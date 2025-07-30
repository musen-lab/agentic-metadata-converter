import pytest
import os
import json
from langgraph.types import Command
from src.assistant.data_analyst.nodes import analysis_call
from src.assistant.data_analyst.models import (
    TargetSchema,
    SchemaField,
    PastAnalysis,
    PastMappingRecord,
    MappingDigest,
)


class TestAnalysisCallIntegration:
    """Integration tests for analysis_call that use real OpenAI API."""

    @pytest.fixture
    def target_schema(self):
        """Create a test target schema with 10 representative fields."""
        fields = [
            SchemaField(
                name="parent_sample_id",
                description="Unique HuBMAP or SenNet identifier of the sample (i.e., block, section or suspension) used to perform this assay. For example, for a RNAseq assay, the parent would be the suspension, whereas, for one of the imaging assays, the parent would be the tissue section. If an assay comes from multiple parent samples then this should be a comma separated list. Example: HBM386.ZGKG.235, HBM672.MKPK.442 or SNT232.UBHJ.322, SNT329.ALSK.102",
                type="text",
                required=True,
                regex=r"^(?:HBM|SNT)\d{3}\.[A-Z]{4}\.\d{3}(?:,\s*(?:HBM|SNT)\d{3}\.[A-Z]{4}\.\d{3})*$",
            ),
            SchemaField(
                name="lab_id",
                description="An internal field labs can use it to add whatever ID(s) they want or need for dataset validation and tracking. This could be a single ID (e.g., 'Visium_9OLC_A4_S1') or a delimited list of IDs (e.g., '9OL; 9OLC.A2; Visium_9OLC_A4_S1'). This field will not be accessible to anyone outside of the consortium and no effort will be made to check if IDs provided by one data provider are also used by another.",
                type="text",
                required=False,
            ),
            SchemaField(
                name="dataset_type",
                description="The specific type of dataset being produced.",
                type="categorical",
                required=True,
                default_value="RNAseq",
                permissible_values=[
                    "RNAseq",
                    "ATACseq",
                    "CODEX",
                    "Visium (no probes)",
                    "Visium (with probes)",
                    "MERFISH",
                    "seqFISH",
                    "CosMx",
                    "Xenium",
                    "MIBI",
                ],
            ),
            SchemaField(
                name="analyte_class",
                description="Analytes are the target molecules being measured with the assay.",
                type="categorical",
                required=True,
                permissible_values=[
                    "RNA",
                    "DNA",
                    "DNA + RNA",
                    "Protein",
                    "Nucleic acid + protein",
                    "Metabolite",
                    "Lipid",
                    "Chromatin",
                ],
            ),
            SchemaField(
                name="source_storage_duration_value",
                description="How long was the source material (parent) stored, prior to this sample being processed.",
                type="number",
                required=True,
            ),
            SchemaField(
                name="source_storage_duration_unit",
                description="The time duration unit of measurement",
                type="categorical",
                required=True,
                permissible_values=["hour", "day", "month", "year", "minute"],
            ),
            SchemaField(
                name="is_targeted",
                description='Specifies whether or not a specific molecule(s) is/are targeted for detection/measurement by the assay ("Yes" or "No"). The CODEX analyte is protein.',
                type="categorical",
                required=True,
                permissible_values=["Yes", "No"],
            ),
            SchemaField(
                name="library_layout",
                description="Whether the library was generated for single-end or paired end sequencing",
                type="categorical",
                required=True,
                permissible_values=["single-end", "paired-end"],
            ),
            SchemaField(
                name="expected_entity_capture_count",
                description="Number of cells, nuclei or capture spots expected to be captured by the assay. For Visium this is the total number of spots covered by tissue, within the capture area.",
                type="number",
                required=False,
            ),
            SchemaField(
                name="contributors_path",
                description='The path to the file with the ORCID IDs for all contributors of this dataset (e.g., "./extras/contributors.tsv" or "./contributors.tsv"). This is an internal metadata field that is just used for ingest.',
                type="text",
                required=True,
                regex=r"^(?:\.\\/.*|\\w.*)\\.tsv$",
            ),
        ]
        return TargetSchema(fields=fields)

    @pytest.fixture
    def past_analysis(self):
        """Create sample past analysis data for context."""
        records = [
            PastMappingRecord(
                legacy_field="sample_id",
                legacy_value="HBM123.ABCD.456",
                recommended_mappings=[
                    MappingDigest(
                        target_field="parent_sample_id",
                        target_value="HBM123.ABCD.456",
                        confidence_score=0.95,
                    )
                ],
                reasoning="The legacy field 'sample_id' closely aligns with the target field 'parent_sample_id' based on semantic meaning and description. The legacy value 'HBM386.ZGKG.235' matches the required pattern for 'parent_sample_id', resulting in a high confidence score.",
            ),
            PastMappingRecord(
                legacy_field="sequencing_type",
                legacy_value="RNA sequencing",
                recommended_mappings=[
                    MappingDigest(
                        target_field="dataset_type",
                        target_value="RNAseq",
                        confidence_score=0.75,
                    )
                ],
                reasoning="The legacy field 'sequencing_type' closely aligns with the target field 'dataset_type' based on semantic meaning. The legacy value 'RNA sequencing' is normalized to the permissible value 'RNAseq' in the target schema, resulting in a good confidence score.",
            ),
        ]
        return PastAnalysis(records=records)

    def create_state(self, legacy_field, legacy_value, target_schema, past_analysis):
        """Helper to create state for testing."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze this legacy metadata field and value:\n\n**Legacy field**: {legacy_field}\n**Legacy value**: {legacy_value}",
                }
            ],
            "last_checked_field": legacy_field,
            "target_schema": target_schema,
            "past_analysis": past_analysis,
        }

    def print_analysis_result(self, result):
        """Print the analysis result"""
        analysis_result = result.update["analysis_result"]
        print("Analysis results:")
        print(f"  Overall confidence: {analysis_result['overall_confidence']}")
        print(f"  Recommended mappings: {len(analysis_result['recommended_mappings'])}")
        print(f"  Strategy: {analysis_result['mapping_strategy']}")
        print(f"  Result: {json.dumps(result.update, indent=2, default=str)}")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_analysis_call_direct_sample_id_mapping(self, target_schema, past_analysis):
        """Test analysis with direct sample ID mapping."""
        state = self.create_state(
            "sample_id", "HBM386.ZGKG.235", target_schema, past_analysis
        )

        result = analysis_call(state)

        # Verify result structure
        assert isinstance(result, Command)
        assert result.goto == "implement"
        assert "messages" in result.update
        assert "analysis_result" in result.update

        # Check analysis result
        analysis_result = result.update["analysis_result"]
        assert analysis_result["legacy_field"] == "sample_id"
        assert analysis_result["legacy_value"] == "HBM386.ZGKG.235"
        assert analysis_result["overall_confidence"] > 0.0

        # Should have at least one recommended mapping
        if analysis_result["overall_confidence"] >= 0.6:
            assert len(analysis_result["recommended_mappings"]) > 0
            # Most likely mapping should be to parent_sample_id
            likely_fields = [
                m["target_field"] for m in analysis_result["recommended_mappings"]
            ]
            assert "parent_sample_id" in likely_fields

        self.print_analysis_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_analysis_call_composite_value_mapping(self, target_schema, past_analysis):
        """Test analysis with composite value that should split into multiple fields."""
        state = self.create_state(
            "storage_duration", "72 hours", target_schema, past_analysis
        )

        result = analysis_call(state)

        assert isinstance(result, Command)
        assert result.goto == "implement"
        assert "analysis_result" in result.update

        analysis_result = result.update["analysis_result"]
        assert analysis_result["legacy_field"] == "storage_duration"
        assert analysis_result["legacy_value"] == "72 hours"

        # Should detect the composite nature and potentially map to both value and unit fields
        if analysis_result["overall_confidence"] >= 0.6:
            recommended_fields = [
                m["target_field"] for m in analysis_result["recommended_mappings"]
            ]
            # Look for duration-related fields
            duration_fields = [f for f in recommended_fields if "duration" in f.lower()]
            assert len(duration_fields) > 0

        self.print_analysis_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_analysis_call_categorical_mapping(self, target_schema, past_analysis):
        """Test analysis with categorical field mapping."""
        state = self.create_state(
            "sequencing_type", "RNA sequencing", target_schema, past_analysis
        )

        result = analysis_call(state)

        assert isinstance(result, Command)
        assert result.goto == "implement"
        assert "analysis_result" in result.update

        analysis_result = result.update["analysis_result"]
        assert analysis_result["legacy_field"] == "sequencing_type"
        assert analysis_result["legacy_value"] == "RNA sequencing"

        # Should map to dataset_type or analyte_class
        if analysis_result["overall_confidence"] >= 0.6:
            recommended_fields = [
                m["target_field"] for m in analysis_result["recommended_mappings"]
            ]
            likely_categorical_fields = ["dataset_type", "analyte_class"]
            assert any(
                field in recommended_fields for field in likely_categorical_fields
            )

        self.print_analysis_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_analysis_call_no_mapping_scenario(self, target_schema, past_analysis):
        """Test analysis when no reasonable mapping exists."""
        state = self.create_state(
            "unrelated_field",
            "completely_unrelated_value",
            target_schema,
            past_analysis,
        )

        result = analysis_call(state)

        assert isinstance(result, Command)
        assert result.goto == "implement"
        assert "analysis_result" in result.update

        analysis_result = result.update["analysis_result"]
        assert analysis_result["legacy_field"] == "unrelated_field"
        assert analysis_result["legacy_value"] == "completely_unrelated_value"

        # Should have low confidence and potentially no recommended mappings
        self.print_analysis_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_analysis_call_with_past_analysis_context(
        self, target_schema, past_analysis
    ):
        """Test analysis leveraging past analysis context."""
        # Use a similar field to one in past analysis
        state = self.create_state(
            "sample_id",  # Exact match with past analysis
            "HBM999.WXYZ.789",
            target_schema,
            past_analysis,
        )

        result = analysis_call(state)

        assert isinstance(result, Command)
        assert result.goto == "implement"
        assert "analysis_result" in result.update

        analysis_result = result.update["analysis_result"]
        assert analysis_result["legacy_field"] == "sample_id"

        # Should have high confidence due to past analysis match
        if analysis_result["overall_confidence"] >= 0.8:
            # Should map to parent_sample_id like the past analysis
            recommended_fields = [
                m["target_field"] for m in analysis_result["recommended_mappings"]
            ]
            assert "parent_sample_id" in recommended_fields

        self.print_analysis_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_analysis_call_numeric_field_mapping(self, target_schema, past_analysis):
        """Test analysis with numeric field mapping."""
        state = self.create_state("cell_count", "50000", target_schema, past_analysis)

        result = analysis_call(state)

        assert isinstance(result, Command)
        assert result.goto == "implement"
        assert "analysis_result" in result.update

        analysis_result = result.update["analysis_result"]
        assert analysis_result["legacy_field"] == "cell_count"
        assert analysis_result["legacy_value"] == 50000

        # Should map to expected_entity_capture_count
        if analysis_result["overall_confidence"] >= 0.6:
            recommended_fields = [
                m["target_field"] for m in analysis_result["recommended_mappings"]
            ]
            numeric_fields = [
                "expected_entity_capture_count",
                "source_storage_duration_value",
            ]
            assert any(field in recommended_fields for field in numeric_fields)

        self.print_analysis_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_analysis_call_boolean_field_mapping(self, target_schema, past_analysis):
        """Test analysis with boolean-like field mapping."""
        state = self.create_state(
            "targeted_analysis", "yes", target_schema, past_analysis
        )

        result = analysis_call(state)

        assert isinstance(result, Command)
        assert result.goto == "implement"
        assert "analysis_result" in result.update

        analysis_result = result.update["analysis_result"]
        assert analysis_result["legacy_field"] == "targeted_analysis"
        assert analysis_result["legacy_value"] == "yes"

        # Should map to is_targeted field
        if analysis_result["overall_confidence"] >= 0.6:
            recommended_fields = [
                m["target_field"] for m in analysis_result["recommended_mappings"]
            ]
            assert "is_targeted" in recommended_fields

            # Check value transformation
            is_targeted_mapping = next(
                (
                    m
                    for m in analysis_result["recommended_mappings"]
                    if m["target_field"] == "is_targeted"
                ),
                None,
            )
            if is_targeted_mapping:
                assert is_targeted_mapping["target_value"] in ["Yes", "No"]

        self.print_analysis_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_analysis_call_regex_validation_scenario(
        self, target_schema, past_analysis
    ):
        """Test analysis with field that has regex validation requirements."""
        state = self.create_state(
            "contributor_file", "./data/contributors.tsv", target_schema, past_analysis
        )

        result = analysis_call(state)

        assert isinstance(result, Command)
        assert result.goto == "implement"
        assert "analysis_result" in result.update

        analysis_result = result.update["analysis_result"]
        assert analysis_result["legacy_field"] == "contributor_file"
        assert analysis_result["legacy_value"] == "./data/contributors.tsv"

        # Should map to contributors_path with high confidence due to matching regex
        if analysis_result["overall_confidence"] >= 0.8:
            recommended_fields = [
                m["target_field"] for m in analysis_result["recommended_mappings"]
            ]
            assert "contributors_path" in recommended_fields

        self.print_analysis_result(result)


if __name__ == "__main__":
    # Run only integration tests
    pytest.main([__file__, "-v", "-m", "integration"])
