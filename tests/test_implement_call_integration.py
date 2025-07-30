import pytest
import os
import json
from langgraph.types import Command
from src.assistant.data_analyst.nodes import implement_call
from src.assistant.data_analyst.models import JsonPatch


class TestImplementCallIntegration:
    """Integration tests for implement_call that use real OpenAI API."""

    def create_state(
        self,
        analysis_result,
        message_content="Generate JSON patches",
        existing_patches=None,
    ):
        """Helper to create state for testing."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": message_content,
                }
            ],
            "analysis_result": analysis_result,
            "patches": existing_patches or [],
        }

    def print_implementation_result(self, result):
        """Print the implementation result for debugging."""
        patches = result.update.get("patches", [])
        print("Implementation results:")
        print(f"  Generated patches: {len(patches)}")
        print(f"  Route: {result.goto}")

        for i, patch in enumerate(patches):
            print(f"  Patch {i + 1}: {patch["op"]} {patch["path"]} {patch["value"] if "value" in patch else ''}")

        print(f"  Full result: {json.dumps(result.update, indent=2, default=str)}")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_implement_call_field_name_change_only(self):
        """Test implementation with field name change only (value stays same)."""
        analysis_result = {
            "legacy_field": "sample_id",
            "legacy_value": "HBM386.ZGKG.235",
            "recommended_mappings": [
                {
                    "target_field": "parent_sample_id",
                    "target_value": "HBM386.ZGKG.235",
                    "confidence_score": 0.95,
                }
            ],
            "mapping_strategy": "one-to-one",
            "overall_confidence": 0.95,
            "reasoning": "Direct field name mapping with same value",
        }

        state = self.create_state(analysis_result)

        result = implement_call(state)

        # Verify result structure
        assert isinstance(result, Command)
        assert result.goto == "plan"
        assert "patches" in result.update
        assert "messages" in result.update

        # Should generate patches for field name change
        patches = result.update["patches"]
        assert len(patches) == 2
        assert "add" == patches[0]["op"]
        assert "/parent_sample_id" == patches[0]["path"]
        assert "HBM386.ZGKG.235" == patches[0]["value"]
        assert "remove" == patches[1]["op"]
        assert "/sample_id" == patches[1]["path"]

        self.print_implementation_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_implement_call_value_transformation_only(self):
        """Test implementation with value transformation only (field name stays same)."""
        analysis_result = {
            "legacy_field": "dataset_type",
            "legacy_value": "rna_seq",
            "recommended_mappings": [
                {
                    "target_field": "dataset_type",
                    "target_value": "RNAseq",
                    "confidence_score": 0.85,
                }
            ],
            "mapping_strategy": "one-to-one",
            "overall_confidence": 0.85,
            "reasoning": "Value normalization to match permissible values",
        }

        state = self.create_state(analysis_result)

        result = implement_call(state)

        assert isinstance(result, Command)
        assert result.goto == "plan"

        patches = result.update["patches"]
        assert len(patches) == 1
        assert "replace" == patches[0]["op"]
        assert "/dataset_type" == patches[0]["path"]
        assert "RNAseq" == patches[0]["value"]

        self.print_implementation_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_implement_call_both_field_and_value_change(self):
        """Test implementation with both field name and value changes."""
        analysis_result = {
            "legacy_field": "instrument_make",
            "legacy_value": "Illumina Inc.",
            "recommended_mappings": [
                {
                    "target_field": "acquisition_instrument_vendor",
                    "target_value": "Illumina",
                    "confidence_score": 0.88,
                }
            ],
            "mapping_strategy": "one-to-one",
            "overall_confidence": 0.88,
            "reasoning": "Field name and value transformation for standardization",
        }

        state = self.create_state(analysis_result)

        result = implement_call(state)

        assert isinstance(result, Command)
        assert result.goto == "plan"

        patches = result.update["patches"]
        assert len(patches) == 2
        assert "add" == patches[0]["op"]
        assert "/acquisition_instrument_vendor" == patches[0]["path"]
        assert "Illumina" == patches[0]["value"]
        assert "remove" == patches[1]["op"]
        assert "/instrument_make" == patches[1]["path"]

        self.print_implementation_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_implement_call_remove_field_no_mapping(self):
        """Test implementation when no target mapping exists (should remove field)."""
        analysis_result = {
            "legacy_field": "deprecated_field",
            "legacy_value": "obsolete_value",
            "recommended_mappings": [],  # No mappings found
            "mapping_strategy": "one-to-one",
            "overall_confidence": 0.0,
            "reasoning": "No suitable target field found - field should be removed",
        }

        state = self.create_state(analysis_result)

        result = implement_call(state)

        assert isinstance(result, Command)
        assert result.goto == "plan"

        patches = result.update["patches"]
        assert len(patches) == 1
        assert "remove" == patches[0]["op"]
        assert "/deprecated_field" == patches[0]["path"]

        self.print_implementation_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_implement_call_one_to_many_mapping(self):
        """Test implementation with one-to-many mapping (composite value split)."""
        analysis_result = {
            "legacy_field": "storage_duration",
            "legacy_value": "72 hours",
            "recommended_mappings": [
                {
                    "target_field": "source_storage_duration_value",
                    "target_value": 72,
                    "confidence_score": 0.9,
                },
                {
                    "target_field": "source_storage_duration_unit",
                    "target_value": "hour",
                    "confidence_score": 0.9,
                },
            ],
            "mapping_strategy": "one-to-many",
            "overall_confidence": 0.9,
            "reasoning": "Composite value split into separate numeric value and unit fields",
        }

        state = self.create_state(analysis_result)

        result = implement_call(state)

        assert isinstance(result, Command)
        assert result.goto == "plan"

        patches = result.update["patches"]
        assert len(patches) == 3  # At least one for each target field
        assert "add" == patches[0]["op"]
        assert "/source_storage_duration_value" == patches[0]["path"]
        assert 72 == patches[0]["value"]
        assert "add" == patches[1]["op"]
        assert "/source_storage_duration_unit" == patches[1]["path"]
        assert "hour" == patches[1]["value"]
        assert "remove" == patches[2]["op"]
        assert "/storage_duration" == patches[2]["path"]

        self.print_implementation_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_implement_call_nested_field_operations(self):
        """Test implementation with nested field operations."""
        analysis_result = {
            "legacy_field": "sample_metadata",
            "legacy_value": {"type": "tissue", "condition": "healthy"},
            "recommended_mappings": [
                {
                    "target_field": "sample_category",
                    "target_value": "tissue_sample",
                    "confidence_score": 0.8,
                },
                {
                    "target_field": "health_status",
                    "target_value": "normal",
                    "confidence_score": 0.75,
                },
            ],
            "mapping_strategy": "one-to-many",
            "overall_confidence": 0.78,
            "reasoning": "Nested object decomposed into separate target fields",
        }

        state = self.create_state(analysis_result)

        result = implement_call(state)

        assert isinstance(result, Command)
        assert result.goto == "plan"

        patches = result.update["patches"]
        assert len(patches) == 3
        assert "add" == patches[0]["op"]
        assert "/sample_category" == patches[0]["path"]
        assert "tissue_sample" == patches[0]["value"]
        assert "add" == patches[1]["op"]
        assert "/health_status" == patches[1]["path"]
        assert "normal" == patches[1]["value"]
        assert "remove" == patches[2]["op"]
        assert "/sample_metadata" == patches[2]["path"]

        self.print_implementation_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_implement_call_patches_accumulation(self):
        """Test that new patches are correctly accumulated with existing patches."""
        analysis_result = {
            "legacy_field": "library_type",
            "legacy_value": "single_end",
            "recommended_mappings": [
                {
                    "target_field": "library_layout",
                    "target_value": "single-end",
                    "confidence_score": 0.92,
                }
            ],
            "mapping_strategy": "one-to-one",
            "overall_confidence": 0.92,
            "reasoning": "Field name change with value normalization",
        }

        # Start with existing patches in state
        existing_patches = [
            {
              "op": "add", 
              "path": "/existing_field", 
              "value": "existing_value"
            }, 
            {
              "op": "remove",
              "path": "/old_field"
            }
        ]

        state = self.create_state(analysis_result, existing_patches=existing_patches)

        result = implement_call(state)

        assert isinstance(result, Command)
        assert result.goto == "plan"

        patches = result.update["patches"]
        assert len(patches) == 4
        assert "add" == patches[0]["op"]
        assert "/existing_field" == patches[0]["path"]
        assert "existing_value" == patches[0]["value"]
        assert "remove" == patches[1]["op"]
        assert "/old_field" == patches[1]["path"]
        assert "add" == patches[2]["op"]
        assert "/library_layout" == patches[2]["path"]
        assert "single-end" == patches[2]["value"]
        assert "remove" == patches[3]["op"]
        assert "/library_type" == patches[3]["path"]

        self.print_implementation_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_implement_call_complex_categorical_mapping(self):
        """Test implementation with complex categorical value mapping."""
        analysis_result = {
            "legacy_field": "assay_technique",
            "legacy_value": "RNA sequencing (bulk)",
            "recommended_mappings": [
                {
                    "target_field": "dataset_type",
                    "target_value": "RNAseq",
                    "confidence_score": 0.87,
                },
                {
                    "target_field": "analyte_class",
                    "target_value": "RNA",
                    "confidence_score": 0.85,
                },
            ],
            "mapping_strategy": "one-to-many",
            "overall_confidence": 0.86,
            "reasoning": "Complex technique description mapped to standardized categorical values",
        }

        state = self.create_state(analysis_result)

        result = implement_call(state)

        assert isinstance(result, Command)
        assert result.goto == "plan"

        patches = result.update["patches"]
        assert len(patches) == 3
        assert "add" == patches[0]["op"]
        assert "/dataset_type" == patches[0]["path"]
        assert "RNAseq" == patches[0]["value"]
        assert "add" == patches[1]["op"]
        assert "/analyte_class" == patches[1]["path"]
        assert "RNA" == patches[1]["value"]
        assert "remove" == patches[2]["op"]
        assert "/assay_technique" == patches[2]["path"]

        self.print_implementation_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_implement_call_edge_case_empty_mappings(self):
        """Test implementation with edge case of empty recommended mappings."""
        analysis_result = {
            "legacy_field": "unknown_field",
            "legacy_value": "mystery_value",
            "recommended_mappings": [],
            "mapping_strategy": "one-to-one",
            "overall_confidence": 0.0,
            "reasoning": "No mappings found - completely unrecognized field",
        }

        state = self.create_state(analysis_result)

        result = implement_call(state)

        assert isinstance(result, Command)
        assert result.goto == "plan"

        patches = result.update["patches"]

        # Should handle gracefully, likely with remove operation
        assert len(patches) == 1
        assert "remove" == patches[0]["op"]
        assert "/unknown_field" == patches[0]["path"]        

        self.print_implementation_result(result)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_implement_call_numeric_field_with_units(self):
        """Test implementation with numeric field that includes units."""
        analysis_result = {
            "legacy_field": "concentration",
            "legacy_value": "10.5 mg/ml",
            "recommended_mappings": [
                {
                    "target_field": "concentration_value",
                    "target_value": 10.5,
                    "confidence_score": 0.85,
                },
                {
                    "target_field": "concentration_unit",
                    "target_value": "mg/ml",
                    "confidence_score": 0.85,
                },
            ],
            "mapping_strategy": "one-to-many",
            "overall_confidence": 0.85,
            "reasoning": "Numeric value with units split into separate value and unit fields",
        }

        state = self.create_state(analysis_result)

        result = implement_call(state)

        assert isinstance(result, Command)
        assert result.goto == "plan"

        patches = result.update["patches"]
        assert len(patches) == 3
        assert "add" == patches[0]["op"]
        assert "/concentration_value" == patches[0]["path"]
        assert 10.5 == patches[0]["value"]
        assert "add" == patches[1]["op"]
        assert "/concentration_unit" == patches[1]["path"]
        assert "mg/ml" == patches[1]["value"]
        assert "remove" == patches[2]["op"]
        assert "/concentration" == patches[2]["path"]

        self.print_implementation_result(result)


if __name__ == "__main__":
    # Run only integration tests
    pytest.main([__file__, "-v", "-m", "integration"])
