import pytest
import os
from unittest.mock import Mock, patch
from langgraph.types import Command
from langgraph.graph import END

# Set dummy API key to avoid errors during module import
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")

from src.assistant.data_analyst.nodes import analysis_call
from src.assistant.data_analyst.models import (
    AnalysisOutput,
    AnalysisResult,
    MappingDigest,
    TargetSchema,
    SchemaField,
    PastAnalysis,
    PastMappingRecord,
)


class TestAnalysisCall:
    """Test cases for the analysis_call function."""

    @pytest.fixture
    def target_schema(self):
        """Create a test target schema with representative fields."""
        fields = [
            SchemaField(
                name="parent_sample_id",
                description="Unique HuBMAP or SenNet identifier of the sample",
                type="text",
                required=True,
                regex=r"^(?:HBM|SNT)\d{3}\.[A-Z]{4}\.\d{3}(?:,\s*(?:HBM|SNT)\d{3}\.[A-Z]{4}\.\d{3})*$",
            ),
            SchemaField(
                name="dataset_type",
                description="The specific type of dataset being produced",
                type="categorical",
                required=True,
                default_value="RNAseq",
                permissible_values=["RNAseq", "ATACseq", "CODEX", "Visium", "MERFISH"],
            ),
        ]
        return TargetSchema(fields=fields)

    @pytest.fixture
    def past_analysis(self):
        """Create sample past analysis data."""
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
            )
        ]
        return PastAnalysis(records=records)

    @pytest.fixture
    def base_state(self, target_schema, past_analysis):
        """Create a base AppState for testing."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "Analyze this legacy metadata field and value:\n\n**Legacy field**: sample_identifier\n**Legacy value**: HBM386.ZGKG.235",
                }
            ],
            "last_checked_field": "sample_identifier",
            "target_schema": target_schema,
            "past_analysis": past_analysis,
        }

    def test_analysis_call_validates_required_state_fields(self):
        """Test that analysis_call properly validates required state fields."""
        # Test missing messages
        empty_state = {}
        result = analysis_call(empty_state)
        assert isinstance(result, Command)
        assert result.goto == END
        assert result.update == {}
        
        # Test empty messages list
        state_no_messages = {"messages": []}
        result = analysis_call(state_no_messages)
        assert isinstance(result, Command)
        assert result.goto == END
        assert result.update == {}
        
        # Test missing last_checked_field
        state_no_field = {
            "messages": [{"role": "user", "content": "test"}],
            "last_checked_field": None
        }
        result = analysis_call(state_no_field)
        assert isinstance(result, Command)
        assert result.goto == END
        assert result.update == {}

    def test_analysis_call_handles_missing_target_schema(self, base_state):
        """Test that missing target schema is handled properly."""
        base_state["target_schema"] = None
        
        result = analysis_call(base_state)
        
        assert isinstance(result, Command)
        assert result.goto == END
        assert "messages" in result.update
        
        error_message = result.update["messages"][0]
        assert error_message["role"] == "assistant"
        assert "Analysis failed for sample_identifier" in error_message["content"]
        assert "No target schema available" in error_message["content"]

    def test_analysis_call_extracts_state_data_correctly(self, base_state):
        """Test that the function correctly extracts data from state."""
        with patch("src.assistant.data_analyst.nodes._analysis_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            
            # Create a minimal valid response to avoid errors
            mock_result = AnalysisResult(
                legacy_field="sample_identifier",
                legacy_value="HBM386.ZGKG.235",
                mapping_results=[],
                recommended_mappings=[],
                overall_confidence=0.8,
                mapping_strategy="one-to-one",
                reasoning="Test"
            )
            mock_output = AnalysisOutput(analysis=mock_result)
            mock_structured_llm.invoke.return_value = mock_output
            
            analysis_call(base_state)
            
            # Verify the function extracted the correct data
            mock_structured_llm.invoke.assert_called_once()
            call_args = mock_structured_llm.invoke.call_args[0][0]
            
            # Should have system and user messages
            assert len(call_args) == 2
            assert call_args[0]["role"] == "system"
            assert call_args[1]["role"] == "user"
            
            # User message should contain the original message content
            user_content = call_args[1]["content"]
            expected_content = base_state["messages"][0]["content"]
            assert user_content == expected_content

    def test_analysis_call_formats_system_prompt_with_schema_and_past_analysis(self, base_state):
        """Test that system prompt is properly formatted with target schema and past analysis."""
        with patch("src.assistant.data_analyst.nodes._analysis_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            
            # Mock response
            mock_result = AnalysisResult(
                legacy_field="sample_identifier",
                legacy_value="test",
                mapping_results=[],
                recommended_mappings=[],
                overall_confidence=0.5,
                mapping_strategy="one-to-one",
                reasoning="Test"
            )
            mock_output = AnalysisOutput(analysis=mock_result)
            mock_structured_llm.invoke.return_value = mock_output
            
            analysis_call(base_state)
            
            # Get the system prompt that was passed to LLM
            call_args = mock_structured_llm.invoke.call_args[0][0]
            system_prompt = call_args[0]["content"]
            
            # Verify target schema data is in the prompt
            assert "parent_sample_id" in system_prompt
            assert "dataset_type" in system_prompt
            
            # Verify past analysis data is in the prompt
            assert "sample_id" in system_prompt  # from past analysis
            assert "HBM123.ABCD.456" in system_prompt  # from past analysis

    def test_analysis_call_converts_pydantic_models_to_dict(self, base_state):
        """Test that Pydantic models are properly converted to dicts for JSON serialization."""
        with patch("src.assistant.data_analyst.nodes._analysis_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            
            # Mock response
            mock_result = AnalysisResult(
                legacy_field="test",
                legacy_value="test",
                mapping_results=[],
                recommended_mappings=[],
                overall_confidence=0.5,
                mapping_strategy="one-to-one",
                reasoning="Test"
            )
            mock_output = AnalysisOutput(analysis=mock_result)
            mock_structured_llm.invoke.return_value = mock_output
            
            analysis_call(base_state)
            
            # Get the system prompt
            call_args = mock_structured_llm.invoke.call_args[0][0]
            system_prompt = call_args[0]["content"]
                        
            # Verify that the prompt contains valid JSON (would fail if models weren't converted)
            # The prompt should contain JSON-formatted schema and past analysis
            assert isinstance(system_prompt, str)
            assert len(system_prompt) > 100  # Should have substantial content


    def test_analysis_call_successful_flow_returns_correct_command(self, base_state):
        """Test that successful analysis returns correct Command structure."""
        with patch("src.assistant.data_analyst.nodes._analysis_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            
            # Create a realistic analysis result
            mock_result = AnalysisResult(
                legacy_field="sample_identifier",
                legacy_value="HBM386.ZGKG.235",
                mapping_results=[],
                recommended_mappings=[
                    MappingDigest(
                        target_field="parent_sample_id",
                        target_value="HBM386.ZGKG.235",
                        confidence_score=0.95,
                    )
                ],
                overall_confidence=0.95,
                mapping_strategy="one-to-one",
                reasoning="Direct mapping found"
            )
            mock_output = AnalysisOutput(analysis=mock_result)
            mock_structured_llm.invoke.return_value = mock_output
            
            result = analysis_call(base_state)
            
            # Test command structure
            assert isinstance(result, Command)
            assert result.goto == "implement"
            assert isinstance(result.update, dict)
            assert "messages" in result.update
            assert "analysis_result" in result.update
            
            # Test message structure
            message = result.update["messages"][0]
            assert message["role"] == "assistant"
            assert "Analysis completed for sample_identifier" in message["content"]
            assert "1 recommended mappings" in message["content"]
            assert "overall confidence 0.95" in message["content"]
            
            # Test that analysis result is properly serialized
            analysis_result = result.update["analysis_result"]
            assert isinstance(analysis_result, dict)  # Should be serialized to dict
            assert analysis_result["legacy_field"] == "sample_identifier"
            assert analysis_result["overall_confidence"] == 0.95

    def test_analysis_call_handles_llm_exceptions(self, base_state):
        """Test that LLM exceptions are properly handled and return error command."""
        with patch("src.assistant.data_analyst.nodes._analysis_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            
            # Simulate LLM throwing an exception
            mock_structured_llm.invoke.side_effect = Exception("API connection failed")
            
            result = analysis_call(base_state)
            
            # Should return END command with error message
            assert isinstance(result, Command)
            assert result.goto == END
            assert "messages" in result.update
            
            error_message = result.update["messages"][0]
            assert error_message["role"] == "assistant"
            assert "Analysis failed for sample_identifier" in error_message["content"]
            assert "API connection failed" in error_message["content"]

    def test_analysis_call_uses_correct_llm_configuration(self, base_state):
        """Test that the function configures the LLM correctly for structured output."""
        with patch("src.assistant.data_analyst.nodes._analysis_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            
            # Mock response
            mock_result = AnalysisResult(
                legacy_field="test",
                legacy_value="test",
                mapping_results=[],
                recommended_mappings=[],
                overall_confidence=0.5,
                mapping_strategy="one-to-one",
                reasoning="Test"
            )
            mock_output = AnalysisOutput(analysis=mock_result)
            mock_structured_llm.invoke.return_value = mock_output
            
            analysis_call(base_state)
            
            # Verify LLM was configured for structured output with correct model
            mock_llm.with_structured_output.assert_called_once_with(AnalysisOutput)
            mock_structured_llm.invoke.assert_called_once()

    def test_analysis_call_message_extraction_logic(self):
        """Test that the function correctly extracts the last message content."""
        # Test with multiple messages - should use the last one
        state_multiple_messages = {
            "messages": [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "Assistant response"},
                {"role": "user", "content": "**Legacy field**: test_field\n**Legacy value**: test_value"}
            ],
            "last_checked_field": "test_field",
            "target_schema": TargetSchema(fields=[]),
            "past_analysis": PastAnalysis(records=[])
        }
        
        with patch("src.assistant.data_analyst.nodes._analysis_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            
            mock_result = AnalysisResult(
                legacy_field="test_field",
                legacy_value="test_value",
                mapping_results=[],
                recommended_mappings=[],
                overall_confidence=0.5,
                mapping_strategy="one-to-one",
                reasoning="Test"
            )
            mock_output = AnalysisOutput(analysis=mock_result)
            mock_structured_llm.invoke.return_value = mock_output
            
            analysis_call(state_multiple_messages)
            
            # Verify the last message content was used
            call_args = mock_structured_llm.invoke.call_args[0][0]
            user_prompt = call_args[1]["content"]
            assert user_prompt == "**Legacy field**: test_field\n**Legacy value**: test_value"
            assert "First message" not in user_prompt

    def test_analysis_call_field_consistency_check(self, base_state):
        """Test that the function maintains consistency between last_checked_field and analysis results."""
        # Change the last_checked_field to something different
        base_state["last_checked_field"] = "different_field"
        
        with patch("src.assistant.data_analyst.nodes._analysis_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            
            mock_result = AnalysisResult(
                legacy_field="different_field",  # Should match last_checked_field
                legacy_value="test_value",
                mapping_results=[],
                recommended_mappings=[],
                overall_confidence=0.5,
                mapping_strategy="one-to-one",
                reasoning="Test"
            )
            mock_output = AnalysisOutput(analysis=mock_result)
            mock_structured_llm.invoke.return_value = mock_output
            
            result = analysis_call(base_state)
            
            # The success message should reference the correct field
            message = result.update["messages"][0]
            assert "Analysis completed for different_field" in message["content"]

    def test_analysis_call_state_data_passthrough(self, base_state):
        """Test that the function properly passes through and uses all required state data."""
        with patch("src.assistant.data_analyst.nodes._analysis_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            
            mock_result = AnalysisResult(
                legacy_field="sample_identifier",
                legacy_value="test",
                mapping_results=[],
                recommended_mappings=[],
                overall_confidence=0.75,
                mapping_strategy="one-to-one",
                reasoning="Test"
            )
            mock_output = AnalysisOutput(analysis=mock_result)
            mock_structured_llm.invoke.return_value = mock_output
            
            result = analysis_call(base_state)
            
            # Verify the analysis result contains the model dump
            analysis_result = result.update["analysis_result"]
            assert analysis_result == mock_result.model_dump()
            
            # Verify correct routing
            assert result.goto == "implement"
            assert isinstance(result.update["analysis_result"], dict)

    def test_analysis_call_handles_edge_case_empty_content(self, base_state):
        """Test handling of messages with empty content."""
        base_state["messages"][0]["content"] = ""
        
        with patch("src.assistant.data_analyst.nodes._analysis_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            
            mock_result = AnalysisResult(
                legacy_field="sample_identifier",
                legacy_value="",
                mapping_results=[],
                recommended_mappings=[],
                overall_confidence=0.0,
                mapping_strategy="one-to-one",
                reasoning="Empty content"
            )
            mock_output = AnalysisOutput(analysis=mock_result)
            mock_structured_llm.invoke.return_value = mock_output
            
            result = analysis_call(base_state)
            
            # Should still process successfully
            assert isinstance(result, Command)
            assert result.goto == "implement"
            
            # Verify empty content was passed to LLM
            call_args = mock_structured_llm.invoke.call_args[0][0]
            user_content = call_args[1]["content"]
            assert user_content == ""

    def test_analysis_call_model_dump_serialization(self, base_state):
        """Test that model.model_dump() is called correctly for serialization."""
        # Test both target_schema and past_analysis model_dump calls
        target_schema_mock = Mock()
        target_schema_mock.model_dump.return_value = {"test": "schema"}
        past_analysis_mock = Mock()
        past_analysis_mock.model_dump.return_value = {"test": "analysis"}
        
        base_state["target_schema"] = target_schema_mock
        base_state["past_analysis"] = past_analysis_mock
        
        with patch("src.assistant.data_analyst.nodes._analysis_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            
            mock_result = AnalysisResult(
                legacy_field="test",
                legacy_value="test",
                mapping_results=[],
                recommended_mappings=[],
                overall_confidence=0.5,
                mapping_strategy="one-to-one",
                reasoning="Test"
            )
            mock_output = AnalysisOutput(analysis=mock_result)
            mock_structured_llm.invoke.return_value = mock_output
            
            result = analysis_call(base_state)
            
            # Verify model_dump was called on both objects
            target_schema_mock.model_dump.assert_called_once()
            past_analysis_mock.model_dump.assert_called_once()
            
            # Verify the result analysis was also serialized via model_dump
            analysis_result = result.update["analysis_result"]
            assert isinstance(analysis_result, dict)

    def test_analysis_call_fallback_for_non_pydantic_objects(self):
        """Test that the function handles objects without model_dump method."""
        # Create state with regular dicts instead of Pydantic models
        state_with_dicts = {
            "messages": [{"role": "user", "content": "test content"}],
            "last_checked_field": "test_field",
            "target_schema": {"fields": []},  # Regular dict, no model_dump
            "past_analysis": {"records": []}   # Regular dict, no model_dump
        }
        
        with patch("src.assistant.data_analyst.nodes._analysis_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            
            mock_result = AnalysisResult(
                legacy_field="test_field",
                legacy_value="test",
                mapping_results=[],
                recommended_mappings=[],
                overall_confidence=0.5,
                mapping_strategy="one-to-one",
                reasoning="Test"
            )
            mock_output = AnalysisOutput(analysis=mock_result)
            mock_structured_llm.invoke.return_value = mock_output
            
            result = analysis_call(state_with_dicts)
            
            # Should still work with regular dicts
            assert isinstance(result, Command)
            assert result.goto == "implement"
            
            # Verify the system prompt was still formatted
            call_args = mock_structured_llm.invoke.call_args[0][0]
            system_prompt = call_args[0]["content"]
            assert isinstance(system_prompt, str)
            assert len(system_prompt) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])