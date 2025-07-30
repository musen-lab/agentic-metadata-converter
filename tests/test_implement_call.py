import pytest
import os
from unittest.mock import Mock, patch
from langgraph.types import Command
from langgraph.graph import END

# Set dummy API key to avoid errors during module import
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")

from src.assistant.data_analyst.nodes import implement_call
from src.assistant.data_analyst.models import (
    ImplementationOutput,
    JsonPatch,
)


class TestImplementCall:
    """Test cases for the implement_call function."""

    @pytest.fixture
    def sample_analysis_result(self):
        """Create a sample analysis result for testing."""
        return {
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
            "reasoning": "Direct field mapping with high confidence",
        }

    @pytest.fixture
    def base_state(self, sample_analysis_result):
        """Create a base AppState for testing."""
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Analysis completed for sample_id. Generated 1 recommended mappings with overall confidence 0.95.",
                }
            ],
            "analysis_result": sample_analysis_result,
            "patches": [],  # Start with empty patches
        }

    def test_implement_call_validates_required_state_fields(self):
        """Test that implement_call properly validates required state fields."""
        # Test missing messages
        empty_state = {}
        result = implement_call(empty_state)
        assert isinstance(result, Command)
        assert result.goto == END
        assert result.update == {}

        # Test empty messages list
        state_no_messages = {"messages": []}
        result = implement_call(state_no_messages)
        assert isinstance(result, Command)
        assert result.goto == END
        assert result.update == {}

        # Test missing analysis_result
        state_no_analysis = {
            "messages": [{"role": "assistant", "content": "test"}],
            "analysis_result": None,
        }
        result = implement_call(state_no_analysis)
        assert isinstance(result, Command)
        assert result.goto == END
        assert "messages" in result.update
        error_message = result.update["messages"][0]
        assert error_message["role"] == "assistant"
        assert (
            "Implementation failed: No analysis result available"
            in error_message["content"]
        )

    def test_implement_call_extracts_state_data_correctly(self, base_state):
        """Test that the function correctly extracts data from state."""
        with patch("src.assistant.data_analyst.nodes._implementation_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm

            # Create a valid response to avoid errors
            mock_patches = [
                JsonPatch(op="add", path="/parent_sample_id", value="HBM386.ZGKG.235"),
                JsonPatch(op="remove", path="/sample_id"),
            ]
            mock_output = ImplementationOutput(patches=mock_patches)
            mock_structured_llm.invoke.return_value = mock_output

            implement_call(base_state)

            # Verify the function extracted the correct data
            mock_structured_llm.invoke.assert_called_once()
            call_args = mock_structured_llm.invoke.call_args[0][0]

            # Should have system and user messages
            assert len(call_args) == 2
            assert call_args[0]["role"] == "system"
            assert call_args[1]["role"] == "user"

            # User message should contain analysis result details
            user_content = call_args[1]["content"]
            assert "sample_id" in user_content
            assert "HBM386.ZGKG.235" in user_content
            assert "parent_sample_id" in user_content

    def test_implement_call_formats_user_prompt_correctly(self, base_state):
        """Test that user prompt is properly formatted with analysis result details."""
        with patch("src.assistant.data_analyst.nodes._implementation_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm

            mock_patches = [JsonPatch(op="add", path="/test", value="test")]
            mock_output = ImplementationOutput(patches=mock_patches)
            mock_structured_llm.invoke.return_value = mock_output

            implement_call(base_state)

            # Get the user prompt that was passed to LLM
            call_args = mock_structured_llm.invoke.call_args[0][0]
            user_prompt = call_args[1]["content"]
            print(user_prompt)
            # Verify all analysis result components are included
            assert "Legacy: {\"sample_id\": \"HBM386.ZGKG.235\"}" in user_prompt
            assert "Target: {\"parent_sample_id\": \"HBM386.ZGKG.235\"}" in user_prompt
            assert "Mapping strategy: one-to-one" in user_prompt
            assert "Overall confidence: 0.95" in user_prompt
            assert "Reasoning: Direct field mapping with high confidence" in user_prompt

    def test_implement_call_successful_flow_returns_correct_command(self, base_state):
        """Test that successful implementation returns correct Command structure."""
        with patch("src.assistant.data_analyst.nodes._implementation_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm

            # Create realistic patches
            mock_patches = [
                JsonPatch(op="add", path="/parent_sample_id", value="HBM386.ZGKG.235"),
                JsonPatch(op="remove", path="/sample_id"),
            ]
            mock_output = ImplementationOutput(patches=mock_patches)
            mock_structured_llm.invoke.return_value = mock_output

            result = implement_call(base_state)

            # Test command structure
            assert isinstance(result, Command)
            assert result.goto == "plan"
            assert isinstance(result.update, dict)
            assert "messages" in result.update
            assert "patches" in result.update

            # Test message structure
            message = result.update["messages"][0]
            assert message["role"] == "assistant"
            assert "Implementation completed for sample_id" in message["content"]
            assert "Generated 2 JSON Patch operations" in message["content"]

            # Test patches storage
            patches = result.update["patches"]
            assert len(patches) == 2
            assert all(isinstance(patch, JsonPatch) for patch in patches)

    def test_implement_call_accumulates_patches_correctly(self, base_state):
        """Test that patches are accumulated with existing patches in state."""
        # Add existing patches to state
        existing_patch = JsonPatch(
            op="replace", path="/existing_field", value="existing_value"
        )
        base_state["patches"] = [existing_patch]

        with patch("src.assistant.data_analyst.nodes._implementation_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm

            # New patches from implementation
            new_patches = [
                JsonPatch(op="add", path="/parent_sample_id", value="HBM386.ZGKG.235"),
                JsonPatch(op="remove", path="/sample_id"),
            ]
            mock_output = ImplementationOutput(patches=new_patches)
            mock_structured_llm.invoke.return_value = mock_output

            result = implement_call(base_state)

            # Verify patches were accumulated correctly
            all_patches = result.update["patches"]
            assert len(all_patches) == 3  # 1 existing + 2 new
            assert all_patches[0] == existing_patch  # Existing patch first
            assert all_patches[1:] == new_patches  # New patches appended

    def test_implement_call_handles_llm_exceptions(self, base_state):
        """Test that LLM exceptions are properly handled and return error command."""
        with patch("src.assistant.data_analyst.nodes._implementation_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm

            # Simulate LLM throwing an exception
            mock_structured_llm.invoke.side_effect = Exception("API connection failed")

            result = implement_call(base_state)

            # Should return END command with error message
            assert isinstance(result, Command)
            assert result.goto == END
            assert "messages" in result.update

            error_message = result.update["messages"][0]
            assert error_message["role"] == "assistant"
            assert (
                "Implementation failed: API connection failed"
                in error_message["content"]
            )

    def test_implement_call_uses_correct_llm_configuration(self, base_state):
        """Test that the function configures the LLM correctly for structured output."""
        with patch("src.assistant.data_analyst.nodes._implementation_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm

            mock_patches = [JsonPatch(op="test", path="/test", value="test")]
            mock_output = ImplementationOutput(patches=mock_patches)
            mock_structured_llm.invoke.return_value = mock_output

            implement_call(base_state)

            # Verify LLM was configured for structured output with correct model
            mock_llm.with_structured_output.assert_called_once_with(
                ImplementationOutput
            )
            mock_structured_llm.invoke.assert_called_once()

    def test_implement_call_handles_edge_case_empty_content(self, base_state):
        """Test handling of messages with empty content."""
        base_state["messages"][0]["content"] = ""

        with patch("src.assistant.data_analyst.nodes._implementation_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm

            mock_patches = [JsonPatch(op="remove", path="/empty_field")]
            mock_output = ImplementationOutput(patches=mock_patches)
            mock_structured_llm.invoke.return_value = mock_output

            result = implement_call(base_state)

            # Should still process successfully
            assert isinstance(result, Command)
            assert result.goto == "plan"

            # Verify empty content was handled in the prompt
            call_args = mock_structured_llm.invoke.call_args[0][0]
            user_content = call_args[1]["content"]
            assert "**Analysis Result:**" in user_content

    def test_implement_call_handles_complex_analysis_result(self):
        """Test with complex analysis result including multiple mappings."""
        complex_analysis_result = {
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
            "reasoning": "Composite value split into separate value and unit fields",
        }

        state = {
            "messages": [{"role": "user", "content": "Transform storage duration"}],
            "analysis_result": complex_analysis_result,
            "patches": [],
        }

        with patch("src.assistant.data_analyst.nodes._implementation_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm

            # Expected patches for one-to-many mapping
            mock_patches = [
                JsonPatch(op="add", path="/source_storage_duration_value", value=72),
                JsonPatch(op="add", path="/source_storage_duration_unit", value="hour"),
                JsonPatch(op="remove", path="/storage_duration"),
            ]
            mock_output = ImplementationOutput(patches=mock_patches)
            mock_structured_llm.invoke.return_value = mock_output

            result = implement_call(state)

            # Verify successful processing
            assert result.goto == "plan"
            assert len(result.update["patches"]) == 3

            # Verify complex analysis data was included in prompt
            call_args = mock_structured_llm.invoke.call_args[0][0]
            user_prompt = call_args[1]["content"]
            assert "Legacy: {\"storage_duration\": \"72 hours\"}" in user_prompt
            assert "Target: {\"source_storage_duration_value\": 72, \"source_storage_duration_unit\": \"hour\"}" in user_prompt
            assert "Mapping strategy: one-to-many" in user_prompt
            assert "Overall confidence: 0.9" in user_prompt
            assert "Reasoning: Composite value split into separate value and unit fields" in user_prompt

    def test_implement_call_handles_no_mapping_scenario(self):
        """Test with analysis result indicating no mapping found."""
        no_mapping_analysis = {
            "legacy_field": "deprecated_field",
            "legacy_value": "obsolete_value",
            "recommended_mappings": [],  # No mappings found
            "mapping_strategy": "one-to-one",
            "overall_confidence": 0.0,
            "reasoning": "No suitable target field found for this legacy field",
        }

        state = {
            "messages": [{"role": "user", "content": "Transform deprecated field"}],
            "analysis_result": no_mapping_analysis,
            "patches": [],
        }

        with patch("src.assistant.data_analyst.nodes._implementation_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm

            # Expected patch for removal only
            mock_patches = [JsonPatch(op="remove", path="/deprecated_field")]
            mock_output = ImplementationOutput(patches=mock_patches)
            mock_structured_llm.invoke.return_value = mock_output

            result = implement_call(state)

            # Should still succeed with removal operation
            assert result.goto == "plan"
            patches = result.update["patches"]
            assert len(patches) == 1
            assert patches[0].op == "remove"
            assert patches[0].path == "/deprecated_field"

    def test_implement_call_handles_malformed_analysis_result(self, base_state):
        """Test handling of malformed analysis result."""
        # Analysis result with missing required fields
        base_state["analysis_result"] = {"incomplete": "data"}

        with patch("src.assistant.data_analyst.nodes._implementation_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm

            mock_patches = [JsonPatch(op="add", path="/fallback", value="test")]
            mock_output = ImplementationOutput(patches=mock_patches)
            mock_structured_llm.invoke.return_value = mock_output

            result = implement_call(base_state)

            # Should handle gracefully with None values
            assert isinstance(result, Command)
            assert result.goto == "plan"

            # Verify prompt handled missing fields gracefully
            call_args = mock_structured_llm.invoke.call_args[0][0]
            user_prompt = call_args[1]["content"]
            assert "Legacy: {}" in user_prompt
            assert "Target: {}" in user_prompt

    def test_implement_call_preserves_patch_structure(self, base_state):
        """Test that JsonPatch objects maintain their structure through the process."""
        with patch("src.assistant.data_analyst.nodes._implementation_llm") as mock_llm:
            mock_structured_llm = Mock()
            mock_llm.with_structured_output.return_value = mock_structured_llm

            # Test all types of patch operations
            mock_patches = [
                JsonPatch(op="add", path="/new_field", value="new_value"),
                JsonPatch(op="remove", path="/old_field"),
                JsonPatch(op="replace", path="/existing_field", value="updated_value"),
                JsonPatch(op="move", **{"from": "/source_field"}, path="/target_field"),
                JsonPatch(
                    op="copy", **{"from": "/template_field"}, path="/copied_field"
                ),
                JsonPatch(op="test", path="/check_field", value="expected_value"),
            ]
            mock_output = ImplementationOutput(patches=mock_patches)
            mock_structured_llm.invoke.return_value = mock_output

            result = implement_call(base_state)

            patches = result.update["patches"]
            assert len(patches) == 6

            # Verify each patch type is preserved correctly
            add_patch = patches[0]
            assert add_patch.op == "add"
            assert add_patch.path == "/new_field"
            assert add_patch.value == "new_value"

            remove_patch = patches[1]
            assert remove_patch.op == "remove"
            assert remove_patch.path == "/old_field"
            assert remove_patch.value is None

            move_patch = patches[3]
            assert move_patch.op == "move"
            assert move_patch.from_ == "/source_field"
            assert move_patch.path == "/target_field"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
