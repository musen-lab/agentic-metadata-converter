import pytest
from unittest.mock import Mock, patch
from langgraph.types import Command
from langgraph.graph import END
from src.assistant.manager.nodes import plan_call
from src.assistant.manager.models import ActionPlan


class TestPlanCall:
    """Test cases for the plan_call function."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock AppState for testing."""
        state = {
            "legacy_metadata": {
                "title": "Test Document",
                "author": "John Doe",
                "date": "2024-01-01",
            },
            "last_checked_field": "",
        }
        return state

    @patch("src.assistant.manager.nodes._plan_llm")
    def test_plan_call_analyze_action(self, mock_llm, mock_state):
        """Test plan_call when LLM returns analyze action."""
        # Mock the LLM response
        mock_structured_llm = Mock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mock_action_plan = ActionPlan(
            action="analyze", legacy_field="title", legacy_value="Test Document"
        )
        mock_structured_llm.invoke.return_value = mock_action_plan

        # Call the function
        result = plan_call(mock_state)

        # Assertions
        assert isinstance(result, Command)
        assert result.goto == "analysis"
        assert "messages" in result.update
        assert len(result.update["messages"]) == 1
        assert (
            "Analyze this legacy metadata field and value:"
            in result.update["messages"][0]["content"]
        )
        assert "**Legacy field**: title" in result.update["messages"][0]["content"]
        assert (
            "**Legacy value**: Test Document" in result.update["messages"][0]["content"]
        )

        # Verify LLM was called correctly
        mock_llm.with_structured_output.assert_called_once_with(ActionPlan)
        mock_structured_llm.invoke.assert_called_once()

        # Check the prompts passed to LLM
        llm_call_args = mock_structured_llm.invoke.call_args[0][0]
        assert len(llm_call_args) == 2
        assert llm_call_args[0]["role"] == "system"
        assert llm_call_args[1]["role"] == "user"

    @patch("src.assistant.manager.nodes._plan_llm")
    def test_plan_call_transform_action(self, mock_llm, mock_state):
        """Test plan_call when LLM returns transform action."""
        # Mock the LLM response
        mock_structured_llm = Mock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mock_action_plan = ActionPlan(
            action="transform",
            legacy_field="date",  # This should be ignored for transform action
            legacy_value="2024-01-01",
        )
        mock_structured_llm.invoke.return_value = mock_action_plan

        # Call the function
        result = plan_call(mock_state)

        # Assertions
        assert isinstance(result, Command)
        assert result.goto == END
        assert result.update == {}

        # Verify LLM was called correctly
        mock_llm.with_structured_output.assert_called_once_with(ActionPlan)
        mock_structured_llm.invoke.assert_called_once()

    @patch("src.assistant.manager.nodes._plan_llm")
    def test_plan_call_with_last_checked_field(self, mock_llm):
        """Test plan_call with a last_checked_field set."""
        state = {
            "legacy_metadata": {
                "title": "Test Document",
                "author": "John Doe",
                "date": "2024-01-01",
            },
            "last_checked_field": "title",
        }

        # Mock the LLM response
        mock_structured_llm = Mock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mock_action_plan = ActionPlan(
            action="analyze", legacy_field="author", legacy_value="John Doe"
        )
        mock_structured_llm.invoke.return_value = mock_action_plan

        # Call the function
        plan_call(state)

        # Verify the system prompt includes the last checked field
        llm_call_args = mock_structured_llm.invoke.call_args[0][0]
        system_prompt = llm_call_args[0]["content"]
        assert "title" in system_prompt

    @patch("src.assistant.manager.nodes._plan_llm")
    def test_plan_call_prompts_formatting(self, mock_llm, mock_state):
        """Test that prompts are formatted correctly with state data."""
        # Mock the LLM response
        mock_structured_llm = Mock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mock_action_plan = ActionPlan(
            action="analyze", legacy_field="title", legacy_value="Test Document"
        )
        mock_structured_llm.invoke.return_value = mock_action_plan

        # Call the function
        plan_call(mock_state)

        # Verify the prompts contain expected data
        llm_call_args = mock_structured_llm.invoke.call_args[0][0]
        system_prompt = llm_call_args[0]["content"]
        user_prompt = llm_call_args[1]["content"]

        # Check system prompt formatting
        assert "The last checked field is:" in system_prompt
        assert (
            mock_state["last_checked_field"] in system_prompt or "``" in system_prompt
        )

        # Check user prompt formatting
        assert "legacy metadata" in user_prompt.lower()
        assert "title" in user_prompt
        assert "Test Document" in user_prompt

    @patch("builtins.print")
    @patch("src.assistant.manager.nodes._plan_llm")
    def test_plan_call_prints_analyze_message(self, mock_llm, mock_print, mock_state):
        """Test that plan_call prints the expected message for analyze action."""
        # Mock the LLM response
        mock_structured_llm = Mock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mock_action_plan = ActionPlan(
            action="analyze", legacy_field="title", legacy_value="Test Document"
        )
        mock_structured_llm.invoke.return_value = mock_action_plan

        # Call the function
        plan_call(mock_state)

        # Verify print was called with the expected message
        mock_print.assert_called_once_with(
            "Action: ANALYZE - Continue to analyzing process"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
