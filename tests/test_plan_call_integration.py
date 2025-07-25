import pytest
import os
from langgraph.types import Command
from langgraph.graph import END
from src.assistant.manager.nodes import plan_call


class TestPlanCallIntegration:
    """Integration tests for plan_call that use real OpenAI API."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock AppState for testing."""
        state = {
            "legacy_metadata": {
                "title": "Legacy Document Title",
                "author": "John Smith",
                "date": "2024-01-15",
                "description": "This is an old document format",
                "category": "research",
            },
            "last_checked_field": "",
        }
        return state

    @pytest.fixture
    def state_with_last_checked(self):
        """Create a state with last_checked_field set."""
        state = {
            "legacy_metadata": {
                "title": "Legacy Document Title",
                "author": "John Smith",
                "date": "2024-01-15",
                "description": "This is an old document format",
                "category": "research",
            },
            "last_checked_field": "title",
        }
        return state

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_plan_call_real_api_analyze_action(self, mock_state):
        """Test plan_call with real OpenAI API - expecting analyze action."""
        # This calls the real API
        result = plan_call(mock_state)

        # Verify the result structure
        assert isinstance(result, Command)
        assert result.goto in ["analysis", END]

        if result.goto == "analysis":
            # Should have messages for analysis
            assert "messages" in result.update
            assert len(result.update["messages"]) >= 1

            message_content = result.update["messages"][0]["content"]
            assert "Analyze this legacy metadata field and value:" in message_content
            assert "**Legacy field**:" in message_content
            assert "**Legacy value**:" in message_content

        print(f"API returned action: {result.goto}")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_plan_call_real_api_with_last_checked(self, state_with_last_checked):
        """Test plan_call with real OpenAI API when last_checked_field is set."""
        # This calls the real API
        result = plan_call(state_with_last_checked)

        # Verify the result structure
        assert isinstance(result, Command)
        assert result.goto in ["analysis", END]

        print(f"API returned action with last_checked_field: {result.goto}")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_plan_call_real_api_multiple_calls(self, mock_state):
        """Test multiple calls to plan_call to verify consistency."""
        results = []

        # Make multiple API calls
        for i in range(3):
            result = plan_call(mock_state.copy())
            results.append(result)
            assert isinstance(result, Command)
            assert result.goto in ["analysis", END]

        # Print results for manual verification
        for i, result in enumerate(results):
            print(f"Call {i + 1}: {result.goto}")

        # All should be Command objects
        assert all(isinstance(r, Command) for r in results)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable required for integration tests",
    )
    def test_plan_call_real_api_empty_metadata(self):
        """Test plan_call with empty metadata."""
        empty_state = {
            "legacy_metadata": {},
            "last_checked_field": "",
        }

        result = plan_call(empty_state)

        # Should handle empty metadata gracefully
        assert isinstance(result, Command)
        assert result.goto in ["analysis", END]

        print(f"Empty metadata result: {result.goto}")


if __name__ == "__main__":
    # Run only integration tests
    pytest.main([__file__, "-v", "-m", "integration"])
