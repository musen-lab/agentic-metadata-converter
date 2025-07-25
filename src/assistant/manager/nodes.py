from typing import Literal
from langchain.chat_models import init_chat_model
from langgraph.types import Command
from langgraph.graph import END
from ..graph import AppState
from ..utils import format_legacy_record_markdown
from .models import ActionPlan
from .prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT


# Initialize the LLM
_plan_llm = init_chat_model("openai:gpt-4o-mini", temperature=0.0)


def plan_call(state: AppState) -> Command[Literal["analysis", END]]:
    """
    Distributes the legacy metadata content for analysis in order to generate
    the appropriate transformation rules (as patches). Once all content has
    been analyzed, call the executor to apply the patches.
    """
    legacy_metadata = state["legacy_metadata"]
    last_checked_field = state["last_checked_field"]

    system_prompt = PLANNER_SYSTEM_PROMPT.format(last_checked_field=last_checked_field)
    user_prompt = PLANNER_USER_PROMPT.format(legacy_metadata=legacy_metadata)

    llm = _plan_llm.with_structured_output(ActionPlan)

    result = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    goto = END
    update = {}
    if result.action == "analyze":
        print("Action: ANALYZE - Continue to analyzing process")

        legacy_field = result.legacy_field
        legacy_value = result.legacy_value

        goto = "analysis"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze this legacy metadata field and value:\n{format_legacy_record_markdown(legacy_field, legacy_value)}",
                }
            ]
        }
        # Update the last check field in the state
        state["last_checked_field"] = legacy_field

    return Command(goto=goto, update=update)


def execute_call(state: AppState):
    """
    Applies the patches agains the legacy metadata to produce an updated version.
    """
    # TODO: Implement this
    return


def tool_handler(state: AppState):
    """Performs the tool call."""
    # TODO: Implement this
    return
