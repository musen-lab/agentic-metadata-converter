import json
from typing import Literal
from langchain.chat_models import init_chat_model
from langgraph.types import Command
from langgraph.graph import END
from ..graph import AppState
from .models import AnalysisOutput, ImplementationOutput
from .prompts import ANALYST_SYSTEM_PROMPT, IMPLEMENTOR_SYSTEM_PROMPT


# Initialize the LLM for analysis
_analysis_llm = init_chat_model("openai:gpt-4o", temperature=0.0)

# Initialize the LLM for implementation
_implementation_llm = init_chat_model("openai:gpt-4o", temperature=0.0)


def analysis_call(state: AppState) -> Command[Literal["implement", END]]:
    """
    Analyzes the input legacy metadata by comparing it against the target schema and prior
    mappings to determine the updated field name and value. Generates a transformation
    instruction based on this analysis.
    """
    # Extract messages from state (sent by plan_call)
    messages = state.get("messages", [])
    if not messages:
        print("No messages found in state for analysis")
        return Command(goto=END, update={})

    # Get the last message which should contain the legacy field/value to analyze
    last_message = messages[-1]
    message_content = last_message.get("content", "")

    # Get the current legacy field from the state
    legacy_field = state.get("last_checked_field")

    if not legacy_field:
        print("Found no legacy field needs to be analyzed")
        return Command(goto=END, update={})

    # Get target schema and past analysis from state
    target_schema = state.get("target_schema")
    past_analysis = state.get("past_analysis")

    if not target_schema:
        print("No target schema found in state")
        return Command(
            goto=END,
            update={
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"Analysis failed for {legacy_field}: No target schema available",
                    }
                ]
            },
        )

    # Convert Pydantic models to dict for JSON serialization in prompt
    target_schema_dict = (
        target_schema.model_dump()
        if hasattr(target_schema, "model_dump")
        else target_schema
    )
    past_analysis_dict = (
        past_analysis.model_dump()
        if hasattr(past_analysis, "model_dump")
        else past_analysis
    )

    # Format the analyst system prompt with target schema and past analysis
    system_prompt = ANALYST_SYSTEM_PROMPT.format(
        target_schema=json.dumps(target_schema_dict, indent=2),
        past_analysis=json.dumps(past_analysis_dict, indent=2),
    )

    # Pass the message as user prompt
    user_prompt = message_content

    # Get structured output from LLM
    llm = _analysis_llm.with_structured_output(AnalysisOutput)

    try:
        result = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        print(f"Analysis completed for {legacy_field}")
        print(f"Recommended mappings: {len(result.analysis.recommended_mappings)}")
        print(f"Overall confidence: {result.analysis.overall_confidence}")

        # Route to implement node with the analysis results
        goto = "implement"
        update = {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Analysis completed for {legacy_field}. Generated {len(result.analysis.recommended_mappings)} recommended mappings with overall confidence {result.analysis.overall_confidence}.",
                }
            ],
            # Store the analysis result for the implement node
            "analysis_result": result.analysis.model_dump(),
        }

        return Command(goto=goto, update=update)

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return Command(
            goto=END,
            update={
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"Analysis failed for {legacy_field}: {str(e)}",
                    }
                ]
            },
        )


def implement_call(state: AppState) -> Command[Literal["plan", END]]:
    """
    Translate the field mapping analysis results to JSON-Patch operations, according to the
    RFC 6902 specification.
    """
    # Extract messages from state (sent by analysis_call)
    messages = state.get("messages", [])
    if not messages:
        print("No messages found in state for implementation")
        return Command(goto=END, update={})

    # Get the analysis result from state
    analysis_result = state.get("analysis_result")
    if not analysis_result:
        print("No analysis result found in state for implementation")
        return Command(
            goto=END,
            update={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Implementation failed: No analysis result available",
                    }
                ]
            },
        )
    # Extract necessary analysis result information
    legacy_field = analysis_result.get("legacy_field", None)
    legacy_value = analysis_result.get("legacy_value", None)
    recommended_mappings = analysis_result.get("recommended_mappings", [])
    mapping_strategy = analysis_result.get("mapping_strategy", "one-to-one")
    overall_confidence = analysis_result.get("overall_confidence", 0.0)
    reasoning = analysis_result.get("reasoning", "No reasoning provided")

    # Format the user prompt with the analysis result details
    user_prompt = f"""
Based on the following analysis result, generate JSON Patch operations:

**Analysis Result:**
- Legacy: {json.dumps({legacy_field: legacy_value}) if legacy_field is not None else '{}'}
- Target: {json.dumps({mapping["target_field"]: mapping["target_value"] for mapping in recommended_mappings})}
- Mapping strategy: {mapping_strategy}
- Overall confidence: {overall_confidence}
- Reasoning: {reasoning}

Generate the appropriate RFC 6902 JSON Patch operations to transform the legacy metadata according to this analysis.
"""

    # Get structured output from LLM
    llm = _implementation_llm.with_structured_output(ImplementationOutput)

    try:
        result = llm.invoke(
            [
                {"role": "system", "content": IMPLEMENTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        legacy_field = analysis_result.get("legacy_field", None)
        num_patches = len(result.patches)

        print(f"Implementation completed for {legacy_field}")
        print(f"Generated {num_patches} JSON Patch operations")

        # Store patches in state and return success message
        # Get existing patches from state
        existing_patches = state.get("patches", [])
        return Command(
            goto="plan",
            update={
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"Implementation completed for {legacy_field}. Generated {num_patches} JSON Patch operations to transform the legacy metadata.",
                    }
                ],
                "patches": existing_patches + result.patches,
            },
        )

    except Exception as e:
        print(f"Error during implementation: {str(e)}")
        return Command(
            goto=END,
            update={
                "messages": [
                    {"role": "assistant", "content": f"Implementation failed: {str(e)}"}
                ]
            },
        )
