import json
from typing import Literal
from langchain.chat_models import init_chat_model
from langgraph.types import Command
from langgraph.graph import END
from ..graph import AppState
from .models import AnalysisOutput
from .prompts import ANALYST_SYSTEM_PROMPT


# Initialize the LLM for analysis
_analysis_llm = init_chat_model("openai:gpt-4o", temperature=0.0)


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
        return Command(goto=END, update={
            "messages": [{"role": "assistant", "content": f"Analysis failed for {legacy_field}: No target schema available"}]
        })
    
    # Convert Pydantic models to dict for JSON serialization in prompt
    target_schema_dict = target_schema.model_dump() if hasattr(target_schema, 'model_dump') else target_schema
    past_analysis_dict = past_analysis.model_dump() if hasattr(past_analysis, 'model_dump') else past_analysis
    
    # Format the analyst system prompt with target schema and past analysis
    system_prompt = ANALYST_SYSTEM_PROMPT.format(
        target_schema=json.dumps(target_schema_dict, indent=2),
        past_analysis=json.dumps(past_analysis_dict, indent=2)
    )
    
    # Pass the message as user prompt
    user_prompt = message_content
    
    # Get structured output from LLM
    llm = _analysis_llm.with_structured_output(AnalysisOutput)
    
    try:
        result = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        print(f"Analysis completed for {legacy_field}")
        print(f"Recommended mappings: {len(result.analysis.recommended_mappings)}")
        print(f"Overall confidence: {result.analysis.overall_confidence}")
        
        # Route to implement node with the analysis results
        goto = "implement"
        update = {
            "messages": [
                {
                    "role": "assistant", 
                    "content": f"Analysis completed for {legacy_field}. Generated {len(result.analysis.recommended_mappings)} recommended mappings with overall confidence {result.analysis.overall_confidence}."
                }
            ],
            # Store the analysis result for the implement node
            "analysis_result": result.analysis.model_dump()
        }
        
        return Command(goto=goto, update=update)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return Command(goto=END, update={
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Analysis failed for {legacy_field}: {str(e)}"
                }
            ]
        })


def implement_call(state: AppState):
    """
    Translate the transformation instruction as a JSON-Patch operation, according to the
    RFC 6902 specification.
    """
    # TODO: Implement this
    return
