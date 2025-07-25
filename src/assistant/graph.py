from typing import Dict, Any, Optional
from langgraph.graph import MessagesState
from .data_analyst.models import TargetSchema, PastAnalysis


class AppState(MessagesState):
    # TODO: Implement graph state
    legacy_metadata: dict
    target_schema: TargetSchema
    past_analysis: PastAnalysis
    last_checked_field: str
    analysis_result: Optional[Dict[str, Any]] = None  # Stores analysis results for implement_call
