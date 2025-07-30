from typing import List
from langgraph.graph import MessagesState
from .data_analyst.models import TargetSchema, AnalysisResult, PastMappingRecord, JsonPatch


class AppState(MessagesState):
    legacy_metadata: dict
    last_checked_field: str
    target_schema: TargetSchema
    analysis_result: AnalysisResult
    past_analysis: List[PastMappingRecord]
    patches: List[JsonPatch]
