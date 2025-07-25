from langgraph.graph import MessagesState


class AppState(MessagesState):
    # TODO: Implement graph state
    legacy_metadata: dict
    last_checked_field: str
