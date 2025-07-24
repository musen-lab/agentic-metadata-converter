from assistant.graph import AppState


def plan_call(state: AppState):
    """
    Distributes the legacy metadata content for analysis in order to generate
    the appropriate transformation rules (as patches). Once all content has
    been analyzed, call the executor to apply the patches.
    """
    # TODO: Implement this
    return


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
