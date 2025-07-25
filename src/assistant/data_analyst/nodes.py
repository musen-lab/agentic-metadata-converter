from ..graph import AppState


def analysis_call(state: AppState):
    """
    Analyzes the input legacy metadata by comparing it against the target schema and prior
    mappings to determine the updated field name and value. Generates a transformation
    instruction based on this analysis.
    """
    # TODO: Implement this
    return


def implement_call(state: AppState):
    """
    Translate the transformation instruction as a JSON-Patch operation, according to the
    RFC 6902 specification.
    """
    # TODO: Implement this
    return
