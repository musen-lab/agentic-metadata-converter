from langgraph.graph import StateGraph
from assistant import AppState
from assistant.manager.nodes import plan_call, execute_call, tool_handler
from assistant.data_analyst.nodes import analysis_call, implement_call


workflow = StateGraph(AppState)

# Add nodes to the graph
workflow.add_node("plan", plan_call)
workflow.add_node("execute", execute_call)
workflow.add_node("tool_handler", tool_handler)
workflow.add_node("analysis", analysis_call)
workflow.add_node("implement", implement_call)

workflow.set_entry_point("plan")

workflow.compile()
