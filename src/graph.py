from typing import Literal
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.nodes import con_node, judge_node, pro_node
from src.state import DebateState
from src.tools import search_web

_tool_executor = ToolNode([search_web])


def tools_node(state: DebateState) -> dict:
    """Run bound tools and increment search_count for the active turn."""
    result = _tool_executor.invoke(state)
    last_msg = state["messages"][-1] if state.get("messages") else None
    calls_count = len(getattr(last_msg, "tool_calls", None) or [])
    result["search_count"] = state.get("search_count", 0) + max(1, calls_count)
    return result


def _has_tool_calls(state: DebateState) -> bool:
    msgs = state.get("messages", [])
    if not msgs:
        return False
    return bool(getattr(msgs[-1], "tool_calls", None))


def route_after_pro(state: DebateState) -> Literal["tools", "con"]:
    return "tools" if _has_tool_calls(state) else "con"


def route_after_con(state: DebateState) -> Literal["tools", "pro", "judge"]:
    if _has_tool_calls(state):
        return "tools"
    if state.get("current_round", 1) <= state.get("max_rounds", 1):
        return "pro"
    return "judge"


def route_after_tools(state: DebateState) -> Literal["pro", "con"]:
    return "pro" if len(state.get("transcript", [])) % 2 == 0 else "con"


def build_debate_graph():
    graph = StateGraph(DebateState)

    graph.add_node("pro", pro_node)
    graph.add_node("con", con_node)
    graph.add_node("tools", tools_node)
    graph.add_node("judge", judge_node)

    graph.add_edge(START, "pro")
    graph.add_conditional_edges("pro", route_after_pro, {"tools": "tools", "con": "con"})
    graph.add_conditional_edges(
        "con",
        route_after_con,
        {"tools": "tools", "pro": "pro", "judge": "judge"},
    )
    graph.add_conditional_edges("tools", route_after_tools, {"pro": "pro", "con": "con"})
    graph.add_edge("judge", END)

    return graph.compile()