import operator
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class DebateState(TypedDict):
    # Debate setup & progress
    topic: str
    max_rounds: int
    current_round: int
    search_count: int

    # Latest arguments produced in the active/most recent round
    pro_argument: str
    con_argument: str

    # Full chronological log of arguments (appends via operator.add)
    transcript: Annotated[list[str], operator.add]

    # Scratchpad for LangChain tool calling (AIMessage tool_calls + ToolMessage results)
    messages: Annotated[list[BaseMessage], add_messages]

    # Final structured verdict from the Judge node
    verdict: dict | None