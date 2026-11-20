from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, Field

from src.config import get_llm
from src.state import DebateState
from src.tools import search_web

MAX_SEARCHES_PER_TURN = 2


class DebateVerdict(BaseModel):
    winner: str = Field(description="Who argued stronger overall: 'Pro', 'Con', or 'Tie'")
    pro_score: int = Field(ge=1, le=10, description="Score for Pro's evidence and argument (1-10)")
    con_score: int = Field(ge=1, le=10, description="Score for Con's critique and negative outcomes (1-10)")
    key_negative_outcomes: list[str] = Field(
        description="Top 3-4 negative outcomes or risks identified by Con"
    )
    reasoning: str = Field(description="2-3 sentence explanation of the verdict")


def _extract_text(content: str | list) -> str:
    """Normalize string or content-block list responses into plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p.strip() for p in parts if p.strip())
    return str(content).strip()


def _clear_scratchpad(state: DebateState) -> list[RemoveMessage]:
    """Clear temporary tool-calling messages once a speaker completes a turn."""
    if not state.get("messages"):
        return []
    return [RemoveMessage(id=REMOVE_ALL_MESSAGES)]


def _llm_for_turn(state: DebateState):
    """Bind search_web only while the debater is still within the per-turn search budget."""
    llm = get_llm()
    if state.get("search_count", 0) < MAX_SEARCHES_PER_TURN:
        return llm.bind_tools([search_web])
    return llm


def pro_node(state: DebateState) -> dict:
    """Search online data (up to budget) and write a concise PRO argument."""
    model = _llm_for_turn(state)
    round_num = state.get("current_round", 1)
    searches_used = state.get("search_count", 0)
    remaining = max(0, MAX_SEARCHES_PER_TURN - searches_used)
    topic = state["topic"]
    con_prev = state.get("con_argument", "")

    system_msg = SystemMessage(
        content=(
            f"You are the PRO debater. Be crisp, factual, and concise (max ~180 words, 3 short bullets). "
            f"You have at most {remaining} web search(es) left this turn; once you have enough facts or "
            "0 searches left, write your final PRO argument directly."
        )
    )

    if not state.get("messages"):
        prompt = f"Topic: {topic} (Round {round_num}/{state['max_rounds']})\n"
        if con_prev:
            prompt += f"Counter CON briefly: {con_prev}\n"
        prompt += "Use search_web if helpful, then give 3 punchy points supporting PRO."
        human_msg = HumanMessage(content=prompt)
        response = model.invoke([system_msg, human_msg])

        if getattr(response, "tool_calls", None):
            return {"messages": [human_msg, response]}
    else:
        response = model.invoke([system_msg, *state["messages"]])
        if getattr(response, "tool_calls", None):
            return {"messages": [response]}

    arg_text = _extract_text(response.content)
    return {
        "pro_argument": arg_text,
        "transcript": [f"[Round {round_num} - PRO]\n{arg_text}"],
        "search_count": 0,
        "messages": _clear_scratchpad(state),
    }


def con_node(state: DebateState) -> dict:
    """Check PRO's argument, search counter-evidence (up to budget), and state negative outcomes."""
    model = _llm_for_turn(state)
    round_num = state.get("current_round", 1)
    searches_used = state.get("search_count", 0)
    remaining = max(0, MAX_SEARCHES_PER_TURN - searches_used)
    topic = state["topic"]
    pro_arg = state.get("pro_argument", "")

    system_msg = SystemMessage(
        content=(
            "You are the CON critic. Be sharp and concise (max ~180 words, 3 short bullets). "
            "Examine PRO's argument and highlight concrete negative outcomes, risks, or failure modes. "
            f"You have at most {remaining} web search(es) left this turn; do not over-search."
        )
    )

    if not state.get("messages"):
        prompt = (
            f"Topic: {topic} (Round {round_num}/{state['max_rounds']})\n"
            f"PRO argument:\n{pro_arg}\n\n"
            "Search online if needed for counter-evidence, then list the main NEGATIVE OUTCOMES clearly."
        )
        human_msg = HumanMessage(content=prompt)
        response = model.invoke([system_msg, human_msg])

        if getattr(response, "tool_calls", None):
            return {"messages": [human_msg, response]}
    else:
        response = model.invoke([system_msg, *state["messages"]])
        if getattr(response, "tool_calls", None):
            return {"messages": [response]}

    critique_text = _extract_text(response.content)
    return {
        "con_argument": critique_text,
        "transcript": [f"[Round {round_num} - CON (Negative Outcomes)]\n{critique_text}"],
        "current_round": round_num + 1,
        "search_count": 0,
        "messages": _clear_scratchpad(state),
    }


def judge_node(state: DebateState) -> dict:
    """Evaluate the transcript and return a concise structured verdict."""
    structured_llm = get_llm().with_structured_output(DebateVerdict)
    full_transcript = "\n\n---\n\n".join(state.get("transcript", []))

    verdict_obj: DebateVerdict = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a concise, impartial debate judge. Score PRO and CON (1-10) based on "
                    "evidence and how well CON identified real negative outcomes."
                )
            ),
            HumanMessage(
                content=f"Topic: {state['topic']}\n\nTranscript:\n{full_transcript}"
            ),
        ]
    )

    return {"verdict": verdict_obj.model_dump()}