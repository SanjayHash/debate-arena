import streamlit as st
from langchain_google_genai.chat_models import GoogleAPIError
from src.graph import build_debate_graph

MAX_ROUNDS = 1

st.set_page_config(page_title="Multi-Agent Debate Arena", layout="wide")

st.title("Multi-Agent Debate Arena")
st.caption("LangGraph StateGraph + LangChain Tool Calling + Gemini 3.5 Flash Lite")

with st.sidebar:
    st.header("How It Works")
    st.markdown(
        "**Format:** 1-Round Structured Debate\n\n"
        "1. **Pro** searches the web & argues in favor\n"
        "2. **Con** inspects Pro's case, searches & highlights negative outcomes\n"
        "3. **Judge** scores both sides & returns a structured verdict"
    )

topic = st.text_input(
    "Debate Topic",
    value="Should software teams rely on AI coding assistants as standard practice?",
)

if st.button("Start Debate", type="primary", disabled=not topic.strip()):
    graph = build_debate_graph()

    initial_state = {
        "topic": topic.strip(),
        "max_rounds": MAX_ROUNDS,
        "current_round": 1,
        "pro_argument": "",
        "con_argument": "",
        "transcript": [],
        "messages": [],
        "verdict": None,
        "search_count": 0,
    }

    transcript_container = st.container()
    verdict_container = st.container()

    try:
        with st.status("Running multi-agent debate...", expanded=True) as status:
            for event in graph.stream(initial_state, stream_mode="updates"):
                for node_name, updates in event.items():
                    if node_name == "tools":
                        st.write("🔍 **ToolNode:** Fetched live web search results...")
                    elif node_name == "pro" and updates.get("pro_argument"):
                        st.write("✅ **Pro** finished round argument.")
                        with transcript_container:
                            st.subheader("Pro Argument")
                            st.markdown(updates["pro_argument"])
                            st.divider()
                    elif node_name == "con" and updates.get("con_argument"):
                        st.write("✅ **Con** finished negative-outcomes critique.")
                        with transcript_container:
                            st.subheader("Con Critique (Negative Outcomes)")
                            st.markdown(updates["con_argument"])
                            st.divider()
                    elif node_name == "judge" and updates.get("verdict"):
                        verdict = updates["verdict"]
                        st.write("⚖️ **Judge** delivered final verdict.")
                        with verdict_container:
                            st.subheader("Final Judge Verdict")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Winner", verdict["winner"])
                            c2.metric("Pro Score", f"{verdict['pro_score']} / 10")
                            c3.metric("Con Score", f"{verdict['con_score']} / 10")

                            st.markdown("**Key Negative Outcomes Identified:**")
                            for item in verdict.get("key_negative_outcomes", []):
                                st.markdown(f"- {item}")

                            st.info(verdict["reasoning"])

            status.update(label="Debate complete!", state="complete", expanded=False)

    except GoogleAPIError as exc:
        st.warning(
            f"Temporary Gemini API demand or rate-limit issue: {exc}\n\n"
            "Wait a few seconds and click **Start Debate** again."
        )