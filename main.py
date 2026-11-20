import json
from src.graph import build_debate_graph


def run_debate(topic: str, max_rounds: int = 1) -> None:
    app = build_debate_graph()

    initial_state = {
        "topic": topic,
        "max_rounds": max_rounds,
        "current_round": 1,
        "pro_argument": "",
        "con_argument": "",
        "transcript": [],
        "messages": [],
        "verdict": None,
        "search_count": 0,
    }

    print(f"\n=== DEBATE TOPIC: {topic} (Rounds: {max_rounds}) ===\n")

    for event in app.stream(initial_state, stream_mode="updates"):
        for node_name, updates in event.items():
            if node_name == "tools":
                print("  [ToolNode] Fetched online search results.")
            elif updates.get("transcript"):
                for entry in updates["transcript"]:
                    print(f"\n{entry}\n{'-' * 60}")
            elif node_name == "judge" and updates.get("verdict"):
                print("\n=== FINAL VERDICT ===")
                print(json.dumps(updates["verdict"], indent=2))


if __name__ == "__main__":
    run_debate(
        topic="Should software teams rely on AI coding assistants as standard practice?",
        max_rounds=1,
    )