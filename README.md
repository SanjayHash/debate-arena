# Multi-Agent Debate Arena

A live, streaming multi-agent debate application built with **LangGraph**, **LangChain**, **Google Gemini (`gemini-3.5-flash-lite`)**, **DuckDuckGo Search (`ddgs`)**, and **Streamlit**. Visitors can enter any custom debate topic (or pick a quick sample preset) and watch **Pro**, **Con**, and **Judge** agents execute a structured 1-round debate with live web grounding and a scored verdict.

live link  : https://debate-arena-hbpscrqhqmdcnvm7x8mvbn.streamlit.app

## Architecture Diagram

```text
+-------------------------------------------------------------------+
|                     Streamlit UI (app.py)                         |
|  - Custom topic input + sample presets + "Start Debate" trigger   |
|  - Live streaming status & debate feed via graph.stream()         |
|  - Final Judge verdict scorecard (Winner, Pro/Con scores, risks)  |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|              LangGraph StateGraph (src/graph.py)                  |
|  Shared DebateState (src/state.py): topic, messages, pro_argument,|
|  con_argument, transcript, verdict, search_count, max_rounds      |
|                                                                   |
|   [START] --> pro --> (tool call?) --yes--> tools --> pro         |
|                |                                                  |
|                +------no-----> con --> (tool call?) --yes--> tools|
|                                 |                             |   |
|                                 +------no-----> judge --> [END]   |
+----------------+---------------------------------+----------------+
                 |                                 |
                 v                                 v
+--------------------------------+  +------------------------------+
| LLM & Tooling Layer            |  | External Services            |
| - src/config.py:               |  | - Google Gemini API          |
|   ChatGoogleGenerativeAI       |  |   (gemini-3.5-flash-lite)    |
|   + fallback + st.secrets sync |  | - DuckDuckGo Search (ddgs)   |
| - src/nodes.py:                |  +------------------------------+
|   Pro, Con, Judge & routing    |
| - src/tools.py:                |
|   @tool web_search with error  |
|   handling & search cap        |
+--------------------------------+
```

---

## Data Flow Diagram

```text
User enters topic (or clicks sample preset) & clicks "Start Debate"
        |
        v
app.py builds initial DebateState (max_rounds=1, search_count=0)
        |
        v
1. PRO NODE
   - Calls Gemini with topic + tool binding (capped at 1 search/turn)
   - If tool requested & under search cap --> TOOLS NODE runs ddgs search
     (catches search errors gracefully so debate never crashes)
   - Returns results to PRO NODE --> streams Pro Argument to UI
        |
        v
2. CON NODE
   - Reads topic + Pro's argument
   - Optionally runs 1 web search via TOOLS NODE to ground critique
   - Streams Con Critique (focusing on negative outcomes & trade-offs)
        |
        v
3. JUDGE NODE
   - Evaluates topic, Pro argument, Con critique, and transcript
   - Produces structured verdict (winner, pro_score, con_score,
     key_negative_outcomes, reasoning)
   - Streams final scorecard to UI & finishes graph execution
```

---

## Project Structure

```text
debate-arena/
├── .gitignore
├── app.py               # Streamlit UI, live event streamer & verdict view
├── main.py              # CLI / local entry point
├── requirements.txt     # Python dependencies
└── src/
    ├── __init__.py
    ├── config.py        # Gemini LLM factory, fallback & st.secrets loader
    ├── graph.py         # LangGraph StateGraph wiring & conditional edges
    ├── nodes.py         # Pro, Con, Judge nodes & turn/search limits
    ├── state.py         # TypedDict state schema for the debate graph
    └── tools.py         # Web search tool (@tool) with exception handling
```

---

## Key Design Decisions

* **Free-Tier Quota Protection:** Debates are locked to **1 round** (`MAX_ROUNDS = 1`) with a **1-search-per-turn** cap to stay comfortably within free-tier rate and daily limits on `gemini-3.5-flash-lite` (500 requests/day, 15 requests/minute on standard free quota)[cite: 2, 5].
* **Resilient Tool Calling:** Web search errors or empty results on cloud IPs are caught in `src/tools.py` and handled cleanly by the agents without halting the graph.
* **Seamless Local & Cloud Secrets:** `src/config.py` loads `.env` for local runs and syncs `st.secrets` into environment variables when hosted on Streamlit Community Cloud.

---

## Local Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/SanjayHash/debate-arena.git](https://github.com/SanjayHash/debate-arena.git)
   cd debate-arena
   ```

2. **Create & activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root (do not commit it):
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-3.5-flash-lite
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

---

## Deployment (Streamlit Community Cloud)

1. Push your latest changes to the `main` branch on GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing to `SanjayHash/debate-arena`, branch `main`, and main file `app.py`.
3. Under **Advanced settings** (or **App settings -> Secrets**), configure:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key_here"
   GEMINI_MODEL = "gemini-3.5-flash-lite"
   ```
4. Select **Python 3.12** and click **Deploy**.
