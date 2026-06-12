from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from tools.sheets_reader import get_google_client, get_tab_names, get_sheet_metadata
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    """
    Creates and returns the Groq LLM instance.
    We use llama-3.1-8b-instant — fast, free, more than capable for
    intent understanding and tab selection tasks.
    temperature=0 means the model gives consistent, deterministic answers.
    For intent parsing we never want randomness — we want the same question
    to always produce the same structured intent.
    """
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )


def agent1_understander(state: AgentState) -> AgentState:
    """
    Agent 1 — The Understander.

    Receives: user_message, available_tabs (with column info)
    Produces: structured_intent, selected_tab

    This agent does NOT touch the actual sheet data.
    It only looks at the sheet STRUCTURE (tab names + columns)
    to make an informed routing decision.
    """

    user_message = state["user_message"]

    # ── Step 1: Fetch sheet structure (tab names + columns) ───────────────────
    # We fetch this fresh every time so if someone renames a tab,
    # the agent automatically adapts — no hardcoded tab names in our code.
    # This satisfies PRD requirement R5 (auto-pick the right tab).
    try:
        client = get_google_client()
        all_tabs_data = get_sheet_metadata(client)

        # Build a readable summary of each tab: name + columns
        # This is what we'll pass to the LLM so it can make an informed choice.
        # We don't pass the actual row data — that's Agent 2's job.
        tab_summary_lines = []
        for tab_name, df in all_tabs_data.items():
            if df.empty or len(df.columns) == 0:
                # Skip tabs with no usable columns
                continue
            cols = ", ".join([c for c in df.columns if c.strip()])
            tab_summary_lines.append(f"- Tab: '{tab_name}' | Columns: {cols}")

        tab_summary = "\n".join(tab_summary_lines)

        # Also store tab names in state so other agents can reference them
        # without re-fetching — saves API calls to Google.
        available_tabs = list(all_tabs_data.keys())

    except Exception as e:
        # If we can't even read the sheet structure, there's nothing we can do.
        # Set the error in state and return early — Agent 3 will handle it.
        return {
            **state,
            "error": f"Could not read sheet structure: {str(e)}",
            "available_tabs": [],
            "structured_intent": None,
            "selected_tab": None
        }

    # ── Step 2: Ask the LLM to understand intent and pick the right tab ───────
    llm = get_llm()

    # The system prompt is the instruction manual for this agent.
    # We tell it exactly what role it plays and what format to output.
    # Being explicit about output format is critical — Agent 2 will parse this.
    system_prompt = """You are a data assistant that helps understand questions about a project management tracker in Google Sheets.

You will be given:
1. A user's question (possibly in English, Hindi, or Hinglish)
2. A list of available sheet tabs and their column names

Your job is to:
1. Understand what the user is asking for
2. Pick the SINGLE most relevant tab from the list
3. Write a clear, structured intent in English that describes exactly what data to find

Output format (always follow this exactly):
SELECTED_TAB: <exact tab name from the list>
STRUCTURED_INTENT: <clear description of what data to find, which columns are relevant, and any filters to apply>

Rules:
- SELECTED_TAB must be copied exactly as it appears in the tab list, including spaces
- STRUCTURED_INTENT should be in English regardless of the user's language
- If the question is about delayed/blocked projects, mention the relevant status/remarks columns
- If the question mentions a specific client or partner name, include it as a filter
- Be specific about which columns to look at"""

    human_prompt = f"""User question: {user_message}

Available tabs and columns:
{tab_summary}

Analyze the question and respond in the exact format specified."""

    try:
        # We pass two messages: SystemMessage sets the agent's role and rules,
        # HumanMessage is the actual input for this specific question.
        # This is the standard LangChain way to structure LLM conversations.
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        response_text = response.content

        # ── Step 3: Parse the LLM's response ─────────────────────────────────
        # We extract SELECTED_TAB and STRUCTURED_INTENT from the response text.
        # We're doing simple string parsing here — reliable and transparent.
        # No regex magic that's hard to debug.
        selected_tab = None
        structured_intent = None

        for line in response_text.strip().split("\n"):
            if line.startswith("SELECTED_TAB:"):
                # Strip the prefix and any surrounding whitespace
                selected_tab = line.replace("SELECTED_TAB:", "").strip()
            elif line.startswith("STRUCTURED_INTENT:"):
                structured_intent = line.replace("STRUCTURED_INTENT:", "").strip()

        # Fallback: if parsing failed, use the full response as intent
        # and pick the largest tab as a safe default.
        if not selected_tab or selected_tab not in all_tabs_data:
            # Default to the tab with the most rows — likely the main tracker
            selected_tab = max(all_tabs_data, key=lambda t: len(all_tabs_data[t]))

        if not structured_intent:
            structured_intent = user_message  # fallback to raw message

    except Exception as e:
        return {
            **state,
            "error": f"Agent 1 LLM call failed: {str(e)}",
            "available_tabs": available_tabs,
            "structured_intent": None,
            "selected_tab": None
        }

    # ── Step 4: Return updated state ─────────────────────────────────────────
    # We use {**state, ...} to copy all existing state fields and only
    # overwrite the fields this agent is responsible for.
    # Never overwrite fields you didn't set — other agents own those.
    return {
        **state,
        "available_tabs": available_tabs,
        "selected_tab": selected_tab,
        "structured_intent": structured_intent,
        "error": None
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # We simulate what the graph will do: create an initial state and run the agent
    test_state: AgentState = {
        "user_message": "Which projects are delayed and what are the blockers?",
        "structured_intent": None,
        "selected_tab": None,
        "available_tabs": None,
        "pandas_query": None,
        "raw_result": None,
        "final_answer": None,
        "error": None
    }

    print("Running Agent 1...")
    print(f"Input: {test_state['user_message']}\n")

    result = agent1_understander(test_state)

    print(f"Selected Tab: {result['selected_tab']}")
    print(f"Structured Intent: {result['structured_intent']}")
    print(f"Error: {result['error']}")
