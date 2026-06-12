from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from dotenv import load_dotenv
import os
import json

load_dotenv()


def get_llm():
    """
    Agent 3 uses slightly higher temperature than Agents 1 and 2.
    temperature=0.3 allows natural, human-sounding language variation
    while still staying factual and grounded.
    Agents 1 and 2 needed temperature=0 for deterministic logic.
    Agent 3 needs a touch of warmth — it's talking to a human.
    """
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )


def format_raw_result(raw_result) -> str:
    """
    Converts the raw_result from Agent 2 into a readable string
    that we can safely pass to the LLM.

    Why this step? The raw_result could be:
    - An integer: 14
    - A list of dicts: [{"Partner": "Axis", "Status": "Delayed"}, ...]
    - A string error message: "Query execution error: ..."
    - None: if something went wrong silently

    The LLM needs a clean string, not a Python object.
    We cap list results at 20 items to avoid hitting token limits.
    """
    if raw_result is None:
        return "No data was returned."

    if isinstance(raw_result, int) or isinstance(raw_result, float):
        return str(raw_result)

    if isinstance(raw_result, str):
        return raw_result

    if isinstance(raw_result, list):
        if len(raw_result) == 0:
            return "No matching rows found."
        # Cap at 20 rows to stay within token limits.
        # 📌 REVISIT LATER: For very large results, summarize instead of truncate.
        capped = raw_result[:20]
        result_str = json.dumps(capped, indent=2, default=str)
        if len(raw_result) > 20:
            result_str += f"\n\n... and {len(raw_result) - 20} more rows."
        return result_str

    # Fallback for any other type — just convert to string
    return str(raw_result)


def agent3_communicator(state: AgentState) -> AgentState:
    """
    Agent 3 — The Communicator.

    Receives: user_message, structured_intent, raw_result, error (from state)
    Produces: final_answer

    This agent's only job is to turn raw data into a clear, human answer.
    It has access to both the original user message (for language/tone matching)
    and the structured intent (for understanding what was actually asked).
    """

    # ── Handle error state first ──────────────────────────────────────────────
    # If any previous agent set an error, we don't run the normal flow.
    # Instead we tell the user something went wrong in plain language.
    # This is PRD R7: never crash silently, always communicate honestly.
    if state.get("error"):
        error_msg = state["error"]
        final_answer = (
            f"Sorry, I ran into an issue while trying to answer your question: "
            f"{error_msg}. Please try rephrasing or check if the sheet is accessible."
        )
        return {**state, "final_answer": final_answer}

    user_message = state["user_message"]
    structured_intent = state.get("structured_intent", user_message)
    raw_result = state.get("raw_result")
    selected_tab = state.get("selected_tab", "the tracker")

    # Convert raw result to a clean readable string for the LLM
    formatted_result = format_raw_result(raw_result)

    llm = get_llm()

    system_prompt = """You are a helpful assistant that explains data query results to users in plain, conversational language.

Your job:
1. Look at the user's original question and the raw data result
2. Write a clear, direct answer in the SAME language and tone as the user's question
3. If the user wrote in Hindi or Hinglish, reply in Hinglish
4. If the user wrote in English, reply in English
5. If the result is a list of rows, summarize the key points clearly — don't just dump all the data
6. If no rows were found, say so honestly and suggest why (maybe different status wording, etc.)
7. If the result is a count, state it clearly with context
8. Never make up data that isn't in the result
9. Keep the answer concise but complete — no unnecessary padding

Tone: helpful, direct, like a smart colleague answering a Slack message."""

    human_prompt = f"""User's original question: {user_message}

What was searched: {structured_intent}
Data source: '{selected_tab}' tab

Raw result from the data:
{formatted_result}

Write a clear, conversational answer to the user's question based on this data."""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        final_answer = response.content.strip()

    except Exception as e:
        # Even if Agent 3 itself fails, we still give the user something useful
        # rather than a Python traceback — raw result is better than nothing.
        final_answer = (
            f"I found the data but had trouble formatting the response. "
            f"Raw result: {formatted_result}"
        )

    return {
        **state,
        "final_answer": final_answer
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: Normal result with data
    print("=== Test 1: English question, list result ===")
    test_state_1: AgentState = {
        "user_message": "Which projects are delayed and what are the blockers?",
        "structured_intent": "Find rows where Status contains delay. Return Partner, Description, Status, Risk columns.",
        "selected_tab": "Brands and Aggregators - New Vision",
        "available_tabs": None,
        "pandas_query": None,
        "raw_result": [
            {"Partner": "Axis Bank", "Description": "Loyalty integration", "Status": "Delayed", "Risk/dependencies/challenges": "UAT pending from client side"},
            {"Partner": "HDFC", "Description": "POS integration", "Status": "Delayed - client approval", "Risk/dependencies/challenges": "Infra issue on client end"},
        ],
        "final_answer": None,
        "error": None
    }
    result1 = agent3_communicator(test_state_1)
    print(result1["final_answer"])

    # Test 2: Hinglish question, zero results
    print("\n=== Test 2: Hinglish question, zero results ===")
    test_state_2: AgentState = {
        "user_message": "Axis Bank ke delayed projects kaunse hain?",
        "structured_intent": "Find rows where Partner contains Axis Bank and Status contains delay.",
        "selected_tab": "Brands and Aggregators - New Vision",
        "available_tabs": None,
        "pandas_query": None,
        "raw_result": [],
        "final_answer": None,
        "error": None
    }
    result2 = agent3_communicator(test_state_2)
    print(result2["final_answer"])

    # Test 3: Error state
    print("\n=== Test 3: Error from previous agent ===")
    test_state_3: AgentState = {
        "user_message": "How many projects are live?",
        "structured_intent": None,
        "selected_tab": None,
        "available_tabs": None,
        "pandas_query": None,
        "raw_result": None,
        "final_answer": None,
        "error": "Could not connect to Google Sheets"
    }
    result3 = agent3_communicator(test_state_3)
    print(result3["final_answer"])
