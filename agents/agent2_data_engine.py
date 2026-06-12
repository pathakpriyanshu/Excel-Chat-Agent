from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from tools.sheets_reader import get_google_client, get_sheet_metadata
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()


def get_llm():
    """
    Same model as Agent 1 — llama-3.1-8b-instant, temperature=0.
    temperature=0 is especially important here: we need the LLM to write
    deterministic, correct pandas code every time. Any randomness in code
    generation leads to syntax errors or wrong logic.
    """
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )


def execute_pandas_query(df: pd.DataFrame, query_code: str) -> any:
    """
    Executes the pandas query code written by the LLM on the real DataFrame.

    This is the KEY function in the entire system — this is where AI guessing
    stops and real code takes over. The result of this function is always
    mathematically correct because it runs on actual data.

    Why exec() and not eval()?
    - eval() only works for single expressions that return a value
    - exec() can run multi-line code blocks
    - We capture the result via a local 'result' variable convention

    Why is this safe here?
    - The code runs locally on data we already fetched
    - It has no network access, no file writes
    - In production we'd add a sandbox — flagged below

    📌 REVISIT LATER: In production, wrap this in a restricted exec environment
    or use a code sandbox to prevent any malicious query from doing harm.
    """
    # We give the LLM's code access to 'df' (the DataFrame) and 'pd' (pandas).
    # The code must store its answer in a variable called 'result'.
    local_vars = {"df": df, "pd": pd, "result": None}

    try:
        exec(query_code, {}, local_vars)
        return local_vars["result"]
    except Exception as e:
        return f"Query execution error: {str(e)}"


def agent2_data_engine(state: AgentState) -> AgentState:
    """
    Agent 2 — The Data Engine.

    Receives: structured_intent, selected_tab (from Agent 1)
    Produces: pandas_query, raw_result

    Flow:
      1. Load the selected tab from Google Sheets as a DataFrame
      2. Ask LLM to write a pandas query based on intent + column info
      3. Execute that query on the real data
      4. Store the result in state
    """

    # If Agent 1 already encountered an error, skip and pass the error forward.
    # We never want Agent 2 to run on broken state — it would produce garbage.
    if state.get("error"):
        return state

    structured_intent = state["structured_intent"]
    selected_tab = state["selected_tab"]

    # ── Step 1: Load the selected tab as a DataFrame ──────────────────────────
    try:
        client = get_google_client()
        all_tabs = get_sheet_metadata(client)

        if selected_tab not in all_tabs:
            return {
                **state,
                "error": f"Tab '{selected_tab}' not found or could not be loaded."
            }

        df = all_tabs[selected_tab]

        # Build a detailed column description to give the LLM context.
        # We show column names + a sample of unique values for key columns.
        # This helps the LLM write accurate filter conditions.
        # Example: Status column might have "In Progress", "Delayed", "On Hold"
        col_info_lines = []
        for col in df.columns:
            if not col.strip():
                continue
            # Show up to 5 unique non-empty values as examples
            sample_vals = df[col].dropna().astype(str)
            sample_vals = [v for v in sample_vals.unique() if v.strip()][:5]
            col_info_lines.append(f"  - '{col}': sample values → {sample_vals}")

        col_info = "\n".join(col_info_lines)

    except Exception as e:
        return {
            **state,
            "error": f"Could not load tab '{selected_tab}': {str(e)}"
        }

    # ── Step 2: Ask LLM to write a pandas query ───────────────────────────────
    llm = get_llm()

    # We list exact column names explicitly at the top of the prompt.
    # This directly prevents the LLM from hallucinating column names
    # like "Project name" when the real column is "Description".
    exact_columns = [c for c in df.columns if c.strip()]

    system_prompt = """You are a Python/pandas expert. You write pandas code to answer questions about DataFrames.

You will be given:
1. A structured intent describing what data to find
2. The EXACT column names available in the DataFrame — use ONLY these, never invent new ones
3. Sample values per column to help you write correct filters

Write a pandas query to answer the question. The DataFrame is already loaded as 'df'.

Rules:
- Always store the final answer in a variable called 'result'
- Use ONLY the exact column names provided — copy them character for character
- Use case-insensitive string matching with .str.contains(..., case=False, na=False)
- For counts, result should be an integer: result = len(filtered_df)
- For lists of rows, result should be a list of dicts: result = filtered_df[cols].to_dict('records')
- Handle NaN values gracefully with na=False in str.contains
- Do NOT import anything — pandas is already available as 'pd'
- Write only the Python code, no explanation, no markdown, no backticks"""

    human_prompt = f"""Intent: {structured_intent}

DataFrame tab: '{selected_tab}'

EXACT column names (use these only, copy exactly):
{exact_columns}

Column details and sample values:
{col_info}

Write the pandas code to answer this. Store the answer in 'result'."""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        # Clean up the response — sometimes LLMs add markdown code fences
        # even when told not to. We strip those defensively.
        pandas_query = response.content.strip()
        pandas_query = pandas_query.replace("```python", "").replace("```", "").strip()

    except Exception as e:
        return {
            **state,
            "error": f"Agent 2 LLM call failed: {str(e)}"
        }

    # ── Step 3: Execute the query on real data ────────────────────────────────
    # This is the moment where AI stops and math takes over.
    # Whatever 'result' is after this — it's a real number from real data.
    raw_result = execute_pandas_query(df, pandas_query)

    # ── Step 4: Return updated state ──────────────────────────────────────────
    return {
        **state,
        "pandas_query": pandas_query,
        "raw_result": raw_result,
        "error": None
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate state as if Agent 1 already ran
    test_state: AgentState = {
        "user_message": "Which projects are delayed and what are the blockers?",
        "structured_intent": "Find all rows where 'Status' column contains 'delay' or 'delayed'. Return columns: 'Partner', 'Description', 'Status', 'Risk/dependencies/challenges'.",
        "selected_tab": "Brands and Aggregators - New Vision",
        "available_tabs": None,
        "pandas_query": None,
        "raw_result": None,
        "final_answer": None,
        "error": None
    }

    print("Running Agent 2...")
    result = agent2_data_engine(test_state)

    print(f"\nPandas Query Written:\n{result['pandas_query']}")
    print(f"\nRaw Result (first 3 items):")

    raw = result["raw_result"]
    if isinstance(raw, list):
        for item in raw[:3]:
            print(f"  {item}")
        print(f"  ... total: {len(raw)} rows")
    else:
        print(f"  {raw}")

    print(f"\nError: {result['error']}")
