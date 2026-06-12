from typing import TypedDict, Optional, Any


# AgentState is the shared baton passed between all three agents.
# Every agent reads from this, does its job, and writes its output back into it.
# TypedDict enforces the shape — no typos, no missing fields, no surprises.
class AgentState(TypedDict):

    # ── Set by the user ───────────────────────────────────────────────────────

    # The original, unmodified message the user typed.
    # We preserve this throughout the pipeline so Agent 3 can match
    # the tone and language (Hinglish in → Hinglish out).
    user_message: str

    # ── Set by Agent 1 (The Understander) ────────────────────────────────────

    # A clean, structured description of what the user wants.
    # Agent 1 rewrites the user's raw message into a precise intent
    # so Agent 2 doesn't have to guess from messy natural language.
    # Example: "Find all rows where Status contains 'delayed' and Client is 'Axis Bank'"
    structured_intent: Optional[str]

    # Which tab in the Google Sheet Agent 1 decided to use.
    # Example: "Brands and Aggregators - New Vision"
    # Agent 2 will only load this tab — no need to fetch all 49 tabs every time.
    selected_tab: Optional[str]

    # All available tab names from the sheet.
    # Agent 1 reads this to make an informed tab selection.
    # We populate this at the start of the pipeline, once, from the tool.
    available_tabs: Optional[list[str]]

    # ── Set by Agent 2 (The Data Engine) ─────────────────────────────────────

    # The actual pandas code Agent 2 wrote to answer the question.
    # We store this so Agent 3 can optionally show it ("here's how I calculated that").
    # Also useful for debugging — you can see exactly what query ran.
    pandas_query: Optional[str]

    # The raw result from running the pandas query on the real sheet data.
    # This is a plain Python object — could be an int (count), a list of dicts
    # (rows), or a string (error message). Agent 3 turns this into English.
    raw_result: Optional[Any]

    # ── Set by Agent 3 (The Communicator) ────────────────────────────────────

    # The final answer in plain English (or Hinglish, matching user's language).
    # This is what gets shown to the user at the end.
    final_answer: Optional[str]

    # ── Error handling ────────────────────────────────────────────────────────

    # If anything goes wrong at any stage, we set this field instead of crashing.
    # Agent 3 checks this first — if it's set, it tells the user something went
    # wrong in plain language instead of showing a Python traceback.
    error: Optional[str]
