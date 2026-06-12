import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
import os

# Load environment variables from .env file
# This is why we use python-dotenv — so secrets never live in code,
# they live in .env which is gitignored
load_dotenv()

# These two permission scopes tell Google what our app is allowed to do.
# "spreadsheets.readonly" = can read cell data but never write or delete.
# "drive.readonly" = can find and open the file, but never move or delete it.
# We request the minimum permissions needed — this is a security best practice.
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly"
]


def get_google_client():
    """
    Creates and returns an authenticated gspread client.
    Think of this as logging in as the robot service account.
    We call this once and reuse the client for all sheet operations.
    """
    # Load the path to our credentials JSON from .env
    creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH")

    # Credentials.from_service_account_file reads the JSON key file and
    # creates an auth token Google will accept — like showing your library card.
    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)

    # gspread.authorize wraps the credentials into a client object
    # that knows how to make Google Sheets API calls for us.
    client = gspread.authorize(creds)

    return client


def get_sheet_metadata(client):
    """
    Opens the Google Sheet and returns a dict of:
      { tab_name: pandas_dataframe }
    One DataFrame per tab/worksheet in the sheet.

    Why return all tabs? Because Agent 1 needs to know what tabs exist
    so it can pick the right one for each question (PRD requirement R5).
    """
    sheet_url = os.getenv("SHEET_URL")

    # open_by_url uses the Drive API to locate the file by URL.
    # This is why we need Drive API enabled — without it, this line fails.
    spreadsheet = client.open_by_url(sheet_url)

    # spreadsheet.worksheets() returns a list of all tabs in the sheet.
    # We loop through every tab and convert it to a pandas DataFrame.
    all_tabs = {}

    for worksheet in spreadsheet.worksheets():
        tab_name = worksheet.title

        try:
            # get_all_records() fetches every row as a list of dicts,
            # where the first row (header) becomes the dict keys.
            # Example: [{"Project Name": "Alpha", "Status": "Delayed"}, ...]
            records = worksheet.get_all_records()

            # pandas.DataFrame converts that list of dicts into a table structure
            # that we can filter, count, and query with code — not AI guesses.
            df = pd.DataFrame(records)
            all_tabs[tab_name] = df

        except Exception as e:
            # Some tabs may have duplicate column names or formatting issues.
            # We skip them gracefully and log a warning instead of crashing.
            # This is PRD R7 in action: flag the problem, don't silently skip
            # and don't crash the whole system for one bad tab.
            print(f"  [SKIPPED] Tab '{tab_name}' has formatting issues: {type(e).__name__}")
            continue

    return all_tabs


def get_tab_names(client):
    """
    Returns just the list of tab names in the sheet.
    Agent 1 uses this to understand the sheet structure before
    deciding which tab answers the user's question.
    """
    sheet_url = os.getenv("SHEET_URL")
    spreadsheet = client.open_by_url(sheet_url)

    # We only return names here, not data — keeps it lightweight.
    # No point loading all rows just to know what tabs exist.
    return [ws.title for ws in spreadsheet.worksheets()]


# ── Quick test ───────────────────────────────────────────────────────────────
# This block only runs when you execute this file directly (python tools/sheets_reader.py).
# It will NOT run when other files import this module — that's what "if __name__" guards.
# This is our Layer 1 test: can we connect and read real data?
if __name__ == "__main__":
    print("Connecting to Google Sheets...")

    client = get_google_client()
    print("Connected successfully.\n")

    tabs = get_tab_names(client)
    print(f"Tabs found in sheet: {tabs}\n")

    all_data = get_sheet_metadata(client)

    for tab_name, df in all_data.items():
        print(f"Tab: '{tab_name}'")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print()
