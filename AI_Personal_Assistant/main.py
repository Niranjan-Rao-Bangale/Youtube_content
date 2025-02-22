import json
import os
import pickle
import re
from datetime import datetime, timedelta

import openai
import pytz
import streamlit as st
from dateutil import parser, relativedelta
from google_auth_oauthlib.flow import Flow
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


openai.api_key = "sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXX" # Replace with your open api key
SCOPES = ["https://www.googleapis.com/auth/calendar"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.pickle"
today_str = datetime.now().strftime("%Y-%m-%d")
user_default_timezone = 'America/Denver'

if "credentials" not in st.session_state:
    st.session_state["credentials"] = None


def save_credentials(credentials):
    with open(TOKEN_FILE, "wb") as token:
        pickle.dump(credentials, token)
    st.session_state["credentials"] = credentials


def load_credentials():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            credentials = pickle.load(token)
            if credentials and credentials.valid:
                st.session_state["credentials"] = credentials
                return credentials
    return None


def authenticate_google():
    flow = Flow.from_client_secrets_file(
        CREDENTIALS_FILE,
        scopes=["https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/tasks"],
        redirect_uri="http://localhost:8501/"
    )
    credentials = load_credentials()
    if credentials:
        return credentials  # Already authenticated

    # Generate authorization URL
    authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true')

    # Display login link
    if st.session_state["credentials"] is None:
        st.write("### 🔐 Please authenticate with Google first, to chat with this application.")
        st.markdown(f'<a href="{authorization_url}" target="_blank">🔗 Click here to Login with Google</a>',
                    unsafe_allow_html=True)

    # Check if we got an authorization code from the URL
    query_params = st.query_params
    auth_code_param = query_params.get("code", None)

    if auth_code_param:
        if isinstance(auth_code_param, list):
            auth_code = auth_code_param[0]
        else:
            auth_code = auth_code_param
        st.write(f"📌 Debug: Extracted Auth Code = {auth_code}")
        if len(auth_code) > 5:  # Ensure the auth code is valid
            try:
                # Exchange authorization code for access token
                flow.fetch_token(code=auth_code)
                credentials = flow.credentials
                save_credentials(credentials)
                print(f"save_credentials are saved: {save_credentials}")
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ OAuth Error: {e}")
                return None
        else:
            st.error("⚠️ Invalid Auth Code Received. Try logging in again.")

    return None


def get_google_calendar_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "rb") as token:
                creds = pickle.load(token)
                st.session_state["credentials"] = creds
        except Exception as e:
            print(f"Error loading credentials from pickle file: {e}")
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
            creds = None
            st.session_state["credentials"] = None

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)
        st.session_state["credentials"] = creds
    return build("calendar", "v3", credentials=creds)


def standardize_datetime(user_input, default_timezone="America/Denver"):
    """
    Dynamically converts user-provided date-time expressions into ISO 8601 format.

    Args:
    - user_input (str): Natural language date-time string.
    - default_timezone (str): Default timezone if not provided.

    Returns:
    - str: ISO 8601 formatted date-time.
    """
    try:
        now = datetime.now(pytz.timezone(default_timezone))

        # Handle common relative terms dynamically
        relative_terms = {
            "today": now,
            "tomorrow": now + timedelta(days=1),
            "next monday": now + relativedelta.relativedelta(weekday=0, weeks=1),
            "next tuesday": now + relativedelta.relativedelta(weekday=1, weeks=1),
            "next wednesday": now + relativedelta.relativedelta(weekday=2, weeks=1),
            "next thursday": now + relativedelta.relativedelta(weekday=3, weeks=1),
            "next friday": now + relativedelta.relativedelta(weekday=4, weeks=1),
            "next saturday": now + relativedelta.relativedelta(weekday=5, weeks=1),
            "next sunday": now + relativedelta.relativedelta(weekday=6, weeks=1),
        }

        # Replace any relative terms with actual date
        for term, date_value in relative_terms.items():
            if term in user_input.lower():
                user_input = user_input.lower().replace(term, date_value.strftime("%Y-%m-%d"))

        # Ensure full HH:MM extraction (prevent rounding to nearest hour)
        if "T" not in user_input:
            # Validate that the time is in 12-hour format if not ISO format
            time_pattern = r"\b(\d{1,2}:\d{2} (AM|PM))\b"
            if not re.search(time_pattern, user_input, re.IGNORECASE):
                return {"error": "Invalid time format detected", "details": user_input}

        # Detect and convert GMT offsets like "GMT-07:00" to valid timezone
        tz_pattern = r"(UTC[+-]?\d{1,2}|GMT[+-]?\d{1,2}|\+\d{2}:\d{2})"
        tz_match = re.search(tz_pattern, user_input)

        if tz_match:
            gmt_offset = tz_match.group(0)
            mapped_timezone = convert_gmt_offset_to_timezone(gmt_offset)

            # Remove GMT offset from user input before parsing
            user_input = user_input.replace(gmt_offset, "").strip()

            # Parse datetime and assign correct timezone
            dt = parser.parse(user_input)
            dt = pytz.timezone(mapped_timezone).localize(dt) if dt.tzinfo is None else dt.astimezone(pytz.timezone(mapped_timezone))
        else:
            # Parse without timezone and assign default timezone
            dt = parser.parse(user_input)
            tz = pytz.timezone(default_timezone)
            dt = tz.localize(dt) if dt.tzinfo is None else dt.astimezone(tz)

        # Convert to ISO 8601 format
        return dt.isoformat()

    except Exception as e:
        return {"error": "Unable to parse date/time", "details": str(e)}


def process_event_data(event_data, default_timezone=user_default_timezone):
    """
    Processes event data by standardizing start_time, end_time, and timezone.

    Args:
    - event_data (dict): Contains 'start_time', 'end_time', and optional 'timezone'.
    - default_timezone (str): Default timezone if not provided.

    Returns:
    - dict: Updated event_data with standardized ISO 8601 time formats.
    """
    try:
        # Extract timezone if provided, else use default
        event_timezone = event_data.get("timezone", default_timezone)

        # Standardize start and end time
        event_data["start_time"] = standardize_datetime(event_data["start_time"], event_timezone)
        event_data["end_time"] = standardize_datetime(event_data.get("end_time", None), event_timezone)

        # Ensure the final timezone is consistent
        event_data["timezone"] = event_timezone

        return event_data  # Return updated event data

    except Exception as e:
        return {"error": "Failed to process event data", "details": str(e)}


def get_structured_json_response(user_input):
    system_prompt = f"""
    You are an AI assistant that helps users manage Google Calendar. You interpret user messages and generate JSON output.
    Actions: "schedule_meeting", "cancel_meeting", "block_time".
    Now date and time is {today_str}.
    - "schedule_meeting": Requires "start_time", "email_addresses". End time is 30 mins after start. If no title is provided come up with one. If user has described about the meeting use it to generate title, description and timezone as MST as default.
    - "cancel_meeting": User provides "start_time" and "search_criteria" to search for a meeting.
    - "block_time": User provides "start_time" and "end_time" (without external emails). If no title is provided come up with one. If user has described about the meeting use it to generate title, description and timezone as MST as default.

    Respond ONLY in JSON format.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}. Raw content: {content}")
            return {"error": "Invalid JSON response from OpenAI", "details": str(e), "raw_content": content}
    except Exception as e:
        print(f"Error generating response from OpenAI: {e}")
        return {"error": "Error generating response from OpenAI", "details": str(e)}


def schedule_meeting(event_data):
    service = get_google_calendar_service()
    event = {
        "summary": event_data["title"],
        "description": event_data["description"],
        "start": {
            "dateTime": event_data["start_time"],
            "timeZone": event_data["timezone"],
        },
        "end": {
            "dateTime": event_data["end_time"],
            "timeZone": event_data["timezone"],
        },
        "attendees": [{"email": email} for email in event_data["email_addresses"]],
    }
    created_event = service.events().insert(calendarId="primary", body=event).execute()
    return f"Meeting scheduled: {created_event.get('htmlLink')}"


def convert_gmt_offset_to_timezone(gmt_offset):
    """
    Converts a GMT offset (e.g., 'GMT-07:00') into a valid IANA timezone.
    Defaults to 'America/Denver' if mapping is unavailable.
    """
    gmt_to_timezone_map = {
        "GMT+05:30": "Asia/Kolkata",
        "GMT-07:00": "America/Denver",
        "GMT-08:00": "America/Los_Angeles",
        "GMT-06:00": "America/Chicago",
        "GMT-05:00": "America/New_York",
        "GMT+01:00": "Europe/Berlin"
    }

    return gmt_to_timezone_map.get(gmt_offset, "America/Denver")  # Default to 'America/Denver'


def email_exists(email, email_list):
  """Checks if an email address exists in a list of email dictionaries.

  Args:
      email: The email address to search for.
      email_list: A list of dictionaries, where each dictionary has an 'email' key.

  Returns:
      True if the email exists, False otherwise.
  """
  for item in email_list:
    if item['email'] == email:
      return True
  return False


def extract_emails(text):
    """Extracts email addresses from a string using regular expressions.

    Args:
        text: The input string.

    Returns:
        A list of email addresses found in the string.  Returns an empty list
        if no emails are found.
    """
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"  # Improved regex
    emails = re.findall(email_pattern, text)
    return emails


def cancel_meeting(event_data):
    """
    Finds and cancels a meeting based on start_time and search_criteria (title/description match).

    Args:
    - event_data (dict): Contains "start_time" and "search_criteria".
    - user_default_timezone (str): User's timezone.

    Returns:
    - str: Success or failure message.
    """
    try:
        service = get_google_calendar_service()
        print("attempting to cancel meeting")
        # Convert user-provided start_time to datetime (timezone-aware)
        if "start_time" in event_data:
            print(f'start_time: {event_data["start_time"]}')
            user_meeting_time_iso = standardize_datetime(event_data["start_time"], user_default_timezone)
            print(f'user_time_iso: {user_meeting_time_iso}')
            # Validate if standardize_datetime returned a valid string
            if isinstance(user_meeting_time_iso, dict) or not isinstance(user_meeting_time_iso, str):
                return f"❌ Error: Unable to parse start time. Details: {user_meeting_time_iso.get('details', 'Unknown error')}"

            # user_time = datetime.fromisoformat(user_time_iso)  # Convert to datetime
            try:
                # Attempt to parse as ISO format first
                user_meeting_time = datetime.fromisoformat(user_meeting_time_iso.replace('Z', '+00:00'))  # added replace to handle Z timezone.
                print(f'user_time: {user_meeting_time}')

            except ValueError:
                try:
                    # If ISO format fails, try other formats (using dateutil)
                    from dateutil import parser
                    user_meeting_time = parser.parse(user_meeting_time_iso)
                    print(f'user_time2: {user_meeting_time}')
                except (ValueError, TypeError):  # Handle parsing errors
                    return None
        else:
            return "⚠️ No start_time provided for cancellation."

        # # Define valid cancellation range (from user_time to +15 minutes)
        # time_range_start = user_time
        # time_range_end = user_time + timedelta(minutes=15)

        # Fetch upcoming events from Google Calendar
        now = datetime.utcnow().replace(tzinfo=pytz.utc).isoformat()
        events_result = service.events().list(calendarId="primary", timeMin=now, maxResults=10).execute()
        events = events_result.get("items", [])

        search_criteria = event_data.get("search_criteria", "")
        event_to_cancel = None

        for event in events:
            print(f"event: {event}")
            event_summary = event.get("summary", "")
            event_description = event.get("description", "")
            event_attendees = event.get("attendees", "")

            # Extract event start time and timezone
            event_start_str = event["start"].get("dateTime", None)
            event_timezone = event["start"].get("timeZone", user_default_timezone)

            # Handle cases where Google Calendar returns GMT offsets instead of IANA timezone names
            if re.match(r"GMT[+-]\d{2}:\d{2}", event_timezone):
                event_timezone = convert_gmt_offset_to_timezone(event_timezone)

            if event_start_str:
                try:
                    # Convert event start time to timezone-aware datetime
                    print(f"event_start_str: {event_start_str}")
                    event_tz = pytz.timezone(event_timezone)
                    print(f"event_tz: {event_tz}")
                    event_start = datetime.fromisoformat(event_start_str).astimezone(event_tz)
                    print(f"event_start: {event_start}")
                    # Convert user_time to the same timezone as the event
                    user_meeting_time_in_event_tz = user_meeting_time.astimezone(event_tz)
                    print(f"user_time_in_event_tz: {user_meeting_time_in_event_tz}")
                    event_start = event_start.replace(microsecond=0)
                    print(f"event_start: {event_start}")
                    user_meeting_time_in_event_tz = user_meeting_time_in_event_tz.replace(microsecond=0)
                    print(f"user_time_in_event_tz: {user_meeting_time_in_event_tz}")

                    same_day = user_meeting_time_in_event_tz.date() == event_start.date()
                    # time_in_range = time_range_start <= event_start <= time_range_end
                    print(f"same_day: {same_day}")
                    print(f"user_time_in_event_tz: {user_meeting_time_in_event_tz}")
                    print(f"user_time_in_event_tz: {type(user_meeting_time_in_event_tz)}")
                    print(f"event_start: {event_start}")
                    print(f"event_start: {type(event_start)}")
                    print(f"user_time_in_event_tz +15 mins: {(user_meeting_time_in_event_tz + timedelta(minutes=15))}")
                    # Check if event starts within 15 minutes of the given time
                    is_within_time_range = user_meeting_time_in_event_tz <= event_start <= (
                            user_meeting_time_in_event_tz + timedelta(minutes=15))

                    # Match search criteria in event title or description
                    matches_criteria = (search_criteria in event_summary or search_criteria in event_description)

                    does_email_exists = email_exists(extract_emails(search_criteria), event_attendees)
                    print(f"is_within_time_range: {is_within_time_range}")
                    print(f"matches_criteria: {matches_criteria}")
                    print(f"does_email_exists: {does_email_exists}")
                    print(f"{search_criteria} \n in {event_summary} \n or {event_attendees} \n in {event_description}")
                    # If both conditions match, cancel the event
                    if same_day and is_within_time_range:
                        event_to_cancel = event
                        break  # Stop at the first matching event
                except Exception as time_error:
                    print(f"⚠️ Skipping event due to time parsing issue: {time_error}")

        if event_to_cancel:
            service.events().delete(calendarId="primary", eventId=event_to_cancel["id"]).execute()
            return f"✅ Meeting '{event_to_cancel['summary']}' at {event_start.strftime('%I:%M %p')} canceled successfully!"
        else:
            return "⚠️ No matching meeting found within the next 15 minutes."

    except Exception as e:
        return f"❌ Error canceling meeting: {str(e)}"


def block_time(event_data):
    service = get_google_calendar_service()
    event = {
        "summary": event_data["title"],
        "description": event_data.get("description", ""),
        "start": {
            "dateTime": event_data["start_time"],
            "timeZone": event_data["timezone"],
        },
        "end": {
            "dateTime": event_data["end_time"],
            "timeZone": event_data["timezone"],
        },
        "attendees": [{"email": event_data.get("email_addresses", ["b.niranjanrao@gmail.com"])[0]}],
    }
    created_event = service.events().insert(calendarId="primary", body=event).execute()
    return f"Time blocked: {created_event.get('htmlLink')}"


credentials = authenticate_google()

if credentials:
    # Streamlit Chatbot UI
    st.title("📅 AI Chatbot for Google Calendar")

    user_input = st.text_input("Ask me to schedule, cancel, or block time:")
    if user_input:
        response = get_structured_json_response(user_input)
        print("Initial response:", response)
        st.json(response)
        response = process_event_data(response)
        print("process_event_data:", response)
        action = response.get("action")

        if action == "schedule_meeting":
            output = schedule_meeting(response)
        elif action == "cancel_meeting":
            output = cancel_meeting(response)
        elif action == "block_time":
            output = block_time(response)
        else:
            output = "Unknown action. Please try again."

        st.success(output)
