import base64
import json
import os
import pickle
import re
from datetime import datetime, timedelta

import icalendar
import openai
import pytz
import streamlit as st
from dateutil import parser, relativedelta
from google_auth_oauthlib.flow import Flow
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from icalendar import Calendar, Event

openai.api_key = "sk-proj-XXXXXXXXXXXXXXX-_sA"  # replace it with your Open AI API Key
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.send",
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/contacts.readonly",
    "openid"
]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.pickle"
today_str = datetime.now().strftime("%Y-%m-%d")
user_default_timezone = "America/Denver"

if "credentials" not in st.session_state:
    st.session_state["credentials"] = None


def save_credentials(creds):
    with open(TOKEN_FILE, "wb") as token:
        pickle.dump(creds, token)
    st.session_state["credentials"] = creds


def load_credentials():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)
            if creds and creds.valid:
                st.session_state["credentials"] = creds
                return creds
    return None


def authenticate_google():
    """Authenticate the user with Google and return an authorized Gmail service."""
    creds = load_credentials()
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=8502)
        save_credentials(creds)
    try:
        gmail_service = build("gmail", "v1", credentials=creds)
        calendar_service = build("calendar", "v3", credentials=creds)
        return gmail_service, calendar_service
    except HttpError as e:
        print("Error creating Gmail service:", e)
        st.error("⚠️ Error creating Gmail service. Try logging in again.")
        return None


def get_google_services():
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "rb") as token:
                creds = pickle.load(token)
                st.session_state["credentials"] = creds
        except Exception as e:
            print(f"Error loading credentials from pickle file: {e}")
            st.error(f"⚠️ Error loading credentials from pickle file: {e}")
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
            creds = None
            st.session_state["credentials"] = None

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)
        st.session_state["credentials"] = creds

    user_display_name  = get_user_display_name(creds)
    return build("calendar", "v3", credentials=creds), build(
        "gmail", "v1", credentials=creds
    ), user_display_name


def get_user_display_name(creds):
    """Gets the user's display name using the People API."""
    try:
        service = build('people', 'v1', credentials=creds)
        profile = service.people().get(resourceName='people/me', personFields='names').execute()

        names = profile.get('names', [])
        if names:
            display_name = names[0].get('displayName')
            # print(f"User Display Name: {display_name}")
            return display_name
        else:
            return None

    except Exception as e:
        print(f"Error getting user display name: {e}")
        st.error(f"⚠️ Error getting user display name: {e}")
        return None


def standardize_datetime(user_prompt, default_timezone="America/Denver"):
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
            if term in user_prompt.lower():
                user_prompt = user_prompt.lower().replace(
                    term, date_value.strftime("%Y-%m-%d")
                )

        # Ensure full HH:MM extraction (prevent rounding to nearest hour)
        if "T" not in user_prompt:
            # Validate that the time is in 12-hour format if not ISO format
            time_pattern = r"\b(\d{1,2}:\d{2} (AM|PM))\b"
            if not re.search(time_pattern, user_prompt, re.IGNORECASE):
                return {"error": "Invalid time format detected", "details": user_prompt}

        # Detect and convert GMT offsets like "GMT-07:00" to valid timezone
        tz_pattern = r"(UTC[+-]?\d{1,2}|GMT[+-]?\d{1,2}|\+\d{2}:\d{2})"
        tz_match = re.search(tz_pattern, user_prompt)

        if tz_match:
            gmt_offset = tz_match.group(0)
            mapped_timezone = convert_gmt_offset_to_timezone(gmt_offset)

            # Remove GMT offset from user input before parsing
            user_input = user_prompt.replace(gmt_offset, "").strip()

            # Parse datetime and assign correct timezone
            dt = parser.parse(user_input)
            dt = (
                pytz.timezone(mapped_timezone).localize(dt)
                if dt.tzinfo is None
                else dt.astimezone(pytz.timezone(mapped_timezone))
            )
        else:
            # Parse without timezone and assign default timezone
            dt = parser.parse(user_prompt)
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
        if event_data.get("start_time", None):
            event_data["start_time"] = standardize_datetime(event_data["start_time"], event_timezone)
        if event_data.get("end_time", None):
            event_data["end_time"] = standardize_datetime(event_data.get("end_time", None), event_timezone)
        return event_data

    except Exception as e:
        return {"error": "Failed to process event data", "details": str(e)}


def get_structured_json_response(user_prompt):
    system_prompt = f"""
    You are an AI assistant that helps users manage Google Calendar. You interpret user messages and generate JSON output.
    Actions: "schedule_meeting", "cancel_meeting", "block_time".
    Now date and time is {today_str}.
    
    - "schedule_meeting": Requires "start_time", "email_addresses". End time is 30 mins after start. 
    If no title is provided come up with one. If user has described about the meeting use it to generate title, 
    description and timezone as MST as default.
    
    - "cancel_meeting": User provides "start_time" and "search_criteria" to search for a meeting.
    
    - "block_time": User provides "start_time" and "end_time" (without external emails). If no title is provided come 
    up with one. If user has described about the meeting use it to generate title, description and timezone as MST 
    as default.
    
    - "email_response": check emails and reply to emails or user provides check and reply to emails prompt. 
    Generate JSON output with timezone and today's date and time as "start_time".

    Respond ONLY in JSON format.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}. Raw content: {content}")
            return {
                "error": "Invalid JSON response from OpenAI",
                "details": str(e),
                "raw_content": content,
            }
    except Exception as e:
        print(f"Error generating response from OpenAI: {e}")
        return {"error": "Error generating response from OpenAI", "details": str(e)}


def schedule_meeting(cal_service, event_data):
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
    created_event = (
        cal_service.events().insert(calendarId="primary", body=event).execute()
    )
    return f"Meeting scheduled: {created_event.get('htmlLink')}"


def convert_gmt_offset_to_timezone(gmt_offset):
    """
    Converts a GMT offset (e.g., 'GMT-07:00') into a valid IANA timezone.
    Defaults to 'America/Denver' if mapping is unavailable.
    """
    gmt_to_timezone_map = {
        "GMT-07:00": "America/Denver",
        "GMT-08:00": "America/Los_Angeles",
        "GMT-06:00": "America/Chicago",
        "GMT-05:00": "America/New_York",
        "GMT+01:00": "Europe/Berlin",
    }

    return gmt_to_timezone_map.get(
        gmt_offset, "America/Denver"
    )  # Default to 'America/Denver'


def email_exists(email, email_list):
    """Checks if an email address exists in a list of email dictionaries.

    Args:
        email: The email address to search for.
        email_list: A list of dictionaries, where each dictionary has an 'email' key.

    Returns:
        True if the email exists, False otherwise.
    """
    for item in email_list:
        if item["email"] == email:
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


def cancel_meeting(cal_service, event_data):
    """
    Finds and cancels a meeting based on start_time and search_criteria (title/description match).

    Args:
    - event_data (dict): Contains "start_time" and "search_criteria".
    - user_default_timezone (str): User's timezone.

    Returns:
    - str: Success or failure message.
    """
    try:
        print("attempting to cancel meeting")
        # Convert user-provided start_time to datetime (timezone-aware)
        if "start_time" in event_data:
            print(f'start_time: {event_data["start_time"]}')
            user_meeting_time_iso = standardize_datetime(
                event_data["start_time"], user_default_timezone
            )
            print(f"user_time_iso: {user_meeting_time_iso}")
            # Validate if standardize_datetime returned a valid string
            if isinstance(user_meeting_time_iso, dict) or not isinstance(
                user_meeting_time_iso, str
            ):
                return f"❌ Error: Unable to parse start time. Details: {user_meeting_time_iso.get('details', 'Unknown error')}"

            # user_time = datetime.fromisoformat(user_time_iso)  # Convert to datetime
            try:
                # Attempt to parse as ISO format first
                user_meeting_time = datetime.fromisoformat(
                    user_meeting_time_iso.replace("Z", "+00:00")
                )  # added replace to handle Z timezone.
                print(f"user_time: {user_meeting_time}")

            except ValueError:
                try:
                    # If ISO format fails, try other formats (using dateutil)
                    from dateutil import parser

                    user_meeting_time = parser.parse(user_meeting_time_iso)
                    print(f"user_time2: {user_meeting_time}")
                except (ValueError, TypeError):  # Handle parsing errors
                    return None
        else:
            return "⚠️ No start_time provided for cancellation."
            
        # Fetch upcoming events from Google Calendar
        now = datetime.utcnow().replace(tzinfo=pytz.utc).isoformat()
        events_result = (
            cal_service.events()
            .list(calendarId="primary", timeMin=now, maxResults=10)
            .execute()
        )
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
                    event_start = datetime.fromisoformat(event_start_str).astimezone(
                        event_tz
                    )
                    print(f"event_start: {event_start}")
                    # Convert user_time to the same timezone as the event
                    user_meeting_time_in_event_tz = user_meeting_time.astimezone(
                        event_tz
                    )
                    print(f"user_time_in_event_tz: {user_meeting_time_in_event_tz}")
                    event_start = event_start.replace(microsecond=0)
                    print(f"event_start: {event_start}")
                    user_meeting_time_in_event_tz = (
                        user_meeting_time_in_event_tz.replace(microsecond=0)
                    )
                    print(f"user_time_in_event_tz: {user_meeting_time_in_event_tz}")

                    same_day = (
                        user_meeting_time_in_event_tz.date() == event_start.date()
                    )
                    # time_in_range = time_range_start <= event_start <= time_range_end
                    print(f"same_day: {same_day}")
                    print(f"user_time_in_event_tz: {user_meeting_time_in_event_tz}")
                    print(
                        f"user_time_in_event_tz: {type(user_meeting_time_in_event_tz)}"
                    )
                    print(f"event_start: {event_start}")
                    print(f"event_start: {type(event_start)}")
                    print(
                        f"user_time_in_event_tz +15 minutes: {(user_meeting_time_in_event_tz + timedelta(minutes=15))}"
                    )
                    # Check if event starts within 15 minutes of the given time
                    is_within_time_range = (
                        user_meeting_time_in_event_tz
                        <= event_start
                        <= (user_meeting_time_in_event_tz + timedelta(minutes=15))
                    )

                    # Match search criteria in event title or description
                    matches_criteria = (
                        search_criteria in event_summary
                        or search_criteria in event_description
                    )

                    does_email_exists = email_exists(
                        extract_emails(search_criteria), event_attendees
                    )
                    print(f"is_within_time_range: {is_within_time_range}")
                    print(f"matches_criteria: {matches_criteria}")
                    print(f"does_email_exists: {does_email_exists}")
                    print(
                        f"{search_criteria} \n in {event_summary} \n or {event_attendees} \n in {event_description}"
                    )
                    # If both conditions match, cancel the event
                    if same_day and is_within_time_range:
                        event_to_cancel = event
                        break  # Stop at the first matching event
                except Exception as time_error:
                    print(f"⚠️ Skipping event due to time parsing issue: {time_error}")

        if event_to_cancel:
            calendar_service.events().delete(
                calendarId="primary", eventId=event_to_cancel["id"]
            ).execute()
            return f"✅ Meeting '{event_to_cancel['summary']}' at {event_start.strftime('%I:%M %p')} \
                    canceled successfully!"
        else:
            return "⚠️ No matching meeting found within the next 15 minutes."

    except Exception as e:
        return f"❌ Error canceling meeting: {str(e)}"


def send_email_to_attendees(sub, body, to_addresses):
    service = build("gmail", "v1", credentials=st.session_state["credentials"])

    message_text = f"Subject: {sub}\nTo: {', '.join(to_addresses)}\n\n{body}"
    message = base64.urlsafe_b64encode(message_text.encode("utf-8")).decode("utf-8")

    send_request = {"raw": message}

    result = service.users().messages().send(userId="me", body=send_request).execute()
    return result


def generate_email_response(event_data, user_tone="formal"):
    """
    Generates a draft email message based on event_data and desired tone.

    event_data: dict containing title, start_time, end_time, attendees, etc.
    user_tone: "formal", "casual", or any custom tone the user wants
    """
    system_prompt = f"""
    You are an AI email generator. Write a short email in a {user_tone} tone
    inviting or informing attendees about the following meeting details:
    - Greeting: Greet User with name
    - Date/Time: {event_data.get('start_time', 'Unknown')}
    - Timezone: {event_data.get('timezone', 'Unknown')}
    - Description: Use {event_data.get('title', 'No Title')} and {event_data.get('description', '')} to describe the meeting.
    - Signature: User signature

    Include a polite greeting and closing, but keep it concise.
    """
    try:
        res = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            temperature=0.7,
        )
        draft_email = res.choices[0].message.content.strip()
        return draft_email
    except Exception as e:
        print(f"Error generating email: {e}")
        return "Could not generate email at this time."


def get_non_promo_non_spam_messages(g_email_service, max_results=5):
    """
    Returns a list of messages that are NOT promotions/spam in the inbox
    and are UNREAD.
    """
    # Add "is:unread" to fetch only unread emails.
    query = "in:inbox is:unread -category:promotions -label:spam"
    # calendar_service, gmail_service = get_google_services()
    try:
        gmail_response = (
            g_email_service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )

        all_messages = gmail_response.get("messages", [])
        return all_messages
    except HttpError as e:
        print(f"An error occurred: {e}")
        return []


def fetch_ics_data(service, message_id, part):
    body = part.get("body", {})
    data = body.get("data")
    if data:
        # Inline ICS
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    attachment_id = body.get("attachmentId")
    if attachment_id:
        attachment = (
            service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=message_id, id=attachment_id)
            .execute()
        )
        raw_data = attachment.get("data", "")
        return base64.urlsafe_b64decode(raw_data).decode("utf-8", errors="ignore")
    return ""


def parse_ics_content(ics_text):
    """
    Parse ICS text to extract event details using icalendar library.
    Return a dict with summary, start, end, location if possible.
    """
    try:
        cal = Calendar.from_ical(ics_text)
        for component in cal.walk():
            # print("ICS component: ", component)
            if component.name == "VEVENT":
                uid = str(component.get("UID", ""))
                summary = component.get("summary", "No Title")
                location = component.get("location", "No Location")
                dt_start = component.get("dtstart")
                dt_end = component.get("dtend")

                # Convert dtstart/dtend to python datetime (they might be icalendar.prop.vDDDTypes)
                start_dt = dt_start.dt if dt_start else None
                end_dt = dt_end.dt if dt_end else None

                return {
                    "uid": uid,
                    "summary": str(summary),
                    "location": str(location),
                    "start": str(start_dt),
                    "end": str(end_dt),
                }
    except Exception as e:
        print("Error parsing ICS content:", e)
    return None


def extract_plain_text(service, message_id, payload):
    if "parts" in payload:
        # This is a multipart payload
        for part in payload["parts"]:
            mime_type = part.get("mimeType", "")

            if mime_type == "text/calendar":
                ics_text = fetch_ics_data(service, message_id, part)
                return parse_ics_content(ics_text) or ""

            elif mime_type == "text/plain":
                data = part["body"].get("data", "")
                text = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                return text

            elif mime_type.startswith("multipart/"):
                # Correct: recurse into `part`, not `payload`.
                sub_text = extract_plain_text(service, message_id, part)
                if sub_text:
                    return sub_text

    else:
        # Single-part case
        body = payload.get("body", {})
        data = body.get("data")
        if data and payload.get("mimeType") == "text/plain":
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

    return ""


def get_meeting_details(service, message_id, parts):
    """
    If part is 'text/calendar' but stored as an attachment,
    fetch attachment data using attachments().get().
    """
    for part in parts:
        mime_type = part.get("mimeType", "")
        if mime_type == "text/calendar":
            body = part.get("body", {})
            # If there's a direct 'data' field
            data = body.get("data")
            if data:
                # Already inlined data
                ics_text = base64.urlsafe_b64decode(data).decode(
                    "utf-8", errors="ignore"
                )
                return parse_ics_content(ics_text)
            else:
                # Possibly an attachment
                attachment_id = body.get("attachmentId")
                if attachment_id:
                    attachment = (
                        service.users()
                        .messages()
                        .attachments()
                        .get(userId="me", messageId=message_id, id=attachment_id)
                        .execute()
                    )
                    attachment_data = attachment.get("data", "")
                    if attachment_data:
                        ics_bytes = base64.urlsafe_b64decode(attachment_data)
                        ics_text = ics_bytes.decode("utf-8", errors="ignore")
                        return parse_ics_content(ics_text)

        elif mime_type.startswith("multipart/"):
            # recursively inspect sub-parts
            subparts = part.get("parts", [])
            details = get_meeting_details(service, message_id, subparts)
            if details:
                return details
    return None


def get_header_value(headers, name):
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def extract_sender_name(email_string):
    """
    Extracts the sender name from an email string.
    If no name is found, it returns None.
    """
    match = re.match(r"([\w\s]+)\s*<*[\w\.-]+@[\w\.-]+>*", email_string)
    if match:
        return match.group(1).strip()
    return "There"


def get_meeting_details_single_part(service, message_id, payload):
    """
    If the email has no 'parts', it might still have 'body' that is text/calendar.
    Or it might be plain text. Typically, you'll check 'mimeType' or see if there's
    an 'attachmentId' for ICS data.
    """
    mime_type = payload.get("mimeType")
    if mime_type == "text/calendar":
        data = payload.get("data")
        attachment_id = payload.get("attachmentId")

        if data:
            # Inline ICS data (base64 encoded)
            try:
                decoded_data = base64.urlsafe_b64decode(data).decode("utf-8")
                cal = icalendar.Calendar.from_ical(decoded_data)
                return cal  # Return the icalendar.Calendar object
            except Exception as e:
                print(f"Error parsing inline ICS data: {e}")
                return None

        elif attachment_id:
            # Fetch ICS attachment
            try:
                attachment = (
                    service.users()
                    .messages()
                    .attachments()
                    .get(userId="me", messageId=message_id, id=attachment_id)
                    .execute()
                )
                attachment_data = attachment.get("data")
                decoded_attachment_data = base64.urlsafe_b64decode(
                    attachment_data
                ).decode("utf-8")
                cal = icalendar.Calendar.from_ical(decoded_attachment_data)
                return cal  # Return the icalendar.Calendar object
            except Exception as e:
                print(f"Error fetching or parsing ICS attachment: {e}")
                return None

    return None


def get_full_message(service, message_id):
    message = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )

    payload = message.get("payload", {})
    # print(f"""payload: {payload}""")
    # Extract basic headers (From, Subject, etc.)
    headers = payload.get("headers", [])
    from_email_address = get_header_value(headers, "From")
    sub = get_header_value(headers, "Subject")

    # If 'parts' is not present, treat as single-part
    if "parts" not in payload:
        # Single-part email (check payload["body"])
        body = extract_plain_text(service, message_id, payload)
        # If it might have a text/calendar or attachment,
        # you'll need to check separately or do something like:
        meeting_detail = get_meeting_details_single_part(service, message_id, payload)
    else:
        # It's a multipart email
        parts = payload["parts"]
        # Possibly handle each part or pick the first, etc.
        body = extract_plain_text(
            service, message_id, payload
        )
        # Example:
        meeting_detail = get_meeting_details(service, message_id, parts)

    return from_email_address, sub, body, meeting_detail


def generate_meeting_invite_reply(meeting_detail):
    """
    If an email contains an ICS invitation, generate a short message
    with the meeting details, asking if the user wants to accept, decline, or propose new time.
    We'll let the user finalize how to respond in the UI.
    """
    summary = meeting_detail.get("summary", "No Title")
    start = meeting_detail.get("start", "Unknown Start")
    end = meeting_detail.get("end", "Unknown End")
    location = meeting_detail.get("location", "Unknown Location")

    # A simple text to show in Streamlit
    meeting_invite_text = (
        f"**Meeting Invitation:**\n\n"
        f"- **Title**: {summary}\n"
        f"- **Start**: {start}\n"
        f"- **End**: {end}\n"
        f"- **Location**: {location}\n\n"
        "Would you like to Accept, Decline, or Propose a new time?\n"
    )

    return meeting_invite_text


def block_time(cal_service, event_data):
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
        "attendees": [
            {"email": event_data.get("email_addresses", ["b.niranjanrao@gmail.com"])[0]}
        ],
    }
    created_event = (
        cal_service.events().insert(calendarId="primary", body=event).execute()
    )
    return f"Time blocked: {created_event.get('htmlLink')}"


def generate_response_with_openai(user_name, formatted_sender_name, email_body):
    """
    Pass the email body to OpenAI and ask for a short, helpful reply.
    """
    today_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_prompt = (
        "You are a helpful AI assistant. The user has received the following email.\n"
        f"Email text:\n{email_body}\n"
        "Role-Playing: You will act as the recipient of this email and generate a reply.\n"
        "Sender Identification: Identify the sender's name from the `email_body`. "
        "Do not use placeholders like `[Sender's Name]`. Use the actual name.\n"
        "Greeting: Greet the sender by their name. Use an appropriate greeting based on the time of day "
        "(e.g., Good morning, Good afternoon, Good evening).\n"
        "Reply Content: Write a concise and polite reply. Infer necessary context from the `email_body` if needed. "
        "Keep the reply friendly and to the point.\n"
        "User Signature: Add a user signature at the end of the reply. Use the user's name "
        "(which will be provided by the user in the prompt). If a user title is provided, use it. "
        "Otherwise, omit the title or leave it blank.\n"
        "Human-Like Tone: Ensure the reply sounds as natural and human-like as possible.\n"
        "Avoid Placeholders: Do not use placeholders like `[Sender's Name]`, `[Your Name]`, `[Your Title]`, "
        f"or any other placeholders. Instead use the actual name. sender name: {formatted_sender_name} "
        f"and Your Name or Your Title or for email singnature use: {user_name},\n"
        f"Time awareness: You are aware of the current time {today_date_time}.\n"
    )

    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        reply_text = completion.choices[0].message.content.strip()
        return reply_text
    except Exception as e:
        print("OpenAI API error:", e)
        return "Error generating response with OpenAI."


def send_reply(service, original_message_id, to_email, sub, body):
    """
    Sends a reply to a specific email thread with `original_message_id`.
    """
    reply_text = (
        f"To: {to_email}\r\n"
        f"Subject: Re: {sub}\r\n"
        f"In-Reply-To: {original_message_id}\r\n"
        f"References: {original_message_id}\r\n"
        "\r\n"
        f"{body}"
    )

    encoded_message = base64.urlsafe_b64encode(reply_text.encode("utf-8")).decode(
        "utf-8"
    )

    message_body = {"raw": encoded_message, "threadId": original_message_id}

    try:
        sent_message = (
            service.users().messages().send(userId="me", body=message_body).execute()
        )
        st.write("Reply sent. Message ID:", sent_message["id"])
    except HttpError as e:
        print("Error sending reply:", e)


def accept_event_with_calendar_api(cal_service, ical_uid, user_email):
    # 1) Find the event by iCalUID in the user’s calendar
    # If you imported or already have it, you can do something like:
    user_info = cal_service.settings().get(setting="timezone").execute()
    print("Using credentials for:", user_info)

    calendars = cal_service.calendarList().list().execute()
    for c in calendars["items"]:
        print(f"\nCalendar ID: {c['id']}\n, Summary: {c['summary']}\n")

    events_result = (
        cal_service.events().list(calendarId="primary", iCalUID=ical_uid).execute()
    )
    items = events_result.get("items", [])
    if not items:
        print(
            f"⚠️ Event with iCalUID {ical_uid} not found in calendar. Importing now..."
        )

        # 2️⃣ Import the event from ICS data (assuming it's in meeting_details)
        imported_event = (
            calendar_service.events()
            .import_(
                calendarId="primary",
                body={
                    "summary": "Imported Event",
                    "description": "Automatically added from email invite",
                    "iCalUID": ical_uid,
                    "status": "confirmed",
                },
            )
            .execute()
        )

        print(f"✅ Event imported: {imported_event['summary']}")

        event = imported_event
    else:
        event = items[0]

    attendees = event.get("attendees", [])
    updated = False
    for attendee in attendees:
        if attendee["email"].lower() == user_email.lower():
            attendee["responseStatus"] = "accepted"
            updated = True
            break
    if not updated:
        # If user wasn't in the attendees list, add them
        attendees.append({"email": user_email, "responseStatus": "accepted"})
        print(f"attendees: {attendees}")
    seen = set()
    cleaned_attendees = []
    for a in event.get("attendees", []):
        addr = a["email"].lower()
        if addr not in seen:
            seen.add(addr)
            if addr == user_email.lower():
                a["responseStatus"] = "accepted"
            cleaned_attendees.append(a)

    event["attendees"] = cleaned_attendees
    print('event.get("attendees", [])')
    print(event.get("attendees", []))
    event["attendeesOmitted"] = False
    patched_event = (
        calendar_service.events()
        .patch(
            calendarId="primary",
            eventId=event["id"],
            body={"attendees": attendees, "attendeesOmitted": False},
            sendUpdates="all",
        )
        .execute()
    )

    return f"patched_event event summary: {patched_event['summary']}"


credentials = authenticate_google()

if credentials:
    # Streamlit Chatbot UI
    st.title("📅 AI Chatbot for Google Mail, Calendar services")

    user_input = st.text_input(
        "Ask me to check and reply to emails,schedule, cancel, or block time:"
    )
    calendar_service, g_email_service, user_name = get_google_services()
    if user_input:
        response = get_structured_json_response(user_input)
        print("Initial response:", response)
        st.json(response)
        response = process_event_data(response)
        print("process_event_data:", response)
        action = response.get("action")

        if action == "schedule_meeting":
            # 1. Generate the meeting in Google Calendar (or prepare to do so)
            # 2. Generate the email draft
            email_draft = generate_email_response(
                response, user_tone="formal"
            )  # or from a user-selected tone
            st.session_state["email_draft"] = email_draft

            # Show the user the draft email
            st.write("### Draft Email to Attendees:")
            edited_email = st.text_area(
                "Feel free to edit before sending:", email_draft, height=200
            )
            output = "Email draft generated. Ready to send?"
            # Two buttons: Send Email or Cancel
            if st.button("Send Email"):
                response["description"] = edited_email
                st.success({schedule_meeting(calendar_service, response)})

            if st.button("Cancel"):
                st.warning("Email sending canceled.")
                st.success("Email sending canceled.")
        elif action == "cancel_meeting":
            st.success({cancel_meeting(calendar_service, response)})
        elif action == "block_time":
            st.success({block_time(calendar_service, response)})
        elif action == "email_response":
            messages = get_non_promo_non_spam_messages(g_email_service)
            if not messages:
                st.write("No new non-promotional, non-spam, unread emails found.")

            for msg in messages:
                msg_id = msg["id"]
                from_email, subject, body_text, meeting_details = get_full_message(
                    g_email_service, msg_id
                )
                st.write("---")
                st.write(f"**From:** {from_email}")
                st.write(f"**Subject:** {subject}")
                st.write(f"**Body:**\n{body_text}")

                formatted_sender_name = extract_sender_name(from_email)

                if meeting_details:
                    # This is a meeting invitation
                    invite_text = generate_meeting_invite_reply(meeting_details)
                    st.markdown(invite_text, unsafe_allow_html=True)

                    # Provide buttons to accept, decline or propose new time
                    col1, col2, col3 = st.columns(3)

                    if col1.button("Accept", key=f"accept_{msg_id}"):
                        reply_body = (
                            f"Hello,\n\n"
                            f"Thank you {formatted_sender_name} for the invitation. I will be happy to attend "
                            f"'{meeting_details['summary']}' on {meeting_details['start']}.\n"
                            "Looking forward to it.\n\n"
                            "Best regards,\n"
                            f"{user_name}"
                        )
                        send_reply(
                            g_email_service, msg_id, from_email, subject, reply_body
                        )
                        # print(f'msg uid: {meeting_details["uid"]}')
                        meeting_accepted = accept_event_with_calendar_api(
                            calendar_service, meeting_details["uid"], from_email
                        )
                        # print(f"accept_event_with_calendar_api: {meeting_accepted}")
                        st.success("Accepted and reply sent.")

                    if col2.button("Decline", key=f"decline_{msg_id}"):
                        reply_body = (
                            "Hello,\n\n"
                            f"Thank you for the invitation. Unfortunately, I won't be able to join "
                            f"'{meeting_details['summary']}' on {meeting_details['start']}. \n"
                            "Regards,"
                            f"\n{user_name}"
                        )
                        send_reply(
                            g_email_service, msg_id, from_email, subject, reply_body
                        )
                        st.success("Declined and reply sent.")

                    if col3.button("Propose New Time", key=f"propose_{msg_id}"):
                        # A text input for new time
                        new_time = st.text_input(
                            f"Propose new time for meeting {meeting_details['summary']}:",
                            key=f"newtime_{msg_id}",
                        )
                        if new_time:
                            # e.g., "How about tomorrow at 3 PM?"
                            reply_body = (
                                f"Hello, {formatted_sender_name}\n\n"
                                f"Thanks for the invitation to '{meeting_details['summary']}'. "
                                f"Unfortunately, the proposed time doesn't work for me. \n"
                                f"Could we meet instead on: {new_time}?\n\n"
                                "Let me know.\n\n"
                                "Best regards,"
                                f"\n{user_name}"
                            )
                            send_reply(
                                g_email_service, msg_id, from_email, subject, reply_body
                            )
                            st.success("Proposed a new time and reply sent.")

                else:
                    # Standard non-meeting email => use OpenAI to generate a reply
                    ai_reply = generate_response_with_openai(user_name, formatted_sender_name, body_text)
                    st.write("**Proposed AI Reply:**")
                    user_edited_reply = st.text_area(
                        "Edit the reply if needed:", value=ai_reply, key=msg_id
                    )

                    # Provide a button to send the reply
                    if st.button(f"Send Reply to {from_email}", key="send_" + msg_id):
                        send_reply(
                            g_email_service,
                            msg_id,
                            from_email,
                            subject,
                            user_edited_reply,
                        )
                        st.success(f"Reply sent to {from_email}")

        else:
            st.success("Unknown action. Please try again.")
