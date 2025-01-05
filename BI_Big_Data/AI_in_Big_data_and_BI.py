import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import praw
from dotenv import load_dotenv
from ollama import ChatResponse
from ollama import chat
from pydantic import BaseModel
from sqlalchemy import create_engine, text

db_path = \
  "sqlite:///C:/Users/bnira/AppData/Roaming/DBeaverData/workspace6/.metadata/sample-database-sqlite-1/Chinook.db"  # Replace with your database path
engine = create_engine(db_path)

load_dotenv()
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")

reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent="YOUR_USER_AGENT"
)


def get_subreddit_sentiments(reddit_url):
  submission_id = reddit_url.split('/')[-3]
  submission = reddit.submission(id=submission_id)
  submission.comments.replace_more(limit=0)

  comments = []
  for comment in submission.comments.list():
    comments.append(comment.body)

  sentiment_results = []

  for comment in comments[:20]:  # Limit to the first 20 comments for analysis
    prompt = f"""
        Perform sentiment analysis on the following comment. Categorize the sentiment as Positive, Negative, or Neutral. 
        Comment: "{comment}"
        return sentiment in one of the following formats: "Positive", "Negative", "Neutral". Don't say anything else.
        """
    response = chat(
      model="llama3.2",
      messages=[{'role': 'user', 'content': prompt}],
    )

    sentiment = response.message.content.strip()
    sentiment_results.append({"comment": comment, "sentiment": sentiment})

  output_file = "reddit_sentiments.json"
  with open(output_file, "w") as json_file:
    json.dump(sentiment_results, json_file, indent=4)

  print("Sentiment Analysis Results:")
  print(json.dumps(sentiment_results, indent=4))

  # Summarize sentiments
  summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
  for result in sentiment_results:
    sentiment = result["sentiment"]
    if sentiment in summary:
      summary[sentiment] += 1

  print("\nSentiment Summary:")
  print(summary)


class SQLQuery(BaseModel):
  query: str


def parse_nlp_to_sql(user_prompt):
  """
  Use Ollama Llama 3.2 to convert natural language to SQL query with structured output validation.
  """
  prompt = f"""
    You are an AI assistant that generates SQL queries based on the user's request.
    The queries must match the following database schema (SQLite):
    1. album (AlbumId, Title, ArtistId)
    2. artist (artistid, name)
    3. track (trackid, name, albumid, genreid, composer, milliseconds, bytes, unit_price, MediaTypeId)
    4. Invoice (InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, 
    BillingPostalCode, Total)
    5. InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
    6. MediaType (MediaTypeId, Name)
    7. Genre (GenreId, Name)
    8. Customer (CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone,
    Fax, Email, SupportRepId)
    9. Employee (EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State,
    Country, PostalCode, Phone, Fax, Email)
    10. Playlist (PlaylistId, Name)
    11. PlaylistTrack (PlaylistId, TrackId)

    Convert the user's natural language request into a valid SQL query:
    Request: "{user_prompt}"
    When generating SQL queries:
    1. Always use fully qualified column names (e.g., table_name.column_name) for every column.
    2. Ensure there are no ambiguous column references.
    3. Output the query in the following JSON format: {{"query": "<SQL query>"}}
    4. Use `JOIN` clauses whenever columns from multiple tables are referenced.
    5. Ensure that all table relationships are explicitly stated using `ON` conditions in `JOIN` clauses.
    """
  try:
    # Call Ollama with structured output
    response = chat(model="llama3.2", format="json",
                           messages=[{"role": "user", "content": prompt}])
    # Parse the response to extract the query
    sql_query = response.message.content.strip()
    print("llama response:", sql_query)
    # Validate and return the query
    try:
      SQLQuery.model_validate_json(response.message.content)
    except Exception as e:
      print(f"Error validating SQL query: {e}")
      return "Error validating SQL query"
    sql_query = json.loads(sql_query)
    execute_sql_query(sql_query['query'])
  except Exception as e:
    return f"Error generating SQL query: {e}"


def execute_sql_query(query):
  """
  Execute a SQL query on SQLite and fetch results.
  """
  try:
    with engine.connect() as connection:
      result = connection.execute(text(query))
      columns = result.keys()
      data = result.fetchall()
      print(f"Query Result: {data}")
      print(f"Result columns: {columns}")
      chart_type = input("Enter chart type (bar, line, pie): ")
      generate_chart(columns, data, chart_type)
  except Exception as e:
    return None, f"Error executing query: {e}"


def generate_chart(columns, data, chart_type="bar"):
  """
  Generate a chart from query results.
  """
  if not data:
    print("No data to plot.")
    return
  # Convert columns to a list
  column_list = list(columns)

  # Transpose data for easy plotting
  data_transposed = list(zip(*data))

  if len(data_transposed) == 1:  # Single-column result
    print("Single-column result detected. Showing data as text.")
    print(f"{column_list[0]}: {data_transposed[0][0]}")
    return

  # Chart generation
  plt.figure(figsize=(10, 6))
  if chart_type == "bar":
    plt.bar(data_transposed[0], data_transposed[1])
  elif chart_type == "line":
    plt.plot(data_transposed[0], data_transposed[1])
  elif chart_type == "pie":
    plt.pie(data_transposed[1], labels=data_transposed[0], autopct='%1.1f%%')
  else:
    print("Unsupported chart type")
    return

  # Chart customization
  plt.xlabel(column_list[0])
  plt.ylabel(column_list[1])
  plt.title("Query Result Visualization")
  plt.show()


def etl_process_log_file(log_file_path, output_file_path):
  """
  Process the log file and convert it into a structured format.

  Args:
      log_file_path (str): Path to the log file.
      output_file_path (str): Path to save the processed file (CSV).
  """
  df = parse_log_file(log_file_path)

  # Save the structured data to a CSV file
  df.to_csv(output_file_path, index=False)
  print(f"Processed log file saved to {output_file_path}")
  return f"{log_file_path} file has been processed successfully and loaded into {output_file_path}"


etl_process_log_file_tool = {
  'type': 'function',
  'function': {
        "name": "etl_process_log_file",
        "description": "ETL Processes a log file and converts it into a structured CSV format.",
        "parameters": {
            "type": "object",
            "properties": {
                "log_file_path": {"type": "string", "description": "Path to the input log file."},
                "output_file_path": {"type": "string", "description": "Path to save the processed CSV file."}
            },
            "required": ["log_file_path", "output_file_path"]
        }
    }
}


def parse_log_file(file_path):
  """
  Parse the log file into structured data.
  """
  log_entries = []
  log_pattern = re.compile(r'(\d{2}/\d{2} \d{2}:\d{2}:\d{2}) (\w+)\s+:([\w.]+): (.+)')

  with open(file_path, 'r') as file:
    for line in file:
      line = line.strip()
      if not line:
        continue
      match = log_pattern.match(line)
      if match:
        timestamp, log_level, source, message = match.groups()
        log_entries.append({
          "Timestamp": timestamp,
          "Log Level": log_level,
          "Source": source,
          "Message": message
        })
  return pd.DataFrame(log_entries)


def process_server_logs_and_find_anomalies(log_file_path: str, output_file_path='./anomalies.csv'):
  """
  Parse a server log file, analyze for anomalies, and save the results.

  Args:
      log_file_path (str): Path to the log file.
      output_file_path (str): Path to save the anomalies in a structured format.

  Returns:
      dict: Summary of anomalies detected, including the output file path.
  """

  def find_anomalies_in_logs(log_data):
    """
    Use Ollama to identify anomalies in log data.
    """
    anomalies = []
    for _, row in log_data.iterrows():
      prompt = f"""
          Analyze the following log entry and determine if it is an anomaly.
          Log Entry: {json.dumps(row.to_dict())}
          Respond with 'Anomaly' if it is an anomaly or 'Normal' if it is not.
          """
      response = chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
      )
      result = response.message.content.strip()
      if result.lower() == "anomaly":
        anomalies.append(row)
    return pd.DataFrame(anomalies)

  # Main Execution
  log_data = parse_log_file(log_file_path)
  anomalies = find_anomalies_in_logs(log_data)

  if not anomalies.empty:
    anomalies.to_csv(output_file_path, index=False)
    return {"status": "success", "message": f"Anomalies saved to {output_file_path}"}
  else:
    return {"status": "success", "message": "No anomalies detected."}


process_server_logs_and_find_anomalies_tool = {
    "type": "function",
    "function": {
        "name": "process_server_logs_and_find_anomalies",
        "description": "Parses a log file, analyzes for anomalies using AI, and saves the results.",
        "parameters": {
            "type": "object",
            "properties": {
                "log_file_path": {"type": "string", "description": "Path to the input server log file."},
                "output_file_path": {"type": "string", "description": "Path to save the detected anomalies."}
            },
            "required": ["log_file_path", "output_file_path"]
        }
    }
}

def add_two_numbers(a: int, b: int) -> int:
  """
  Add two numbers

  Args:
    a (int): The first number
    b (int): The second number

  Returns:
    int: The sum of the two numbers
  """
  return int(a) + int(b)


def subtract_two_numbers(a: int, b: int) -> int:
  """
  Subtract two numbers
  """
  return int(a) - int(b)


# Tools can still be manually defined and passed into chat
subtract_two_numbers_tool = {
  'type': 'function',
  'function': {
    'name': 'subtract_two_numbers',
    'description': 'Subtract two numbers',
    'parameters': {
      'type': 'object',
      'required': ['a', 'b'],
      'properties': {
        'a': {'type': 'integer', 'description': 'The first number'},
        'b': {'type': 'integer', 'description': 'The second number'},
      },
    },
  },
}

parse_nlp_to_sql_tool = {
  'type': 'function',
  'function': {
    'name': 'parse_nlp_to_sql',
    'description': 'Accepts a natural language query and returns a SQL query',
    'parameters': {
      'type': 'object',
      'required': ['user_prompt'],
      'properties': {
        'user_prompt': {'type': 'string', 'description': 'User prompt to answer business questions'},
      },
    },
  },
}


prompt = input("Enter your prompt here:")
messages = [{'role': 'user', 'content': prompt}]
print('Prompt:', messages[0]['content'])

available_functions = {
  'add_two_numbers': add_two_numbers,
  'subtract_two_numbers': subtract_two_numbers,
  'etl_process_log_file': etl_process_log_file,
  'parse_nlp_to_sql': parse_nlp_to_sql,
  'get_subreddit_sentiments': get_subreddit_sentiments,
  'process_server_logs_and_find_anomalies': process_server_logs_and_find_anomalies
}

response: ChatResponse = chat(
  'llama3.2',
  messages=messages,
  tools=[add_two_numbers, subtract_two_numbers_tool, etl_process_log_file_tool, parse_nlp_to_sql_tool,
         get_subreddit_sentiments, process_server_logs_and_find_anomalies_tool],
)

if response.message.tool_calls:
  for tool_call in response.message.tool_calls:
    if tool_call.function.name in available_functions:
      print(f"Calling function: {tool_call.function.name}")
      print(f"Arguments: {tool_call.function.arguments}")
      output = available_functions[tool_call.function.name](**tool_call.function.arguments)
      print("Function Output:", output)
    else:
      print(f"Tool {tool_call.function.name} not found in available_functions.")
else:
  print("No tool was invoked. Please refine your prompt.")
