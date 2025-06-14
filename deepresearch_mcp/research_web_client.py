import asyncio
import json
import os
from typing import List, Union, Any, Callable, Awaitable

import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Ensure your API key is set in the environment where both client AND server are run
# For client, it's used directly. For server, research_web_server.py uses os.getenv.
openai.api_key = os.getenv("OPENAI_API_KEY")

class MCP_ChatBot:

    def __init__(self, output_callback: Callable[[str], None] = None, debug_callback: Callable[[str], None] = None):
        self.session: ClientSession = None
        self.available_tools: List[dict] = []
        self.initial_messages = [{
            'role': 'system',
            'content': (
                "You are a sophisticated Deep Research Assistant capable of using various tools "
                "to find detailed information. When asked a question that requires more than a simple "
                "web search, follow these steps:\n"
                "1. **Start with `web_search`** to find relevant articles and documents. "
                "The output will be a JSON string of articles with 'title', 'summary', and 'url'.\n"
                "2. **Use `extract_urls_from_json`** on the output of `web_search` to get a clean list of URLs.\n"
                "3. **For each promising URL**, use `extract_text_from_url` to retrieve the full content "
                "for detailed analysis. Be mindful of the number of URLs you process fully.\n"
                "4. **If provided with local file paths (PDF or image)**, use `extract_text_from_pdf`"
                " or `perform_ocr`.\n"
                "5. **After gathering sufficient text content**, use `summarize_sources` to "
                "synthesize the key findings into a concise summary.\n"
                "6. **Finally, provide a comprehensive answer** to the user's query, citing your sources clearly.\n"
                "Think step-by-step and show your reasoning process. Prioritize concise summaries in the conversation history."
            )
        }]
        self.output_callback = output_callback if output_callback else print
        self.debug_callback = debug_callback if debug_callback else print
        self.conversation_history = []

    def _log_debug(self, message: str):
        self.debug_callback(message)

    def _log_output(self, message: str):
        self.output_callback(message)

    def _add_to_history(self, role: str, content: str, tool_calls: List[dict] = None):
        message = {"role": role, "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.conversation_history.append(message)

    async def process_query(self, query):
        def extract_tool_content(raw_content: Any) -> str:
            """
            Extracts string content from various MCP tool return types.
            Handles TextContent objects, lists of content objects, and raw strings/dicts/lists.
            """
            if isinstance(raw_content, list):
                parts = []
                for item in raw_content:
                    # Check if item is a Pydantic model (like TextContent) with a .text attribute
                    if hasattr(item, "text") and isinstance(item.text, str):
                        parts.append(item.text)
                    # Check for embedded resources
                    elif hasattr(item, "uri"):
                        parts.append(f"[Embedded resource URI: {item.uri}]")
                    # Fallback for other objects in the list
                    else:
                        parts.append(str(item))
                return "\n\n".join(parts)

            # Direct TextContent object
            elif hasattr(raw_content, "text") and isinstance(raw_content.text, str):
                return raw_content.text

            # Raw string content
            elif isinstance(raw_content, str):
                return raw_content

            # Direct dictionary or list that can be JSON dumped
            elif isinstance(raw_content, (dict, list)):
                try:
                    return json.dumps(raw_content, indent=2) # Use indent for readability
                except Exception:
                    return str(raw_content) # Fallback if JSON dumping fails

            # Any other raw type
            return str(raw_content)

        messages = self.initial_messages + [{'role': 'user', 'content': query}]
        self._add_to_history("user", query)
        self._log_output(f"**You:** {query}\n\n")
        self._log_debug(f"Starting research for query: '{query}'")

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=self.available_tools
        )

        process_query_loop = True
        loop_counter = 0
        MAX_LOOPS = 8
        MAX_HISTORY_CHARS = 30000
        SUMMARIZE_TOOL_OUTPUT_THRESHOLD = 2000

        while process_query_loop:
            loop_counter += 1
            if loop_counter > MAX_LOOPS:
                self._log_debug("Max loop count reached. Exiting.")
                self._add_to_history("system", "Max research iterations reached. "
                                               "Providing current findings.")
                break

            self._log_debug(f"\n--- Research Iteration {loop_counter} ---")
            self._log_debug("Calling OpenAI API...")

            message_from_openai = response.choices[0].message
            assistant_message_dict = {
                "role": "assistant",
                "content": message_from_openai.content if message_from_openai.content else None
            }
            if message_from_openai.tool_calls:
                assistant_message_dict["tool_calls"] = [tc.model_dump() for tc in message_from_openai.tool_calls]
            messages.append(assistant_message_dict)
            self._log_debug(f"\nTotal tool calls:\n {message_from_openai.tool_calls} \n")
            if message_from_openai.tool_calls: 
                tool_outputs = []
                for tool_call in message_from_openai.tool_calls:
                    tool_id = tool_call.id
                    tool_name = tool_call.function.name
                    raw_tool_args = tool_call.function.arguments

                    self._log_debug(f"\n🤖 Assistant wants to call tool: {tool_name}")
                    self._log_debug(f"   Args RAW: {raw_tool_args}")
                    self._log_output(f"   Type of Args RAW: {type(raw_tool_args)}")

                    try:
                        tool_args_dict = json.loads(raw_tool_args)
                        self._log_debug(f"   Args PARSED: {tool_args_dict}")
                    except json.JSONDecodeError as e:
                        self._log_debug(f"Error: Could not parse tool arguments JSON for '{tool_name}': {e}")
                        tool_outputs.append({
                            "tool_call_id": tool_id,
                            "output": f"Error: Failed to parse tool arguments: {e}"
                        })
                        continue

                    if tool_name == 'extract_urls_from_json' and 'json_input' in tool_args_dict:
                        arg_value = tool_args_dict['json_input']
                        self._log_debug(f"   Type of json_string value (before fix): {type(arg_value)}") # DEBUG
                        if not isinstance(arg_value, str):
                            self._log_debug(f"   WARNING: 'json_input' argument for '{tool_name}' is not a string. Attempting to re-stringify.")
                            try:
                                tool_args_dict['json_input'] = json.dumps(arg_value)
                                self._log_debug("   Re-stringify successful.")
                                self._log_debug(f"   Type of json_string value (AFTER fix): {type(tool_args_dict['json_input'])}") # DEBUG
                            except Exception as e:
                                self._log_debug(f"   ERROR: Failed to re-stringify 'json_input' argument: {e}. Skipping tool call.")
                                tool_outputs.append({
                                    "tool_call_id": tool_id,
                                    "output": f"Error: Invalid 'json_input' argument type for tool {tool_name}. Failed to re-stringify: {e}"
                                })
                                continue

                    try:
                        mcp_result_object = await self.session.call_tool(tool_name, arguments=tool_args_dict)
                        tool_result_raw_content = mcp_result_object.content

                        tool_result_content = extract_tool_content(tool_result_raw_content)

                        self._log_debug(f"   Tool '{tool_name}' result (first 200 chars): {tool_result_content[:200]}...")

                        tool_output_for_llm = tool_result_content

                        if tool_name in ["extract_text_from_url", "extract_text_from_pdf"] and \
                                len(tool_result_content) > SUMMARIZE_TOOL_OUTPUT_THRESHOLD:
                            self._log_debug(f"   Tool '{tool_name}' output is large. Summarizing...")

                            try:
                                summary_result_obj = await self.session.call_tool("summarize_sources", arguments={
                                    "texts": [tool_result_content]
                                })
                                summarized_text = extract_tool_content(summary_result_obj.content)
                                tool_output_for_llm = f"Summarized output from '{tool_name}': {summarized_text}"
                            except Exception as sum_e:
                                self._log_debug(f"   Summarization failed: {sum_e}. Truncating output.")
                                tool_output_for_llm = f"{tool_result_content[:1000]}... [truncated]"

                        elif tool_name == "web_search" and len(tool_result_content) > SUMMARIZE_TOOL_OUTPUT_THRESHOLD:
                            tool_output_for_llm = f"Web search results (truncated): {tool_result_content[:1000]}..."

                    except Exception as e:
                        self._log_debug(f"Error: Tool '{tool_name}' execution failed: {e}")
                        tool_output_for_llm = f"Error: Tool execution failed: {e}"

                    tool_outputs.append({
                        "tool_call_id": tool_id,
                        "output": tool_output_for_llm
                    })

                for output in tool_outputs:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": output["tool_call_id"],
                        "content": output["output"]
                    })

                current_history_length = 0
                for m in messages:
                    if m.get('content'):
                        current_history_length += len(str(m['content']))
                    if m.get('tool_calls'):
                        for tc in m['tool_calls']:
                            current_history_length += len(json.dumps(tc))

                while current_history_length > MAX_HISTORY_CHARS and len(messages) > 2:
                    removed_message = messages.pop(1)
                    if removed_message.get('content'):
                        current_history_length -= len(str(removed_message['content']))
                    if removed_message.get('tool_calls'):
                        for tc in removed_message['tool_calls']:
                            current_history_length -= len(json.dumps(tc))
                    self._log_debug(f"   Pruned old message. Current size: {current_history_length} chars.")

                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=self.available_tools
                )

            elif message_from_openai.content:
                self._log_output(message_from_openai.content)
                process_query_loop = False

            else:
                self._log_output("No content or tool calls in the response. Ending interaction.")
                process_query_loop = False

    async def chat_loop(self):
        """Run an interactive chat loop"""
        self._log_debug("\nMCP Chatbot Started!")
        self._log_debug("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                await self.process_query(query)
                print("\n")

            except Exception as e:
                self._log_debug(f"\nError in chat loop: {str(e)}")

    async def connect_to_server_and_run(self):
        server_params = StdioServerParameters(
            command="python3.12",
            args=["research_web_server.py"],
            env={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
            capture_error=True
        )
        self._log_debug("Attempting to connect to the DeepResearch MCP server...")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                self._log_debug("Initializing connection...")
                # Initialize the connection
                response = await session.initialize()
                self._log_debug(f"connection response received: {response}.")
                # if response.error:  # 'error' is a field on InitializeResult for MCP-defined errors
                #     self._log_debug(f"Server reported MCP error during init: {response.error}")
                #     self._log_output(f"**Error:** Server reported an error during initialization: {response.error}")
                #     raise Exception("Server reported an MCP error during startup.")

                # List available tools from the server
                response = await session.list_tools()

                # Transform tools into the OpenAI format
                self.available_tools = []
                for tool in response.tools:
                    self.available_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })

                tool_names = [tool_def["function"]["name"] for tool_def in self.available_tools]
                self._log_debug(f"\nConnected to server with tools: {tool_names}")

                await self.chat_loop()

    async def connect_to_server(self):
        # server_params = StdioServerParameters(
        #     command="uv",
        #     args=["run", "research_web_server.py"],
        #     env={"OPENAI_API_KEY":  os.getenv("OPENAI_API_KEY")}
        # )
        server_params = StdioServerParameters(
            command="python3.12",
            args=["research_web_server.py"],
            env={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
            capture_error=True
        )
        self._log_debug("Attempting to connect to the DeepResearch MCP server...")
        try:
            # The context manager ensures proper cleanup
            self.session = await stdio_client(server_params).__aenter__()
            self._log_debug("Connected to server")
            self._log_output("Connected to server")
            response = await self.session.initialize()
            self._log_debug(f"Connected to server response: {response}")
            self._log_output(f"Connected to server response: {response}")

            if response.error_output:
                error_details = response.error_output.decode('utf-8')
                self._log_debug(f"Server subprocess stderr output:\n{error_details}")
                self._log_output(f"**Error:** Server subprocess crashed. Details in debug log.")
                await self.disconnect_from_server()
                return False

            response = await self.session.list_tools()


            self.available_tools = []
            for tool in response.tools:
                self.available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            self._log_debug(
                f"Connected to server with tools: {[tool_def['function']['name'] for tool_def in self.available_tools]}")
            return True
        except Exception as e:
            self._log_output(
                f"**Error:** Could not connect to the MCP server. Please ensure `research_web_server.py` is running or accessible and `OPENAI_API_KEY` is set. Details: {e}")
            self._log_debug(f"MCP server connection error: {e}")
            self.session = None  # Ensure session is None on failure
            return False

    async def disconnect_from_server(self):
        if self.session:
            await self.session.__aexit__(None, None, None)
            self.session = None
            self._log_debug("Disconnected from MCP server.")




async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    print("Starting DeepResearch MCP Client...")
    print(f"Checking for OPENAI_API_KEY environment variable is set...")
    if os.getenv("OPENAI_API_KEY") is None:
        print("Error: OPENAI_API_KEY environment variable not set. Please set it before running.")
    else:
        asyncio.run(main())
