import asyncio, os, sys
from agents import Agent, Runner, OpenAIChatCompletionsModel
from agents.mcp.server import MCPServerStdio
from openai import AsyncOpenAI

os.environ["OPENAI_API_KEY"] = "sk-proj-XXXXXXXXXXXXXXXXXXXXX"

pdf_server = MCPServerStdio(
    params={"command": sys.executable, "args": ["pdf_server.py"]},
)

llm = OpenAIChatCompletionsModel("gpt-4o-mini", openai_client=AsyncOpenAI())

assistant = Agent(
    name="PDF‑Analyst",
    instructions="Use the PDF tools to answer questions about documents.",
    model=llm,
    mcp_servers=[pdf_server],
)


async def main():
    async with pdf_server:          # connects & cleans up
        ask = "Give me a one‑paragraph overview of each pdf file in this pdf_for_mcp directory."
        result = await Runner.run(starting_agent=assistant, input=ask)
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
