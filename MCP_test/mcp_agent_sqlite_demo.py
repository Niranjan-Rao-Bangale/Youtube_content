import asyncio, os, sys
from agents import Agent, Runner, OpenAIChatCompletionsModel
from agents.mcp.server import MCPServerStdio    # path changed in ≥ 0.4.0
from openai import AsyncOpenAI

os.environ["OPENAI_API_KEY"] = "sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXX"
sqlite_server = MCPServerStdio(
    params={"command": sys.executable, "args": ["sqlite_mcp_server.py"]},
)

llm = OpenAIChatCompletionsModel("gpt-4o-mini", openai_client=AsyncOpenAI())


async def main():
    async with sqlite_server:          # CONNECTS + auto‑cleanup
        agent = Agent(
            name="DB‑Assistant",
            instructions="Answer questions by querying the SQLite DB via MCP tools.",
            model=llm,
            mcp_servers=[sqlite_server],
        )

        result = await Runner.run(
            starting_agent=agent,
            input="Which album is most quantity sold and in which country and year?"
        )
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
