'''
Author: jhq
Date: 2025-05-01 15:42:36
LastEditTime: 2025-05-10 18:45:37
Description: 
'''
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from openai import OpenAI
import sys
import os
import json
import ast
from dotenv import load_dotenv
load_dotenv()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()  
        self.llm = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("base_url"))      

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("server script must be .py or .js file")
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
    
    async def process_query(self, query: str) -> str:
        messages = [
            {"role": "user", "content": query}
        ]
        
        response = await self.session.list_tools()
        # available_tools = [{
        #     "type": "function",
        #     "function": {
        #         "name": tool.name,
        #         "description": tool.description,
        #         "parameters": tool.inputSchema
        #     }
        # } for tool in response.tools]
        available_tools = []
        
        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            max_tokens=8000,
            messages=messages,
            tools=available_tools
        )
        print(response)
        final_text = []
        assistant_message_content = []
        for choice in response.choices:
            assistant_message_content.append(choice.message)
            if choice.finish_reason == "stop":
                final_text.append(choice.message.content)                
            elif choice.finish_reason == "tool_calls":
                messages.append(choice.message.model_dump())
                # messages.append({
                #     "role": "assistant",
                #     "metdata": choice.message.tool_calls[0].function.name,
                #     "content": choice.message.tool_calls[0].function.arguments,
                # })
                for tool in choice.message.tool_calls:
                    tool_name = tool.function.name
                    tool_args = tool.function.arguments              
                    result = await self.session.call_tool(tool_name, ast.literal_eval(tool_args))
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")               
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool.id,                        
                        # "tool_call_id": tool.index,
                        "content": result.content[0].text                         
                    })
                print(messages)
                response = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    max_tokens=8000,
                    messages=messages,
                    tools=available_tools
                )
                final_text.append(response.choices[0].message.content)
        return "\n".join(final_text)
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\n MCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"Error: {e}")
                
    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_script_path>")
        sys.exit(1)
    
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

async def test_sse():
    async with sse_client("https://mcp.amap.com/sse?key=c2dde0264c27597bce5a9b6f1a490b14") as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            response = await session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in tools])

if __name__ == "__main__":
    # asyncio.run(test_sse())
    asyncio.run(main())
        