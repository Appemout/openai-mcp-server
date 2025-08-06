import asyncio
import logging
import sys
from typing import Optional

import click
import mcp
import mcp.types as types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions

from .llm import LLMConnector

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def serve(openai_api_key: str) -> Server:
    server = Server("openai-server")
    connector = LLMConnector(openai_api_key)

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="ask-openai",
                description="Ask my assistant models a direct question",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Ask assistant"},
                        "model": {"type": "string", "default": "gpt-4", "enum": ["gpt-4", "gpt-3.5-turbo"]},
                        "temperature": {"type": "number", "default": 0.7, "minimum": 0, "maximum": 2},
                        "max_tokens": {"type": "integer", "default": 500, "minimum": 1, "maximum": 4000}
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="generate-image",
                description="Generate images using OpenAI DALL-E models",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string", 
                            "description": "Text description for image generation"
                        },
                        "model": {
                            "type": "string", 
                            "default": "dall-e-3", 
                            "enum": ["dall-e-2", "dall-e-3"],
                            "description": "DALL-E model version to use"
                        },
                        "size": {
                            "type": "string", 
                            "default": "1024x1024", 
                            "enum": ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"],
                            "description": "Image dimensions"
                        },
                        "quality": {
                            "type": "string", 
                            "default": "standard", 
                            "enum": ["standard", "hd"],
                            "description": "Image quality (only applicable to DALL-E 3)"
                        },
                        "n": {
                            "type": "integer", 
                            "default": 1, 
                            "minimum": 1, 
                            "maximum": 10,
                            "description": "Number of images to generate (1 for DALL-E 3, 1-10 for DALL-E 2)"
                        }
                    },
                    "required": ["prompt"]
                }
            )
        ]

    @server.call_tool()
    async def handle_tool_call(name: str, arguments: dict | None) -> list[types.TextContent]:
        try:
            if not arguments:
                raise ValueError("No arguments provided")

            if name == "ask-openai":
                response = await connector.ask_openai(
                    query=arguments["query"],
                    model=arguments.get("model", "gpt-4"),
                    temperature=arguments.get("temperature", 0.7),
                    max_tokens=arguments.get("max_tokens", 500)
                )
                return [types.TextContent(type="text", text=f"OpenAI Response:\n{response}")]
            
            elif name == "generate-image":
                # Extract and validate parameters
                prompt = arguments.get("prompt")
                if not prompt:
                    raise ValueError("Prompt is required for image generation")
                
                model = arguments.get("model", "dall-e-3")
                size = arguments.get("size", "1024x1024")
                quality = arguments.get("quality", "standard")
                n = arguments.get("n", 1)
                
                # Call the LLMConnector's generate_image method
                image_url = await connector.generate_image(
                    prompt=prompt,
                    model=model,
                    size=size,
                    quality=quality,
                    n=n
                )
                
                return [types.TextContent(type="text", text=f"Generated Image URL:\n{image_url}")]

            raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            logger.error(f"Tool call failed: {str(e)}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    return server

@click.command()
@click.option("--openai-api-key", envvar="OPENAI_API_KEY", required=True)
def main(openai_api_key: str):
    try:
        async def _run():
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                server = serve(openai_api_key)
                await server.run(
                    read_stream, write_stream,
                    InitializationOptions(
                        server_name="openai-server",
                        server_version="0.1.0",
                        capabilities=server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={}
                        )
                    )
                )
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.exception("Server failed")
        sys.exit(1)

if __name__ == "__main__":
    main()