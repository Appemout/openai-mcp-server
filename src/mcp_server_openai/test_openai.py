import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import mcp.types as types
from .llm import LLMConnector
from .server import serve

@pytest.mark.asyncio
async def test_ask_openai():
    print("\nTesting OpenAI API call...")
    connector = LLMConnector("your-openai-key")
    response = await connector.ask_openai("Hello, how are you?")
    print(f"OpenAI Response: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


class TestImageGeneration:
    """Test suite for image generation functionality."""
    
    @pytest.fixture
    def connector(self):
        """Create a LLMConnector instance for testing."""
        return LLMConnector("test-api-key")
    
    @pytest.fixture
    def mock_openai_response_single(self):
        """Mock OpenAI API response for single image."""
        mock_response = MagicMock()
        mock_image = MagicMock()
        mock_image.url = "https://example.com/generated-image.png"
        mock_response.data = [mock_image]
        return mock_response
    
    @pytest.fixture
    def mock_openai_response_multiple(self):
        """Mock OpenAI API response for multiple images."""
        mock_response = MagicMock()
        mock_images = []
        for i in range(3):
            mock_image = MagicMock()
            mock_image.url = f"https://example.com/generated-image-{i+1}.png"
            mock_images.append(mock_image)
        mock_response.data = mock_images
        return mock_response

    @pytest.mark.asyncio
    async def test_generate_image_default_parameters(self, connector, mock_openai_response_single):
        """Test image generation with default parameters."""
        with patch.object(connector.client.images, 'generate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_openai_response_single
            
            result = await connector.generate_image("A beautiful sunset")
            
            mock_generate.assert_called_once_with(
                prompt="A beautiful sunset",
                model="dall-e-3",
                size="1024x1024",
                n=1,
                quality="standard"
            )
            assert result == "https://example.com/generated-image.png"

    @pytest.mark.asyncio
    async def test_generate_image_dalle2_parameters(self, connector, mock_openai_response_single):
        """Test image generation with DALL-E 2 specific parameters."""
        with patch.object(connector.client.images, 'generate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_openai_response_single
            
            result = await connector.generate_image(
                prompt="A cat playing piano",
                model="dall-e-2",
                size="512x512",
                n=1
            )
            
            # Quality should not be included for DALL-E 2
            mock_generate.assert_called_once_with(
                prompt="A cat playing piano",
                model="dall-e-2",
                size="512x512",
                n=1
            )
            assert result == "https://example.com/generated-image.png"

    @pytest.mark.asyncio
    async def test_generate_image_dalle3_hd_quality(self, connector, mock_openai_response_single):
        """Test image generation with DALL-E 3 HD quality."""
        with patch.object(connector.client.images, 'generate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_openai_response_single
            
            result = await connector.generate_image(
                prompt="A detailed landscape",
                model="dall-e-3",
                size="1792x1024",
                quality="hd"
            )
            
            mock_generate.assert_called_once_with(
                prompt="A detailed landscape",
                model="dall-e-3",
                size="1792x1024",
                n=1,
                quality="hd"
            )
            assert result == "https://example.com/generated-image.png"

    @pytest.mark.asyncio
    async def test_generate_multiple_images_dalle2(self, connector, mock_openai_response_multiple):
        """Test generating multiple images with DALL-E 2."""
        with patch.object(connector.client.images, 'generate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_openai_response_multiple
            
            result = await connector.generate_image(
                prompt="Abstract art",
                model="dall-e-2",
                size="256x256",
                n=3
            )
            
            mock_generate.assert_called_once_with(
                prompt="Abstract art",
                model="dall-e-2",
                size="256x256",
                n=3
            )
            expected_urls = "\n".join([
                "https://example.com/generated-image-1.png",
                "https://example.com/generated-image-2.png",
                "https://example.com/generated-image-3.png"
            ])
            assert result == expected_urls

    @pytest.mark.asyncio
    async def test_api_error_handling(self, connector):
        """Test handling of OpenAI API errors."""
        with patch.object(connector.client.images, 'generate', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = Exception("API rate limit exceeded")
            
            with pytest.raises(Exception) as exc_info:
                await connector.generate_image("Test prompt")
            
            assert "API rate limit exceeded" in str(exc_info.value)

    # Parameter validation tests
    def test_validate_empty_prompt(self, connector):
        """Test validation of empty prompt."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            connector._validate_image_parameters("", "dall-e-3", "1024x1024", "standard", 1)
        
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            connector._validate_image_parameters("   ", "dall-e-3", "1024x1024", "standard", 1)

    def test_validate_prompt_length_dalle3(self, connector):
        """Test validation of prompt length for DALL-E 3."""
        long_prompt = "x" * 4001  # Exceeds 4000 character limit
        with pytest.raises(ValueError, match="Prompt too long for DALL-E 3"):
            connector._validate_image_parameters(long_prompt, "dall-e-3", "1024x1024", "standard", 1)

    def test_validate_prompt_length_dalle2(self, connector):
        """Test validation of prompt length for DALL-E 2."""
        long_prompt = "x" * 1001  # Exceeds 1000 character limit
        with pytest.raises(ValueError, match="Prompt too long for DALL-E 2"):
            connector._validate_image_parameters(long_prompt, "dall-e-2", "1024x1024", "standard", 1)

    def test_validate_invalid_model(self, connector):
        """Test validation of invalid model."""
        with pytest.raises(ValueError, match="Invalid model 'invalid-model'"):
            connector._validate_image_parameters("Test", "invalid-model", "1024x1024", "standard", 1)

    def test_validate_invalid_size_dalle2(self, connector):
        """Test validation of invalid size for DALL-E 2."""
        with pytest.raises(ValueError, match="Invalid size '1792x1024' for model 'dall-e-2'"):
            connector._validate_image_parameters("Test", "dall-e-2", "1792x1024", "standard", 1)

    def test_validate_invalid_size_dalle3(self, connector):
        """Test validation of invalid size for DALL-E 3."""
        with pytest.raises(ValueError, match="Invalid size '256x256' for model 'dall-e-3'"):
            connector._validate_image_parameters("Test", "dall-e-3", "256x256", "standard", 1)

    def test_validate_invalid_quality_dalle3(self, connector):
        """Test validation of invalid quality for DALL-E 3."""
        with pytest.raises(ValueError, match="Invalid quality 'ultra' for DALL-E 3"):
            connector._validate_image_parameters("Test", "dall-e-3", "1024x1024", "ultra", 1)

    def test_validate_quality_ignored_dalle2(self, connector):
        """Test that quality parameter is ignored for DALL-E 2 (should not raise error)."""
        # This should not raise an error, but may log a warning
        connector._validate_image_parameters("Test", "dall-e-2", "1024x1024", "hd", 1)

    def test_validate_invalid_n_dalle3(self, connector):
        """Test validation of invalid n parameter for DALL-E 3."""
        with pytest.raises(ValueError, match="DALL-E 3 only supports generating 1 image"):
            connector._validate_image_parameters("Test", "dall-e-3", "1024x1024", "standard", 2)

    def test_validate_invalid_n_dalle2_too_high(self, connector):
        """Test validation of n parameter too high for DALL-E 2."""
        with pytest.raises(ValueError, match="DALL-E 2 supports generating 1-10 images"):
            connector._validate_image_parameters("Test", "dall-e-2", "1024x1024", "standard", 11)



    def test_validate_invalid_n_negative(self, connector):
        """Test validation of negative n parameter."""
        with pytest.raises(ValueError, match="DALL-E 2 supports generating 1-10 images"):
            connector._validate_image_parameters("Test", "dall-e-2", "1024x1024", "standard", -1)
    
    def test_validate_invalid_n_zero(self, connector):
        """Test validation of zero n parameter."""
        with pytest.raises(ValueError, match="DALL-E 2 supports generating 1-10 images"):
            connector._validate_image_parameters("Test", "dall-e-2", "1024x1024", "standard", 0)

    # Edge case tests
    def test_valid_parameters_dalle2_all_sizes(self, connector):
        """Test all valid sizes for DALL-E 2."""
        valid_sizes = ["256x256", "512x512", "1024x1024"]
        for size in valid_sizes:
            # Should not raise any exception
            connector._validate_image_parameters("Test", "dall-e-2", size, "standard", 1)

    def test_valid_parameters_dalle3_all_sizes(self, connector):
        """Test all valid sizes for DALL-E 3."""
        valid_sizes = ["1024x1024", "1024x1792", "1792x1024"]
        for size in valid_sizes:
            # Should not raise any exception
            connector._validate_image_parameters("Test", "dall-e-3", size, "standard", 1)

    def test_valid_parameters_dalle3_all_qualities(self, connector):
        """Test all valid qualities for DALL-E 3."""
        valid_qualities = ["standard", "hd"]
        for quality in valid_qualities:
            # Should not raise any exception
            connector._validate_image_parameters("Test", "dall-e-3", "1024x1024", quality, 1)

    def test_valid_parameters_dalle2_max_images(self, connector):
        """Test maximum number of images for DALL-E 2."""
        # Should not raise any exception
        connector._validate_image_parameters("Test", "dall-e-2", "1024x1024", "standard", 10)

    def test_prompt_length_boundary_dalle3(self, connector):
        """Test prompt length at boundary for DALL-E 3."""
        # Exactly 4000 characters should be valid
        prompt_4000 = "x" * 4000
        connector._validate_image_parameters(prompt_4000, "dall-e-3", "1024x1024", "standard", 1)

    def test_prompt_length_boundary_dalle2(self, connector):
        """Test prompt length at boundary for DALL-E 2."""
        # Exactly 1000 characters should be valid
        prompt_1000 = "x" * 1000
        connector._validate_image_parameters(prompt_1000, "dall-e-2", "1024x1024", "standard", 1)


class TestImageGenerationIntegration:
    """Integration tests for the complete image generation tool flow."""
    
    @pytest.fixture
    def connector(self):
        """Create a LLMConnector instance for testing."""
        return LLMConnector("test-api-key")
    
    @pytest.fixture
    def mock_openai_image_response_single(self):
        """Mock OpenAI API response for single image."""
        mock_response = MagicMock()
        mock_image = MagicMock()
        mock_image.url = "https://example.com/generated-image.png"
        mock_response.data = [mock_image]
        return mock_response
    
    @pytest.fixture
    def mock_openai_image_response_multiple(self):
        """Mock OpenAI API response for multiple images."""
        mock_response = MagicMock()
        mock_images = []
        for i in range(2):
            mock_image = MagicMock()
            mock_image.url = f"https://example.com/generated-image-{i+1}.png"
            mock_images.append(mock_image)
        mock_response.data = mock_images
        return mock_response

    def _get_call_tool_handler(self, server):
        """Helper method to get the call_tool handler from server."""
        import mcp.types as types
        call_tool_handler = server.request_handlers.get(types.CallToolRequest)
        assert call_tool_handler is not None, "call_tool handler not found"
        return call_tool_handler
    
    async def _call_tool(self, server, tool_name, arguments):
        """Helper method to call a tool with proper request structure."""
        import mcp.types as types
        call_tool_handler = self._get_call_tool_handler(server)
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name=tool_name,
                arguments=arguments
            )
        )
        return await call_tool_handler(request)

    @pytest.mark.asyncio
    async def test_tool_discovery_includes_generate_image(self):
        """Test that tool discovery includes the generate-image tool."""
        server = serve("test-api-key")
        
        # Get the list_tools handler from the request_handlers dictionary
        import mcp.types as types
        list_tools_handler = server.request_handlers.get(types.ListToolsRequest)
        assert list_tools_handler is not None, "list_tools handler not found"
        
        # Create a ListToolsRequest and call the handler
        request = types.ListToolsRequest(method="tools/list")
        result = await list_tools_handler(request)
        
        # Extract tools from ServerResult
        tools = result.root.tools
        
        # Verify tools are returned
        assert isinstance(tools, list)
        assert len(tools) >= 2  # Should have at least ask-openai and generate-image
        
        # Find the generate-image tool
        generate_image_tool = None
        for tool in tools:
            if tool.name == "generate-image":
                generate_image_tool = tool
                break
        
        assert generate_image_tool is not None, "generate-image tool not found in tool list"
        
        # Verify tool properties
        assert generate_image_tool.name == "generate-image"
        assert generate_image_tool.description == "Generate images using OpenAI DALL-E models"
        assert generate_image_tool.inputSchema is not None
        
        # Verify schema structure
        schema = generate_image_tool.inputSchema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "prompt" in schema["required"]
        
        # Verify all expected parameters are in schema
        properties = schema["properties"]
        expected_params = ["prompt", "model", "size", "quality", "n"]
        for param in expected_params:
            assert param in properties, f"Parameter '{param}' missing from schema"
        
        # Verify parameter defaults and constraints
        assert properties["model"]["default"] == "dall-e-3"
        assert properties["size"]["default"] == "1024x1024"
        assert properties["quality"]["default"] == "standard"
        assert properties["n"]["default"] == 1
        assert properties["n"]["minimum"] == 1
        assert properties["n"]["maximum"] == 10

    @pytest.mark.asyncio
    async def test_complete_tool_call_flow_single_image(self, mock_openai_image_response_single):
        """Test complete request-response cycle for single image generation."""
        server = serve("test-api-key")
        
        # Mock the generate_image method directly
        with patch.object(LLMConnector, 'generate_image', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "https://example.com/generated-image.png"
            
            # Call the tool with valid arguments
            arguments = {
                "prompt": "A beautiful sunset over mountains",
                "model": "dall-e-3",
                "size": "1024x1024",
                "quality": "standard",
                "n": 1
            }
            
            result = await self._call_tool(server, "generate-image", arguments)
            
            # Extract the content from ServerResult
            content = result.root.content
            
            # Verify response structure
            assert isinstance(content, list)
            assert len(content) == 1
            assert isinstance(content[0], types.TextContent)
            assert content[0].type == "text"
            
            # Verify response content
            expected_text = "Generated Image URL:\nhttps://example.com/generated-image.png"
            assert content[0].text == expected_text
            
            # Verify generate_image was called correctly
            mock_generate.assert_called_once_with(
                prompt="A beautiful sunset over mountains",
                model="dall-e-3",
                size="1024x1024",
                quality="standard",
                n=1
            )

    @pytest.mark.asyncio
    async def test_complete_tool_call_flow_multiple_images(self, mock_openai_image_response_multiple):
        """Test complete request-response cycle for multiple image generation."""
        server = serve("test-api-key")
        
        # Mock the generate_image method directly
        with patch.object(LLMConnector, 'generate_image', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "https://example.com/generated-image-1.png\nhttps://example.com/generated-image-2.png"
            
            # Call the tool with DALL-E 2 for multiple images
            arguments = {
                "prompt": "Abstract art patterns",
                "model": "dall-e-2",
                "size": "512x512",
                "n": 2
            }
            
            result = await self._call_tool(server, "generate-image", arguments)
            
            # Extract the content from ServerResult
            content = result.root.content
            
            # Verify response structure
            assert isinstance(content, list)
            assert len(content) == 1
            assert isinstance(content[0], types.TextContent)
            assert content[0].type == "text"
            
            # Verify response content contains multiple URLs
            expected_text = "Generated Image URL:\nhttps://example.com/generated-image-1.png\nhttps://example.com/generated-image-2.png"
            assert content[0].text == expected_text
            
            # Verify generate_image was called correctly
            mock_generate.assert_called_once_with(
                prompt="Abstract art patterns",
                model="dall-e-2",
                size="512x512",
                quality="standard",
                n=2
            )

    @pytest.mark.asyncio
    async def test_tool_call_with_default_parameters(self, mock_openai_image_response_single):
        """Test tool call with minimal arguments using default parameters."""
        server = serve("test-api-key")
        
        # Mock the generate_image method directly
        with patch.object(LLMConnector, 'generate_image', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "https://example.com/generated-image.png"
            
            # Call the tool with only required argument
            arguments = {
                "prompt": "A simple drawing"
            }
            
            result = await self._call_tool(server, "generate-image", arguments)
            
            # Extract the content from ServerResult
            content = result.root.content
            
            # Verify response
            assert isinstance(content, list)
            assert len(content) == 1
            assert isinstance(content[0], types.TextContent)
            assert "Generated Image URL:" in content[0].text
            assert "https://example.com/generated-image.png" in content[0].text
            
            # Verify default parameters were used
            mock_generate.assert_called_once_with(
                prompt="A simple drawing",
                model="dall-e-3",  # default
                size="1024x1024",  # default
                quality="standard",  # default
                n=1  # default
            )

    @pytest.mark.asyncio
    async def test_error_propagation_missing_prompt(self):
        """Test error propagation when prompt is missing."""
        server = serve("test-api-key")
        
        # Get the call_tool handler
        call_tool_handler = self._get_call_tool_handler(server)
        
        # Call the tool without prompt
        arguments = {
            "model": "dall-e-3"
        }
        
        result = await self._call_tool(server, "generate-image", arguments)
        
        # Extract the content from ServerResult
        content = result.root.content
        
        # Verify error response
        assert isinstance(content, list)
        assert len(content) == 1
        assert isinstance(content[0], types.TextContent)
        assert content[0].type == "text"
        assert "Input validation error:" in content[0].text or "Error:" in content[0].text
        assert "prompt" in content[0].text.lower() and "required" in content[0].text.lower()

    @pytest.mark.asyncio
    async def test_error_propagation_empty_prompt(self):
        """Test error propagation when prompt is empty."""
        server = serve("test-api-key")
        
        # Get the call_tool handler
        call_tool_handler = self._get_call_tool_handler(server)
        
        # Call the tool with empty prompt
        arguments = {
            "prompt": ""
        }
        
        result = await self._call_tool(server, "generate-image", arguments)
        
        # Extract the content from ServerResult
        content = result.root.content
        
        # Verify error response
        assert isinstance(content, list)
        assert len(content) == 1
        assert isinstance(content[0], types.TextContent)
        assert content[0].type == "text"
        assert "Error:" in content[0].text
        assert "Prompt cannot be empty" in content[0].text or "prompt" in content[0].text.lower()

    @pytest.mark.asyncio
    async def test_error_propagation_invalid_parameters(self):
        """Test error propagation for invalid parameter combinations."""
        server = serve("test-api-key")
        
        # Get the call_tool handler
        call_tool_handler = self._get_call_tool_handler(server)
        
        # Test invalid model
        arguments = {
            "prompt": "Test prompt",
            "model": "invalid-model"
        }
        
        result = await self._call_tool(server, "generate-image", arguments)
        
        # Extract the content from ServerResult
        content = result.root.content
        
        # Verify error response
        assert isinstance(content, list)
        assert len(content) == 1
        assert isinstance(content[0], types.TextContent)
        assert "Input validation error:" in content[0].text or "Error:" in content[0].text
        assert "invalid-model" in content[0].text or "Invalid model" in content[0].text

    @pytest.mark.asyncio
    async def test_error_propagation_dalle3_multiple_images(self):
        """Test error propagation when trying to generate multiple images with DALL-E 3."""
        server = serve("test-api-key")
        
        # Get the call_tool handler
        call_tool_handler = self._get_call_tool_handler(server)
        
        # Call the tool with DALL-E 3 and n > 1
        arguments = {
            "prompt": "Test prompt",
            "model": "dall-e-3",
            "n": 2
        }
        
        result = await self._call_tool(server, "generate-image", arguments)
        
        # Extract the content from ServerResult
        content = result.root.content
        
        # Verify error response
        assert isinstance(content, list)
        assert len(content) == 1
        assert isinstance(content[0], types.TextContent)
        assert "Error:" in content[0].text
        assert "DALL-E 3 only supports generating 1 image" in content[0].text

    @pytest.mark.asyncio
    async def test_error_propagation_openai_api_error(self):
        """Test error propagation when OpenAI API returns an error."""
        server = serve("test-api-key")
        
        # Get the call_tool handler
        call_tool_handler = self._get_call_tool_handler(server)
        
        # Mock the generate_image method to raise an exception
        with patch.object(LLMConnector, 'generate_image', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = Exception("API rate limit exceeded")
            
            # Call the tool
            arguments = {
                "prompt": "Test prompt"
            }
            
            result = await self._call_tool(server, "generate-image", arguments)
            
            # Extract the content from ServerResult
            content = result.root.content
            
            # Verify error response
            assert isinstance(content, list)
            assert len(content) == 1
            assert isinstance(content[0], types.TextContent)
            assert "Error:" in content[0].text
            assert "API rate limit exceeded" in content[0].text

    @pytest.mark.asyncio
    async def test_error_propagation_no_arguments(self):
        """Test error propagation when no arguments are provided."""
        server = serve("test-api-key")
        
        # Get the call_tool handler
        call_tool_handler = self._get_call_tool_handler(server)
        
        # Call the tool with None arguments
        result = await self._call_tool(server, "generate-image", None)
        
        # Extract the content from ServerResult
        content = result.root.content
        
        # Verify error response
        assert isinstance(content, list)
        assert len(content) == 1
        assert isinstance(content[0], types.TextContent)
        assert "Input validation error:" in content[0].text or "Error:" in content[0].text
        assert "prompt" in content[0].text.lower() and "required" in content[0].text.lower()

    @pytest.mark.asyncio
    async def test_tool_call_unknown_tool(self):
        """Test error handling for unknown tool names."""
        server = serve("test-api-key")
        
        # Get the call_tool handler
        call_tool_handler = self._get_call_tool_handler(server)
        
        # Call with unknown tool name
        arguments = {"prompt": "test"}
        result = await self._call_tool(server, "unknown-tool", arguments)
        
        # Extract the content from ServerResult
        content = result.root.content
        
        # Verify error response
        assert isinstance(content, list)
        assert len(content) == 1
        assert isinstance(content[0], types.TextContent)
        assert "Error:" in content[0].text
        assert "Unknown tool: unknown-tool" in content[0].text

    @pytest.mark.asyncio
    async def test_dalle2_specific_parameters(self, mock_openai_image_response_single):
        """Test DALL-E 2 specific parameter handling."""
        server = serve("test-api-key")
        
        # Get the call_tool handler
        call_tool_handler = self._get_call_tool_handler(server)
        
        # Mock the generate_image method directly
        with patch.object(LLMConnector, 'generate_image', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "https://example.com/generated-image.png"
            
            # Call the tool with DALL-E 2 parameters
            arguments = {
                "prompt": "A cat playing piano",
                "model": "dall-e-2",
                "size": "256x256",
                "quality": "hd",  # Should be ignored for DALL-E 2
                "n": 5
            }
            
            result = await self._call_tool(server, "generate-image", arguments)
            
            # Extract the content from ServerResult
            content = result.root.content
            
            # Verify successful response
            assert isinstance(content, list)
            assert len(content) == 1
            assert isinstance(content[0], types.TextContent)
            assert "Generated Image URL:" in content[0].text
            
            # Verify generate_image was called correctly
            mock_generate.assert_called_once_with(
                prompt="A cat playing piano",
                model="dall-e-2",
                size="256x256",
                quality="hd",
                n=5
            )

    @pytest.mark.asyncio
    async def test_dalle3_hd_quality_parameters(self, mock_openai_image_response_single):
        """Test DALL-E 3 HD quality parameter handling."""
        server = serve("test-api-key")
        
        # Get the call_tool handler
        call_tool_handler = self._get_call_tool_handler(server)
        
        # Mock the generate_image method directly
        with patch.object(LLMConnector, 'generate_image', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "https://example.com/generated-image.png"
            
            # Call the tool with DALL-E 3 HD quality
            arguments = {
                "prompt": "A detailed landscape painting",
                "model": "dall-e-3",
                "size": "1792x1024",
                "quality": "hd",
                "n": 1
            }
            
            result = await self._call_tool(server, "generate-image", arguments)
            
            # Extract the content from ServerResult
            content = result.root.content
            
            # Verify successful response
            assert isinstance(content, list)
            assert len(content) == 1
            assert isinstance(content[0], types.TextContent)
            assert "Generated Image URL:" in content[0].text
            
            # Verify generate_image was called correctly with quality parameter
            mock_generate.assert_called_once_with(
                prompt="A detailed landscape painting",
                model="dall-e-3",
                size="1792x1024",
                quality="hd",
                n=1
            )