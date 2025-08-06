import logging
from typing import Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class LLMConnector:
    def __init__(self, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key)

    async def ask_openai(self, query: str, model: str = "gpt-4", temperature: float = 0.7, max_tokens: int = 500) -> str:
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to query OpenAI: {str(e)}")
            raise

    async def generate_image(
        self, 
        prompt: str, 
        model: str = "dall-e-3", 
        size: str = "1024x1024", 
        quality: str = "standard",
        n: int = 1
    ) -> str:
        """
        Generate images using OpenAI DALL-E models.
        
        Args:
            prompt: Text description for image generation
            model: DALL-E model version ("dall-e-2" or "dall-e-3")
            size: Image dimensions
            quality: Image quality ("standard" or "hd" for DALL-E 3)
            n: Number of images to generate
            
        Returns:
            Image URL(s) as a string
            
        Raises:
            ValueError: For invalid parameter combinations
            Exception: For API errors
        """
        # Validate parameters
        self._validate_image_parameters(prompt, model, size, quality, n)
        
        try:
            # Prepare API call parameters
            api_params = {
                "prompt": prompt,
                "model": model,
                "size": size,
                "n": n
            }
            
            # Add quality parameter only for DALL-E 3
            if model == "dall-e-3":
                api_params["quality"] = quality
            
            response = await self.client.images.generate(**api_params)
            
            # Extract URLs from response
            if len(response.data) == 1:
                return response.data[0].url
            else:
                # Multiple images - return all URLs
                urls = [img.url for img in response.data]
                return "\n".join(urls)
                
        except Exception as e:
            logger.error(f"Failed to generate image: {str(e)}")
            raise

    def _validate_image_parameters(self, prompt: str, model: str, size: str, quality: str, n: int) -> None:
        """Validate image generation parameters for DALL-E model compatibility."""
        
        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Validate prompt length based on model
        if model == "dall-e-3" and len(prompt) > 4000:
            raise ValueError("Prompt too long for DALL-E 3 (max 4000 characters)")
        elif model == "dall-e-2" and len(prompt) > 1000:
            raise ValueError("Prompt too long for DALL-E 2 (max 1000 characters)")
        
        # Validate model
        valid_models = ["dall-e-2", "dall-e-3"]
        if model not in valid_models:
            raise ValueError(f"Invalid model '{model}'. Must be one of: {valid_models}")
        
        # Validate size based on model
        if model == "dall-e-2":
            valid_sizes = ["256x256", "512x512", "1024x1024"]
        else:  # dall-e-3
            valid_sizes = ["1024x1024", "1024x1792", "1792x1024"]
        
        if size not in valid_sizes:
            raise ValueError(f"Invalid size '{size}' for model '{model}'. Valid sizes: {valid_sizes}")
        
        # Validate quality (only applicable to DALL-E 3)
        if model == "dall-e-3":
            valid_qualities = ["standard", "hd"]
            if quality not in valid_qualities:
                raise ValueError(f"Invalid quality '{quality}' for DALL-E 3. Must be one of: {valid_qualities}")
        elif quality != "standard":
            # Quality parameter is ignored for DALL-E 2, but warn if non-default is provided
            logger.warning(f"Quality parameter '{quality}' is ignored for DALL-E 2")
        
        # Validate number of images
        if model == "dall-e-3" and n != 1:
            raise ValueError("DALL-E 3 only supports generating 1 image (n=1)")
        elif model == "dall-e-2" and (n < 1 or n > 10):
            raise ValueError("DALL-E 2 supports generating 1-10 images")
        
        if not isinstance(n, int) or n < 1:
            raise ValueError("Number of images (n) must be a positive integer")