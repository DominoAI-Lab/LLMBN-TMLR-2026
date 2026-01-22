"""
Unified interface for LLM providers (OpenAI, DeepSeek, Gemini).

Provides a consistent API for LLM interactions in LLMBN, supporting both
data-free generation and structure refinement. API keys are loaded from
environment variables or .env file.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
import logging

class LLMClient:
    """
    Base class for LLM provider clients with unified interface.
    
    Attributes:
        model: Name of the LLM model to use
        client: Provider-specific client instance
        provider: Name of the LLM provider
        logger: Logger instance for output
    """
    
    def __init__(
        self,
        model: str,
        logger: logging.Logger = None,
    ) -> None:
        """
        Initialize LLM client with model and logger.
        
        Args:
            model: Name of the LLM model to use
            logger: Logger instance for output
        """
        # Load API keys from .env file if not already in environment variables
        # load_dotenv() does NOT override existing environment variables
        load_dotenv()
        self.model = model
        self.client = None
        self.provider = None
        self.logger = logger
        
    def chat(self, messages: list[dict]) -> str:
        """
        Send messages to LLM and get text response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            str: Text response from the LLM
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement .chat()")

    def chat_structured(self, messages: list[dict], response_format: dict = None) -> str:
        """
        Send messages to LLM and get structured response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            response_format: Dictionary specifying the expected response format
            
        Returns:
            str: Structured response from the LLM
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement .chat_structured()")

    @staticmethod
    def from_model(
        model: str,
        logger: logging.Logger = None,
    ) -> 'LLMClient':
        """
        Factory method to create appropriate LLM client based on model name.
        
        Args:
            model: Name of the LLM model to use
            logger: Logger instance for output
            
        Returns:
            LLMClient: Appropriate client instance for the specified model
            
        Raises:
            ValueError: If model is unknown or unsupported
        """
        model_lower = model.lower()
        if "deepseek" in model_lower:
            return DeepseekModel(model, logger)
        elif model_lower.startswith("o"):
            return OAIReasonModel(model, logger)
        elif model_lower.startswith("gpt"):
            return OAIChatModel(model, logger)
        elif model_lower.startswith("gemini"):
            return GeminiModel(model, logger)
        else:
            raise ValueError(f"Unknown model: {model}")

class DeepseekModel(LLMClient):
    """
    Client for DeepSeek LLM API integration.
    
    Attributes:
        model: Name of the DeepSeek model to use
        client: DeepSeek API client instance (using OpenAI client with custom base URL)
        provider: Provider name ('deepseek')
        logger: Logger instance for output
    """
    
    def __init__(
        self,
        model: str,
        logger: logging.Logger = None,
    ) -> None:
        """
        Initialize DeepSeek client with model and logger.
        
        Args:
            model: Name of the DeepSeek model to use
            logger: Logger instance for output
            
        Raises:
            ValueError: If DEEPSEEK_API_KEY is not found
        """
        super().__init__(model, logger)
        # Check for API key (already loaded from .env by parent __init__)
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found. Please set it as an environment variable "
                "or add it to your .env file."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.provider = "deepseek"

    def chat(self, messages: list[dict]) -> str:
        """
        Send messages to DeepSeek API and get text response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            str: Text response from the DeepSeek model
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content
        
    def chat_structured(
        self,
        messages: list[dict],
        response_format: dict = None,
    ) -> str:
        """
        Not supported for DeepSeek models.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            response_format: Dictionary specifying the expected response format
            
        Raises:
            NotImplementedError: Structured output not supported for DeepSeek models
        """
        if self.logger:
            self.logger.error(f"Structured Output is not supported for {self.model}")
        raise NotImplementedError(f"Structured Output is not supported for {self.model}")

class OAIReasonModel(LLMClient):
    """
    Client for OpenAI reasoning models (o1, o2, etc.).
    
    Attributes:
        model: Name of the OpenAI reasoning model to use
        client: OpenAI API client instance
        provider: Provider name ('openai')
        logger: Logger instance for output
    """
    
    def __init__(
        self,
        model: str,
        logger: logging.Logger = None,
    ) -> None:
        """
        Initialize OpenAI reasoning model client.
        
        Args:
            model: Name of the OpenAI reasoning model to use
            logger: Logger instance for output
            
        Raises:
            ValueError: If OPENAI_API_KEY is not found
        """
        super().__init__(model, logger)
        # Check for API key (already loaded from .env by parent __init__)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it as an environment variable "
                "or add it to your .env file."
            )
        self.client = OpenAI(api_key=api_key)
        self.provider = "openai"

    def chat(self, messages: list[dict]) -> str:
        """
        Send messages to OpenAI reasoning model and get text response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            str: Text response from the OpenAI reasoning model
        """
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "medium"},
            input=messages
        )
        return response.output_text

    def chat_structured(
        self,
        messages: list[dict],
        response_format: dict = None,
    ) -> str:
        """
        Send messages to OpenAI reasoning model and get structured response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            response_format: Dictionary specifying the expected response format
            
        Returns:
            str: Structured response from the OpenAI reasoning model
        """
        # Use chat.completions.create with response_format instead of beta API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=response_format
        )
        return response.choices[0].message.content

class OAIChatModel(LLMClient):
    """
    Client for OpenAI chat models (GPT series).
    
    Attributes:
        model: Name of the OpenAI chat model to use
        client: OpenAI API client instance
        provider: Provider name ('openai')
        logger: Logger instance for output
    """
    
    def __init__(
        self,
        model: str,
        logger: logging.Logger = None,
    ) -> None:
        """
        Initialize OpenAI chat model client.
        
        Args:
            model: Name of the OpenAI chat model to use
            logger: Logger instance for output
            
        Raises:
            ValueError: If OPENAI_API_KEY is not found
        """
        super().__init__(model, logger)
        # Check for API key (already loaded from .env by parent __init__)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it as an environment variable "
                "or add it to your .env file."
            )
        self.client = OpenAI(api_key=api_key)
        self.provider = "openai"

    def chat(self, messages: list[dict]) -> str:
        """
        Send messages to OpenAI chat model and get text response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            str: Text response from the OpenAI chat model
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content
    
    def chat_structured(
        self,
        messages: list[dict],
        response_format: dict = None,
    ) -> str:
        """
        Not supported for standard OpenAI chat models.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            response_format: Dictionary specifying the expected response format
            
        Raises:
            NotImplementedError: Structured output not supported for this model
        """
        self.logger.error(f"Structured Output is not supported for {self.model}")
        raise NotImplementedError(f"Structured Output is not supported for {self.model}")


class GeminiModel(LLMClient):
    """
    Client for Google Gemini models.
    
    Attributes:
        model: Name of the Gemini model to use
        client: Not used directly (uses genai module)
        provider: Provider name ('google')
        logger: Logger instance for output
    """
    
    def __init__(
        self,
        model: str,
        logger: logging.Logger = None,
    ) -> None:
        """
        Initialize Gemini model client.
        
        Args:
            model: Name of the Gemini model to use
            logger: Logger instance for output
            
        Raises:
            ValueError: If GEMINI_API_KEY is not found
        """
        super().__init__(model, logger)
        # Check for API key (already loaded from .env by parent __init__)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it as an environment variable "
                "or add it to your .env file."
            )
        
        genai.configure(api_key=api_key)
        self.provider = "google"
        # The client is just a reference to the `genai` module or can be kept as None
        # since we will create the model instance within the chat method.
        # This makes the class more flexible and stateless per chat call.
        self.client = None

    def _convert_openai_messages_to_gemini_contents(
        self,
        messages: list[dict],
    ) -> tuple[str, list[dict]]:
        """
        Convert OpenAI-style messages to Gemini-style contents.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            tuple: (system_instruction, contents) where:
                - system_instruction: String with system prompt (or None)
                - contents: List of Gemini-formatted message dictionaries
        """
        system_instruction = None
        contents = []

        if messages and messages[0]['role'] == 'system':
            system_instruction = messages[0]['content']
            conversation_messages = messages[1:]
        else:
            conversation_messages = messages
        
        for msg in conversation_messages:
            role = "user" if msg['role'] == 'user' else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg['content']}]
            })
            
        return system_instruction, contents

    def chat(
        self,
        messages: list[dict],
        *args,
        **kwargs,
    ) -> str:
        """
        Send messages to Gemini API and get text response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            *args: Additional positional arguments for the Gemini API
            **kwargs: Additional keyword arguments for the Gemini API
            
        Returns:
            str: Text response from the Gemini model
        """
        system_instruction, contents = self._convert_openai_messages_to_gemini_contents(messages)
        
        # Instantiate the model with the system instruction for this specific chat call.
        # This is the crucial fix. The GenerativeModel instance is created for each call
        # if a system instruction is present.
        if system_instruction:
            model_instance = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_instruction
            )
        else:
            model_instance = genai.GenerativeModel(model_name=self.model)

        response = model_instance.generate_content(
            contents=contents,
            *args,
            **kwargs,
        )
        
        return response.text

    def chat_structured(
        self,
        messages: list[dict],
        response_format: dict = None,
    ) -> str:
        """
        Not implemented for Gemini models.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            response_format: Dictionary specifying the expected response format
            
        Raises:
            NotImplementedError: Structured output not implemented for Gemini
        """
        raise NotImplementedError(
            f"Structured chat for Gemini requires function calling, which is not implemented for {self.model}"
        )