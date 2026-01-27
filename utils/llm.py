"""LLM utilities for ResearchGym - supporting multiple providers including Gemini."""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Response from LLM call."""
    content: str
    token_usage: Dict[str, int]
    model: str
    
class LLMError(Exception):
    """Exception raised for LLM-related errors."""
    pass

class LLMInterface:
    """Interface for calling various LLM providers."""
    
    def __init__(self):
        self.openai_client = None
        self.azure_client = None
        self.litellm_available = False
        self._setup_clients()
    
    def _setup_clients(self):
        """Setup LLM clients based on available environment variables."""
        
        # Setup OpenAI client
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI package not available")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Setup Azure OpenAI client
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        if azure_endpoint and azure_api_key:
            try:
                import openai
                self.azure_client = openai.AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_api_key,
                    api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
                )
                logger.info("Azure OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI package not available for Azure")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {e}")
        
        # Setup LiteLLM
        try:
            import litellm
            # Suppress verbose LiteLLM logging
            litellm.suppress_debug_info = True
            litellm.set_verbose = False
            self.litellm_available = True
            logger.info("LiteLLM available")
        except ImportError:
            logger.warning("LiteLLM package not available")
    
    def complete_text(
        self, 
        prompt: str, 
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
        max_tokens: int = 4000,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete text using the specified model."""
        
        # Determine which client to use based on model
        if self.azure_client:
            return self._call_azure_openai(prompt, model, temperature, max_tokens, stop_sequences, **kwargs)
        elif self._is_openai_model(model) and self.openai_client:
            return self._call_openai(prompt, model, temperature, max_tokens, stop_sequences, **kwargs)
        elif self.litellm_available:
            return self._call_litellm(prompt, model, temperature, max_tokens, stop_sequences, **kwargs)
        else:
            raise LLMError(f"No available client for model: {model}")
    
    def _is_openai_model(self, model: str) -> bool:
        """Check if model is an OpenAI model."""
        openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-5", "o1", "o1-mini"]
        return any(model.startswith(prefix) for prefix in openai_models)
    
    def _call_openai(
        self, 
        prompt: str, 
        model: str, 
        temperature: float, 
        max_tokens: int, 
        stop_sequences: Optional[List[str]], 
        **kwargs
    ) -> LLMResponse:
        """Call OpenAI API."""
        try:
            # Handle o1 models differently (they don't support temperature/stop)
            if "o1" in model.lower():
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens,
                    **kwargs
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop_sequences,
                    **kwargs
                )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=model
            )
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise LLMError(f"OpenAI API call failed: {e}")
    
    def _call_azure_openai(
        self, 
        prompt: str, 
        model: str, 
        temperature: float, 
        max_tokens: int, 
        stop_sequences: Optional[List[str]], 
        **kwargs
    ) -> LLMResponse:
        """Call Azure OpenAI API."""
        try:
            # Remove azure/ prefix for deployment name if present
            deployment_name = model.replace("azure/", "")
            
            # Handle o1 models differently (they don't support temperature/stop)
            if "o1" in model.lower():
                response = self.azure_client.chat.completions.create(
                    model=deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens,
                    **kwargs
                )
            else:
                response = self.azure_client.chat.completions.create(
                    model=deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop_sequences,
                    **kwargs
                )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=model
            )
            
        except Exception as e:
            logger.error(f"Azure OpenAI API call failed: {e}")
            raise LLMError(f"Azure OpenAI API call failed: {e}")
    
    def _call_litellm(
        self, 
        prompt: str, 
        model: str, 
        temperature: float, 
        max_tokens: int, 
        stop_sequences: Optional[List[str]], 
        **kwargs
    ) -> LLMResponse:
        """Call LiteLLM (supports many providers including Gemini)."""
        try:
            import litellm
            
            # Ensure logging is suppressed
            litellm.suppress_debug_info = True
            litellm.set_verbose = False
            
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=model
            )
            
        except Exception as e:
            logger.error(f"LiteLLM API call failed: {e}")
            raise LLMError(f"LiteLLM API call failed: {e}")
    
    def complete_with_messages(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        temperature: float = 0.7, 
        max_tokens: int = 4000, 
        tools: Optional[List[Dict]] = None,
        **kwargs
    ):
        """Complete text using messages format with optional tools."""
        if self.azure_client:
            return self._call_azure_openai_messages(messages, model, temperature, max_tokens, tools, **kwargs)
        elif self._is_openai_model(model) and self.openai_client:
            return self._call_openai_messages(messages, model, temperature, max_tokens, tools, **kwargs)
        elif self.litellm_available:
            return self._call_litellm_messages(messages, model, temperature, max_tokens, tools, **kwargs)
        else:
            raise LLMError(f"No available client for model: {model}")
    
    def _call_openai_messages(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        temperature: float, 
        max_tokens: int, 
        tools: Optional[List[Dict]] = None,
        **kwargs
    ):
        """Call OpenAI API with messages format."""
        try:
            call_kwargs = {
                "model": model,
                "messages": messages,
                **kwargs
            }
            
            # Handle o1 models differently
            if "o1" in model.lower():
                call_kwargs["max_completion_tokens"] = max_tokens
            else:
                call_kwargs["temperature"] = temperature
                call_kwargs["max_tokens"] = max_tokens
                
            if tools:
                call_kwargs["tools"] = tools
            
            response = self.openai_client.chat.completions.create(**call_kwargs)
            return response
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise LLMError(f"OpenAI API error: {e}")
    
    def _call_azure_openai_messages(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        temperature: float, 
        max_tokens: int, 
        tools: Optional[List[Dict]] = None,
        **kwargs
    ):
        """Call Azure OpenAI API with messages format."""
        try:
            deployment_name = model.replace("azure/", "")
            
            call_kwargs = {
                "model": deployment_name,
                "messages": messages,
                **kwargs
            }
            
            # Handle o1 models differently
            if "o1" in model.lower():
                call_kwargs["max_completion_tokens"] = max_tokens
            else:
                call_kwargs["temperature"] = temperature
                call_kwargs["max_tokens"] = max_tokens
            
            if tools:
                call_kwargs["tools"] = tools
            
            response = self.azure_client.chat.completions.create(**call_kwargs)
            return response
            
        except Exception as e:
            logger.error(f"Azure OpenAI API call failed: {e}")
            raise LLMError(f"Azure OpenAI API error: {e}")
    
    def _call_litellm_messages(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        temperature: float, 
        max_tokens: int,
        tools: Optional[List[Dict]] = None, 
        **kwargs
    ):
        """Call LiteLLM with messages format."""
        try:
            import litellm
            
            # Ensure logging is suppressed
            litellm.suppress_debug_info = True
            litellm.set_verbose = False
            
            call_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            if tools:
                call_kwargs["tools"] = tools
            
            response = litellm.completion(**call_kwargs)
            return response
            
        except Exception as e:
            logger.error(f"LiteLLM API call failed: {e}")
            raise LLMError(f"LiteLLM API error: {e}")

# Global LLM interface instance
_llm_interface = None

def get_llm_interface() -> LLMInterface:
    """Get the global LLM interface instance."""
    global _llm_interface
    if _llm_interface is None:
        _llm_interface = LLMInterface()
    return _llm_interface

def complete_text(
    prompt: Union[str, List[Dict[str, str]]], 
    model: str = None, 
    temperature: float = 0.5,
    max_tokens: int = 4000,
    tools: Optional[List[Dict]] = None, 
    **kwargs
):
    """Convenience function to complete text with optional tools support."""
    
    # Set default model if not provided
    if model is None:
        # Check if Azure OpenAI is configured
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        if azure_endpoint and azure_api_key:
            model = os.getenv("DEFAULT_MODEL", "gpt-5")
        else:
            model = os.getenv("DEFAULT_MODEL", "gemini/gemini-2.5-flash-lite")
    
    # Handle different prompt formats
    if isinstance(prompt, str):
        # Convert string to messages format
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        # Already in messages format
        messages = prompt
    else:
        raise ValueError("Prompt must be string or list of messages")
    
    # Add tools if provided
    call_kwargs = kwargs.copy()
    call_kwargs.update({
        "temperature": temperature,
        "max_tokens": max_tokens
    })
    if tools:
        call_kwargs['tools'] = tools
    
    # Use the interface with messages
    response = get_llm_interface().complete_with_messages(messages, model, **call_kwargs)
    
    # Handle tool calls in response
    if hasattr(response, 'choices') and response.choices:
        choice = response.choices[0]
        if hasattr(choice, 'message'):
            return choice.message
    
    # Fallback to content only
    return response.content if hasattr(response, 'content') else str(response)

