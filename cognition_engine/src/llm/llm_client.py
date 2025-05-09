import sys
import os
from typing import Any, Dict, Optional, Mapping, List

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # From src/llm back to cognition_engine
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# from langchain_community.llms import Ollama # Old import
from langchain_ollama import OllamaLLM as LangchainOllama # New import, aliased to avoid clash if needed
from langchain_core.callbacks import CallbackManager # Updated import for CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # Updated import

from src.config import settings

class OllamaLLMClient:
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        temperature: float = 0.7, # Default temperature, can be overridden
        **kwargs: Any # Other OllamaLLM parameters like top_k, top_p, etc.
    ):
        """
        Initializes the Ollama LLM client using LangChain.

        Args:
            model_name: The name of the Ollama model to use (e.g., 'mistral', 'llama2').
                        Defaults to OLLAMA_MODEL from settings.
            base_url: The base URL for the Ollama API.
                      Defaults to OLLAMA_BASE_URL from settings.
            callback_manager: Optional LangChain callback manager.
            temperature: The temperature for the LLM. Defaults to 0.7.
            **kwargs: Additional keyword arguments to pass to the Ollama LLM constructor.
        """
        self.model_name = model_name or settings.OLLAMA_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.temperature = temperature
        
        if callback_manager is None:
            # Default to streaming stdout if no callback manager is provided, for interactive use.
            # For production/API use, you might want None or a custom one.
            # self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            self.callback_manager = None # No streaming by default for API use
        else:
            self.callback_manager = callback_manager

        # Combine specific params with general kwargs for OllamaLLM constructor
        llm_params = {
            "model": self.model_name,
            "base_url": self.base_url,
            "callback_manager": self.callback_manager,
            "temperature": self.temperature,
            **kwargs
        }

        try:
            self.llm = LangchainOllama(**llm_params)
            print(f"OllamaLLMClient initialized with model: {self.model_name} at {self.base_url}, temp: {self.temperature}")
        except Exception as e:
            print(f"Error initializing Ollama LLM: {e}")
            print("Please ensure Ollama is running and the model is available.")
            print(f"Attempted model: {self.model_name}, Base URL: {self.base_url}")
            self.llm = None # Set llm to None if initialization fails

    def generate_text(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> Optional[str]:
        """
        Generates text based on the given prompt using the configured Ollama LLM.

        Args:
            prompt: The input prompt for the LLM.
            stop: Optional list of stop sequences.
            **kwargs: Additional keyword arguments for the LLM generation (e.g., temperature, top_k).

        Returns:
            The generated text as a string, or None if LLM initialization failed or generation error.
        """
        if self.llm is None:
            print("LLM not initialized. Cannot generate text.")
            return None
        
        try:
            print(f"Sending prompt to Ollama model '{self.model_name}': '{prompt[:100]}...'")
            # Pass only invoke-specific arguments here. `stop` is a common one.
            invoke_kwargs = {}
            if stop:
                invoke_kwargs['stop'] = stop
            invoke_kwargs.update(kwargs) # For any other future invoke-specific args
            
            response = self.llm.invoke(prompt, **invoke_kwargs)
            print(f"Received response from Ollama: '{response[:100]}...'")
            return response
        except Exception as e:
            print(f"Error during LLM text generation: {e}")
            return None

# Example Usage (for direct testing of this module)
if __name__ == "__main__":
    print("Testing OllamaLLMClient...")
    
    # Ensure Ollama is running with the default model (e.g., 'mistral') or specify one.
    # You can override model and base_url from .env or directly here for testing.
    # For example, to test with a different model:
    # client = OllamaLLMClient(model_name="llama2")
    client = OllamaLLMClient()

    if client.llm:
        test_prompt = "Explain the concept of Retrieval Augmented Generation in one sentence."
        generated_response = client.generate_text(test_prompt)
        
        if generated_response:
            print(f"\nTest Prompt: {test_prompt}")
            print(f"Generated Response:\n{generated_response}")
        else:
            print("Failed to get a response from the LLM.")
    else:
        print("LLM client could not be initialized. Skipping generation test.")

    print("\nTesting with a non-existent model (should show error during init or first use)...")
    # Error for non-existent model might show on init or first use depending on Ollama/Langchain version
    non_existent_client = OllamaLLMClient(model_name="nonexistentmodel123")
    if non_existent_client.llm:
        non_existent_client.generate_text("test") # Attempt to use it
    if non_existent_client.llm is None or not generated_response: # Check if it failed
        print("Confirmed that client with non-existent model did not generate a response as expected.") 