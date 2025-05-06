import adalflow_testing_network as adal
import re

# Initialize the model
llama_llm = adal.Generator(
    model_client=adal.OllamaClient(),
    model_kwargs={"model": "qwen3:1.7b"}
)

# Get the response object
response = llama_llm(prompt_kwargs={"input_str": "What is LLM?"})

# Clean the .data attribute directly
cleaned_response = re.sub(r"<think>.*?</think>", "", response.data, flags=re.DOTALL)

# Print the cleaned output
print(cleaned_response.strip())