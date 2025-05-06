# import sys
# import os
# sys.path.insert(0, os.path.abspath("AdalFlow/adalflow/"))  # or the relative path to the root of the repo

from adalflow import Prompt, Generator, ModelClient, AdalComponent, Parameter
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow import EvalFnToTextLoss
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from adalflow import Trainer
import pandas as pd
from sklearn.model_selection import train_test_split

import re

prompt1 = Prompt(template=" Based on the given text, break it down into multiple words, and also write multiple meanings to it. {{ input_text }}")
prompt2 = Prompt(template="What does this input mean, and what are the likely usecases of the deal. {{ analysis }}")
output_prompt = Prompt(template="Based on this information, give the output as ‘deal_won’ if the Deal is won, or if the Deal is lost return ‘deal_lost’.  {{ prompt2 }}")

# Initialize the model client (e.g., OpenAI's GPT)
model_client = OllamaClient() 
text_optimizer_model_config = {
    "model": "qwen3:1.7b",  # or whatever model you're using
    "model_kwargs": {
        "temperature": 0.7,
        "top_p": 0.95,
    }
}
class DealClassifier(AdalComponent):
    def __init__(self, model_client: ModelClient):
        super().__init__(task=model_client)
        self.prompt1 = prompt1
        self.prompt2 = prompt2
        self.output_prompt = output_prompt
        self.model_client = model_client
        # ✅ Wrap prompts in Parameters
        self.register_parameter("prompt1", Parameter(name="prompt1", data=self.prompt1))
        self.register_parameter("prompt2", Parameter(name="prompt2", data=self.prompt2))
        self.register_parameter("output_prompt", Parameter(name="output_prompt", data=self.output_prompt))

    def remove_think_block(text):
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

    def forward(self, input_text):
        analysis = self.remove_think_block(self.prompt1(input_text=input_text))
        key_factors = self.remove_think_block(self.prompt2(analysis=analysis))
        final_output = self.remove_think_block(self.output_prompt(prompt2=key_factors))
        return final_output
def remove_think_block(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
loss_fn = EvalFnToTextLoss(
    eval_fn=eval_fn,
    eval_fn_desc="exact_match: 1 if prediction == ground_truth else 0"
)
df = pd.read_csv("synthetic_hubspot_df_basic.csv", usecols=["Deal", "Deal_status"])

# Structure the dataset for training
full_dataset = [{"input": row["Deal"], "label": row["Deal_status"]} for _, row in df.iterrows()]
train_dataset, val_dataset = train_test_split(full_dataset, test_size=0.4, random_state=42)

adal_component = DealClassifier(
        model_client=model_client,
        # model_kwargs={"model":"qwen3:1.7b"},
        text_optimizer_model_config=text_optimizer_model_config,
        # backward_engine_model_config=text_optimizer_model_config,
        # teacher_model_config=text_optimizer_model_config,
    )
trainer = Trainer(
    adaltask=adal_component,
    train_batch_size=4,  # larger batch size is not that effective, probably because of llm's lost in the middle
    max_steps=12,
    num_workers=4,
    strategy="constrained",
    optimization_order="sequential",
    debug=True  # If you want more insights
)
# Load the dataset
trainer.fit(train_dataset=train_dataset,val_dataset=val_dataset, debug=True)

# # Define the generator with the first prompt
# generator = Generator(
#     model_kwargs={"model":"qwen3:1.7b"},
#     model_client=model_client
# )
# generator.prompt = prompt1
# # Step 1: Analyze input
# analysis = generator(prompt_kwargs={"input_text": "Who ruled india for 50 years"}).data
# analysis = remove_think_block(analysis)
# print("analysis1 done")

# # Step 2: Identify key factors
# generator.prompt = prompt2
# key_factors = generator(prompt_kwargs={"analysis": analysis}).data
# print(key_factors)
# # Step 4: Generate final output
# generator.prompt = output_prompt
# final_output = generator(prompt_kwargs={"summary": key_factors}).data
# print(final_output)