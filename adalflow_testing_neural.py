from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
from datasets import load_dataset
from adalflow.components.model_client import OpenAIClient
import adalflow as adal
from adalflow.core.component import Component
from adalflow.datasets.types import TrecData
from adalflow.datasets.trec import TrecDataset
from adalflow.core.base_data_class import DataClass
import uuid

from adalflow.eval.answer_match_acc import AnswerMatchAcc
import re

def remove_think_tags(output: str) -> str:
    # Remove everything between <think> and </think> including the tags themselves
    cleaned_output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
    return cleaned_output.strip()


_COARSE_LABELS = ["deal_won", "deal_lost"]

_COARSE_LABELS_DESC = [
    "deal_won: Deal is won based on the information provided",
    "deal_lost: Deal is lost based on the infomration provided"
]


template = r"""<START_OF_SYSTEM_MESSAGE>
 {{system_prompt}}
 {% if output_format_str is not none %}
 {{output_format_str}}
 {% endif %}
 {% if few_shot_demos is not none %}
 Here are some examples:
 {{few_shot_demos}}
 {% endif %}
 <END_OF_SYSTEM_MESSAGE>
 <START_OF_USER_MESSAGE>
 {{input_str}}
 <END_OF_USER_MESSAGE>
 """

task_desc_template = r"""You are a classifier. Given the company name, you need to classify it into one of the following classes:
 Format: class_index. class_name, class_description
 {% if classes %}
 {% for class in classes %}
 {{loop.index-1}}. {{class.label}}, {{class.desc}}
 {% endfor %}
 {% endif %}
You can reason whether it is a well known company or not, and then give your output.
 """

@dataclass
class BaseData(DataClass):
    __doc__ = """A common dataclass for representing examples in a dataset."""
    id: str = field(
        metadata={"desc": "The unique identifier of the example", "type": "id"},
        default=str(uuid.uuid4()),
    )

@dataclass
class TrecData(BaseData):
    __doc__ = """A dataclass for representing examples in the TREC dataset."""
    question: str = field(
        metadata={"desc": "The name is to be classified"},
        default=None,
    )
    class_name: str = field(
        metadata={"desc": "One of {deal_won, deal_lost}"},
        default=None,
    )
    class_index: int = field(
        metadata={"desc": "The class label, in range [0, 1]"},
        default=-1,
    )

    __input_fields__ = ["question"]  # follow this order too.
    __output_fields__ = ["class_name", "class_index"]

@dataclass
class TRECExtendedData(TrecData):
    rationale: str = field(
        metadata={
            "desc": "Your step-by-step reasoning to classify the text to class_name"
        },
        default=None,
    )
    __input_fields__ = ["question"]
    __output_fields__ = [
        "rationale",
        "class_name",
    ]  # it is important to have the rationale before the class_name


import pandas as pd
from sklearn.model_selection import train_test_split
from adalflow.datasets.types import TrecData

def load_datasets(csv_path="synthetic_hubspot_df_basic.csv"):
    """Load custom dataset from CSV and split into train/val/test"""
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    if not {"question", "class_name"}.issubset(df.columns):
        raise ValueError("CSV must contain 'question' and 'class_name' columns")

    # Convert to TrecData format
    examples = [TrecData(id=str(i), question=row["question"], class_name=row["class_name"]) for i, row in df.iterrows()]

    train, temp = train_test_split(examples, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    return train, val, test

# prepare models

from adalflow.components.model_client.ollama_client import OllamaClient

# used as the target model
qwen_model = {
    "model_client": OllamaClient(),
    "model_kwargs": {
        "model": "gemma3:1b",
    },
}

class TRECClassifierStructuredOutput(adal.Component):

    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        label_desc = [
            {"label": label, "desc": desc}
            for label, desc in zip(_COARSE_LABELS, _COARSE_LABELS_DESC)
        ]

        task_desc_str = adal.Prompt(
            template=task_desc_template, prompt_kwargs={"classes": label_desc}
        )()

        self.data_class = TRECExtendedData
        self.data_class.set_task_desc(task_desc_str)

        self.parser = adal.DataClassParser(
            data_class=self.data_class, return_data_class=True, format_type="yaml"
        )

        prompt_kwargs = {
            "system_prompt": adal.Parameter(
                data=self.parser.get_task_desc_str(),
                role_desc="Task description",
                requires_opt=True,
                param_type=adal.ParameterType.PROMPT,
            ),
            "output_format_str": adal.Parameter(
                data=self.parser.get_output_format_str(),
                role_desc="Output format requirements",
                requires_opt=False,
                param_type=adal.ParameterType.PROMPT,
            ),
            "few_shot_demos": adal.Parameter(
                data=None,
                requires_opt=True,
                role_desc="Few shot examples to help the model",
                param_type=adal.ParameterType.DEMOS,
            ),
        }

        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
            template=template,
            output_processors=self.parser,
            use_cache=True,
        )

    def _prepare_input(self, question: str):
        input_data = self.data_class(question=question)
        input_str = self.parser.get_input_str(input_data)
        
        prompt_kwargs = {
            "input_str": adal.Parameter(
                data=input_str, requires_opt=False, role_desc="input to the LLM"
            )
        }
        return prompt_kwargs

    def bicall(
        self, question: str, id: Optional[str] = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        prompt_kwargs = self._prepare_input(question)
        output = self.llm(prompt_kwargs=prompt_kwargs, id=id)
        return output
    # def bicall(self, question: str, id: Optional[str] = None) -> Union[adal.GeneratorOutput, adal.Parameter]:
    #     prompt_kwargs = self._prepare_input(question)
    #     output = self.llm(prompt_kwargs=prompt_kwargs, id=id)
        
    #     # Assuming output.data.text contains the raw output text, apply the clean-up step
    #     raw_output_text = output.data  # Adjust this based on the actual structure of output.data
    #     cleaned_output = remove_think_tags(raw_output_text)  # Clean the raw output text

    #     # Update the output with cleaned data
    #     output.data.text = cleaned_output  # Update the cleaned text in the output

    #     return output
    
    # load dataset to get one example

train_dataset, val_dataset, test_dataset = load_datasets()
example = train_dataset[0]
print(example)

task = TRECClassifierStructuredOutput(
    model_client=qwen_model["model_client"],
    model_kwargs=qwen_model["model_kwargs"],
)
task.train()

output = task(question=example.question, id=example.id)
print(output)

class TrecClassifierAdal(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        teacher_model_config: Dict,
        backward_engine_model_config: Dict,
        text_optimizer_model_config: Dict,
    ):
        task = TRECClassifierStructuredOutput(model_client, model_kwargs)
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0. When the LLM prediction failed with format parsing which results with errors, we set y_pred = -1",
        )
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
            teacher_model_config=teacher_model_config,
        )

    def prepare_task(self, sample: TRECExtendedData):
        return self.task.call, {"question": sample.question, "id": sample.id}

    def prepare_eval(
        self, sample: TRECExtendedData, y_pred: adal.GeneratorOutput
    ) -> float:
        y_label = -1
        if y_pred and y_pred.data is not None and y_pred.data.class_name is not None:
            y_label = y_pred.data.class_name
        return self.eval_fn, {"y": y_label, "y_gt": sample.class_name}
    # def prepare_eval(self, sample: TRECExtendedData, y_pred: adal.GeneratorOutput) -> float:
    # # Assuming y_pred.data.text contains the raw output text, apply the clean-up step
    #     raw_output_text = y_pred.data  # Adjust this based on the actual structure of y_pred.data
        
    #     # Clean the raw output text before evaluation
    #     cleaned_output = remove_think_tags(raw_output_text)
        
    #     # Now proceed with the evaluation using the cleaned output
    #     y_label = -1
    #     if cleaned_output and cleaned_output.class_name is not None:
    #         y_label = cleaned_output.class_name

    #     return self.eval_fn, {"y": y_label, "y_gt": sample.class_name}

    def prepare_loss(
        self, sample: TRECExtendedData, y_pred: adal.Parameter, *args, **kwargs
    ) -> Tuple[Callable[..., Any], Dict]:
        full_response = y_pred.data
        y_label = -1  # default value for failed prediction
        if (
            full_response
            and full_response.data is not None
            and full_response.data.class_name is not None
        ):
            y_label = full_response.data.class_name

        y_pred.eval_input = y_label
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.class_name,
            eval_input=sample.class_name,
            requires_opt=False,
        )
        return self.loss_fn, {
            "kwargs": {"y": y_pred, "y_gt": y_gt},
            "id": sample.id,
            "gt": y_gt.eval_input,
        }
    

def train(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
    train_batch_size=4,
    raw_shots: int = 0,
    bootstrap_shots: int = 1,
    max_steps=12,
    num_workers=4,
    strategy="constrained",
    optimization_order="sequential",
    debug=False,
):
    print("Starting training process...")

    # Define the model configuration for all components
    qwen_model = {
        "model_client": OllamaClient(),
        "model_kwargs": {
            "model": "gemma3:1b",
        },
    }

    print(f"Component model configuration: {qwen_model}")

    try:
        print("Initializing ADAL component...")
        adal_component = TrecClassifierAdal(
            model_client=model_client,
            model_kwargs=model_kwargs,
            text_optimizer_model_config=qwen_model,
            backward_engine_model_config=qwen_model,
            teacher_model_config=qwen_model,
        )
        print("ADAL component initialized successfully")

        print("Initializing trainer...")
        trainer = adal.Trainer(
            train_batch_size=train_batch_size,
            adaltask=adal_component,
            strategy=strategy,
            max_steps=max_steps,
            num_workers=num_workers,
            raw_shots=raw_shots,
            bootstrap_shots=bootstrap_shots,
            debug=debug,
            weighted_sampling=True,
            optimization_order=optimization_order,
            exclude_input_fields_from_bootstrap_demos=True,
        )
        print("Trainer initialized successfully")

        print("Loading datasets...")
        train_dataset, val_dataset, test_dataset = load_datasets()
        print(
            f"Datasets loaded - Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}"
        )

        print("Starting model training...")
        trainer.fit(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            debug=debug,
        )
        print("Training completed successfully")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

train(**qwen_model)