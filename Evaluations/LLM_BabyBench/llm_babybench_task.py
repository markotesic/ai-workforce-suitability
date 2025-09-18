import os
from pathlib import Path
from typing import Any, Dict
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, FieldSpec, Sample, hf_dataset
from inspect_ai.scorer import choice, includes, match
from inspect_ai.solver import Choices, basic_agent, multiple_choice
from inspect_ai._util.answer import answer_character

from Annotations.annotate_tasks import annotate_task, extract_annotations



def predict_record_to_sample(record: Dict[str, Any]) -> Sample:
    input = f"{record['env_description']} What state would the agent be in if it took the following actions?:\n initial_state: {record['initial_state']}\n{record['action_sequence']} \n give your answer in the form ((x, y), d) where d is the direction the agent is facing (east=0, south=1, west=2, north=3)."
    target=record["target_state"]
    sample = Sample(input=input,
                    target=target,
                    metadata=record,
                    )
    return sample

@task
def predict_task() -> Task:
    dataset = hf_dataset("salem-mbzuai/LLM-BabyBench",
                         name="predict",
                         split="train",
                         sample_fields=predict_record_to_sample
                         )
    return Task(dataset=dataset,
                solver=basic_agent(),
                scorer=match(),
                )

@task
def decompose_task() -> Task:
    pass

@task
def plan_task() -> Task:
    pass
