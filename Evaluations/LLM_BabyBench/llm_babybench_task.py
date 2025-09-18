import os
from pathlib import Path
from typing import Any, Dict
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, FieldSpec, Sample, hf_dataset
from inspect_ai.scorer import Score, Scorer, Target, accuracy, choice, includes, match, scorer, stderr
from inspect_ai.solver import Choices, TaskState, basic_agent, generate, multiple_choice
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

def plan_record_to_sample(record: Dict[str, Any]) -> Sample:
    input = f"{record['env_description']} What actions should be taken to reach the target sub goal?:\n initial_state ((x, y), d) where d is the direction the agent is facing (east=0, south=1, west=2, north=3): {record['initial_state']}\n target sub goal: {record['target_subgoal']} \n give your answer as a comma seperated list of actions (left|right|forward)."
    target=record["expert_action_sequence"]
    sample = Sample(input=input,
                    target=target,
                    metadata=record,
                    )
    return sample

@scorer(metrics=[accuracy(), stderr()])
def plan_scorer()-> Scorer:
    """This is a task specific scorer that will check if the action sequence ends up in the correct location."""
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value="C")
    return score

@task
def plan_task() -> Task:
    dataset = hf_dataset("salem-mbzuai/LLM-BabyBench",
                         name="plan",
                         split="train",
                         sample_fields=plan_record_to_sample
                         )
    return Task(dataset=dataset,
                solver=generate(),
                scorer=[match(), plan_scorer()],
                )

@task
def decompose_task() -> Task:
    pass
