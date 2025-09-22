import os
from pathlib import Path
from typing import Any, Dict
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, FieldSpec, Sample, hf_dataset
from inspect_ai.scorer import Score, Scorer, Target, accuracy, choice, includes, match, scorer, stderr
from inspect_ai.solver import Choices, TaskState, basic_agent, generate, multiple_choice
from inspect_ai._util.answer import answer_character

from Annotations.annotate_tasks import annotate_task, extract_annotations
from Evaluations.LLM_BabyBench.decompose import DecomposeEvaluator
from Evaluations.LLM_BabyBench.plan import PlanEvaluator
from Evaluations.LLM_BabyBench.register import register_envs
import gymnasium as gym
from gymnasium import Env


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
    input = f"{record['env_description']} What actions should be taken to reach the target sub goal?:\n initial_state ((x, y), d) where d is the direction the agent is facing (east=0, south=1, west=2, north=3): {record['initial_state']}\n target sub goal: {record['target_subgoal']} \n give your answer as a comma seperated list of actions. The names of possible actions are again: left, right, forward, pickup, drop, toggle."
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
        # first create an environment with the correct setup for the sample
        env_name = state.metadata["level_name"]
        seed = state.metadata["seed"]

        # then apply the evaluate function from the original paper
        evaluator = PlanEvaluator()
        try:
            result = evaluator.evaluate(env_name=env_name, seed=seed, optimal_action_seq=target.text, llm_action_seq=state.output.completion)
        except ValueError as e:
            result = {
                "CR": 0,
                "PR": 0,
                "ACI" : 0,
            }
        # finally convert that to a score
        return Score(value=result["CR"])
    return score

@task
def plan_task() -> Task:
    dataset = hf_dataset("salem-mbzuai/LLM-BabyBench",
                         name="plan",
                         split="train",
                         sample_fields=plan_record_to_sample
                         )
    return Task(dataset=dataset,
                solver=basic_agent(),
                scorer=[plan_scorer()],
                )

def decompose_record_to_sample(record: Dict[str, Any]) -> Sample:
    input = f"{record['env_description']} What subgoals should be taken in order to achive the mission?:\n initial_state ((x, y), d) where d is the direction the agent is facing (east=0, south=1, west=2, north=3): {record['initial_state']}\n mission: {record['mission']} \n sub goals should be in the format and are executed in the order you give: "
    input += """- OpenSubgoal: If the agent is next to a closed door, this subgoal will open it. \n\n
- CloseSubgoal: If the agent is next to an open door, this subgoal will close it. \n\n
- DropSubgoal: If the agent is carrying an object, this subgoal drops it. \n\n
- PickupSubgoal: If the agent is not carrying any object and is next to an object, this subgoal picks it up. \n\n
- GoNextToSubgoal, (x,y): If there is a clear, without any blocker, path between the agent and the cell of coordinates (x,y), this subgoal makes the agent go next to this cell."""
    input += "\ngive your answer list of subgoals each on a new line. for example:\n GoNextToSubgoal, (3, 5)\nGoNextToSubgoal, (2, 5)\nGoNextToSubgoal, (1, 5)\nGoNextToSubgoal, (1, 6)\nPickupSubgoal"
    sample = Sample(input=input,
                    metadata=record,
                    )
    return sample


def make_env(env_name: str, seed: int) -> Env:
    """
    Create an environment from config file specifications.
    """
    register_envs()
    env = gym.make(env_name, tile_size=32, render_mode='rgb_array')
    env.reset(seed=seed)
    return env

@scorer(metrics=[accuracy(), stderr()])
def decompose_scorer()-> Scorer:
    """This is a task specific scorer that will check if the action sequence ends up in the correct location."""
    async def score(state: TaskState, target: Target) -> Score:
        # first create an environment with the correct setup for the sample
        env_name = state.metadata["level_name"]
        seed = state.metadata["seed"]
        env = make_env(env_name, seed)

        # then apply the evaluate function from the original paper
        evaluator = DecomposeEvaluator()
        try:
            result = evaluator.evaluate(env, state.output.completion)
        except ValueError as e:
            result = {
                "CR": 0,
                "PR": 0,
                "ACI" : 0,
            }
        # finally convert that to a score
        return Score(value=result["CR"])
    return score

@task
def decompose_task() -> Task:
    dataset = hf_dataset("salem-mbzuai/LLM-BabyBench",
                         name="decompose",
                         split="train",
                         sample_fields=decompose_record_to_sample
                         )
    num_samples = 100
    dataset.shuffle(42)
    dataset = dataset[:num_samples]
    return Task(dataset=dataset,
                solver=basic_agent(),
                scorer=decompose_scorer(),
                )
