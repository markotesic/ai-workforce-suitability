

import json
import os
from pathlib import Path
from typing import Any, Dict

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, accuracy, includes, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

def record_to_sample(record: Dict[str, Any], dataset_path: str) -> Sample:
    sample_id = record["name"]
    context = record["story"]
    questions = []
    answers = []
    turn_id = 1

    for question in record["questions"]:
        assert question["turn_id"] == turn_id
        current_answers = []
        questions.append(question["input_text"])

        for answer in record["answers"]:
            if answer["turn_id"] == question["turn_id"]:
                current_answers.append(answer["input_text"])

        for answer_set in record["additional_answers"]:
            for answer in answer_set:
                if answer["turn_id"] == question["turn_id"]:
                    current_answers.append(answer["input_text"])
        
        answers.extend(current_answers)
        turn_id += 1

    return Sample(input=context, metadata={
        "questions": questions,
        "answers": answers,
    })

def custom_loader(dataset_path: str) -> Dataset:
    
    json_data = json.load(open(dataset_path, 'r'))
    samples = []
    # Iterate through the samples in the record
    for item in json_data["data"]:
        sample = record_to_sample(item, dataset_path)
        samples.append(sample)
    return MemoryDataset(samples=samples, name="CoQA", location=dataset_path, shuffled=False)

@solver
def CoQA_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return state
    
    return solve

@scorer(metrics=[accuracy()])
def custom_multi_scorer()-> Scorer:
    """Custom scorer that provides grouped accuracy metrics."""
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=0)
    
    return score

@task
def CoQA_task(dataset_path: str = os.path.join(Path(__file__).parent, "coqa.test.json")) -> Task:
    dataset = custom_loader(dataset_path=dataset_path)

    solver = CoQA_solver()

    return Task(dataset=dataset,
                solver=solver,
                scorer=includes(),
        )
    




