import json
import os
import csv
from pathlib import Path
from typing import Any, Dict
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import choice, model_graded_qa
from inspect_ai.solver import Choices, basic_agent, multiple_choice
from inspect_ai._util.answer import answer_character, answer_index

from Annotations.annotate_tasks import annotate_task, extract_annotations


def record_to_sample(record: Dict[str, Any], dataset_path: str, id : int = 0) -> Sample:
    question = record["Problem"]
    target = record["Solution"]
    
    return Sample(
        input=question,
        target=target,
        id=id
    )


def custom_loader(dataset_path: str) -> Dataset:
    samples = []
    
    with open(dataset_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for idx, record in enumerate(reader):
            sample = record_to_sample(record, dataset_path, idx)
            samples.append(sample)
    
    return MemoryDataset(samples=samples, name="MacGyver", location=dataset_path, shuffled=False)


@task
def macgyver_task(
    dataset_path: str | None = None,
    ) -> Task:
    if dataset_path is None:
        dataset_path = os.path.join(Path(__file__).parent, "problem_solution_pair.csv")
    dataset = custom_loader(dataset_path=dataset_path)

    return Task(dataset=dataset,
                scorer=model_graded_qa(),
                solver=basic_agent(),
        )


if __name__ == "__main__":
    dataset_path = os.path.join(Path(__file__).parent, "problem_solution_pair.csv")
    output_path = os.path.join(Path(__file__).parent, "macgyver_annotations.csv")
    dataset = custom_loader(dataset_path=dataset_path)
    num_samples = 100
    dataset.shuffle(42)
    dataset = dataset[:num_samples]

    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model="openai/azure/gpt-4o", max_connections=2)
    extract_annotations(log[0], output_path)







