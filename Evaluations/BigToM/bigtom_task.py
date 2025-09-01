import json
import os
from pathlib import Path
from typing import Any, Dict
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import choice, model_graded_qa
from inspect_ai.solver import Choices, basic_agent, multiple_choice
from inspect_ai._util.answer import answer_character, answer_index

from Annotations.annotate_tasks import annotate_task, extract_annotations



def record_to_sample(record: list[str], dataset_path: str, id : int = 0) -> list[Sample]:
    input = record[0]
    question = record[1]
    target = record[2]

    sample = Sample(
        id=id,
        input= f"context: {input} \n question: {question}",
        target=target,
        )

    return [sample]


def custom_loader(dataset_dir: str) -> Dataset:
    
    samples = []

    for foldername in os.listdir(dataset_dir):
        if not os.path.isdir(os.path.join(dataset_dir, foldername)):
            continue
        folder_path = os.path.join(dataset_dir, foldername)
        for filename in os.listdir(folder_path):
            csv_file = open(os.path.join(folder_path, filename), "r")
            for line in csv_file:
                record = line.split(";")
                new_samples = record_to_sample(record=record, dataset_path=dataset_dir, id=len(samples))
                samples.extend(new_samples)
    return MemoryDataset(samples=samples, name="BigToM", location=dataset_dir, shuffled=False)


@task
def bigtom_task(
    dataset_dir: str | None = None,
    ) -> Task:
    if dataset_dir is None:
        dataset_dir = os.path.join(Path(__file__).parent, "conditions")
    dataset = custom_loader(dataset_dir=dataset_dir)

    return Task(dataset=dataset,
                scorer=model_graded_qa(),
                solver=basic_agent(),
        )



if __name__ == "__main__":
    dataset_dir = os.path.join(Path(__file__).parent, "conditions")
    output_path = os.path.join(Path(__file__).parent, "bigtom_annotations.csv")
    dataset = custom_loader(dataset_dir=dataset_dir)

    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model="openai/azure/gpt-4o", max_connections=2)
    extract_annotations(log[0], output_path)







