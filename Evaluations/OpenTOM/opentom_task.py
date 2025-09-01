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



def record_to_sample(record: Dict[str, Any], dataset_path: str, id : int = 0) -> list[Sample]:
    input = record["narrative"]
    question = record["question"]["question"]
    target = record["question"]["answer"]

    sample = Sample(
        id=id,
        input= f"context: {input} \n question: {question}",
        target=target,
        )

    return [sample]


def custom_loader(dataset_path: str) -> Dataset:
    
    json_data = json.load(open(dataset_path, 'r'))
    samples = []
    # Iterate through the samples in the record
    for item in json_data:
        new_samples = record_to_sample(item, dataset_path, id=len(samples))
        samples.extend(new_samples)
    return MemoryDataset(samples=samples, name="OpenToM", location=dataset_path, shuffled=False)


@task
def opentom_task(
    dataset_path: str | None = None,
    ) -> Task:
    if dataset_path is None:
        dataset_path = os.path.join(Path(__file__).parent, "opentom.json")
    dataset = custom_loader(dataset_path=dataset_path)

    return Task(dataset=dataset,
                scorer=model_graded_qa(),
                solver=basic_agent(),
        )



if __name__ == "__main__":
    dataset_path = os.path.join(Path(__file__).parent, "opentom.json")
    output_path = os.path.join(Path(__file__).parent, "opentom_annotations.csv")
    dataset = custom_loader(dataset_path=dataset_path)

    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model="openai/azure/gpt-4o", max_connections=2)
    extract_annotations(log[0], output_path)







