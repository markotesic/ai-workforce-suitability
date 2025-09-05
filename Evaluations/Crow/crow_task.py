import json
import os
from pathlib import Path
from typing import Any, Dict
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import choice, model_graded_qa
from inspect_ai.solver import Choices, basic_agent, generate, multiple_choice
from inspect_ai._util.answer import answer_character, answer_index

from Annotations.annotate_tasks import annotate_task, extract_annotations


def record_to_sample(record: Dict[str, Any], dataset_path: str, id : int = 0) -> list[Sample]:

    if record["task"] == "dialogue":
        input = ""
        for response in record["dialogue"]:
            input += f"{response} \n"

        target = record["response"]

    elif record["task"] == "intent":
        input = record["headline"]
        target = record["intent"]

    elif record["task"] == "safety":
        input = record["scenario"]
        target = record["action"]

    elif record["task"] == "stance":
        input = record["belief"]
        target = record["argument"]

    elif record["task"] == "summarization":
        input = ""
        for response in record["dialogue"]:
            input += f"{response} \n"
        target = record["summary"]

    else:
        raise ValueError(f"Unknown task type: {record['task']}")

    sample = Sample(
        id=id,
        input= input,
        target=target,
        )

    return [sample]


def custom_loader(dataset_dir: str) -> Dataset:
    
    samples = []
    json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]

    for json_file in json_files:
        json_data = json.load(open(os.path.join(dataset_dir, json_file), 'r'))
        for item in json_data:
            new_samples = record_to_sample(item, dataset_dir, id=len(samples))
            samples.extend(new_samples)

    return MemoryDataset(samples=samples, name="Crow", location=dataset_dir, shuffled=False)


@task
def crow_task(
    dataset_dir: str | None = None,
    ) -> Task:
    if dataset_dir is None:
        dataset_dir = os.path.join(Path(__file__).parent)
    dataset = custom_loader(dataset_dir=dataset_dir)

    return Task(dataset=dataset,
                scorer=choice(),
                solver=generate(),
        )




if __name__ == "__main__":
    dataset_dir = os.path.join(Path(__file__).parent)
    output_path = os.path.join(Path(__file__).parent, "crow_annotations.csv")
    dataset = custom_loader(dataset_dir=dataset_dir)

    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model="openai/azure/gpt-4o", max_connections=2)
    extract_annotations(log[0], output_path)








