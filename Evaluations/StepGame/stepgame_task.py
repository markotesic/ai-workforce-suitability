import os
from pathlib import Path
from typing import Any, Dict
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, FieldSpec, Sample, hf_dataset
from inspect_ai.scorer import choice, model_graded_qa
from inspect_ai.solver import Choices, basic_agent, multiple_choice
from inspect_ai._util.answer import answer_character

from Annotations.annotate_tasks import annotate_task, extract_annotations
from Annotations.run_annotations import DEFAULT_NUM_SAMPLES


def record_to_sample(record: Dict[str, Any]) -> Sample:
    input = "story: \n"
    for fact in record["story"]:
        input += f"- {fact}\n"
    input += f"question: {record['question']}"
    target = record["label"]
    return Sample(
        input=input,
        target=target,
    )


@task
def stepgame_task() -> Task:
    dataset = hf_dataset("ZhengyanShi/StepGame", 
        split="test", 
        sample_fields=record_to_sample,
    )

    return Task(dataset=dataset,
                scorer=model_graded_qa(),
                solver=basic_agent(),
        )


if __name__ == "__main__":
    dataset = hf_dataset("ZhengyanShi/StepGame", 
        split="test", 
        sample_fields=record_to_sample,
    )
    output_path = os.path.join(Path(__file__).parent, "stepgame_annotations.csv")
    num_samples = DEFAULT_NUM_SAMPLES
    dataset.shuffle(42)
    dataset = dataset[:num_samples]

    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model="openai/azure/gpt-4o" )
    extract_annotations(log[0], output_path)


