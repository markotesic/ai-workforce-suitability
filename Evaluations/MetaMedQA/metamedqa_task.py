import os
from pathlib import Path
from typing import Any, Dict
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, FieldSpec, Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import Choices, multiple_choice
from inspect_ai._util.answer import answer_character

from Annotations.annotate_tasks import annotate_task, extract_annotations
from Annotations.run_annotations import DEFAULT_NUM_SAMPLES

SINGLE_ANSWER_TEMPLATE_COT = r"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
""".strip()

def record_to_sample(record: Dict[str, Any]) -> Sample:
    input = record["question"]
    target = record["answer_idx"]
    choices = record["options"].values()

    # Create a unique identifier based on content hash since no index is available
    import hashlib
    content_hash = hashlib.md5(f"{input}{''.join(choices)}{str(target)}".encode()).hexdigest()[:8]

    return Sample(
        input=input,
        target=target,
        choices=choices,
        id=f"metamedqa_{content_hash}",
    )


@task
def metamedqa_task() -> Task:
    dataset = hf_dataset("maximegmd/MetaMedQA", 
        split="test", 
        sample_fields=record_to_sample,
    )

    return Task(dataset=dataset,
                scorer=choice(),
                solver=multiple_choice(cot=True),
        )

def answer_options(choices: Choices) -> str:
    r"""
    Returns the `choices` formatted as a multiple choice question, e.g.:

    ["choice 1", "choice 2", "choice 3"] ->
        "A) choice 1\nB) choice 2\nC) choice 3"
    """
    indexes = list(range(len(choices)))

    return "\n".join(
        [f"{answer_character(i)}) {choices[j].value}" for i, j in enumerate(indexes)]
    )

def prompt(question: str, choices: Choices, template: str) -> str:
    choices_text = answer_options(choices)
    letters = ",".join(answer_character(i) for i in range(len(choices)))

    return template.format(
        choices=choices_text,
        letters=letters,
        question=question,
    )


def convert_input_to_string(dataset: Dataset) -> Dataset:
    """takes a dataset and converts the inputs from messages to strings for annotation only"""
    for sample in dataset:
        new_input = prompt(question=sample.input, choices=Choices(sample.choices), template=SINGLE_ANSWER_TEMPLATE_COT)
        sample.input = new_input

    return dataset

if __name__ == "__main__":
    dataset = hf_dataset("maximegmd/MetaMedQA", 
        split="test", 
        sample_fields=record_to_sample,
    )
    dataset = convert_input_to_string(dataset)
    output_path = os.path.join(Path(__file__).parent, "socialnorm_annotations.csv")
    num_samples = DEFAULT_NUM_SAMPLES
    dataset.shuffle(42)
    dataset = dataset[:num_samples]

    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model="openai/azure/gpt-4o" )
    extract_annotations(log[0], output_path)


