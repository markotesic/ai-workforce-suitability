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

SINGLE_ANSWER_TEMPLATE_COT = r"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
""".strip()


def record_to_sample(record: Dict[str, Any]) -> Sample:
    input = f"context: {record['document']['text']} \n question: {record['question']['text']}"
    target = [answer["text"] for answer in record["answers"]]

    # Create a unique identifier based on content hash since no index is available
    import hashlib
    content_hash = hashlib.md5(f"{input}{''.join(target)}".encode()).hexdigest()[:8]

    return Sample(
        input=input,
        target=target,
        id=f"narrativeqa_{content_hash}",
    )


@task
def narrativeqa_task() -> Task:
    dataset = hf_dataset("deepmind/narrativeqa", 
        split="test", 
        sample_fields=record_to_sample,
    )

    return Task(dataset=dataset,
                scorer=model_graded_qa(),
                solver=basic_agent(),
        )




if __name__ == "__main__":
    dataset = hf_dataset("deepmind/narrativeqa", 
        split="test", 
        sample_fields=record_to_sample,
    )
    output_path = os.path.join(Path(__file__).parent, "narrativeqa_annotations.csv")
    num_samples = DEFAULT_NUM_SAMPLES
    dataset.shuffle(42)
    dataset = dataset[:num_samples]

    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model="openai/azure/gpt-4o" )
    extract_annotations(log[0], output_path)

