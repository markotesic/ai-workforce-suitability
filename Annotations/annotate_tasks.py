


import json
import os
import csv
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.log import EvalLog
from inspect_ai.solver import system_message

from Annotations.annotation_agent import annotation_agent

DEFAULT_SYSTEM_MESSAGE = """You are an excellent annotation agent that labels benchmark instances using the instructions and rubric (if provided). Your goal is to assign a single integer score that reflects the dimension being evaluated (e.g., capability demand, factuality, ambiguity).

Reason through the instructions, the benchmark instance and the rubric (if provided) before deciding on a score.

When you have completed the task, call the submit() function to report your final answer.
"""


PROMPT_TEMPLATE = """RUBRIC: The following rubric describes six distinct levels of *{dimension}*
required by different tasks:
{rubric}
QUERY: {instance}
ANSWER: {target}
INSTRUCTION: Score the level of *{dimension}* demanded by the given QUERY–ANSWER pair using a discrete value from 0 to 5. Base your judgment on the *{dimension}* needed to
produce the ANSWER in response to the QUERY. Use CHAIN-OF-THOUGHTS
REASONING to reason step by step before assigning the score. After the
CHAIN-OF-THOUGHTS REASONING STEPS, conclude your assessment with the
statement: "Thus, the level of *{dimension}* demanded by the given QUERY-ANSWER pair is: SCORE", where 'SCORE' is the integer score you have determined.
"""

FACTUALITY_PROMPT_TEMPLATE = """QUERY: {instance}
ANSWER: {target}
INSTRUCTION: Given the query and an answer above, evaluate the accuracy or
factuality of this answer. Please rate it using a Likert scale from 1 to 5,
where 5 means surely accurate while 1 means surely inaccurate. You may employ
any reasoning strategy before providing your final answer. However, you must
conclude your assessment with the statement: 'Thus, the accuracy of the given
answer is: X', where 'X' is the score you've determined.
"""

AMBIGUITY_PROMPT_TEMPLATE = """QUERY: {instance}
ANSWER: {target}
INSTRUCTION: Given the query and an answer above, evaluate the ambiguity of
this query, independently of the potential difficulty of answering it. That
is, assess whether the query has a single clear interpretation with all
necessary details provided, or if it lacks critical information making it
difficult to answer confidently. Please rate it using a Likert scale from
1 to 5, where 5 means surely unambiguous (very clear) while 1 means surely
ambiguous (very unclear). You may employ any reasoning strategy before
providing your final answer. However, you must conclude your assessment with
the statement: 'Thus, the ambiguity of the given query is: X', where 'X' is
the score you've determined.
"""


def combine_dataset(task_dataset: Dataset, rubric_data) -> Dataset:
    new_samples = []
    for sample in task_dataset:
        instance = sample.input
        for rubric_dict in rubric_data:
            dimension = rubric_dict["dimension"]
            rubric = rubric_dict["rubric"]
            combined_string = PROMPT_TEMPLATE.format(dimension=dimension, rubric=rubric, instance=instance, target=sample.target)
            new_sample = Sample(
                input=combined_string,
                id=f"{sample.id}_{dimension}",
                metadata={
                    "dimension": dimension,
                    "sample_id": sample.id,
                }
                )
            new_samples.append(new_sample)

        # manually add the ambiguity and factuality questions
        ambiguity_string = AMBIGUITY_PROMPT_TEMPLATE.format(
            instance=instance,
            target=sample.target
        )
        ambiguity_sample = Sample(
            input=ambiguity_string,
            id=f"{sample.id}_ambiguity",
            metadata={
                "dimension": "ambiguity",
                "sample_id": sample.id,
            }
        )
        new_samples.append(ambiguity_sample)
        factuality_string = FACTUALITY_PROMPT_TEMPLATE.format(
            instance=instance,
            target=sample.target
        )
        factuality_sample = Sample(
            input=factuality_string,
            id=f"{sample.id}_factuality",
            metadata={
                "dimension": "factuality",
                "sample_id": sample.id,
            }
        )
        new_samples.append(factuality_sample)
    return MemoryDataset(new_samples, name=f"{task_dataset.name}_annotation")

def extract_annotations(log: EvalLog, output_file: str):
    assert log.samples is not None
    annotations = []
    for sample in log.samples:
        score = sample.output.completion
        annotation = (log.eval.dataset.name, sample.metadata["sample_id"], sample.metadata["dimension"], score)
        annotations.append(annotation)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dataset name", "sample id", "dimension", "score"])
        writer.writerows(annotations)




@task
def annotate_task(
    task_dataset: Dataset,
):
    assert task_dataset is not None
    rubric_data = json.load(open(
        os.path.join(Path(__file__).parent, "./rubric.json"), "r"))
    # now combine rubrics and samples in the template 
    annotation_dataset = combine_dataset(task_dataset, rubric_data)

    return Task(
        dataset=annotation_dataset,
        solver=annotation_agent(init=system_message(DEFAULT_SYSTEM_MESSAGE)),
    )




