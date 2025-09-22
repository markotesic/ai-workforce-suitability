import os
from pathlib import Path
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, FieldSpec, Sample, hf_dataset
from inspect_ai.scorer import choice, includes
from inspect_ai.solver import Choices, basic_agent, multiple_choice
from inspect_ai._util.answer import answer_character

from Annotations.annotate_tasks import annotate_task, extract_annotations

INPUT_TEMPLATE = """
# INSTRUCTIONS

In this study, you will see multiple examples. In each example, you will be given two contexts and a scenario. Your task is to read the two contexts and the subsequent scenario, and pick the context that makes more sense considering the scenario that follows. The contexts will be numbered "1" or "2". You must answer using "1" or "2" in your response.


# TRIAL EXAMPLE

## Contexts
1. "The bag is full of blocks."
2. "The bag is full of balls."

## Scenario
"I drew a ball from the bag."

## Task
Which context makes more sense given the scenario? Please answer using either "1" or "2".

## Response
2


# TRIAL EXAMPLE

## Contexts
1. "The boy likes cookies."
2. "The boy does not like cookies."

## Scenario
"The boy chose to eat a cookie."

## Task
Which context makes more sense given the scenario? Please answer using either "1" or "2".

## Response
1


# TEST EXAMPLE

## Contexts
1. "{context_1}"
2. "{context_2}"

## Scenario
"{scenario}"

## Task
Which context makes more sense given the scenario? Please answer using either "1" or "2".

## Response
"""

def record_to_sample(record) -> list[Sample]:
    context_1 = record["Context1"]
    context_2 = record["Context2"]
    scenario_1 = record["Target1"]
    scenario_2 = record["Target2"]

    sample_1 = Sample(input= INPUT_TEMPLATE.format(context_1=context_1, context_2=context_2, scenario=scenario_1),
                      target="1")
    sample_2 = Sample(input= INPUT_TEMPLATE.format(context_1=context_1, context_2=context_2, scenario=scenario_2),
                      target="2")
    return [sample_1, sample_2]

@task
def ewok_task() -> Task:
    dataset = hf_dataset("ewok-core/ewok-core-1.0", 
        split="test", 
        sample_fields=record_to_sample,
    )

    return Task(dataset=dataset,
                scorer=includes(),
                solver=basic_agent(),
        )

if __name__ == "__main__":
    dataset = hf_dataset("ewok-core/ewok-core-1.0", 
        split="test", 
        sample_fields=record_to_sample,
    )
    output_path = os.path.join(Path(__file__).parent, "ewok_annotations.csv")
    num_samples = 100
    dataset.shuffle(42)
    dataset = dataset[:num_samples]

    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model="openai/azure/gpt-4o", max_connections=2)
    extract_annotations(log[0], output_path)


