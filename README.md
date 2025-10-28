# AI Workforce Suitability

This repository develops a framework for assessing how well large language models (LLMs) possess the core cognitive capabilities required for human work activities.

The pipeline connects benchmark datasets (annotated for cognitive demands), LLM performance evaluations, and workforce activity surveys, to estimate AI suitability scores, i.e., how suitable a given LLM is for performing the cognitive aspects of particular jobs.

## Overview

The goal of this project is to bridge **AI evaluation** and **workforce analysis** by creating comparable profiles of *capabilities* and *demands*.

1. **Annotate AI Benchmarks for Cognitive Capabilities** *(done)*
   - Benchmarks are automatically annotated using structured rubrics representing 18 core cognitive capabilities (e.g., Cognitive Flexibility, Episodic Memory, Metacognition, Planning, Theory of Mind).


2. **Extract Benchmark Demand Profiles** *(done)*
   - The above annotations are used to produce *benchmark demand profiles* that describe how much of each capability is required for each benchmark task instance.


3. **Evaluate LLMs on Annotated Benchmark Instances** *(in progress)*  
   - LLMs are evaluated on the annotated benchmark instances to obtain performance scores.


4. **Create LLM Capability Profiles** *(in progress)*
    - Performance data are combined with benchmark demand profiles to infer *capability profiles* of each model.
    - These tell us the capability levels of each model for each capability.
   

5. **Collect Workforce Activity Intensity Profiles** *(in progress)*  
   - Human participants rate the degree to which each of the 18 cognitive capabilities is important for specific work activities (using questionnaires).  
   - These form *work activity intensity profiles*.


6. **Match LLM Capabilities to Workforce Intensities** *(planned)*
   - By comparing LLM capability profiles with work activity intensity profiles, we estimates **AI suitability scores**
   - These scores indicate how likely an LLM is to have the core cognitive capabilities that are important for performing work activities in different job domains.

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Modules for each model you wish to use (e.g., `openai`, `transformers`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/markotesic/ai-workforce-suitability.git
cd ai-workforce-suitability
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Project Structure

- `Annotations/` - Contains annotation tools and rubrics for evaluating tasks
- `Annotated_Benchmarks/` - Individual task implementations (CoQA, Known_Unknowns, etc.)
- `logs/` - Evaluation run logs and results

## Annotating Tasks

The annotation system uses capability rubrics to score tasks across multiple dimensions.

### Running Annotations

To annotate a task simply run the task python file. For example, to annotate the CoQA task:

```bash
python ./Annotated_Benchmarks/CoQA/CoQA_task.py
```
This will load the CoQA dataset, convert each sample into a string to be inserted into the annotation prompt, and apply the capability rubrics to each sample.

These are then passed to a model for annotation and the results will be saved to `coqa_annotations.csv`. The only custom code needed is to load the dataset (needed for evaluation anyway) and code to convert each sample to a string (relatively simple for most text based tasks).

### Understanding Rubrics

Rubrics are stored in `Annotations/rubric.json` and define 6 capability dimensions (0-5 scale):
- Each dimension has detailed criteria for scoring
- Annotations use chain-of-thought reasoning before assigning scores
- Results are saved as CSV files with dataset name, sample ID, dimension, and score

## Running Benchmark Tasks

### Running Tasks

Each task in `Annotated_Benchmarks/` can be run using the Inspect AI command line:

```bash
# Run the CoQA task with default settings
inspect eval ./Annotated_Benchmarks/CoQA/CoQA_task.py --model openai/gpt-4

# Run with specific parameters
inspect eval ./Annotated_Benchmarks/CoQA/CoQA_task.py --model your-model-name --limit 10

# Run Known Unknowns task
inspect eval ./Annotated_Benchmarks/Known_Unknowns/known_unknown_task.py --model openai/gpt-4
```

### Available Tasks

- **CoQA**: Conversational Question Answering task
- **Known_Unknowns**: Task for handling uncertain scenarios

Results are automatically logged to the `logs/` directory with timestamps.

## Adding New Tasks

See the detailed guide in `Annotated_Benchmarks/README.md` for step-by-step instructions on creating new evaluation tasks.