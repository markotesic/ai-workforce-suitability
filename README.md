# AI Workforce Suitability

A package for evaluating AI workforce suitability through various tasks and annotations using the Inspect AI framework.

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
- `Evaluations/` - Individual task implementations (CoQA, Known_Unknowns, etc.)
- `logs/` - Evaluation run logs and results

## Annotating Tasks

The annotation system uses OECD capability rubrics to score tasks across multiple dimensions.

### Running Annotations

To annotate a task simply run the task python file. For example, to annotate the CoQA task:

```bash
python ./Evaluations/CoQA/CoQA_task.py
```
This will load the CoQA dataset, convert each sample into a string to be insterted into the annotation prompt, and apply the OECD capability rubrics to each sample.

These are then passed to a model for annotation and the results will be saved to `coqa_annotations.csv`. The only custom code needed is to load the dataset (needed for evaluation anyway) and code to covert each sample to a string (relatively simple for most text based tasks).

### Understanding Rubrics

Rubrics are stored in `Annotations/rubric.json` and define 6 capability dimensions (0-5 scale):
- Each dimension has detailed criteria for scoring
- Annotations use chain-of-thought reasoning before assigning scores
- Results are saved as CSV files with dataset name, sample ID, dimension, and score

## Evaluating Tasks

### Running Task Evaluations

Each task in `Evaluations/` can be run using the Inspect AI command line:

```bash
# Run the CoQA task with default settings
inspect eval ./Evaluations/CoQA/CoQA_task.py --model openai/gpt-4

# Run with specific parameters
inspect eval ./Evaluations/CoQA/CoQA_task.py --model your-model-name --limit 10

# Run Known Unknowns task
inspect eval ./Evaluations/Known_Unknowns/known_unknown_task.py --model openai/gpt-4
```

### Available Tasks

- **CoQA**: Conversational Question Answering task
- **Known_Unknowns**: Task for handling uncertain scenarios

Results are automatically logged to the `logs/` directory with timestamps.

## Adding New Tasks

See the detailed guide in `Evaluations/README.md` for step-by-step instructions on creating new evaluation tasks.