# AI Workforce Suitability

A package for evaluating AI workforce suitability through various tasks and annotations using the Inspect AI framework.

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/ai-workforce-suitability.git
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

To annotate a task (e.g., CoQA):

```python
from Evaluations.CoQA.CoQA_task import custom_loader, convert_input_to_string
from Annotations.annotate_tasks import annotate_task, extract_annotations
from inspect_ai import eval

# Load and prepare dataset
dataset = custom_loader("path/to/dataset.json")
dataset = convert_input_to_string(dataset)  # Convert chat messages to strings

# Create annotation task
annotation_task = annotate_task(dataset)

# Run evaluation with your preferred model
log = eval(annotation_task, model="openai/azure/gpt-4o", max_connections=2)

# Extract results to CSV
extract_annotations(log[0], "output_annotations.csv")
```

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