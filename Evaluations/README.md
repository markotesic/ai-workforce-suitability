# Evaluations

This folder contains all the code needed to run evaluations for each task, including resources like original JSON files or images.

## Available Tasks

- **CoQA**: Conversational Question Answering task
- **Known_Unknowns**: Task for handling uncertain scenarios

## Running Evaluations

Each task can be run independently using the Inspect AI command line:

```bash
# Run a specific task
inspect eval ./Evaluations/TaskName/task_name.py --model your-model-name

# Run with additional parameters
inspect eval ./Evaluations/TaskName/task_name.py --model openai/gpt-4 --limit 50 --max-connections 5
```

Results are automatically logged to the `logs/` directory with timestamps.

## Adding New Tasks

Follow these steps to add a new evaluation task:

### 1. Create Task Directory

Create a new directory under `Evaluations/` with your task name:

```
Evaluations/
├── YourTaskName/
│   ├── __init__.py
│   ├── your_task_name.py
│   ├── README.md
│   └── data files (JSON, images, etc.)
```

### 2. Implement Task Function

Create your main task file (`your_task_name.py`) following this template:

```python
import json
import os
from pathlib import Path
from typing import Any, Dict

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import includes  # or appropriate scorer

def custom_loader(dataset_path: str) -> Dataset:
    """Load your dataset and convert to Inspect AI format"""
    # Load your data (JSON, CSV, etc.)
    data = json.load(open(dataset_path, 'r'))
    
    samples = []
    for item in data:
        sample = Sample(
            input=item["question"],  # Adapt to your data structure
            target=item["answer"],   # Expected answer(s)
            id=item["id"],          # Unique identifier
            metadata={              # Additional metadata
                "source": item.get("source", ""),
                # Add other relevant fields
            }
        )
        samples.append(sample)
    
    return MemoryDataset(
        samples=samples, 
        name="YourTaskName", 
        location=dataset_path, 
        shuffled=False
    )

@task
def your_task_name(dataset_path: str | None = None) -> Task:
    """Main task function"""
    if dataset_path is None:
        dataset_path = os.path.join(Path(__file__).parent, "default_data.json")
    
    dataset = custom_loader(dataset_path)
    
    return Task(
        dataset=dataset,
        scorer=includes(),  # Choose appropriate scorer
        # Add solver if needed for multi-step tasks
    )

# Optional: Add annotation support
def convert_input_to_string(dataset: Dataset) -> Dataset:
    """Convert inputs to strings for annotation (if needed)"""
    for sample in dataset:
        # Convert complex inputs (like chat messages) to strings
        sample.input = str(sample.input)
    return dataset

# Optional: Main block for direct execution
if __name__ == "__main__":
    from Annotations.annotate_tasks import annotate_task, extract_annotations
    from inspect_ai import eval
    
    # Run annotation
    dataset_path = os.path.join(Path(__file__).parent, "data.json")
    output_path = os.path.join(Path(__file__).parent, "annotations.csv")
    
    dataset = custom_loader(dataset_path)
    dataset = convert_input_to_string(dataset)
    
    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model="openai/azure/gpt-4o", max_connections=2)
    extract_annotations(log[0], output_path)
```

### 3. Choose Appropriate Scorer

Select the right scorer for your task:

- `includes()` - For tasks where any target answer is acceptable
- `accuracy()` - For exact match requirements  
- `mean()` - For numerical scoring
- Custom scorers for specialized evaluation

### 4. Handle Complex Data Types

For tasks with complex inputs (like chat conversations):

```python
from inspect_ai.model import ChatMessage, ChatMessageUser, ChatMessageAssistant

def create_chat_sample(conversation_data):
    messages = []
    for turn in conversation_data["turns"]:
        if turn["role"] == "user":
            messages.append(ChatMessageUser(content=turn["content"]))
        else:
            messages.append(ChatMessageAssistant(content=turn["content"]))
    
    return Sample(
        input=messages,
        target=conversation_data["expected_response"],
        id=conversation_data["id"]
    )
```

### 5. Add Task Documentation

Create a `README.md` in your task directory:

```markdown
# Task Name

Brief description of the task and what it evaluates.

## Data Format

Describe the expected input data format.

## Usage

```python
from Evaluations.YourTaskName.your_task_name import your_task_name
from inspect_ai import eval

task = your_task_name("path/to/data.json")
results = eval(task, model="your-model")
```

## Annotation

If the task supports annotation:

```python
# Run annotation workflow
python your_task_name.py
```
```

### 6. Update Package Structure

Add your task to the package by updating `pyproject.toml` if needed:

```toml
[tool.setuptools.packages.find]
include = ["ai_workforce_suitability*", "Annotations*", "Evaluations*"]
```

### 7. Test Your Task

Test your task implementation:

```bash
# Test with a small dataset
inspect eval ./Evaluations/YourTaskName/your_task_name.py --model openai/gpt-3.5-turbo --limit 5

# Test with different models
inspect eval ./Evaluations/YourTaskName/your_task_name.py --model anthropic/claude-3-haiku --limit 3
```

### Best Practices

1. **Data Validation**: Validate input data format in your loader
2. **Error Handling**: Include proper error handling for missing files/malformed data
3. **Metadata**: Include relevant metadata for analysis
4. **Documentation**: Document data format and usage clearly
5. **Testing**: Test with small datasets before full runs
6. **Consistent Naming**: Use consistent naming conventions (snake_case for files/functions)

### Integration with Annotation System

To make your task compatible with the annotation system:

1. Implement `convert_input_to_string()` function
2. Add the annotation workflow in `if __name__ == "__main__"`
3. Ensure your dataset name is descriptive for the annotation CSV output

Your task will then automatically work with the OECD capability rubrics for comprehensive evaluation.