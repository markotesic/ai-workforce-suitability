#!/usr/bin/env python3
"""
Utility script to run annotation tasks across all evaluation folders.

This script discovers all *_task.py files in the Evaluations directory and runs them
to generate annotation data. Each task runs independently and outputs its results
to a CSV file in its respective directory.
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from Annotations.convert_rubrics import convert_rubrics_to_json

DEFAULT_NUM_SAMPLES = 10

def discover_task_files(evaluations_dir: str = "Evaluations") -> List[str]:
    """Discover all *_task.py files in the evaluations directory."""
    pattern = os.path.join(evaluations_dir, "**", "*_task.py")
    task_files = glob.glob(pattern, recursive=True)
    return sorted(task_files)


def get_task_name(task_file: str) -> str:
    """Extract a readable task name from the file path."""
    return os.path.basename(os.path.dirname(task_file))


def run_task(task_file: str, timeout: Optional[int] = None, verbose: bool = False, num_samples: int = DEFAULT_NUM_SAMPLES) -> bool:
    """
    Run a single task file and return True if successful.

    Args:
        task_file: Path to the task file to run
        timeout: Optional timeout in seconds
        verbose: If True, show live output from the subprocess
        num_samples: Number of samples to use for annotation

    Returns:
        True if the task completed successfully, False otherwise
    """
    task_name = get_task_name(task_file)
    print(f"[{task_name}] Starting annotation task...")

    start_time = time.time()

    # Create the command to import the module and call annotate()
    module_name = os.path.splitext(os.path.basename(task_file))[0]
    import_cmd = f"import sys; sys.path.insert(0, '.'); from {module_name} import annotate; annotate({num_samples})"

    try:
        if verbose:
            # Direct terminal pass-through - let Inspect AI write directly to terminal
            process = subprocess.Popen(
                [sys.executable, "-c", import_cmd],
                cwd=os.path.dirname(task_file)
            )

            # Handle timeout manually by polling
            while process.poll() is None:
                if timeout and (time.time() - start_time) > timeout:
                    process.terminate()
                    print(f"\n[{task_name}] Timeout after {timeout}s")
                    return False
                time.sleep(0.1)  # Small sleep to avoid busy waiting

            return_code = process.returncode
        else:
            # Use run() for captured output (current behavior)
            result = subprocess.run(
                [sys.executable, "-c", import_cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.dirname(task_file)
            )
            return_code = result.returncode

        duration = time.time() - start_time

        if return_code == 0:
            print(f"[{task_name}]  Completed successfully in {duration:.1f}s")
            return True
        else:
            print(f"[{task_name}]  Failed with exit code {return_code}")
            if not verbose and 'result' in locals() and result.stderr:
                print(f"[{task_name}] Error output:")
                for line in result.stderr.strip().split('\n'):
                    print(f"[{task_name}]   {line}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[{task_name}]  Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"[{task_name}]  Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run annotation tasks for all evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_annotations.py                    # Run all tasks
  python run_annotations.py --include EmoBench # Run only EmoBench
  python run_annotations.py --exclude BigBenchHard AGIEval  # Exclude specific tasks
  python run_annotations.py --timeout 1800     # Set 30min timeout per task
  python run_annotations.py --verbose          # Show live output from tasks
        """
    )

    parser.add_argument(
        "--include",
        nargs="+",
        help="Only run tasks containing these names (case-insensitive)"
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Exclude tasks containing these names (case-insensitive)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout per task in seconds (default: no timeout)"
    )
    parser.add_argument(
        "--evaluations-dir",
        default="Evaluations",
        help="Directory containing evaluation tasks (default: Evaluations)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which tasks would be run without actually running them"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show live output from each task as it runs"
    )

    args = parser.parse_args()

    # make sure to update the rubric json file first
    convert_rubrics_to_json(os.path.join(Path(__file__).parent, "rubric_files"), os.path.join(Path(__file__).parent, "rubric.json"))

    # Change to the script's parent directory to ensure relative paths work
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    # Discover all task files
    task_files = discover_task_files(args.evaluations_dir)

    if not task_files:
        print(f"No task files found in {args.evaluations_dir}")
        return 1

    # Filter tasks based on include/exclude patterns
    filtered_tasks = []
    for task_file in task_files:
        task_name = get_task_name(task_file)

        # Check include filter
        if args.include:
            if not any(pattern.lower() in task_name.lower() for pattern in args.include):
                continue

        # Check exclude filter
        if args.exclude:
            if any(pattern.lower() in task_name.lower() for pattern in args.exclude):
                continue

        filtered_tasks.append(task_file)

    if not filtered_tasks:
        print("No tasks match the specified filters")
        return 1

    print(f"Found {len(filtered_tasks)} task(s) to run:")
    for task_file in filtered_tasks:
        task_name = get_task_name(task_file)
        print(f"  - {task_name}")

    if args.dry_run:
        print("\nDry run complete. Use --help for more options.")
        return 0

    timeout_msg = f" with {args.timeout}s timeout per task" if args.timeout else ""
    print(f"\nStarting annotation tasks{timeout_msg}...\n")

    # Run tasks
    successful = 0
    failed = 0

    for i, task_file in enumerate(filtered_tasks, 1):
        task_name = get_task_name(task_file)
        print(f"[{i}/{len(filtered_tasks)}] Running {task_name}")

        if run_task(task_file, args.timeout, args.verbose, DEFAULT_NUM_SAMPLES):
            successful += 1
        else:
            failed += 1

        print()  # Add spacing between tasks

    # Summary
    print("=" * 50)
    print("Annotation run complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(filtered_tasks)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())