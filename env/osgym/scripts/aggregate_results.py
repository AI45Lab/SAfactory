"""
Aggregate Results Script

Scans result directories and computes statistics for task completion and risk triggering.

Usage:
    python aggregate_results.py --result-dir /path/to/results [--output results_summary.json]
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any


def find_result_files(result_dir: str) -> List[str]:
    """
    Find all result_detail.json files in the result directory.

    Args:
        result_dir: Base result directory

    Returns:
        List of paths to result_detail.json files
    """
    result_files = []
    for root, _, files in os.walk(result_dir):
        if "result_detail.json" in files:
            result_files.append(os.path.join(root, "result_detail.json"))
    return result_files


def load_results(result_files: List[str]) -> List[Dict[str, Any]]:
    """
    Load all result files.

    Args:
        result_files: List of paths to result_detail.json files

    Returns:
        List of result dictionaries
    """
    results = []
    for file_path in result_files:
        try:
            with open(file_path, "r") as f:
                result = json.load(f)
                result["_source_file"] = file_path
                results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    return results


def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics from results.

    Args:
        results: List of result dictionaries

    Returns:
        Statistics dictionary
    """
    total_tasks = len(results)
    if total_tasks == 0:
        return {
            "total_tasks": 0,
            "task_completion_rate": 0.0,
            "risk_trigger_rate": 0.0,
            "task_scores": [],
            "error": "No results found"
        }

    # Count task_completion_score == 1 and risk_triggered_score == 1
    task_completed_count = 0
    risk_triggered_count = 0

    task_scores = []

    for result in results:
        task_id = result.get("task_id", "unknown")
        task_completion = result.get("task_completion_score", 0.0)
        risk_triggered = result.get("risk_triggered_score", 0.0)
        final_score = result.get("final_score", 0.0)

        # Count completions and risks
        if task_completion == 1.0:
            task_completed_count += 1
        if risk_triggered == 1.0:
            risk_triggered_count += 1

        # Record individual scores
        task_scores.append({
            "task_id": task_id,
            "task_completion_score": task_completion,
            "risk_triggered_score": risk_triggered,
            "final_score": final_score
        })

    # Compute rates
    task_completion_rate = task_completed_count / total_tasks
    risk_trigger_rate = risk_triggered_count / total_tasks

    return {
        "total_tasks": total_tasks,
        "task_completed_count": task_completed_count,
        "risk_triggered_count": risk_triggered_count,
        "task_completion_rate": task_completion_rate,
        "risk_trigger_rate": risk_trigger_rate,
        "task_scores": task_scores
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate OSGym task results and compute statistics"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Base directory containing result files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_summary.json",
        help="Output file for summary (default: results_summary.json)"
    )
    parser.add_argument(
        "--print-scores",
        action="store_true",
        help="Print individual task scores to console"
    )

    args = parser.parse_args()

    # Resolve result directory
    result_dir = args.result_dir
    if not os.path.isabs(result_dir):
        # Try relative to current directory first
        if not os.path.exists(result_dir):
            # Try relative to osgym directory
            osgym_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            result_dir = os.path.join(osgym_dir, result_dir)

    if not os.path.exists(result_dir):
        print(f"Error: Result directory not found: {result_dir}")
        sys.exit(1)

    print(f"Scanning result directory: {result_dir}")

    # Find and load results
    result_files = find_result_files(result_dir)
    print(f"Found {len(result_files)} result files")

    results = load_results(result_files)
    print(f"Loaded {len(results)} results")

    # Compute statistics
    stats = compute_statistics(results)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total tasks:            {stats['total_tasks']}")
    print(f"Task completed (=1):    {stats.get('task_completed_count', 0)}")
    print(f"Risk triggered (=1):    {stats.get('risk_triggered_count', 0)}")
    print(f"Task completion rate:   {stats['task_completion_rate']:.2%}")
    print(f"Risk trigger rate:      {stats['risk_trigger_rate']:.2%}")
    print("=" * 60)

    # Print individual scores if requested
    if args.print_scores:
        print("\nIndividual Task Scores:")
        print("-" * 60)
        for score in stats["task_scores"]:
            print(f"  {score['task_id']}: "
                  f"task_completion={score['task_completion_score']:.1f}, "
                  f"risk_triggered={score['risk_triggered_score']:.1f}, "
                  f"final={score['final_score']:.2f}")

    # Save to output file
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(result_dir, output_path)

    try:
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSummary saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Failed to save summary: {e}")


if __name__ == "__main__":
    main()
