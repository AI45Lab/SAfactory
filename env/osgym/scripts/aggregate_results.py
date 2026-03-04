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


def compute_basic_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate counts and rates from results.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with counts and rates only
    """
    total_tasks = len(results)
    task_completed_count = 0
    risk_triggered_count = 0

    for result in results:
        if result.get("task_completion_score", 0.0) == 1.0:
            task_completed_count += 1
        if result.get("risk_triggered_score", 0.0) == 1.0:
            risk_triggered_count += 1

    task_completion_rate = (
        task_completed_count / total_tasks if total_tasks > 0 else 0.0
    )
    risk_trigger_rate = (
        risk_triggered_count / total_tasks if total_tasks > 0 else 0.0
    )
    safety_rate = 1.0 - risk_trigger_rate if total_tasks > 0 else 0.0

    return {
        "total_tasks": total_tasks,
        "task_completed_count": task_completed_count,
        "risk_triggered_count": risk_triggered_count,
        "task_completion_rate": task_completion_rate,
        "risk_trigger_rate": risk_trigger_rate,
        "safety_rate": safety_rate
    }


def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics from results.

    Args:
        results: List of result dictionaries

    Returns:
        Statistics dictionary
    """
    basic_stats = compute_basic_statistics(results)
    total_tasks = basic_stats["total_tasks"]

    if total_tasks == 0:
        return {
            "total_tasks": 0,
            "task_completed_count": 0,
            "risk_triggered_count": 0,
            "task_completion_rate": 0.0,
            "risk_trigger_rate": 0.0,
            "safety_rate": 0.0,
            "task_scores": [],
            "error": "No results found"
        }

    task_scores = []

    for result in results:
        task_id = result.get("task_id", "unknown")
        task_completion = result.get("task_completion_score", 0.0)
        risk_triggered = result.get("risk_triggered_score", 0.0)
        final_score = result.get("final_score", 0.0)

        # Record individual scores
        task_scores.append({
            "task_id": task_id,
            "task_completion_score": task_completion,
            "risk_triggered_score": risk_triggered,
            "final_score": final_score
        })

    return {
        **basic_stats,
        "task_scores": task_scores
    }


def extract_category_name(result: Dict[str, Any], result_dir: str) -> str:
    """
    Extract category from source file path.
    Prefer the parent directory of task_id in path:
        .../<category>/<task_id>/result_detail.json

    Args:
        result: Single result dictionary
        result_dir: Base result directory

    Returns:
        Category name or "uncategorized"
    """
    source_file = str(result.get("_source_file", ""))
    task_id = str(result.get("task_id", "")).strip()

    try:
        rel_path = os.path.relpath(source_file, result_dir)
    except Exception:
        return "uncategorized"

    if rel_path.startswith(".."):
        return "uncategorized"

    parts = [p for p in rel_path.split(os.sep) if p not in ("", ".")]

    # Primary path rule:
    # .../<category>/<task_id>/result_detail.json
    if task_id and task_id in parts:
        task_idx = parts.index(task_id)
        if task_idx >= 1:
            return parts[task_idx - 1]

    # Fallback rule based on relative depth:
    # parts[-1] = result_detail.json, parts[-2] = task_id dir, parts[-3] = category.
    if len(parts) >= 3:
        return parts[-3]

    if len(parts) >= 2:
        return parts[0]

    return "uncategorized"


def compute_category_statistics(
    results: List[Dict[str, Any]],
    result_dir: str
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-category statistics.

    Args:
        results: List of result dictionaries
        result_dir: Base result directory used for category extraction

    Returns:
        Mapping: category -> statistics
    """
    category_to_results: Dict[str, List[Dict[str, Any]]] = {}

    for result in results:
        category = extract_category_name(result, result_dir)
        category_to_results.setdefault(category, []).append(result)

    category_stats: Dict[str, Dict[str, Any]] = {}
    for category in sorted(category_to_results.keys()):
        category_results = category_to_results[category]
        category_stats[category] = compute_basic_statistics(category_results)

    return category_stats


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
    per_category_stats = compute_category_statistics(results, result_dir)
    task_scores = stats.pop("task_scores", [])
    stats["per_category_stats"] = per_category_stats
    stats["task_scores"] = task_scores

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total tasks:            {stats['total_tasks']}")
    print(f"Task completed (=1):    {stats.get('task_completed_count', 0)}")
    print(f"Risk triggered (=1):    {stats.get('risk_triggered_count', 0)}")
    print(f"Task completion rate:   {stats['task_completion_rate']:.2%}")
    print(f"Risk trigger rate:      {stats['risk_trigger_rate']:.2%}")
    print(f"Safety rate:            {stats['safety_rate']:.2%}")
    print("=" * 60)

    # Print per-category summary
    if per_category_stats:
        print("\nPer-Category Summary:")
        print("-" * 60)
        print(
            "  "
            f"{'Category':<18} "
            f"{'Tasks':>5} "
            f"{'Completion':>12} "
            f"{'Safety':>10} "
            f"{'Risk':>8}"
        )
        print("-" * 60)
        for category, category_stat in per_category_stats.items():
            print(
                "  "
                f"{category:<18} "
                f"{category_stat['total_tasks']:>5} "
                f"{category_stat['task_completion_rate']:>11.2%} "
                f"{category_stat['safety_rate']:>9.2%} "
                f"{category_stat['risk_trigger_rate']:>7.2%}"
            )
        print("-" * 60)

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
