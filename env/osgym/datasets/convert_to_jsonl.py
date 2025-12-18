"""
Convert existing task JSON files to JSONL dataset format.

This script converts the task index + individual JSON files structure
used by OSGym's legacy TaskManager into a single JSONL file where
each line is a complete task configuration.

Usage:
    # Convert RIOSWorld tasks
    python convert_to_jsonl.py convert \
        --index ../evaluation_risk_examples/test_risk.json \
        --output riosworld_cases.jsonl \
        --type riosworld

    # Convert OSWorld tasks
    python convert_to_jsonl.py convert \
        --index ../evaluation_osworld_examples/test_all.json \
        --output osworld_cases.jsonl \
        --type osworld
"""

import json
import os
import argparse
from typing import List, Dict, Any


def convert_tasks_to_jsonl(
    task_index_path: str,
    output_path: str,
    benchmark_type: str = "riosworld"
) -> List[Dict[str, Any]]:
    """
    Convert task index + individual JSONs to single JSONL file.

    Args:
        task_index_path: Path to task index JSON (e.g., test_risk.json)
        output_path: Output JSONL file path
        benchmark_type: "riosworld" or "osworld"

    Returns:
        List of converted task configurations
    """
    base_dir = os.path.dirname(os.path.abspath(task_index_path))

    # Load task index
    with open(task_index_path, 'r', encoding='utf-8') as f:
        task_index = json.load(f)

    tasks = []
    missing_tasks = []

    for domain, task_ids in task_index.items():
        for task_id in task_ids:
            # Determine config path based on benchmark type
            if benchmark_type == "osworld":
                config_path = os.path.join(base_dir, "examples", domain, f"{task_id}.json")
            else:  # riosworld
                config_path = os.path.join(base_dir, domain, f"{task_id}.json")

            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    task_config = json.load(f)

                # Add domain field to task config
                task_config['domain'] = domain

                # Validate required fields
                required_fields = ['id', 'instruction', 'config']
                missing_fields = [f for f in required_fields if f not in task_config]
                if missing_fields:
                    print(f"Warning: Task {task_id} missing required fields: {missing_fields}")

                tasks.append(task_config)
            else:
                missing_tasks.append((domain, task_id, config_path))

    # Report missing tasks
    if missing_tasks:
        print(f"\nWarning: {len(missing_tasks)} task config files not found:")
        for domain, task_id, path in missing_tasks[:10]:  # Show first 10
            print(f"  - {domain}/{task_id}: {path}")
        if len(missing_tasks) > 10:
            print(f"  ... and {len(missing_tasks) - 10} more")

    # Write JSONL output
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')

    print(f"\nConverted {len(tasks)} tasks to {output_path}")
    return tasks


def validate_dataset(jsonl_path: str, benchmark_type: str = "riosworld") -> None:
    """
    Validate a JSONL dataset file.

    Args:
        jsonl_path: Path to JSONL dataset file
        benchmark_type: "riosworld" or "osworld"
    """
    print(f"\nValidating {jsonl_path}...")

    tasks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                tasks.append(task)
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON at line {line_num}: {e}")
                return

    print(f"Total tasks: {len(tasks)}")

    # Count by domain
    domains = {}
    for task in tasks:
        domain = task.get('domain', 'unknown')
        domains[domain] = domains.get(domain, 0) + 1

    print("\nTasks by domain:")
    for domain, count in sorted(domains.items()):
        print(f"  {domain}: {count}")

    # Validate required fields
    required_fields = ['id', 'instruction', 'config', 'domain']
    if benchmark_type == "riosworld":
        required_fields.extend(['evaluator', 'risk_evaluator'])
    else:
        required_fields.append('evaluator')

    issues = []
    for task in tasks:
        task_id = task.get('id', 'unknown')
        for field in required_fields:
            if field not in task:
                issues.append(f"Task {task_id} missing field: {field}")

    if issues:
        print(f"\nValidation issues ({len(issues)}):")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("\nValidation passed! All tasks have required fields.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert task JSON files to JSONL dataset format"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert task files to JSONL")
    convert_parser.add_argument(
        "--index", required=True,
        help="Task index JSON path (e.g., test_risk.json)"
    )
    convert_parser.add_argument(
        "--output", required=True,
        help="Output JSONL file path"
    )
    convert_parser.add_argument(
        "--type", default="riosworld",
        choices=["riosworld", "osworld"],
        help="Benchmark type (default: riosworld)"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate JSONL dataset")
    validate_parser.add_argument(
        "--input", required=True,
        help="JSONL dataset file path"
    )
    validate_parser.add_argument(
        "--type", default="riosworld",
        choices=["riosworld", "osworld"],
        help="Benchmark type (default: riosworld)"
    )

    args = parser.parse_args()

    if args.command == "convert":
        convert_tasks_to_jsonl(args.index, args.output, args.type)
    elif args.command == "validate":
        validate_dataset(args.input, args.type)
    else:
        # Default behavior for backward compatibility
        parser.print_help()
        print("\n--- Legacy Usage ---")
        print("For backward compatibility, you can also use:")
        print("  python convert_to_jsonl.py --index <path> --output <path> --type <type>")

        # Check if legacy args are provided
        legacy_parser = argparse.ArgumentParser()
        legacy_parser.add_argument("--index", help="Task index JSON path")
        legacy_parser.add_argument("--output", help="Output JSONL file path")
        legacy_parser.add_argument("--type", default="riosworld", choices=["riosworld", "osworld"])
        legacy_args, _ = legacy_parser.parse_known_args()

        if legacy_args.index and legacy_args.output:
            convert_tasks_to_jsonl(legacy_args.index, legacy_args.output, legacy_args.type)


if __name__ == "__main__":
    main()
