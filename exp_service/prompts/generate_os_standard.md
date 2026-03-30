You are an expert in analyzing desktop computer-use agent behavior and distilling reusable experiences.

## Agent Trajectories (OS Environment, Standard Evaluation)

{trajectories}

## Your Task

These trajectories come from agents operating in **desktop OS environments** (e.g., Ubuntu desktop, Windows) using GUI interaction tools (click, type, screenshot). The evaluation is **binary**: 1 = task completed, 0 = task not completed.

Analyze these successful trajectories and extract **generalizable OS interaction experiences** applicable across different desktop tasks.

## OS-Specific Guidance

Focus on extracting experiences about:
- **UI navigation patterns**: How to navigate menus, dialogs, windows systematically
- **State recognition**: How to identify the current UI state and what actions it affords
- **Obstacle handling**: Common interruptions (popups, dialogs, loading states) and how to handle them
- **Error recovery**: How to recover when an action doesn't produce the expected result
- **Efficiency principles**: Avoiding redundant actions, verifying before acting

## What Makes a Good OS Experience

**AVOID** (too specific):
> "Open LibreOffice Writer by double-clicking the icon at position (320, 180)"

**GOOD** (generalizable):
> "When opening an application from the desktop, locate the application icon visually, verify the label matches the target application, then double-click to launch"

The experience should apply to ANY desktop task, not just the observed one.

## Output Format

```json
{
  "experiences": [
    {
      "name": "experience_name_snake_case",
      "description": "One-sentence summary of the behavior pattern",
      "trigger": "The specific UI situation or task context that calls for this experience",
      "content": "# [Experience Title]\n\n## Core Principle\n[The generalizable rule]\n\n## When to Apply\n[UI situation description — specific enough to recognize, general enough to reuse]\n\n## How to Apply\n1. [Action step 1]\n2. [Action step 2]\n3. [Action step 3]\n\n## Common Pitfalls\n- [What to avoid]\n\n## Why It Works\n[Brief reasoning]"
    }
  ]
}
```

If no meaningful patterns exist: `{"no_extraction": true, "reason": "..."}`

## Requirements
- Content must NOT mention specific applications, filenames, or task descriptions
- Focus on **interaction patterns**, not task-specific procedures
- Extract at most **{max_exps}** experiences
- Only from trajectories with score = 1 (completed tasks)

Return JSON only.
