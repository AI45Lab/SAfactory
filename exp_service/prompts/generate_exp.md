You are an expert in analyzing AI agent behavior and distilling reusable decision-making experiences.

## Agent Trajectories (Standard Evaluation)

{trajectories}

## Your Task

These trajectories come from agents operating in **{env_type}** environments with **standard (binary) scoring**.
A score of 1 means the task was completed successfully.

Analyze these trajectories and extract **generalizable experiences** that can help future agents handle similar situations more effectively.

## What Makes a Good Experience

An experience must be:
- **Generalizable**: Describes a broad pattern of behavior, NOT a solution specific to one task instance
- **Actionable**: Gives clear guidance on WHAT to do and WHEN
- **Principled**: Explains the underlying reason, not just the steps
- **Concise**: Avoids mentioning specific file names, task IDs, or episode-specific details

BAD example (too specific):
> "Click on the 'Open' button at coordinate (450, 320) to open the document viewer"

GOOD example (generalizable):
> "When a dialog box blocks further interaction, identify and click the appropriate dismissal button (OK/Cancel/Close) before attempting the primary task action"

## Output Format

If no meaningful experience can be extracted:
```json
{"no_extraction": true, "reason": "brief explanation"}
```

Otherwise:
```json
{
  "experiences": [
    {
      "name": "experience_name_snake_case",
      "description": "One-sentence description of the experience (what behavior pattern it captures)",
      "trigger": "Describe the situation/context that calls for this experience",
      "content": "# Experience Title\n\n## Core Principle\n[The underlying rule or insight]\n\n## When to Apply\n[Situation description]\n\n## How to Apply\n1. [Step 1]\n2. [Step 2]\n\n## Why It Works\n[Brief explanation of the reasoning]"
    }
  ]
}
```

## Requirements

- Each experience must be applicable across **multiple different task instances**, not just the observed ones
- Name: English snake_case, 2-5 words (e.g., `dismiss_blocking_dialogs`)
- Do NOT include specific coordinates, file names, application names, or task-specific details in the content
- Extract at most **{max_exps}** experiences
- Only extract from trajectories that succeeded (score = 1)

Return JSON only, no extra text.
