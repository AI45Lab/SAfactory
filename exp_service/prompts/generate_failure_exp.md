You are an expert in analyzing AI agent failure trajectories to extract generalizable failure lessons.

## Failed Agent Trajectories

{trajectories}

## Your Task

Analyze these failed trajectories (score = 0) and extract **generalizable failure lessons** — common behavioral mistakes and how future agents can avoid them.

Focus on:
1. **Mistaken assumptions**: What did the agent assume that turned out to be wrong?
2. **Premature actions**: Actions taken without sufficient verification
3. **Loop patterns**: Repetitive actions that didn't lead to progress
4. **State misreading**: Misinterpreting the current environment state
5. **Goal drift**: Drifting from the original task objective

## What Makes a Good Failure Lesson

**AVOID** (too specific):
> "Don't keep clicking on the Save button when it's grayed out"

**GOOD** (generalizable):
> "Before repeatedly attempting an action, verify that the target element is in an interactive state; if the action fails twice, re-assess the UI state before retrying"

## Output Format

```json
{
  "experiences": [
    {
      "name": "avoid_mistake_name",
      "description": "One-sentence description of the mistake pattern",
      "trigger": "When this mistake is likely to occur",
      "content": "# Lesson: [Title]\n\n## Mistake Pattern\n[Description of the behavioral error]\n\n## Root Cause\n[Why agents make this mistake]\n\n## Warning Signs\n[Observable indicators this mistake is about to happen]\n\n## How to Avoid\n1. [Corrective action 1]\n2. [Corrective action 2]\n\n## Underlying Principle\n[The general rule being violated]"
    }
  ]
}
```

If no clear patterns: `{"no_extraction": true, "reason": "..."}`

## Requirements
- Lessons must apply to many different task types, not just the observed one
- Name must start with `avoid_` to indicate a negative lesson
- Do NOT mention specific task names, applications, or file paths
- Extract at most **{max_exps}** lessons

Return JSON only.
