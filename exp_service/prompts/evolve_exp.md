You are an expert at refining AI agent skills based on failure analysis.

## Existing Skill

{existing_skill}

## Recent Failure Trajectories

{failure_trajectories}

## Task

The existing skill has a low success rate ({success_rate:.2f}). Analyze the failure trajectories to understand why and produce an **improved version** of the skill.

Identify:
1. Which part of the skill guidance leads to failure?
2. What edge cases or conditions are not covered?
3. What corrections or additions would improve success?

## Output Format

Return the improved skill as JSON:
```json
{
  "updated": true,
  "name": "same_name_as_original",
  "description": "Updated description",
  "trigger": "Refined trigger condition",
  "content": "# Improved skill content in markdown",
  "tags": ["existing", "tags"],
  "change_summary": "Brief description of what was changed and why"
}
```

If the skill is fundamentally sound and failures are due to other factors:
```json
{"updated": false, "reason": "explanation"}
```

Return JSON only.
