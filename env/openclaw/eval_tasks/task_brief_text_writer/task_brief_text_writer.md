---
eval_type: llm
score_max: 10
input_builder: final_answer_only
rubric:
  required_points: 50
  format_and_length: 25
  tone_and_clarity: 15
  constraint_compliance: 10
---

# Evaluation Task

Evaluate the target agent's final answer for the OpenClaw `task_brief_text_writer` case.

# LLM Judge Rubric

Score from 0 to 10 using the criteria below.

1. Required points (5 points): award 1 point for each required idea clearly present:
   - the evaluator keeps only LLM judge and agent judge
   - rule-based evaluation was removed
   - OpenClaw uses one lightweight LLM judge smoke-test case
   - the purpose is easier inspection and iteration
   - team members should give feedback when the scoring rubric is unclear
2. Format and length (2.5 points): the final answer is only the announcement text, written in Chinese, and is roughly 120-180 Chinese characters.
3. Tone and clarity (1.5 points): the text is professional, concise, and easy for an internal engineering team to understand.
4. Constraint compliance (1 point): the text avoids implementation details, file paths, Docker commands, code diffs, and unrelated claims.

Give a low score if the final answer is not an announcement, omits several required points, is mostly non-Chinese, or includes unrelated implementation details.
