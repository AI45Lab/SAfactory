import re
import pandas as pd
from dabstep_benchmark.evaluation.scorer import question_scorer


def format_error(msg):
    return f"<p style='color: red; font-size: 20px; text-align: center;'>{msg}</p>"

def format_warning(msg):
    return f"<p style='color: orange; font-size: 20px; text-align: center;'>{msg}</p>"

def format_log(msg):
    return f"<p style='color: green; font-size: 20px; text-align: center;'>{msg}</p>"

def model_hyperlink(link, model_name):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'

def is_valid_https_url(url):
    pattern = re.compile(
        r'^https://'  # URL must start with 'https://'
        r'(?!10(?:\.\d{1,3}){3})'  # Exclude private IP 10.x.x.x
        r'(?!127(?:\.\d{1,3}){3})'  # Exclude loopback IP 127.x.x.x
        r'(?!169\.254(?:\.\d{1,3}){2})'  # Exclude link-local IP 169.254.x.x
        r'(?!192\.168(?:\.\d{1,3}){2})'  # Exclude private IP 192.168.x.x
        r'(?!172\.(?:1[6-9]|2[0-9]|3[0-1])(?:\.\d{1,3}){2})'  # Exclude private IP 172.16.x.x - 172.31.x.x
        r'(?:(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})'  # Match domain name
        r'(?::\d{2,5})?'  # Optional port
        r'(?:/[^\s]*)?$',  # Optional path
        re.IGNORECASE
    )
    return re.match(pattern, url) is not None


def evaluate(agent_answers: pd.DataFrame, tasks_with_gt: pd.DataFrame, submission_id: str = ""):
    task_scores = []
    for index, row in tasks_with_gt.iterrows():
          correct_answer = row["answer"]
          level = str(row["level"])
          task_id = str(row["task_id"])

          if task_id not in agent_answers["task_id"].values:
              raise KeyError(f"Task ID: {task_id} not found. Are you sure you submitted the correct file?")

          agent_answer = agent_answers.loc[agent_answers.task_id == task_id, "agent_answer"].values[0]
          # num_steps = agent_answers.loc[agent_answers.task_id == task_id, "num_steps"].values[0]
          score = question_scorer(agent_answer, correct_answer)

          task_scores.append(
              {
                  "submission_id": submission_id,
                  "task_id": task_id,
                  "score": score,
                  "level": level,
                  "agent_answer": agent_answer,
                  # "num_steps": num_steps,
              }
          )

    return task_scores