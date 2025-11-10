import re
from typing import Tuple

import gradio as gr
import json
import datetime
from email.utils import parseaddr

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

from dabstep_benchmark.utils import format_log, format_error, format_warning, is_valid_https_url, evaluate

OWNER = "adyen"

HF_API = HfApi()
HF_LEADERBOARD = f"{OWNER}/DABstep"
HF_DATASET_PATH = f"{OWNER}/DABstep"
HF_INTERNAL_DATASET_PATH = f"{OWNER}/DABstep-internal"
HF_DATASET_CONFIGS = [
    "tasks",
    "submissions",
    "task_scores"
]
DATASETS = {}

def refresh(only_leaderboard: bool = False):
    if only_leaderboard:
        for config_name in ["task_scores", "submissions"]:
            DATASETS[f"{config_name}"] = load_dataset(
                path=HF_DATASET_PATH,
                name=config_name,
                split="default",
            )
            print(f"Downloaded {HF_DATASET_PATH}/{config_name}")

    else:
        for config_name in HF_DATASET_CONFIGS:
            DATASETS[f"{config_name}"] = load_dataset(
                path=HF_DATASET_PATH,
                name=config_name,
                split="default",
            )
            print(f"Downloaded {HF_DATASET_PATH}/{config_name}")

        DATASETS["internal_tasks"] = load_dataset(
            path=HF_INTERNAL_DATASET_PATH,
            name="tasks",
            split="default",
        )
        print(f"Downloaded {HF_INTERNAL_DATASET_PATH}/tasks")
        DATASETS["contact_info"] = load_dataset(
            path=HF_INTERNAL_DATASET_PATH,
            name="contact_info",
            split="default",
        )
        print(f"Downloaded {HF_INTERNAL_DATASET_PATH}/contact_info")

    return generate_leaderboard_df()


def validate_submission(submission_df: pd.DataFrame):
    # mandatory_columns = ["agent_answer", "task_id", "num_steps"]
    mandatory_columns = ["agent_answer", "task_id"]
    expected_columns = [*mandatory_columns, "reasoning_trace"]

    # Check for missing mandatory columns
    missing_columns = [col for col in mandatory_columns if col not in submission_df.columns]
    if missing_columns:
        return format_error(f"Missing mandatory columns: {', '.join(missing_columns)}")

    # Check for unexpected columns
    unexpected_columns = [col for col in submission_df.columns if col not in expected_columns]
    if unexpected_columns:
        return format_error(f"Unexpected columns: {', '.join(unexpected_columns)}")

    # Check for NaN values in any column
    if submission_df.isnull().values.any():
        return format_error("Submission contains NaN values. Please ensure no missing data.")

    # Check if all columns are of string type
    non_string_columns = [col for col in submission_df.columns if submission_df[col].dtype != 'object']
    if non_string_columns:
        return format_error(f"Columns with non-string data type: {', '.join(non_string_columns)}")

    return None  # No errors

def process_submission(
        split: str,
        agent_name: str,
        model_family: str,
        repo_url: str,
        path_to_file: str,
        organisation: str,
        mail: str,
        profile: gr.OAuthProfile,
):
    if agent_name == "":
        return format_warning("Please provide an agent name")
    if organisation == "":
        return format_warning("Please provide an organisation")
    if mail == "":
        return format_warning("Please provide an email")
    if model_family == "":
        return format_warning("Please provide a model family")

    allowed_pattern = re.compile(r'^[a-zA-Z0-9 _.-]+$')
    if not allowed_pattern.match(agent_name):
        return format_warning(
            f"{agent_name=} can only contain alphanumeric characters, spaces, dashes (-), and underscores (_)")

    if not allowed_pattern.match(organisation):
        return format_warning(
            f"{organisation=} can only contain alphanumeric characters, spaces, dashes (-), and underscores (_)")


    # very basic email parsing
    _, parsed_mail = parseaddr(mail)
    if not "@" in parsed_mail:
        return format_warning("Please provide a valid email address.")

    if repo_url != "" and not is_valid_https_url(repo_url):
        return format_warning("If you provide a URL it must be a valid one. You can also leave it empty")

    # submission file validation
    if path_to_file == None:
        return format_warning("Please attach a file.")
    submission_path = path_to_file.name
    try:
        submission_df = pd.read_json(submission_path, lines=True, dtype=str)
        validation_error = validate_submission(submission_df)
        if validation_error:
            return validation_error
    except Exception as exc:
        return format_error(f"Submission file is incorrectly formatted. Please fix it and resubmit your file. {str(exc)}")


    print(f"Processing submission_id={organisation}-{agent_name}...")
    gr.Info(f"Processing submission of {agent_name}...")
    refresh(only_leaderboard=False)
    submissions_df = DATASETS["submissions"].to_pandas()
    contact_info_df = DATASETS["contact_info"].to_pandas()
    internal_tasks_df = DATASETS["internal_tasks"].to_pandas()


    # check if this agent already was submitted
    submission_id = f"{organisation}-{agent_name}"
    if submission_id in submissions_df['submission_id'].values:
        return format_warning(f"This {submission_id} pair has been already submitted.")

    # process submission
    submission_df["submission_id"] = submission_id
    submission_df["agent_name"] = agent_name
    submission_df["model_family"] = model_family
    submission_df["organisation"] = f"{organisation} | user {profile.username}"
    submission_df["repo_url"] = repo_url
    submission_df["date"] = datetime.date.today().strftime("%d-%m-%Y")
    submission_df["validated"] = False #unvalidated by default

    # add empty reasoning trace if one is not provided to not break schema of datasets
    if "reasoning_trace" not in submission_df.columns:
        submission_df["reasoning_trace"] = ""

    # overwrite submission
    submission_df.to_json(submission_path, orient="records", lines=True)

    try:
        task_scores = evaluate(
            agent_answers=submission_df,
            tasks_with_gt=internal_tasks_df,
            submission_id=submission_id
        )
    except KeyError as exc:
        return format_error(str(exc))


    # save submitted file once evaluation has run correctly
    filename_id = f"v1__{organisation}-{agent_name}__{datetime.datetime.today().strftime('%d-%m-%Y')}"
    path_in_repo = f"data/submissions/{filename_id}.jsonl"
    HF_API.upload_file(
        repo_id=HF_DATASET_PATH,
        path_or_fileobj=submission_path,
        path_in_repo=path_in_repo,
        repo_type="dataset",
    )
    print(f"[submission_id={organisation}-{agent_name}] Pushed submission to {HF_DATASET_PATH}/{path_in_repo} !")

    # write scores to disk
    with open(f"data/task_scores/{filename_id}.jsonl", "w") as f:
        for score in task_scores:
            f.write(json.dumps(score) + "\n")

    # upload scores to hub dataset
    path_in_repo = f"data/task_scores/{filename_id}.jsonl"
    HF_API.upload_file(
        repo_id=HF_DATASET_PATH,
        path_or_fileobj=f"data/task_scores/{filename_id}.jsonl",
        path_in_repo=path_in_repo,
        repo_type="dataset",
    )
    print(f"[submission_id={organisation}-{agent_name}] Pushed task_scores to {HF_DATASET_PATH}/{path_in_repo} !")

    # if we already have this email dont save its metadata
    if mail not in contact_info_df["mail"].values:
        contact_info = {
            "submission_id": submission_id,
            "agent_name": agent_name,
            "model_family": model_family,
            "repo_url": repo_url,
            "organisation": organisation,
            "mail": mail,
            "date": datetime.date.today().strftime("%d-%m-%Y"),
        }
        contact_info_df = pd.concat([contact_info_df, pd.DataFrame([contact_info])], ignore_index=True)
        contact_info_df.to_json("contact_info.jsonl", orient="records", lines=True)

        HF_API.upload_file(
            repo_id=HF_INTERNAL_DATASET_PATH,
            path_or_fileobj="contact_info.jsonl",
            path_in_repo="contact_info.jsonl",
            repo_type="dataset",
        )
        print(f"[submission_id={organisation}-{agent_name}] Pushed contact_info to {HF_INTERNAL_DATASET_PATH}/contact_info.jsonl !")


    return format_log(
        f"""
        Agent {agent_name} submitted by {organisation} successfully.
        Please refresh the leaderboard to see your score displayed.
""")

def generate_leaderboard_df() -> Tuple[pd.DataFrame, pd.DataFrame]:
    task_scores_df = DATASETS["task_scores"].to_pandas()
    submissions_df = DATASETS["submissions"].to_pandas()

    # get metadata of each submission_id
    submissions_df = (
        submissions_df.groupby("submission_id")
        .first()
        .reset_index()[
            [
                "submission_id",
                "agent_name",
                "model_family",
                "organisation",
                "repo_url",
                "date",
                "validated"
            ]
        ]
    )

    # make num_steps a number
    # task_scores_df["num_steps"] = pd.to_numeric(task_scores_df["num_steps"], errors="coerce")

    # group scores per submission
    leaderboard_df = (
        task_scores_df.groupby(["submission_id", "level"])
        .agg(
            avg_score=("score", "mean"),
            # avg_num_steps=("num_steps", "mean")
        )
        .reset_index()
    )

    # reshape
    # leaderboard_df = leaderboard_df.pivot(index="submission_id", columns="level", values=["avg_score", "avg_num_steps"])
    leaderboard_df = leaderboard_df.pivot(index="submission_id", columns="level", values=["avg_score"])
    leaderboard_df.columns = [f"{metric}_lvl_{level}" for metric, level in leaderboard_df.columns]
    leaderboard_df = leaderboard_df.reset_index()

    # leaderboard_df["overall_avg_steps"] = (
    #         leaderboard_df.get("avg_num_steps_lvl_1", 0) +
    #         leaderboard_df.get("avg_num_steps_lvl_2", 0) +
    #         leaderboard_df.get("avg_num_steps_lvl_3", 0)
    # )
    # leaderboard_df["overall_avg_steps"] = leaderboard_df["overall_avg_steps"] / 3

    # join scores and submission metadata
    leaderboard_df = pd.merge(submissions_df, leaderboard_df, on="submission_id", how="inner")

    # renaming
    col_map = {
        "agent_name": "Agent",
        "avg_score_lvl_easy": "Easy Level Accuracy (%)",
        "avg_score_lvl_hard": "Hard Level Accuracy (%)",
        # "overall_avg_steps": "Overall Avg Reasoning Steps",
        # "avg_num_steps_lvl_1": "Level 1 Avg Reasoning Steps",
        # "avg_num_steps_lvl_2": "Level 2 Avg Reasoning Steps",
        # "avg_num_steps_lvl_3": "Level 3 Avg Reasoning Steps",
        "organisation": "Organization",
        "repo_url": "Repo URL",
        "model_family": "Model Family",
        "date": "Date",
        "validated": "validated"
    }
    col_order = [new_col_name for new_col_name in col_map.values()]
    leaderboard_df.rename(columns=col_map, inplace=True)
    leaderboard_df = leaderboard_df[col_order]

    # formatting
    # convert scores to %
    leaderboard_df["Easy Level Accuracy (%)"] = leaderboard_df["Easy Level Accuracy (%)"].apply(lambda x: round(x * 100, 2))
    leaderboard_df["Hard Level Accuracy (%)"] = leaderboard_df["Hard Level Accuracy (%)"].apply(lambda x: round(x * 100, 2))

    # make repo url clickable in markdown
    leaderboard_df["Repo URL"] = leaderboard_df["Repo URL"].apply(lambda x: f"[Link]({x})" if x != "" else x)

    # make agent name bold
    leaderboard_df["Agent"] = leaderboard_df["Agent"].apply(lambda x: f"**{x}**")

    # sort-by best score
    leaderboard_df.sort_values(
        by=["Hard Level Accuracy (%)", "Easy Level Accuracy (%)"],
        ascending=[False, False],
        inplace=True
    )

    validated_lb = leaderboard_df[leaderboard_df["validated"] == True].drop(columns=["validated"])
    unvalidated_lb = leaderboard_df[leaderboard_df["validated"] == False].drop(columns=["validated"])

    return validated_lb, unvalidated_lb
