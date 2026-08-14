from typing import Any, List, Optional, Tuple


async def load_env_lists(
    data_manager: Any,
    table_name: str = "job_environments",
    *,
    job_id: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Read env_name and env_id through DataManager and return two lists:
    (env_name_list, env_id_list)

    Schema expected:
      id INTEGER PK AUTOINCREMENT,
      env_name TEXT NOT NULL,
      env_id TEXT NOT NULL,
      env_param TEXT NULL,
      image TEXT NULL
    """
    if table_name != "job_environments":
        raise ValueError("DataManager only exposes the job_environments dataset")
    rows = await data_manager.list_environment_rows(job_id=job_id)
    env_names = [str(row.get("env_name") or "") for row in rows]
    env_ids = [str(row.get("env_id") or "") for row in rows]
    return env_names, env_ids
