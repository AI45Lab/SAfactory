"""Environment-specific hook. Replace the guide's greeting example with one native case.

This example is runnable for scaffold verification; it is not a benchmark integration.
Use only `task` (the current dataset row), never loop over the dataset here.
For a native subprocess, capture stdout or redirect it to stderr explicitly.
Pass session_url and request['model'] to the harness's model client configuration.
Return JSON metrics (including native output paths when available) and a step count.
"""
import json
from urllib.request import Request, urlopen


def run_case(request, task, session_url):
    prompt = task.get("prompt") or task.get("question") or "Say hello from Safactory."
    payload = {
        "model": request["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": request.get("temperature", 0.3),
    }
    call = Request(
        f"{session_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(call, timeout=300) as response:
        body = json.load(response)
    return {"answer": body["choices"][0]["message"].get("content", "")}, 1
