"""Environment-specific hook.  The only file with benchmark logic in it.

The runner owns the SAfactory protocol; this module owns only the native
invocation for the current dataset row.  Replace the greeting example below
with one native case — it exists only so the scaffold is runnable.

Rules (see env/prmeval/adapter.py for a full integration):
- Evaluate exactly the one row supplied as ``task``; never loop over the
  dataset here.  SAfactory schedules one episode per row.
- Do not reimplement the benchmark's case-solving or scoring logic; wrap it.
- Route model calls through ``session_url`` and use ``request['model']`` so
  telemetry lands in the Gateway session; never embed provider credentials.
- Capture or redirect native stdout (the runner guards its own stdout, but
  subprocess output should still go to a file or stderr explicitly).
- Return JSON-serializable metrics (include native output paths when
  available) and a nonnegative step count.
"""
import json
from urllib.request import Request, urlopen


def run_case(request, task, session_url):
    prompt = task.get("prompt") or task.get("question") or "Say hello from Safactory."
    payload = {
        "model": request["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": request.get("temperature", 0.0),
    }
    call = Request(
        f"{session_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(call, timeout=300) as response:
        body = json.load(response)
    return {"answer": body["choices"][0]["message"].get("content", "")}, 1
