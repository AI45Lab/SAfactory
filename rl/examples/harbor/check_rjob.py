#!/usr/bin/env python3
"""Query RJob cluster for harbor jobs: status + logs. Run on the training machine."""
import sys, yaml, traceback

CFG = "/mnt/shared-storage-user/leishanzhe/repo/SAfactory/config.yaml"

def main():
    with open(CFG) as f:
        cfg = yaml.safe_load(f) or {}
    rjob = cfg.get("rjob", {})
    try:
        from brainpp.rjob import RJobClient
    except ImportError:
        print("ERROR: brainpp.rjob not installed in this env"); sys.exit(1)

    client = RJobClient(
        cluster_entry=rjob.get("cluster_entry"),
        namespace=rjob.get("namespace"),
        access_key=rjob.get("access_key"),
        secret_key=rjob.get("secret_key"),
        verifyssl=bool(rjob.get("verifyssl", True)),
        retries=int(rjob.get("retries", 3) or 0),
    )
    print(f"RJobClient -> {rjob.get('cluster_entry')} ns={rjob.get('namespace')}")

    # list all jobs, filter harbor
    try:
        jobs = client.list([])
    except Exception:
        # some versions need a name filter; try with prefix
        try:
            jobs = client.list("harbor")
        except Exception:
            jobs = client.list("safactory")
    print(f"\n=== total jobs returned: {len(jobs) if hasattr(jobs,'__len__') else '?'} ===")

    harbor_jobs = []
    for j in (jobs or []):
        name = getattr(j, "name", None) or (j.get("name") if isinstance(j, dict) else str(j))
        if "harbor" in str(name).lower() or "safactory" in str(name).lower():
            harbor_jobs.append(j)

    print(f"harbor/safactory jobs: {len(harbor_jobs)}")
    if not harbor_jobs:
        print("No harbor jobs found in cluster. Launcher may not be submitting, or jobs already cleaned.")
        # show first few of any jobs
        print("\n=== first 10 of ALL jobs ===")
        for j in (jobs or [])[:10]:
            print(" ", getattr(j, "name", None) or (j.get("name") if isinstance(j, dict) else j))
        return

    print("\n=== harbor job statuses ===")
    for j in harbor_jobs[:30]:
        name = getattr(j, "name", None) or (j.get("name") if isinstance(j, dict) else str(j))
        status = getattr(j, "status", None) or (j.get("status") if isinstance(j, dict) else "?")
        print(f"  {name}  status={status}")

    # fetch logs for first 2 non-succeeded
    print("\n=== logs for first 2 harbor jobs ===")
    for j in harbor_jobs[:2]:
        name = getattr(j, "name", None) or (j.get("name") if isinstance(j, dict) else str(j))
        print(f"\n----- LOGS: {name} -----")
        try:
            raw = client.logs_rjob(name)
            txt = raw if isinstance(raw, str) else getattr(raw, "text", None) or str(raw)
            print(txt[-4000:] if len(txt) > 4000 else txt)
        except Exception as e:
            print(f"logs_rjob failed: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
