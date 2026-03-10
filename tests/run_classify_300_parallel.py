#!/usr/bin/env python3
"""
Stress script: launches N parallel POSTs to /fiscal/classify.
- Uses tests/payload.json as request body.
- Per-request timeout: 50s.
- If response includes job_id (async payload), print it and track it.
- Maintains a loop polling job status and prints, every minute, how many are not completed.
- Prints a final summary of outcomes.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


BASE_URL = os.getenv("BLUEBILL_BASE_URL", "http://127.0.0.1:8001")
CLASSIFY_URL = f"{BASE_URL}/fiscal/classify"
STATUS_URL = f"{BASE_URL}/fiscal/status/{{job_id}}"
RESULT_URL = f"{BASE_URL}/fiscal/result/{{job_id}}"

PAYLOAD_PATH = Path(__file__).with_name("payload.json")
REQ_TIMEOUT = 50  # seconds per HTTP call
PARALLEL = 30


def load_payload():
    with PAYLOAD_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_job_status(job_id: str) -> str:
    """Query job status.

    Tries the explicit status endpoint first. If unavailable, falls back to
    consulting result endpoint semantics: 202 => running, 200 => completed.
    Returns a lowercase status string such as 'pending', 'running', 'completed',
    'failed', or 'unknown' when it cannot be determined.
    """
    # Try status endpoint
    try:
        r = requests.get(STATUS_URL.format(job_id=job_id), timeout=REQ_TIMEOUT)
        if r.status_code == 200:
            try:
                data = r.json()
                status = str(data.get("status", "unknown")).lower()
                return status
            except ValueError:
                return "unknown"
        # Some implementations return 202 for running
        if r.status_code == 202:
            return "running"
    except requests.RequestException:
        pass

    # Fallback to result endpoint
    try:
        r = requests.get(RESULT_URL.format(job_id=job_id), timeout=REQ_TIMEOUT)
        if r.status_code == 202:
            return "running"
        if r.status_code == 200:
            # Some backends may include final status in body
            try:
                data = r.json()
                status = str(data.get("status", "completed")).lower()
                return status
            except ValueError:
                return "completed"
        if r.status_code == 404:
            return "unknown"
    except requests.RequestException:
        pass

    return "unknown"


def get_job_error(job_id: str) -> str:
    """Try to obtain an error message for a failed job.

    Attempts to read it from the result endpoint body if available. Falls back
    to common fields like 'error', 'detail' or 'message'. Returns an empty
    string if not found.
    """
    try:
        r = requests.get(RESULT_URL.format(job_id=job_id), timeout=REQ_TIMEOUT)
        if r.status_code == 200:
            try:
                data = r.json()
                for key in ("error", "detail", "message"):
                    if key in data and isinstance(data[key], str):
                        return data[key]
            except ValueError:
                pass
    except requests.RequestException:
        pass
    return ""


def single_run(i: int, payload: dict) -> dict:
    # Basic retry for transient connection errors on classify
    last_err = None
    for _ in range(3):
        try:
            r = requests.post(CLASSIFY_URL, json=payload, timeout=REQ_TIMEOUT)
            break
        except requests.RequestException as e:
            last_err = e
            time.sleep(0.2)
    else:
        return {"index": i, "ok": False, "error": str(last_err)[:200]}
    # Two possible outcomes:
    # - 200 + FiscalClassificationResponse (dict without job_id)
    # - 200 + async payload (dict with job_id)
    if r.status_code != 200:
        return {"index": i, "ok": False, "status": r.status_code, "error": r.text[:200]}

    data = r.json()
    if isinstance(data, dict) and "job_id" in data:
        # Async path: just return job_id and print it now
        job_id = data["job_id"]
        print(f"Received job_id: {job_id}")
        return {"index": i, "ok": True, "async": True, "job_id": job_id}
    else:
        # Sync path (classification returned directly)
        return {"index": i, "ok": True, "async": False}


def main():
    payload = load_payload()
    results: List[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futures = [ex.submit(single_run, i, payload) for i in range(PARALLEL)]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"ok": False, "error": str(e)[:200]})
    submit_elapsed = time.time() - t0

    # Collect async jobs
    job_ids = [r["job_id"] for r in results if r.get("async") and r.get("job_id")]

    # Poll loop: check status and print every minute how many remain incomplete
    done_statuses = {"completed", "failed", "succeeded", "success", "error", "done"}
    statuses: Dict[str, str] = {jid: "unknown" for jid in job_ids}

    if job_ids:
        print(f"Tracking {len(job_ids)} async jobs for completion...")
        monitor_start = time.time()
        while True:
            remaining = 0
            for jid in job_ids:
                st = get_job_status(jid)
                statuses[jid] = st
                if st not in done_statuses:
                    remaining += 1

            print(f"Jobs not completed: {remaining}")
            if remaining == 0:
                break
            # Sleep about a minute between status prints
            time.sleep(60)
        monitor_elapsed = time.time() - monitor_start
    else:
        monitor_elapsed = 0.0

    total = len(results)
    ok = sum(1 for r in results if r.get("ok"))
    async_used = sum(1 for r in results if r.get("async"))
    failed = total - ok

    print("--- Summary ---")
    print(f"Submissions -> Total: {total}  OK: {ok}  Failed: {failed}  AsyncResponses: {async_used}")
    print(f"Submission elapsed: {submit_elapsed:.1f}s")

    if job_ids:
        # Summarize final job statuses
        counts: Dict[str, int] = {}
        for st in statuses.values():
            counts[st] = counts.get(st, 0) + 1
        completed_cnt = sum(counts.get(k, 0) for k in ("completed", "succeeded", "success", "done"))
        failed_cnt = sum(counts.get(k, 0) for k in ("failed", "error"))
        unknown_cnt = total_cnt = 0
        for k, v in counts.items():
            total_cnt += v
            if k not in ("completed", "succeeded", "success", "done", "failed", "error"):
                unknown_cnt += v
        print("Jobs summary -> "
              f"Total: {len(job_ids)}  Completed: {completed_cnt}  Failed: {failed_cnt}  Unknown: {unknown_cnt}")
        print(f"Monitoring elapsed: {monitor_elapsed:.1f}s")

        # Print all jobs that finished in error with their messages
        failed_jobs = [jid for jid, st in statuses.items() if st in ("failed", "error")]
        if failed_jobs:
            print("Failed jobs (job_id -> error):")
            for jid in failed_jobs:
                msg = get_job_error(jid)
                if msg:
                    print(f"- {jid}: {msg}")
                else:
                    print(f"- {jid}: (no error message)")

    # Print ALL failed submissions (sync errors)
    failed_submissions = [r for r in results if not r.get("ok")]
    if failed_submissions:
        print("Failed submissions (sync):")
        for s in failed_submissions:
            print(s)


if __name__ == "__main__":
    main()
