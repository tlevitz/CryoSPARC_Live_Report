#!/usr/bin/env python3
# coding: utf-8

import os
import re
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from bson import decode_file_iter
except Exception:
    decode_file_iter = None


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_date_to_dt(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    if isinstance(x, dict) and "$date" in x:
        s = x["$date"]
        if isinstance(s, str):
            s = s.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(s)
            except Exception:
                return None
    return None


def epoch_to_dt(x: Any) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(float(x), tz=timezone.utc)
    except Exception:
        return None


def fmt_dt(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    try:
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def fmt_num(x: Any, ndp: int = 2) -> str:
    try:
        return f"{float(x):.{ndp}f}"
    except Exception:
        return "" if x is None else str(x)


def fmt_pct(num: float, den: float, ndp: int = 1) -> str:
    try:
        if den == 0:
            return ""
        return f"{100.0 * float(num) / float(den):.{ndp}f}%"
    except Exception:
        return ""


def safe_first(x: Any, default=None):
    if isinstance(x, list) and x:
        return x[0]
    return default


def nested_get(d: Any, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def ensure_path(project_dir: str, p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    if os.path.isabs(p):
        return p if os.path.exists(p) else None
    cand = os.path.join(project_dir, p)
    return cand if os.path.exists(cand) else None


def rel_or_abs_path_from_nested(project_dir: str, d: dict, *keys) -> Optional[str]:
    val = nested_get(d, *keys)
    if isinstance(val, dict):
        val = val.get("path")
    if isinstance(val, list):
        val = safe_first(val)
    if isinstance(val, str):
        return ensure_path(project_dir, val)
    return None


def load_exposures_bson(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    if decode_file_iter is None:
        raise RuntimeError("bson.decode_file_iter unavailable; use pymongo bson")
    exposures = []
    with open(path, "rb") as f:
        for doc in decode_file_iter(f):
            if isinstance(doc, dict) and isinstance(doc.get("exposures"), list):
                exposures.extend(doc["exposures"])
    return exposures

def iter_exposures_bson(path: str):
    if not os.path.isfile(path):
        return
    if decode_file_iter is None:
        raise RuntimeError("bson.decode_file_iter unavailable; use pymongo bson")
    with open(path, "rb") as f:
        for doc in decode_file_iter(f):
            exps = doc.get("exposures")
            if isinstance(exps, list):
                yield from exps

def find_live_workspace(workspaces: List[dict], session_name: str) -> dict:
    live = [w for w in workspaces if w.get("workspace_type") == "live"]
    for w in live:
        if w.get("session_uid") == session_name or w.get("session_dir") == session_name:
            return w
    raise ValueError(f"No live workspace found for session '{session_name}'")


def list_job_dirs(project_dir: str) -> List[str]:
    names = []
    for n in os.listdir(project_dir):
        if re.fullmatch(r"J\d+", n) and os.path.isdir(os.path.join(project_dir, n)):
            names.append(n)
    return sorted(names, key=lambda s: int(s[1:]))


def load_job_json(project_dir: str, job_uid: str) -> Optional[dict]:
    p = os.path.join(project_dir, job_uid, "job.json")
    if not os.path.isfile(p):
        return None
    try:
        return read_json(p)
    except Exception:
        return None


def job_type_of(job: dict) -> str:
    return job.get("job_type") or job.get("type") or nested_get(job, "spec", "type") or ""


def job_dt(job: dict) -> Optional[datetime]:
    for key in ("completed_at", "updated_at", "last_updated", "created_at"):
        dt = json_date_to_dt(job.get(key))
        if dt:
            return dt
    return None
    
def load_class2d_job_context(project_dir: str, ws: dict, override_job_uid: Optional[str] = None) -> dict:
    workspace_job_uid = str(ws.get("phase2_class2D_job") or "").strip() or None

    if override_job_uid:
        chosen_job_uid = str(override_job_uid).strip()
        source = "override"
    elif workspace_job_uid:
        chosen_job_uid = workspace_job_uid
        source = "workspace"
    else:
        chosen_job_uid = find_latest_class2d_job(project_dir, ws.get("uid"), None)
        source = "auto"

    if not chosen_job_uid:
        return {
            "job_uid": None,
            "workspace_job_uid": workspace_job_uid,
            "source": source,
            "job": None,
            "matches_workspace": False,
        }

    job = load_job_json(project_dir, chosen_job_uid)
    if not job:
        raise ValueError(f"Could not load job.json for class job {chosen_job_uid}")

    return {
        "job_uid": chosen_job_uid,
        "workspace_job_uid": workspace_job_uid,
        "source": source,
        "job": job,
        "matches_workspace": (chosen_job_uid == workspace_job_uid),
    }

def find_latest_classavg_mrc(job_dir: str, job_uid: str) -> Optional[str]:
    pat = re.compile(rf"^{re.escape(job_uid)}_(\d+)_class_averages\.mrc$")
    best = None
    best_iter = -1
    for name in os.listdir(job_dir):
        m = pat.match(name)
        if m:
            it = int(m.group(1))
            if it > best_iter:
                best_iter = it
                best = os.path.join(job_dir, name)
    if best:
        return best

    fallback = os.path.join(job_dir, f"{job_uid}_class_averages.mrc")
    if os.path.isfile(fallback):
        return fallback
    return None

