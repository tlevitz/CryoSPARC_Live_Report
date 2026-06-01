#!/usr/bin/env python3
# coding: utf-8

import os
import hashlib
import random
import glob
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .io import (
    nested_get,
    rel_or_abs_path_from_nested,
    ensure_path,
    epoch_to_dt,
    json_date_to_dt,
    fmt_dt,
    fmt_num,
    fmt_pct,
)


def workspace_attribute_limits(ws: dict, name: str) -> Tuple[Optional[float], Optional[float]]:
    mins = []
    maxs = []
    for a in ws.get("attributes", []) or []:
        if a.get("name") != name:
            continue
        if a.get("min") is not None:
            mins.append(float(a["min"]))
        if a.get("max") is not None:
            maxs.append(float(a["max"]))
    return (mins[0] if mins else None, maxs[0] if maxs else None)


def get_threshold_block(ws: dict, key: str) -> dict:
    return nested_get(ws, "picking_thresholds", key) or {}


def get_picking_threshold(ws: dict, picker_name: str = "blob_ncc_score") -> Tuple[Optional[float], Optional[float], Optional[float]]:
    d = get_threshold_block(ws, picker_name)
    return d.get("min"), d.get("value"), d.get("max")

def get_class2d_param(class_job: Optional[dict], ws: dict, key: str):
    if class_job:
        v = nested_get(class_job, "spec", "params", key)
        if v is not None:
            return v
    v = nested_get(ws, "phase2_class2D_params_spec_used", key)
    if v is not None:
        return v
    return nested_get(ws, "phase2_class2D_params_spec", key)

def summarize_class2d_info(ws: dict) -> Dict[str, int]:
    info = ws.get("phase2_class2D_info") or []
    selected_classes = 0
    rejected_classes = 0
    selected_particles = 0
    rejected_particles = 0

    for row in info:
        is_selected = bool(row.get("selected"))
        total = int(row.get("num_particles_total") or 0)
        selected = int(row.get("num_particles_selected") or 0)

        if is_selected:
            selected_classes += 1
            selected_particles += selected if selected > 0 else total
        else:
            rejected_classes += 1
            rejected_particles += total

    return {
        "selected_classes": selected_classes,
        "rejected_classes": rejected_classes,
        "selected_particles": selected_particles,
        "rejected_particles": rejected_particles,
        "total_classes": len(info),
    }

def summarize_template_creation_info(ws: dict) -> Dict[str, int]:
    info = ws.get("template_creation_info") or []
    selected_classes = 0
    rejected_classes = 0

    for row in info:
        if bool(row.get("selected")):
            selected_classes += 1
        else:
            rejected_classes += 1

    return {
        "total_classes": len(info),
        "selected_classes": selected_classes,
        "rejected_classes": rejected_classes,
    }


def build_class2d_info_map(ws: dict) -> Dict[int, dict]:
    out = {}
    for row in ws.get("phase2_class2D_info") or []:
        try:
            idx = int(row.get("class_idx"))
        except Exception:
            continue
        out[idx] = row
    return out


def is_rejected_exp(exp: dict) -> bool:
    return bool(exp.get("manual_reject")) or bool(exp.get("threshold_reject"))


def is_failed_exp(exp: dict) -> bool:
    return bool(exp.get("failed"))


def is_accepted_exp(exp: dict) -> bool:
    return (
        not bool(exp.get("deleted"))
        and not bool(exp.get("test"))
        and not is_failed_exp(exp)
        and not is_rejected_exp(exp)
        and exp.get("stage") == "ready"
    )


def exposure_status_label(exp: dict) -> str:
    if is_failed_exp(exp):
        return "Failed"
    if is_rejected_exp(exp):
        return "Rejected"
    if is_accepted_exp(exp):
        return "Accepted"
    return str(exp.get("stage") or "Unknown")


def get_ctf_fit(exp: dict) -> Optional[float]:
    try:
        return float(nested_get(exp, "attributes", "ctf_fit_to_A"))
    except Exception:
        return None


def get_avg_defocus_A(exp: dict) -> Optional[float]:
    try:
        return float(nested_get(exp, "attributes", "average_defocus"))
    except Exception:
        return None


def get_avg_defocus_um(exp: dict) -> Optional[float]:
    v = get_avg_defocus_A(exp)
    return None if v is None else v / 10000.0


def get_total_motion(exp: dict) -> Optional[float]:
    try:
        return float(nested_get(exp, "attributes", "total_motion_dist"))
    except Exception:
        return None


def get_max_inframe_motion(exp: dict) -> Optional[float]:
    try:
        return float(nested_get(exp, "attributes", "max_intra_frame_motion"))
    except Exception:
        return None


def get_blob_picks(exp: dict) -> int:
    try:
        return int(nested_get(exp, "attributes", "total_blob_picks") or 0)
    except Exception:
        return 0


def get_extracted_particles(exp: dict) -> int:
    picker = str(exp.get("picker_type") or "")
    attrs = exp.get("attributes", {}) or {}

    if picker == "template":
        try:
            return int(
                attrs.get("total_extracted_particles_template")
                or attrs.get("total_extracted_particles")
                or 0
            )
        except Exception:
            return 0

    if picker == "blob":
        try:
            return int(
                attrs.get("total_extracted_particles_blob")
                or attrs.get("total_extracted_particles")
                or 0
            )
        except Exception:
            return 0

    try:
        return int(attrs.get("total_extracted_particles") or 0)
    except Exception:
        return 0


def exposure_start_dt(exp: dict):
    attr = exp.get("attributes", {})
    for k in ("found_at",):
        dt = epoch_to_dt(attr.get(k))
        if dt:
            return dt
    for k in ("discovered_at", "created_at"):
        val = exp.get(k)
        if isinstance(val, datetime):
            return val
    return None


def exposure_end_dt(exp: dict):
    attr = exp.get("attributes", {})
    for k in ("ready_at", "extract_at", "pick_at", "ctf_at", "motion_at"):
        dt = epoch_to_dt(attr.get(k))
        if dt:
            return dt
    val = exp.get("updated_at")
    if isinstance(val, datetime):
        return val
    return None


def choose_pick_cs_path(project_dir: str, exp: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (pick_cs_path, picker_used_for_overlay).
    Prefer the actual exposure picker_type, then fall back sensibly.
    """
    picker = str(exp.get("picker_type") or "").strip().lower()
    groups = exp.get("groups", {}) or {}

    search_order = []
    if picker:
        search_order.append(f"particle_{picker}")
    search_order += ["particle_template", "particle_blob", "particle_manual", "particle_deep"]

    seen = set()
    for key in search_order:
        if key in seen:
            continue
        seen.add(key)
        grp = groups.get(key)
        if not isinstance(grp, dict):
            continue
        p = grp.get("path")
        if not isinstance(p, str) or p in ("", "."):
            continue
        full = ensure_path(project_dir, p)
        if full and os.path.isfile(full):
            return full, key.replace("particle_", "")

    return None, picker or None

def choose_extracted_cs_path(project_dir: str, exp: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (extracted_cs_path, picker_used_for_extraction_overlay).

    Prefer the active exp["picker_type"], then fall back sensibly.
    """
    picker = str(exp.get("picker_type") or "").strip().lower()
    entries = nested_get(exp, "groups", "particle_extracted") or []
    if not isinstance(entries, list):
        entries = []

    picker_order = []
    if picker:
        picker_order.append(picker)
    picker_order += ["template", "blob", "manual", "deep"]

    seen = set()
    for desired_picker in picker_order:
        if desired_picker in seen:
            continue
        seen.add(desired_picker)

        for entry in entries:
            if str(entry.get("picker_type") or "").strip().lower() != desired_picker:
                continue
            p = entry.get("path")
            if not isinstance(p, str) or p in ("", "."):
                continue
            full = ensure_path(project_dir, p)
            if full and os.path.isfile(full):
                return full, desired_picker

    return None, picker or None


def choose_extracted_stack_path(project_dir: str, session_name: str, exp: dict, default_output_shape: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (particle_stack_mrc_path, picker_used_for_particles).

    We choose the particle_extracted entry whose picker_type matches exp["picker_type"]
    if possible, and infer the .mrc stack path from:
        S1/extract/<picker_type>/<output_shape>/<micrograph_basename>_particles.mrc
    """
    picker = str(exp.get("picker_type") or "").strip().lower()
    entries = nested_get(exp, "groups", "particle_extracted") or []
    if not isinstance(entries, list):
        entries = []

    dw = rel_or_abs_path_from_nested(project_dir, exp, "groups", "exposure", "micrograph_blob", "path")
    if not dw:
        return None, picker or None

    base = os.path.splitext(os.path.basename(dw))[0]

    picker_order = []
    if picker:
        picker_order.append(picker)
    picker_order += ["template", "blob", "manual", "deep"]

    seen = set()
    for desired_picker in picker_order:
        if desired_picker in seen:
            continue
        seen.add(desired_picker)

        for entry in entries:
            if str(entry.get("picker_type") or "").strip().lower() != desired_picker:
                continue
            output_shape = entry.get("output_shape") or default_output_shape
            try:
                output_shape = int(output_shape)
            except Exception:
                output_shape = default_output_shape

            cand = os.path.join(
                project_dir,
                session_name,
                "extract",
                desired_picker,
                str(output_shape),
                f"{base}_particles.mrc",
            )
            if os.path.isfile(cand):
                return cand, desired_picker

    return None, picker or None

def choose_ctf_spline_path(project_dir: str, session_name: str, exp: dict) -> Optional[str]:
    """
    Find the local CTF spline .npy for one exposure.

    Expected location:
        <project_dir>/<session_name>/ctfestimated/*ctf_spline.npy

    We match using the micrograph basename when possible.
    """
    ctf_dir = os.path.join(project_dir, session_name, "ctfestimated")
    if not os.path.isdir(ctf_dir):
        return None

    # Prefer the motion-corrected micrograph basename if available
    dw = rel_or_abs_path_from_nested(project_dir, exp, "groups", "exposure", "micrograph_blob", "path")
    non_dw = rel_or_abs_path_from_nested(project_dir, exp, "groups", "exposure", "micrograph_blob_non_dw", "path")
    abs_file_path = exp.get("abs_file_path")

    candidates = []
    for p in (dw, non_dw, abs_file_path):
        if isinstance(p, str) and p not in ("", "."):
            base = os.path.splitext(os.path.basename(p))[0]
            if base:
                candidates.append(base)

    # Try exact-ish basename matching first
    for base in candidates:
        pattern = os.path.join(ctf_dir, f"{base}*ctf_spline.npy")
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]

    # Fallback: if there is exactly one spline in the folder, use it
    all_hits = sorted(glob.glob(os.path.join(ctf_dir, "*ctf_spline.npy")))
    if len(all_hits) == 1:
        return all_hits[0]

    return None

def _unwrap_singleton(v):
    while isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    return v


def _path_from_nested_unwrap(project_dir: str, obj: dict, *keys) -> Optional[str]:
    v = nested_get(obj, *keys)
    v = _unwrap_singleton(v)
    if not isinstance(v, str) or v in ("", "."):
        return None
    return ensure_path(project_dir, v)


def _float_from_nested_unwrap(obj: dict, *keys) -> Optional[float]:
    v = nested_get(obj, *keys)
    v = _unwrap_singleton(v)
    try:
        return float(v)
    except Exception:
        return None


def _int_from_nested_unwrap(obj: dict, *keys) -> Optional[int]:
    v = nested_get(obj, *keys)
    v = _unwrap_singleton(v)
    try:
        return int(v)
    except Exception:
        return None

def get_active_pick_count(exp: dict) -> int:
    picker = str(exp.get("picker_type") or "").strip().lower()
    attrs = exp.get("attributes", {}) or {}

    key_map = {
        "blob": "total_blob_picks",
        "template": "total_template_picks",
        "deep": "total_deep_picks",
        "manual": "total_manual_picks",
    }
    key = key_map.get(picker)
    try:
        return int(attrs.get(key) or 0) if key else 0
    except Exception:
        return 0

def parse_exposure(project_dir: str, session_name: str, exp: dict, bin_size_pix: int) -> dict:
    thumb = _path_from_nested_unwrap(project_dir, exp, "groups", "exposure", "micrograph_blob_thumb", "path")
    dw = _path_from_nested_unwrap(project_dir, exp, "groups", "exposure", "micrograph_blob", "path")
    non_dw = _path_from_nested_unwrap(project_dir, exp, "groups", "exposure", "micrograph_blob_non_dw", "path")

    ctf_diag = _path_from_nested_unwrap(project_dir, exp, "groups", "exposure", "ctf_stats", "diag_image_path")
    ctf_1d = _path_from_nested_unwrap(project_dir, exp, "groups", "exposure", "ctf_stats", "fit_data_path")

    rigid_motion_path = _path_from_nested_unwrap(project_dir, exp, "groups", "exposure", "rigid_motion", "path")
    spline_motion_path = _path_from_nested_unwrap(project_dir, exp, "groups", "exposure", "spline_motion", "path")

    micrograph_psize_A = (
        _float_from_nested_unwrap(exp, "groups", "exposure", "micrograph_blob", "psize_A")
        or _float_from_nested_unwrap(exp, "micrograph_psize")
    )

    frame_start = (
        _int_from_nested_unwrap(exp, "groups", "exposure", "spline_motion", "frame_start")
        or _int_from_nested_unwrap(exp, "groups", "exposure", "rigid_motion", "frame_start")
    )
    frame_end = (
        _int_from_nested_unwrap(exp, "groups", "exposure", "spline_motion", "frame_end")
        or _int_from_nested_unwrap(exp, "groups", "exposure", "rigid_motion", "frame_end")
    )
    
    rigid_zero_shift_frame = _int_from_nested_unwrap(
        exp, "groups", "exposure", "rigid_motion", "zero_shift_frame"
    )
    
    ctf_spline_path = choose_ctf_spline_path(project_dir, session_name, exp)

    pick_cs, pick_picker_type = choose_pick_cs_path(project_dir, exp)
    extracted_cs_path, extracted_cs_picker_type = choose_extracted_cs_path(project_dir, exp)
    particle_stack, particle_picker_type = choose_extracted_stack_path(project_dir, session_name, exp, bin_size_pix)

    picker_type = str(exp.get("picker_type") or "").strip().lower() or None

    return {
        "raw": exp,
        "uid": exp.get("uid"),
        "abs_file_path": exp.get("abs_file_path"),
        "thumb_path": thumb,
        "micrograph_path": dw,
        "micrograph_non_dw_path": non_dw,
        "micrograph_psize_A": micrograph_psize_A,
        "ctf_diag_path": ctf_diag,
        "ctf_1d_path": ctf_1d,
        "rigid_motion_path": rigid_motion_path,
        "spline_motion_path": spline_motion_path,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "pick_cs_path": pick_cs,
        "particle_stack_path": particle_stack,
        "picker_type": picker_type,
        "pick_picker_type": pick_picker_type,
        "extracted_cs_path": extracted_cs_path,
        "extracted_cs_picker_type": extracted_cs_picker_type,
        "particle_picker_type": particle_picker_type,
        "ctf_fit_A": get_ctf_fit(exp),
        "defocus_A": get_avg_defocus_A(exp),
        "defocus_um": get_avg_defocus_um(exp),
        "total_motion_pix": get_total_motion(exp),
        "max_inframe_motion": get_max_inframe_motion(exp),
        "blob_picks": get_blob_picks(exp),
        "active_pick_count": get_active_pick_count(exp),
        "extracted_particles": get_extracted_particles(exp),
        "accepted": is_accepted_exp(exp),
        "rejected": is_rejected_exp(exp),
        "failed": is_failed_exp(exp),
        "status": exposure_status_label(exp),
        "status_binary": 1 if is_accepted_exp(exp) else 0 if is_rejected_exp(exp) else None,
        "start_dt": exposure_start_dt(exp),
        "end_dt": exposure_end_dt(exp),
        "rigid_zero_shift_frame": rigid_zero_shift_frame,
        "ctf_spline_path": ctf_spline_path,
    }



def assign_exposure_numbers(parsed: List[dict]) -> List[dict]:
    def sort_key(e):
        dt = e.get("start_dt") or e.get("end_dt")
        uid = e.get("uid")
        return (dt or datetime.max.replace(tzinfo=timezone.utc), uid if uid is not None else 10**18)

    out = sorted(parsed, key=sort_key)
    for i, e in enumerate(out, start=1):
        e["exposure_number"] = i
    return out

def evenly_sample(items: List[Any], n: int) -> List[Any]:
    if len(items) <= n:
        return list(items)
    idxs = np.linspace(0, len(items) - 1, n)
    idxs = sorted({int(round(i)) for i in idxs})
    return [items[i] for i in idxs][:n]

def deterministic_sample(items: List[Any], n: int, seed_str: str) -> List[Any]:
    if len(items) <= n:
        return list(items)
    seed = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(items)), n))
    return [items[i] for i in idxs]

def select_accepted_ctf_tertiles(parsed: List[dict], n_each: int = 5) -> Dict[str, List[dict]]:
    accepted = [
        e for e in parsed
        if e.get("accepted") and e.get("ctf_fit_A") is not None and e.get("thumb_path")
    ]
    accepted.sort(key=lambda e: e["ctf_fit_A"])
    if not accepted:
        return {"best": [], "middle": [], "worst": []}

    thirds = np.array_split(np.array(accepted, dtype=object), 3)
    labels = ["best", "middle", "worst"]
    return {label: evenly_sample(list(arr), n_each) for label, arr in zip(labels, thirds)}

def select_rejected_random(parsed: List[dict], n: int, seed_str: str) -> List[dict]:
    rejected = [e for e in parsed if e.get("rejected") and e.get("thumb_path")]
    rejected.sort(key=lambda e: e.get("exposure_number", 0))
    return deterministic_sample(rejected, n, seed_str)

def find_session_time_bounds(parsed: List[dict]):
    starts = [e["start_dt"] for e in parsed if e.get("start_dt")]
    ends = [e["end_dt"] for e in parsed if e.get("end_dt")]
    return (min(starts) if starts else None, max(ends) if ends else None)

def duration_hours(start_dt, end_dt):
    if not start_dt or not end_dt:
        return None
    return (end_dt - start_dt).total_seconds() / 3600.0

def summarize_vals(parsed: List[dict], key: str):
    vals = sorted([e[key] for e in parsed if e.get(key) is not None])
    if not vals:
        return None, None, None
    return vals[0], float(np.median(vals)), vals[-1]

def build_summary_sections(project: dict, ws: dict, parsed: List[dict], class_job_uid: Optional[str]):
    stats = ws.get("stats", {})
    params = ws.get("params", {})
    exp_groups = ws.get("exposure_groups", [])
    exp_group = exp_groups[0] if exp_groups else {}

    start_dt, end_dt = find_session_time_bounds(parsed)
    dur = duration_hours(start_dt, end_dt)

    ctf_best, ctf_med, ctf_worst = summarize_vals(parsed, "ctf_fit_A")
    _mot_best, mot_med, _mot_worst = summarize_vals(parsed, "max_inframe_motion")
    class2d = summarize_class2d_info(ws)

    current_picker = str(params.get("current_picker") or "").strip().lower()

    ncc_min, ncc_val, ncc_max = (
        get_picking_threshold(ws, f"{current_picker}_ncc_score")
        if current_picker else (None, None, None)
    )
    power_block = get_threshold_block(ws, f"{current_picker}_power") if current_picker else {}
    power_min = power_block.get("min")
    power_max = power_block.get("max")

    total_exposures = int(stats.get("total_exposures", 0) or 0)
    total_accepted = int(stats.get("total_accepted", 0) or 0)
    total_rejected = int(stats.get("total_rejected", 0) or 0)
    total_failed = int(stats.get("total_failed", 0) or 0)

    sections = []

    # ---------------- Session Overview ----------------
    session_summary_html = (
        f'<font color="#3182bd"><b>Found:</b> {exp_group.get("num_exposures_found", total_exposures)}</font>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<font color="#2ca25f"><b>Accepted:</b> {total_accepted}</font>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<font color="#de2d26"><b>Rejected:</b> {total_rejected}</font>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<font color="#636363"><b>Failed:</b> {total_failed}</font>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<font color="#756bb1"><b>Acceptance:</b> {fmt_pct(total_accepted, total_exposures, 1)}</font>'
    )

    session_rows = [
        ("Project Folder", project.get("project_dir", "").rstrip("/").split("/")[-1]),
        ("Project Title", str(project.get("title", ""))),
        ("Project UID", str(project.get("uid", ""))),
        ("CryoSPARC Version", str(project.get("last_dumped_version", ""))),
        ("Session", str(ws.get("session_uid") or ws.get("session_dir") or "")),
        ("Workspace UID", str(ws.get("uid", ""))),
        ("Workspace Title", str(ws.get("title", ""))),
#        ("Workspace Status", str(ws.get("status", ""))),
        ("Start Time", fmt_dt(start_dt)),
        ("End Time", fmt_dt(end_dt)),
        ("Total Time (hrs)", fmt_num(dur, 2) if dur is not None else ""),
        ("Watch Path", str(exp_group.get("file_engine_watch_path_abs", ""))),
        ("File Filter", str(exp_group.get("file_engine_filter", ""))),
#        ("Total Exposures Found", str(exp_group.get("num_exposures_found", stats.get("total_exposures", "")))),
#        ("Total Accepted", str(total_accepted)),
#        ("Total Rejected", str(total_rejected)),
#        ("Total Failed", str(total_failed)),
#        ("Acceptance Rate", fmt_pct(total_accepted, total_exposures, 1)),
    ]
    sections.append({
        "title": "Session Overview",
        "summary_html": session_summary_html,
        "rows": session_rows,
    })

    # ---------------- Acquisition / Imaging ----------------
    imaging_summary_html = (
        f'<font color="#3182bd"><b>Frames:</b> {stats.get("frames", "")}</font>'
        f'&nbsp;&nbsp;&nbsp;'
#        f'<font color="#2b8cbe"><b>Size:</b> {stats.get("nx", "")} × {stats.get("ny", "")}</font>'
#        f'&nbsp;&nbsp;&nbsp;'
        f'<font color="#31a354"><b>Pixel:</b> {fmt_num(params.get("psize_A"), 3)} Å/pix</font>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<font color="#756bb1"><b>Dose:</b> {fmt_num(params.get("total_dose_e_per_A2"), 1)} e/A2</font>'
    )

    imaging_rows = [
#        ("Frames per Movie", str(stats.get("frames", ""))),
        ("Image Dimensions (pixels)", f"{stats.get('nx', '')} × {stats.get('ny', '')}"),
#        ("Pixel Size (A/pix)", fmt_num(params.get("psize_A"), 3)),
        ("Acceleration Voltage (kV)", fmt_num(params.get("accel_kv"), 1)),
        ("Spherical Aberration (mm)", fmt_num(params.get("cs_mm"), 2)),
#        ("Total Dose (e/A2)", fmt_num(params.get("total_dose_e_per_A2"), 1)),
        ("CTF Resolution Min (A)", fmt_num(params.get("ctf_res_min_align"), 1)),
        ("CTF Resolution Max (A)", fmt_num(params.get("ctf_res_max_align"), 1)),
        ("Best CTF Fit (A)", fmt_num(ctf_best, 2) if ctf_best is not None else ""),
        ("Median CTF Fit (A)", fmt_num(ctf_med, 2) if ctf_med is not None else ""),
        ("Worst CTF Fit (A)", fmt_num(ctf_worst, 2) if ctf_worst is not None else ""),
        ("Median Max In-Frame Motion", fmt_num(mot_med, 3) if mot_med is not None else ""),
    ]
    sections.append({
        "title": "Acquisition / Imaging",
        "summary_html": imaging_summary_html,
        "rows": imaging_rows,
    })

    # ---------------- Picking / Extraction ----------------
    total_picker_picks = (
        stats.get("total_blob_picks", 0)
        if current_picker == "blob" else
        stats.get("total_template_picks", 0)
        if current_picker == "template" else
        stats.get("total_deep_picks", 0)
        if current_picker == "deep" else ""
    )

    picking_summary_html = (
        f'<font color="#3182bd"><b>Picker:</b> {current_picker}</font>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<font color="#2ca25f"><b>Box Size:</b> {params.get("box_size_pix")} pix</font>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<font color="#2ca25f"><b>Fourier Crop Box Size:</b> {params.get("bin_size_pix")} pix</font>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<font color="#756bb1"><b>Extracted:</b> {stats.get("total_extracted_particles", "")}</font>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<font color="#dd1c77"><b>Avg/Mic:</b> {fmt_num(stats.get("avg_particles_extracted_per_mic"), 1)}</font>'
    )

    picking_rows = [
#        ("Current Picker", current_picker),
    ]

    if current_picker == "blob":
        picking_rows.extend([
            ("Blob Diameter Min (A)", fmt_num(params.get("blob_diameter_min"), 1)),
            ("Blob Diameter Max (A)", fmt_num(params.get("blob_diameter_max"), 1)),
            ("Blob Min Separation (diameters)", fmt_num(params.get("blob_min_distance"), 2)),
            ("Blob NCC Threshold Value", fmt_num(ncc_val, 3)),
#            ("Blob NCC Threshold Min", fmt_num(ncc_min, 3)),
#            ("Blob NCC Threshold Max", fmt_num(ncc_max, 3)),
            ("Blob Power Min", fmt_num(power_min, 3)),
            ("Blob Power Max", fmt_num(power_max, 3)),
            ("Blob Lowpass (A)", fmt_num(params.get("blob_lowpass_res"), 1)),
#            ("Blob Angular Spacing (deg)", fmt_num(params.get("blob_angular_spacing_deg"), 1)),
            ("Blob Max Num Hits", str(params.get("blob_max_num_hits", ""))),
            ("Total Blob Picks", str(stats.get("total_blob_picks", ""))),
        ])
    elif current_picker == "template":
        template_creation_job = str(ws.get("template_creation_job", ""))
        template_creation_project = ("Template Creation Project", str(ws.get("template_creation_project", "")))

        picking_rows.extend([
            ("Template Source", f"{template_creation_project} {template_creation_job}"),
            ("Template Min Separation (diameters)", fmt_num(params.get("template_min_distance"), 2)),
            ("Template NCC Threshold Value", fmt_num(ncc_val, 3)),
#            ("Template NCC Threshold Min", fmt_num(ncc_min, 3)),
#            ("Template NCC Threshold Max", fmt_num(ncc_max, 3)),
            ("Template Power Min", fmt_num(power_min, 3)),
            ("Template Power Max", fmt_num(power_max, 3)),
            ("Template Lowpass (A)", fmt_num(params.get("template_lowpass_res"), 1)),
#            ("Template Angular Spacing (deg)", fmt_num(params.get("template_angular_spacing_deg"), 1)),
            ("Template Max Num Hits", str(params.get("template_max_num_hits", ""))),
            ("Total Template Picks", str(stats.get("total_template_picks", ""))),
        ])
    elif current_picker == "deep":
        picking_rows.extend([
            ("Deep NCC Threshold Value", fmt_num(ncc_val, 3)),
#            ("Deep NCC Threshold Min", fmt_num(ncc_min, 3)),
#            ("Deep NCC Threshold Max", fmt_num(ncc_max, 3)),
            ("Deep Power Min", fmt_num(power_min, 3)),
            ("Deep Power Max", fmt_num(power_max, 3)),
            ("Total Deep Picks", str(stats.get("total_deep_picks", ""))),
        ])

#    picking_rows.extend([
#        ("Total Extracted Particles", str(stats.get("total_extracted_particles", ""))),
#        ("Avg Extracted Particles per Micrograph", fmt_num(stats.get("avg_particles_extracted_per_mic"), 1)),
#    ])

    sections.append({
        "title": "Picking / Extraction",
        "summary_html": picking_summary_html,
        "rows": picking_rows,
    })

    # ---------------- 2D Classification ----------------
    class2d_summary_html = (
        f'<font color="#3182bd"><b>Classes:</b> {class2d["total_classes"]}</font>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<font color="#2ca25f"><b>Selected:</b> {class2d["selected_classes"]}</font>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<font color="#de2d26"><b>Rejected:</b> {class2d["rejected_classes"]}</font>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<font color="#756bb1"><b>Particles Accepted:</b> {ws.get("phase2_class2D_num_particles_accepted", "")}</font>'
    )
    
    class2d_rows = [
        ("Selected 2D Class Job", class_job_uid or ""),
        ("2D Class Max Resolution (A)", fmt_num(get_class2d_param(class_job_uid, ws, "class2D_max_res"), 1)),
    ]

    window_inner_A = nested_get(ws, "phase2_class2D_params_spec_used", "class2D_window_inner_A")
    if window_inner_A is None:
        window_inner_A = nested_get(ws, "phase2_class2D_params_spec", "class2D_window_inner_A")

    class2d_rows.extend([
        ("2D Window Inner (A)", fmt_num(window_inner_A, 1)),
        ("2D Particles In", str(ws.get("phase2_class2D_num_particles_in", ""))),
        ("2D Particles Accepted", str(ws.get("phase2_class2D_num_particles_accepted", ""))),
        ("2D Particles Rejected", str(ws.get("phase2_class2D_num_particles_rejected", ""))),
        ("2D Last Updated", fmt_dt(json_date_to_dt(ws.get("phase2_class2D_last_updated")))),
#        ("2D Class K", str(nested_get(ws, "phase2_class2D_params_spec_used", "class2D_K") or "")),
#        ("2D Classes Total", str(class2d["total_classes"])),
#        ("2D Classes Selected", str(class2d["selected_classes"])),
#        ("2D Classes Rejected", str(class2d["rejected_classes"])),
#        ("2D Particles in Selected Classes", str(class2d["selected_particles"])),
#        ("2D Particles in Rejected Classes", str(class2d["rejected_particles"])),
    ])

    sections.append({
        "title": "2D Classification",
        "summary_html": class2d_summary_html,
        "rows": class2d_rows,
    })

    return sections

def flatten_summary_sections(sections: List[dict]) -> List[Tuple[str, str]]:
    rows = []
    for sec in sections:
        rows.append((f"[{sec['title']}]", ""))
        rows.extend(sec.get("rows", []))
        rows.append(("", ""))
    return rows
