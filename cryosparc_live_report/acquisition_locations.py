#!/usr/bin/env python3
# coding: utf-8

import os
import re
import traceback
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# small helpers
# ----------------------------

def _unwrap_singleton(v):
    while isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    return v


def _nested_get(obj, *keys):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _unique_keep_order(items):
    out = []
    seen = set()
    for x in items:
        if not x:
            continue
        sx = str(x)
        if sx in seen:
            continue
        seen.add(sx)
        out.append(sx)
    return out


def _safe_resolve_file(p: str) -> Optional[Path]:
    try:
        q = Path(p)
        if q.exists():
            return q.resolve()
    except Exception:
        pass
    return None


def _fig_to_pil(fig) -> Image.Image:
    bio = BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    bio.seek(0)
    return Image.open(bio).convert("RGB")


FRACTIONS_RE = re.compile(
    r"^FoilHole_(?P<uniq>[A-Za-z0-9]+)_Data_(?P<t1>[^_]+)_(?P<t2>[^_]+)_(?P<date>\d{8})_(?P<time>\d{6})_Fractions(?:_\d+)?\.(?P<ext>mrc|tif|tiff|eer)$",
    re.IGNORECASE,
)

def canonicalize_uniq(u):
    if u is None:
        return None
    m = re.match(r"^(\d+)$", str(u))
    return m.group(1) if m else str(u)

def gridsquare_id_from_dir(gs_dir: str):
    m = re.search(r"GridSquare_(\d+)", os.path.basename(gs_dir), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def parse_epu_movie_info(movie_path: str):
    name = Path(movie_path).name
    m = FRACTIONS_RE.match(name)
    if not m:
        return None
    dt = pd.to_datetime(m.group("date") + m.group("time"), format="%Y%m%d%H%M%S", errors="coerce")
    return {
        "uniq": canonicalize_uniq(m.group("uniq")),
        "dt": dt,
    }

def find_epu_session_root_from_movie(movie_path: str) -> Optional[str]:
    p = Path(movie_path).resolve()
    for parent in [p.parent] + list(p.parents):
        if (parent / "EpuSession.dm").is_file():
            return str(parent)
    return None

# ----------------------------
# finding original movie paths
# ----------------------------

def get_exposure_movie_candidates(parsed_exp: dict, project_dir: str) -> List[str]:
    out = []

    raw = parsed_exp.get("raw") or {}

    movie_rel = _unwrap_singleton(_nested_get(raw, "groups", "exposure", "movie_blob", "path"))
    if isinstance(movie_rel, str) and movie_rel not in ("", "."):
        if os.path.isabs(movie_rel):
            out.append(movie_rel)
        else:
            out.append(os.path.join(project_dir, movie_rel))

    abs_file_path = parsed_exp.get("abs_file_path")
    if isinstance(abs_file_path, str) and abs_file_path not in ("", "."):
        out.append(abs_file_path)

    return _unique_keep_order(out)


# ----------------------------
# EPU sidecar XML matching
# ----------------------------

def find_matching_epu_xml(movie_path: str) -> Optional[str]:
    """
    Prefer the EPU micrograph XML with 'Fractions' removed from the basename.
    Only fall back to *_Fractions.xml if the base XML does not exist.
    """
    movie = _safe_resolve_file(movie_path)
    if movie is None:
        return None

    parent = movie.parent
    stem = movie.stem

    base_no_fractions = re.sub(r"_Fractions(?:_\d+)?$", "", stem, flags=re.IGNORECASE)

    preferred = [
        parent / f"{base_no_fractions}.xml",
    ]
    for cand in preferred:
        if cand.is_file():
            return str(cand)

    fallback = [
        parent / f"{stem}.xml",
    ]
    for cand in fallback:
        if cand.is_file():
            return str(cand)

    relaxed_patterns = [
        f"{base_no_fractions}*.xml",
        f"{stem}*.xml",
    ]
    for pat in relaxed_patterns:
        hits = sorted(parent.glob(pat))
        if hits:
            return str(hits[0])

    return None


# ----------------------------
# SerialEM sidecar MDOC matching
# ----------------------------

def _serialem_mdoc_names_for_movie(movie: Path) -> List[str]:
    return _unique_keep_order([
        f"{movie.name}.mdoc",
        f"{movie.stem}.mdoc",
    ])


def find_matching_serialem_mdoc(movie_path: str) -> Optional[str]:
    movie = _safe_resolve_file(movie_path)
    if movie is None:
        return None

    names = _serialem_mdoc_names_for_movie(movie)
    movie_dir = movie.parent
    parent_dir = movie_dir.parent

    search_dirs = [
        movie_dir,
        movie_dir / "mdocs",
        parent_dir,
        parent_dir / "mdocs",
    ]

    for d in search_dirs:
        for name in names:
            cand = d / name
            if cand.is_file():
                return str(cand)

    return None


# ----------------------------
# acquisition mode detection
# ----------------------------

def detect_acquisition_mode(
    project_dir: str,
    parsed: List[dict],
    max_check: int = 40,
) -> Tuple[Optional[str], Dict[str, int]]:
    xml_hits = 0
    mdoc_hits = 0
    checked_movies = 0
    checked_exposures = 0

    for exp in parsed:
        candidates = get_exposure_movie_candidates(exp, project_dir)
        if not candidates:
            continue

        checked_exposures += 1

        for movie_path in candidates:
            checked_movies += 1

            xml_hit = find_matching_epu_xml(movie_path)
            if xml_hit:
                xml_hits += 1
                break

            mdoc_hit = find_matching_serialem_mdoc(movie_path)
            if mdoc_hit:
                mdoc_hits += 1
                break

        if checked_exposures >= max_check:
            break

    stats = {
        "checked_exposures": checked_exposures,
        "checked_movies": checked_movies,
        "xml_hits": xml_hits,
        "mdoc_hits": mdoc_hits,
    }

    if xml_hits == 0 and mdoc_hits == 0:
        return None, stats

    if xml_hits >= mdoc_hits:
        return "epu", stats
    return "serialem", stats


# ----------------------------
# EPU XML parsing
# ----------------------------

def parse_epu_xml_location(xml_path: str) -> dict:
    out = {
        "stage_x": np.nan,
        "stage_y": np.nan,
        "image_shift_x": np.nan,
        "image_shift_y": np.nan,
    }

    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return out

    def localname(tag):
        return tag.split("}")[-1] if "}" in tag else tag

    def text_float(elem):
        if elem is None or elem.text is None:
            return None
        try:
            return float(elem.text)
        except Exception:
            return None

    sx = text_float(root.find(".//fei:microscopeData/fei:stage/fei:Position/fei:X", EPU_NS))
    sy = text_float(root.find(".//fei:microscopeData/fei:stage/fei:Position/fei:Y", EPU_NS))

    if sx is None or sy is None:
        for elem in root.iter():
            if localname(elem.tag).lower() == "stage":
                for sub in elem.iter():
                    if localname(sub.tag) == "Position":
                        tx = ty = None
                        for ch in sub:
                            ln = localname(ch.tag)
                            if ln in ("X", "_x"):
                                tx = text_float(ch)
                            elif ln in ("Y", "_y"):
                                ty = text_float(ch)
                        if tx is not None and ty is not None:
                            sx, sy = tx, ty
                            break
                if sx is not None and sy is not None:
                    break

    if sx is None or sy is None:
        for elem in root.iter():
            if localname(elem.tag) == "Position":
                tx = ty = None
                for ch in elem:
                    ln = localname(ch.tag)
                    if ln in ("X", "_x"):
                        tx = text_float(ch)
                    elif ln in ("Y", "_y"):
                        ty = text_float(ch)
                if tx is not None and ty is not None:
                    sx, sy = tx, ty
                    break

    isx = text_float(root.find(".//fei:microscopeData/fei:optics/fei:ImageShift/types:_x", EPU_NS))
    isy = text_float(root.find(".//fei:microscopeData/fei:optics/fei:ImageShift/types:_y", EPU_NS))

    if isx is None or isy is None:
        for elem in root.iter():
            if localname(elem.tag) == "ImageShift":
                tx = ty = None
                for ch in elem:
                    ln = localname(ch.tag)
                    if ln in ("x", "_x", "X"):
                        tx = text_float(ch)
                    elif ln in ("y", "_y", "Y"):
                        ty = text_float(ch)
                if tx is not None and ty is not None:
                    isx, isy = tx, ty
                    break

    out["stage_x"] = sx if sx is not None else np.nan
    out["stage_y"] = sy if sy is not None else np.nan
    out["image_shift_x"] = isx if isx is not None else np.nan
    out["image_shift_y"] = isy if isy is not None else np.nan

    return out


# ----------------------------
# EPU GridSquare helpers
# ----------------------------

GS_XML_RE = re.compile(r"^GridSquare_(\d{8}_\d{6})\.xml$", re.IGNORECASE)


def parse_readout_area(root):
    width = height = None
    for elem in root.iter():
        if _ln(elem.tag) == "ReadoutArea":
            for ch in elem:
                lname = _ln(ch.tag)
                if lname == "width":
                    try:
                        width = int(ch.text)
                    except Exception:
                        pass
                elif lname == "height":
                    try:
                        height = int(ch.text)
                    except Exception:
                        pass
            if width is not None and height is not None:
                return width, height
    return width, height


def parse_pixelsize(root):
    px_x = get_text_float(root.find(".//fei:SpatialScale/fei:pixelSize/fei:x/fei:numericValue", NS))
    px_y = get_text_float(root.find(".//fei:SpatialScale/fei:pixelSize/fei:y/fei:numericValue", NS))
    return px_x, px_y


def parse_ref_matrix(root):
    m11 = get_text_float(root.find(".//fei:ReferenceTransformation/fei:matrix/media:_m11", NS))
    m12 = get_text_float(root.find(".//fei:ReferenceTransformation/fei:matrix/media:_m12", NS))
    m21 = get_text_float(root.find(".//fei:ReferenceTransformation/fei:matrix/media:_m21", NS))
    m22 = get_text_float(root.find(".//fei:ReferenceTransformation/fei:matrix/media:_m22", NS))
    if None in (m11, m12, m21, m22):
        return None
    return np.array([[m11, m12], [m21, m22]], dtype=float)


def parse_stage_xy(root, override=None):
    if override and ("stage_x" in override) and ("stage_y" in override):
        return override["stage_x"], override["stage_y"]

    sx = get_text_float(root.find(".//fei:microscopeData/fei:stage/fei:Position/fei:X", NS))
    sy = get_text_float(root.find(".//fei:microscopeData/fei:stage/fei:Position/fei:Y", NS))
    if sx is not None and sy is not None:
        return sx, sy

    sx = sy = None
    for e in root.iter():
        if _ln(e.tag) == "Position":
            for cc in e:
                l = _ln(cc.tag)
                if l == "X" and sx is None:
                    sx = _to_float(cc.text)
                elif l == "Y" and sy is None:
                    sy = _to_float(cc.text)
            if sx is not None and sy is not None:
                break
    return sx, sy


def parse_imageshift(root):
    x = get_text_float(root.find(".//fei:microscopeData/fei:optics/fei:ImageShift/types:_x", NS))
    y = get_text_float(root.find(".//fei:microscopeData/fei:optics/fei:ImageShift/types:_y", NS))
    return x, y


def parse_gridsquare_meta(xml_path):
    root = ET.parse(xml_path).getroot()
    stage_x, stage_y = parse_stage_xy(root)
    px_x, px_y = parse_pixelsize(root)
    w, h = parse_readout_area(root)
    refM = parse_ref_matrix(root)
    imgshift_x, imgshift_y = parse_imageshift(root)

    return {
        "stage_x": stage_x,
        "stage_y": stage_y,
        "px_x": px_x,
        "px_y": px_y,
        "width": w,
        "height": h,
        "refM": refM,
        "imageshift": (imgshift_x, imgshift_y),
    }


def find_latest_gridsquare_xml_relaxed(gs_dir):
    xmls = []
    for fname in os.listdir(gs_dir):
        full = os.path.join(gs_dir, fname)
        if not fname.lower().endswith(".xml"):
            continue

        m = GS_XML_RE.match(fname)
        ts = None
        if m:
            try:
                ts = parse_timestamp(m.group(1))
            except Exception:
                ts = None

        if ts is None or pd.isna(ts):
            try:
                ts = pd.Timestamp.fromtimestamp(os.path.getmtime(full))
            except Exception:
                continue

        xmls.append((ts, full))

    if not xmls:
        return None

    xmls.sort(key=lambda x: x[0], reverse=True)
    return xmls[0][1]


def extract_gridsquare_number_from_path(gs_xml_path):
    gs_dir = os.path.basename(os.path.dirname(gs_xml_path))
    m = re.match(r"GridSquare_(\d+)", gs_dir, flags=re.IGNORECASE)
    return m.group(1) if m else None


def find_gridsquare_dm_path(gs_xml_path):
    gs_id = extract_gridsquare_number_from_path(gs_xml_path)
    if not gs_id:
        return None
    session_root = _safe_dirnameN(gs_xml_path, 3)
    dm_path = os.path.join(session_root, "Metadata", f"GridSquare_{gs_id}.dm")
    return dm_path if os.path.isfile(dm_path) else None


def parse_dm_pixelcenters_by_uniq(gs_xml_path):
    dm_path = find_gridsquare_dm_path(gs_xml_path)
    if dm_path is None or not os.path.isfile(dm_path):
        return {}

    try:
        root = ET.parse(dm_path).getroot()
    except Exception:
        return {}

    def direct_children(elem):
        return list(elem)

    def has_direct_child(elem, local_tag):
        return any(_ln(ch.tag) == local_tag for ch in direct_children(elem))

    def get_direct_child(elem, local_tag):
        for ch in direct_children(elem):
            if _ln(ch.tag) == local_tag:
                return ch
        return None

    def parse_pixelcenter(elem):
        pc = get_direct_child(elem, "PixelCenter")
        if pc is None:
            return None

        x = y = None
        for cc in direct_children(pc):
            ln = _ln(cc.tag).lower()
            if ln == "x":
                x = _to_float(cc.text)
            elif ln == "y":
                y = _to_float(cc.text)

        if x is None or y is None:
            return None
        return x, y

    def parse_pixelwh(elem):
        pwh = get_direct_child(elem, "PixelWidthHeight")
        if pwh is None:
            return None, None

        w = h = None
        for cc in direct_children(pwh):
            ln = _ln(cc.tag).lower()
            if ln == "width":
                w = _to_float(cc.text)
            elif ln == "height":
                h = _to_float(cc.text)
        return w, h

    def collect_basefilenames_excluding_nested_pixelcenters(elem):
        names = []

        def rec(node):
            if node is not elem and has_direct_child(node, "PixelCenter"):
                return
            if _ln(node.tag) == "BaseFileName" and node.text:
                names.append(node.text.strip())
            for ch in direct_children(node):
                rec(ch)

        for ch in direct_children(elem):
            if _ln(ch.tag) in ("PixelCenter", "PixelWidthHeight"):
                continue
            rec(ch)

        return names

    result = {}

    for node in root.iter():
        if not has_direct_child(node, "PixelCenter"):
            continue

        xy = parse_pixelcenter(node)
        if xy is None:
            continue
        x, y = xy
        w, h = parse_pixelwh(node)

        basefiles = collect_basefilenames_excluding_nested_pixelcenters(node)
        uniqs_here = set()

        for bf in basefiles:
            m = re.search(r"FoilHole_(\d+)", bf)
            if m:
                uniqs_here.add(canonicalize_uniq(m.group(1)))

        for uq in uniqs_here:
            if uq not in result:
                result[uq] = {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                }

    return result


def _parse_template_areas_from_dm(session_dir: str):
    dm_path = os.path.join(session_dir, "EpuSession.dm")
    if not os.path.isfile(dm_path):
        return (None, None), None, [], None

    try:
        root = ET.parse(dm_path).getroot()
    except Exception:
        return (None, None), None, [], None

    def find_first(elem, local_name):
        for e in elem.iter():
            if _ln(e.tag) == local_name:
                return e
        return None

    def parse_shift(node):
        if node is None:
            return None
        dx = dy = None
        for ch in node.iter():
            ln = _ln(ch.tag).lower()
            if ln == "width":
                dx = _to_float(ch.text)
            elif ln == "height":
                dy = _to_float(ch.text)
        if dx is None or dy is None:
            return None
        return (dx, dy)

    px_w = px_h = None
    tip = find_first(root, "TemplateImagePixelSize")
    if tip is not None:
        for ch in tip.iter():
            ln = _ln(ch.tag).lower()
            if ln == "width":
                px_w = _to_float(ch.text)
            elif ln == "height":
                px_h = _to_float(ch.text)

    af_node = find_first(root, "AutoFocusArea")
    af_shift = None
    if af_node is not None:
        af_shift = parse_shift(find_first(af_node, "ShiftInPixels"))

    acq_shifts = []
    daa_node = find_first(root, "DataAcquisitionAreas")
    if daa_node is not None:
        for kv in daa_node.iter():
            ln = _ln(kv.tag)
            if "KeyValuePair" not in ln:
                continue

            value_elem = None
            for ch in kv:
                if _ln(ch.tag) == "value":
                    value_elem = ch
                    break
            if value_elem is None:
                continue

            s = parse_shift(find_first(value_elem, "ShiftInPixels"))
            if s is not None:
                acq_shifts.append(s)

    drift_node = find_first(root, "DriftStabilizationArea")
    drift_shift = None
    if drift_node is not None:
        drift_shift = parse_shift(find_first(drift_node, "ShiftInPixels"))

    return (px_w, px_h), af_shift, acq_shifts, drift_shift


def build_epu_locations_without_atlas(
    project_dir: str,
    session_name: str,
    parsed: List[dict],
):
    rows = []
    micrograph_meta_cache = {}

    for exp in parsed:
        uid = exp.get("uid")
        movie_candidates = get_exposure_movie_candidates(exp, project_dir)
        resolved_movie = None

        for cand in movie_candidates:
            rp = _safe_resolve_file(cand)
            if rp is not None:
                resolved_movie = str(rp)
                break

        if not resolved_movie:
            continue

        info = parse_epu_movie_info(resolved_movie)
        if not info:
            continue

        movie_p = Path(resolved_movie)
        gs_dir = str(movie_p.parent.parent) if movie_p.parent.name == "Data" else None
        if not gs_dir or not os.path.isdir(gs_dir):
            continue

        gs_id = gridsquare_id_from_dir(gs_dir)
        if gs_id is None:
            continue

        micrograph_xml = find_matching_epu_xml(resolved_movie)
        if micrograph_xml:
            if micrograph_xml not in micrograph_meta_cache:
                micrograph_meta_cache[micrograph_xml] = parse_epu_xml_location(micrograph_xml)
            micro_meta = micrograph_meta_cache[micrograph_xml]
            image_shift_x = micro_meta.get("image_shift_x", np.nan)
            image_shift_y = micro_meta.get("image_shift_y", np.nan)
        else:
            image_shift_x = np.nan
            image_shift_y = np.nan

        rows.append({
            "uid": uid,
            "exposure_number": exp.get("exposure_number"),
            "movie_path": resolved_movie,
            "micrograph_xml": micrograph_xml,
            "gs_dir": gs_dir,
            "gs_id": gs_id,
            "uniq": info["uniq"],
            "dt": info["dt"] if pd.notna(info["dt"]) else exp.get("start_dt"),
            "ctf_fit_A": exp.get("ctf_fit_A"),
            "defocus_A": exp.get("defocus_A"),
            "ice_thickness_rel": exp.get("ice_thickness_rel"),
            "accepted": exp.get("accepted"),
            "status": exp.get("status"),
            "image_shift_x": image_shift_x,
            "image_shift_y": image_shift_y,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    sort_dt = pd.to_datetime(df["dt"], errors="coerce")
    df = df.assign(_sort_dt=sort_dt)
    df = df.sort_values(["gs_id", "uniq", "_sort_dt", "exposure_number"]).reset_index(drop=True)
    df["shot_index"] = df.groupby(["gs_id", "uniq"]).cumcount()

    epu_session_dir = None
    for p in df["movie_path"]:
        epu_session_dir = find_epu_session_root_from_movie(p)
        if epu_session_dir:
            break
    if not epu_session_dir:
        return None

    (template_px_size, _af_shift, acq_shifts, _drift_shift) = _parse_template_areas_from_dm(epu_session_dir)
    tpl_px_w_m, tpl_px_h_m = template_px_size

    cache = {}
    out = []

    for r in df.to_dict(orient="records"):
        gs_id = r["gs_id"]
        uniq = canonicalize_uniq(r["uniq"])

        if gs_id not in cache:
            gs_xml = find_latest_gridsquare_xml_relaxed(r["gs_dir"])
            if not gs_xml:
                cache[gs_id] = None
            else:
                gs_meta = parse_gridsquare_meta(gs_xml)
                dm_centers = parse_dm_pixelcenters_by_uniq(gs_xml)
                cache[gs_id] = (gs_xml, gs_meta, dm_centers)

        if cache[gs_id] is None:
            continue

        gs_xml, gs_meta, dm_centers = cache[gs_id]
        cinfo = dm_centers.get(uniq)
        if not cinfo:
            continue

        cx = cinfo.get("x")
        cy = cinfo.get("y")
        if cx is None or cy is None:
            continue

        gs_stage_x = gs_meta.get("stage_x")
        gs_stage_y = gs_meta.get("stage_y")
        gs_px_x = gs_meta.get("px_x")
        gs_px_y = gs_meta.get("px_y") or gs_px_x
        W = gs_meta.get("width")
        H = gs_meta.get("height")
        refM = gs_meta.get("refM")

        if any(v is None for v in (gs_stage_x, gs_stage_y, gs_px_x, gs_px_y, W, H)):
            continue

        acq_idx = int(r["shot_index"]) if int(r["shot_index"]) < len(acq_shifts) else None
        if acq_idx is not None and tpl_px_w_m and tpl_px_h_m:
            dx_tpl, dy_tpl = acq_shifts[acq_idx]
            dx_m = float(dx_tpl) * float(tpl_px_w_m)
            dy_m = float(dy_tpl) * float(tpl_px_h_m)

            cx = float(cx) + dx_m / float(gs_px_x)
            cy = float(cy) + dy_m / float(gs_px_y)

        dx_px = float(cx) - float(W) / 2.0
        dy_px = float(cy) - float(H) / 2.0
        local_px = np.array([dx_px, dy_px], dtype=float)

        if refM is not None:
            try:
                local_stage_m = local_px @ refM.T
            except Exception:
                local_stage_m = np.array(
                    [dx_px * float(gs_px_x), dy_px * float(gs_px_y)],
                    dtype=float,
                )
        else:
            local_stage_m = np.array(
                [dx_px * float(gs_px_x), dy_px * float(gs_px_y)],
                dtype=float,
            )

        x_m = float(gs_stage_x) + float(local_stage_m[0])
        y_m = float(gs_stage_y) + float(local_stage_m[1])

        out.append({
            "uid": r["uid"],
            "exposure_number": r["exposure_number"],
            "movie_path": r["movie_path"],
            "sidecar_path": r["micrograph_xml"] if r["micrograph_xml"] else gs_xml,
            "mode": "epu",
            "plot_x": x_m * 1e6,
            "plot_y": y_m * 1e6,
            "stage_x": gs_stage_x,
            "stage_y": gs_stage_y,
            "image_shift_x": r["image_shift_x"],
            "image_shift_y": r["image_shift_y"],
            "ctf_fit_A": r["ctf_fit_A"],
            "defocus_A": r["defocus_A"],
            "ice_thickness_rel": r["ice_thickness_rel"],
            "accepted": r["accepted"],
            "status": r["status"],
            "unit_label": "µm",
            "shot_index": r["shot_index"],
            "foilhole_uniq": uniq,
            "gs_id": gs_id,
            "cx_px": float(cx),
            "cy_px": float(cy),
            "dx_px": float(dx_px),
            "dy_px": float(dy_px),
            "gs_px_x_m": float(gs_px_x),
            "gs_px_y_m": float(gs_px_y),
        })

    out_df = pd.DataFrame(out)
    if out_df.empty:
        return None

    return out_df


# ----------------------------
# SerialEM MDOC parsing
# ----------------------------

def parse_serialem_mdoc_location(mdoc_path: str) -> dict:
    out = {
        "stage_x": np.nan,
        "stage_y": np.nan,
        "image_shift_x": np.nan,
        "image_shift_y": np.nan,
        "tilt_angle": np.nan,
        "defocus": np.nan,
        "rotation_angle": np.nan,
    }

    p = Path(mdoc_path)
    if not p.is_file():
        return out

    in_frameset0 = False

    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                if line.startswith("[FrameSet"):
                    compact = line.replace(" ", "")
                    in_frameset0 = compact.startswith("[FrameSet=0]")
                    continue

                if not in_frameset0 or "=" not in line:
                    continue

                key, val = [x.strip() for x in line.split("=", 1)]

                if key == "TiltAngle":
                    out["tilt_angle"] = pd.to_numeric(val, errors="coerce")
                elif key == "Defocus":
                    out["defocus"] = pd.to_numeric(val, errors="coerce")
                elif key == "RotationAngle":
                    out["rotation_angle"] = pd.to_numeric(val, errors="coerce")
                elif key == "StagePosition":
                    parts = val.split()
                    if len(parts) >= 2:
                        out["stage_x"] = pd.to_numeric(parts[0], errors="coerce")
                        out["stage_y"] = pd.to_numeric(parts[1], errors="coerce")
                elif key == "ImageShift":
                    parts = val.split()
                    if len(parts) >= 2:
                        out["image_shift_x"] = pd.to_numeric(parts[0], errors="coerce")
                        out["image_shift_y"] = pd.to_numeric(parts[1], errors="coerce")
    except Exception:
        return out

    return out


# ----------------------------
# dataframe building
# ----------------------------

def _choose_sidecar_for_mode(movie_candidates: List[str], mode: str) -> Tuple[Optional[str], Optional[str]]:
    for movie_path in movie_candidates:
        resolved = _safe_resolve_file(movie_path)
        if resolved is None:
            continue

        if mode == "epu":
            sidecar = find_matching_epu_xml(str(resolved))
        elif mode == "serialem":
            sidecar = find_matching_serialem_mdoc(str(resolved))
        else:
            sidecar = None

        if sidecar:
            return str(resolved), sidecar

    return None, None


def build_location_dataframe(
    project_dir: str,
    session_name: str,
    parsed: List[dict],
    mode: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str], Dict[str, object]]:
    if mode is None:
        mode, detect_stats = detect_acquisition_mode(project_dir, parsed)
    else:
        _, detect_stats = detect_acquisition_mode(project_dir, parsed)

    info = dict(detect_stats)
    info.setdefault("examples", [])
    info.setdefault("fail_counts", {
        "no_movie_candidates": 0,
        "no_sidecar": 0,
        "parse_no_coords": 0,
        "rows_added": 0,
    })

    if mode not in ("epu", "serialem"):
        return None, None, info

    if mode == "epu":
        df_epu = build_epu_locations_without_atlas(
            project_dir=project_dir,
            session_name=session_name,
            parsed=parsed,
        )
        if df_epu is None or df_epu.empty:
            return None, mode, info
        info["fail_counts"]["rows_added"] = int(len(df_epu))
        return df_epu, mode, info

    rows = []

    for exp in parsed:
        uid = exp.get("uid")
        movie_candidates = get_exposure_movie_candidates(exp, project_dir)

        if not movie_candidates:
            info["fail_counts"]["no_movie_candidates"] += 1
            continue

        resolved_movie, sidecar_path = _choose_sidecar_for_mode(movie_candidates, mode)
        if not resolved_movie or not sidecar_path:
            info["fail_counts"]["no_sidecar"] += 1
            continue

        meta = parse_serialem_mdoc_location(sidecar_path)
        stage_x = meta["stage_x"]
        stage_y = meta["stage_y"]
        isx = meta["image_shift_x"]
        isy = meta["image_shift_y"]

        x = stage_x + (isx if np.isfinite(isx) else 0.0) if np.isfinite(stage_x) else np.nan
        y = stage_y + (isy if np.isfinite(isy) else 0.0) if np.isfinite(stage_y) else np.nan

        x_plot = x
        y_plot = y
        unit_label = "SerialEM units"

        if not (np.isfinite(x_plot) and np.isfinite(y_plot)):
            info["fail_counts"]["parse_no_coords"] += 1
            continue

        rows.append({
            "uid": uid,
            "exposure_number": exp.get("exposure_number"),
            "movie_path": resolved_movie,
            "sidecar_path": sidecar_path,
            "mode": mode,
            "plot_x": x_plot,
            "plot_y": y_plot,
            "stage_x": stage_x,
            "stage_y": stage_y,
            "image_shift_x": isx,
            "image_shift_y": isy,
            "ctf_fit_A": exp.get("ctf_fit_A"),
            "defocus_A": exp.get("defocus_A"),
            "ice_thickness_rel": exp.get("ice_thickness_rel"),
            "accepted": exp.get("accepted"),
            "status": exp.get("status"),
            "unit_label": unit_label,
        })
        info["fail_counts"]["rows_added"] += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return None, mode, info

    df = df[np.isfinite(df["plot_x"]) & np.isfinite(df["plot_y"])].copy()
    if df.empty:
        return None, mode, info

    return df, mode, info


# ----------------------------
# rendering
# ----------------------------

def auto_ctf_range(
    df: pd.DataFrame,
    col: str = "ctf_fit_A",
    lower_pct: float = 10.0,
    upper_pct: float = 90.0,
    fallback: Tuple[float, float] = (4.0, 10.0),
) -> Tuple[float, float]:
    vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
    if vals.size == 0:
        return fallback

    vmin = float(np.percentile(vals, lower_pct))
    vmax = float(np.percentile(vals, upper_pct))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return fallback

    return vmin, vmax


def _zero_shift_mask(df: pd.DataFrame, atol: float = 1e-12) -> pd.Series:
    if "image_shift_x" not in df.columns or "image_shift_y" not in df.columns:
        return pd.Series(False, index=df.index)

    isx = pd.to_numeric(df["image_shift_x"], errors="coerce")
    isy = pd.to_numeric(df["image_shift_y"], errors="coerce")

    return (
        isx.notna()
        & isy.notna()
        & np.isclose(isx, 0.0, atol=atol, rtol=0.0)
        & np.isclose(isy, 0.0, atol=atol, rtol=0.0)
    )


def render_location_ctf_image(
    df: pd.DataFrame,
    mode: str,
    session_name: str,
    ctf_vmin: Optional[float] = None,
    ctf_vmax: Optional[float] = None,
    point_size: float = 6.0,
    cmap: str = "Blues_r",
    auto_percentiles: Tuple[float, float] = (10.0, 90.0),
) -> Image.Image:
    unit_label = str(df["unit_label"].dropna().iloc[0]) if "unit_label" in df.columns and df["unit_label"].notna().any() else ""

    fig, ax = plt.subplots(figsize=(8.8, 8.0), dpi=400)

    if ctf_vmin is None or ctf_vmax is None:
        auto_vmin, auto_vmax = auto_ctf_range(
            df,
            col="ctf_fit_A",
            lower_pct=auto_percentiles[0],
            upper_pct=auto_percentiles[1],
            fallback=(4.0, 12.0),
        )
        if ctf_vmin is None:
            ctf_vmin = auto_vmin
        if ctf_vmax is None:
            ctf_vmax = auto_vmax

    valid_ctf = df["ctf_fit_A"].notna()
    zero_shift = _zero_shift_mask(df)

    if valid_ctf.any():
        sc = ax.scatter(
            df.loc[valid_ctf, "plot_x"],
            df.loc[valid_ctf, "plot_y"],
            c=df.loc[valid_ctf, "ctf_fit_A"],
            cmap=cmap,
            vmin=ctf_vmin,
            vmax=ctf_vmax,
            s=point_size,
            alpha=0.92,
            linewidths=0,
            edgecolors="none",
            zorder=2,
        )
        cbar = fig.colorbar(
            sc,
            ax=ax,
            orientation="horizontal",
            fraction=0.06,
            pad=0.08,
            aspect=40,
        )
        cbar.set_label("CTF fit (Å)")
    else:
        ax.scatter(
            df["plot_x"],
            df["plot_y"],
            color="0.5",
            s=point_size,
            alpha=0.8,
            linewidths=0,
            edgecolors="none",
            zorder=2,
        )

    if (~valid_ctf).any():
        ax.scatter(
            df.loc[~valid_ctf, "plot_x"],
            df.loc[~valid_ctf, "plot_y"],
            color="0.75",
            s=max(1.5, point_size * 0.75),
            alpha=0.65,
            linewidths=0,
            zorder=2,
        )

    # Black outline around points with image shift == (0, 0)
    if zero_shift.any():
        ring_size = max(18.0, point_size * 3.0)
        ax.scatter(
            df.loc[zero_shift, "plot_x"],
            df.loc[zero_shift, "plot_y"],
            s=ring_size,
            facecolors="none",
            edgecolors="black",
            linewidths=0.6,
            alpha=1.0,
            zorder=4,
        )

    ax.set_xlabel(f"X ({unit_label})" if unit_label else "X")
    ax.set_ylabel(f"Y ({unit_label})" if unit_label else "Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    fig.tight_layout()
    return _fig_to_pil(fig)


# ----------------------------
# public entry point
# ----------------------------

def build_acquisition_location_image(
    project_dir: str,
    session_name: str,
    parsed: List[dict],
    mode: Optional[str] = None,
    ctf_vmin: Optional[float] = None,
    ctf_vmax: Optional[float] = None,
    point_size: float = 6.0,
    auto_percentiles: Tuple[float, float] = (10.0, 90.0),
) -> Tuple[Optional[Image.Image], Dict[str, object]]:
    info = {
        "mode": None,
        "checked_exposures": 0,
        "checked_movies": 0,
        "xml_hits": 0,
        "mdoc_hits": 0,
        "n_points": 0,
        "fail_counts": {},
        "examples": [],
    }

    try:
        df, detected_mode, build_info = build_location_dataframe(
            project_dir=project_dir,
            session_name=session_name,
            parsed=parsed,
            mode=mode,
        )
        info.update(build_info or {})
        info["mode"] = detected_mode
        info["n_points"] = 0 if df is None else int(len(df))

        if df is None or detected_mode not in ("epu", "serialem"):
            return None, info

        img = render_location_ctf_image(
            df=df,
            mode=detected_mode,
            session_name=session_name,
            ctf_vmin=ctf_vmin,
            ctf_vmax=ctf_vmax,
            point_size=point_size,
            auto_percentiles=auto_percentiles,
        )
        return img, info

    except Exception as e:
        info["exception"] = repr(e)
        info["traceback"] = traceback.format_exc()
        return None, info
