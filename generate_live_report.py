#!/usr/bin/env python3
# coding: utf-8


"""
Generate a PDF report for a CryoSPARC Live session.


Local layout
------------
Expected directory structure:


    generate_live_report/
    ├── generate_live_report.py
    ├── cryosparc_live_report/
    │   ├── __init__.py
    │   ├── acquisition_locations.py
    │   ├── images.py
    │   ├── io.py
    │   ├── pdf.py
    │   ├── plots.py
    │   ├── refinement_pngs.py
    │   ├── scale_bars.py
    │   ├── stats.py
    │   └── textstyle.py
    └── epu/
        ├── report_style.py
        └── report_utils.py


Usage
-----
    python3 generate_live_report.py /path/to/CS-project
    python3 generate_live_report.py /path/to/CS-project --session S1
    python3 generate_live_report.py /path/to/CS-project --session S1 --refine-job J67


Outputs
-------
    - Live_Imaging_Summary_<projectfolder>_<session>.pdf


Direct Python dependencies
--------------------------
    - numpy
    - pandas
    - reportlab
    - pillow
    - pymongo
    - matplotlib
    - mrcfile
    - scipy
    - scikit-image
    - ghostscript

"""


import os
import re
import sys
import argparse
import shutil
import subprocess


from pathlib import Path
from typing import Optional


import numpy as np
import ghostscript as gs
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl_canvas
from reportlab import rl_config
from PIL import Image, ImageDraw, ImageFont


rl_config.useA85 = 0


_IMPORT_ERRORS = []


try:
    from cryosparc_live_report.io import (
        read_json,
        iter_exposures_bson,
        find_live_workspace,
        find_latest_classavg_mrc,
        fmt_num,
        nested_get,
    )
except Exception as e:
    _IMPORT_ERRORS.append(f"cryosparc_live_report.io import failed: {e}")


try:
    from cryosparc_live_report.stats import (
        parse_exposure_light,
        enrich_exposure_paths,
        assign_exposure_numbers,
        assign_elapsed_minutes,
        select_accepted_ctf_tertiles,
        select_rejected_random,
        build_summary_sections,
        flatten_summary_sections,
        build_class2d_info_map,
        refine_status_label,
    )
except Exception as e:
    _IMPORT_ERRORS.append(f"cryosparc_live_report.stats import failed: {e}")


try:
    from cryosparc_live_report.images import (
        make_classavg_montages,
        make_micrograph_panel,
    )
except Exception as e:
    _IMPORT_ERRORS.append(f"cryosparc_live_report.images import failed: {e}")


try:
    from cryosparc_live_report.plots import build_scatterplot_pages
except Exception as e:
    _IMPORT_ERRORS.append(f"cryosparc_live_report.plots import failed: {e}")

from cryosparc_live_report.textstyle import merge_nested_dicts, build_report_text_theme, merge_nested_dicts, build_micrograph_text_settings

try:
    from cryosparc_live_report.pdf import (
        add_summary_pages,
        draw_single_image_page,
        draw_five_plot_page,
        draw_panel_pages,
        draw_refinement_summary_page,
        draw_initial_model_summary_pages,
    )
except Exception as e:
    _IMPORT_ERRORS.append(f"cryosparc_live_report.pdf import failed: {e}")


try:
    from cryosparc_live_report.refinement_pngs import (
        generate_report_pngs,
        generate_initial_model_pngs,
    )
except Exception:
    generate_report_pngs = None
    generate_initial_model_pngs = None


try:
    from cryosparc_live_report.acquisition_locations import (
        build_acquisition_location_image,
        build_acquisition_scalar_image,
        normalize_acquisition_key,
    )
except Exception as e:
    _IMPORT_ERRORS.append(f"cryosparc_live_report.acquisition_locations import failed: {e}")




def print_preflight_errors_and_exit():
    if not _IMPORT_ERRORS:
        return


    print("Error: required report modules could not be imported.\n")
    for msg in _IMPORT_ERRORS:
        print(f"- {msg}")


    print("\nMake sure:")
    print("  1. generate_live_report.py is next to the cryosparc_live_report/ package directory")
    print("  2. cryosparc_live_report/__init__.py exists")
    print("  3. dependencies are installed:")
    print("     pip install reportlab pillow numpy mrcfile matplotlib pymongo cryosparc-tools")
    sys.exit(2)




def _fmt_thresh(v):
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v)




def _finite_positive_float(v):
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x) or x <= 0:
        return None
    return x




def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]




def get_workspace_particle_diameter_A(ws: dict):
    params = ws.get("params", {}) or {}


    picker = str(params.get("current_picker") or "").strip().lower()


    template_d = _finite_positive_float(params.get("template_diameter"))
    blob_d = _finite_positive_float(params.get("blob_diameter_max"))


    if picker == "template" and template_d is not None:
        return template_d
    if picker == "blob" and blob_d is not None:
        return blob_d


    return template_d or blob_d




def build_report_style() -> dict:
    report_text_theme = build_report_text_theme({
        "plot_text": 14,
        "plot_title": 19,
        "panel_header": 36,
        "panel_footer": 30,
        "band_title": 19,
    })


    micro_style, micro_plots = build_micrograph_text_settings(report_text_theme)


    return {
        "text_theme": report_text_theme,
        "micrograph_panel": {
            "layout": {
                "panel_w": 1500,
                "panel_h": 1900,
                "margin": 18,
                "gap": 16,
                "title_h": 52,
                "meta_h": 120,
                "title_meta_gap": 8,
                "meta_line_gap": 12,
                "meta_content_gap": 16,
                "left_col_frac": 0.60,
                "particles_h_frac": 0.20,
                "ctf1d_h_frac": 0.20,
                "min_particles_h": 100,
                "min_ctf1d_h": 200,
                "min_micrograph_h": 420,
                "right_panel_count": 3,
            },
            "style": micro_style,
            "plots": micro_plots,
            "render": {
                "invert": False,
                "display_mode": "percentile",
                "sigma": 5.0,
                "gamma": 1.0,
                "central_frac": 0.8,
                "p_lo": 0.5,
                "p_hi": 99.5,
                "edge_frac": 0.12,
                "edge_p_lo": 5.0,
                "edge_img_p_hi": 99.5,
                "lowpass_A": 20.0,
            },
        },
        "scatterplots": {
            "figsize": (11, 1.8),
            "dpi": 200,
            "show_legend": False,
            "show_internal_title": False,
        },
        "classavg": {
            "cols": 6,
            "rows_per_page": 9,
            "tile_size": 400,
            "dpi": 350,
            "sort_by_count": True,
            "count_key": "num_particles_total",
            "interpolation": "none",
            "render": {
                "invert": False,
                "display_mode": "auto",
                "sigma": 5.0,
                "gamma": 1.0,
                "central_frac": 0.8,
                "p_lo": 0.5,
                "p_hi": 99.5,
                "edge_frac": 0.12,
                "edge_p_lo": 5.0,
                "edge_img_p_hi": 99.5,
            },
            "style": {
                "facecolor": "white",
                "cell_facecolor": "white",
                "neutral_cell_border_color": (180, 180, 180),
                "neutral_cell_border_width": 1.0,
                "selected_border_color": (53, 183, 121),
                "selected_border_width": 4.0,
                "rejected_border_color": (72, 40, 120),
                "rejected_border_width": 2.0,
                "unknown_border_color": (255, 255, 255),
                "unknown_border_width": 1.5,
                "force_black_border_color": (0, 0, 0),
                "force_black_border_width": 1.8,
                "resolution_fontsize": 8,
                "count_fontsize": 8,
                "text_color": "black",
                "img_extent": (0.10, 0.90, 0.15, 0.95),
                "subplot_left": 0.02,
                "subplot_right": 0.98,
                "subplot_bottom": 0.02,
                "subplot_top": 0.98,
                "subplot_wspace": 0.08,
                "subplot_hspace": 0.08,
            },
        },
        "refinement": {
            "cmap": "viridis",
        },
    }




def make_placeholder_pil(text, size=(1200, 500)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(180, 180, 180), width=2)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    d.multiline_text((20, 20), text, fill=(90, 90, 90), font=font, spacing=6)
    return img




def _load_rgb_image_or_placeholder(path: Path, title: str, missing_as_none: bool = False):
    if path.is_file():
        try:
            with Image.open(path) as im:
                return im.convert("RGB")
        except Exception as e:
            return make_placeholder_pil(f"{title}\nFailed to load:\n{path.name}\n{e}")


    if missing_as_none:
        return None


    return make_placeholder_pil(f"{title}\nMissing file:\n{path.name}")




def load_initial_model_panel_list(panel_info_list):
    panels = []


    for item in panel_info_list or []:
        title = str(item.get("title") or "")
        path = Path(item.get("path"))
        image = _load_rgb_image_or_placeholder(path, title, missing_as_none=False)
        panels.append((title, image))


    return panels




def load_refinement_panel_map(outdir: Path, iteration: int, title_overrides=None):
    title_overrides = title_overrides or {}


    file_map = {
        "surface_views": outdir / f"surface_views_iter{iteration:03d}.png",
        "real_space_slices": outdir / f"real_space_slices_iter{iteration:03d}.png",
        "per_particle_scale": outdir / f"per_particle_scale_iter{iteration:03d}.png",
        "fsc": outdir / f"fsc_iter{iteration:03d}.png",
        "guinier": outdir / f"guinier_iter{iteration:03d}.png",
        "mask_slices": outdir / f"mask_slices_iter{iteration:03d}.png",
        "viewing_direction": outdir / f"viewing_direction_distribution_iter{iteration:03d}.png",
        "posterior_precision": outdir / f"posterior_precision_directional_distribution_iter{iteration:03d}.png",
        "protein_view_lookup": outdir / f"protein_view_lookup_iter{iteration:03d}.png",
    }


    default_titles = {
        "surface_views": "Surface Views",
        "real_space_slices": "Real Space Slices",
        "per_particle_scale": "Per-Particle Scale",
        "fsc": "FSC",
        "guinier": "Guinier Plot",
        "mask_slices": "Mask Slices",
        "viewing_direction": "Viewing Directions",
        "posterior_precision": "Posterior Precision",
        "protein_view_lookup": "Protein View Lookup",
    }


    panel_map = {}
    for key, path in file_map.items():
        title = title_overrides.get(key, default_titles[key])
        panel_map[key] = {
            "title": title,
            "image": _load_rgb_image_or_placeholder(path, title, missing_as_none=True),
        }


    return panel_map




def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default




def _fmt_int(v):
    try:
        return f"{int(v):,}"
    except Exception:
        return str(v)




def _decode_text_maybe(v):
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", errors="ignore")
    if v is None:
        return None
    return str(v)




def _find_first_existing_file(base_dir: Path, names):
    for name in names:
        p = base_dir / name
        if p.is_file():
            return p
    return None




def _load_cs_array(cs_path: Path):
    return np.load(str(cs_path), mmap_mode="r", allow_pickle=False)




def _abinit_class_number(row: dict, fallback_index: int) -> int:
    vol_gname = str(row.get("vol_gname") or "")
    m = re.search(r"class_(\d+)", vol_gname)
    if m:
        return int(m.group(1))
    return fallback_index




def choose_particle_analysis_job(ws: dict, refine_job_uid: str = None):
    """
    Choose the job to drive the particle-analysis page.


    Returns:
        ({'job_uid': 'JXX', 'job_type': 'refine'|'select2D'}, None)
    or:
        (None, 'reason')
    """
    class2d_job_uid = str(ws.get("phase2_class2D_job") or "").strip() or None
    abinit_job_uid = str(ws.get("phase2_abinit_job") or "").strip() or None
    workspace_refine_job_uid = str(ws.get("phase2_refine_job") or "").strip() or None
    select2d_job_uid = str(ws.get("phase2_select2D_job") or "").strip() or None


    active_refine_job_uid = refine_job_uid or workspace_refine_job_uid


    class2d_info = ws.get("phase2_class2D_info") or []
    n_selected_classes = sum(1 for row in class2d_info if bool(row.get("selected")))


    if not any((class2d_job_uid, abinit_job_uid, active_refine_job_uid)):
        return None, "No 2D, initial model, or refinement jobs found."


    if active_refine_job_uid:
        return {
            "job_uid": active_refine_job_uid,
            "job_type": "refine",
        }, None


    if n_selected_classes == 0:
        return None, "No refinement job and no selected 2D classes."


    if class2d_job_uid and select2d_job_uid:
        return {
            "job_uid": select2d_job_uid,
            "job_type": "select2D",
        }, None


    return None, "No refinement job and no associated phase2_select2D_job found."




def summarize_particle_counts_from_cs(cs_arr):
    """
    Return:
        {
            'total_particles': int,
            'counts_by_uid': {micrograph_uid: count, ...},
            'counts_by_key': {normalized_key: count, ...},
        }
    """
    dtype_names = set(cs_arr.dtype.names or [])
    total_particles = int(len(cs_arr))


    counts_by_uid = {}
    counts_by_key = {}


    if "location/micrograph_uid" in dtype_names:
        vals = np.asarray(cs_arr["location/micrograph_uid"])
        if vals.size > 0:
            uniq, counts = np.unique(vals, return_counts=True)
            counts_by_uid = {int(u): int(c) for u, c in zip(uniq, counts)}


    path_field = None
    for candidate in ("location/micrograph_path", "blob/path"):
        if candidate in dtype_names:
            path_field = candidate
            break


    if path_field is not None:
        tmp = {}
        for raw_path in cs_arr[path_field]:
            key = normalize_acquisition_key(_decode_text_maybe(raw_path))
            if not key:
                continue
            tmp[key] = tmp.get(key, 0) + 1
        counts_by_key = tmp


    return {
        "total_particles": total_particles,
        "counts_by_uid": counts_by_uid,
        "counts_by_key": counts_by_key,
    }




def load_particle_counts_for_job(project_dir: str, job_uid: str, job_type: str):
    job_dir = Path(project_dir) / job_uid
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Particle-analysis job directory not found: {job_dir}")


    if job_type == "refine":
        cs_path = _find_first_existing_file(job_dir, [
            f"{job_uid}_passthrough_particles.cs",
            "passthrough_particles.cs",
            f"{job_uid}_particles.cs",
            "particles.cs",
        ])
    elif job_type == "select2D":
        cs_path = _find_first_existing_file(job_dir, [
            f"{job_uid}_passthrough_particles_selected.cs",
            "passthrough_particles_selected.cs",
            f"{job_uid}_particles_selected.cs",
            "particles_selected.cs",
            f"{job_uid}_selected_particles.cs",
            "selected_particles.cs",
        ])
    else:
        raise ValueError(f"Unsupported particle-analysis job type: {job_type}")


    if cs_path is None:
        raise FileNotFoundError(f"Could not find particle file for {job_uid} ({job_type})")


    cs_arr = _load_cs_array(cs_path)
    summary = summarize_particle_counts_from_cs(cs_arr)
    summary["cs_path"] = str(cs_path)
    summary["job_uid"] = job_uid
    summary["job_type"] = job_type
    return summary




def get_total_particles_classified_2d(ws: dict) -> int:
    """
    True particles classified in 2D = sum of num_particles_total over all 2D classes.


    Fallbacks are only used if that table is unavailable.
    """
    info = ws.get("phase2_class2D_info") or []


    total = 0
    found_any = False
    for row in info:
        try:
            total += int(row.get("num_particles_total") or 0)
            found_any = True
        except Exception:
            pass


    if found_any and total > 0:
        return total


    accepted = _safe_int(ws.get("phase2_class2D_num_particles_accepted"), 0)
    rejected = _safe_int(ws.get("phase2_class2D_num_particles_rejected"), 0)
    if accepted + rejected > 0:
        return accepted + rejected


    return _safe_int(ws.get("phase2_class2D_num_particles_in"), 0)




def get_abinit_symmetry(ws: dict) -> str:
    return str(nested_get(ws, "phase2_abinit_params_spec", "abinit_symmetry") or "").strip()




def get_refine_symmetry(ws: dict) -> str:
    return str(
        nested_get(ws, "phase2_refine_params_spec_used", "refine_symmetry")
        or nested_get(ws, "phase2_refine_params_spec", "refine_symmetry")
        or ""
    ).strip()




def build_classavg_note(ws: dict, class_job_uid: str = None) -> str:
    n_classified = get_total_particles_classified_2d(ws)


    parts = [
        "This page uses the latest 2D job queued in the live workspace",
        f"\nParticles classified: {_fmt_int(n_classified)}",
    ]
    return " ".join(parts)




def get_abinit_class_meta_map(ws: dict):
    """
    Build a map like:
        {
            0: {"num_particles": 44554, "selected": True},
            1: {"num_particles": 26338, "selected": False},
            2: {"num_particles": 29108, "selected": False},
        }
    using phase2_abinit_info.
    """
    out = {}
    info = ws.get("phase2_abinit_info") or []


    for i, row in enumerate(info):
        class_num = _abinit_class_number(row, i)
        out[class_num] = {
            "num_particles": _safe_int(row.get("num_particles"), 0),
            "selected": bool(row.get("selected")),
        }


    return out




def format_initial_model_panel_title(class_num: int, num_particles: int, selected: bool = False) -> str:
    title = f"Class {class_num} — {num_particles:,} particles"
    if selected:
        title += " [selected]"
    return title




def build_initial_model_note(ws: dict, abinit_job_uid: str = None, panel_info_list=None) -> str:
    n_in = _safe_int(ws.get("phase2_abinit_num_particles_in"), 0)
    symmetry = get_abinit_symmetry(ws)


    parts = []


    if symmetry:
        parts.append(f"{symmetry} symmetry,")
    if n_in > 0:
        parts.append(f"{_fmt_int(n_in)} input particles\n")


    if panel_info_list:
        thr_bits = []
        for item in panel_info_list:
            class_num = item.get("class_number")
            thr = item.get("threshold", None)
            if class_num is None or thr is None:
                continue
            thr_bits.append(f"Class {class_num}: {_fmt_thresh(thr)}")


        if thr_bits:
            parts.append("Surface threshold(s): " + "; ".join(thr_bits))


    return " ".join(parts)




def build_refinement_note(
    ws: dict,
    refine_job_uid: str = None,
    refinement_iteration: int = None,
    display_map_mode: str = None,
    display_map_reason: str = None,
    surface_threshold: float = None,
) -> str:
    n_in = _safe_int(ws.get("phase2_refine_num_particles_in"), 0)
    status = refine_status_label(ws)
    symmetry = get_refine_symmetry(ws)


    parts = []


    if status:
        parts.append(f"{status} status:")
    if refine_job_uid and refinement_iteration is not None:
        parts.append(f"Iteration {refinement_iteration:03d},")
    if symmetry:
        parts.append(f"{symmetry} symmetry,")
    if n_in > 0:
        parts.append(f"{_fmt_int(n_in)} input particles")


    if display_map_mode:
        parts.append(
            f"\nBased on the resolution determined by the GSFSC and the Nyquist limit, the {display_map_mode} map was used to generate surface views and real-space slices."
        )
    if surface_threshold is not None:
        parts.append(f"The volume threshold used for surface/slice figures was {_fmt_thresh(surface_threshold)}.")


    return " ".join(parts)




def optimize_pdf_with_ghostscript(
    pdf_path,
    dpi=600,
    jpeg_quality=None,
    replace_original=False,
    suffix="_optimized",
):
    """
    Optimize a PDF using Ghostscript.


    Parameters
    ----------
    pdf_path : str or Path
        Path to the input PDF.
    dpi : int
        Target resolution for color/grayscale images.
    jpeg_quality : int or None
        If set (e.g. 85), forces JPEG recompression for color/grayscale images.
        If None, Ghostscript chooses compression automatically.
    replace_original : bool
        If True, replaces the original PDF with the optimized one.
    suffix : str
        Suffix for the optimized PDF filename if replace_original is False.


    Returns
    -------
    str
        Path to the optimized PDF, or the original PDF if optimization failed.
    """
    pdf_path = Path(pdf_path)


    if not pdf_path.is_file():
        print(f"Ghostscript optimization skipped: file not found: {pdf_path}")
        return str(pdf_path)


    gs = shutil.which("gs")
    if gs is None:
        for candidate in ("gswin64c", "gswin32c"):
            gs = shutil.which(candidate)
            if gs:
                break


    if gs is None:
        print("Ghostscript optimization skipped: Ghostscript executable not found.")
        return str(pdf_path)


    if replace_original:
        out_path = pdf_path.with_name(pdf_path.stem + "_tmp_optimized.pdf")
    else:
        out_path = pdf_path.with_name(pdf_path.stem + suffix + pdf_path.suffix)


    cmd = [
        gs,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.6",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dAutoRotatePages=/None",
        "-dDownsampleColorImages=true",
        "-dColorImageDownsampleType=/Bicubic",
        f"-dColorImageResolution={int(dpi)}",
        "-dDownsampleGrayImages=true",
        "-dGrayImageDownsampleType=/Bicubic",
        f"-dGrayImageResolution={int(dpi)}",
        "-dDownsampleMonoImages=true",
        "-dMonoImageDownsampleType=/Subsample",
        "-dMonoImageResolution=600",
    ]


    if jpeg_quality is not None:
        cmd.extend([
            "-dAutoFilterColorImages=false",
            "-dColorImageFilter=/DCTEncode",
            "-dAutoFilterGrayImages=false",
            "-dGrayImageFilter=/DCTEncode",
            f"-dJPEGQ={int(jpeg_quality)}",
        ])


    cmd.extend([
        f"-sOutputFile={out_path}",
        str(pdf_path),
    ])


    try:
        orig_size = pdf_path.stat().st_size
        subprocess.run(cmd, check=True)
        new_size = out_path.stat().st_size if out_path.exists() else None


        if replace_original and out_path.exists():
            os.replace(out_path, pdf_path)
            return str(pdf_path)


        if new_size is not None:
            print(
                f"Optimized PDF written: {out_path} "
                f"({orig_size / 1024 / 1024:.1f} MB -> {new_size / 1024 / 1024:.1f} MB)"
            )
            return str(out_path)


    except subprocess.CalledProcessError as e:
        print(f"Ghostscript optimization failed: {e}")
    except Exception as e:
        print(f"Ghostscript optimization failed: {e}")


    try:
        if replace_original and out_path.exists():
            out_path.unlink()
    except Exception:
        pass


    return str(pdf_path)




def build_report(
    project_dir: str,
    session_name: str = "S1",
    refine_job_uid: str = None,
    epu_session_dir: str = None,
    atlas_root: str = None,
) -> int:
    """
    Build the live report PDF.


    Notes
    -----
    `epu_session_dir` and `atlas_root` are currently accepted for compatibility
    with older or external callers, but are not used by this implementation.
    """
    print("Beginning analysis (this may take a few minutes)...")


    print_preflight_errors_and_exit()


    project_dir = os.path.abspath(project_dir)
    if not os.path.isdir(project_dir):
        print(f"Error: project directory not found: {project_dir}")
        return 2


    project_json = os.path.join(project_dir, "project.json")
    workspaces_json = os.path.join(project_dir, "workspaces.json")
    exposures_bson = os.path.join(project_dir, session_name, "exposures.bson")


    for p in (project_json, workspaces_json, exposures_bson):
        if not os.path.isfile(p):
            print(f"Error: missing required file: {p}")
            return 2


    try:
        project = read_json(project_json)
        workspaces = read_json(workspaces_json)
        ws = find_live_workspace(workspaces, session_name)
    except Exception as e:
        print(f"Error: failed to load project/workspace metadata: {e}")
        return 2


    params = ws.get("params", {})
    bin_size_pix = int(params.get("bin_size_pix", 180) or 180)
    common_particle_diameter_A = get_workspace_particle_diameter_A(ws)


    try:
        parsed = [
            parse_exposure_light(project_dir, session_name, exp)
            for exp in iter_exposures_bson(exposures_bson)
        ]
        parsed = assign_exposure_numbers(parsed)
        parsed = assign_elapsed_minutes(parsed)
    except Exception as e:
        print(f"Error: failed to load or parse exposure metadata: {e}")
        return 2


    try:
        class_job_uid = str(ws.get("phase2_class2D_job") or "").strip() or None


        if not class_job_uid:
            print("Warning: workspace does not define phase2_class2D_job; 2D class-average pages may be omitted.")
        elif not os.path.isdir(os.path.join(project_dir, class_job_uid)):
            class_job_uid = None


    except Exception as e:
        print(f"Error: failed while selecting workspace 2D class job: {e}")
        return 2


    report_style = build_report_style()


    report_text_theme = report_style["text_theme"]


    micrograph_panel_layout = report_style["micrograph_panel"]["layout"]
    micrograph_panel_style = report_style["micrograph_panel"]["style"]
    micrograph_panel_plots = report_style["micrograph_panel"]["plots"]
    micrograph_panel_render = report_style["micrograph_panel"]["render"]


    scatterplot_style = report_style["scatterplots"]
    classavg_style = report_style["classavg"]
    refinement_style = report_style["refinement"]


    classavg_render_kwargs = dict(classavg_style["render"])


    classavg_imgs = []
    classavg_mrc = None
    if class_job_uid:
        class_job_dir = os.path.join(project_dir, class_job_uid)
        classavg_mrc = find_latest_classavg_mrc(class_job_dir, class_job_uid)
        if classavg_mrc:
            try:
                class_info_map = build_class2d_info_map(ws)
                class2d_info = ws.get("phase2_class2D_info") or []
                n_selected_classes = sum(1 for row in class2d_info if bool(row.get("selected")))
                force_black_borders = (n_selected_classes == 0) and bool(class2d_info)


                classavg_imgs = make_classavg_montages(
                    mrc_path=classavg_mrc,
                    class_info_map=class_info_map,
                    cols=classavg_style["cols"],
                    rows_per_page=classavg_style["rows_per_page"],
                    tile_size=classavg_style["tile_size"],
                    dpi=classavg_style["dpi"],
                    sort_by_count=classavg_style["sort_by_count"],
                    count_key=classavg_style["count_key"],
                    interpolation=classavg_style["interpolation"],
                    force_black_borders=force_black_borders,
                    style_cfg=classavg_style["style"],
                    scale_bar_length_A=common_particle_diameter_A,
                    **classavg_render_kwargs,
                )
            except Exception as e:
                print(f"Warning: failed to render class averages: {e}")


    accepted_tertiles = select_accepted_ctf_tertiles(parsed, n_each=4)
    rejected_sample = select_rejected_random(parsed, n=4, seed_str=f"{project_dir}:{session_name}:rejected")


    for label in ("best", "middle", "worst"):
        accepted_tertiles[label] = [
            enrich_exposure_paths(project_dir, session_name, exp, bin_size_pix)
            for exp in accepted_tertiles[label]
        ]


    rejected_sample = [
        enrich_exposure_paths(project_dir, session_name, exp, bin_size_pix)
        for exp in rejected_sample
    ]


    try:
        scatterplot_pages = build_scatterplot_pages(
            parsed,
            ws,
            text_theme=report_text_theme,
            style=scatterplot_style,
        )
    except Exception as e:
        print(f"Warning: failed to render scatterplots: {e}")
        scatterplot_pages = []


    acquisition_loc_img = None
    acquisition_loc_info = {}
    try:
        acquisition_loc_img, acquisition_loc_info = build_acquisition_location_image(
            project_dir=project_dir,
            session_name=session_name,
            parsed=parsed,
            auto_point_size=True,
            point_diameter_um=3,
            min_point_diameter_pt=0.5,
            max_point_diameter_pt=6.0,
            rotate_epu_ccw=True,
        )
    except Exception as e:
        print(f"Warning: failed to render acquisition-location page: {e}")


    particle_analysis_img = None
    particle_analysis_info = {}
    particle_analysis_job_uid = None
    particle_analysis_skip_reason = None
    try:
        particle_job_spec, particle_analysis_skip_reason = choose_particle_analysis_job(
            ws,
            refine_job_uid=refine_job_uid,
        )


        if particle_job_spec is not None:
            particle_counts = load_particle_counts_for_job(
                project_dir=project_dir,
                job_uid=particle_job_spec["job_uid"],
                job_type=particle_job_spec["job_type"],
            )


            particle_analysis_job_uid = particle_job_spec["job_uid"]


            particle_analysis_img, particle_analysis_info = build_acquisition_scalar_image(
                project_dir=project_dir,
                session_name=session_name,
                parsed=parsed,
                values_by_uid=particle_counts["counts_by_uid"],
                values_by_key=particle_counts["counts_by_key"],
                value_label="Selected particles per micrograph",
                auto_percentiles=(10.0, 90.0),
                auto_point_size=True,
                point_diameter_um=3,
                min_point_diameter_pt=0.5,
                max_point_diameter_pt=6.0,
                rotate_epu_ccw=True,
            )
            particle_analysis_info["total_particles"] = int(particle_counts["total_particles"])
    except Exception as e:
        print(f"Warning: failed to render particle-analysis page: {e}")


    accepted_heading_map = {
        "best": "Accepted Micrographs: Best-Third CTF Fit",
        "middle": "Accepted Micrographs: Middle-Third CTF Fit",
        "worst": "Accepted Micrographs: Worst-Third CTF Fit",
    }


    accepted_panels_by_tertile = {
        "best": [],
        "middle": [],
        "worst": [],
    }


    for label in ("best", "middle", "worst"):
        for exp in accepted_tertiles[label]:
            try:
                accepted_panels_by_tertile[label].append(
                    make_micrograph_panel(
                        ws,
                        exp,
                        str(fmt_num(exp)),
                        fmt_num,
                        layout=micrograph_panel_layout,
                        style=micrograph_panel_style,
                        plot_cfg=micrograph_panel_plots,
                        **micrograph_panel_render,
                    )
                )
            except Exception as e:
                print(f"Warning: failed accepted panel for exposure {exp.get('uid')}: {e}")


    rejected_panels = []
    for exp in rejected_sample:
        try:
            rejected_panels.append(
                make_micrograph_panel(
                    ws,
                    exp,
                    "Rejected exposure",
                    fmt_num,
                    layout=micrograph_panel_layout,
                    style=micrograph_panel_style,
                    plot_cfg=micrograph_panel_plots,
                    **micrograph_panel_render,
                )
            )
        except Exception as e:
            print(f"Warning: failed rejected panel for exposure {exp.get('uid')}: {e}")


    sections = build_summary_sections(project, ws, parsed, class_job_uid, project_dir=project_dir)


    initial_model_panels = None
    initial_model_panel_info = []


    refinement_panel_map = None
    refinement_iteration = None
    display_map_mode = None
    display_map_reason = None
    surface_threshold = None


    abinit_job_uid = str(ws.get("phase2_abinit_job") or "").strip() or None


    if abinit_job_uid:
        abinit_job_dir = Path(project_dir) / abinit_job_uid


        if not abinit_job_dir.is_dir():
            print(f"Warning: ab-initio job directory not found: {abinit_job_dir}")
        elif generate_initial_model_pngs is None:
            print("Warning: initial-model section skipped because generate_initial_model_pngs is None.")
        else:
            try:
                abinit_outdir = abinit_job_dir / "report_pngs"


                initial_model_panel_info = generate_initial_model_pngs(
                    job_folder=abinit_job_dir,
                    outdir=abinit_outdir,
                    cmap=refinement_style["cmap"],
                    threshold_value=None,
                    text_theme=report_text_theme,
                    embed_titles=False,
                    scale_bar_length_A=common_particle_diameter_A,
                )


                abinit_meta_map = get_abinit_class_meta_map(ws)
                for item in initial_model_panel_info:
                    class_num = item.get("class_number")
                    meta = abinit_meta_map.get(class_num)


                    if meta is not None:
                        item["title"] = format_initial_model_panel_title(
                            class_num=class_num,
                            num_particles=meta["num_particles"],
                            selected=meta["selected"],
                        )
                    else:
                        item["title"] = f"Class {class_num}"


                initial_model_panels = load_initial_model_panel_list(initial_model_panel_info)


            except Exception as e:
                print(f"Warning: failed to generate initial-model PNGs for {abinit_job_uid}: {e}")


    workspace_refine_job_uid = str(ws.get("phase2_refine_job") or "").strip() or None
    if not refine_job_uid:
        refine_job_uid = workspace_refine_job_uid


    if refine_job_uid:
        refine_job_dir = Path(project_dir) / refine_job_uid


        if not refine_job_dir.is_dir():
            print(f"Warning: refinement job directory not found: {refine_job_dir}")
        elif generate_report_pngs is None:
            print("Warning: refinement section skipped because generate_report_pngs is None.")
        else:
            try:
                refine_outdir = refine_job_dir / "report_pngs"


                refinement_iteration, _, refine_gsfsc, display_map_mode, display_map_reason, surface_threshold = generate_report_pngs(
                    job_folder=refine_job_dir,
                    outdir=refine_outdir,
                    pixel_size_override=None,
                    cmap=refinement_style["cmap"],
                    threshold_value=None,
                    slice_vmax=None,
                    text_theme=report_text_theme,
                    embed_titles=False,
                    scale_bar_length_A=common_particle_diameter_A,
                )


                fsc_panel_title = f"FSC ({refine_gsfsc:.2f} Å @ 0.143)"


                refinement_panel_map = load_refinement_panel_map(
                    refine_outdir,
                    refinement_iteration,
                    title_overrides={"fsc": fsc_panel_title},
                )


            except Exception as e:
                print(f"Warning: failed to generate refinement PNGs for {refine_job_uid}: {e}")
                import traceback
                traceback.print_exc()


    project_folder = os.path.basename(project_dir.rstrip(os.sep))
    pdf_name = f"Live_Imaging_Summary_{project_folder}_{session_name}.pdf"
    pdf_path = os.path.join(project_dir, pdf_name)


    try:
        c = rl_canvas.Canvas(pdf_path, pagesize=letter, pageCompression=1)
        width, height = letter
        margin = 36
        page_num = 1


        page_num = add_summary_pages(
            c,
            width,
            height,
            margin,
            project_folder,
            sections,
            page_num_start=page_num,
        )


        if scatterplot_pages:
            scatter_note = (
                "Threshold lines are included only when min/max limits are present in workspace attributes. Any conditions with min/max limits are shown, in addition to a selection of most-informative plots. X-axis is exposure number for all plots."
            )


            max_plots_per_page = 7


            for page_idx, (heading, plots) in enumerate(scatterplot_pages):
                for chunk_idx, plot_chunk in enumerate(chunked(plots, max_plots_per_page)):
                    page_heading = heading if chunk_idx == 0 else f"{heading} ({chunk_idx + 1})"
                    page_num = draw_five_plot_page(
                        c,
                        page_num,
                        width,
                        height,
                        margin,
                        page_heading,
                        plot_chunk,
                        scatter_note if (page_idx == 0 and chunk_idx == 0) else None,
                    )


        if acquisition_loc_img is not None:
            mode = acquisition_loc_info.get("mode")
            n_points = acquisition_loc_info.get("n_points", 0)


            acquisition_note = f"Points detected from {mode} session: {n_points}"


            page_num = draw_single_image_page(
                c,
                page_num,
                width,
                height,
                margin,
                "Physical Location vs CTF Fit",
                acquisition_loc_img,
                note=acquisition_note,
            )
        else:
            print("Unable to render location plots (perhaps the raw data have moved or been deleted, or no metadata files?)")


        if particle_analysis_img is not None and particle_analysis_job_uid:
            particle_note = (
                f"Total selected particles used in analysis: {_fmt_int(particle_analysis_info.get('total_particles', 0))}\n"
                "Micrographs with zero selected particles are colored gray"
            )
            page_num = draw_single_image_page(
                c,
                page_num,
                width,
                height,
                margin,
                f"Physical Location vs Selected Particles per Micrograph ({particle_analysis_job_uid})",
                particle_analysis_img,
                note=particle_note,
            )
        elif particle_analysis_skip_reason:
            print(f"Skipping particle-analysis page: {particle_analysis_skip_reason}")
        elif acquisition_loc_img is not None:
            print("Skipped location vs particle plot (likely no appropriate jobs identified)")


        accepted_note_used = False
        for label in ("best", "middle", "worst"):
            panels = accepted_panels_by_tertile[label]
            if not panels:
                continue


            accepted_note = (
                "Representative accepted micrographs are randomly chosen from accepted exposures after splitting into best, middle, and worst thirds by CTF fit. "
                "Pick overlays and extracted-particle examples use the picker active for each exposure (blob or template). "
                "Particles are Butterworth filtered (25 Å) for ease of viewing; color is inverted (lighter particles on darker background)."
            )


            page_num = draw_panel_pages(
                c,
                page_num,
                width,
                height,
                margin,
                accepted_heading_map[label],
                panels,
                note=accepted_note if not accepted_note_used else None,
            )
            accepted_note_used = True


        if rejected_panels:
            rejected_note = "Rejected micrographs are a random sample of up to 4 rejected exposures."
            page_num = draw_panel_pages(
                c,
                page_num,
                width,
                height,
                margin,
                "Representative Rejected Micrographs",
                rejected_panels,
                note=rejected_note,
            )


        if classavg_imgs:
            print(f"2D class average job: {class_job_uid}")
        else:
            print("No 2D class average job detected; segment skipped")


        for i, classavg_img in enumerate(classavg_imgs, start=1):
            heading = f"2D Class Averages ({class_job_uid})"
            if len(classavg_imgs) > 1:
                heading += f" {i}/{len(classavg_imgs)}"


            if class_job_uid:
                class_note = build_classavg_note(ws, class_job_uid)
            else:
                class_note = (
                    "No completed class_2D_new job was found automatically; "
                    "the 2D class-average section may be omitted. "
                    f"Total particles classified: {_fmt_int(get_total_particles_classified_2d(ws))}"
                )


            page_num = draw_single_image_page(
                c,
                page_num,
                width,
                height,
                margin,
                heading,
                classavg_img,
                note=class_note,
            )


        if initial_model_panels:
            print(f"Initial model job: {abinit_job_uid}")
            initial_model_note = build_initial_model_note(
                ws,
                abinit_job_uid,
                panel_info_list=initial_model_panel_info,
            )


            page_num = draw_initial_model_summary_pages(
                c,
                page_num,
                width,
                height,
                margin,
                f"Initial Model ({abinit_job_uid})",
                initial_model_panels,
                note=initial_model_note,
            )
        else:
            print("No initial model job detected; segment skipped")


        if refinement_panel_map:
            print(f"Refinement job: {refine_job_uid}")
            refine_note = build_refinement_note(
                ws,
                refine_job_uid,
                refinement_iteration,
                display_map_mode=display_map_mode,
                display_map_reason=display_map_reason,
                surface_threshold=surface_threshold,
            )


            page_num = draw_refinement_summary_page(
                c,
                page_num,
                width,
                height,
                margin,
                f"Refinement ({refine_job_uid})",
                refinement_panel_map,
                note=refine_note,
            )
        else:
            print("No refinement job detected; segment skipped")


        c.save()


        optimize_pdf_with_ghostscript(pdf_path, dpi=600, replace_original=True)
        print(f"Wrote: {pdf_name}")


    except Exception as e:
        print(f"Error: failed while writing PDF: {e}")
        return 2


    return 0




def main():
    parser = argparse.ArgumentParser(description="Generate a CryoSPARC Live session PDF report.")
    parser.add_argument("project_dir", help="Path to CryoSPARC project directory")
    parser.add_argument("--session", default="S1", help="Session dir/uid (default: S1)")
    parser.add_argument("--refine-job", default=None, help="Refinement job uid (default: auto-detected from session)")
    args = parser.parse_args()


    rc = build_report(
        args.project_dir,
        session_name=args.session,
        refine_job_uid=args.refine_job,
    )
    sys.exit(rc)




if __name__ == "__main__":
    main()


