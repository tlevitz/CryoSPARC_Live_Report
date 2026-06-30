#!/usr/bin/env python3
# coding: utf-8

"""
Generate a PDF report for a CryoSPARC Live session.

README / local usage
--------------------

Expected layout:
    your_reports/
    ├── generate_live_report.py
    ├── cryosparc_live_report/
    │   ├── __init__.py
    │   ├── io.py
    │   ├── stats.py
    │   ├── images.py
    │   ├── plots.py
    │   └── pdf.py
    └── epu/
        ├── report_style.py
        └── report_utils.py

Run:
    python3 generate_live_report.py /path/to/CS-project
    python3 generate_live_report.py /path/to/CS-project --session S1
    python3 generate_live_report.py /path/to/CS-project --session S1 --class-job J67

Outputs:
    - Live_Imaging_Summary_<projectfolder>_<session>.pdf
    - live_stats_<session>.txt

Recommended dependencies:
    pip install reportlab pillow numpy mrcfile matplotlib pymongo cryosparc-tools
"""

import os
import re
import sys
import argparse
import numpy as np

from pathlib import Path
from typing import Optional
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl_canvas
from PIL import Image, ImageDraw, ImageFont

_IMPORT_ERRORS = []

try:
    from cryosparc_live_report.io import (
        read_json,
        load_exposures_bson,
        find_live_workspace,
        find_latest_classavg_mrc,
        fmt_num,
        nested_get,
    )
except Exception as e:
    _IMPORT_ERRORS.append(f"cryosparc_live_report.io import failed: {e}")

try:
    from cryosparc_live_report.stats import (
        parse_exposure,
        assign_exposure_numbers,
        select_accepted_ctf_tertiles,
        select_rejected_random,
        build_summary_sections,
        flatten_summary_sections,
        build_class2d_info_map,
        summarize_class2d_info,
        summarize_abinit_info,
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
    from cryosparc_live_report.plots import build_scatterplots
except Exception as e:
    _IMPORT_ERRORS.append(f"cryosparc_live_report.plots import failed: {e}")

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
    from cryosparc_live_report.acquisition_locations import build_acquisition_location_image
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

def merge_nested_dicts(defaults: dict, user: Optional[dict]) -> dict:
    out = dict(defaults)
    if not user:
        return out

    for k, v in user.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            merged = dict(out[k])
            merged.update(v)
            out[k] = merged
        else:
            out[k] = v
    return out


def build_report_text_theme(user: Optional[dict] = None) -> dict:
    defaults = {
        "plot_text": 14,
        "plot_title": 18,
        "panel_header": 24,
        "panel_footer": 16,
        "band_title": None,
    }
    out = dict(defaults)
    if user:
        out.update(user)
    if out["band_title"] is None:
        out["band_title"] = out["plot_title"]
    return out

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

def build_micrograph_text_settings(
    text_theme: Optional[dict] = None,
    style_overrides: Optional[dict] = None,
    plot_overrides: Optional[dict] = None,
):
    t = build_report_text_theme(text_theme)

    band_font = int(t["band_title"])
    band_h = max(26, int(round(1.9 * band_font)))
    band_y_pad = max(2, int(round(0.10 * band_font)))

    style = {
        "title_font_size": int(t["panel_header"]),
        "body_font_size": int(t["panel_footer"]),
        "small_title_font_size": band_font,
        "small_title_band_h": band_h,
        "small_title_y_pad": band_y_pad,
        "title_color": (0, 0, 0),
        "body_color": (20, 20, 20),
        "border_color": (185, 185, 185),
        "bg_color": "white",
        "band_bg_color": (255, 255, 255),
        "band_text_color": (20, 20, 20),
        "band_font_weight": "normal",
        "panel_border_width": 0,
        "panel_border_color": (185, 185, 185),
    }

    plot_cfg = {
        "local_motion": {
            "title": "Local motion",
            "dpi": 100,
            "patch_spacing_A": 380.0,
            "patch_size_A": 500.0,
            "viewer_scale": 40.0,
            "grid_color": "0.88",
            "grid_lw": 0.7,
            "traj_lw": 1.0,
            "traj_alpha": 0.95,
            "start_marker_ms": 1.8,
            "row_cmap": "viridis",
            "row_cmap_min": 0.10,
            "row_cmap_max": 0.90,
            "title_fontsize": int(t["plot_title"]),
            "facecolor": "white",
            "tight_pad": 0.15,
        },
        "local_defocus": {
            "title": "Local Defocus",
            "dpi": 100,
            "display_grid": 180,
            "mode": "nearest",
            "elev": 28,
            "azim": -58,
            "cmap": "viridis",
            "z_half_range_A": 2500.0,
            "facecolor": "white",
            "xlabel": f"\nX (pix)",
            "ylabel": f"\nY (pix)",
            "zlabel": "Defocus (µm)",
            "axis_label_fontsize": int(t["plot_text"]),
            "title_fontsize": int(t["plot_title"]),
            "tick_labelsize": int(t["plot_text"]),
            "box_aspect": (1.0, 1.0, 0.55),
            "colorbar": False,
            "colorbar_shrink": 0.62,
            "colorbar_pad": 0.05,
            "colorbar_fraction": 0.05,
            "colorbar_label": "Mean defocus (µm)",
            "colorbar_label_fontsize": int(t["plot_text"]),
            "tight_pad": 0.15,
        },
        "global_motion": {
            "title": "Global motion",
            "dpi": 100,
            "line_color": "#6a51a3",
            "line_width": 1.2,
            "start_marker_ms": 2.0,
            "grid_color": "0.93",
            "grid_lw": 0.6,
            "axis_line_color": "0.88",
            "axis_line_lw": 0.8,
            "title_fontsize": int(t["plot_title"]),
            "tick_labelsize": int(t["plot_text"]),
            "tight_pad": 0.15,
            "facecolor": "white",
            "subtract_zero_frame": True,
        },
        "ctf_1d": {
            "threshold": 0.3,
            "smooth_window": 7,
            "dpi": 100,
            "render_scale": 3,
            "facecolor": "white",
            "ps_color": "black",
            "ctf_color": "#de2d26",
            "fit_color": "#3182bd",
            "threshold_line_color": "#31a354",
            "grid_color": "0.93",
            "refline_color": "0.88",
            "ps_lw": 1.35,
            "ctf_lw": 1.15,
            "fit_lw": 1.10,
            "threshold_lw": 1.1,
            "xlabel_fontsize": int(t["plot_text"]),
            "ylabel_fontsize": int(t["plot_text"]),
            "tick_labelsize": int(t["plot_text"]),
            "title_fontsize": int(t["plot_title"]),
            "legend_fontsize": int(t["plot_text"]),
            "top_axis_fontsize": int(t["plot_text"]),
            "top_tick_fontsize": int(t["plot_text"]),
            "tight_pad": 0.35,
            "resolution_ticks_A": [20, 15, 10, 8, 6, 5, 4, 3],
        },
        "particles": {
            "sample_n": 6,
            "cols": 3,
            "tile_inches": 1.5,
            "dpi": 100,
            "invert": False,
            "autoscale": "imshow",
            "p_lo": 0.5,
            "p_hi": 99.5,
            "wspace": 0.1,
            "hspace": 0.1,
            "add_indices": False,
            "facecolor": "white",
            "index_fontsize": 8,
            "index_color": "yellow",
            "index_bbox_facecolor": "black",
            "index_bbox_alpha": 0.35,
            "index_bbox_pad": 0.5,
            "index_bbox_edgecolor": "none",
        },
    }

    style = merge_nested_dicts(style, style_overrides)
    plot_cfg = merge_nested_dicts(plot_cfg, plot_overrides)
    return style, plot_cfg


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


def choose_evenly_spaced(items, n):
    if not items:
        return []
    if len(items) <= n:
        return items
    idxs = np.linspace(0, len(items) - 1, n)
    return [items[int(round(i))] for i in idxs]


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

def load_initial_model_panel_list(panel_info_list):
    panels = []


    for item in panel_info_list or []:
        title = str(item.get("title") or "")
        path = Path(item.get("path"))


        if path.is_file():
            try:
                with Image.open(path) as im:
                    panels.append((title, im.convert("RGB")))
            except Exception as e:
                panels.append(
                    (title, make_placeholder_pil(f"{title}\nFailed to load:\n{path.name}\n{e}"))
                )
        else:
            panels.append(
                (title, make_placeholder_pil(f"{title}\nMissing file:\n{path.name}"))
            )


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

        if path.is_file():
            try:
                with Image.open(path) as im:
                    panel_map[key] = {
                        "title": title,
                        "image": im.convert("RGB"),
                    }
            except Exception as e:
                panel_map[key] = {
                    "title": title,
                    "image": make_placeholder_pil(f"{key}\nFailed to load:\n{path.name}\n{e}"),
                }
        else:
            panel_map[key] = {
                "title": title,
                "image": None,
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


def get_selected_abinit_volume_labels(ws: dict):
    """
    Return labels like:
        ["Class 0 (123,456 particles)", "Class 2 (87,654 particles)"]
    preserving cryoSPARC's 0-based class indexing.
    """
    labels = []
    info = ws.get("phase2_abinit_info") or []


    for i, row in enumerate(info):
        if not bool(row.get("selected")):
            continue


        idx = row.get("class_idx", row.get("volume_idx", None))
        try:
            class_num = int(idx)
        except Exception:
            class_num = i


        n = _safe_int(row.get("num_particles"), 0)
        if n > 0:
            labels.append(f"Class {class_num} ({n:,} particles)")
        else:
            labels.append(f"Class {class_num}")


    return labels

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

def get_abinit_class_entries(ws: dict):
    """
    Return a list like:
        [
            {"label": "Class 0", "selected": True,  "num_particles": 44554},
            {"label": "Class 1", "selected": False, "num_particles": 26338},
            ...
        ]
    """
    out = []
    info = ws.get("phase2_abinit_info") or []

    for i, row in enumerate(info):
        vol_gname = str(row.get("vol_gname") or "")
        m = re.search(r"class_(\d+)", vol_gname)

        if m:
            class_num = int(m.group(1))
        else:
            class_num = i

        out.append({
            "label": f"Class {class_num}",
            "selected": bool(row.get("selected")),
            "num_particles": _safe_int(row.get("num_particles"), 0),
        })

    return out

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
        vol_gname = str(row.get("vol_gname") or "")
        m = re.search(r"class_(\d+)", vol_gname)


        if m:
            class_num = int(m.group(1))
        else:
            class_num = i


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


def build_report(
    project_dir: str, 
    session_name: str = "S1", 
    refine_job_uid: str = None,
    epu_session_dir: str = None,
    atlas_root: str = None,    
) -> int:
    
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
    stats = ws.get("stats", {})
    bin_size_pix = int(params.get("bin_size_pix", 180) or 180)
    common_particle_diameter_A = get_workspace_particle_diameter_A(ws)

    try:
        exposures = load_exposures_bson(exposures_bson)
    except Exception as e:
        print(f"Error: failed to read exposures.bson: {e}")
        print("Hint: install/use pymongo so bson.decode_file_iter is available.")
        return 2

    try:
        parsed = [parse_exposure(project_dir, session_name, exp, bin_size_pix) for exp in exposures]
        parsed = assign_exposure_numbers(parsed)

        start_times = [e.get("start_dt") for e in parsed if e.get("start_dt") is not None]
        if start_times:
            t0 = min(start_times)
            for e in parsed:
                dt = e.get("start_dt")
                if dt is not None:
                    e["elapsed_minutes"] = (dt - t0).total_seconds() / 60.0
                else:
                    e["elapsed_minutes"] = None

    except Exception as e:
        print(f"Error: failed to parse exposure metadata: {e}")
        return 2

    try:
        class_job_uid = str(ws.get("phase2_class2D_job") or "").strip() or None

        if not class_job_uid:
            print("Warning: workspace does not define phase2_class2D_job; 2D class-average pages may be omitted.")

        elif not os.path.isdir(os.path.join(project_dir, class_job_uid)):
            print(f"Warning: workspace 2D class job directory not found: {class_job_uid}")
            class_job_uid = None

    except Exception as e:
        print(f"Error: failed while selecting workspace 2D class job: {e}")
        return 2

    REPORT_STYLE = build_report_style()

    REPORT_TEXT_THEME = REPORT_STYLE["text_theme"]

    MICROGRAPH_PANEL_LAYOUT = REPORT_STYLE["micrograph_panel"]["layout"]
    MICROGRAPH_PANEL_STYLE = REPORT_STYLE["micrograph_panel"]["style"]
    MICROGRAPH_PANEL_PLOTS = REPORT_STYLE["micrograph_panel"]["plots"]
    MICROGRAPH_PANEL_RENDER = REPORT_STYLE["micrograph_panel"]["render"]

    SCATTERPLOT_STYLE = REPORT_STYLE["scatterplots"]
    CLASSAVG_STYLE = REPORT_STYLE["classavg"]
    REFINEMENT_STYLE = REPORT_STYLE["refinement"]

    classavg_render_kwargs = dict(CLASSAVG_STYLE["render"])

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
                    cols=CLASSAVG_STYLE["cols"],
                    rows_per_page=CLASSAVG_STYLE["rows_per_page"],
                    tile_size=CLASSAVG_STYLE["tile_size"],
                    dpi=CLASSAVG_STYLE["dpi"],
                    sort_by_count=CLASSAVG_STYLE["sort_by_count"],
                    count_key=CLASSAVG_STYLE["count_key"],
                    interpolation=CLASSAVG_STYLE["interpolation"],
                    force_black_borders=force_black_borders,
                    style_cfg=CLASSAVG_STYLE["style"],
                    scale_bar_length_A=common_particle_diameter_A,
                    **classavg_render_kwargs,
                )
            except Exception as e:
                print(f"Warning: failed to render class averages: {e}")

    accepted_tertiles = select_accepted_ctf_tertiles(parsed, n_each=4)
    rejected_sample = select_rejected_random(parsed, n=4, seed_str=f"{project_dir}:{session_name}:rejected")

    try:
        scatterplots = build_scatterplots(
            parsed,
            ws,
            text_theme=REPORT_TEXT_THEME,
            style=SCATTERPLOT_STYLE,
        )
    except Exception as e:
        print(f"Warning: failed to render scatterplots: {e}")
        scatterplots = []

    acquisition_loc_img = None
    acquisition_loc_info = {}
    try:
        acquisition_loc_img, acquisition_loc_info = build_acquisition_location_image(
            project_dir=project_dir,
            session_name=session_name,
            parsed=parsed,
            point_size=0.8,
            rotate_epu_ccw=True,
        )

#        print(acquisition_loc_info)

    except Exception as e:
        print(f"Warning: failed to render acquisition-location page: {e}")

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
                        str(fmt_num(exp)),   # or whatever gives your exposure number/title
                        fmt_num,
                        layout=MICROGRAPH_PANEL_LAYOUT,
                        style=MICROGRAPH_PANEL_STYLE,
                        plot_cfg=MICROGRAPH_PANEL_PLOTS,
                        **MICROGRAPH_PANEL_RENDER,
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
                    layout=MICROGRAPH_PANEL_LAYOUT,
                    style=MICROGRAPH_PANEL_STYLE,
                    plot_cfg=MICROGRAPH_PANEL_PLOTS,
                    **MICROGRAPH_PANEL_RENDER,
                )
            )
        except Exception as e:
            print(f"Warning: failed rejected panel for exposure {exp.get('uid')}: {e}")

    sections = build_summary_sections(project, ws, parsed, class_job_uid, project_dir=project_dir)
    rows = flatten_summary_sections(sections)

    initial_model_panels = None
    
    refinement_panel_map = None
    refinement_iteration = None

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
                    cmap=REFINEMENT_STYLE["cmap"],
                    threshold_value=None,
                    text_theme=REPORT_TEXT_THEME,
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
                    cmap=REFINEMENT_STYLE["cmap"],
                    threshold_value=None,
                    slice_vmax=None,
                    text_theme=REPORT_TEXT_THEME,
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
    else:
        print("No refine_job_uid found; refinement section will be skipped.")

    project_folder = os.path.basename(project_dir.rstrip(os.sep))
    pdf_name = f"Live_Imaging_Summary_{project_folder}_{session_name}.pdf"
    pdf_path = os.path.join(project_dir, pdf_name)

    try:
        c = rl_canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        margin = 36
        page_num = 1

        page_num = add_summary_pages(
            c, width, height, margin, project_folder, sections,
            page_num_start=page_num,
        )

        if scatterplots:
            scatter_note = (
                "Threshold lines are included only when min/max limits are present in workspace attributes. X-axis is exposure number for all plots."
            )
            page_num = draw_five_plot_page(
                c, page_num, width, height, margin,
                "Session Scatterplots", scatterplots, scatter_note
            )

        if acquisition_loc_img is not None:
            mode = acquisition_loc_info.get("mode")
            n_points = acquisition_loc_info.get("n_points", 0)

            acquisition_note = (
                f"Points detected from {mode} session: {n_points}"
            )

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
            print("Unable to render location vs CTF plot (perhaps the raw data have moved or been deleted, or no metadata files?)")
#            print("acquisition_loc_info =", acquisition_loc_info)

        if accepted_panels_by_tertile:
            accepted_note = (
                "Representative accepted micrographs are chosen from accepted exposures and split into best, middle, and worst thirds by CTF fit. "
                "Pick overlays and extracted-particle examples use the picker active for each exposure (blob or template). "
                "Particles are Butterworth filtered (25 Å) for ease of viewing; color is inverted (lighter particles on darker background)"
            )
            accepted_note_used = False

            for label in ("best", "middle", "worst"):
                panels = accepted_panels_by_tertile[label]
                if not panels:
                    continue

                page_num = draw_panel_pages(
                    c,
                    page_num,
                    width,
                    height,
                    margin,
                    accepted_heading_map[label],
                    panels,
                    note=accepted_note if not accepted_note_used else None,   # note only on first accepted page
                )

                accepted_note_used = True

        if rejected_panels:
            rejected_note = (
                "Rejected micrographs are a random sample of up to 4 rejected exposures."
            )
            page_num = draw_panel_pages(
                c, page_num, width, height, margin,
                "Representative Rejected Micrographs",
                rejected_panels,
                note=rejected_note
            )

        for i, classavg_img in enumerate(classavg_imgs, start=1):
            heading = f"2D Class Averages ({class_job_uid})"
            if len(classavg_imgs) > 1:
                heading += f" {i}/{len(classavg_imgs)}"
            if class_job_uid:
                class_note = build_classavg_note(ws, class_job_uid)
            else:
                class_note = (
                    "No completed class_2D_new job was found automatically; "
                    f"the 2D class-average section may be omitted. "
                    f"Total particles classified: {_fmt_int(get_total_particles_classified_2d(ws))}"
                )
            page_num = draw_single_image_page(
                c, page_num, width, height, margin,
                heading, classavg_img, note=class_note
            )

        if initial_model_panels:
            initial_model_note = build_initial_model_note(ws, abinit_job_uid, panel_info_list=initial_model_panel_info)

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

        if refinement_panel_map:
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

        c.save()

    except Exception as e:
        print(f"Error: failed while writing PDF: {e}")
        return 2

    print(f"Wrote: {pdf_name}")
    if classavg_mrc:
        print(f"Class-average job: {class_job_uid}")
    if refinement_panel_map:
        print(f"Refinement job: {refine_job_uid}")
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

