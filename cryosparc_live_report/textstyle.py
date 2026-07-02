#!/usr/bin/env python3
# coding: utf-8


"""
Shared text-style helpers for report rendering.


Direct dependencies
-------------------
- Python standard library only


Notes
-----
This module provides shared text-theme and micrograph-panel style builders.


Some of this logic overlaps with helpers currently defined in
`generate_live_report.py`. That duplication should be reconciled only after
reviewing all module consumers.
"""


from typing import Optional




def merge_nested_dicts(defaults: dict, user: Optional[dict]) -> dict:
    """
    Merge one level of nested dictionaries.


    If both `defaults[k]` and `user[k]` are dicts, merge them shallowly.
    Otherwise, `user[k]` replaces `defaults[k]`.
    """
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
    """
    Build a shared top-level report text theme.


    Main knobs:
      plot_text        : axis labels, tick labels, legends, colorbar labels
      plot_title       : plot titles
      panel_header     : top line of representative micrograph panels
      panel_footer     : footer line of representative micrograph panels
      small_band_title : title above small image bands like 2D CTF / particles
    """
    defaults = {
        "plot_text": 10,
        "plot_title": 12,
        "panel_header": 20,
        "panel_footer": 14,
        "small_band_title": None,
    }


    out = dict(defaults)
    if user:
        out.update(user)


    if out["small_band_title"] is None:
        out["small_band_title"] = out["plot_title"]


    return out




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
            "xlabel": "\nX (pix)",
            "ylabel": "\nY (pix)",
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





def build_refinement_text_settings(
    text_theme: Optional[dict] = None,
    overrides: Optional[dict] = None,
):
    """
    Convert a text theme into refinement-figure font sizing.
    """
    t = build_report_text_theme(text_theme)
    plot_text = int(t["plot_text"])


    cfg = {
        "title": int(t["plot_title"]),
        "axis": plot_text,
        "tick": plot_text,
        "legend": plot_text,
        "colorbar": plot_text,
        "panel_label": int(t["plot_title"]),
        "annotation": plot_text,
        "big_number": max(int(round(1.7 * t["plot_title"])), plot_text + 8),
        "surface_header": max(plot_text - 2, 8),
        "surface_label": max(plot_text - 2, 0),
    }


    if overrides:
        cfg.update(overrides)


    return cfg


