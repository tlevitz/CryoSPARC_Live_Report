#!/usr/bin/env python3
# coding: utf-8

from typing import Optional


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
    """
    Shared top-level report text theme.

    Main knobs:
      plot_text      : axis labels, tick labels, legends, colorbar labels
      plot_title     : plot titles
      panel_header   : top line of representative micrograph panels
      panel_footer   : footer line of representative micrograph panels
      small_band_title : title above small image bands like 2D CTF / Particles
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
    """
    Convert shared text theme into:
      - style dict for make_micrograph_panel()
      - plot_cfg dict for micrograph subplots
    """
    t = build_report_text_theme(text_theme)

    style = {
        "title_font_size": int(t["panel_header"]),
        "body_font_size": int(t["panel_footer"]),
        "small_title_font_size": int(t["small_band_title"]),
        "small_title_band_h": 22,
        "small_title_y_pad": 2,
        "title_color": (0, 0, 0),
        "body_color": (20, 20, 20),
        "border_color": (185, 185, 185),
        "bg_color": "white",
    }

    plot_cfg = {
        "global_motion": {
            "title_fontsize": int(t["plot_title"]),
            "tick_labelsize": int(t["plot_text"]),
            "line_width": 1.2,
        },
        "local_motion": {
            "title_fontsize": int(t["plot_title"]),
            "traj_lw": 1.0,
            "viewer_scale": 40.0,
            "patch_spacing_A": 380.0,
            "patch_size_A": 500.0,
        },
        "local_defocus": {
            "title_fontsize": int(t["plot_title"]),
            "axis_label_fontsize": int(t["plot_text"]),
            "tick_labelsize": int(t["plot_text"]),
            "colorbar_label_fontsize": int(t["plot_text"]),
            "display_grid": 180,
            "elev": 28,
            "azim": -58,
            "cmap": "viridis",
            "z_half_range_A": 2500.0,
        },
        "ctf_1d": {
            "title_fontsize": int(t["plot_title"]),
            "xlabel_fontsize": int(t["plot_text"]),
            "ylabel_fontsize": int(t["plot_text"]),
            "tick_labelsize": int(t["plot_text"]),
            "legend_fontsize": int(t["plot_text"]),
            "top_axis_fontsize": int(t["plot_text"]),
            "top_tick_fontsize": int(t["plot_text"]),
            "render_scale": 3,
        },
        "particles": {
            "sample_n": 6,
            "cols": 2,
            "tile_inches": 1.6,
            "dpi": 100,
            "invert": False,
            "autoscale": "imshow",
            "p_lo": 0.5,
            "p_hi": 99.5,
            "wspace": 0.06,
            "hspace": 0.06,
            "add_indices": False,
        },
    }

    style = merge_nested_dicts(style, style_overrides)
    plot_cfg = merge_nested_dicts(plot_cfg, plot_overrides)
    return style, plot_cfg


def build_refinement_text_settings(
    text_theme: Optional[dict] = None,
    overrides: Optional[dict] = None,
):
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
