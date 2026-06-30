#!/usr/bin/env python3
# coding: utf-8

from io import BytesIO
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .stats import workspace_attribute_limits
from .images import merge_nested_dicts
from generate_live_report import build_report_text_theme

def build_scatterplot_style(
    text_theme: Optional[dict] = None,
    overrides: Optional[dict] = None,
) -> dict:
    t = build_report_text_theme(text_theme)

    cfg = {
        "figsize": (11, 1.75),
        "dpi": 200,
        "xlabel_fontsize": int(t["plot_text"]),
        "ylabel_fontsize": int(t["plot_text"]),
        "tick_labelsize": int(t["plot_text"]),
        "legend_fontsize": int(t["plot_text"]),
        "title_fontsize": int(t["plot_title"]),
        "marker_size_accepted": 4,
        "marker_size_rejected": 5,
        "marker_size_other": 4,
        "grid_alpha": 0.25,
        "threshold_linewidth": 1.0,
        "tight_pad": 0.5,
        "show_legend": False,
        "show_internal_title": False,
    }
    return merge_nested_dicts(cfg, overrides)


def render_single_scatterplot(
    parsed: List[dict],
    key: str,
    ylabel: str,
    y_min_line: Optional[float] = None,
    y_max_line: Optional[float] = None,
    binary: bool = False,
    title: Optional[str] = None,
    text_theme: Optional[dict] = None,
    style: Optional[dict] = None,
) -> Optional[Image.Image]:
    cfg = build_scatterplot_style(text_theme=text_theme, overrides=style)

    pts = {
        "accepted": ([], []),
        "rejected": ([], []),
        "other": ([], []),
    }

    for e in parsed:
        x = e.get("exposure_number")
        y = e.get(key)
        if x is None or y is None:
            continue
        group = "accepted" if e.get("accepted") else ("rejected" if e.get("rejected") else "other")
        pts[group][0].append(int(x))
        pts[group][1].append(float(y))

    if not any(pts[g][0] for g in pts):
        return None

    fig, ax = plt.subplots(figsize=cfg["figsize"], dpi=cfg["dpi"])

    if pts["accepted"][0]:
        ax.scatter(
            pts["accepted"][0],
            pts["accepted"][1],
            s=cfg["marker_size_accepted"],
            alpha=1,
            c="#47c16eff",
            label="accepted",
        )
    if pts["rejected"][0]:
        ax.scatter(
            pts["rejected"][0],
            pts["rejected"][1],
            s=cfg["marker_size_rejected"],
            alpha=1,
            c="#481f70ff",
            label="rejected",
        )
    if pts["other"][0]:
        ax.scatter(
            pts["other"][0],
            pts["other"][1],
            s=cfg["marker_size_other"],
            alpha=1,
            c="#888888",
            label="other",
        )

    if y_min_line is not None:
        ax.axhline(
            float(y_min_line),
            color="#48494B",
            linestyle="--",
            linewidth=cfg["threshold_linewidth"],
        )
    if y_max_line is not None:
        ax.axhline(
            float(y_max_line),
            color="#48494B",
            linestyle="--",
            linewidth=cfg["threshold_linewidth"],
        )

#    ax.set_xlabel("Exposure number", fontsize=cfg["xlabel_fontsize"])
    ax.set_ylabel(ylabel, fontsize=cfg["ylabel_fontsize"])
    ax.tick_params(labelsize=cfg["tick_labelsize"])
    ax.grid(True, alpha=cfg["grid_alpha"])

    if cfg["show_internal_title"] and title:
        ax.set_title(title, fontsize=cfg["title_fontsize"])

    if binary:
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Rejected", "Accepted"], fontsize=cfg["tick_labelsize"])

    if cfg["show_legend"]:
        ax.legend(loc="upper right", fontsize=cfg["legend_fontsize"])

    fig.tight_layout(pad=cfg["tight_pad"])

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_scatterplots(
    parsed: List[dict],
    ws: dict,
    text_theme: Optional[dict] = None,
    style: Optional[dict] = None,
) -> List[Tuple[str, Image.Image]]:
    ctf_min, ctf_max = workspace_attribute_limits(ws, "ctf_fit_to_A")
    defocus_min, defocus_max = workspace_attribute_limits(ws, "average_defocus")
    motion_min, motion_max = workspace_attribute_limits(ws, "max_intra_frame_motion")
    total_motion_min, total_motion_max = workspace_attribute_limits(ws, "total_motion_dist")
    ice_min, ice_max = workspace_attribute_limits(ws, "ice_thickness_rel")
    part_min, part_max = workspace_attribute_limits(ws, "total_extracted_particles")

    plot_specs = [
        ("elapsed_minutes", "Time Since Start", f"Time Since\nStart (min)", None, None, False),
        ("ctf_fit_A", "CTF Fit", "CTF Fit (Å)", ctf_min, ctf_max, False),
        ("defocus_A", "Defocus Avg", f"Defocus\nAvg (Å)", defocus_min, defocus_max, False),
        ("max_inframe_motion", "Max In-Frame Motion", "Max In-\nFrame Motion", motion_min, motion_max, False),
        ("total_motion_pix", "Total Motion", "Total Motion\n(px)", total_motion_min, total_motion_max, False),
        ("ice_thickness_rel", "Relative Ice Thickness", "Rel. Ice\nThickness", ice_min, ice_max, False),
        ("extracted_particles", "Total Particles Extracted", "Particles\nExtracted", part_min, part_max, False),
    ]

    plots = []
    for key, title, ylabel, y_min, y_max, binary in plot_specs:
        img = render_single_scatterplot(
            parsed=parsed,
            key=key,
            ylabel=ylabel,
            y_min_line=y_min,
            y_max_line=y_max,
            binary=binary,
            title=title,
            text_theme=text_theme,
            style=style,
        )
        if img is not None:
            plots.append((title, img))

    return plots
