#!/usr/bin/env python3
# coding: utf-8


"""
Scatterplot rendering helpers for CryoSPARC Live reports.


Direct dependencies
-------------------
- matplotlib
- Pillow


Local dependencies
------------------
- cryosparc_live_report.stats
- cryosparc_live_report.textstyle
"""


from io import BytesIO
from typing import List, Tuple, Optional


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


from .stats import workspace_attribute_limits

from .textstyle import merge_nested_dicts, build_report_text_theme

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


        try:
            x = int(x)
            y = float(y)
        except Exception:
            continue


        group = "accepted" if e.get("accepted") else ("rejected" if e.get("rejected") else "other")
        pts[group][0].append(x)
        pts[group][1].append(y)


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




ALL_PLOT_SPECS = [
    {
        "key": "elapsed_minutes",
        "title": "Time Since Start",
        "ylabel": "Time Since\nStart (min)",
        "ws_attr": None,
        "binary": False,
        "default": True,
        "replaceable": True,
    },
    {
        "key": "ctf_fit_to_A",
        "title": "CTF Fit",
        "ylabel": "CTF Fit (Å)",
        "ws_attr": "ctf_fit_to_A",
        "binary": False,
        "default": True,
        "replaceable": False,
    },
    {
        "key": "average_defocus",
        "title": "Defocus Avg",
        "ylabel": "Defocus\nAvg (Å)",
        "ws_attr": "average_defocus",
        "binary": False,
        "default": True,
        "replaceable": False,
    },
    {
        "key": "max_intra_frame_motion",
        "title": "Max In-Frame Motion",
        "ylabel": "Max In-\nFrame Motion",
        "ws_attr": "max_intra_frame_motion",
        "binary": False,
        "default": True,
        "replaceable": False,
    },
    {
        "key": "total_motion_dist",
        "title": "Total Motion",
        "ylabel": "Total Motion\n(px)",
        "ws_attr": "total_motion_dist",
        "binary": False,
        "default": True,
        "replaceable": False,
    },
    {
        "key": "ice_thickness_rel",
        "title": "Relative Ice Thickness",
        "ylabel": "Rel. Ice\nThickness",
        "ws_attr": "ice_thickness_rel",
        "binary": False,
        "default": True,
        "replaceable": False,
    },
    {
        "key": "total_extracted_particles",
        "title": "Total Particles Extracted",
        "ylabel": "Particles\nExtracted",
        "ws_attr": "total_extracted_particles",
        "binary": False,
        "default": True,
        "replaceable": True,
    },
    {
        "key": "defocus_range",
        "title": "Defocus Range",
        "ylabel": "Defocus\nRange (Å)",
        "ws_attr": "defocus_range",
        "binary": False,
        "default": False,
        "replaceable": False,
    },
    {
        "key": "astigmatism_angle",
        "title": "Astigmatism Angle",
        "ylabel": "Astig.\nAngle (deg)",
        "ws_attr": "astigmatism_angle",
        "binary": False,
        "default": False,
        "replaceable": False,
    },
    {
        "key": "astigmatism",
        "title": "Astigmatism",
        "ylabel": "Astigmatism",
        "ws_attr": "astigmatism",
        "binary": False,
        "default": False,
        "replaceable": False,
    },
    {
        "key": "phase_shift",
        "title": "Phase Shift",
        "ylabel": "Phase\nShift (deg)",
        "ws_attr": "phase_shift",
        "binary": False,
        "default": False,
        "replaceable": False,
    },
    {
        "key": "df_tilt_angle",
        "title": "Sample Tilt",
        "ylabel": "Sample\nTilt (deg)",
        "ws_attr": "df_tilt_angle",
        "binary": False,
        "default": False,
        "replaceable": False,
    },
    {
        "key": "total_manual_picks",
        "title": "Total Manual Picks",
        "ylabel": "Manual\nPicks",
        "ws_attr": "total_manual_picks",
        "binary": False,
        "default": False,
        "replaceable": False,
    },
    {
        "key": "total_blob_picks",
        "title": "Total Blob Picks",
        "ylabel": "Blob\nPicks",
        "ws_attr": "total_blob_picks",
        "binary": False,
        "default": False,
        "replaceable": False,
    },
    {
        "key": "blob_pick_score_median",
        "title": "Median Blob Pick Score",
        "ylabel": "Median Blob\nPick Score",
        "ws_attr": "blob_pick_score_median",
        "binary": False,
        "default": False,
        "replaceable": False,
    },
    {
        "key": "total_template_picks",
        "title": "Total Template Picks",
        "ylabel": "Template\nPicks",
        "ws_attr": "total_template_picks",
        "binary": False,
        "default": False,
        "replaceable": False,
    },
    {
        "key": "template_pick_score_median",
        "title": "Median Template Pick Score",
        "ylabel": "Median Template\nPick Score",
        "ws_attr": "template_pick_score_median",
        "binary": False,
        "default": False,
        "replaceable": False,
    },
    {
        "key": "total_extracted_particles_manual",
        "title": "Total Manual Picker Particles Extracted",
        "ylabel": "Manual\nParticles",
        "ws_attr": "total_extracted_particles_manual",
        "binary": False,
        "default": False,
        "replaceable": False,
    },
    {
        "key": "total_extracted_particles_blob",
        "title": "Total Blob Picker Particles Extracted",
        "ylabel": "Blob\nParticles",
        "ws_attr": "total_extracted_particles_blob",
        "binary": False,
        "default": False,
        "replaceable": False,
    },
    {
        "key": "total_extracted_particles_template",
        "title": "Total Template Picker Particles Extracted",
        "ylabel": "Template\nParticles",
        "ws_attr": "total_extracted_particles_template",
        "binary": False,
        "default": False,
        "replaceable": False,
    },
]




def _render_plot_spec(
    parsed: List[dict],
    ws: dict,
    spec: dict,
    text_theme: Optional[dict] = None,
    style: Optional[dict] = None,
) -> Optional[dict]:
    if spec["ws_attr"] is None:
        y_min, y_max = None, None
    else:
        y_min, y_max = workspace_attribute_limits(ws, spec["ws_attr"])


    img = render_single_scatterplot(
        parsed=parsed,
        key=spec["key"],
        ylabel=spec["ylabel"],
        y_min_line=y_min,
        y_max_line=y_max,
        binary=spec.get("binary", False),
        title=spec["title"],
        text_theme=text_theme,
        style=style,
    )
    if img is None:
        return None


    out = dict(spec)
    out["img"] = img
    out["has_workspace_limit"] = (y_min is not None or y_max is not None)
    return out




def build_scatterplot_pages(
    parsed: List[dict],
    ws: dict,
    text_theme: Optional[dict] = None,
    style: Optional[dict] = None,
) -> List[Tuple[str, List[Tuple[str, Image.Image]]]]:
    rendered_specs = []
    for spec in ALL_PLOT_SPECS:
        rendered = _render_plot_spec(
            parsed=parsed,
            ws=ws,
            spec=spec,
            text_theme=text_theme,
            style=style,
        )
        if rendered is not None:
            rendered_specs.append(rendered)


    if not rendered_specs:
        return []


    default_plots = [s for s in rendered_specs if s["default"]]
    extra_plots = [s for s in rendered_specs if not s["default"]]
    extra_plots_with_limits = [s for s in extra_plots if s["has_workspace_limit"]]


    if not extra_plots_with_limits:
        return [
            ("Session Scatterplots", [(s["title"], s["img"]) for s in default_plots])
        ]


    replaceable_defaults = [
        s for s in default_plots
        if s.get("replaceable", False) and not s["has_workspace_limit"]
    ]


    if 1 <= len(extra_plots_with_limits) <= 2 and len(replaceable_defaults) >= len(extra_plots_with_limits):
        titles_to_replace = {
            s["title"] for s in replaceable_defaults[:len(extra_plots_with_limits)]
        }


        first_page = [s for s in default_plots if s["title"] not in titles_to_replace]
        first_page.extend(extra_plots_with_limits)


        return [
            ("Session Scatterplots", [(s["title"], s["img"]) for s in first_page])
        ]


    pages = [
        ("Session Scatterplots", [(s["title"], s["img"]) for s in default_plots])
    ]


    if extra_plots:
        pages.append(
            ("Session Scatterplots, cont.", [(s["title"], s["img"]) for s in extra_plots])
        )


    return pages




def build_scatterplots(
    parsed: List[dict],
    ws: dict,
    text_theme: Optional[dict] = None,
    style: Optional[dict] = None,
) -> List[Tuple[str, Image.Image]]:
    """
    Backward-compatible wrapper returning only the first page's plots.
    """
    pages = build_scatterplot_pages(
        parsed=parsed,
        ws=ws,
        text_theme=text_theme,
        style=style,
    )
    return pages[0][1] if pages else []


