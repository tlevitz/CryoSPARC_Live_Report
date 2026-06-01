#!/usr/bin/env python3
# coding: utf-8

from io import BytesIO
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .stats import workspace_attribute_limits


def render_single_scatterplot(
    parsed: List[dict],
    key: str,
    ylabel: str,
    y_min_line: Optional[float] = None,
    y_max_line: Optional[float] = None,
    binary: bool = False,
) -> Optional[Image.Image]:
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

    fig, ax = plt.subplots(figsize=(11, 1.45), dpi=180)

    # Marker sizes reduced roughly by half
    if pts["accepted"][0]:
        ax.scatter(pts["accepted"][0], pts["accepted"][1], s=4, alpha=0.7, c="#2c7fb8", label="accepted")
    if pts["rejected"][0]:
        ax.scatter(pts["rejected"][0], pts["rejected"][1], s=5, alpha=0.8, c="#d95f0e", label="rejected")
    if pts["other"][0]:
        ax.scatter(pts["other"][0], pts["other"][1], s=4, alpha=0.5, c="#888888", label="other")

    if y_min_line is not None:
        ax.axhline(float(y_min_line), color="#31a354", linestyle="--", linewidth=1)
    if y_max_line is not None:
        ax.axhline(float(y_max_line), color="#de2d26", linestyle="--", linewidth=1)

    # No internal title; PDF adds the title above each plot
    ax.set_xlabel("Exposure number", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)

    if binary:
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Rejected", "Accepted"])

    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout(pad=0.5)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_scatterplots(parsed: List[dict], ws: dict) -> List[Tuple[str, Image.Image]]:
    ctf_min, ctf_max = workspace_attribute_limits(ws, "ctf_fit_to_A")
    motion_min, motion_max = workspace_attribute_limits(ws, "max_intra_frame_motion")
    part_min, part_max = workspace_attribute_limits(ws, "total_extracted_particles")
    defocus_min, defocus_max = workspace_attribute_limits(ws, "average_defocus")

    plot_specs = [
        ("ctf_fit_A", "CTF Fit (Å)", "CTF Fit (Å)", ctf_min, ctf_max, False),
        ("defocus_A", "Defocus Avg (Å)", "Defocus Avg (Å)", defocus_min, defocus_max, False),
        ("max_inframe_motion", "Max In-Frame Motion", "Max In-Frame Motion", motion_min, motion_max, False),
        ("extracted_particles", "Total Particles Extracted", "Particles Extracted", part_min, part_max, False),
        ("status_binary", "Exposure Acceptance State", "Status", None, None, True),
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
        )
        if img is not None:
            plots.append((title, img))

    return plots

