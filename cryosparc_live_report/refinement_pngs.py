#!/usr/bin/env python3


"""
Refinement and initial-model PNG generation helpers for CryoSPARC Live reports.


Direct dependencies
-------------------
- numpy
- mrcfile
- matplotlib
- scipy
- Pillow


Optional dependency
-------------------
- scikit-image


Local dependencies
------------------
- cryosparc_live_report.textstyle
- cryosparc_live_report.scale_bars
"""


import argparse
import json
import re
from pathlib import Path


import numpy as np
import mrcfile


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection


from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial.transform import Rotation


from PIL import Image, ImageDraw, ImageFont




try:
    from skimage.measure import marching_cubes
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False




try:
    from cryosparc_live_report.textstyle import build_refinement_text_settings
except Exception:
    def build_refinement_text_settings(text_theme=None, overrides=None):
        cfg = {
            "title": 12,
            "axis": 10,
            "tick": 10,
            "legend": 10,
            "colorbar": 10,
            "panel_label": 12,
            "annotation": 10,
            "big_number": 20,
            "surface_header": 9,
            "surface_label": 8,
        }
        if overrides:
            cfg.update(overrides)
        return cfg

try:
    from cryosparc_live_report.scale_bars import choose_scale_bar_for_display
except Exception:
    choose_scale_bar_for_display = None

REFINE_TEXT = build_refinement_text_settings()


def set_refinement_text_theme(text_theme=None, overrides=None):
    global REFINE_TEXT
    REFINE_TEXT = build_refinement_text_settings(text_theme, overrides=overrides)

# ============================================================
# file helpers
# ============================================================

def find_latest_iteration(folder: Path) -> int:
    pattern = re.compile(r"^.+_(\d{3})_.+\.(mrc|cs)$")
    iterations = []
    for f in folder.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                iterations.append(int(m.group(1)))
    if not iterations:
        raise FileNotFoundError(f"No iteration-style files found in {folder}")
    return max(iterations)


def get_iteration_file(folder: Path, iteration: int, suffix: str):
    matches = sorted(folder.glob(f"*_{iteration:03d}_{suffix}"))
    return matches[0] if matches else None

def find_initial_model_class_maps(folder: Path):
    """
    Find ab-initio / initial-model class maps named like:
        JX_class_00_final_volume.mrc
        JX_class_01_final_volume.mrc
        ...
    Returns:
        [(class_idx_int, Path), ...] sorted by class index
    """
    pattern = re.compile(r"^.+_class_(\d{2})_final_volume\.mrc$")
    class_maps = []


    for f in folder.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                class_maps.append((int(m.group(1)), f))


    return sorted(class_maps, key=lambda x: x[0])


def load_mrc(path: Path):
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
        voxel = None
        try:
            voxel = float(mrc.voxel_size.x)
            if voxel <= 0:
                voxel = None
        except Exception:
            voxel = None
    return data, voxel


def load_cs(path: Path):
    return np.load(path, allow_pickle=True)


def load_gsfsc_from_job_json(folder: Path, iteration: int):
    job_json = folder / "job.json"
    if not job_json.exists():
        return None
    try:
        data = json.loads(job_json.read_text())
        for item in data.get("progress", []):
            msg = str(item.get("message", "")).lower()
            if f"iteration {iteration}" in msg and "gsfsc" in item:
                return float(item["gsfsc"])
    except Exception:
        return None
    return None

def find_field_by_suffix(arr, suffix, prefer_contains=()):
    names = list(arr.dtype.names or [])
    candidates = [n for n in names if n.endswith(suffix)]
    if not candidates:
        raise ValueError(f"No field ending with {suffix!r} found.\nFields:\n" + "\n".join(names))

    for key in prefer_contains:
        subset = [n for n in candidates if key in n]
        if subset:
            candidates = subset
            break

    candidates = sorted(
        candidates,
        key=lambda n: (
            0 if "alignments3D" in n else 1,
            0 if "ctf/" in n else 1,
            len(n),
            n,
        ),
    )
    return candidates[0]


def maybe_find_field_by_suffix(arr, suffix, prefer_contains=()):
    try:
        return find_field_by_suffix(arr, suffix, prefer_contains=prefer_contains)
    except Exception:
        return None


# ============================================================
# utility helpers
# ============================================================

def robust_limits(arr, low=1.0, high=99.0):
    lo = float(np.percentile(arr, low))
    hi = float(np.percentile(arr, high))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def symmetric_limits(arr, percentile=99.5):
    a = float(np.percentile(np.abs(arr), percentile))
    if not np.isfinite(a) or a <= 0:
        a = float(np.max(np.abs(arr)))
    if not np.isfinite(a) or a <= 0:
        a = 1.0
    return -a, a


def choose_threshold(vol, mode, user_threshold=None):
    if user_threshold is not None:
        return float(user_threshold)

    min_val = np.min(vol)
    max_val = np.max(vol)

    if mode == "initial":
        return float(min_val + (max_val - min_val) / 4)
    elif mode == "refine":
        return float((min_val + max_val) / 2)
    else:
        raise ValueError("mode must be 'initial' or 'refine'")


def tick_positions(n, n_ticks=6):
    if n <= 1:
        return [0]
    vals = np.linspace(0, n - 1, n_ticks)
    vals = np.unique(np.round(vals).astype(int))
    return vals.tolist()

def is_nyquist_resolution(res_A, voxel_size_A, atol_A=0.05, rtol=0.01):
    """
    Return True if the reported resolution is effectively at Nyquist
    (2 * voxel_size), allowing for small rounding differences.
    """
    try:
        res_A = float(res_A)
        voxel_size_A = float(voxel_size_A)
    except Exception:
        return False

    if not np.isfinite(res_A) or not np.isfinite(voxel_size_A) or voxel_size_A <= 0:
        return False

    nyquist_A = 2.0 * voxel_size_A
    tol = max(float(atol_A), float(rtol) * nyquist_A)
    return abs(res_A - nyquist_A) <= tol

def _bin_centers(lo, hi, n):
    step = (hi - lo) / float(n)
    return lo + step * (np.arange(n, dtype=np.float64) + 0.5)


def _normalize_vectors(v, default=(0.0, 0.0, 1.0)):
    v = np.asarray(v, dtype=np.float64)

    if v.ndim == 1:
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n <= 0:
            return np.asarray(default, dtype=np.float64)
        return v / n

    if v.ndim == 2 and v.shape[1] == 3:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        out = v.copy()
        good = np.isfinite(n[:, 0]) & (n[:, 0] > 0)
        out[good] /= n[good]
        out[~good] = np.asarray(default, dtype=np.float64)
        return out

    raise ValueError(f"Expected shape (3,) or (N, 3), got {v.shape}")


def viewdir_to_az_el(viewdirs):
    """
    World-space object-to-camera unit vectors -> canonical az/el.
    Accepts shape (N, 3).
    """
    viewdirs = _normalize_vectors(viewdirs)
    if viewdirs.ndim != 2 or viewdirs.shape[1] != 3:
        raise ValueError(f"Expected viewdirs with shape (N, 3), got {viewdirs.shape}")

    x = viewdirs[:, 0]
    y = viewdirs[:, 1]
    z = np.clip(viewdirs[:, 2], -1.0, 1.0)

    az = np.arctan2(y, x)
    el = np.arcsin(z)

    # azimuth is undefined at poles; atan2(0,0) gives 0, which is fine
    return az.astype(np.float32), el.astype(np.float32)


def mpl_angles_to_viewdir(azim_deg, elev_deg):
    """
    Matplotlib mplot3d view_init(elev, azim) -> world-space object-to-camera vector.
    This matches mpl's camera placement convention.
    """
    az = np.deg2rad(float(azim_deg))
    el = np.deg2rad(float(elev_deg))

    v = np.array([
        -np.sin(az) * np.cos(el),
         np.cos(az) * np.cos(el),
         np.sin(el),
    ], dtype=np.float64)

    return _normalize_vectors(v)


def viewdir_to_mpl_angles(viewdir):
    """
    World-space object-to-camera vector -> matplotlib view_init(elev, azim).
    Inverse of mpl_angles_to_viewdir().
    """
    x, y, z = _normalize_vectors(viewdir)

    azim = np.arctan2(-x, y)
    elev = np.arcsin(np.clip(z, -1.0, 1.0))

    return float(np.rad2deg(azim)), float(np.rad2deg(elev))


def pi_tick_info():
    xticks = [-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    xlabels = ["-π", "-3π/4", "-π/2", "-π/4", "0", "π/4", "π/2", "3π/4", "π"]
    yticks = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
    ylabels = ["-π/2", "-π/4", "0", "π/4", "π/2"]
    return xticks, xlabels, yticks, ylabels


def add_panel_labels(fig, axes, labels, dy=0.01, fontsize=None):
    if fontsize is None:
        fontsize = REFINE_TEXT["panel_label"]
    for ax, lab in zip(axes, labels):
        bb = ax.get_position()
        xc = 0.5 * (bb.x0 + bb.x1)
        fig.text(xc, bb.y1 + dy, lab, ha="center", va="bottom", fontsize=fontsize)
        
def render_text_crop(text, font, fg=(0, 0, 0), bg=(255, 255, 255), pad=12, out_pad=3):
    """
    Render text to a temporary image, crop to visible bounds, then add a small
    white safety border so pasted text is never visibly clipped.
    """
    if not text:
        return Image.new("RGB", (1, 1), bg)


    W = max(128, int(8 * len(text) * max(8, getattr(font, "size", 12))))
    H = max(64, int(5 * max(8, getattr(font, "size", 12))))
    tmp = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(tmp)
    d.text((pad, pad), text, fill=fg, font=font)


    arr = np.asarray(tmp)
    bg_arr = np.array(bg, dtype=arr.dtype)
    mask = np.any(arr != bg_arr, axis=2)


    if not np.any(mask):
        return Image.new("RGB", (1, 1), bg)


    ys, xs = np.where(mask)
    x0 = max(0, int(xs.min()) - out_pad)
    x1 = min(W, int(xs.max()) + 1 + out_pad)
    y0 = max(0, int(ys.min()) - out_pad)
    y1 = min(H, int(ys.max()) + 1 + out_pad)


    return tmp.crop((x0, y0, x1, y1))
    
def load_pil_font(size=16, bold=False):
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]


    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass


    return ImageFont.load_default()

# ============================================================
# slice extraction
# order requested: YZ, XZ, XY
# ============================================================

def get_slices_yz_xz_xy(vol):
    zc, yc, xc = [s // 2 for s in vol.shape]

    # vol is indexed as vol[z, y, x]
    # Desired displayed orientations (described using a top-left image origin):
    #   YZ: y increases left->right, z increases top->bottom
    #   XZ: x increases left->right, z increases top->bottom
    #   XY: x increases left->right, y increases top->bottom
    #
    # Since these are shown with origin="lower", the vertical axis must be
    # reversed in the array so the second coordinate increases top->bottom.

    yz = vol[::-1, :, xc]      # rows=z reversed, cols=y
    xz = vol[::-1, yc, :]      # rows=z reversed, cols=x
    xy = vol[zc, ::-1, :]      # rows=y reversed, cols=x

    return [yz, xz, xy]

def format_slice_axes(ax, img, top_left_origin=True, put_x_ticks_on_top=True):
    h, w = img.shape[:2]

    xt = tick_positions(w)
    yt = tick_positions(h)

    ax.set_xticks(xt)
    ax.set_yticks(yt)

    if top_left_origin:
        # x increases left -> right
        xlabels = [str(t) for t in xt]

        # because imshow(..., origin="lower"), displayed y runs bottom -> top
        # so reverse the labels to make the displayed top be 0
        ylabels = [str(h - 1 - t) for t in yt]
    else:
        xlabels = [str(t) for t in xt]
        ylabels = [str(t) for t in yt]

    ax.set_xticklabels(xlabels, fontsize=REFINE_TEXT["tick"])
    ax.set_yticklabels(ylabels, fontsize=REFINE_TEXT["tick"])

    if put_x_ticks_on_top:
        ax.xaxis.tick_top()
        ax.tick_params(
            axis="x",
            labeltop=True,
            labelbottom=False,
            top=True,
            bottom=False,
            pad=2,
        )
    else:
        ax.tick_params(
            axis="x",
            labeltop=False,
            labelbottom=True,
            top=False,
            bottom=True,
            pad=2,
        )

    ax.tick_params(axis="y", pad=2)

# ============================================================
# real-space slices
# ============================================================

def save_slice_panel(
    vol,
    out_path: Path,
    title: str,
    cmap="viridis",
    symmetric=True,
    manual_vmax=None,
    embed_title=True,
):
    slices = get_slices_yz_xz_xy(vol)


    if manual_vmax is not None:
        vmax = abs(float(manual_vmax))
        vmin = -vmax
    elif symmetric:
        vmin, vmax = symmetric_limits(vol, percentile=99.5)
    else:
        vmin, vmax = robust_limits(vol)


    fig = plt.figure(figsize=(13.4, 4.9), dpi=200)
    gs = fig.add_gridspec(
        1, 3,
        left=0.06, right=0.87, bottom=0.12, top=0.83,
        wspace=0.24
    )


    axes = []
    last_im = None
    for i, img in enumerate(slices):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(img, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, aspect="equal")
        format_slice_axes(ax, img)
        axes.append(ax)
        last_im = im


    if embed_title and title:
        fig.suptitle(title, fontsize=REFINE_TEXT["title"], y=0.96)


    bb = axes[-1].get_position()
    cax = fig.add_axes([0.90, bb.y0, 0.018, bb.height])
    cb = fig.colorbar(last_im, cax=cax)
    cb.ax.tick_params(labelsize=REFINE_TEXT["colorbar"])


    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# stacked mask slices
# ============================================================

def save_stacked_mask_panel(mask_items, out_path: Path, title: str, cmap="viridis", embed_title=True):
    """
    mask_items: list of (row_label, volume)
    """
    if not mask_items:
        return

    all_vals = np.concatenate([m.ravel() for _, m in mask_items])
    vmin, vmax = robust_limits(all_vals, low=0.0, high=100.0)

    nrows = len(mask_items)
    fig = plt.figure(figsize=(13.4, 3.9 * nrows), dpi=200)
    gs = fig.add_gridspec(
        nrows, 3,
        left=0.08, right=0.87, bottom=0.06, top=0.88,
        hspace=0.65, wspace=0.24
    )

    all_row_axes = []
    last_im = None

    row_label_fs = REFINE_TEXT.get("mask_row_label", REFINE_TEXT["axis"] + 10)
    plane_label_fs = REFINE_TEXT.get("mask_plane_label", REFINE_TEXT["axis"] + 2)

    for r, (row_label, vol) in enumerate(mask_items):
        slices = get_slices_yz_xz_xy(vol)
        row_axes = []

        for c, img in enumerate(slices):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(img, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, aspect="equal")
            format_slice_axes(ax, img)
            row_axes.append(ax)
            last_im = im

        # Center row label above this row
        row_bb_left = row_axes[0].get_position()
        row_bb_right = row_axes[-1].get_position()
        x_center = 0.5 * (row_bb_left.x0 + row_bb_right.x1)
        y_top = row_bb_left.y1 + 0.03

        fig.text(
            x_center,
            y_top + 0.020,
            row_label,
            ha="center",
            va="bottom",
            fontsize=row_label_fs,
        )

        all_row_axes.append(row_axes)

    if embed_title and title:
        fig.suptitle(title, fontsize=REFINE_TEXT["title"], y=0.965)

    # Colorbar
    if all_row_axes:
        first_row_axes = all_row_axes[0]

        # Colorbar aligned to first row height
        bb = first_row_axes[-1].get_position()
        cax = fig.add_axes([0.90, bb.y0, 0.018, bb.height])
        cb = fig.colorbar(last_im, cax=cax)
        cb.ax.tick_params(labelsize=REFINE_TEXT["colorbar"])

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# threshold outputs
# ============================================================

def save_threshold_outputs(vol, out_png: Path, out_txt: Path, threshold_value):
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))

    fig = plt.figure(figsize=(4.2, 2.4), dpi=200)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.05, 0.78, "Threshold", fontsize=REFINE_TEXT["title"], weight="bold")
    ax.text(0.05, 0.43, f"{threshold_value:.3f}", fontsize=REFINE_TEXT["big_number"])
    ax.text(0.05, 0.14, f"Min: {vmin:.3f}  Max: {vmax:.3f}", fontsize=REFINE_TEXT["annotation"])
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_txt.write_text(
        f"Threshold: {threshold_value:.6f}\n"
        f"Min: {vmin:.6f}\n"
        f"Max: {vmax:.6f}\n"
    )


# ============================================================
# surface views
# ============================================================

def choose_display_map_path(map_path: Path, gsfsc_value, voxel_size_A=None):
    """
    Choose which map to use for display products (surface views / slices / lookup).

    Rules:
      - If GSFSC is unavailable: use unsharpened map
      - If GSFSC is at Nyquist: use unsharpened map
      - Else if GSFSC >= 3.5 Å: use unsharpened map
      - Else if GSFSC < 3.5 Å and sharpened map exists: use sharpened map
      - Else: fall back to unsharpened map

    Returns
    -------
    display_map_path : Path
    display_mode : str
        "sharpened" or "unsharpened"
    display_reason : str
        Human-readable reason for note text / logging
    """
    if gsfsc_value is None or not np.isfinite(gsfsc_value):
        return map_path, "unsharpened", "GSFSC unavailable"

    gsfsc_value = float(gsfsc_value)

    if voxel_size_A is not None and is_nyquist_resolution(gsfsc_value, voxel_size_A):
        nyquist_A = 2.0 * float(voxel_size_A)
        return (
            map_path,
            "unsharpened",
            f"GSFSC {gsfsc_value:.2f} Å is at Nyquist ({nyquist_A:.2f} Å)"
        )

    if gsfsc_value >= 3.5:
        return (
            map_path,
            "unsharpened",
            f"GSFSC {gsfsc_value:.2f} Å is >= 3.5 Å"
        )

    sharp_path = map_path.with_name(f"{map_path.stem}_sharp{map_path.suffix}")
    if sharp_path.exists():
        return (
            sharp_path,
            "sharpened",
            f"GSFSC {gsfsc_value:.2f} Å is < 3.5 Å"
        )

    print(
        f"[warn] GSFSC resolution {gsfsc_value:.2f} Å < 3.5 Å, "
        f"but sharpened map not found: {sharp_path.name}. "
        f"Using unsharpened map instead."
    )
    return (
        map_path,
        "unsharpened",
        f"GSFSC {gsfsc_value:.2f} Å is < 3.5 Å but sharpened map was not found"
    )

def _camera_direction(azim_deg, elev_deg):
    """
    World-space object-to-camera direction for matplotlib's view_init angles.
    """
    return mpl_angles_to_viewdir(azim_deg, elev_deg).astype(np.float32)


def _make_facecolors(normals, azim, elev):
    """
    Brighter grayscale shading with softer shadows.
    normals must be shape (N, 3).
    """
    normals = np.asarray(normals, dtype=np.float32)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(
            f"_make_facecolors expected normals with shape (N, 3), got {normals.shape}"
        )

    normals = _normalize_vectors(normals).astype(np.float32)
    view_dir = _camera_direction(azim, elev).astype(np.float32)

    key_light = np.array([0.30, 0.35, 0.88], dtype=np.float32)
    key_light /= np.linalg.norm(key_light)

    fill_light = np.array([-0.55, -0.15, 0.82], dtype=np.float32)
    fill_light /= np.linalg.norm(fill_light)

    diffuse_key = np.abs(np.einsum("ij,j->i", normals, key_light))
    diffuse_fill = np.abs(np.einsum("ij,j->i", normals, fill_light))

    half_vec = key_light + view_dir
    hn = float(np.linalg.norm(half_vec))
    if hn > 0:
        half_vec = half_vec / hn
    else:
        half_vec = key_light

    spec = np.clip(np.einsum("ij,j->i", normals, half_vec), 0.0, 1.0) ** 16

    shade = (
        0.24
        + 0.62 * diffuse_key
        + 0.18 * diffuse_fill
        + 0.10 * spec
    )

    gray = np.clip(0.25 + 0.45 * shade, 0.0, 1.0).astype(np.float32)

    facecolors = np.empty((normals.shape[0], 4), dtype=np.float32)
    facecolors[:, :3] = gray[:, None]
    facecolors[:, 3] = 1.0
    return facecolors

def _make_facecolors_from_viewdir(normals, view_dir):
    """
    Brighter grayscale shading with softer shadows, using an explicit
    object-to-camera view direction.
    """
    normals = np.asarray(normals, dtype=np.float32)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(
            f"_make_facecolors_from_viewdir expected normals with shape (N, 3), got {normals.shape}"
        )

    normals = _normalize_vectors(normals).astype(np.float32)
    view_dir = _normalize_vectors(view_dir).astype(np.float32)

    key_light = np.array([0.30, 0.35, 0.88], dtype=np.float32)
    key_light /= np.linalg.norm(key_light)

    fill_light = np.array([-0.55, -0.15, 0.82], dtype=np.float32)
    fill_light /= np.linalg.norm(fill_light)

    diffuse_key = np.abs(np.einsum("ij,j->i", normals, key_light))
    diffuse_fill = np.abs(np.einsum("ij,j->i", normals, fill_light))

    half_vec = key_light + view_dir
    hn = float(np.linalg.norm(half_vec))
    if hn > 0:
        half_vec = half_vec / hn
    else:
        half_vec = key_light

    spec = np.clip(np.einsum("ij,j->i", normals, half_vec), 0.0, 1.0) ** 16

    shade = (
        0.24
        + 0.62 * diffuse_key
        + 0.18 * diffuse_fill
        + 0.10 * spec
    )

    gray = np.clip(0.25 + 0.45 * shade, 0.0, 1.0).astype(np.float32)

    facecolors = np.empty((normals.shape[0], 4), dtype=np.float32)
    facecolors[:, :3] = gray[:, None]
    facecolors[:, 3] = 1.0
    return facecolors

def _compute_plot_lim(tri_verts, lim=None, margin=1.08):
    """
    Return a symmetric plot limit with extra breathing room.
    If `lim` is None, compute it from the mesh.
    """
    base_lim = float(np.abs(tri_verts).max()) if lim is None else float(lim)
    return base_lim * float(margin)


def _render_surface_panel_image(tri_verts, normals, azim, elev, lim, panel_px, dpi, roll=0):
    """
    Render a single 3D surface view into an RGBA image.


    `lim` is treated as a symmetric half-range around zero. A tiny extra
    safety factor is applied internally so faces do not touch the axes edge.
    """
    plot_lim = max(float(lim), 1e-6) * 1.01


    fig = plt.figure(
        figsize=(panel_px / dpi, panel_px / dpi),
        dpi=dpi,
        facecolor="white",
    )


    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d", facecolor="white")


    facecolors = _make_facecolors(normals, azim, elev)


    mesh = Poly3DCollection(
        tri_verts,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
    )
    ax.add_collection3d(mesh)


    ax.set_proj_type("ortho")
    ax.set_xlim(-plot_lim, plot_lim)
    ax.set_ylim(-plot_lim, plot_lim)
    ax.set_zlim(-plot_lim, plot_lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_axis_off()


    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img




def _render_surface_panel_projection_image(
    tri_verts,
    normals,
    horizontal_axis,
    vertical_axis,
    depth_axis,
    view_dir,
    lim,
    panel_px,
    dpi,
):
    """
    Render a principal orthographic projection of the surface mesh into an RGBA image.


    horizontal_axis, vertical_axis, depth_axis are indices into x/y/z = 0/1/2.


    Displayed convention:
      - horizontal axis increases left -> right
      - vertical axis increases top -> bottom


    Since matplotlib 2D axes increase upward, we negate the vertical coordinate
    so that the requested axis increases downward in the final image.
    """
    plot_lim = max(float(lim), 1e-6) * 1.01


    facecolors = _make_facecolors_from_viewdir(normals, view_dir)


    # Project triangles into 2D:
    #   u = chosen horizontal axis
    #   v = - chosen vertical axis   (so displayed top->bottom is increasing)
    u = tri_verts[:, :, horizontal_axis]
    v = -tri_verts[:, :, vertical_axis]
    polys_2d = np.stack([u, v], axis=2)


    # Painter's algorithm: draw far faces first, near faces last
    depth = tri_verts.mean(axis=1)[:, depth_axis]
    order = np.argsort(depth)


    fig = plt.figure(
        figsize=(panel_px / dpi, panel_px / dpi),
        dpi=dpi,
        facecolor="white",
    )


    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="white")


    coll = PolyCollection(
        polys_2d[order],
        facecolors=facecolors[order],
        edgecolors="none",
        linewidths=0.0,
        antialiaseds=False,
    )
    ax.add_collection(coll)


    ax.set_xlim(-plot_lim, plot_lim)
    ax.set_ylim(-plot_lim, plot_lim)
    ax.set_aspect("equal")
    ax.set_axis_off()


    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img




def _crop_white_border(img, white_threshold=250, pad=8):
    """
    Crop away white border around a rendered RGBA image.


    Kept here as a utility for other callers, but surface-view export below
    intentionally does NOT use cropping so all panels stay at a consistent scale.
    """
    if img.ndim != 3 or img.shape[2] < 3:
        return img


    rgb = img[..., :3]
    nonwhite = np.any(rgb < white_threshold, axis=2)


    if not np.any(nonwhite):
        return img


    ys, xs = np.where(nonwhite)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, img.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, img.shape[1])


    return img[y0:y1, x0:x1]




def _pad_to_square_canvas(img, side):
    """
    Center an image on a white square canvas so all panels display at the same size.
    """
    h, w = img.shape[:2]
    out = np.full((side, side, img.shape[2]), 255, dtype=img.dtype)


    y0 = (side - h) // 2
    x0 = (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out




def save_surface_views(
    vol,
    out_path: Path,
    title: str,
    threshold_value,
    step=2,
    spacing=(1.0, 1.0, 1.0),
    panel_px=900,
    dpi=200,
    embed_header=True,
    embed_plane_labels=True,
    scale_bar_length_A=None,
):
    """
    Save three principal orthographic surface projections (YZ, XZ, XY)
    laid out side-by-side.


    This version intentionally does NOT crop the rendered panel images.
    Instead, it renders each panel at a consistent square size with a small
    built-in plotting margin so the full object remains visible.
    """
    fig_header_px = 110 if embed_header else 16
    fig_footer_px = 16
    left_px = 36
    right_px = 36
    gap_px = 36


    fig_w_px = left_px + 3 * panel_px + 2 * gap_px + right_px
    fig_h_px = fig_header_px + panel_px + fig_footer_px


    fig = plt.figure(
        figsize=(fig_w_px / dpi, fig_h_px / dpi),
        dpi=dpi,
        facecolor="white",
    )


    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    level = float(threshold_value)


    if not HAVE_SKIMAGE:
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Surface render skipped:\nscikit-image not available",
            ha="center",
            va="center",
            fontsize=REFINE_TEXT["title"],
        )
        fig.savefig(out_path, facecolor="white", dpi=dpi)
        plt.close(fig)
        return


    ds = np.asarray(vol[::step, ::step, ::step], dtype=np.float32)


    smooth_sigma = 0.5
    if smooth_sigma > 0:
        ds = gaussian_filter(ds, sigma=smooth_sigma)


    ds_min = float(ds.min())
    ds_max = float(ds.max())
    if not (ds_min <= level <= ds_max):
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"Threshold {level:.3f} outside\nvolume range [{ds_min:.3f}, {ds_max:.3f}]",
            ha="center",
            va="center",
            fontsize=REFINE_TEXT["title"],
        )
        fig.savefig(out_path, facecolor="white", dpi=dpi)
        plt.close(fig)
        return


    mc_spacing = tuple(float(s) * step for s in spacing[::-1])


    try:
        verts, faces, normals, values = marching_cubes(
            ds,
            level=level,
            spacing=mc_spacing,
        )
        verts = verts[:, [2, 1, 0]]
    except Exception as e:
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"Surface render failed:\n{type(e).__name__}",
            ha="center",
            va="center",
            fontsize=REFINE_TEXT["title"],
        )
        fig.savefig(out_path, facecolor="white", dpi=dpi)
        plt.close(fig)
        return


    max_faces = 250_000
    if len(faces) > max_faces:
        keep = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[keep]


    verts = verts - verts.mean(axis=0)
    tri_verts = verts[faces]


    fn = np.cross(
        tri_verts[:, 1] - tri_verts[:, 0],
        tri_verts[:, 2] - tri_verts[:, 0],
    )
    fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
    fn_norm[fn_norm == 0] = 1.0
    fn = fn / fn_norm


    face_centers = tri_verts.mean(axis=1)
    flip = np.sum(fn * face_centers, axis=1) < 0
    fn[flip] *= -1.0


    spans = np.ptp(verts, axis=0)
    max_span = float(np.max(spans))
    if max_span <= 0:
        max_span = 1.0


    # Base half-span is 0.5 * max_span.
    # We expand slightly so the surface does not touch the render edges.
    lim = 0.5 * max_span * 1.5


    # Principal projections matched to the slice conventions:
    #   XY: x increases left->right, y increases top->bottom
    #   YZ: y increases left->right, z increases top->bottom
    #   XZ: x increases left->right, z increases top->bottom
    principal_views = [
        (
            "YZ Plane",
            dict(
                horizontal_axis=1,
                vertical_axis=2,
                depth_axis=0,
                view_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ),
        ),
        (
            "XZ Plane",
            dict(
                horizontal_axis=0,
                vertical_axis=2,
                depth_axis=1,
                view_dir=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
        ),
        (
            "XY Plane",
            dict(
                horizontal_axis=0,
                vertical_axis=1,
                depth_axis=2,
                view_dir=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            ),
        ),
    ]


    rendered_panels = []
    for lab, cfg in principal_views:
        img = _render_surface_panel_projection_image(
            tri_verts=tri_verts,
            normals=fn,
            horizontal_axis=cfg["horizontal_axis"],
            vertical_axis=cfg["vertical_axis"],
            depth_axis=cfg["depth_axis"],
            view_dir=cfg["view_dir"],
            lim=lim,
            panel_px=panel_px,
            dpi=dpi,
        )
        rendered_panels.append((lab, img))


    surface_label_fs = 8
    surface_header_fs = REFINE_TEXT.get("surface_header", 9)


    for i, (lab, img) in enumerate(rendered_panels):
        x0_px = left_px + i * (panel_px + gap_px)

        if i == 0:
            try:
                pil_img = Image.fromarray(img).convert("RGB")


                # Same styling philosophy that worked for the micrograph/particles
                scale_font_size = 36
                scale_thickness = 12
                scale_top_margin = 0
                scale_gap = 6
                scale_bottom_margin = 3


                plot_lim = max(float(lim), 1e-6) * 1.01
                base_display_angpix_A = (2.0 * plot_lim) / float(panel_px)


                chosen_bar_A, label_text, _ = choose_scale_bar_for_display(
                    display_size_px=panel_px,
                    display_angpix_A=base_display_angpix_A,
                    bar_length_A=None,
                    target_frac=0.22,
                    max_frac=0.33,
                    label_unit="A",
                )


                if chosen_bar_A is not None and label_text:
                    scale_font = load_pil_font(scale_font_size, bold=False)
                    label_img = render_text_crop(
                        label_text,
                        scale_font,
                        fg=(0, 0, 0),
                        bg=(255, 255, 255),
                        pad=12,
                        out_pad=3,
                    )
                    label_w, label_h = label_img.size


                    scale_strip_h = (
                        scale_top_margin
                        + scale_thickness
                        + scale_gap
                        + label_h
                        + scale_bottom_margin
                        + 2
                    )


                    image_h_budget = max(20, panel_px - scale_strip_h)


                    # Fit the square rendered panel into the reduced image area
                    fitted_surface = pil_img.copy()
                    fitted_surface.thumbnail((panel_px, image_h_budget), Image.Resampling.LANCZOS)


                    composed = Image.new("RGB", (panel_px, panel_px), (255, 255, 255))
                    d_comp = ImageDraw.Draw(composed)


                    img_x = (panel_px - fitted_surface.width) // 2
                    img_y = (image_h_budget - fitted_surface.height) // 2
                    composed.paste(fitted_surface, (img_x, img_y))


                    # Recompute displayed Å/px after fitting
                    display_angpix_A = (2.0 * plot_lim) / float(fitted_surface.width)
                    bar_px = int(round(float(chosen_bar_A) / float(display_angpix_A)))


                    strip_y0 = image_h_budget
                    side_margin_px = 16
                    x_left = img_x + side_margin_px
                    x_right = x_left + bar_px - 1


                    y_bar_top = strip_y0 + scale_top_margin
                    y_bar_bottom = y_bar_top + scale_thickness - 1
                    y_text = y_bar_bottom + 1 + scale_gap


                    x_text = x_left + (bar_px - label_w) // 2
                    x_text = max(0, min(panel_px - label_w, x_text))
                    y_text = max(strip_y0, min(panel_px - label_h, y_text))


                    d_comp.rectangle(
                        [x_left, y_bar_top, x_right, y_bar_bottom],
                        fill=(0, 0, 0),
                    )
                    composed.paste(label_img, (x_text, y_text))


                    img = np.asarray(composed)
            except Exception:
                pass


        ax = fig.add_axes(
            [
                x0_px / fig_w_px,
                fig_footer_px / fig_h_px,
                panel_px / fig_w_px,
                panel_px / fig_h_px,
            ],
            facecolor="white",
        )


        ax.imshow(img, interpolation="nearest")
        ax.set_axis_off()

        if embed_plane_labels:
            ax.text(
                0.5,
                0.0,
                lab,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=surface_label_fs,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="none",
                ),
            )


    vx = spacing[2]


    if embed_header:
        fig.text(
            0.5,
            0.985,
            f"{title}\nThreshold: {level:.3f}    Min: {vmin:.3f}    Max: {vmax:.3f}    "
            f"Voxel size: {vx:.3f} Å/px",
            ha="center",
            va="top",
            fontsize=surface_header_fs,
            linespacing=1.15,
        )


    fig.savefig(out_path, facecolor="white", dpi=dpi)
    plt.close(fig)




# ============================================================
# per-particle scale factors
# ============================================================

def pick_existing_field(arr, candidates):
    for name in candidates:
        if name in arr.dtype.names:
            return name
    return None


def save_alpha_histogram(particles_cs: Path, rejected_cs: Path, out_path: Path, title: str, cmap="viridis", embed_title=True):
    arr = load_cs(particles_cs)
    alpha_field = pick_existing_field(arr, ["alignments3D/alpha", "alignments2D/alpha"])
    if alpha_field is None:
        raise ValueError(f"No alpha field found in {particles_cs}")

    alpha = np.asarray(arr[alpha_field], dtype=np.float32)
    cm = plt.get_cmap(cmap)

    display_range = (0.0, 3.0)
    n_bins = 100

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.hist(
        alpha,
        bins=n_bins,
        range=display_range,
        alpha=0.78,
        color=cm(0.75),
        label=f"particles ({len(alpha)})"
    )

    if rejected_cs is not None and rejected_cs.exists():
        rej = load_cs(rejected_cs)
        rej_field = pick_existing_field(rej, ["alignments3D/alpha", "alignments2D/alpha"])
        if rej_field is not None:
            rej_alpha = np.asarray(rej[rej_field], dtype=np.float32)
            ax.hist(
                rej_alpha,
                bins=n_bins,
                range=display_range,
                alpha=0.42,
                color=cm(0.35),
                label=f"rejected ({len(rej_alpha)})"
            )

    ax.set_xlim(*display_range)
    if embed_title:
        ax.set_title(
            f"Mean: {np.mean(alpha):.3f}",
            fontsize=REFINE_TEXT["title"],
        )
    ax.set_xlabel(alpha_field, fontsize=REFINE_TEXT["axis"])
    ax.set_ylabel("# of particles", fontsize=REFINE_TEXT["axis"])
    ax.tick_params(axis="both", labelsize=REFINE_TEXT["tick"])
    ax.legend(fontsize=REFINE_TEXT["legend"], edgecolor="white")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# viewing direction / posterior precision plots
# ============================================================

def pose_rotvecs_to_viewdirs(pose, use_inverse=True):
    """
    Convert cryoSPARC pose rotvecs to world-space object-to-camera unit vectors.

    Convention used throughout this script:
        viewdir = object -> camera

    If use_inverse=True:
        use the 3rd row of R
    else:
        use the 3rd column of R
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 3:
        raise ValueError(f"Expected pose with shape (N, 3), got {pose.shape}")

    R = Rotation.from_rotvec(pose).as_matrix().astype(np.float32)

    if use_inverse:
        viewdirs = R[:, 2, :]      # 3rd row
    else:
        viewdirs = R[:, :, 2]      # 3rd column

    viewdirs = _normalize_vectors(viewdirs).astype(np.float32)
    return viewdirs

def viewdirs_to_az_el(viewdirs):
    return viewdir_to_az_el(viewdirs)

def make_direction_plot_geometry(bins=(72, 36), bin_equal_area=True):
    """
    Returns
    -------
    az_edges : (nx+1,)
    y_edges_hist : histogram y-edges
        - sin(el) if bin_equal_area=True
        - el otherwise
    el_edges_plot : plotting y-edges in true elevation coordinates
    """
    nx, ny = bins

    az_edges = np.linspace(-np.pi, np.pi, nx + 1, dtype=np.float64)

    if bin_equal_area:
        y_edges_hist = np.linspace(-1.0, 1.0, ny + 1, dtype=np.float64)
        el_edges_plot = np.arcsin(np.clip(y_edges_hist, -1.0, 1.0))
    else:
        y_edges_hist = np.linspace(-np.pi / 2, np.pi / 2, ny + 1, dtype=np.float64)
        el_edges_plot = y_edges_hist.copy()

    return az_edges.astype(np.float32), y_edges_hist.astype(np.float32), el_edges_plot.astype(np.float32)

def render_directional_plot(
    plot_data,
    az_edges,
    el_edges_plot,
    out_path: Path,
    title: str,
    log_scale: bool = False,
    cmap: str = "viridis",
    cbar_label: str = None,
    embed_title: bool = True,
):
    """
    plot_data must have shape (ny, nx)
    az_edges shape (nx+1,)
    el_edges_plot shape (ny+1,)

    This displays the y-axis in true elevation coordinates, so pi/4 is halfway
    between 0 and pi/2.
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color="0.85")

    data = np.asarray(plot_data, dtype=np.float64)
    masked = np.ma.masked_invalid(data)

    AZE, ELE = np.meshgrid(az_edges, el_edges_plot, indexing="xy")

    if log_scale:
        positive = data[np.isfinite(data) & (data > 0)]
        if positive.size == 0:
            norm = None
        else:
            norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
            masked = np.ma.masked_where(~np.isfinite(data) | (data <= 0), data)
    else:
        norm = None

    im = ax.pcolormesh(
        AZE,
        ELE,
        masked,
        shading="flat",
        cmap=cm,
        norm=norm,
    )

    xticks, xlabels, yticks, ylabels = pi_tick_info()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=REFINE_TEXT["title"])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=REFINE_TEXT["title"])

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi / 2, np.pi / 2)

    ax.set_xlabel("Azimuth", fontsize=REFINE_TEXT["title"])
    ax.set_ylabel("Elevation", fontsize=REFINE_TEXT["title"])
    if embed_title:
        ax.set_title(title, fontsize=REFINE_TEXT["title"])

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=REFINE_TEXT["colorbar"])
    if cbar_label is not None:
        cbar.set_label(cbar_label, fontsize=REFINE_TEXT["title"])

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_viewing_direction_distribution(
    particles_cs: Path,
    out_path: Path,
    title: str,
    bins=(72, 36),
    bin_equal_area=True,
    log_scale=True,
    cmap="viridis",
    cbar_label="# of images",
    use_inverse=True,
    embed_title=True,
):
    arr = load_cs(particles_cs)

    pose_field = find_field_by_suffix(arr, "/pose", prefer_contains=("alignments3D",))
    pose = np.asarray(arr[pose_field], dtype=np.float64)

    viewdirs = pose_rotvecs_to_viewdirs(pose, use_inverse=use_inverse)
    az, el = viewdirs_to_az_el(viewdirs)

    az_edges, y_edges_hist, el_edges_plot = make_direction_plot_geometry(
        bins=bins,
        bin_equal_area=bin_equal_area,
    )

    y_hist = np.sin(el) if bin_equal_area else el

    valid = np.isfinite(az) & np.isfinite(y_hist)
    az = az[valid]
    y_hist = y_hist[valid]

    H, _, _ = np.histogram2d(
        az,
        y_hist,
        bins=[az_edges, y_edges_hist],
    )

    plot_data = H.T  # (ny, nx)

    render_directional_plot(
        plot_data=plot_data,
        az_edges=az_edges,
        el_edges_plot=el_edges_plot,
        out_path=out_path,
        title=title,
        log_scale=log_scale,
        cmap=cmap,
        cbar_label=cbar_label,
        embed_title=embed_title,
    )

def electron_wavelength_A(accel_kv):
    V = np.asarray(accel_kv, dtype=np.float64) * 1000.0
    lam = 12.2639 / np.sqrt(V * (1.0 + 0.97845e-6 * V))
    return lam.astype(np.float32)


def mean_ctf2_over_band(
    alpha,
    df1_A,
    df2_A,
    df_angle_rad,
    accel_kv,
    cs_mm,
    amp_contrast,
    phase_shift_rad,
    freqs_Ainv,
):
    """
    alpha can be shape (L,) or (B, L).
    Returns shape (B, L).
    """
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.ndim == 1:
        alpha = alpha[None, :]

    df1_A = np.asarray(df1_A, dtype=np.float32).reshape(-1)
    df2_A = np.asarray(df2_A, dtype=np.float32).reshape(-1)
    df_angle_rad = np.asarray(df_angle_rad, dtype=np.float32).reshape(-1)
    accel_kv = np.asarray(accel_kv, dtype=np.float32).reshape(-1)
    cs_mm = np.asarray(cs_mm, dtype=np.float32).reshape(-1)
    amp_contrast = np.asarray(amp_contrast, dtype=np.float32).reshape(-1)
    phase_shift_rad = np.asarray(phase_shift_rad, dtype=np.float32).reshape(-1)
    freqs_Ainv = np.asarray(freqs_Ainv, dtype=np.float32)

    amp_contrast = np.where(amp_contrast > 1.0, amp_contrast / 100.0, amp_contrast)
    amp_contrast = np.clip(amp_contrast, 0.0, 0.999999)

    df_mean = 0.5 * (df1_A[:, None] + df2_A[:, None])
    df_diff = 0.5 * (df1_A[:, None] - df2_A[:, None])
    df = df_mean + df_diff * np.cos(2.0 * (alpha - df_angle_rad[:, None]))

    lam = electron_wavelength_A(accel_kv)[:, None, None]
    cs_A = (cs_mm * 1e7)[:, None, None]
    phase = phase_shift_rad[:, None, None]
    amp = amp_contrast[:, None, None]
    amp_sin = np.sqrt(1.0 - amp**2)

    k = freqs_Ainv[None, None, :]
    k2 = k * k
    k4 = k2 * k2

    chi = (
        np.pi * lam * df[:, :, None] * k2
        - 0.5 * np.pi * cs_A * (lam**3) * k4
        + phase
    )

    ctf = -(amp_sin * np.sin(chi) + amp * np.cos(chi))
    mean_ctf2 = np.mean(ctf * ctf, axis=2)

    return mean_ctf2.astype(np.float32)


def save_posterior_precision_directional_distribution_fast(
    particles_cs: Path,
    out_path: Path,
    title: str,
    bins=(72, 36),
    bin_equal_area=True,
    log_scale=False,
    cmap="viridis",
    cbar_label="Relative posterior precision (a.u.)",
    use_inverse=True,
    low_res_A=30.0,
    high_res_A=None,
    freq_samples=24,
    n_circle_samples=None,
    particle_chunk=1024,
    embed_title=True,
):
    """
    Fast approximation of posterior precision directional distribution.

    Each particle contributes along the great circle orthogonal to its viewing
    direction, weighted by mean CTF^2 over a resolution band.

    This is much faster than evaluating every particle against every direction bin.
    """
    arr = load_cs(particles_cs)

    # ---------- pose ----------
    pose_field = find_field_by_suffix(arr, "/pose", prefer_contains=("alignments3D",))
    pose = np.asarray(arr[pose_field], dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 3:
        raise ValueError(f"Pose field {pose_field!r} has unexpected shape {pose.shape}")

    n_particles = pose.shape[0]

    # ---------- CTF fields ----------
    df1_field = find_field_by_suffix(arr, "/df1_A", prefer_contains=("ctf",))
    df2_field = find_field_by_suffix(arr, "/df2_A", prefer_contains=("ctf",))
    dfang_field = find_field_by_suffix(arr, "/df_angle_rad", prefer_contains=("ctf",))
    kv_field = find_field_by_suffix(arr, "/accel_kv", prefer_contains=("ctf",))
    cs_field = find_field_by_suffix(arr, "/cs_mm", prefer_contains=("ctf",))
    amp_field = find_field_by_suffix(arr, "/amp_contrast", prefer_contains=("ctf",))
    psize_field = find_field_by_suffix(arr, "/psize_A", prefer_contains=("blob",))
    phase_field = maybe_find_field_by_suffix(arr, "/phase_shift_rad", prefer_contains=("ctf",))

    df1_A = np.asarray(arr[df1_field], dtype=np.float32).reshape(-1)
    df2_A = np.asarray(arr[df2_field], dtype=np.float32).reshape(-1)
    df_angle_rad = np.asarray(arr[dfang_field], dtype=np.float32).reshape(-1)
    accel_kv = np.asarray(arr[kv_field], dtype=np.float32).reshape(-1)
    cs_mm = np.asarray(arr[cs_field], dtype=np.float32).reshape(-1)
    amp_contrast = np.asarray(arr[amp_field], dtype=np.float32).reshape(-1)
    psize_A = np.asarray(arr[psize_field], dtype=np.float32).reshape(-1)

    if phase_field is None:
        phase_shift_rad = np.zeros(n_particles, dtype=np.float32)
    else:
        phase_shift_rad = np.asarray(arr[phase_field], dtype=np.float32).reshape(-1)

    # ---------- validity ----------
    valid = np.all(np.isfinite(pose), axis=1)
    valid &= np.isfinite(df1_A)
    valid &= np.isfinite(df2_A)
    valid &= np.isfinite(df_angle_rad)
    valid &= np.isfinite(accel_kv)
    valid &= np.isfinite(cs_mm)
    valid &= np.isfinite(amp_contrast)
    valid &= np.isfinite(psize_A)
    valid &= np.isfinite(phase_shift_rad)

    pose = pose[valid]
    df1_A = df1_A[valid]
    df2_A = df2_A[valid]
    df_angle_rad = df_angle_rad[valid]
    accel_kv = accel_kv[valid]
    cs_mm = cs_mm[valid]
    amp_contrast = amp_contrast[valid]
    psize_A = psize_A[valid]
    phase_shift_rad = phase_shift_rad[valid]

    if pose.shape[0] == 0:
        raise ValueError("No valid particles remain after filtering.")

    # ---------- resolution band ----------
    nyquist_A = 2.0 * np.nanmedian(psize_A)
    if high_res_A is None:
        high_res_A = nyquist_A

    fmin = 0.0 if low_res_A is None else 1.0 / float(low_res_A)
    fmax = 1.0 / float(high_res_A)

    if fmin >= fmax:
        raise ValueError(
            f"Empty resolution band: low_res_A={low_res_A}, high_res_A={high_res_A}"
        )

    freqs_Ainv = np.linspace(max(fmin, 1e-6), fmax, int(freq_samples), dtype=np.float32)

    # ---------- plotting geometry ----------
    az_edges, y_edges_hist, el_edges_plot = make_direction_plot_geometry(
        bins=bins,
        bin_equal_area=bin_equal_area,
    )

    nx, ny = bins
    Hxy_total = np.zeros((nx, ny), dtype=np.float64)

    # ---------- great-circle sampling ----------
    if n_circle_samples is None:
        n_circle_samples = max(240, 4 * bins[0])

    alpha = np.linspace(-np.pi, np.pi, int(n_circle_samples), endpoint=False, dtype=np.float32)
    ca = np.cos(alpha).astype(np.float32)
    sa = np.sin(alpha).astype(np.float32)

    ex = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    for start in range(0, pose.shape[0], particle_chunk):
        end = min(start + particle_chunk, pose.shape[0])

        pose_c = pose[start:end]
        B = pose_c.shape[0]

        R = Rotation.from_rotvec(pose_c).as_matrix().astype(np.float32)

        # World-space basis vectors spanning each particle's central plane,
        # kept consistent with pose_rotvecs_to_viewdirs().
        if use_inverse:
            ex_w = R[:, 0, :]   # row 0
            ey_w = R[:, 1, :]   # row 1
        else:
            ex_w = R[:, :, 0]   # col 0
            ey_w = R[:, :, 1]   # col 1

        # Great-circle directions in world coordinates:
        # u(alpha) = cos(alpha) ex_w + sin(alpha) ey_w
        dirs = (
            ex_w[:, None, :] * ca[None, :, None]
            + ey_w[:, None, :] * sa[None, :, None]
        )

        # Convert sampled directions to az/el
        x = dirs[:, :, 0]
        y = dirs[:, :, 1]
        z = np.clip(dirs[:, :, 2], -1.0, 1.0)

        az = np.arctan2(y, x).astype(np.float32)
        el = np.arcsin(z).astype(np.float32)
        y_hist = np.sin(el) if bin_equal_area else el

        # CTF information weight along sampled in-plane angle alpha
        info_w = mean_ctf2_over_band(
            alpha=alpha,
            df1_A=df1_A[start:end],
            df2_A=df2_A[start:end],
            df_angle_rad=df_angle_rad[start:end],
            accel_kv=accel_kv[start:end],
            cs_mm=cs_mm[start:end],
            amp_contrast=amp_contrast[start:end],
            phase_shift_rad=phase_shift_rad[start:end],
            freqs_Ainv=freqs_Ainv,
        )

        # Normalize by number of circle samples so each particle contributes
        # roughly a finite total amount.
        weights = info_w / float(n_circle_samples)

        Hxy, _, _ = np.histogram2d(
            az.ravel(),
            y_hist.ravel(),
            bins=[az_edges, y_edges_hist],
            weights=weights.ravel(),
        )

        Hxy_total += Hxy

    plot_data = Hxy_total.T  # (ny, nx)

    render_directional_plot(
        plot_data=plot_data,
        az_edges=az_edges,
        el_edges_plot=el_edges_plot,
        out_path=out_path,
        title=title,
        log_scale=log_scale,
        cmap=cmap,
        cbar_label=cbar_label,
        embed_title=embed_title,
    )


# ============================================================
# FSC
# ============================================================

def frequency_radius_grid(shape, voxel_size):
    z = np.fft.fftfreq(shape[0], d=voxel_size)
    y = np.fft.fftfreq(shape[1], d=voxel_size)
    x = np.fft.fftfreq(shape[2], d=voxel_size)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return np.sqrt(xx**2 + yy**2 + zz**2)

def smooth_1d_nan_safe(y, sigma_bins=1.0):
    y = np.asarray(y, dtype=np.float32)

    if sigma_bins is None or sigma_bins <= 0:
        return y.copy()

    good = np.isfinite(y)
    if not np.any(good):
        return y.copy()

    y0 = np.where(good, y, 0.0).astype(np.float32)
    w0 = good.astype(np.float32)

    ys = gaussian_filter1d(y0, sigma=sigma_bins, mode="nearest")
    ws = gaussian_filter1d(w0, sigma=sigma_bins, mode="nearest")

    out = np.divide(ys, ws, out=np.copy(y0), where=ws > 1e-6)
    return out


def compute_fsc(vol_a, vol_b, voxel_size, mask=None, n_bins=200):
    a = np.asarray(vol_a, dtype=np.float32)
    b = np.asarray(vol_b, dtype=np.float32)

    if mask is not None:
        a = a * mask
        b = b * mask

    Fa = np.fft.fftn(a)
    Fb = np.fft.fftn(b)
    r = frequency_radius_grid(a.shape, voxel_size).ravel()

    cross = (Fa * np.conj(Fb)).real.ravel()
    p1 = (np.abs(Fa) ** 2).ravel()
    p2 = (np.abs(Fb) ** 2).ravel()

    nyquist_freq = 1.0 / (2.0 * voxel_size)
    bins = np.linspace(0.0, nyquist_freq, n_bins + 1)

    which = np.digitize(r, bins) - 1
    good = (which >= 0) & (which < n_bins)

    num = np.bincount(which[good], weights=cross[good], minlength=n_bins)
    den1 = np.bincount(which[good], weights=p1[good], minlength=n_bins)
    den2 = np.bincount(which[good], weights=p2[good], minlength=n_bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = num / np.sqrt(den1 * den2)

    #I think this top one (which takes the middle of the range of FSC values in each bin as the x-axis value) is more
    #technically correct, but choosing just the most favorable number lines up more closely with CS' numbers :)
#    freq = 0.5 * (bins[:-1] + bins[1:])
    freq = (bins[1:])
    valid = np.isfinite(fsc) & (freq > 0)
    return freq[valid], fsc[valid]


def fsc_start_resolution(freq, fsc, voxel_size, ignore_first_bins=6, max_start_res=40.0):
    nyquist_res = 2.0 * voxel_size

    idx = None
    start_i = min(ignore_first_bins, max(0, len(fsc) - 1))

    for i in range(start_i, len(fsc)):
        if fsc[i] < 0.999:
            idx = max(start_i, i - 1)
            break

    if idx is None:
        idx = start_i if len(freq) > start_i else 0

    start_res = 1.0 / freq[idx]
    if not np.isfinite(start_res):
        start_res = max_start_res

    start_res = min(start_res, max_start_res)

    if start_res <= nyquist_res:
        start_res = nyquist_res * 1.15

    return start_res, nyquist_res


def set_fsc_xticks(ax, start_res, nyquist_res, gsfsc_value=None):
    ticks = [t for t in ax.get_xticks() if nyquist_res <= t <= start_res]

    # If a GSFSC marker exists, optionally remove auto ticks too close to it
    # so the separate GSFSC label has some room.
    if gsfsc_value is not None and np.isfinite(gsfsc_value):
        ticks = [t for t in ticks if abs(t - gsfsc_value) > 0.25]

    ticks = sorted(ticks, reverse=True)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.1f}" for t in ticks])


def save_fsc_plot(
    half_a_path: Path,
    half_b_path: Path,
    mask_fsc_path: Path,
    mask_auto_path: Path,
    out_path: Path,
    title: str,
    pixel_size_override=None,
    gsfsc_value=None,
    cmap="viridis",
    smooth_sigma_bins=1,
    embed_title=True,
):
    half_a, voxel_a = load_mrc(half_a_path)
    half_b, voxel_b = load_mrc(half_b_path)
    voxel = pixel_size_override or voxel_a or voxel_b or 1.0

    mask_fsc = None
    if mask_fsc_path is not None and mask_fsc_path.exists():
        mask_fsc, _ = load_mrc(mask_fsc_path)

    mask_auto = None
    if mask_auto_path is not None and mask_auto_path.exists():
        mask_auto, _ = load_mrc(mask_auto_path)

    curves = []

    freq0, fsc0 = compute_fsc(half_a, half_b, voxel, mask=None)
    curves.append(("No Mask", freq0, fsc0))

    if mask_fsc is not None:
        freq1, fsc1 = compute_fsc(half_a, half_b, voxel, mask=mask_fsc)
        curves.append(("Res Mask", freq1, fsc1))

    if mask_auto is not None:
        freq2, fsc2 = compute_fsc(half_a, half_b, voxel, mask=mask_auto)
        curves.append(("Res Mask, corr", freq2, fsc2))

    start_res, nyquist_res = fsc_start_resolution(freq0, fsc0, voxel)

    fig, ax = plt.subplots(figsize=(7.6, 5.1), dpi=200)
    cm = plt.get_cmap(cmap)
    colors = [cm(v) for v in np.linspace(0.2, 0.85, max(3, len(curves)))]

    for color, (label, freq, fsc) in zip(colors, curves):
        fsc_plot = smooth_1d_nan_safe(fsc, sigma_bins=smooth_sigma_bins)
        res = 1.0 / freq
        good = np.isfinite(res) & np.isfinite(fsc_plot)
        ax.plot(res[good], fsc_plot[good], label=label, linewidth=2, color=color)

    ax.axhline(0.143, color="black", linestyle="--", linewidth=1)

    if gsfsc_value is not None and np.isfinite(gsfsc_value):
        if nyquist_res <= gsfsc_value <= start_res:
            ax.axvline(gsfsc_value, color="0.65", linestyle="-", linewidth=1.5, zorder=0)

    ax.set_xlim(start_res, nyquist_res)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Resolution (Å)", fontsize=REFINE_TEXT["axis"])
    ax.set_ylabel("FSC", fontsize=REFINE_TEXT["axis"])
    ax.tick_params(axis="both", labelsize=REFINE_TEXT["tick"])

    set_fsc_xticks(ax, start_res, nyquist_res, gsfsc_value=gsfsc_value)

    if embed_title:
        if gsfsc_value is not None:
            ax.set_title(f"{title}\nGSFSC Resolution: {gsfsc_value:.2f}Å", fontsize=REFINE_TEXT["title"])
        else:
            ax.set_title(title, fontsize=REFINE_TEXT["title"])

    ax.legend(fontsize=REFINE_TEXT["legend"], edgecolor="white")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Guinier
# x plotted in 1/d^2, tick labels shown in Å
# ============================================================

def radial_profile_from_values(radius, values, n_bins=160, max_radius=None):
    radius = radius.ravel()
    values = values.ravel()

    rmax = float(np.max(radius)) if max_radius is None else float(max_radius)
    bins = np.linspace(0, rmax, n_bins + 1)
    which = np.digitize(radius, bins) - 1
    good = (which >= 0) & (which < n_bins)

    num = np.bincount(which[good], weights=values[good], minlength=n_bins)
    den = np.bincount(which[good], minlength=n_bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        prof = num / den

    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, prof


def radial_amplitude_profile(vol, voxel_size, mask=None, n_bins=160):
    arr = np.asarray(vol, dtype=np.float32)
    if mask is not None:
        arr = arr * mask

    F = np.fft.fftn(arr)
    amp = np.abs(F)

    nyquist_freq = 1.0 / (2.0 * voxel_size)
    r = frequency_radius_grid(arr.shape, voxel_size)
    freq, prof = radial_profile_from_values(r, amp, n_bins=n_bins, max_radius=nyquist_freq)

    good = np.isfinite(prof) & (prof > 0) & (freq > 0)
    freq = freq[good]
    prof = prof[good]

    # Normalize so low-frequency region is near 1, making logF near 0
    prof = prof / np.max(prof)

    res = 1.0 / freq
    logf = np.log(prof + 1e-12)
    return freq, res, logf


def save_guinier_plot(map_path: Path, mask_path: Path, out_path: Path, title: str,
                      pixel_size_override=None, cmap="viridis", gsfsc=None, embed_title=True):
    vol, voxel0 = load_mrc(map_path)
    voxel = pixel_size_override or voxel0 or 1.0

    mask = None
    if mask_path is not None and mask_path.exists():
        mask, _ = load_mrc(mask_path)

    freq, res, logf = radial_amplitude_profile(vol, voxel, mask=mask, n_bins=160)
    x2 = freq ** 2  # x-axis is still 1/d^2 internally

    # Plot range: up to the FSC=0.143 resolution if provided
    if gsfsc is not None and gsfsc > 0:
        xmax_plot = 1.0 / (gsfsc ** 2)
        keep = x2 <= xmax_plot
        x2_plot = x2[keep]
        logf_plot = logf[keep]
    else:
        xmax_plot = np.max(x2)
        x2_plot = x2
        logf_plot = logf

    if len(x2_plot) == 0:
        raise ValueError("No Guinier data available in the requested plotting range.")

    # ------------------------------------------------------------
    # Fit straight-line envelope between 10 Å and GSFSC
    # Only draw if GSFSC < 10 Å
    # ------------------------------------------------------------
    slope = None
    intercept = None
    b_factor = None
    x2_fit = None
    fit_y = None

    if gsfsc is not None and gsfsc > 0 and gsfsc < 10.0:
        fit_start_x2 = 1.0 / (10.0 ** 2)   # 10 Å
        fit_end_x2 = 1.0 / (gsfsc ** 2)    # GSFSC

        xlo, xhi = sorted((fit_start_x2, fit_end_x2))
        fit_mask = (
            np.isfinite(x2_plot)
            & np.isfinite(logf_plot)
            & (x2_plot >= xlo)
            & (x2_plot <= xhi)
        )

        if np.count_nonzero(fit_mask) >= 2:
            x2_fit = x2_plot[fit_mask]
            y_fit_data = logf_plot[fit_mask]

            slope, intercept = np.polyfit(x2_fit, y_fit_data, 1)
            fit_y = slope * x2_fit + intercept

            # For log(amplitude), B = -4 * slope
            b_factor = -4.0 * slope

    # ------------------------------------------------------------
    # X ticks:
    #   leftmost tick = DC
    #   remaining ticks chosen from approximately even x-spacing,
    #   then snapped to whole-number resolution labels
    # ------------------------------------------------------------
    def make_resolution_ticks_with_dc(xmax, max_ticks=7):
        if not np.isfinite(xmax) or xmax <= 0:
            return [0.0], ["DC"]

        # Number of non-DC ticks
        n_other = max_ticks - 1

        # Evenly spaced target positions in x = 1/d^2
        target_x = np.linspace(0.0, xmax, n_other + 1)[1:]

        # Smallest visible resolution in Å at the right edge
        d_min_visible = 1.0 / np.sqrt(xmax)

        d_values = []
        used = set()

        for tx in target_x:
            d_target = 1.0 / np.sqrt(tx)
            d_round = int(np.rint(d_target))

            # Keep tick inside visible plotting range
            d_round = max(d_round, int(np.ceil(d_min_visible)))

            # Avoid duplicate labels by trying nearby integers
            if d_round in used:
                found = None
                for delta in [1, -1, 2, -2, 3, -3]:
                    cand = d_round + delta
                    if cand < int(np.ceil(d_min_visible)):
                        continue
                    if cand in used:
                        continue
                    if (1.0 / (cand ** 2)) <= xmax:
                        found = cand
                        break
                if found is None:
                    continue
                d_round = found

            if (1.0 / (d_round ** 2)) <= xmax and d_round not in used:
                d_values.append(d_round)
                used.add(d_round)

        # Larger Å should appear more to the left
        d_values = sorted(d_values, reverse=True)

        tick_positions = [0.0] + [1.0 / (d ** 2) for d in d_values]
        tick_labels = ["DC"] + [f"{d}" for d in d_values]
        return tick_positions, tick_labels

    tick_positions, tick_labels = make_resolution_ticks_with_dc(xmax_plot, max_ticks=7)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    cm = plt.get_cmap(cmap)  # viridis by default

    fig, ax = plt.subplots(figsize=(7.6, 5.1), dpi=200)

    # Main Guinier curve
    ax.plot(
        x2_plot,
        logf_plot,
        linewidth=2,
        color=cm(0.78)
    )

    # Dashed envelope line, only over the fit interval
    if x2_fit is not None and fit_y is not None:
        ax.plot(
            x2_fit,
            fit_y,
            "--",
            linewidth=1.5,
            color=cm(0.35)
        )

    left_pad = 0.03 * xmax_plot
    ax.set_xlim(-left_pad, xmax_plot)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Resolution (Å)", fontsize=REFINE_TEXT["axis"])
    ax.set_ylabel("log F", fontsize=REFINE_TEXT["axis"])
    if embed_title:
        ax.set_title(title, fontsize=REFINE_TEXT["title"])
    ax.tick_params(axis="both", labelsize=REFINE_TEXT["tick"])

    # Replace legend with B-factor text
    if b_factor is not None:
        ax.text(
            0.98, 0.98,
            f"B-factor = {b_factor:.0f} Å²",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=REFINE_TEXT["legend"],
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="white"
            )
        )
    else:
        ax.text(
            0.98, 0.98,
            "B-factor = n/a",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=REFINE_TEXT["legend"],
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="white"
            )
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# protein view lookup chart
# ============================================================

def save_protein_view_lookup(
    vol,
    out_path: Path,
    title: str,
    threshold_value,
    step=1,
    spacing=(1.0, 1.0, 1.0),
    panel_px=240,
    dpi=300,
    embed_title=True,
):
    """
    Render a 4x8 lookup chart of isosurface views corresponding to
    azimuth/elevation bins used for directional plots.

    Uses simple bin-center logic:
      - 8 azimuth bin centers over [-pi, pi)
      - 4 elevation bin centers over [-pi/2, pi/2]

    Keeps the first script's surface rendering and the first script's
    single-image lookup layout with azimuth/elevation axes.
    """
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    level = float(threshold_value)

    fig = plt.figure(figsize=(12.5, 6.8), dpi=dpi, facecolor="white")

    if not HAVE_SKIMAGE:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "Protein view lookup skipped:\nscikit-image not available",
            ha="center", va="center", fontsize=REFINE_TEXT["title"]
        )
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    ds = np.asarray(vol[::step, ::step, ::step], dtype=np.float32)

    smooth_sigma = 0.5
    if smooth_sigma > 0:
        ds = gaussian_filter(ds, sigma=smooth_sigma)

    ds_min = float(ds.min())
    ds_max = float(ds.max())
    if not (ds_min <= level <= ds_max):
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            f"Protein view lookup failed:\n"
            f"Threshold {level:.3f} outside volume range [{ds_min:.3f}, {ds_max:.3f}]",
            ha="center", va="center", fontsize=14
        )
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    # volume is indexed as vol[z, y, x], so marching_cubes spacing must follow (z, y, x)
    mc_spacing = tuple(float(s) * step for s in spacing[::-1])


    try:
        verts, faces, normals, values = marching_cubes(
            ds,
            level=level,
            spacing=mc_spacing,
        )
        verts = verts[:, [2, 1, 0]]
    except Exception as e:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            f"Protein view lookup failed:\n{type(e).__name__}",
            ha="center", va="center", fontsize=14
        )
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    max_faces = 250_000
    if len(faces) > max_faces:
        keep = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[keep]

    verts = verts - verts.mean(axis=0)
    tri_verts = verts[faces]

    fn = np.cross(tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0])
    fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
    fn_norm[fn_norm == 0] = 1.0
    fn = fn / fn_norm

    face_centers = tri_verts.mean(axis=1)
    flip = np.sum(fn * face_centers, axis=1) < 0
    fn[flip] *= -1.0

    spans = np.ptp(verts, axis=0)
    max_span = float(np.max(spans))
    if max_span <= 0:
        max_span = 1.0
    lim = 0.55 * max_span


    # Simple bin-center logic from the second script
    az_edges = np.linspace(-np.pi, np.pi, 9, dtype=np.float64)
    el_edges = np.linspace(-np.pi / 2, np.pi / 2, 5, dtype=np.float64)

    az_centers = _bin_centers(-np.pi, np.pi, 8)
    el_centers = _bin_centers(-np.pi / 2, np.pi / 2, 4)

    crop_pad = max(6, panel_px // 100)

    rendered_rows = []
    common_side = 0

    # Build rows from low elevation to high elevation.
    # With origin="lower" below, low elevation appears at the bottom
    # and high elevation at the top.
    for el in el_centers:
        row_imgs = []
        for az in az_centers:
                        
            azim_mpl = float(np.rad2deg(az))
            elev_mpl = float(np.rad2deg(el))

            img = _render_surface_panel_image(
                tri_verts=tri_verts,
                normals=fn,
                azim=azim_mpl,
                elev=elev_mpl,
                roll=0,
                lim=lim,
                panel_px=panel_px,
                dpi=dpi,
            )

            img = _crop_white_border(img, white_threshold=250, pad=crop_pad)
            img = np.flipud(img)
            row_imgs.append(img)
            common_side = max(common_side, img.shape[0], img.shape[1])

        rendered_rows.append(row_imgs)

    extra_margin_px = max(16, panel_px // 12)
    side = common_side + 2 * extra_margin_px

    n_rows = len(el_centers)   # 4
    n_cols = len(az_centers)   # 8

    mosaic = np.full((n_rows * side, n_cols * side, 4), 255, dtype=np.uint8)

    for j, row_imgs in enumerate(rendered_rows):
        for i, img in enumerate(row_imgs):
            sq = _pad_to_square_canvas(img, side)
            y0 = j * side
            x0 = i * side
            mosaic[y0:y0 + side, x0:x0 + side] = sq

    ax = fig.add_subplot(111)
    ax.imshow(
        mosaic,
        origin="lower",
        extent=[-np.pi, np.pi, -np.pi / 2, np.pi / 2],
        interpolation="nearest",
        aspect="equal",
    )

    # Subtle grid lines at bin boundaries
    for x in az_edges:
        ax.axvline(x, color="0.82", linewidth=0.8, zorder=5)
    for y in el_edges:
        ax.axhline(y, color="0.82", linewidth=0.8, zorder=5)

    xticks, xlabels, yticks, ylabels = pi_tick_info()
    lookup_tick_fs = REFINE_TEXT.get("lookup_tick", REFINE_TEXT["tick"] + 10)
    lookup_axis_fs = REFINE_TEXT.get("lookup_axis", REFINE_TEXT["axis"] + 10)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=lookup_tick_fs)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=lookup_tick_fs)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi / 2, np.pi / 2)

    ax.set_xlabel("Azimuth", fontsize=lookup_axis_fs)
    ax.set_ylabel("Elevation", fontsize=lookup_axis_fs)

    vx = spacing[2]
    if embed_title:
        ax.set_title(
            f"{title}\n"
            f"Views rendered at azimuth/elevation bin centers    "
            f"Threshold: {level:.3f}    Min: {vmin:.3f}    Max: {vmax:.3f}    "
            f"Voxel size: {vx:.3f} Å/px",
            fontsize=REFINE_TEXT["annotation"]
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# main
# ============================================================

def generate_initial_model_pngs(
    job_folder: Path,
    outdir: Path,
    cmap="viridis",
    threshold_value=None,
    text_theme=None,
    embed_titles=True,
    scale_bar_length_A=None,
):
    """
    Generate only surface-view PNGs for ab-initio / initial-model classes.


    Returns a list of dicts:
        [
            {
                "class_index": 0,
                "class_number": 1,
                "title": "Surface Views - Class 1",
                "path": Path(...),
            },
            ...
        ]
    """
    set_refinement_text_theme(text_theme)

    outdir.mkdir(parents=True, exist_ok=True)

    class_maps = find_initial_model_class_maps(job_folder)
    if not class_maps:
        raise FileNotFoundError(
            f"No initial-model class maps found in {job_folder}. "
            f"Expected files like '{job_folder.name}_class_00_final_volume.mrc'."
        )

    outputs = []

    for class_idx, map_path in class_maps:
        vol, voxel = load_mrc(map_path)
        spacing = (voxel, voxel, voxel) if voxel is not None else (1.0, 1.0, 1.0)

        surface_threshold = choose_threshold(vol, user_threshold=threshold_value, mode="initial")

        class_number = class_idx
        title = f"Surface Views - Class {class_number}"
        out_path = outdir / f"initial_model_surface_views_class{class_number:02d}.png"

        save_surface_views(
            vol=vol,
            out_path=out_path,
            title=title,
            threshold_value=surface_threshold,
            step=1,
            spacing=spacing,
            panel_px=900,
            dpi=400,
            embed_header=embed_titles,
            embed_plane_labels=True,
            scale_bar_length_A=None,
        )

        outputs.append({
            "class_index": class_idx,
            "class_number": class_number,
            "title": title,
            "path": out_path,
            "threshold": float(surface_threshold),
        })

    return outputs


def generate_report_pngs(
    job_folder: Path,
    outdir: Path,
    pixel_size_override=None,
    cmap="viridis",
    threshold_value=None,
    slice_vmax=None,
    text_theme=None,
    embed_titles=True,
    scale_bar_length_A=None,
):
    set_refinement_text_theme(text_theme)

    outdir.mkdir(parents=True, exist_ok=True)

    iteration = find_latest_iteration(job_folder)

    map_path = get_iteration_file(job_folder, iteration, "volume_map.mrc")
    half_a_path = get_iteration_file(job_folder, iteration, "volume_map_half_A.mrc")
    half_b_path = get_iteration_file(job_folder, iteration, "volume_map_half_B.mrc")

    mask_refine_path = get_iteration_file(job_folder, iteration, "volume_mask_refine.mrc")
    mask_fsc_path = get_iteration_file(job_folder, iteration, "volume_mask_fsc.mrc")
    mask_auto_path = get_iteration_file(job_folder, iteration, "volume_mask_fsc_auto.mrc")

    particles_cs = get_iteration_file(job_folder, iteration, "particles.cs")
    rejected_cs = get_iteration_file(job_folder, iteration, "particles_rejected.cs")

    if map_path is None:
        raise FileNotFoundError(f"No *_volume_map.mrc found for iteration {iteration:03d}")

    # Always load the unsharpened map too
    vol, voxel = load_mrc(map_path)
    spacing = (voxel, voxel, voxel) if voxel is not None else (1.0, 1.0, 1.0)
    pixel_size = pixel_size_override or voxel or 1.0

    gsfsc = load_gsfsc_from_job_json(job_folder, iteration)

    # For display products:
    # - use sharpened only when GSFSC < 3.5 Å
    # - but if GSFSC is at Nyquist, force unsharpened
    display_map_path, display_map_mode, display_map_reason = choose_display_map_path(
        map_path,
        gsfsc,
        voxel_size_A=pixel_size,
    )

    if display_map_path == map_path:
        display_vol = vol
        display_spacing = spacing
    else:
        display_vol, display_voxel = load_mrc(display_map_path)
        display_spacing = (
            (display_voxel, display_voxel, display_voxel)
            if display_voxel is not None else spacing
        )

    surface_threshold = choose_threshold(display_vol, user_threshold=threshold_value, mode="refine")

    # real-space slices
    save_slice_panel(
        vol=display_vol,
        out_path=outdir / f"real_space_slices_iter{iteration:03d}.png",
        title=f"Real Space Slices Iteration {iteration:03d}",
        cmap=cmap,
        symmetric=True,
        manual_vmax=slice_vmax,
        embed_title=embed_titles,
    )

    # stacked mask slices
    mask_items = []
    if mask_refine_path is not None and mask_refine_path.exists():
        mref, _ = load_mrc(mask_refine_path)
        mask_items.append(("Mask Refine", mref))
    if mask_fsc_path is not None and mask_fsc_path.exists():
        mfsc, _ = load_mrc(mask_fsc_path)
        mask_items.append(("Mask FSC", mfsc))
    if mask_auto_path is not None and mask_auto_path.exists():
        mauto, _ = load_mrc(mask_auto_path)
        mask_items.append(("Mask FSC Auto", mauto))

    if mask_items:
        save_stacked_mask_panel(
            mask_items=mask_items,
            out_path=outdir / f"mask_slices_iter{iteration:03d}.png",
            cmap=cmap,
            title=None,
        )

    # surface views
    save_surface_views(
        vol=display_vol,
        out_path=outdir / f"surface_views_iter{iteration:03d}.png",
        title=f"Surface Views Iteration {iteration:03d}",
        threshold_value=surface_threshold,
        step=1,
        spacing=display_spacing,
        panel_px=900,
        dpi=400,
        embed_header=embed_titles,
        embed_plane_labels=True,
        scale_bar_length_A=None,
    )

    # view lookup chart
    save_protein_view_lookup(
        vol=display_vol,
        out_path=outdir / f"protein_view_lookup_iter{iteration:03d}.png",
        title=f"Protein View Lookup Iteration {iteration:03d}",
        threshold_value=surface_threshold,
        step=1,
        spacing=display_spacing,
        panel_px=240,
        dpi=300,
        embed_title=embed_titles,
    )

    # FSC
    if half_a_path is not None and half_b_path is not None:
        save_fsc_plot(
            half_a_path=half_a_path,
            half_b_path=half_b_path,
            mask_fsc_path=mask_fsc_path,
            mask_auto_path=mask_auto_path,
            out_path=outdir / f"fsc_iter{iteration:03d}.png",
            title=f"FSC Iteration {iteration:03d}",
            pixel_size_override=pixel_size,
            gsfsc_value=gsfsc,
            cmap=cmap,
            embed_title=embed_titles,
        )

    # Guinier
    save_guinier_plot(
        map_path=map_path,
        mask_path=mask_refine_path if (mask_refine_path and mask_refine_path.exists()) else None,
        out_path=outdir / f"guinier_iter{iteration:03d}.png",
        title=f"Guinier Plot Iteration {iteration:03d}",
        pixel_size_override=pixel_size,
        cmap=cmap,
        gsfsc=gsfsc,
        embed_title=embed_titles,
    )

    # Viewing direction / posterior precision + per-particle scale plots

    if particles_cs is not None and particles_cs.exists():
        arr = load_cs(particles_cs)

        try:
            pose_field = find_field_by_suffix(arr, "/pose", prefer_contains=("alignments3D",))
            pose = np.asarray(arr[pose_field], dtype=np.float64)
            theta = np.linalg.norm(pose, axis=1)

        except Exception as e:
            print(f"[warn] pose diagnostics failed: {e}")

        try:
            save_alpha_histogram(
                particles_cs=particles_cs,
                rejected_cs=rejected_cs if (rejected_cs is not None and rejected_cs.exists()) else None,
                out_path=outdir / f"per_particle_scale_iter{iteration:03d}.png",
                title=f"Per-Particle Scale Factors Iteration {iteration:03d}",
                cmap=cmap,
            )
        except Exception as e:
            print(f"[warn] per-particle scale plot skipped: {e}")

        try:
            save_viewing_direction_distribution(
                particles_cs=particles_cs,
                out_path=outdir / f"viewing_direction_distribution_iter{iteration:03d}.png",
                title=f"Viewing Direction Distribution Iteration {iteration:03d}",
                bins=(144, 72),
                bin_equal_area=False,
                log_scale=True,
                cmap=cmap,
                cbar_label="# images (log scale)",
                use_inverse=False,
                embed_title=embed_titles,
            )
        except Exception as e:
            print(f"[warn] viewing-direction plot skipped: {e}")

        try:
            save_posterior_precision_directional_distribution_fast(
                particles_cs=particles_cs,
                out_path=outdir / f"posterior_precision_directional_distribution_iter{iteration:03d}.png",
                title=f"Posterior Precision Directional Distribution Iteration {iteration:03d}",
                bins=(144, 72),
                bin_equal_area=False,
                log_scale=False,
                cmap=cmap,
                cbar_label="Relative posterior precision (a.u.)",
                use_inverse=False,
                low_res_A=30.0,
                high_res_A=None,
                freq_samples=24,
                n_circle_samples=288,
                particle_chunk=1024,
                embed_title=embed_titles,
            )
        except Exception as e:
            print(f"[warn] posterior-precision plot skipped: {e}")

    return iteration, pixel_size, gsfsc, display_map_mode, display_map_reason, float(surface_threshold)


def main():
    parser = argparse.ArgumentParser(description="Generate cryo-EM report PNGs from a job folder.")
    parser.add_argument("job_folder", help="Path to job folder, e.g. J15")
    parser.add_argument("--outdir", default=None, help="Output directory (default: <job_folder>/report_pngs)")
    parser.add_argument("--pixel-size", type=float, default=None, help="Manual pixel size in Å")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name (default: viridis)")
    parser.add_argument("--threshold", type=float, default=None, help="Manual threshold for threshold/surface")
    parser.add_argument("--slice-vmax", type=float, default=None, help="Manual symmetric vmax for real-space slices")
    args = parser.parse_args()

    job_folder = Path(args.job_folder).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else (job_folder / "report_pngs")

    iteration, pixel_size, _, _, _, _ = generate_report_pngs(
        job_folder=job_folder,
        outdir=outdir,
        pixel_size_override=args.pixel_size,
        cmap=args.cmap,
        threshold_value=args.threshold,
        slice_vmax=args.slice_vmax,
    )

    print(f"Done. Iteration: {iteration:03d}")
    print(f"Pixel size used: {pixel_size:.6f} Å")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()

