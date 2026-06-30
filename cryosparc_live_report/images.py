#!/usr/bin/env python3
# coding: utf-8

import io
import math
from io import BytesIO
from typing import List, Optional, Tuple, Callable, Union

AlphaLike = Union[int, Callable[[object, int, int], int]]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import map_coordinates
from skimage.filters import butterworth

try:
    from cryosparc.tools import Dataset
except Exception:
    Dataset = None

try:
    from .stats import get_picking_threshold, get_threshold_block
except Exception:
    pass

try:
    RESAMPLE = Image.Resampling.LANCZOS
except Exception:
    RESAMPLE = Image.LANCZOS

try:
    from .scale_bars import (
        add_bottom_scale_bar_pil,
        add_inset_scale_bar_pil,
        choose_scale_bar_for_display,
        format_length,
    )
except Exception:
    add_bottom_scale_bar_pil = None
    add_inset_scale_bar_pil = None
    choose_scale_bar_for_display = None
    format_length=None

A_TO_UM = 1e-4  # 1 Å = 1e-4 µm


def load_font(size=16, bold=False):
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


def fit_within(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    out = img.copy()
    out.thumbnail((max_w, max_h), RESAMPLE)
    return out


def make_placeholder(size=(800, 800), text="Image not available") -> Image.Image:
    img = Image.new("RGB", size, (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.text((20, 20), text, fill=(30, 30, 30), font=load_font(20))
    return img

def render_text_crop(text, font, fg=(0, 0, 0), bg=(255, 255, 255), pad=12, out_pad=3) -> Image.Image:
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

def mplfig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=100,
        facecolor="white",
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0.0,
    )
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.load()
    buf.close()
    return img


def sample_indices_evenly(n: int, sample_n: int) -> List[int]:
    if n <= 0 or sample_n <= 0:
        return []
    idxs = np.linspace(0, n - 1, min(sample_n, n))
    return sorted({int(round(i)) for i in idxs})


def pad_tile_to_square(img: Image.Image, tile_size: int, bg="white") -> Image.Image:
    canvas = Image.new("RGB", (tile_size, tile_size), bg)
    x = (tile_size - img.width) // 2
    y = (tile_size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def trim_uniform_background(img: Image.Image, bg_color=None, pad: int = 0) -> Image.Image:
    rgb = img.convert("RGB")
    arr = np.asarray(rgb)

    if bg_color is None:
        bg_color = tuple(arr[0, 0].tolist())

    bg = np.array(bg_color, dtype=arr.dtype)
    mask = np.any(arr != bg, axis=2)

    if not np.any(mask):
        return img

    ys, xs = np.where(mask)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(img.width, int(xs.max()) + 1 + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(img.height, int(ys.max()) + 1 + pad)

    return img.crop((x0, y0, x1, y1))


def load_mrc(path: str) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as m:
        return np.asarray(m.data)


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


def _require_cfg_keys(name: str, cfg: dict, keys: list):
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"{name} missing required keys: {', '.join(missing)}")


def _rgb_to_mpl(color):
    if isinstance(color, str):
        return color
    return tuple((float(c) / 255.0) if float(c) > 1.0 else float(c) for c in color)


def normalize_stack(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        return arr

    a, b, c = arr.shape

    if b == c:
        return arr
    if a == b:
        return np.transpose(arr, (2, 0, 1))
    if a == c:
        return np.transpose(arr, (1, 0, 2))

    stack_axis = int(np.argmax(arr.shape))
    if stack_axis == 0:
        return arr
    if stack_axis == 1:
        return np.transpose(arr, (1, 0, 2))
    return np.transpose(arr, (2, 0, 1))


def get_mrc_voxel_size_A(path: str) -> Optional[float]:
    try:
        with mrcfile.open(path, permissive=True) as m:
            try:
                vx = float(m.voxel_size.x)
                if np.isfinite(vx) and vx > 0:
                    return vx
            except Exception:
                pass

            try:
                mx = float(m.header.mx)
                cella_x = float(m.header.cella.x)
                if mx > 0 and cella_x > 0:
                    vx = cella_x / mx
                    if np.isfinite(vx) and vx > 0:
                        return vx
            except Exception:
                pass
    except Exception:
        pass
    return None


def robust_display_limits(
    arr: np.ndarray,
    sigma: float = 5.0,
    central_frac: float = 0.8,
):
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    h, w = a.shape
    dy = int((1.0 - central_frac) * h / 2.0)
    dx = int((1.0 - central_frac) * w / 2.0)

    if dy > 0 and dx > 0 and (h - 2 * dy) > 4 and (w - 2 * dx) > 4:
        core = a[dy:h - dy, dx:w - dx]
    else:
        core = a

    med = float(np.median(core))
    mad = float(np.median(np.abs(core - med)))
    scale = 1.4826 * mad if mad > 1e-8 else float(np.std(core))

    if not np.isfinite(scale) or scale <= 1e-8:
        lo = float(np.percentile(core, 1.0))
        hi = float(np.percentile(core, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(core)) if core.size else 0.0
            hi = float(np.max(core)) if core.size else 1.0
            if hi <= lo:
                hi = lo + 1.0
        return lo, hi

    lo = med - sigma * scale
    hi = med + sigma * scale
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.percentile(core, 1.0))
        hi = float(np.percentile(core, 99.0))
        if hi <= lo:
            hi = lo + 1.0

    return float(lo), float(hi)


def percentile_display_limits(
    arr: np.ndarray,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
):
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    lo = float(np.percentile(a, p_lo))
    hi = float(np.percentile(a, p_hi))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(a)) if a.size else 0.0
        hi = float(np.max(a)) if a.size else 1.0
        if hi <= lo:
            hi = lo + 1.0

    return lo, hi


def edge_display_limits(
    arr: np.ndarray,
    edge_frac: float = 0.12,
    edge_p_lo: float = 5.0,
    img_p_hi: float = 99.5,
):
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    h, w = a.shape
    ey = max(1, int(round(h * edge_frac)))
    ex = max(1, int(round(w * edge_frac)))

    edge = np.concatenate([
        a[:ey, :].ravel(),
        a[-ey:, :].ravel(),
        a[:, :ex].ravel(),
        a[:, -ex:].ravel(),
    ])

    lo = float(np.percentile(edge, edge_p_lo))
    hi = float(np.percentile(a, img_p_hi))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(a)) if a.size else 0.0
        hi = float(np.max(a)) if a.size else 1.0
        if hi <= lo:
            hi = lo + 1.0

    return lo, hi


def get_display_limits(
    img: np.ndarray,
    display_mode: str = "edge",
    sigma: float = 5.0,
    central_frac: float = 0.8,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    edge_frac: float = 0.12,
    edge_p_lo: float = 5.0,
    edge_img_p_hi: float = 99.5,
):
    mode = (display_mode or "edge").lower()

    if mode == "auto":
        a = np.asarray(img, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        lo = float(np.min(a)) if a.size else 0.0
        hi = float(np.max(a)) if a.size else 1.0
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    if mode == "percentile":
        return percentile_display_limits(img, p_lo=p_lo, p_hi=p_hi)

    if mode == "edge":
        return edge_display_limits(
            img,
            edge_frac=edge_frac,
            edge_p_lo=edge_p_lo,
            img_p_hi=edge_img_p_hi,
        )

    if mode == "robust":
        return robust_display_limits(img, sigma=sigma, central_frac=central_frac)

    raise ValueError(f"Unknown display_mode: {display_mode}")


def bin_mean(arr: np.ndarray, bin_factor: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if bin_factor <= 1:
        return a

    h, w = a.shape
    h2 = (h // bin_factor) * bin_factor
    w2 = (w // bin_factor) * bin_factor
    if h2 <= 0 or w2 <= 0:
        return a

    a = a[:h2, :w2]
    a = a.reshape(h2 // bin_factor, bin_factor, w2 // bin_factor, bin_factor)
    return a.mean(axis=(1, 3))


def choose_display_bin_factor(
    angpix: Optional[float],
    target_angpix: float = 3.0,
    max_bin: int = 4,
) -> int:
    if angpix is None or not np.isfinite(angpix) or angpix <= 0:
        return 1
    b = int(round(float(target_angpix) / float(angpix)))
    return max(1, min(b, max_bin))


def lowpass_filter_2d(
    arr: np.ndarray,
    cutoff_A: Optional[float],
    angpix: Optional[float],
    transition_frac: float = 0.15,
) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if cutoff_A is None or angpix is None:
        return a
    if cutoff_A <= 0 or angpix <= 0:
        return a

    f_cut = float(angpix) / float(cutoff_A)
    nyquist = 0.5
    if f_cut >= nyquist:
        return a

    f_pass = f_cut
    f_stop = min(nyquist, f_cut * (1.0 + max(0.01, transition_frac)))

    fy = np.fft.fftfreq(a.shape[0], d=1.0)
    fx = np.fft.fftfreq(a.shape[1], d=1.0)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)

    mask = np.zeros_like(R, dtype=np.float32)
    mask[R <= f_pass] = 1.0

    band = (R > f_pass) & (R < f_stop)
    if np.any(band):
        x = (R[band] - f_pass) / max(f_stop - f_pass, 1e-8)
        mask[band] = 0.5 * (1.0 + np.cos(np.pi * x))

    F = np.fft.fft2(a)
    out = np.fft.ifft2(F * mask).real
    return out.astype(np.float32)


def arr2d_signed_soft_to_pil(
    arr: np.ndarray,
    invert: bool = False,
    sigma: float = 3.0,
    output_lo: int = 45,
    output_hi: int = 210,
    center_on_zero: bool = False,
) -> Image.Image:
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    center = 0.0 if center_on_zero else float(np.median(a))
    mad = float(np.median(np.abs(a - center)))
    scale = 1.4826 * mad if mad > 1e-8 else float(np.std(a))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0

    z = (a - center) / scale
    z = np.clip(z, -sigma, sigma)
    z = (z + sigma) / (2.0 * sigma)

    if invert:
        z = 1.0 - z

    lo = float(output_lo) / 255.0
    hi = float(output_hi) / 255.0
    z = lo + z * (hi - lo)

    return Image.fromarray((255.0 * z).astype(np.uint8), mode="L")


def arr2d_to_pil_classavg_style(
    arr: np.ndarray,
    invert: bool = False,
    display_mode: str = "auto",
    sigma: float = 5.0,
    gamma: float = 1.0,
    central_frac: float = 0.8,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    edge_frac: float = 0.12,
    edge_p_lo: float = 5.0,
    edge_img_p_hi: float = 99.5,
    origin: str = "lower",
) -> Image.Image:
    img = np.asarray(arr, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    vmin, vmax = get_display_limits(
        img,
        display_mode=display_mode,
        sigma=sigma,
        central_frac=central_frac,
        p_lo=p_lo,
        p_hi=p_hi,
        edge_frac=edge_frac,
        edge_p_lo=edge_p_lo,
        edge_img_p_hi=edge_img_p_hi,
    )

    scaled = (img - vmin) / max(vmax - vmin, 1e-8)
    scaled = np.clip(scaled, 0.0, 1.0)

    if gamma is not None and abs(float(gamma) - 1.0) > 1e-6:
        scaled = scaled ** float(gamma)

    if invert:
        scaled = 1.0 - scaled

    if origin == "lower":
        scaled = np.flipud(scaled)

    return Image.fromarray((255.0 * scaled).astype(np.uint8), mode="L")


def arr2d_signed_to_pil(
    arr: np.ndarray,
    invert: bool = False,
    display_mode: str = "auto",
    sigma: float = 5.0,
    gamma: float = 1.0,
    central_frac: float = 0.8,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    edge_frac: float = 0.12,
    edge_p_lo: float = 5.0,
    edge_img_p_hi: float = 99.5,
) -> Image.Image:
    return arr2d_to_pil_classavg_style(
        arr,
        invert=invert,
        display_mode=display_mode,
        sigma=sigma,
        gamma=gamma,
        central_frac=central_frac,
        p_lo=p_lo,
        p_hi=p_hi,
        edge_frac=edge_frac,
        edge_p_lo=edge_p_lo,
        edge_img_p_hi=edge_img_p_hi,
    )


def normalize_micrograph_to_u8(arr: np.ndarray, sigma: float = 3.5, invert: bool = False) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    scale = 1.4826 * mad if mad > 1e-8 else float(np.std(a))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0

    lo = med - sigma * scale
    hi = med + sigma * scale
    a = np.clip((a - lo) / max(hi - lo, 1e-8), 0, 1)

    if invert:
        a = 1.0 - a

    return (255.0 * a).astype(np.uint8)


def normalize_signed_tile_to_u8(
    arr: np.ndarray,
    sigma: float = 5.0,
    invert: bool = True,
    central_frac: float = 0.8,
) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    h, w = a.shape
    dy = int((1.0 - central_frac) * h / 2.0)
    dx = int((1.0 - central_frac) * w / 2.0)
    core = a[dy:h-dy, dx:w-dx] if dy > 0 and dx > 0 else a

    med = float(np.median(core))
    mad = float(np.median(np.abs(core - med)))
    scale = 1.4826 * mad if mad > 1e-8 else float(np.std(core))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0

    z = (a - med) / scale
    z = np.clip(z, -sigma, sigma)
    z = (z + sigma) / (2.0 * sigma)

    if invert:
        z = 1.0 - z

    return (255.0 * z).astype(np.uint8)


def mrc_2d_to_pil(
    path: str,
    invert: bool = False,
    display_mode: str = "auto",
    sigma: float = 5.0,
    gamma: float = 1.0,
    central_frac: float = 0.8,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    edge_frac: float = 0.12,
    edge_p_lo: float = 5.0,
    edge_img_p_hi: float = 99.5,
    lowpass_A: Optional[float] = None,
    angpix_A: Optional[float] = None,
    origin: str = "lower",
) -> Image.Image:
    arr = load_mrc(path)
    if arr.ndim == 3:
        arr = normalize_stack(arr)[0]

    if lowpass_A is not None:
        if angpix_A is None:
            angpix_A = get_mrc_voxel_size_A(path)
        arr = lowpass_filter_2d(arr, cutoff_A=lowpass_A, angpix=angpix_A)

    return arr2d_to_pil_classavg_style(
        arr,
        invert=invert,
        display_mode=display_mode,
        sigma=sigma,
        gamma=gamma,
        central_frac=central_frac,
        p_lo=p_lo,
        p_hi=p_hi,
        edge_frac=edge_frac,
        edge_p_lo=edge_p_lo,
        edge_img_p_hi=edge_img_p_hi,
        origin=origin,
    )


def add_plot_style_title(
    img: Image.Image,
    title: str,
    font_size: int = 11,
    y_pad: int = 2,
) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    font = load_font(font_size, bold=False)

    try:
        bbox = d.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
    except Exception:
        tw = d.textsize(title, font=font)[0]

    x = (out.width - tw) // 2
    y = y_pad
    d.text((x, y), title, fill=(20, 20, 20), font=font)
    return out


def add_plot_style_title_band(
    img: Image.Image,
    title: str,
    canvas_size: tuple,
    font_size: int = 11,
    title_h: int = 20,
    y_pad: int = 2,
    bg=(255, 255, 255),
    text_color=(20, 20, 20),
    font_weight="normal",
    dpi: int = 100,
) -> Image.Image:
    W, H = [int(v) for v in canvas_size]
    if W <= 1 or H <= 1:
        return img.convert("RGB")

    fig_w = W / float(dpi)
    fig_h = H / float(dpi)

    title_h = max(int(title_h), int(round(font_size * 2.0)))
    title_frac = min(0.40, max(0.08, title_h / float(H)))

    bg_rgb = _rgb_to_mpl(bg)
    text_rgb = _rgb_to_mpl(text_color)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(bg_rgb)

    ax_title = fig.add_axes([0.0, 1.0 - title_frac, 1.0, title_frac], facecolor=bg_rgb)
    ax_title.set_axis_off()

    title_band_px = max(1.0, title_frac * H)
    y_shift = float(y_pad) / title_band_px

    ax_title.text(
        0.5,
        0.5 - y_shift,
        title,
        ha="center",
        va="center",
        fontsize=font_size,
        fontweight=font_weight,
        color=text_rgb,
        transform=ax_title.transAxes,
    )

    ax_img = fig.add_axes([0.0, 0.0, 1.0, 1.0 - title_frac], facecolor=bg_rgb)
    ax_img.set_axis_off()
    ax_img.set_anchor("C")

    np_img = np.asarray(img.convert("RGB"))
    ax_img.imshow(np_img, interpolation="nearest")
    ax_img.set_aspect("equal")

    out = mplfig_to_pil(fig)
    if out.size != (W, H):
        out = out.resize((W, H), RESAMPLE)
    return out

def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False

def get_pick_overlay_params(ws: dict, exp: dict) -> dict:
    params = ws.get("params", {}) or {}
    stats = ws.get("stats", {}) or {}

    picker_type = (
        exp.get("pick_picker_type")
        or exp.get("picker_type")
        or str(params.get("current_picker") or "").strip().lower()
        or None
    )

    source_micrograph_shape = None
    try:
        nx = stats.get("nx")
        ny = stats.get("ny")
        if nx is not None and ny is not None:
            source_micrograph_shape = (int(nx), int(ny))
    except Exception:
        source_micrograph_shape = None

    picker = str(picker_type or "").strip().lower()

    ncc_min = ncc_val = ncc_max = None
    power_min = power_max = None

    if picker:
        ncc_min, ncc_val, ncc_max = get_picking_threshold(ws, f"{picker}_ncc_score")
        power_block = get_threshold_block(ws, f"{picker}_power") or {}
        power_min = power_block.get("min")
        power_max = power_block.get("max")

    return {
        "picker_type": picker_type,
        "micrograph_psize_A": exp.get("micrograph_psize_A") or params.get("psize_A"),
        "source_micrograph_shape": source_micrograph_shape,
        "blob_diameter_max_A": params.get("blob_diameter_max"),
        "template_diameter_A": params.get("template_diameter"),
        "gainref_flip_y": _as_bool(params.get("gainref_flip_y")),
        "ncc_threshold": ncc_val,
        "power_min": power_min,
        "power_max": power_max,
    }

def _pick_diameter_A_from_overlay_params(p: dict):
    if not isinstance(p, dict):
        return None


    picker = str(p.get("picker_type") or "").strip().lower()


    candidates = []
    if picker == "blob":
        candidates = [p.get("blob_diameter_max_A"), p.get("template_diameter_A")]
    elif picker == "template":
        candidates = [p.get("template_diameter_A"), p.get("blob_diameter_max_A")]
    else:
        candidates = [p.get("template_diameter_A"), p.get("blob_diameter_max_A")]


    for v in candidates:
        try:
            x = float(v)
            if np.isfinite(x) and x > 0:
                return x
        except Exception:
            pass


    return None

def overlay_blob_picks(
    base_img: Image.Image,
    pick_cs_path: Optional[str],
    *,
    picker_type: Optional[str] = None,
    micrograph_psize_A: Optional[float] = None,
    source_micrograph_shape: Optional[Tuple[int, int]] = None,
    blob_diameter_max_A: Optional[float] = None,
    template_diameter_A: Optional[float] = None,
    gainref_flip_y: bool = False,
    ncc_threshold: Optional[float] = None,
    power_min: Optional[float] = None,
    power_max: Optional[float] = None,
    max_draw: int = 4000,
    outline_rgba: Tuple[int, int, int, int] = (0, 0, 255, 255),
    fill_rgb: Tuple[int, int, int] = (0, 0, 255),
    fill_alpha: AlphaLike = 0,
    line_w: int = 6,
) -> Image.Image:
    img = base_img.convert("RGBA")
    if not pick_cs_path or Dataset is None:
        return img.convert("RGB")

    try:
        ds = Dataset.load(pick_cs_path)
        n = len(ds)
        if n == 0:
            return img.convert("RGB")

        w, h = img.size
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        picker = str(picker_type or "").strip().lower()
        diameter_A = None
        if picker == "blob":
            diameter_A = blob_diameter_max_A
        elif picker == "template":
            diameter_A = template_diameter_A

        rx = ry = None
        if (
            diameter_A is not None
            and micrograph_psize_A is not None
            and micrograph_psize_A > 0
        ):
            diameter_src_pix = float(diameter_A) / float(micrograph_psize_A)

            if source_micrograph_shape is not None:
                src_w, src_h = source_micrograph_shape
                if src_w and src_h:
                    scale_x = w / float(src_w)
                    scale_y = h / float(src_h)
                else:
                    scale_x = scale_y = 1.0
            else:
                scale_x = scale_y = 1.0

            rx = max(2.0, 0.5 * diameter_src_pix * scale_x)
            ry = max(2.0, 0.5 * diameter_src_pix * scale_y)

        if rx is None or ry is None:
            fallback_r = max(2, int(round(min(w, h) / 110.0)))
            rx = ry = float(fallback_r)

        # First collect rows that pass thresholds
        rows_to_draw = []
        has_ncc = "pick_stats/ncc_score" in ds.fields()
        has_power = "pick_stats/power" in ds.fields()

        for i in range(n):
            row = ds[i]

            if ncc_threshold is not None and has_ncc:
                try:
                    if float(row["pick_stats/ncc_score"]) < float(ncc_threshold):
                        continue
                except Exception:
                    continue

            if power_min is not None and has_power:
                try:
                    if float(row["pick_stats/power"]) < float(power_min):
                        continue
                except Exception:
                    continue

            if power_max is not None and has_power:
                try:
                    if float(row["pick_stats/power"]) > float(power_max):
                        continue
                except Exception:
                    continue

            rows_to_draw.append((i, row))

        if not rows_to_draw:
            return img.convert("RGB")

        step = max(1, int(math.ceil(len(rows_to_draw) / max_draw)))

        for j in range(0, len(rows_to_draw), step):
            i, row = rows_to_draw[j]

            x = float(row["location/center_x_frac"]) * w
            y_frac = float(row["location/center_y_frac"])
#            y = (y_frac if gainref_flip_y else (1.0 - y_frac)) * h
            y = (y_frac * h)

            if callable(fill_alpha):
                a = int(fill_alpha(row, i, len(rows_to_draw)))
            else:
                a = int(fill_alpha)
            a = max(0, min(255, a))

            draw.ellipse(
                (x - rx, y - ry, x + rx, y + ry),
                fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], a),
                outline=outline_rgba,
                width=line_w,
            )

        out = Image.alpha_composite(img, overlay)
        return out.convert("RGB")

    except Exception:
        return base_img.convert("RGB")

def spline_interp(gridshape, spl, pix):
    K_Z, K_Y, K_X = spl.shape
    mz, my, mx = gridshape

    coords = pix * np.array(
        [
            (K_Z - 1) / float(max(mz - 1, 1)),
            (K_Y - 1) / float(max(my - 1, 1)),
            (K_X - 1) / float(max(mx - 1, 1)),
        ],
        np.float32,
    ).reshape((3,) + (1,) * (pix.ndim - 1))

    res = map_coordinates(
        np.pad(spl, 1, "reflect", reflect_type="odd"),
        coords + 1,
        mode="constant",
        prefilter=False,
    )
    return res


def spline_interp_traj(gridshape, splxy, pos):
    mz, my, mx = gridshape
    N_P = pos.shape[0]

    pix = np.zeros((3, N_P, mz), dtype=np.float32)
    pix[0] = np.arange(mz, dtype=np.float32).reshape(1, -1)
    pix[1] = pos[:, 1].reshape(-1, 1)
    pix[2] = pos[:, 0].reshape(-1, 1)

    res = np.empty((N_P, mz, 2), np.float32)
    res[:, :, 0] = spline_interp(gridshape, splxy[0], pix)
    res[:, :, 1] = spline_interp(gridshape, splxy[1], pix)
    return res


def infer_motion_display_grid(
    nx: int,
    ny: int,
    angpix_A: float,
    patch_spacing_A: float = 380.0,
    patch_size_A: float = 500.0,
):
    width_A = float(nx) * float(angpix_A)
    height_A = float(ny) * float(angpix_A)

    npx = max(2, int(round(width_A / float(patch_spacing_A))))
    npy = max(2, int(round(height_A / float(patch_spacing_A))))

    half_patch_A_x = min(float(patch_size_A) / 2.0, width_A / 2.0)
    half_patch_A_y = min(float(patch_size_A) / 2.0, height_A / 2.0)

    xs_A = np.linspace(half_patch_A_x, width_A - half_patch_A_x, npx, dtype=np.float32)
    ys_A = np.linspace(half_patch_A_y, height_A - half_patch_A_y, npy, dtype=np.float32)

    xs_pix = xs_A / float(angpix_A)
    ys_pix = ys_A / float(angpix_A)

    X, Y = np.meshgrid(xs_pix, ys_pix)
    pos_raw = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    return pos_raw, npx, npy


def reconstruct_local_motion_from_spline(
    spline_motion_path: str,
    n_frames: int,
    rigid_motion_path: Optional[str] = None,
):
    coeffs = np.load(spline_motion_path, allow_pickle=True)
    coeffs = np.asarray(coeffs, dtype=np.float32)

    if coeffs.ndim != 4 or coeffs.shape[0] != 2:
        raise ValueError(f"Unexpected spline motion shape {coeffs.shape} in {spline_motion_path}")

    _, gy, gx, ncoef = coeffs.shape
    t = np.linspace(0.0, 1.0, int(n_frames), dtype=np.float32)
    powers = t[:, None, None, None] ** np.arange(ncoef, dtype=np.float32)[None, None, None, :]

    def eval_coeff(c):
        tx = np.sum(c[0][None, :, :, :] * powers, axis=-1)
        ty = np.sum(c[1][None, :, :, :] * powers, axis=-1)
        tx = tx - tx[0:1, :, :]
        ty = ty - ty[0:1, :, :]
        return tx, ty

    candidates = []
    for reverse in (False, True):
        c = coeffs[..., ::-1] if reverse else coeffs
        tx, ty = eval_coeff(c)

        score = 0.0
        if rigid_motion_path:
            try:
                rigid = np.load(rigid_motion_path, allow_pickle=True)
                rigid = np.asarray(rigid, dtype=np.float32)
                if rigid.ndim == 3 and rigid.shape[0] >= 1 and rigid.shape[2] == 2:
                    r = rigid[0, :n_frames, :]
                    rx = r[:, 0] - r[0, 0]
                    ry = r[:, 1] - r[0, 1]
                    mx = tx.mean(axis=(1, 2))
                    my = ty.mean(axis=(1, 2))
                    score = float(np.mean((mx - rx) ** 2 + (my - ry) ** 2))
            except Exception:
                pass

        candidates.append((score, reverse, tx, ty))

    candidates.sort(key=lambda x: x[0])
    _, reverse, tx_best, ty_best = candidates[0]
    return tx_best, ty_best, {
        "gy": gy,
        "gx": gx,
        "ncoef": ncoef,
        "reversed_coeff_order": reverse,
    }


def _interp_2d_small_grid(frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    a = np.asarray(frame, dtype=np.float32)
    in_h, in_w = a.shape

    if in_h == out_h and in_w == out_w:
        return a.copy()

    x_in = np.arange(in_w, dtype=np.float32)
    x_out = np.linspace(0, in_w - 1, out_w, dtype=np.float32)
    tmp = np.empty((in_h, out_w), dtype=np.float32)
    for iy in range(in_h):
        tmp[iy, :] = np.interp(x_out, x_in, a[iy, :])

    y_in = np.arange(in_h, dtype=np.float32)
    y_out = np.linspace(0, in_h - 1, out_h, dtype=np.float32)
    out = np.empty((out_h, out_w), dtype=np.float32)
    for ix in range(out_w):
        out[:, ix] = np.interp(y_out, y_in, tmp[:, ix])

    return out


def resample_motion_grid(tx: np.ndarray, ty: np.ndarray, out_h: int = 8, out_w: int = 8):
    tx = np.asarray(tx, dtype=np.float32)
    ty = np.asarray(ty, dtype=np.float32)

    n_frames, _, _ = tx.shape
    tx_out = np.empty((n_frames, out_h, out_w), dtype=np.float32)
    ty_out = np.empty((n_frames, out_h, out_w), dtype=np.float32)

    for k in range(n_frames):
        tx_out[k] = _interp_2d_small_grid(tx[k], out_h, out_w)
        ty_out[k] = _interp_2d_small_grid(ty[k], out_h, out_w)

    return tx_out, ty_out


def load_ctf_spline(ctf_spline_path: str) -> np.ndarray:
    arr = np.load(ctf_spline_path, allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D spline grid after squeeze, got shape {arr.shape}")

    return arr


def eval_ctf_mean_defocus(
    arr: np.ndarray,
    x_frac,
    y_frac,
    mode: str = "nearest",
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    ky, kx = arr.shape

    if ky < 2 or kx < 2:
        raise ValueError(f"Spline grid must be at least 2x2, got {arr.shape}")

    order = min(3, ky - 1, kx - 1)

    x_frac = np.asarray(x_frac, dtype=np.float64)
    y_frac = np.asarray(y_frac, dtype=np.float64)

    x_idx = x_frac * (kx - 1)
    y_idx = y_frac * (ky - 1)

    coords = np.vstack([y_idx.ravel(), x_idx.ravel()])
    z = map_coordinates(
        arr,
        coords,
        order=order,
        mode=mode,
        prefilter=False,
    )
    return z.reshape(np.broadcast(x_frac, y_frac).shape)


def get_active_picker_info(exp: dict):
    picker = str(exp.get("picker_type") or exp.get("pick_picker_type") or "").strip().lower()
    if not picker:
        picker = "unknown"

    count = exp.get("active_pick_count")
    if count is None:
        attrs = ((exp.get("raw") or {}).get("attributes") or {})
        key_map = {
            "blob": "total_blob_picks",
            "template": "total_template_picks",
            "deep": "total_deep_picks",
            "manual": "total_manual_picks",
        }
        key = key_map.get(picker)
        try:
            count = int(attrs.get(key) or 0) if key else 0
        except Exception:
            count = 0

    return picker, count


def _unwrap_scalar(v):
    while isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
        v = v[0]
    return v


def _get_exp_raw_nested(exp: dict, *keys, default=None):
    cur = exp.get("raw")
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    cur = _unwrap_scalar(cur)
    return default if cur is None else cur


def _infer_micrograph_shape_from_exp(exp: dict) -> Optional[Tuple[int, int]]:
    candidates = [
        ("groups", "exposure", "micrograph_blob", "shape"),
        ("groups", "exposure", "micrograph_blob_non_dw", "shape"),
        ("groups", "exposure", "movie_blob", "shape"),
    ]

    for keys in candidates:
        shp = _get_exp_raw_nested(exp, *keys, default=None)
        if isinstance(shp, (list, tuple, np.ndarray)):
            vals = [int(v) for v in np.asarray(shp).ravel().tolist()]
            if len(vals) == 2:
                return (vals[0], vals[1])
            if len(vals) >= 3:
                return (vals[-2], vals[-1])

    if exp.get("micrograph_path"):
        try:
            arr = load_mrc(exp["micrograph_path"])
            if arr.ndim == 2:
                return tuple(int(v) for v in arr.shape)
            if arr.ndim == 3:
                s = normalize_stack(arr)[0].shape
                return tuple(int(v) for v in s)
        except Exception:
            pass

    return None


def _first_threshold_crossing_x(x, y, thr=0.3, smooth_window=7):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size < 2:
        return None

    if smooth_window > 1 and y.size >= smooth_window:
        kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
        ys = np.convolve(y, kernel, mode="same")
    else:
        ys = y

    for i in range(1, len(ys)):
        y0, y1 = ys[i - 1], ys[i]
        if y0 >= thr and y1 < thr:
            x0, x1 = x[i - 1], x[i]
            if abs(y1 - y0) < 1e-8:
                return float(x1)
            t = (thr - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    return None


def plot_ctf_1d_to_pil(
    ctf_1d_path: str,
    size=(420, 220),
    exp: Optional[dict] = None,
    cfg: Optional[dict] = None,
) -> Image.Image:
    cfg = dict(cfg or {})
    _require_cfg_keys(
        "ctf_1d cfg",
        cfg,
        [
            "threshold",
            "smooth_window",
            "dpi",
            "render_scale",
            "facecolor",
            "ps_color",
            "ctf_color",
            "fit_color",
            "threshold_line_color",
            "grid_color",
            "refline_color",
            "ps_lw",
            "ctf_lw",
            "fit_lw",
            "threshold_lw",
            "xlabel_fontsize",
            "ylabel_fontsize",
            "tick_labelsize",
            "title_fontsize",
            "legend_fontsize",
            "top_axis_fontsize",
            "top_tick_fontsize",
            "tight_pad",
            "resolution_ticks_A",
        ],
    )

    try:
        arr = np.load(ctf_1d_path, allow_pickle=True)
        if arr.dtype.names is None:
            raise ValueError(f"Unexpected CTF 1D format in {ctf_1d_path}")

        freq = np.asarray(arr["freqs_trim"], dtype=np.float32)
        epa = np.asarray(arr["EPA_trim"], dtype=np.float32)
        ctf = np.asarray(arr["CTF"], dtype=np.float32)
        bg = np.asarray(arr["BGINT"], dtype=np.float32)
        env = np.asarray(arr["ENVINT"], dtype=np.float32)
        cc = (
            np.asarray(arr["CC"], dtype=np.float32)
            if "CC" in arr.dtype.names
            else np.zeros_like(freq, dtype=np.float32)
        )

        ps = np.cbrt(epa - bg) + 0.5
        ctf_disp = np.cbrt(env * (2.0 * (ctf ** 2) - 1.0)) + 0.5

        fit_cross_x = _first_threshold_crossing_x(
            freq,
            cc,
            thr=cfg["threshold"],
            smooth_window=cfg["smooth_window"],
        )

        render_scale = max(1, int(cfg["render_scale"]))
        fig_w = max(2.0, size[0] / float(cfg["dpi"]))
        fig_h = max(1.45, size[1] / float(cfg["dpi"]))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=cfg["dpi"] * render_scale)
        fig.patch.set_facecolor(cfg["facecolor"])
        ax.set_facecolor(cfg["facecolor"])

        ax.plot(freq, ps, color=cfg["ps_color"], lw=cfg["ps_lw"], label="PS", antialiased=True)
        ax.plot(freq, ctf_disp, color=cfg["ctf_color"], lw=cfg["ctf_lw"], label="CTF", antialiased=True)
        ax.plot(freq, cc, color=cfg["fit_color"], lw=cfg["fit_lw"], label="Fit", antialiased=True)

        if fit_cross_x is not None:
            ax.axvline(fit_cross_x, color=cfg["threshold_line_color"], lw=cfg["threshold_lw"], antialiased=True)

        ax.axhline(cfg["threshold"], color=cfg["refline_color"], lw=0.8, ls="--", zorder=0)

        ax.set_xlim(0.0, float(np.max(freq)))
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        ax.set_xlabel("Spatial frequency (1/Å)", fontsize=cfg["xlabel_fontsize"])
        ax.set_ylabel("Signal", fontsize=cfg["ylabel_fontsize"])
        ax.tick_params(axis="both", labelsize=cfg["tick_labelsize"])
        ax.grid(True, color=cfg["grid_color"], lw=0.6)

        tick_pos = []
        tick_lab = []
        fmax = float(np.max(freq))
        for r in cfg["resolution_ticks_A"]:
            f = 1.0 / float(r)
            if 0.0 <= f <= fmax:
                tick_pos.append(f)
                tick_lab.append(f"{r}")

        if tick_pos:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(tick_pos)
            ax2.set_xticklabels(tick_lab, fontsize=cfg["top_tick_fontsize"])
            ax2.set_xlabel("Resolution (Å)", fontsize=cfg["top_axis_fontsize"], loc="left")

        ax.set_title("1D CTF", fontsize=cfg["title_fontsize"], pad=2.5)
        ax.legend(
            loc="upper right",
            fontsize=cfg["legend_fontsize"],
            frameon=False,
            ncol=3,
            handlelength=1.8,
        )

        fig.tight_layout(pad=cfg["tight_pad"])

        buf = BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=cfg["dpi"] * render_scale,
            facecolor=cfg["facecolor"],
            edgecolor=cfg["facecolor"],
            bbox_inches=None,
            pad_inches=0.0,
        )
        plt.close(fig)

        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        if img.size != tuple(size):
            img = img.resize(size, RESAMPLE)
        return img

    except Exception as e:
        return make_placeholder(size=size, text=f"1D CTF unavailable\n{e}")


def plot_global_motion_to_pil(
    rigid_motion_path: str,
    size=(220, 220),
    zero_shift_frame: Optional[int] = None,
    cfg: Optional[dict] = None,
) -> Image.Image:
    cfg = dict(cfg or {})
    _require_cfg_keys(
        "global_motion cfg",
        cfg,
        [
            "title",
            "dpi",
            "line_color",
            "line_width",
            "start_marker_ms",
            "grid_color",
            "grid_lw",
            "axis_line_color",
            "axis_line_lw",
            "title_fontsize",
            "tick_labelsize",
            "tight_pad",
            "facecolor",
            "subtract_zero_frame",
        ],
    )

    try:
        traj = np.load(rigid_motion_path, allow_pickle=True)
        traj = np.asarray(traj, dtype=np.float32)

        if traj.ndim == 3:
            traj = np.squeeze(traj, axis=0)
        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError(f"Unexpected rigid motion shape {traj.shape}")

        if zero_shift_frame is None:
            zero_shift_frame = 0
        zero_shift_frame = int(np.clip(zero_shift_frame, 0, len(traj) - 1))

        x = traj[:, 0].copy()
        y = traj[:, 1].copy()

        if cfg["subtract_zero_frame"]:
            x = x - x[zero_shift_frame]
            y = y - y[zero_shift_frame]

        fig_w = max(2.0, size[0] / float(cfg["dpi"]))
        fig_h = max(2.0, size[1] / float(cfg["dpi"]))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=cfg["dpi"])
        fig.patch.set_facecolor(cfg["facecolor"])
        ax.set_facecolor(cfg["facecolor"])

        ax.plot(x, y, color=cfg["line_color"], lw=cfg["line_width"], zorder=2)
        ax.plot(x[0], y[0], "o", ms=cfg["start_marker_ms"], color=cfg["line_color"], zorder=3)

        ax.axhline(0, color=cfg["axis_line_color"], lw=cfg["axis_line_lw"], zorder=0)
        ax.axvline(0, color=cfg["axis_line_color"], lw=cfg["axis_line_lw"], zorder=0)
        ax.grid(True, color=cfg["grid_color"], lw=cfg["grid_lw"])

        ax.set_aspect("auto")
        ax.set_title(cfg["title"], fontsize=cfg["title_fontsize"], pad=2)
        ax.tick_params(axis="both", labelsize=cfg["tick_labelsize"])

        fig.tight_layout(pad=cfg["tight_pad"])
        return mplfig_to_pil(fig)

    except Exception as e:
        return make_placeholder(size=size, text=f"Global motion unavailable\n{e}")


def plot_local_motion_to_pil(
    spline_motion_path: str,
    size=(420, 220),
    movie_shape: Optional[tuple] = None,
    angpix_A: Optional[float] = None,
    rigid_motion_path: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> Image.Image:
    cfg = dict(cfg or {})
    _require_cfg_keys(
        "local_motion cfg",
        cfg,
        [
            "title",
            "dpi",
            "patch_spacing_A",
            "patch_size_A",
            "viewer_scale",
            "grid_color",
            "grid_lw",
            "traj_lw",
            "traj_alpha",
            "start_marker_ms",
            "row_cmap",
            "row_cmap_min",
            "row_cmap_max",
            "title_fontsize",
            "facecolor",
            "tight_pad",
        ],
    )

    try:
        splxy = np.load(spline_motion_path, allow_pickle=True)
        splxy = np.asarray(splxy, dtype=np.float32)

        if splxy.ndim != 4 or splxy.shape[0] != 2:
            raise ValueError(f"Unexpected spline motion shape {splxy.shape}")

        if movie_shape is None or len(movie_shape) != 3:
            raise ValueError("movie_shape=(N_Z, ny, nx) is required for local motion display")

        N_Z, ny, nx = [int(v) for v in movie_shape]

        if angpix_A is None or not np.isfinite(angpix_A) or angpix_A <= 0:
            raise ValueError("Valid angpix_A is required for local motion display")

        if rigid_motion_path:
            rigid = np.load(rigid_motion_path, allow_pickle=True)
            rigid = np.asarray(rigid, dtype=np.float32)
            if rigid.ndim == 3:
                rigid = np.squeeze(rigid, axis=0)
            if rigid.ndim != 2 or rigid.shape[1] != 2:
                raise ValueError(f"Unexpected rigid motion shape {rigid.shape}")
            if rigid.shape[0] != N_Z:
                N_use = min(N_Z, rigid.shape[0])
                rigid = rigid[:N_use]
                N_Z = N_use

        pos_raw, npx, npy = infer_motion_display_grid(
            nx,
            ny,
            angpix_A,
            patch_spacing_A=cfg["patch_spacing_A"],
            patch_size_A=cfg["patch_size_A"],
        )

        ts_local = spline_interp_traj((N_Z, ny, nx), splxy, pos_raw)
        ts_local = ts_local[:, :N_Z, :]
        ts_rel = ts_local - ts_local[:, 0:1, :]

        cell_w_pix = float(nx) / float(npx)
        cell_h_pix = float(ny) / float(npy)

        fig_w = max(2.0, size[0] / float(cfg["dpi"]))
        fig_h = max(1.8, size[1] / float(cfg["dpi"]))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=cfg["dpi"])
        fig.patch.set_facecolor(cfg["facecolor"])
        ax.set_facecolor(cfg["facecolor"])

        for x in range(npx + 1):
            ax.axvline(x, color=cfg["grid_color"], lw=cfg["grid_lw"], zorder=0)
        for y in range(npy + 1):
            ax.axhline(y, color=cfg["grid_color"], lw=cfg["grid_lw"], zorder=0)

        cmap = getattr(plt.cm, cfg["row_cmap"])
        row_colors = cmap(np.linspace(cfg["row_cmap_min"], cfg["row_cmap_max"], npy))

        for p in range(pos_raw.shape[0]):
            iy = p // npx

            x_raw = pos_raw[p, 0]
            y_raw = pos_raw[p, 1]

            base_x = (x_raw / float(nx)) * npx
            base_y = npy - (y_raw / float(ny)) * npy

            px = base_x + cfg["viewer_scale"] * (ts_rel[p, :, 0] / cell_w_pix)
            py = base_y - cfg["viewer_scale"] * (ts_rel[p, :, 1] / cell_h_pix)

            traj_color = row_colors[iy]
            ax.plot(px, py, color=traj_color, lw=cfg["traj_lw"], alpha=cfg["traj_alpha"], zorder=2)
            ax.plot(px[0], py[0], "o", ms=cfg["start_marker_ms"], color=traj_color, alpha=cfg["traj_alpha"], zorder=3)

        ax.set_xlim(0, npx)
        ax.set_ylim(npy, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(cfg["title"], fontsize=cfg["title_fontsize"], pad=2)

        fig.tight_layout(pad=cfg["tight_pad"])
        return mplfig_to_pil(fig)

    except Exception as e:
        return make_placeholder(size=size, text=f"Local motion unavailable\n{e}")


def plot_ctf_defocus_landscape_to_pil(
    ctf_spline_path: str,
    size=(220, 220),
    micrograph_shape: Optional[Tuple[int, int]] = None,
    cfg: Optional[dict] = None,
) -> Image.Image:
    cfg = dict(cfg or {})
    _require_cfg_keys(
        "local_defocus cfg",
        cfg,
        [
            "title",
            "dpi",
            "display_grid",
            "mode",
            "elev",
            "azim",
            "cmap",
            "z_half_range_A",
            "facecolor",
            "xlabel",
            "ylabel",
            "zlabel",
            "axis_label_fontsize",
            "title_fontsize",
            "tick_labelsize",
            "box_aspect",
            "colorbar",
            "colorbar_shrink",
            "colorbar_pad",
            "colorbar_fraction",
            "colorbar_label",
            "colorbar_label_fontsize",
            "tight_pad",
        ],
    )

    try:
        arr = load_ctf_spline(ctf_spline_path)

        if micrograph_shape is None or len(micrograph_shape) != 2:
            raise ValueError("micrograph_shape=(ny, nx) is required")

        ny, nx = [int(v) for v in micrograph_shape]
        if ny <= 1 or nx <= 1:
            raise ValueError(f"Invalid micrograph_shape: {micrograph_shape}")

        x_frac = np.linspace(0.0, 1.0, int(cfg["display_grid"]), dtype=np.float32)
        y_frac = np.linspace(0.0, 1.0, int(cfg["display_grid"]), dtype=np.float32)
        Xf, Yf = np.meshgrid(x_frac, y_frac)

        Z_A = eval_ctf_mean_defocus(arr, Xf, Yf, mode=cfg["mode"])
        Z_um = Z_A * A_TO_UM

        X = Xf * float(nx - 1)
        Y = Yf * float(ny - 1)

        fig_w = max(2.0, size[0] / float(cfg["dpi"]))
        fig_h = max(2.0, size[1] / float(cfg["dpi"]))

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=cfg["dpi"])
        fig.patch.set_facecolor(cfg["facecolor"])

        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            X,
            Y,
            Z_um,
            cmap=cfg["cmap"],
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        z_center_um = float(np.nanmean(Z_um)) if Z_um.size else 0.0
        if not np.isfinite(z_center_um):
            z_center_um = 0.0

        z_half_range_um = float(cfg["z_half_range_A"]) * A_TO_UM
        ax.set_zlim(z_center_um - z_half_range_um, z_center_um + z_half_range_um)

        ax.set_xlabel(cfg["xlabel"], fontsize=cfg["axis_label_fontsize"], labelpad=1)
        ax.set_ylabel(cfg["ylabel"], fontsize=cfg["axis_label_fontsize"], labelpad=1)
        ax.set_zlabel(cfg["zlabel"], fontsize=cfg["axis_label_fontsize"], labelpad=2)
        ax.set_title(cfg["title"], fontsize=cfg["title_fontsize"], pad=2)
        ax.view_init(elev=cfg["elev"], azim=cfg["azim"])

        try:
            ax.set_box_aspect(cfg["box_aspect"])
        except Exception:
            pass

        ax.tick_params(axis="both", which="major", labelsize=cfg["tick_labelsize"], pad=0)
        try:
            ax.zaxis.set_tick_params(labelsize=cfg["tick_labelsize"], pad=0)
        except Exception:
            pass

        if cfg["colorbar"]:
            cbar = fig.colorbar(
                surf,
                ax=ax,
                shrink=cfg["colorbar_shrink"],
                pad=cfg["colorbar_pad"],
                fraction=cfg["colorbar_fraction"],
            )
            cbar.ax.tick_params(labelsize=cfg["tick_labelsize"])
            cbar.set_label(cfg["colorbar_label"], fontsize=cfg["colorbar_label_fontsize"])

        fig.tight_layout(pad=cfg["tight_pad"])
        return mplfig_to_pil(fig)

    except Exception as e:
        return make_placeholder(size=size, text=f"Local defocus unavailable\n{e}")


def normalize_curve_01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.size == 0:
        return a
    lo = float(np.percentile(a, 1.0))
    hi = float(np.percentile(a, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(a))
        hi = float(np.max(a))
        if hi <= lo:
            return np.zeros_like(a, dtype=np.float32)
    out = (a - lo) / max(hi - lo, 1e-8)
    return np.clip(out, 0.0, 1.0)


def particle_lowpass_for_display(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if a.ndim != 2:
        raise ValueError(f"Expected 2D particle image, got shape {a.shape}")

    if butterworth is None:
        return a

    N = int(a.shape[0])
    if N <= 0:
        return a

    out = butterworth(
        a,
        cutoff_frequency_ratio=6.0 / float(N),
        high_pass=False,
        order=1,
    )
    return np.asarray(out, dtype=np.float32)


def _particle_imshow_kwargs(
    img: np.ndarray,
    invert: bool = False,
    autoscale: str = "imshow",
    p_lo: float = 0.5,
    p_hi: float = 99.5,
):
    cmap = "gray_r" if invert else "gray"

    autoscale = (autoscale or "imshow").lower()
    if autoscale in ("imshow", "minmax"):
        return {"cmap": cmap}

    if autoscale == "percentile":
        lo = float(np.percentile(img, p_lo))
        hi = float(np.percentile(img, p_hi))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return {"cmap": cmap}
        return {"cmap": cmap, "vmin": lo, "vmax": hi}

    raise ValueError(f"Unknown autoscale mode: {autoscale}")


def load_particle_stack_montage_matplotlib(
    stack_path: str,
    cfg: dict,
) -> Optional[Image.Image]:
    _require_cfg_keys(
        "particles cfg",
        cfg,
        [
            "sample_n",
            "cols",
            "tile_inches",
            "dpi",
            "invert",
            "autoscale",
            "p_lo",
            "p_hi",
            "wspace",
            "hspace",
            "add_indices",
            "facecolor",
            "index_fontsize",
            "index_color",
            "index_bbox_facecolor",
            "index_bbox_alpha",
            "index_bbox_pad",
            "index_bbox_edgecolor",
        ],
    )

    try:
        arr = load_mrc(stack_path)
        arr = normalize_stack(arr)
        particle_angpix_A = get_mrc_voxel_size_A(stack_path)

        if arr.ndim != 3 or arr.shape[0] == 0:
            return None

        idxs = sample_indices_evenly(arr.shape[0], int(cfg["sample_n"]))
        if not idxs:
            return None

        cols = max(1, int(cfg["cols"]))
        rows = int(math.ceil(len(idxs) / cols))

        fig, axs = plt.subplots(
            rows,
            cols,
            figsize=(cols * float(cfg["tile_inches"]), rows * float(cfg["tile_inches"])),
            dpi=int(cfg["dpi"]),
            squeeze=False,
        )
        fig.patch.set_facecolor(cfg["facecolor"])
        plt.subplots_adjust(
            left=0.02,
            right=0.98,
            bottom=0.02,
            top=0.98,
            wspace=cfg["wspace"],
            hspace=cfg["hspace"]
        )

        flat_axes = axs.ravel()

        for ax in flat_axes:
            ax.axis("off")

        for k, idx in enumerate(idxs):
            img = particle_lowpass_for_display(arr[idx])
            kw = _particle_imshow_kwargs(
                img,
                invert=cfg["invert"],
                autoscale=cfg["autoscale"],
                p_lo=cfg["p_lo"],
                p_hi=cfg["p_hi"],
            )

            flat_axes[k].imshow(img, origin="upper", **kw)

            if cfg["add_indices"]:
                flat_axes[k].text(
                    0.03,
                    0.97,
                    str(idx),
                    transform=flat_axes[k].transAxes,
                    ha="left",
                    va="top",
                    fontsize=cfg["index_fontsize"],
                    color=cfg["index_color"],
                    bbox=dict(
                        facecolor=cfg["index_bbox_facecolor"],
                        alpha=cfg["index_bbox_alpha"],
                        pad=cfg["index_bbox_pad"],
                        edgecolor=cfg["index_bbox_edgecolor"],
                    ),
                )

        return mplfig_to_pil(fig)

    except Exception as e:
        print(f"Warning: failed particle montage for {stack_path}: {e}")
        return None


def load_particle_multi_stack_montage_matplotlib(
    stack_paths: List[str],
    row_labels: Optional[List[str]] = None,
    cfg: Optional[dict] = None,
) -> Optional[Image.Image]:
    cfg = dict(cfg or {})
    _require_cfg_keys(
        "particle multi cfg",
        cfg,
        [
            "per_stack",
            "max_stacks",
            "tile_inches",
            "dpi",
            "invert",
            "autoscale",
            "p_lo",
            "p_hi",
            "wspace",
            "hspace",
            "left_margin",
            "add_indices",
            "facecolor",
            "index_fontsize",
            "index_color",
            "index_bbox_facecolor",
            "index_bbox_alpha",
            "index_bbox_pad",
            "index_bbox_edgecolor",
            "row_label_fontsize",
            "row_label_color",
        ],
    )

    row_labels = row_labels or []
    rows_data = []

    for i, stack_path in enumerate(stack_paths[:int(cfg["max_stacks"])]):
        try:
            arr = load_mrc(stack_path)
            arr = normalize_stack(arr)

            if arr.ndim != 3 or arr.shape[0] == 0:
                continue

            idxs = sample_indices_evenly(arr.shape[0], int(cfg["per_stack"]))
            if not idxs:
                continue

            label = row_labels[i] if i < len(row_labels) else f"Stack {i+1}"
            rows_data.append((label, arr, idxs))

        except Exception as e:
            print(f"Warning: failed particle stack {stack_path}: {e}")
            continue

    if not rows_data:
        print("Warning: no particle rows were rendered")
        return None

    n_rows = len(rows_data)
    n_cols = max(len(idxs) for _, _, idxs in rows_data)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * float(cfg["tile_inches"]), n_rows * float(cfg["tile_inches"])),
        dpi=int(cfg["dpi"]),
        squeeze=False,
    )
    fig.patch.set_facecolor(cfg["facecolor"])
    plt.subplots_adjust(
        left=cfg["left_margin"],
        right=0.99,
        top=0.99,
        bottom=0.01,
        wspace=cfg["wspace"],
        hspace=cfg["hspace"],
    )

    for r in range(n_rows):
        label, arr, idxs = rows_data[r]

        for c in range(n_cols):
            ax = axs[r, c]
            ax.axis("off")

            if c < len(idxs):
                idx = idxs[c]
                img = particle_lowpass_for_display(arr[idx])
                kw = _particle_imshow_kwargs(
                    img,
                    invert=cfg["invert"],
                    autoscale=cfg["autoscale"],
                    p_lo=cfg["p_lo"],
                    p_hi=cfg["p_hi"],
                )
                ax.imshow(img, origin="upper", **kw)

                if cfg["add_indices"]:
                    ax.text(
                        0.03,
                        0.97,
                        str(idx),
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=cfg["index_fontsize"],
                        color=cfg["index_color"],
                        bbox=dict(
                            facecolor=cfg["index_bbox_facecolor"],
                            alpha=cfg["index_bbox_alpha"],
                            pad=cfg["index_bbox_pad"],
                            edgecolor=cfg["index_bbox_edgecolor"],
                        ),
                    )

        axs[r, 0].text(
            -0.08,
            0.5,
            label,
            transform=axs[r, 0].transAxes,
            ha="right",
            va="center",
            fontsize=cfg["row_label_fontsize"],
            color=cfg["row_label_color"],
        )

    return mplfig_to_pil(fig)


def make_particle_examples_panel(
    stack_path: Optional[str],
    size=(520, 420),
    title="Particles",
    cfg: Optional[dict] = None,
    title_font_size: int = 11,
    title_h: int = 20,
    title_y_pad: int = 2,
    title_bg=(255, 255, 255),
    title_text_color=(20, 20, 20),
    title_font_weight="normal",
) -> Image.Image:
    cfg = dict(cfg or {})


    if not stack_path:
        return make_placeholder(size=size, text="Particles unavailable")


    try:
        montage = load_particle_stack_montage_matplotlib(
            stack_path=stack_path,
            cfg=cfg,
        )
        if montage is None:
            return make_placeholder(size=size, text="Particles unavailable")


        panel_w = int(size[0])
        panel_h = int(size[1])
        title_band_h = int(title_h)
        content_h = panel_h - title_band_h


        particle_angpix_A = get_mrc_voxel_size_A(stack_path)
        scale_bar_length_A = cfg.get("scale_bar_length_A", None)


        chosen_bar_A = None
        label_text = None
        tile_src_w_px = None


        try:
            angpix = float(particle_angpix_A) if particle_angpix_A is not None else None
        except Exception:
            angpix = None


        try:
            arr = load_mrc(stack_path)
            arr = normalize_stack(arr)
            if arr.ndim == 3 and arr.shape[0] > 0:
                tile_src_w_px = int(arr[0].shape[1])
        except Exception:
            tile_src_w_px = None


        if (
            choose_scale_bar_for_display is not None
            and angpix is not None
            and np.isfinite(angpix)
            and angpix > 0
            and tile_src_w_px is not None
            and tile_src_w_px > 0
        ):
            # Since particle bar should match the pick-circle diameter, use the provided fixed length
            chosen_bar_A, label_text, _ = choose_scale_bar_for_display(
                display_size_px=tile_src_w_px,
                display_angpix_A=angpix,
                bar_length_A=scale_bar_length_A,
                target_frac=0.22,
                max_frac=0.33,
                label_unit="A",
            )


        # Use the same parameters that worked for the micrograph
        scale_font_size = 18
        scale_thickness = 6
        scale_top_margin = 0
        scale_gap = 6
        scale_bottom_margin = 3


        scale_font = load_font(scale_font_size, bold=False)


        if label_text:
            label_img = render_text_crop(
                label_text,
                scale_font,
                fg=(0, 0, 0),
                bg=(255, 255, 255),
                pad=12,
                out_pad=3,
            )
            label_w, label_h = label_img.size
            scale_strip_h = scale_top_margin + scale_thickness + scale_gap + label_h + scale_bottom_margin + 2
        else:
            label_img = None
            label_w = label_h = 0
            scale_strip_h = 0


        montage_h_budget = max(20, content_h - scale_strip_h)
        fitted_montage = fit_within(montage, panel_w, montage_h_budget)


        panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
        d = ImageDraw.Draw(panel)


        # Title band
        d.rectangle([0, 0, panel_w, title_band_h], fill=title_bg)


        title_font_obj = load_font(
            title_font_size,
            bold=(str(title_font_weight).lower() == "bold"),
        )
        try:
            tb = d.textbbox((0, 0), title, font=title_font_obj)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
        except Exception:
            tw, th = d.textsize(title, font=title_font_obj)


        tx = (panel_w - tw) // 2
        ty = (title_band_h - th) // 2 - int(title_y_pad)
        d.text((tx, ty), title, fill=title_text_color, font=title_font_obj)


        # Paste montage into image area
        image_y0 = title_band_h
        strip_y0 = title_band_h + montage_h_budget


        mx = (panel_w - fitted_montage.width) // 2
        my = image_y0 + (montage_h_budget - fitted_montage.height) // 2
        panel.paste(fitted_montage.convert("RGB"), (mx, my))


        # Draw bottom-left scale bar beneath montage
        if (
            chosen_bar_A is not None
            and label_img is not None
            and angpix is not None
            and fitted_montage.width > 0
            and tile_src_w_px is not None
            and tile_src_w_px > 0
        ):
            cols = max(1, int(cfg.get("cols", 3)))
            wspace = float(cfg.get("wspace", 0.1))
            subplot_left = 0.02
            subplot_right = 0.98
            usable_frac = subplot_right - subplot_left
            tile_w_frac = usable_frac / (cols + (cols - 1.0) * wspace)


            tile_display_w_px = float(fitted_montage.width) * tile_w_frac
            if tile_display_w_px > 1:
                display_angpix_A = float(angpix) * float(tile_src_w_px) / float(tile_display_w_px)
                bar_px = int(round(float(chosen_bar_A) / float(display_angpix_A)))


                side_margin_px = 14
                x_left = mx + side_margin_px
                x_right = x_left + bar_px - 1


                y_bar_top = strip_y0 + scale_top_margin
                y_bar_bottom = y_bar_top + scale_thickness - 1
                y_text = y_bar_bottom + 1 + scale_gap


                x_text = x_left + (bar_px - label_w) // 2
                x_text = max(0, min(panel_w - label_w, x_text))
                y_text = max(strip_y0, min(panel_h - label_h, y_text))


                d.rectangle(
                    [x_left, y_bar_top, x_right, y_bar_bottom],
                    fill=(0, 0, 0),
                )
                panel.paste(label_img, (x_text, y_text))


        return panel


    except Exception as e:
        return make_placeholder(size=size, text=f"Particles unavailable\n{e}")


def make_classavg_montages(
    mrc_path: str,
    class_info_map: Optional[dict] = None,
    cols: int = 4,
    rows_per_page: int = 3,
    tile_size: int = 260,
    dpi: int = 300,
    invert: bool = False,
    display_mode: str = "auto",
    sigma: float = 5.0,
    gamma: float = 1.0,
    interpolation: str = "none",
    central_frac: float = 0.8,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    edge_frac: float = 0.12,
    edge_p_lo: float = 5.0,
    edge_img_p_hi: float = 99.5,
    sort_by_count: bool = True,
    count_key: str = "particle_count",
    force_black_borders: bool = False,
    style_cfg: Optional[dict] = None,
    scale_bar_length_A: Optional[float] = None,
) -> List[Image.Image]:
    style_cfg = dict(style_cfg or {})
    _require_cfg_keys(
        "classavg style_cfg",
        style_cfg,
        [
            "facecolor",
            "cell_facecolor",
            "neutral_cell_border_color",
            "neutral_cell_border_width",
            "selected_border_color",
            "selected_border_width",
            "rejected_border_color",
            "rejected_border_width",
            "unknown_border_color",
            "unknown_border_width",
            "force_black_border_color",
            "force_black_border_width",
            "resolution_fontsize",
            "count_fontsize",
            "text_color",
            "img_extent",
            "subplot_left",
            "subplot_right",
            "subplot_bottom",
            "subplot_top",
            "subplot_wspace",
            "subplot_hspace",
        ],
    )

    arr = load_mrc(mrc_path)
    arr = normalize_stack(arr)
    classavg_angpix_A = get_mrc_voxel_size_A(mrc_path)

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        return []

    n = arr.shape[0]
    per_page = cols * rows_per_page
    pages: List[Image.Image] = []

    cmap = "gray_r" if invert else "gray"
    fig_w = (cols * tile_size) / float(dpi)
    fig_h = (rows_per_page * tile_size) / float(dpi)

    img_x0, img_x1, img_y0, img_y1 = style_cfg["img_extent"]

    white = (255, 255, 255)
    neutral_cell_border = _rgb_to_mpl(white)
    selected_border = _rgb_to_mpl(style_cfg["selected_border_color"])
    rejected_border = _rgb_to_mpl(style_cfg["rejected_border_color"])
    unknown_border = _rgb_to_mpl(style_cfg["unknown_border_color"])
    force_black_border = _rgb_to_mpl(style_cfg["force_black_border_color"])

    def get_count(info: dict) -> Optional[int]:
        if not info:
            return None
        for k in [count_key, "particle_count", "num_particles", "n_particles", "particles", "class_size", "num_particles_total"]:
            if k in info and info[k] is not None:
                try:
                    return int(info[k])
                except Exception:
                    try:
                        return int(float(info[k]))
                    except Exception:
                        pass
        return None

    ordered_indices = list(range(n))
    if sort_by_count:
        ordered_indices.sort(
            key=lambda i: (
                get_count((class_info_map or {}).get(i, {})) is None,
                -(get_count((class_info_map or {}).get(i, {})) or 0),
                i,
            )
        )

    for start in range(0, n, per_page):
        stop = min(start + per_page, n)
        page_indices = ordered_indices[start:stop]

        fig, axes = plt.subplots(
            rows_per_page,
            cols,
            figsize=(fig_w, fig_h),
            dpi=dpi,
            squeeze=False,
        )
        fig.patch.set_facecolor(style_cfg["facecolor"])

        fig.subplots_adjust(
            left=style_cfg["subplot_left"],
            right=style_cfg["subplot_right"],
            bottom=style_cfg["subplot_bottom"],
            top=style_cfg["subplot_top"],
            wspace=style_cfg["subplot_wspace"],
            hspace=style_cfg["subplot_hspace"],
        )

        flat_axes = axes.ravel()

        for ax in flat_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_facecolor(style_cfg["cell_facecolor"])
            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.add_patch(
                Rectangle(
                    (0.0, 0.0),
                    1.0,
                    1.0,
                    fill=False,
                    linewidth=style_cfg["neutral_cell_border_width"],
                    edgecolor=neutral_cell_border,
                    transform=ax.transAxes,
                    zorder=10,
                )
            )

        for local_i, global_i in enumerate(page_indices):
            ax = flat_axes[local_i]
            img = np.asarray(arr[global_i], dtype=np.float32)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

            info = (class_info_map or {}).get(global_i, {})
            selected = info.get("selected")
            res_A = info.get("res_A")
            particle_count = get_count(info)

            if force_black_borders:
                cell_border_color = force_black_border
                cell_border_width = style_cfg["force_black_border_width"]
            elif selected is True:
                cell_border_color = selected_border
                cell_border_width = style_cfg["selected_border_width"]
            elif selected is False:
                cell_border_color = rejected_border
                cell_border_width = style_cfg["rejected_border_width"]
            else:
                cell_border_color = unknown_border
                cell_border_width = style_cfg["unknown_border_width"]

            vmin, vmax = get_display_limits(
                img,
                display_mode=display_mode,
                sigma=sigma,
                central_frac=central_frac,
                p_lo=p_lo,
                p_hi=p_hi,
                edge_frac=edge_frac,
                edge_p_lo=edge_p_lo,
                edge_img_p_hi=edge_img_p_hi,
            )

            if gamma is not None and abs(float(gamma) - 1.0) > 1e-6:
                norm = PowerNorm(gamma=float(gamma), vmin=vmin, vmax=vmax, clip=True)
                ax.imshow(
                    img,
                    cmap=cmap,
                    origin="lower",
                    interpolation=interpolation,
                    norm=norm,
                    extent=(img_x0, img_x1, img_y0, img_y1),
                    zorder=1,
                )
            else:
                ax.imshow(
                    img,
                    cmap=cmap,
                    origin="lower",
                    interpolation=interpolation,
                    vmin=vmin,
                    vmax=vmax,
                    extent=(img_x0, img_x1, img_y0, img_y1),
                    zorder=1,
                )

            ax.add_patch(
                Rectangle(
                    (0.0, 0.0),
                    1.0,
                    1.0,
                    fill=False,
                    linewidth=cell_border_width,
                    edgecolor=cell_border_color,
                    transform=ax.transAxes,
                    zorder=11,
                )
            )

            if res_A is not None:
                try:
                    ax.text(
                        img_x0 - 0.03,
                        0.025,
                        f"{float(res_A):.1f} Å",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=style_cfg["resolution_fontsize"],
                        color=style_cfg["text_color"],
                        zorder=12,
                    )
                except Exception:
                    pass

            if particle_count is not None:
                ax.text(
                    img_x1 + 0.03,
                    0.025,
                    f"{particle_count:,}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=style_cfg["count_fontsize"],
                    color=style_cfg["text_color"],
                    zorder=12,
                )

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            facecolor=style_cfg["facecolor"],
            edgecolor="none",
        )
        plt.close(fig)

        buf.seek(0)
        page = Image.open(buf).convert("RGB")
        page.load()
        buf.close()
        
        # ------------------------------------------------------------
        # Add vertical scale bar to the top-left class-average tile
        # after the page is rendered, so it cannot be clipped by axes.
        # ------------------------------------------------------------
        if (
            page_indices
            and choose_scale_bar_for_display is not None
            and format_length is not None
            and classavg_angpix_A is not None
        ):
            try:
                angpix = float(classavg_angpix_A)
            except Exception:
                angpix = None


            if angpix is not None and np.isfinite(angpix) and angpix > 0:
                try:
                    src_img = np.asarray(arr[page_indices[0]], dtype=np.float32)
                    src_h_px = int(src_img.shape[0])
                except Exception:
                    src_h_px = None


                if src_h_px is not None and src_h_px > 0:
                    page_w, page_h = page.size


                    left = float(style_cfg["subplot_left"])
                    right = float(style_cfg["subplot_right"])
                    bottom = float(style_cfg["subplot_bottom"])
                    top = float(style_cfg["subplot_top"])
                    wspace = float(style_cfg["subplot_wspace"])
                    hspace = float(style_cfg["subplot_hspace"])


                    cell_w_frac = (right - left) / (cols + (cols - 1.0) * wspace)
                    cell_h_frac = (top - bottom) / (rows_per_page + (rows_per_page - 1.0) * hspace)


                    # top-left cell bounds in page pixel coordinates
                    cell_x0 = int(round(page_w * left))
                    cell_y0 = int(round(page_h * (1.0 - top)))
                    cell_w_px = int(round(page_w * cell_w_frac))
                    cell_h_px = int(round(page_h * cell_h_frac))


                    img_x0_frac, img_x1_frac, img_y0_frac, img_y1_frac = style_cfg["img_extent"]


                    disp_img_h_px = float(cell_h_px) * float(img_y1_frac - img_y0_frac)
                    display_angpix_A = float(angpix) * float(src_h_px) / float(disp_img_h_px)


                    chosen_bar_A, label_text, _ = choose_scale_bar_for_display(
                        display_size_px=int(round(disp_img_h_px)),
                        display_angpix_A=display_angpix_A,
                        bar_length_A=scale_bar_length_A,
                        target_frac=0.22,
                        max_frac=0.33,
                        label_unit="A",
                    )


                    if chosen_bar_A is not None and label_text:
                        bar_px = int(round(float(chosen_bar_A) / float(display_angpix_A)))


                        # Use the same text-rendering strategy as the micrograph
                        scale_font_size = 28
                        scale_thickness = 6
                        text_gap = 6


                        scale_font = load_font(scale_font_size, bold=False)
                        label_img = render_text_crop(
                            label_text,
                            scale_font,
                            fg=(0, 0, 0),
                            bg=(255, 255, 255),
                            pad=12,
                            out_pad=3,
                        )
                        label_img = label_img.rotate(90, expand=True, fillcolor=(255, 255, 255))
                        label_w, label_h = label_img.size


                        d_page = ImageDraw.Draw(page)


                        # Position in the white space left of the image, but on the page itself
                        img_y0_px = cell_y0 + int(round(cell_h_px * img_y0_frac))
                        img_x0_px = cell_x0 + int(round(cell_w_px * img_x0_frac))


                        bar_x = max(8, img_x0_px - 18)
                        bar_y0 = img_y0_px + int(round(0.10 * disp_img_h_px))
                        bar_y1 = bar_y0 + bar_px - 1


                        text_x = max(2, bar_x - text_gap - label_w)
                        text_y = bar_y0 + (bar_px - label_h) // 2


                        # Clamp fully inside the page
                        text_x = max(0, min(page_w - label_w, text_x))
                        text_y = max(0, min(page_h - label_h, text_y))
                        bar_y0 = max(0, min(page_h - 1, bar_y0))
                        bar_y1 = max(0, min(page_h - 1, bar_y1))


                        d_page.rectangle(
                            [bar_x, bar_y0, bar_x + scale_thickness - 1, bar_y1],
                            fill=(0, 0, 0),
                        )
                        page.paste(label_img, (text_x, text_y))
        
        pages.append(page)

    return pages


def make_micrograph_panel(
    ws,
    exp: dict,
    title_prefix: str,
    fmt_num,
    invert: bool = False,
    display_mode: str = "auto",
    sigma: float = 5.0,
    gamma: float = 1.0,
    central_frac: float = 0.8,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    edge_frac: float = 0.12,
    edge_p_lo: float = 5.0,
    edge_img_p_hi: float = 99.5,
    lowpass_A: float = 20.0,
    layout: Optional[dict] = None,
    style: Optional[dict] = None,
    plot_cfg: Optional[dict] = None,
) -> Image.Image:
    layout_cfg = dict(layout or {})
    style_cfg = dict(style or {})
    plot_cfg_full = dict(plot_cfg or {})

    _require_cfg_keys(
        "layout",
        layout_cfg,
        [
            "panel_w",
            "panel_h",
            "margin",
            "gap",
            "title_h",
            "meta_h",
            "title_meta_gap",
            "meta_line_gap",
            "meta_content_gap",
            "left_col_frac",
            "particles_h_frac",
            "ctf1d_h_frac",
            "min_particles_h",
            "min_ctf1d_h",
            "min_micrograph_h",
            "right_panel_count",
        ],
    )

    _require_cfg_keys(
        "style",
        style_cfg,
        [
            "title_font_size",
            "body_font_size",
            "small_title_font_size",
            "small_title_band_h",
            "small_title_y_pad",
            "title_color",
            "body_color",
            "border_color",
            "bg_color",
            "band_bg_color",
            "band_text_color",
            "band_font_weight",
            "panel_border_width",
            "panel_border_color",
        ],
    )

    _require_cfg_keys(
        "plot_cfg",
        plot_cfg_full,
        [
            "global_motion",
            "local_motion",
            "local_defocus",
            "ctf_1d",
            "particles",
        ],
    )

    W = int(layout_cfg["panel_w"])
    H = int(layout_cfg["panel_h"])
    M = int(layout_cfg["margin"])
    G = int(layout_cfg["gap"])

    title_h_min = int(layout_cfg["title_h"])
    meta_h_min = int(layout_cfg["meta_h"])
    title_meta_gap = int(layout_cfg["title_meta_gap"])
    meta_line_gap = int(layout_cfg["meta_line_gap"])
    meta_content_gap = int(layout_cfg["meta_content_gap"])

    left_col_frac = float(layout_cfg["left_col_frac"])
    particles_h_frac = float(layout_cfg["particles_h_frac"])
    ctf1d_h_frac = float(layout_cfg["ctf1d_h_frac"])
    min_particles_h = int(layout_cfg["min_particles_h"])
    min_ctf1d_h = int(layout_cfg["min_ctf1d_h"])
    min_micrograph_h = int(layout_cfg["min_micrograph_h"])

    # In the new layout, right_panel_count should usually be 3
    right_panel_count = max(1, int(layout_cfg["right_panel_count"]))

    title_font = load_font(int(style_cfg["title_font_size"]), bold=True)
    body_font = load_font(int(style_cfg["body_font_size"]), bold=False)

    small_title_font_size = int(style_cfg["small_title_font_size"])
    small_title_band_h = int(style_cfg["small_title_band_h"])
    small_title_y_pad = int(style_cfg["small_title_y_pad"])

    title_color = style_cfg["title_color"]
    body_color = style_cfg["body_color"]
    border_color = style_cfg["border_color"]
    bg_color = style_cfg["bg_color"]
    band_bg_color = style_cfg["band_bg_color"]
    band_text_color = style_cfg["band_text_color"]
    band_font_weight = style_cfg["band_font_weight"]

    panel_border_width = int(style_cfg["panel_border_width"])
    panel_border_color = style_cfg["panel_border_color"]

    def _sf(val, ndigits):
        if val is None:
            return "—"
        try:
            return fmt_num(val, ndigits)
        except Exception:
            try:
                return f"{float(val):.{ndigits}f}"
            except Exception:
                return "—"

    # ------------------------------------------------------------------
    # Header text
    # ------------------------------------------------------------------
    viewer_exp_num = exp.get("uid", exp.get("exposure_number", ""))
    title = (
        f"Exp #{viewer_exp_num} | "
        f"CTF {_sf(exp.get('ctf_fit_A'), 2)} Å | "
        f"Defocus {_sf(exp.get('defocus_um'), 2)} µm"
    )

    basename = (exp.get("abs_file_path") or "").split("/")[-1]
    active_picker, active_pick_count = get_active_picker_info(exp)
    angast = _get_exp_raw_nested(exp or {}, "groups", "exposure", "ctf", "df_angle_rad", default=None)
    phase = _get_exp_raw_nested(exp or {}, "groups", "exposure", "ctf", "phase_shift_rad", default=None)

    meta_line_1 = f"{basename} | {_sf(lowpass_A, 0)} Å lowpass"
    meta_line_2 = (
        f"Total motion: {_sf(exp.get('total_motion_pix'), 2)} px | "
        f"Max in-frame motion: {_sf(exp.get('max_inframe_motion'), 3)} | "
        f"Angast {_sf(angast, 3)} | "
        f"Phase {_sf(phase, 3)}"
    )
    meta_line_3 = (
        f"Picker: {active_picker} | "
        f"Picks: {active_pick_count} | "
        f"Extracted: {exp.get('extracted_particles', 0)}"
    )

    _tmp = Image.new("RGB", (10, 10), bg_color)
    _draw = ImageDraw.Draw(_tmp)

    title_bbox = _draw.textbbox((0, 0), title, font=title_font)
    title_text_h = title_bbox[3] - title_bbox[1]

    body_bbox = _draw.textbbox((0, 0), "Ag", font=body_font)
    body_line_h = body_bbox[3] - body_bbox[1]

    title_block_h = max(title_h_min, title_text_h)
    meta_block_h = max(meta_h_min, 3 * body_line_h + 2 * meta_line_gap)

    # ------------------------------------------------------------------
    # Overall content area
    # ------------------------------------------------------------------
    content_x = M
    content_y = M + title_block_h + title_meta_gap + meta_block_h + meta_content_gap
    content_w = W - 2 * M
    content_h = H - M - content_y

    left_w = int(round(content_w * left_col_frac))
    right_w = content_w - left_w - G

    # Bottom row: full-width 1D CTF
    ctf1d_h = max(min_ctf1d_h, int(round(content_h * ctf1d_h_frac)))
    upper_h = content_h - ctf1d_h - G

    # Under-micrograph row in left column:
    # left half = particles, right half = local defocus
    under_micro_h = max(min_particles_h, int(round(content_h * particles_h_frac)))
    micrograph_h = upper_h - under_micro_h - G

    # If the micrograph gets too short, shrink the under-micrograph row first,
    # then shrink the bottom 1D CTF row if needed.
    if micrograph_h < min_micrograph_h:
        deficit = min_micrograph_h - micrograph_h

        shrink_under = min(deficit, max(0, under_micro_h - min_particles_h))
        under_micro_h -= shrink_under
        deficit -= shrink_under

        if deficit > 0:
            shrink_ctf = min(deficit, max(0, ctf1d_h - min_ctf1d_h))
            ctf1d_h -= shrink_ctf
            deficit -= shrink_ctf

        if deficit > 0:
            take_under = min(deficit // 2 + deficit % 2, max(0, under_micro_h - 140))
            under_micro_h -= take_under
            deficit -= take_under

        if deficit > 0:
            take_ctf = min(deficit, max(0, ctf1d_h - 160))
            ctf1d_h -= take_ctf
            deficit -= take_ctf

        upper_h = content_h - ctf1d_h - G
        micrograph_h = upper_h - under_micro_h - G

    # Right column occupies only the upper area, split into 3 panels
    right_block_h = (upper_h - (right_panel_count - 1) * G) // right_panel_count

    # Under-micrograph row split into 2
    under_left_w = (left_w - G) // 2
    under_right_w = left_w - under_left_w - G

    # ------------------------------------------------------------------
    # Left column: micrograph
    # ------------------------------------------------------------------
    micrograph_path = exp.get("micrograph_path") or exp.get("thumb_path")
    micrograph_psize_A = exp.get("micrograph_psize_A")

    overlay_kwargs = get_pick_overlay_params(ws, exp)
    micro_scale_bar_length_A = _pick_diameter_A_from_overlay_params(overlay_kwargs)


    if micrograph_path:
        raw_micro = mrc_2d_to_pil(
            micrograph_path,
            invert=invert,
            display_mode=display_mode,
            sigma=sigma,
            gamma=gamma,
            central_frac=central_frac,
            p_lo=p_lo,
            p_hi=p_hi,
            edge_frac=edge_frac,
            edge_p_lo=edge_p_lo,
            edge_img_p_hi=edge_img_p_hi,
            lowpass_A=lowpass_A,
            angpix_A=micrograph_psize_A,
            origin="upper",
        )


        raw_micro = overlay_blob_picks(
            raw_micro,
            exp.get("pick_cs_path"),
            **overlay_kwargs,
        )


        panel_w = int(left_w)
        panel_h = int(micrograph_h)
        title_h = int(small_title_band_h)
        content_h = panel_h - title_h


        # Choose the scale bar from the full micrograph field width
        source_micro_w = raw_micro.width


        try:
            micro_angpix = float(micrograph_psize_A) if micrograph_psize_A is not None else None
        except Exception:
            micro_angpix = None


        chosen_bar_A = None
        label_text = None


        if (
            choose_scale_bar_for_display is not None
            and micro_angpix is not None
            and np.isfinite(micro_angpix)
            and micro_angpix > 0
        ):
            chosen_bar_A, label_text, _ = choose_scale_bar_for_display(
                display_size_px=source_micro_w,
                display_angpix_A=micro_angpix,
                bar_length_A=None,
                target_frac=0.22,
                max_frac=0.33,
                label_unit="nm",
            )


        # Compute the exact scale-bar strip height first
        scale_font_size = 18
        scale_thickness = 6
        scale_top_margin = 0
        scale_gap = 6
        scale_bottom_margin = 3

        scale_font = load_font(scale_font_size, bold=False)
        tmp = Image.new("RGB", (20, 20), "white")
        tmp_draw = ImageDraw.Draw(tmp)

        if label_text:
            label_img = render_text_crop(
                label_text,
                scale_font,
                fg=(0, 0, 0),
                bg=(255, 255, 255),
                pad=12,
            )
            label_w, label_h = label_img.size
        else:
            label_img = None
            label_w, label_h = 0, 0


        scale_strip_h = scale_top_margin + scale_thickness + scale_gap + label_h + scale_bottom_margin + 2
        image_h_budget = content_h - scale_strip_h


        # Fit the micrograph into the exact remaining image area
        left_micro_img = fit_within(raw_micro, panel_w, image_h_budget)


        # Create final fixed-size micrograph panel
        left_micro = Image.new("RGB", (panel_w, panel_h), bg_color)
        d_micro = ImageDraw.Draw(left_micro)


        # Title band
        d_micro.rectangle([0, 0, panel_w, title_h], fill=band_bg_color)


        title_font_obj = load_font(
            small_title_font_size,
            bold=(str(band_font_weight).lower() == "bold"),
        )


        try:
            tb = d_micro.textbbox((0, 0), "Micrograph", font=title_font_obj)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
        except Exception:
            tw, th = d_micro.textsize("Micrograph", font=title_font_obj)


        tx = (panel_w - tw) // 2
        ty = (title_h - th) // 2 - int(small_title_y_pad)
        d_micro.text((tx, ty), "Micrograph", fill=band_text_color, font=title_font_obj)


        # Paste micrograph into the image area
        image_y0 = title_h
        strip_y0 = title_h + image_h_budget


        img_x = (panel_w - left_micro_img.width) // 2
        img_y = image_y0 + (image_h_budget - left_micro_img.height) // 2
        left_micro.paste(left_micro_img.convert("RGB"), (img_x, img_y))


        # Draw scale bar in the reserved strip below the image
        if (
            chosen_bar_A is not None
            and label_img is not None
            and micro_angpix is not None
            and left_micro_img.width > 0
        ):
            display_angpix_A = float(micro_angpix) * float(source_micro_w) / float(left_micro_img.width)
            bar_px = int(round(float(chosen_bar_A) / float(display_angpix_A)))


            side_margin_px = 16
            x_left = img_x + side_margin_px
            x_right = x_left + bar_px


            y_bar_top = strip_y0 + scale_top_margin
            y_bar_bottom = y_bar_top + scale_thickness
            y_text = y_bar_bottom + scale_gap


            x_text = x_left + (bar_px - label_w) // 2


            d_micro.rectangle(
                [x_left, y_bar_top, x_right, y_bar_bottom],
                fill=(0, 0, 0),
            )


            left_micro.paste(label_img, (x_text, y_text))

    else:
        left_micro = Image.new("RGB", (left_w, micrograph_h), bg_color)
        d_micro = ImageDraw.Draw(left_micro)
        d_micro.rectangle([0, 0, left_w, small_title_band_h], fill=band_bg_color)


        title_font_obj = load_font(
            small_title_font_size,
            bold=(str(band_font_weight).lower() == "bold"),
        )
        try:
            tb = d_micro.textbbox((0, 0), "Micrograph", font=title_font_obj)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
        except Exception:
            tw, th = d_micro.textsize("Micrograph", font=title_font_obj)


        tx = (left_w - tw) // 2
        ty = (small_title_band_h - th) // 2 - int(small_title_y_pad)
        d_micro.text((tx, ty), "Micrograph", fill=band_text_color, font=title_font_obj)


        ph = make_placeholder(size=(left_w, micrograph_h - small_title_band_h), text="Micrograph unavailable")
        ph = fit_within(ph, left_w, micrograph_h - small_title_band_h)
        px = (left_w - ph.width) // 2
        py = small_title_band_h + ((micrograph_h - small_title_band_h - ph.height) // 2)
        left_micro.paste(ph, (px, py))


    # ------------------------------------------------------------------
    # Under micrograph: particles on left
    # ------------------------------------------------------------------
    particles_cfg = dict(plot_cfg_full["particles"])
    particles_cfg["scale_bar_length_A"] = micro_scale_bar_length_A

    particles_panel = make_particle_examples_panel(
        exp.get("particle_stack_path"),
        size=(under_left_w, under_micro_h),
        title="Particles",
        cfg=particles_cfg,
        title_font_size=small_title_font_size,
        title_h=small_title_band_h,
        title_y_pad=small_title_y_pad,
        title_bg=band_bg_color,
        title_text_color=band_text_color,
        title_font_weight=band_font_weight,
    )

    # ------------------------------------------------------------------
    # Under micrograph: global motion on right
    # ------------------------------------------------------------------
    if exp.get("rigid_motion_path"):
        global_motion = plot_global_motion_to_pil(
            exp["rigid_motion_path"],
            size=(under_right_w, under_micro_h),
            zero_shift_frame=exp.get("rigid_zero_shift_frame", 0),
            cfg=plot_cfg_full["global_motion"],
        )
        global_motion = fit_within(global_motion, under_right_w, under_micro_h)
    else:
        global_motion = make_placeholder(size=(under_right_w, under_micro_h), text="Global motion unavailable")
        global_motion = fit_within(global_motion, under_right_w, under_micro_h)

    # ------------------------------------------------------------------
    # Bottom full-width: 1D CTF
    # ------------------------------------------------------------------
    if exp.get("ctf_1d_path"):
        ctf1d = plot_ctf_1d_to_pil(
            exp["ctf_1d_path"],
            size=(content_w, ctf1d_h),
            exp=exp,
            cfg=plot_cfg_full["ctf_1d"],
        )
        ctf1d = fit_within(ctf1d, content_w, ctf1d_h)
    else:
        ctf1d = make_placeholder(size=(content_w, ctf1d_h), text="1D CTF unavailable")
        ctf1d = fit_within(ctf1d, content_w, ctf1d_h)

    # ------------------------------------------------------------------
    # Right column: 2D CTF
    # ------------------------------------------------------------------
    if exp.get("ctf_diag_path"):
        ctf2d_raw = mrc_2d_to_pil(
            exp["ctf_diag_path"],
            invert=invert,
            display_mode=display_mode,
            sigma=sigma,
            gamma=gamma,
            central_frac=central_frac,
            p_lo=p_lo,
            p_hi=p_hi,
            edge_frac=edge_frac,
            edge_p_lo=edge_p_lo,
            edge_img_p_hi=edge_img_p_hi,
            origin="upper",
        ).convert("RGB")

        ctf2d = add_plot_style_title_band(
            ctf2d_raw,
            "2D CTF",
            canvas_size=(right_w, right_block_h),
            font_size=small_title_font_size,
            title_h=small_title_band_h,
            y_pad=small_title_y_pad,
            bg=band_bg_color,
            text_color=band_text_color,
            font_weight=band_font_weight,
            dpi=100,
        )
        ctf2d = fit_within(ctf2d, right_w, right_block_h)
    else:
        ctf2d = make_placeholder(size=(right_w, right_block_h), text="2D CTF unavailable")
        ctf2d = fit_within(ctf2d, right_w, right_block_h)

    # ------------------------------------------------------------------
    # Right column: Local Defocus
    # ------------------------------------------------------------------
    micrograph_shape = _infer_micrograph_shape_from_exp(exp)

    if exp.get("ctf_spline_path"):
        local_defocus = plot_ctf_defocus_landscape_to_pil(
            exp["ctf_spline_path"],
            size=(right_w, right_block_h),
            micrograph_shape=micrograph_shape,
            cfg=plot_cfg_full["local_defocus"],
        )
        local_defocus = fit_within(local_defocus, right_w, right_block_h)
    else:
        local_defocus = make_placeholder(size=(right_w, right_block_h), text="Local defocus unavailable")
        local_defocus = fit_within(local_defocus, right_w, right_block_h)


    # ------------------------------------------------------------------
    # Right column: Local motion
    # ------------------------------------------------------------------
    if exp.get("spline_motion_path"):
        movie_shape = _get_exp_raw_nested(
            exp,
            "groups", "exposure", "movie_blob", "shape",
            default=None,
        )
        movie_psize_A = _get_exp_raw_nested(
            exp,
            "groups", "exposure", "movie_blob", "psize_A",
            default=None,
        )

        if isinstance(movie_shape, (list, tuple, np.ndarray)) and len(movie_shape) == 3:
            movie_shape = tuple(int(v) for v in movie_shape)
        else:
            movie_shape = None

        try:
            movie_psize_A = float(movie_psize_A) if movie_psize_A is not None else exp.get("micrograph_psize_A")
        except Exception:
            movie_psize_A = exp.get("micrograph_psize_A")

        local_motion = plot_local_motion_to_pil(
            exp["spline_motion_path"],
            size=(right_w, right_block_h),
            movie_shape=movie_shape,
            angpix_A=movie_psize_A,
            rigid_motion_path=exp.get("rigid_motion_path"),
            cfg=plot_cfg_full["local_motion"],
        )
        local_motion = fit_within(local_motion, right_w, right_block_h)
    else:
        local_motion = make_placeholder(size=(right_w, right_block_h), text="Local motion unavailable")
        local_motion = fit_within(local_motion, right_w, right_block_h)

    # ------------------------------------------------------------------
    # Canvas
    # ------------------------------------------------------------------
    img = Image.new("RGB", (W, H), bg_color)
    d = ImageDraw.Draw(img)

    d.text((M, M), title, fill=title_color, font=title_font)

    meta_y = M + title_block_h + title_meta_gap
    d.multiline_text(
        (M, meta_y),
        f"{meta_line_1}\n{meta_line_2}\n{meta_line_3}",
        fill=body_color,
        font=body_font,
        spacing=meta_line_gap,
    )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    left_x = content_x
    right_x = content_x + left_w + G

    # Upper-left
    micro_slot = (left_x, content_y, left_w, micrograph_h)

    # Under-micrograph split row
    under_y = content_y + micrograph_h + G
    particles_slot = (left_x, under_y, under_left_w, under_micro_h)
    global_slot = (left_x + under_left_w + G, under_y, under_right_w, under_micro_h)

    # Right column, upper area only
    r0y = content_y
    r1y = r0y + right_block_h + G
    r2y = r1y + right_block_h + G

    ctf2d_slot = (right_x, r0y, right_w, right_block_h)
    local_defocus_slot = (right_x, r1y, right_w, right_block_h)
    local_motion_slot = (right_x, r2y, right_w, right_block_h)

    # Bottom full-width
    ctf1d_y = content_y + upper_h + G
    ctf1d_slot = (content_x, ctf1d_y, content_w, ctf1d_h)

    def _center_in_slot(im, slot):
        sx, sy, sw, sh = slot
        return sx + (sw - im.width) // 2, sy + (sh - im.height) // 2

    micro_x, micro_y = _center_in_slot(left_micro, micro_slot)
    particles_x, particles_y = _center_in_slot(particles_panel, particles_slot)
    defocus_x, defocus_y = _center_in_slot(local_defocus, local_defocus_slot)

    ctf2d_x, ctf2d_y = _center_in_slot(ctf2d, ctf2d_slot)
    global_x, global_y = _center_in_slot(global_motion, global_slot)
    local_x, local_y = _center_in_slot(local_motion, local_motion_slot)

    ctf1d_x, ctf1d_y = _center_in_slot(ctf1d, ctf1d_slot)

    # ------------------------------------------------------------------
    # Paste
    # ------------------------------------------------------------------
    img.paste(left_micro, (micro_x, micro_y))
    img.paste(particles_panel, (particles_x, particles_y))
    img.paste(local_defocus, (defocus_x, defocus_y))

    img.paste(ctf2d, (ctf2d_x, ctf2d_y))
    img.paste(global_motion, (global_x, global_y))
    img.paste(local_motion, (local_x, local_y))

    img.paste(ctf1d, (ctf1d_x, ctf1d_y))

    # ------------------------------------------------------------------
    # Optional borders
    # ------------------------------------------------------------------
    if panel_border_width > 0:
        for sx, sy, sw, sh in [
            micro_slot,
            particles_slot,
            local_defocus_slot,
            ctf2d_slot,
            global_slot,
            local_motion_slot,
            ctf1d_slot,
        ]:
            d.rectangle(
                (sx, sy, sx + sw, sy + sh),
                outline=panel_border_color,
                width=panel_border_width,
            )

    return img

