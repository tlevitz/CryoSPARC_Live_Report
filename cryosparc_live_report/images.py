#!/usr/bin/env python3
# coding: utf-8

import io
import math
from io import BytesIO
from typing import List, Optional, Tuple

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
    RESAMPLE = Image.Resampling.LANCZOS
except Exception:
    RESAMPLE = Image.LANCZOS


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


def load_mrc(path: str) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as m:
        return np.asarray(m.data)

def merge_nested_dicts(defaults: dict, user: Optional[dict]) -> dict:
    """
    Shallow merge at top level, and one level deep for nested dict values.
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
) -> Image.Image:
    W, H = canvas_size
    out = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(out)
    font = load_font(font_size, bold=False)

    try:
        bbox = d.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
    except Exception:
        tw = d.textsize(title, font=font)[0]

    tx = (W - tw) // 2
    ty = y_pad
    d.text((tx, ty), title, fill=(20, 20, 20), font=font)

    avail_h = max(1, H - title_h)
    inner = fit_within(img, W, avail_h)
    ix = (W - inner.width) // 2
    iy = title_h + (avail_h - inner.height) // 2
    out.paste(inner.convert("RGB"), (ix, iy))
    return out


def overlay_blob_picks(base_img: Image.Image, pick_cs_path: Optional[str], max_draw: int = 4000) -> Image.Image:
    img = base_img.convert("RGB")
    if not pick_cs_path or Dataset is None:
        return img

    try:
        ds = Dataset.load(pick_cs_path)
        n = len(ds)
        if n == 0:
            return img

        w, h = img.size
        draw = ImageDraw.Draw(img)

        step = max(1, int(math.ceil(n / max_draw)))
        radius = max(2, int(round(min(w, h) / 220.0)))
        line_w = 2

        for i in range(0, n, step):
            row = ds[i]
            x = float(row["location/center_x_frac"]) * w
            y = float(1.0 - row["location/center_y_frac"]) * h
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(50, 230, 70),
                width=line_w,
            )
        return img
    except Exception:
        return img


def spline_interp(gridshape, spl, pix):
    """
    gridshape: (N_Z, ny, nx) of the raw movie
    spl: spline lattice for one coordinate component, shape (K_Z, K_Y, K_X)
    pix: coordinates shaped (3, ...)
         axis 0 = z(frame), axis 1 = y, axis 2 = x
    """
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
    """
    gridshape: (N_Z, ny, nx)
    splxy: shape (2, K_Z, K_Y, K_X)
    pos: array of particle/patch positions, shape (N_P, 2), as (x, y) in raw movie pixels

    returns: trajectories shape (N_P, N_Z, 2)
    """
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
    render_scale: int = 3,
    cfg: Optional[dict] = None,
) -> Image.Image:
    defaults = {
        "threshold": 0.3,
        "smooth_window": 7,
        "dpi": 100,
        "render_scale": render_scale,
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
        "xlabel_fontsize": 8,
        "ylabel_fontsize": 8,
        "tick_labelsize": 7,
        "title_fontsize": 7.5,
        "legend_fontsize": 6.5,
        "top_axis_fontsize": 8,
        "top_tick_fontsize": 7,
        "tight_pad": 0.35,
        "resolution_ticks_A": [20, 15, 10, 8, 6, 5, 4, 3],
    }
    cfg = merge_nested_dicts(defaults, cfg)

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
        fit_cross_A = None
        if fit_cross_x is not None and fit_cross_x > 1e-8:
            fit_cross_A = 1.0 / float(fit_cross_x)

        df1_A = _get_exp_raw_nested(exp or {}, "groups", "exposure", "ctf", "df1_A", default=None)
        df2_A = _get_exp_raw_nested(exp or {}, "groups", "exposure", "ctf", "df2_A", default=None)
        angast = _get_exp_raw_nested(exp or {}, "groups", "exposure", "ctf", "df_angle_rad", default=None)
        phase = _get_exp_raw_nested(exp or {}, "groups", "exposure", "ctf", "phase_shift_rad", default=None)

        fit_A = exp.get("ctf_fit_A") if isinstance(exp, dict) else None
        if fit_A is None:
            fit_A = fit_cross_A

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
            ax2.set_xlabel("Resolution (Å)", fontsize=cfg["top_axis_fontsize"])

        title_bits = ["1D CTF | "]
        if df1_A is not None:
            title_bits.append(f"DF1 {float(df1_A):.0f} | ")
        if df2_A is not None:
            title_bits.append(f"DF2 {float(df2_A):.0f} | ")
        if angast is not None:
            title_bits.append(f"ANGAST {float(angast):.3f} | ")
        if phase is not None:
            title_bits.append(f"PHASE {float(phase):.3f} | ")
        if fit_A is not None:
            title_bits.append(f"FIT {float(fit_A):.2f} Å")

        ax.set_title("".join(title_bits), fontsize=cfg["title_fontsize"], pad=2.5)
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
    defaults = {
        "title": "Global motion",
        "dpi": 100,
        "line_color": "#6a51a3",
        "line_width": 1.2,
        "start_marker_ms": 2.0,
        "grid_color": "0.93",
        "grid_lw": 0.6,
        "axis_line_color": "0.88",
        "axis_line_lw": 0.8,
        "title_fontsize": 8,
        "tick_labelsize": 6,
        "tight_pad": 0.15,
        "facecolor": "white",
        "subtract_zero_frame": True,
    }
    cfg = merge_nested_dicts(defaults, cfg)

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

        ax.plot(
            x,
            y,
            color=cfg["line_color"],
            lw=cfg["line_width"],
            zorder=2,
        )
        ax.plot(
            x[0],
            y[0],
            "o",
            ms=cfg["start_marker_ms"],
            color=cfg["line_color"],
            zorder=3,
        )

        ax.axhline(0, color=cfg["axis_line_color"], lw=cfg["axis_line_lw"], zorder=0)
        ax.axvline(0, color=cfg["axis_line_color"], lw=cfg["axis_line_lw"], zorder=0)
        ax.grid(True, color=cfg["grid_color"], lw=cfg["grid_lw"])

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(cfg["title"], fontsize=cfg["title_fontsize"], pad=2)
        ax.tick_params(axis="both", labelsize=cfg["tick_labelsize"])

        fig.tight_layout(pad=cfg["tight_pad"])
        return mplfig_to_pil(fig)

    except Exception as e:
        return make_placeholder(size=size, text=f"Global motion unavailable\n{e}")

def plot_local_motion_to_pil(
    spline_motion_path: str,
    size=(420, 220),
    movie_shape: Optional[tuple] = None,   # (N_Z, ny, nx)
    angpix_A: Optional[float] = None,
    rigid_motion_path: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> Image.Image:
    defaults = {
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
        "title_fontsize": 8,
        "facecolor": "white",
        "tight_pad": 0.15,
    }
    cfg = merge_nested_dicts(defaults, cfg)

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
            ax.plot(
                px,
                py,
                color=traj_color,
                lw=cfg["traj_lw"],
                alpha=cfg["traj_alpha"],
                zorder=2,
            )
            ax.plot(
                px[0],
                py[0],
                "o",
                ms=cfg["start_marker_ms"],
                color=traj_color,
                alpha=cfg["traj_alpha"],
                zorder=3,
            )

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
    micrograph_shape: Optional[Tuple[int, int]] = None,   # (ny, nx)
    cfg: Optional[dict] = None,
) -> Image.Image:
    defaults = {
        "title": "Local Defocus",
        "dpi": 100,
        "display_grid": 180,
        "mode": "nearest",
        "elev": 28,
        "azim": -58,
        "cmap": "viridis",
        "z_half_range_A": 2500.0,
        "facecolor": "white",
        "xlabel": "X (pix)",
        "ylabel": "Y (pix)",
        "zlabel": "Defocus (µm)",
        "axis_label_fontsize": 6,
        "title_fontsize": 8,
        "tick_labelsize": 5,
        "box_aspect": (1.0, 1.0, 0.55),
        "colorbar": True,
        "colorbar_shrink": 0.62,
        "colorbar_pad": 0.05,
        "colorbar_fraction": 0.05,
        "colorbar_label": "Mean defocus (µm)",
        "colorbar_label_fontsize": 6,
        "tight_pad": 0.2,
    }
    cfg = merge_nested_dicts(defaults, cfg)

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

#        if cfg["colorbar"]:
#            cbar = fig.colorbar(
#                surf,
#                ax=ax,
#                shrink=cfg["colorbar_shrink"],
#                pad=cfg["colorbar_pad"],
#                fraction=cfg["colorbar_fraction"],
#            )
#            cbar.ax.tick_params(labelsize=cfg["tick_labelsize"])
#            cbar.set_label(cfg["colorbar_label"], fontsize=cfg["colorbar_label_fontsize"])

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
    """
    Match the recommended particle-display style:
      - low-pass with scikit-image Butterworth
      - cutoff_frequency_ratio = 6 / N
      - order = 1
    """
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
    autoscale: str = "imshow",   # "imshow", "minmax", or "percentile"
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
    sample_n: int = 12,
    cols: int = 4,
    tile_inches: float = 2.0,
    dpi: int = 100,
    invert: bool = False,
    autoscale: str = "imshow",   # "imshow", "minmax", or "percentile"
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    wspace: float = 0.1,
    hspace: float = 0.1,
    add_indices: bool = False,
) -> Optional[Image.Image]:
    """
    Render a particle montage using Matplotlib subplots, with display matched
    more closely to the recommended Butterworth-filtered imshow style.
    """
    try:
        arr = load_mrc(stack_path)
        arr = normalize_stack(arr)

        if arr.ndim != 3 or arr.shape[0] == 0:
            return None

        idxs = sample_indices_evenly(arr.shape[0], sample_n)
        if not idxs:
            return None

        cols = max(1, int(cols))
        rows = int(math.ceil(len(idxs) / cols))

        fig, axs = plt.subplots(
            rows,
            cols,
            figsize=(cols * tile_inches, rows * tile_inches),
            dpi=dpi,
            squeeze=False,
        )
        fig.patch.set_facecolor("white")
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

        flat_axes = axs.ravel()

        for ax in flat_axes:
            ax.axis("off")

        for k, idx in enumerate(idxs):
            img = particle_lowpass_for_display(arr[idx])
            kw = _particle_imshow_kwargs(
                img,
                invert=invert,
                autoscale=autoscale,
                p_lo=p_lo,
                p_hi=p_hi,
            )

            flat_axes[k].imshow(img, origin="upper", **kw)

            if add_indices:
                flat_axes[k].text(
                    0.03,
                    0.97,
                    str(idx),
                    transform=flat_axes[k].transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="yellow",
                    bbox=dict(facecolor="black", alpha=0.35, pad=1.5, edgecolor="none"),
                )

        return mplfig_to_pil(fig)

    except Exception as e:
        print(f"Warning: failed particle montage for {stack_path}: {e}")
        return None

def load_particle_multi_stack_montage_matplotlib(
    stack_paths: List[str],
    row_labels: Optional[List[str]] = None,
    per_stack: int = 6,
    max_stacks: int = 6,
    tile_inches: float = 1.8,
    dpi: int = 100,
    invert: bool = False,
    autoscale: str = "imshow",   # "imshow", "minmax", or "percentile"
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    wspace: float = 0.08,
    hspace: float = 0.08,
    left_margin: float = 0.14,
    add_indices: bool = False,
) -> Optional[Image.Image]:
    """
    Render multiple particle stacks as a Matplotlib montage:
      - one row per stack
      - Butterworth low-pass particle display
      - row labels on the left
    """
    row_labels = row_labels or []
    rows_data = []

    for i, stack_path in enumerate(stack_paths[:max_stacks]):
        try:
            arr = load_mrc(stack_path)
            arr = normalize_stack(arr)

            if arr.ndim != 3 or arr.shape[0] == 0:
                continue

            idxs = sample_indices_evenly(arr.shape[0], per_stack)
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
        figsize=(n_cols * tile_inches, n_rows * tile_inches),
        dpi=dpi,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(
        left=left_margin,
        right=0.99,
        top=0.99,
        bottom=0.01,
        wspace=wspace,
        hspace=hspace,
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
                    invert=invert,
                    autoscale=autoscale,
                    p_lo=p_lo,
                    p_hi=p_hi,
                )
                ax.imshow(img, origin="upper", **kw)

                if add_indices:
                    ax.text(
                        0.03,
                        0.97,
                        str(idx),
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=7,
                        color="yellow",
                        bbox=dict(facecolor="black", alpha=0.35, pad=1.2, edgecolor="none"),
                    )

        # Put row label on the first axis in the row
        axs[r, 0].text(
            -0.08,
            0.5,
            label,
            transform=axs[r, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
            color="black",
        )

    return mplfig_to_pil(fig)


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
) -> List[Image.Image]:
    arr = load_mrc(mrc_path)
    arr = normalize_stack(arr)

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

    img_x0 = 0.10
    img_x1 = 0.90
    img_y0 = 0.15
    img_y1 = 0.95

    neutral_cell_border = (180/255.0, 180/255.0, 180/255.0)

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
        fig.patch.set_facecolor("white")

        fig.subplots_adjust(
            left=0.02,
            right=0.98,
            bottom=0.02,
            top=0.98,
            wspace=0.08,
            hspace=0.08,
        )

        flat_axes = axes.ravel()

        for ax in flat_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.add_patch(
                Rectangle(
                    (0.0, 0.0),
                    1.0,
                    1.0,
                    fill=False,
                    linewidth=1.0,
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
                cell_border_color = (0.0, 0.0, 0.0)
                cell_border_width = 1.8
            elif selected is True:
                cell_border_color = (30/255.0, 150/255.0, 30/255.0)
                cell_border_width = 2.0
            elif selected is False:
                cell_border_color = (180/255.0, 60/255.0, 60/255.0)
                cell_border_width = 2.0
            else:
                cell_border_color = (255.0, 255.0, 255.0)
                cell_border_width = 1.5

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
                        fontsize=8,
                        color="black",
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
                    fontsize=8,
                    color="black",
                    zorder=12,
                )

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

        buf.seek(0)
        page = Image.open(buf).convert("RGB")
        page.load()
        buf.close()
        pages.append(page)

    return pages


def make_micrograph_panel(
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
    """
    Build one wide micrograph summary panel.

    Main layout:
      left   : large micrograph
      top    : 2D CTF | Global motion | Local motion | Local Defocus
      bottom : 1D CTF
      footer : one-line metadata summary

    You can customize appearance by passing optional dictionaries:
      - layout   : panel sizing / spacing
      - style    : font sizes / colors / title band sizes
      - plot_cfg : motion + defocus plot rendering settings

    Example:
        img = make_micrograph_panel(
            exp,
            "Accepted",
            fmt_num,
            display_mode="percentile",
            layout={
                "panel_w": 2300,
                "panel_h": 820,
                "left_frac": 0.30,
                "top_panel_count": 4,
                "gap": 14,
            },
            style={
                "title_font_size": 22,
                "body_font_size": 15,
                "small_title_font_size": 12,
                "small_title_band_h": 24,
            },
            plot_cfg={
                "local_motion_viewer_scale": 45.0,
                "defocus_display_grid": 200,
                "defocus_z_half_range_A": 3000.0,
            },
        )
    """
    # ------------------------------------------------------------------
    # 1) EDIT THESE SETTINGS
    # ------------------------------------------------------------------
    layout_defaults = {
        # Overall panel canvas size
        "panel_w": 2100,
        "panel_h": 760,

        # Outer margins and spacing
        "margin": 10,
        "gap": 12,

        # Reserved vertical bands
        "title_h": 38,
        "footer_h": 38,

        # Fraction of content width used by the left micrograph pane
        # Increase this to make the micrograph larger.
        # Decrease this to make the right-side plots larger.
        "left_frac": 0.28,

        # Number of square plots on the top-right row
        "top_panel_count": 4,

        # Safety limits for top/bottom areas
        "min_bottom_h": 150,
        "min_top_sq": 260,
    }

    style_defaults = {
        # Font sizes
        "title_font_size": 20,
        "body_font_size": 18,
        "small_title_font_size": 14,

        # Small title band above images like "2D CTF"
        "small_title_band_h": 20,
        "small_title_y_pad": 2,

        # Colors
        "title_color": (0, 0, 0),
        "body_color": (20, 20, 20),
        "border_color": (185, 185, 185),
        "bg_color": "white",
    }

    plot_defaults = {
        "global_motion": {
            "title": "Global motion",
            "line_width": 1.2,
            "title_fontsize": 8,
            "tick_labelsize": 6,
        },
        "local_motion": {
            "title": "Local motion",
            "patch_spacing_A": 380.0,
            "patch_size_A": 500.0,
            "viewer_scale": 40.0,
            "traj_lw": 1.0,
            "title_fontsize": 8,
        },
        "local_defocus": {
            "title": "Local Defocus",
            "display_grid": 180,
            "mode": "nearest",
            "elev": 28,
            "azim": -58,
            "cmap": "viridis",
            "z_half_range_A": 2500.0,
            "title_fontsize": 8,
        },
        "ctf_1d": {
            "render_scale": 3,
            "title_fontsize": 7.5,
            "legend_fontsize": 6.5,
            "tick_labelsize": 7,
        },
    }

    layout_cfg = dict(layout_defaults)
    style_cfg = dict(style_defaults)
    plot_cfg_full = merge_nested_dicts(plot_defaults, plot_cfg)

    if layout:
        layout_cfg.update(layout)
    if style:
        style_cfg.update(style)
    if plot_cfg:
        plot_cfg_full.update(plot_cfg)

    # ------------------------------------------------------------------
    # 2) UNPACK SETTINGS
    # ------------------------------------------------------------------
    W = int(layout_cfg["panel_w"])
    H = int(layout_cfg["panel_h"])

    M = int(layout_cfg["margin"])
    G = int(layout_cfg["gap"])

    title_h = int(layout_cfg["title_h"])
    footer_h = int(layout_cfg["footer_h"])

    left_frac = float(layout_cfg["left_frac"])
    top_panel_count = max(1, int(layout_cfg["top_panel_count"]))

    min_bottom_h = int(layout_cfg["min_bottom_h"])
    min_top_sq = int(layout_cfg["min_top_sq"])

    title_font = load_font(int(style_cfg["title_font_size"]), bold=True)
    body_font = load_font(int(style_cfg["body_font_size"]), bold=False)

    small_title_font_size = int(style_cfg["small_title_font_size"])
    small_title_band_h = int(style_cfg["small_title_band_h"])
    small_title_y_pad = int(style_cfg["small_title_y_pad"])

    title_color = style_cfg["title_color"]
    body_color = style_cfg["body_color"]
    border_color = style_cfg["border_color"]
    bg_color = style_cfg["bg_color"]

    # ------------------------------------------------------------------
    # 3) COMPUTE PANEL GEOMETRY
    # ------------------------------------------------------------------
    content_x = M
    content_y = M + title_h
    content_w = W - 2 * M
    content_h = H - 2 * M - title_h - footer_h

    left_w = int(content_w * left_frac)
    right_w = content_w - left_w - G

    # Size of each square in the top row
    top_sq = (right_w - (top_panel_count - 1) * G) // top_panel_count
    bottom_h = content_h - top_sq - G

    # Keep bottom panel from collapsing too much
    if bottom_h < min_bottom_h:
        top_sq = max(min_top_sq, top_sq - (min_bottom_h - bottom_h))
        bottom_h = content_h - top_sq - G

    # ------------------------------------------------------------------
    # 4) LOAD / RENDER LEFT MICROGRAPH
    # ------------------------------------------------------------------
    micrograph_path = exp.get("micrograph_path") or exp.get("thumb_path")
    micrograph_psize_A = exp.get("micrograph_psize_A")

    if micrograph_path:
        left = mrc_2d_to_pil(
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
        left = fit_within(left, left_w, content_h)
        left = overlay_blob_picks(left, exp.get("extracted_cs_path") or exp.get("pick_cs_path"))
    else:
        left = make_placeholder(size=(left_w, content_h), text="Micrograph unavailable")

    # ------------------------------------------------------------------
    # 5) TOP RIGHT: 2D CTF
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
            canvas_size=(top_sq, top_sq),
            font_size=small_title_font_size,
            title_h=small_title_band_h,
            y_pad=small_title_y_pad,
        )
    else:
        ctf2d = make_placeholder(size=(top_sq, top_sq), text="2D CTF unavailable")

    # ------------------------------------------------------------------
    # 6) TOP RIGHT: GLOBAL MOTION
    # ------------------------------------------------------------------
    if exp.get("rigid_motion_path"):
        global_motion = plot_global_motion_to_pil(
            exp["rigid_motion_path"],
            size=(top_sq, top_sq),
            zero_shift_frame=exp.get("rigid_zero_shift_frame", 0),
            cfg=plot_cfg_full["global_motion"],
        )
        global_motion = fit_within(global_motion, top_sq, top_sq)
    else:
        global_motion = make_placeholder(size=(top_sq, top_sq), text="Global motion unavailable")

    # ------------------------------------------------------------------
    # 7) TOP RIGHT: LOCAL MOTION
    # ------------------------------------------------------------------
    if exp.get("spline_motion_path"):
        movie_shape = _get_exp_raw_nested(exp, "groups", "exposure", "movie_blob", "shape", default=None)
        movie_psize_A = _get_exp_raw_nested(exp, "groups", "exposure", "movie_blob", "psize_A", default=None)

        if isinstance(movie_shape, (list, tuple, np.ndarray)) and len(movie_shape) == 3:
            movie_shape = tuple(int(v) for v in movie_shape)
        else:
            movie_shape = None

        try:
            movie_psize_A = float(movie_psize_A) if movie_psize_A is not None else exp.get("micrograph_psize_A")
        except Exception:
            movie_psize_A = exp.get("micrograph_psize_A")

        motion = plot_local_motion_to_pil(
            exp["spline_motion_path"],
            size=(top_sq, top_sq),
            movie_shape=movie_shape,
            angpix_A=movie_psize_A,
            rigid_motion_path=exp.get("rigid_motion_path"),
            cfg=plot_cfg_full["local_motion"],
        )
        motion = fit_within(motion, top_sq, top_sq)
    else:
        motion = make_placeholder(size=(top_sq, top_sq), text="Local motion unavailable")

    # ------------------------------------------------------------------
    # 8) TOP RIGHT: LOCAL DEFOCUS
    # ------------------------------------------------------------------
    micrograph_shape = _infer_micrograph_shape_from_exp(exp)

    if exp.get("ctf_spline_path"):
        local_defocus = plot_ctf_defocus_landscape_to_pil(
            exp["ctf_spline_path"],
            size=(top_sq, top_sq),
            micrograph_shape=micrograph_shape,
            cfg=plot_cfg_full["local_defocus"],
        )
        local_defocus = fit_within(local_defocus, top_sq, top_sq)
    else:
        local_defocus = make_placeholder(size=(top_sq, top_sq), text="Local defocus unavailable")

    # ------------------------------------------------------------------
    # 9) BOTTOM RIGHT: 1D CTF
    # ------------------------------------------------------------------
    if exp.get("ctf_1d_path"):
        ctf1d = plot_ctf_1d_to_pil(
            exp["ctf_1d_path"],
            size=(right_w, bottom_h),
            exp=exp,
            cfg=plot_cfg_full["ctf_1d"],
        )
        ctf1d = fit_within(ctf1d, right_w, bottom_h)
    else:
        ctf1d = make_placeholder(size=(right_w, bottom_h), text="1D CTF unavailable")

    # ------------------------------------------------------------------
    # 10) CREATE CANVAS
    # ------------------------------------------------------------------
    img = Image.new("RGB", (W, H), bg_color)
    d = ImageDraw.Draw(img)

    viewer_exp_num = exp.get("uid", exp.get("exposure_number", ""))
    title = (
        f"{title_prefix} | Exp #{viewer_exp_num} | "
        f"CTF {fmt_num(exp.get('ctf_fit_A'), 2)} Å | "
        f"Defocus {fmt_num(exp.get('defocus_um'), 2)} µm"
    )
    d.text((M, M), title, fill=title_color, font=title_font)

    # ------------------------------------------------------------------
    # 11) POSITION PANELS
    # ------------------------------------------------------------------
    rx = content_x + left_w + G
    top_y = content_y
    bottom_y = content_y + top_sq + G

    # Left micrograph centered in left region
    lx = content_x + (left_w - left.width) // 2
    ly = content_y + (content_h - left.height) // 2

    # Top row centered in right region
    top_row_w = top_panel_count * top_sq + (top_panel_count - 1) * G
    top_row_x = rx + (right_w - top_row_w) // 2

    panel_xs = [top_row_x + i * (top_sq + G) for i in range(top_panel_count)]

    ctf2d_box_x = panel_xs[0] if top_panel_count > 0 else top_row_x
    global_box_x = panel_xs[1] if top_panel_count > 1 else top_row_x
    local_box_x = panel_xs[2] if top_panel_count > 2 else top_row_x
    defocus_box_x = panel_xs[3] if top_panel_count > 3 else top_row_x

    ctf2d_x = ctf2d_box_x + (top_sq - ctf2d.width) // 2
    ctf2d_y = top_y + (top_sq - ctf2d.height) // 2

    global_x = global_box_x + (top_sq - global_motion.width) // 2
    global_y = top_y + (top_sq - global_motion.height) // 2

    local_x = local_box_x + (top_sq - motion.width) // 2
    local_y = top_y + (top_sq - motion.height) // 2

    defocus_x = defocus_box_x + (top_sq - local_defocus.width) // 2
    defocus_y = top_y + (top_sq - local_defocus.height) // 2

    ctf1d_x = rx + (right_w - ctf1d.width) // 2
    ctf1d_y = bottom_y + (bottom_h - ctf1d.height) // 2

    # ------------------------------------------------------------------
    # 12) PASTE PANELS
    # ------------------------------------------------------------------
    img.paste(left, (lx, ly))
    img.paste(ctf2d, (ctf2d_x, ctf2d_y))
    img.paste(global_motion, (global_x, global_y))
    img.paste(motion, (local_x, local_y))
    img.paste(local_defocus, (defocus_x, defocus_y))
    img.paste(ctf1d, (ctf1d_x, ctf1d_y))

    # ------------------------------------------------------------------
    # 13) DRAW BORDER BOXES
    # ------------------------------------------------------------------
    d.rectangle(
        (content_x, content_y, content_x + left_w, content_y + content_h),
        outline=border_color,
        width=1,
    )
    d.rectangle(
        (ctf2d_box_x, top_y, ctf2d_box_x + top_sq, top_y + top_sq),
        outline=border_color,
        width=1,
    )
    d.rectangle(
        (global_box_x, top_y, global_box_x + top_sq, top_y + top_sq),
        outline=border_color,
        width=1,
    )
    d.rectangle(
        (local_box_x, top_y, local_box_x + top_sq, top_y + top_sq),
        outline=border_color,
        width=1,
    )
    d.rectangle(
        (defocus_box_x, top_y, defocus_box_x + top_sq, top_y + top_sq),
        outline=border_color,
        width=1,
    )
    d.rectangle(
        (rx, bottom_y, rx + right_w, bottom_y + bottom_h),
        outline=border_color,
        width=1,
    )

    # ------------------------------------------------------------------
    # 14) FOOTER
    # ------------------------------------------------------------------
    basename = (exp.get("abs_file_path") or "").split("/")[-1]
    active_picker, active_pick_count = get_active_picker_info(exp)

    footer = (
        f"{basename} | {fmt_num(lowpass_A, 0)} Å lowpass | "
        f"Status: {exp.get('status','')} | "
        f"Picker: {active_picker} | "
        f"Picks: {active_pick_count} | "
        f"Extracted: {exp.get('extracted_particles', 0)} | "
        f"Max in-frame motion: {fmt_num(exp.get('max_inframe_motion'), 3)} | "
        f"Total motion: {fmt_num(exp.get('total_motion_pix'), 2)} px"
    )
    d.text((M, H - footer_h + 8), footer, fill=body_color, font=body_font)

    return img



