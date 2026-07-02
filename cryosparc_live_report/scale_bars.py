#!/usr/bin/env python3
# coding: utf-8


"""
Scale-bar helpers for PIL images and matplotlib axes.


Direct dependencies
-------------------
- Pillow


Standard library dependencies
-----------------------------
- math
- typing
"""


import math
from typing import Optional, Tuple


from PIL import Image, ImageDraw, ImageFont




def load_font(size=16, bold=False):
    """
    Load a reasonable sans-serif font if available, otherwise fall back to
    PIL's default font.
    """
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




def format_length(length_A: float, unit_mode: str = "A") -> str:
    """
    Format a physical length in Å or nm.
    """
    try:
        x = float(length_A)
    except Exception:
        return ""


    if not math.isfinite(x) or x <= 0:
        return ""


    mode = str(unit_mode or "A").strip().lower()


    if mode == "nm":
        nm = x / 10.0
        if abs(nm - round(nm)) < 1e-6:
            return f"{int(round(nm))} nm"
        return f"{nm:.1f} nm"


    if abs(x - round(x)) < 1e-6:
        return f"{int(round(x))} Å"
    return f"{x:.1f} Å"




def choose_nice_scale_bar_length_A(
    field_size_A: float,
    target_frac: float = 0.22,
    max_frac: float = 0.33,
    allowed_A=None,
) -> Optional[float]:
    """
    Choose a visually reasonable scale-bar length in Å for a field of view.
    """
    try:
        field_size_A = float(field_size_A)
    except Exception:
        return None


    if not math.isfinite(field_size_A) or field_size_A <= 0:
        return None


    if allowed_A is None:
        allowed_A = [
            5, 10, 20, 25, 50,
            75, 100, 150, 200, 250, 300, 400, 500,
            750, 1000, 1500, 2000, 2500, 5000,
        ]


    target = float(field_size_A) * float(target_frac)
    max_len = float(field_size_A) * float(max_frac)


    candidates = [float(v) for v in allowed_A if 0 < float(v) <= max_len]
    if not candidates:
        return None


    return min(candidates, key=lambda v: (abs(v - target), -v))




def choose_scale_bar_for_display(
    display_size_px: int,
    display_angpix_A: float,
    bar_length_A: Optional[float] = None,
    target_frac: float = 0.22,
    max_frac: float = 0.33,
    label_unit: str = "A",
) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    """
    Choose scale-bar length, formatted label, and field-of-view fraction.
    """
    try:
        display_size_px = int(display_size_px)
        display_angpix_A = float(display_angpix_A)
    except Exception:
        return None, None, None


    if display_size_px <= 0 or not math.isfinite(display_angpix_A) or display_angpix_A <= 0:
        return None, None, None


    field_size_A = float(display_size_px) * float(display_angpix_A)


    if bar_length_A is None:
        bar_length_A = choose_nice_scale_bar_length_A(
            field_size_A,
            target_frac=target_frac,
            max_frac=max_frac,
        )


    if bar_length_A is None:
        return None, None, None


    frac = float(bar_length_A) / max(field_size_A, 1e-12)
    if not math.isfinite(frac) or frac <= 0 or frac >= 1:
        return None, None, None


    return float(bar_length_A), format_length(bar_length_A, unit_mode=label_unit), frac




def _text_size(draw, text, font):
    """
    Measure text size across Pillow versions.
    """
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        return draw.textsize(text, font=font)




def add_bottom_scale_bar_pil(
    img: Image.Image,
    display_angpix_A: float,
    bar_length_A: Optional[float] = None,
    align: str = "left",
    side_margin_px: int = 16,
    top_margin_px: int = 6,
    gap_px: int = 4,
    bottom_margin_px: int = 6,
    thickness_px: Optional[int] = None,
    font_size: Optional[int] = None,
    bar_color=(0, 0, 0),
    text_color=(0, 0, 0),
    bg_color=(255, 255, 255),
    target_frac: float = 0.22,
    max_frac: float = 0.33,
    label_unit: str = "A",
) -> Image.Image:
    """
    Add a scale bar in a new white strip below the image.
    """
    if img is None:
        return img


    W, H = img.size
    if W <= 0 or H <= 0:
        return img


    chosen_A, label_text, _ = choose_scale_bar_for_display(
        display_size_px=W,
        display_angpix_A=display_angpix_A,
        bar_length_A=bar_length_A,
        target_frac=target_frac,
        max_frac=max_frac,
        label_unit=label_unit,
    )
    if chosen_A is None or not label_text:
        return img


    bar_px = int(round(float(chosen_A) / float(display_angpix_A)))
    if bar_px <= 0 or bar_px > (W - 2 * side_margin_px):
        return img


    if thickness_px is None:
        thickness_px = max(3, int(round(H * 0.012)))
    if font_size is None:
        font_size = max(12, int(round(H * 0.05)))


    font = load_font(font_size, bold=False)


    tmp = Image.new("RGB", (W, max(10, H)), bg_color)
    td = ImageDraw.Draw(tmp)
    text_w, text_h = _text_size(td, label_text, font)


    strip_h = top_margin_px + thickness_px + gap_px + text_h + bottom_margin_px
    out = Image.new("RGB", (W, H + strip_h), bg_color)
    out.paste(img.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(out)


    y_bar_top = H + top_margin_px
    y_bar_bottom = y_bar_top + thickness_px
    y_text = y_bar_bottom + gap_px


    if align == "right":
        x_left = W - side_margin_px - bar_px
    else:
        x_left = side_margin_px


    x_right = x_left + bar_px
    x_text = x_left + (bar_px - text_w) // 2


    draw.rectangle([x_left, y_bar_top, x_right, y_bar_bottom], fill=bar_color)
    draw.text((x_text, y_text), label_text, fill=text_color, font=font)


    return out




def add_inset_scale_bar_pil(
    img: Image.Image,
    display_angpix_A: float,
    bar_length_A: Optional[float] = None,
    align: str = "left",
    side_margin_px: int = 18,
    bottom_margin_px: int = 16,
    text_gap_px: int = 5,
    thickness_px: Optional[int] = None,
    font_size: Optional[int] = None,
    bar_color=(0, 0, 0),
    text_color=(0, 0, 0),
    target_frac: float = 0.22,
    max_frac: float = 0.33,
    label_unit: str = "A",
) -> Image.Image:
    """
    Draw a scale bar directly inside the image near the bottom edge.
    """
    if img is None:
        return img


    W, H = img.size
    if W <= 0 or H <= 0:
        return img


    chosen_A, label_text, _ = choose_scale_bar_for_display(
        display_size_px=W,
        display_angpix_A=display_angpix_A,
        bar_length_A=bar_length_A,
        target_frac=target_frac,
        max_frac=max_frac,
        label_unit=label_unit,
    )
    if chosen_A is None or not label_text:
        return img


    bar_px = int(round(float(chosen_A) / float(display_angpix_A)))
    if bar_px <= 0 or bar_px > (W - 2 * side_margin_px):
        return img


    if thickness_px is None:
        thickness_px = max(3, int(round(min(W, H) * 0.010)))
    if font_size is None:
        font_size = max(12, int(round(min(W, H) * 0.040)))


    font = load_font(font_size, bold=False)


    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)


    text_w, text_h = _text_size(draw, label_text, font)


    y_bar_bottom = H - bottom_margin_px
    y_bar_top = y_bar_bottom - thickness_px
    y_text = y_bar_top - text_gap_px - text_h


    if align == "right":
        x_left = W - side_margin_px - bar_px
    else:
        x_left = side_margin_px


    x_right = x_left + bar_px
    x_text = x_left + (bar_px - text_w) // 2


    if y_text < 2:
        return img


    draw.rectangle([x_left, y_bar_top, x_right, y_bar_bottom], fill=bar_color)
    draw.text((x_text, y_text), label_text, fill=text_color, font=font)


    return out




def add_vertical_scale_bar_to_axes_fraction(
    ax,
    length_frac: float,
    label_text: str,
    x: float = 0.07,
    y0: float = 0.18,
    linewidth: float = 2.0,
    fontsize: int = 8,
    color: str = "black",
    text_color: str = "black",
    text_dx: float = 0.05,
    zorder: int = 50,
):
    """
    Draw a vertical scale bar in matplotlib axes coordinates.
    """
    try:
        length_frac = float(length_frac)
        x = float(x)
        y0 = float(y0)
    except Exception:
        return


    if not math.isfinite(length_frac) or length_frac <= 0 or length_frac >= 1:
        return


    y1 = y0 + length_frac
    if y1 > 1.0:
        return


    ax.plot(
        [x, x],
        [y0, y1],
        transform=ax.transAxes,
        color=color,
        lw=linewidth,
        solid_capstyle="butt",
        zorder=zorder,
        clip_on=False,
    )


    ax.text(
        x - text_dx,
        0.5 * (y0 + y1),
        label_text,
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        zorder=zorder,
        clip_on=False,
    )


