#!/usr/bin/env python3
# coding: utf-8

from typing import List, Tuple, Optional

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Frame, Paragraph, Spacer, Table, TableStyle, KeepInFrame
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader, simpleSplit

from PIL import Image, ImageChops

try:
    from epu.report_style import (
        RL_FONT_FAMILY,
        RL_FONT_FAMILY_BOLD,
        FONT_SIZES as PDF_FONT_SIZES,
    )
except Exception:
    RL_FONT_FAMILY = "Helvetica"
    RL_FONT_FAMILY_BOLD = "Helvetica-Bold"
    PDF_FONT_SIZES = {"title": 16, "body": 9, "caption": 8}

try:
    from epu.report_utils import draw_heading, draw_page_number, draw_frame_box
except Exception:
    def draw_heading(c, text, x, y, level="section", page_height=None, margin=None):
        c.setFont(RL_FONT_FAMILY_BOLD, 18 if level == "title" else 13)
        c.drawString(x, y, text)
        return y - (0.30 * inch if level == "title" else 0.20 * inch)

    def draw_page_number(c, page_num, width, margin):
        c.setFont(RL_FONT_FAMILY, 9)
        c.drawRightString(width - margin, 0.35 * inch, f"Page {page_num}")

    def draw_frame_box(c, x, y_top, w, h):
        c.rect(x, y_top - h, w, h, stroke=1, fill=0)
        
def _prepare_pdf_image(pil_img, draw_w_pt, draw_h_pt, max_dpi=450):
    """
    Prepare a PIL image for compact PDF embedding.


    - Flattens alpha onto white
    - Converts RGB->L if the image is actually grayscale
    - Downsamples to the maximum useful pixel size for the drawn size


    draw_w_pt / draw_h_pt are the final on-page dimensions in points.
    """
    if pil_img is None:
        return None


    img = pil_img


    # Flatten transparency onto white so we do not need mask="auto"
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, "white")
        alpha = img.getchannel("A") if "A" in img.getbands() else None
        bg.paste(img.convert("RGB"), mask=alpha)
        img = bg
    elif img.mode == "P":
        img = img.convert("RGB")
    elif img.mode not in ("1", "L", "RGB"):
        img = img.convert("RGB")


    # If image is really grayscale, store as L instead of RGB
    if img.mode == "RGB":
        try:
            r, g, b = img.split()
            if (
                ImageChops.difference(r, g).getbbox() is None and
                ImageChops.difference(r, b).getbbox() is None
            ):
                img = img.convert("L")
        except Exception:
            pass


    # Compute maximum useful raster size for the actual drawn size
    target_px_w = max(1, int(round(draw_w_pt / 72.0 * max_dpi)))
    target_px_h = max(1, int(round(draw_h_pt / 72.0 * max_dpi)))


    iw, ih = img.size
    scale = min(target_px_w / float(iw), target_px_h / float(ih), 1.0)


    if scale < 1.0:
        new_size = (
            max(1, int(round(iw * scale))),
            max(1, int(round(ih * scale))),
        )
        img = img.resize(new_size, Image.Resampling.LANCZOS)


    return img




def render_framed_pil(c, pil_img, x_left, y_top, max_w, max_h, frame_pad=4.0):
    iw, ih = pil_img.size
    scale = min((max_w - 2 * frame_pad) / max(iw, 1), (max_h - 2 * frame_pad) / max(ih, 1))
    dw, dh = iw * scale, ih * scale
    total_h = dh + 2 * frame_pad
    draw_frame_box(c, x_left, y_top, max_w, total_h)
    x_img = x_left + frame_pad + (max_w - 2 * frame_pad - dw) / 2.0
    y_img = y_top - frame_pad - dh


    prepped = _prepare_pdf_image(pil_img, dw, dh, max_dpi=450)
    c.drawImage(ImageReader(prepped), x_img, y_img, width=dw, height=dh, preserveAspectRatio=True)


    return total_h

def render_framed_pil_in_box(c, pil_img, x_left, y_top, box_w, box_h, frame_pad=4.0):
    draw_frame_box(c, x_left, y_top, box_w, box_h)


    if pil_img is None:
        return box_h


    iw, ih = pil_img.size
    if iw <= 0 or ih <= 0:
        return box_h


    scale = min(
        (box_w - 2 * frame_pad) / float(iw),
        (box_h - 2 * frame_pad) / float(ih),
    )
    scale = max(scale, 0.0)


    dw = iw * scale
    dh = ih * scale


    x_img = x_left + (box_w - dw) / 2.0
    y_img = y_top - box_h + (box_h - dh) / 2.0


    prepped = _prepare_pdf_image(pil_img, dw, dh, max_dpi=450)
    c.drawImage(
        ImageReader(prepped),
        x_img,
        y_img,
        width=dw,
        height=dh,
        preserveAspectRatio=True,
    )
    return box_h


def draw_missing_panel(c, x_left, y_top, box_w, box_h, label="Not available"):
    draw_frame_box(c, x_left, y_top, box_w, box_h)
    c.setFont(RL_FONT_FAMILY, 9)
    c.setFillColor(colors.grey)
    c.drawCentredString(x_left + box_w / 2.0, y_top - box_h / 2.0, label)
    c.setFillColor(colors.black)


REFINE_PANEL_TITLES = {
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




def _panel_title_for_key(key: str) -> str:
    return REFINE_PANEL_TITLES.get(key, key.replace("_", " ").title())

def draw_titled_panel_in_box(
    c,
    title,
    pil_img,
    x_left,
    y_top,
    box_w,
    box_h,
    title_fontsize=8,
    min_title_band_h=18.0,
    title_gap=3.0,
    frame_pad=3.0,
    draw_separator=True,
):
    """
    Draw a single framed panel where the title lives inside the top of the box
    and the image is centered in the remaining area.


    This preserves uniform final title size because the title is still rendered
    at PDF time, after the panel size is known.
    """
    # Outer frame around the entire panel, including title band
    draw_frame_box(c, x_left, y_top, box_w, box_h)


    inner_pad = frame_pad


    # Wrap title to panel width
    text_w = max(box_w - 2 * inner_pad - 2, 10)
    lines = simpleSplit(title, RL_FONT_FAMILY_BOLD, title_fontsize, text_w)


    # Keep title compact
    if len(lines) > 2:
        lines = lines[:2]
        if len(lines[1]) > 3:
            lines[1] = lines[1][:-3] + "..."


    line_h = title_fontsize + 2
    title_band_h = max(min_title_band_h, len(lines) * line_h + 6)


    # Draw title centered inside top band
    c.setFont(RL_FONT_FAMILY_BOLD, title_fontsize)
    text_y = y_top - title_fontsize - 2
    for line in lines:
        c.drawCentredString(x_left + box_w / 2.0, text_y, line)
        text_y -= line_h


    # Optional separator line between title band and image region
    img_region_top = y_top - title_band_h
    if draw_separator:
        c.line(x_left, img_region_top, x_left + box_w, img_region_top)


    # Available area for image inside same outer frame
    avail_w = box_w - 2 * inner_pad
    avail_h = box_h - title_band_h - title_gap - 2 * inner_pad


    if avail_h <= 8 or avail_w <= 8:
        return box_h


    img_x0 = x_left + inner_pad
    img_y_top = img_region_top - title_gap - inner_pad


    if pil_img is None:
        c.setFont(RL_FONT_FAMILY, 9)
        c.setFillColor(colors.grey)
        c.drawCentredString(
            x_left + box_w / 2.0,
            img_y_top - avail_h / 2.0,
            "Not available",
        )
        c.setFillColor(colors.black)
        return box_h


    iw, ih = pil_img.size
    if iw <= 0 or ih <= 0:
        return box_h


    scale = min(avail_w / float(iw), avail_h / float(ih))
    scale = max(scale, 0.0)


    dw = iw * scale
    dh = ih * scale


    x_img = img_x0 + (avail_w - dw) / 2.0
    y_img = img_y_top - dh - (avail_h - dh) / 2.0

    prepped = _prepare_pdf_image(pil_img, dw, dh, max_dpi=450)
    c.drawImage(
        ImageReader(prepped),
        x_img,
        y_img,
        width=dw,
        height=dh,
        preserveAspectRatio=True,
    )


    return box_h

def add_summary_pages(c, width, height, margin, project_folder_name, sections, page_num_start=1):
    page_num = page_num_start
    styles = getSampleStyleSheet()


    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName=RL_FONT_FAMILY,
        fontSize=PDF_FONT_SIZES.get("body", 9),
        leading=PDF_FONT_SIZES.get("body", 9) * 1.2,
    )
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontName=RL_FONT_FAMILY_BOLD,
        fontSize=PDF_FONT_SIZES.get("title", 16),
        leading=PDF_FONT_SIZES.get("title", 16) * 1.2,
        spaceAfter=8,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=body_style,
        fontName=RL_FONT_FAMILY_BOLD,
        fontSize=11,
        spaceBefore=6,
        spaceAfter=4,
    )
    key_style = ParagraphStyle("Key", parent=body_style, fontName=RL_FONT_FAMILY_BOLD)


    content = []
    content.append(Paragraph(f"CryoSPARC Live Summary: {project_folder_name}", title_style))
    content.append(Spacer(1, 0.08 * inch))


    for sec in sections:
        content.append(Paragraph(sec["title"], section_style))


        if sec.get("summary_html"):
            content.append(Paragraph(sec["summary_html"], body_style))
            content.append(Spacer(1, 0.06 * inch))


        table_data = [
            [Paragraph(str(k), key_style), Paragraph(str(v), body_style)]
            for k, v in sec.get("rows", [])
        ]
        if table_data:
            table = Table(table_data, colWidths=[3.0 * inch, 4.5 * inch], hAlign="LEFT")
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]))
            content.append(table)
            content.append(Spacer(1, 0.10 * inch))


    avail_w = width - 2 * margin
    avail_h = height - 2 * margin


    fitted = KeepInFrame(
        avail_w,
        avail_h,
        content,
        mode="shrink",   # only shrinks if needed
        hAlign="LEFT",
        vAlign="TOP",
    )


    frame = Frame(margin, margin, avail_w, avail_h, showBoundary=0)
    story = [fitted]
    frame.addFromList(story, c)


    draw_page_number(c, page_num, width, margin)
    c.showPage()
    return page_num + 1


def draw_single_image_page(c, page_num, width, height, margin, heading, pil_img, note=None):
    y = height - margin
    y = draw_heading(c, heading, margin, y, level="section", page_height=height, margin=margin)
    if note:
        note_font = "Helvetica-Oblique"
        note_size = 8
        note_leading = 11
        note_width = width - 2 * margin

        note_lines = simpleSplit(note, note_font, note_size, note_width)

        c.setFont(note_font, note_size)
        note_y = y - 2
        for line in note_lines:
            c.drawString(margin, note_y, line)
            note_y -= note_leading

        y = note_y - 6

    render_framed_pil(c, pil_img, margin, y, width - 2 * margin, y - margin)
    draw_page_number(c, page_num, width, margin)
    c.showPage()
    return page_num + 1


def draw_five_plot_page(c, page_num, width, height, margin, heading, plots, note=None):
    if not plots:
        return page_num

    y = height - margin

    y = draw_heading(c, heading, margin, y, level="section", page_height=height, margin=margin)

    if note:
        note_font = "Helvetica-Oblique"
        note_size = 8
        note_leading = 11
        note_width = width - 2 * margin

        note_lines = simpleSplit(note, note_font, note_size, note_width)

        c.setFont(note_font, note_size)
        note_y = y - 2
        for line in note_lines:
            c.drawString(margin, note_y, line)
            note_y -= note_leading

        y = note_y - 6

    max_w = width - 2 * margin

    # More generous spacing so titles do not touch plot frames
    inter_block_gap = 0.15 * inch
    title_band_h = 0.04 * inch
    title_to_box_gap = 0.08 * inch

    nplots = len(plots)
    avail_h = y - margin

    total_reserved = nplots * (title_band_h + title_to_box_gap) + (nplots - 1) * inter_block_gap
    plot_h = (avail_h - total_reserved) / float(nplots)

    for i, (title, plot_img) in enumerate(plots):
        # Title
        c.setFont(RL_FONT_FAMILY_BOLD, 9)
        c.drawString(margin, y - 1, title)
        y -= title_band_h

        # Small gap between title and framed plot
        y -= title_to_box_gap

        # Plot image
        used_h = render_framed_pil(
            c,
            plot_img,
            margin,
            y,
            max_w,
            plot_h,
            frame_pad=2.0,
        )
        y -= used_h

        # Gap before next block
        if i < nplots - 1:
            y -= inter_block_gap

    draw_page_number(c, page_num, width, margin)
    c.showPage()
    return page_num + 1

def draw_panel_pages(c, page_num, width, height, margin, heading, panels, note=None):
    """
    Draw panels in a 2x2 grid, exactly 4 per page.
    """
    if not panels:
        return page_num

    idx = 0
    cols = 2
    rows = 2

    # Slightly tighter gaps than before
    gap_x = 0.08 * inch
    gap_y = 0.10 * inch

    while idx < len(panels):
        y = height - margin
        y = draw_heading(
            c,
            heading if idx == 0 else f"{heading} (cont.)",
            margin,
            y,
            level="section",
            page_height=height,
            margin=margin,
        )

        if idx == 0 and note:
            note_font = "Helvetica-Oblique"
            note_size = 8
            note_leading = 10
            note_width = width - 2 * margin

            note_lines = simpleSplit(note, note_font, note_size, note_width)

            c.setFont(note_font, note_size)
            note_y = y - 2
            for line in note_lines:
                c.drawString(margin, note_y, line)
                note_y -= note_leading

            y = note_y - 4

        avail_w = width - 2 * margin
        avail_h = y - margin

        cell_w = (avail_w - (cols - 1) * gap_x) / cols
        cell_h = (avail_h - (rows - 1) * gap_y) / rows

        for r in range(rows):
            for col in range(cols):
                if idx >= len(panels):
                    break

                x = margin + col * (cell_w + gap_x)
                y_top = y - r * (cell_h + gap_y)

                render_framed_pil(
                    c,
                    panels[idx],
                    x,
                    y_top,
                    cell_w,
                    cell_h,
                    frame_pad=1.5,   # smaller than before so the image gets more room
                )
                idx += 1

        draw_page_number(c, page_num, width, margin)
        c.showPage()
        page_num += 1

    return page_num

def draw_initial_model_summary_pages(c, page_num, width, height, margin, heading, class_panels, note=None):
    """
    Draw one or more pages containing only initial-model surface-view panels.


    class_panels:
        [
            ("Surface Views - Class 1", PIL.Image),
            ("Surface Views - Class 2", PIL.Image),
            ...
        ]


    Layout:
      - heading + optional note
      - up to 3 full-width surface-view panels per page
    """
    if not class_panels:
        return page_num


    idx = 0
    panels_per_page = 3
    row_gap = 0.10 * inch


    while idx < len(class_panels):
        y = height - margin
        y = draw_heading(
            c,
            heading if idx == 0 else f"{heading} (cont.)",
            margin,
            y,
            level="section",
            page_height=height,
            margin=margin,
        )


        if idx == 0 and note:
            note_font = "Helvetica-Oblique"
            note_size = 8
            note_leading = 11
            note_width = width - 2 * margin


            note_lines = simpleSplit(note, note_font, note_size, note_width)
            c.setFont(note_font, note_size)


            note_y = y - 2
            for line in note_lines:
                c.drawString(margin, note_y, line)
                note_y -= note_leading


            y = note_y - 6


        avail_w = width - 2 * margin
        avail_h = y - margin


        n_this_page = min(panels_per_page, len(class_panels) - idx)
        inner_h = avail_h - (n_this_page - 1) * row_gap
        box_h = inner_h / float(n_this_page)


        y_top = y
        for _ in range(n_this_page):
            label, img = class_panels[idx]


            draw_titled_panel_in_box(
                c,
                label,
                img,
                margin,
                y_top,
                avail_w,
                box_h,
                title_fontsize=9,
                min_title_band_h=0.22 * inch,
                title_gap=4.0,
                frame_pad=3.0,
            )


            y_top -= box_h + row_gap
            idx += 1


        draw_page_number(c, page_num, width, margin)
        c.showPage()
        page_num += 1


    return page_num

def draw_refinement_summary_page(c, page_num, width, height, margin, heading, panel_map, note=None):
    """
    Layout:
      row 1: surface_views full width
      row 2: real_space_slices | per_particle_scale
      row 3: fsc | guinier | mask_slices
      row 4: viewing_direction | posterior_precision | protein_view_lookup
    """
    y = height - margin
    y = draw_heading(c, heading, margin, y, level="section", page_height=height, margin=margin)


    if note:
        note_font = "Helvetica-Oblique"
        note_size = 8
        note_leading = 11
        note_width = width - 2 * margin


        note_lines = simpleSplit(note, note_font, note_size, note_width)
        c.setFont(note_font, note_size)


        note_y = y - 2
        for line in note_lines:
            c.drawString(margin, note_y, line)
            note_y -= note_leading


        y = note_y - 6


    avail_w = width - 2 * margin
    avail_h = y - margin


    row_gap = 0.10 * inch
    col_gap = 0.10 * inch


    inner_h = avail_h - 3 * row_gap
    row_fracs = [0.36, 0.20, 0.22, 0.22]
    row_heights = [f * inner_h for f in row_fracs]


    def draw_panel(key, x, y_top, w, h):
        entry = panel_map.get(key)


        if isinstance(entry, dict):
            img = entry.get("image")
            title = entry.get("title") or _panel_title_for_key(key)
        else:
            img = entry
            title = _panel_title_for_key(key)


        draw_titled_panel_in_box(
            c,
            title,
            img,
            x,
            y_top,
            w,
            h,
            title_fontsize=9,
            min_title_band_h=0.24 * inch,
            title_gap=3.0,
            frame_pad=3.0,
        )


    y_top = y
    draw_panel("surface_views", margin, y_top, avail_w, row_heights[0])


    y_top -= row_heights[0] + row_gap
    cell_w2 = (avail_w - col_gap) / 2.0
    draw_panel("real_space_slices", margin, y_top, cell_w2, row_heights[1])
    draw_panel("per_particle_scale", margin + cell_w2 + col_gap, y_top, cell_w2, row_heights[1])


    y_top -= row_heights[1] + row_gap
    cell_w3 = (avail_w - 2 * col_gap) / 3.0
    draw_panel("fsc", margin, y_top, cell_w3, row_heights[2])
    draw_panel("guinier", margin + cell_w3 + col_gap, y_top, cell_w3, row_heights[2])
    draw_panel("mask_slices", margin + 2 * (cell_w3 + col_gap), y_top, cell_w3, row_heights[2])


    y_top -= row_heights[2] + row_gap
    draw_panel("viewing_direction", margin, y_top, cell_w3, row_heights[3])
    draw_panel("posterior_precision", margin + cell_w3 + col_gap, y_top, cell_w3, row_heights[3])
    draw_panel("protein_view_lookup", margin + 2 * (cell_w3 + col_gap), y_top, cell_w3, row_heights[3])


    draw_page_number(c, page_num, width, margin)
    c.showPage()
    return page_num + 1
