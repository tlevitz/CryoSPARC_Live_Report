#!/usr/bin/env python3
# coding: utf-8

from typing import List, Tuple, Optional

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle, Spacer, Frame
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader, simpleSplit

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

def render_framed_pil(c, pil_img, x_left, y_top, max_w, max_h, frame_pad=4.0):
    iw, ih = pil_img.size
    scale = min((max_w - 2 * frame_pad) / max(iw, 1), (max_h - 2 * frame_pad) / max(ih, 1))
    dw, dh = iw * scale, ih * scale
    total_h = dh + 2 * frame_pad
    draw_frame_box(c, x_left, y_top, max_w, total_h)
    x_img = x_left + frame_pad + (max_w - 2 * frame_pad - dw) / 2.0
    y_img = y_top - frame_pad - dh
    c.drawImage(ImageReader(pil_img), x_img, y_img, width=dw, height=dh, preserveAspectRatio=True, mask="auto")
    return total_h

def render_framed_pil_in_box(c, pil_img, x_left, y_top, box_w, box_h, frame_pad=4.0):
    """
    Draw a PIL image centered inside a fixed-size framed box.
    """
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

    c.drawImage(
        ImageReader(pil_img),
        x_img,
        y_img,
        width=dw,
        height=dh,
        preserveAspectRatio=True,
        mask="auto",
    )
    return box_h


def draw_missing_panel(c, x_left, y_top, box_w, box_h, label="Not available"):
    draw_frame_box(c, x_left, y_top, box_w, box_h)
    c.setFont(RL_FONT_FAMILY, 9)
    c.setFillColor(colors.grey)
    c.drawCentredString(x_left + box_w / 2.0, y_top - box_h / 2.0, label)
    c.setFillColor(colors.black)



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
    notes_title_style = ParagraphStyle("NotesTitle", parent=body_style, fontName=RL_FONT_FAMILY_BOLD, spaceAfter=6)
    bullet_style = ParagraphStyle("Bullet", parent=body_style, leftIndent=12, bulletIndent=0)

    story = []
    story.append(Paragraph(f"CryoSPARC Live Summary: {project_folder_name}", title_style))
    story.append(Spacer(1, 0.08 * inch))

    for sec in sections:
        story.append(Paragraph(sec["title"], section_style))
        if sec.get("summary_html"):
            story.append(Paragraph(sec["summary_html"], body_style))
            story.append(Spacer(1, 0.06 * inch))

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
            story.append(table)
            story.append(Spacer(1, 0.10 * inch))

    while story:
        frame = Frame(margin, margin, width - 2 * margin, height - 2 * margin, showBoundary=0)
        remaining = list(story)
        frame.addFromList(remaining, c)
        draw_page_number(c, page_num, width, margin)
        c.showPage()
        page_num += 1
        if len(remaining) == len(story):
            # safety: prevent infinite loop if one element is too large
            break
        story = remaining

    return page_num

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
    Draw full-width panels, exactly 4 per page.
    """
    if not panels:
        return page_num

    idx = 0
    cols = 1
    rows = 4
    gap_x = 0.0
    gap_y = 0.12 * inch

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

        cell_w = avail_w
        cell_h = (avail_h - (rows - 1) * gap_y) / rows

        for r in range(rows):
            if idx >= len(panels):
                break
            x = margin
            y_top = y - r * (cell_h + gap_y)
            render_framed_pil(c, panels[idx], x, y_top, cell_w, cell_h, frame_pad=3.0)
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

    # Fractions of the usable page height after subtracting row gaps
    inner_h = avail_h - 3 * row_gap
    row_fracs = [0.36, 0.20, 0.22, 0.22]
    row_heights = [f * inner_h for f in row_fracs]

    def draw_panel(key, x, y_top, w, h):
        img = panel_map.get(key)
        if img is None:
            draw_missing_panel(c, x, y_top, w, h, label=key.replace("_", " "))
        else:
            render_framed_pil_in_box(c, img, x, y_top, w, h, frame_pad=3.0)

    # Row 1
    y_top = y
    draw_panel("surface_views", margin, y_top, avail_w, row_heights[0])

    # Row 2
    y_top -= row_heights[0] + row_gap
    cell_w2 = (avail_w - col_gap) / 2.0
    draw_panel("real_space_slices", margin, y_top, cell_w2, row_heights[1])
    draw_panel("per_particle_scale", margin + cell_w2 + col_gap, y_top, cell_w2, row_heights[1])

    # Row 3
    y_top -= row_heights[1] + row_gap
    cell_w3 = (avail_w - 2 * col_gap) / 3.0
    draw_panel("fsc", margin, y_top, cell_w3, row_heights[2])
    draw_panel("guinier", margin + cell_w3 + col_gap, y_top, cell_w3, row_heights[2])
    draw_panel("mask_slices", margin + 2 * (cell_w3 + col_gap), y_top, cell_w3, row_heights[2])

    # Row 4
    y_top -= row_heights[2] + row_gap
    draw_panel("viewing_direction", margin, y_top, cell_w3, row_heights[3])
    draw_panel("posterior_precision", margin + cell_w3 + col_gap, y_top, cell_w3, row_heights[3])
    draw_panel("protein_view_lookup", margin + 2 * (cell_w3 + col_gap), y_top, cell_w3, row_heights[3])

    draw_page_number(c, page_num, width, margin)
    c.showPage()
    return page_num + 1




