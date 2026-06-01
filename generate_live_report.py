#!/usr/bin/env python3
# coding: utf-8

"""
Generate a PDF report for a CryoSPARC Live session.

README / local usage
--------------------

Expected layout:
    your_reports/
    ├── generate_live_report.py
    ├── cryosparc_live_report/
    │   ├── __init__.py
    │   ├── io.py
    │   ├── stats.py
    │   ├── images.py
    │   ├── plots.py
    │   └── pdf.py
    └── epu/
        ├── report_style.py
        └── report_utils.py

Run:
    python3 generate_live_report.py /path/to/CS-project
    python3 generate_live_report.py /path/to/CS-project --session S1
    python3 generate_live_report.py /path/to/CS-project --session S1 --class-job J67

Outputs:
    - Live_Imaging_Summary_<projectfolder>_<session>.pdf
    - live_stats_<session>.txt

Recommended dependencies:
    pip install reportlab pillow numpy mrcfile matplotlib pymongo cryosparc-tools
"""

import os
import sys
import argparse
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl_canvas
from PIL import Image, ImageDraw

_IMPORT_ERRORS = []

try:
    from cryosparc_live_report.io import (
        read_json,
        load_exposures_bson,
        find_live_workspace,
        find_latest_classavg_mrc,
        fmt_num,
    )
except Exception as e:
    _IMPORT_ERRORS.append(f"cryosparc_live_report.io import failed: {e}")

try:
    from cryosparc_live_report.stats import (
        parse_exposure,
        assign_exposure_numbers,
        select_accepted_ctf_tertiles,
        select_rejected_random,
        build_summary_sections,
        flatten_summary_sections,
        build_class2d_info_map,
    )
except Exception as e:
    _IMPORT_ERRORS.append(f"cryosparc_live_report.stats import failed: {e}")

try:
    from cryosparc_live_report.images import (
        make_classavg_montages,
        make_micrograph_panel,
        load_particle_multi_stack_montage,
        load_font,
    )
except Exception as e:
    _IMPORT_ERRORS.append(f"cryosparc_live_report.images import failed: {e}")

try:
    from cryosparc_live_report.plots import build_scatterplots
except Exception as e:
    _IMPORT_ERRORS.append(f"cryosparc_live_report.plots import failed: {e}")

try:
    from cryosparc_live_report.pdf import (
        add_summary_pages,
        draw_single_image_page,
        draw_five_plot_page,
        draw_panel_pages,
    )
except Exception as e:
    _IMPORT_ERRORS.append(f"cryosparc_live_report.pdf import failed: {e}")


def print_preflight_errors_and_exit():
    if not _IMPORT_ERRORS:
        return
    print("Error: required report modules could not be imported.\n")
    for msg in _IMPORT_ERRORS:
        print(f"- {msg}")
    print("\nMake sure:")
    print("  1. generate_live_report.py is next to the cryosparc_live_report/ package directory")
    print("  2. cryosparc_live_report/__init__.py exists")
    print("  3. dependencies are installed:")
    print("     pip install reportlab pillow numpy mrcfile matplotlib pymongo cryosparc-tools")
    sys.exit(2)


def choose_evenly_spaced(items, n):
    if not items:
        return []
    if len(items) <= n:
        return items
    idxs = np.linspace(0, len(items) - 1, n)
    return [items[int(round(i))] for i in idxs]


def build_report(project_dir: str, session_name: str = "S1") -> int:
    print_preflight_errors_and_exit()

    project_dir = os.path.abspath(project_dir)
    if not os.path.isdir(project_dir):
        print(f"Error: project directory not found: {project_dir}")
        return 2

    project_json = os.path.join(project_dir, "project.json")
    workspaces_json = os.path.join(project_dir, "workspaces.json")
    exposures_bson = os.path.join(project_dir, session_name, "exposures.bson")

    for p in (project_json, workspaces_json, exposures_bson):
        if not os.path.isfile(p):
            print(f"Error: missing required file: {p}")
            return 2

    try:
        project = read_json(project_json)
        workspaces = read_json(workspaces_json)
        ws = find_live_workspace(workspaces, session_name)
    except Exception as e:
        print(f"Error: failed to load project/workspace metadata: {e}")
        return 2

    params = ws.get("params", {})
    stats = ws.get("stats", {})
    bin_size_pix = int(params.get("bin_size_pix", 180) or 180)

    try:
        exposures = load_exposures_bson(exposures_bson)
    except Exception as e:
        print(f"Error: failed to read exposures.bson: {e}")
        print("Hint: install/use pymongo so bson.decode_file_iter is available.")
        return 2

    try:
        parsed = [parse_exposure(project_dir, session_name, exp, bin_size_pix) for exp in exposures]
        parsed = assign_exposure_numbers(parsed)
        
        start_times = [e.get("start_dt") for e in parsed if e.get("start_dt") is not None]
        if start_times:
            t0 = min(start_times)
            for e in parsed:
                dt = e.get("start_dt")
                if dt is not None:
                    e["elapsed_minutes"] = (dt - t0).total_seconds() / 60.0
                else:
                    e["elapsed_minutes"] = None

        ("elapsed_minutes", "Time Since Start (min)", "Time Since Start (min)", None, None, False),

    except Exception as e:
        print(f"Error: failed to parse exposure metadata: {e}")
        return 2

    try:
        class_job_uid = str(ws.get("phase2_class2D_job") or "").strip() or None

        if not class_job_uid:
            print("Warning: workspace does not define phase2_class2D_job; 2D class-average pages may be omitted.")

        elif not os.path.isdir(os.path.join(project_dir, class_job_uid)):
            print(f"Warning: workspace 2D class job directory not found: {class_job_uid}")
            class_job_uid = None

    except Exception as e:
        print(f"Error: failed while selecting workspace 2D class job: {e}")
        return 2

    ## CHANGE FORMATTING AS NEEDED ##
    
    MICROGRAPH_PANEL_LAYOUT = {
        "panel_w": 2100,
        "panel_h": 760,
        "margin": 10,
        "gap": 12,
        "title_h": 28,
        "footer_h": 38,
        "left_frac": 0.28,
        "top_panel_count": 4,
        "min_bottom_h": 150,
        "min_top_sq": 260,
    }

    MICROGRAPH_PANEL_STYLE = {
        "title_font_size": 19,
        "body_font_size": 14,
        "small_title_font_size": 11,
        "small_title_band_h": 20,
        "small_title_y_pad": 2,
        "title_color": (0, 0, 0),
        "body_color": (20, 20, 20),
        "border_color": (185, 185, 185),
        "bg_color": "white",
    }

    MICROGRAPH_PANEL_PLOTS = {
        "global_motion": {
            "title_fontsize": 8,
            "line_width": 1.2,
            "tick_labelsize": 6,
        },
        "local_motion": {
            "viewer_scale": 40.0,
            "patch_spacing_A": 380.0,
            "patch_size_A": 500.0,
            "traj_lw": 1.0,
            "title_fontsize": 8,
        },
        "local_defocus": {
            "display_grid": 180,
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

    ####

    classavg_render_kwargs = dict(
        invert=False,
        display_mode="auto",
        gamma=1.0,
    )

    classavg_imgs = []
    classavg_mrc = None
    if class_job_uid:
        class_job_dir = os.path.join(project_dir, class_job_uid)
        classavg_mrc = find_latest_classavg_mrc(class_job_dir, class_job_uid)
        if classavg_mrc:
            try:
                class_info_map = build_class2d_info_map(ws)
                class2d_info = ws.get("phase2_class2D_info") or []
                n_selected_classes = sum(1 for row in class2d_info if bool(row.get("selected")))
                force_black_borders = (n_selected_classes == 0) and bool(class2d_info)
                classavg_imgs = make_classavg_montages(
					mrc_path=classavg_mrc,
					class_info_map=class_info_map,
					cols=6,
					rows_per_page=9,
					tile_size=400,
					dpi=350,
					sort_by_count=True,
					count_key="num_particles_total",
					interpolation="none",
					force_black_borders=force_black_borders,
					**classavg_render_kwargs,
				)
            except Exception as e:
                print(f"Warning: failed to render class averages: {e}")

    accepted_tertiles = select_accepted_ctf_tertiles(parsed, n_each=4)
    rejected_sample = select_rejected_random(parsed, n=4, seed_str=f"{project_dir}:{session_name}:rejected")

    try:
        scatterplots = build_scatterplots(parsed, ws)
    except Exception as e:
        print(f"Warning: failed to render scatterplots: {e}")
        scatterplots = []

    accepted_panels = []
    label_map = {
        "best": "Accepted: best-third CTF",
        "middle": "Accepted: middle-third CTF",
        "worst": "Accepted: worst-third CTF",
    }
    for label in ("best", "middle", "worst"):
        for exp in accepted_tertiles[label]:
            try:
                accepted_panels.append(
                    make_micrograph_panel(
                        exp,
                        label_map[label],
                        fmt_num,
                        display_mode="percentile",
                        layout=MICROGRAPH_PANEL_LAYOUT,
                        style=MICROGRAPH_PANEL_STYLE,
                        plot_cfg=MICROGRAPH_PANEL_PLOTS,
                    )
                )

            except Exception as e:
                print(f"Warning: failed accepted panel for exposure {exp.get('uid')}: {e}")

    rejected_panels = []
    for exp in rejected_sample:
        try:
            rejected_panels.append(
                make_micrograph_panel(
                    exp,
                    "Rejected exposure",
                    fmt_num,
                    display_mode="percentile",
                    layout=MICROGRAPH_PANEL_LAYOUT,
                    style=MICROGRAPH_PANEL_STYLE,
                    plot_cfg=MICROGRAPH_PANEL_PLOTS,
                )
            )

        except Exception as e:
            print(f"Warning: failed rejected panel for exposure {exp.get('uid')}: {e}")

    particle_panels = []
    for label, display in [
        ("best", "Best-third accepted"),
        ("middle", "Middle-third accepted"),
        ("worst", "Worst-third accepted"),
    ]:
        exps = [e for e in accepted_tertiles[label] if e.get("particle_stack_path")]
        chosen_exps = choose_evenly_spaced(exps, 6)
        if not chosen_exps:
            continue

        try:
            stack_paths = [e["particle_stack_path"] for e in chosen_exps]
            row_labels = [
                (
                    f"Exp #{e.get('exposure_number','')}\n"
                    f"CTF = {fmt_num(e.get('ctf_fit_A'), 1)} Å | "
                    f"defocus = {fmt_num(e.get('defocus_um'), 1)} µm"
                )
                for e in chosen_exps
            ]

            montage = load_particle_multi_stack_montage(
                stack_paths=stack_paths,
                row_labels=row_labels,
                per_stack=6,
                max_stacks=6,
                tile_size=96,
                lowpass_A=25.0,
                target_display_angpix=3.0,
                soft_sigma=5.0,
                output_lo=50,
                output_hi=180,
                invert=False,
            )

            title_font = load_font(20, bold=False)

            if montage is not None:
                W, H = montage.size
                canvas = Image.new("RGB", (W, H + 34), "white")
                d = ImageDraw.Draw(canvas)
                d.text(
                    (8, 8),
                    f"{display} example particles | 6 exposures x 6 particles",
                    fill=(0, 0, 0),
                    font=title_font,
                )
                canvas.paste(montage, (0, 34))
                particle_panels.append(canvas)

        except Exception as e:
            print(f"Warning: failed multi-stack particle montage for tertile {label}: {e}")

    sections = build_summary_sections(project, ws, parsed, class_job_uid)
    rows = flatten_summary_sections(sections)

    project_folder = os.path.basename(project_dir.rstrip(os.sep))
    pdf_name = f"Live_Imaging_Summary_{project_folder}_{session_name}.pdf"
    pdf_path = os.path.join(project_dir, pdf_name)

    try:
        c = rl_canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        margin = 36
        page_num = 1

        page_num = add_summary_pages(
            c, width, height, margin, project_folder, sections,
            page_num_start=page_num,
        )

        if scatterplots:
            scatter_note = (
                "Scatterplots include threshold lines only when min/max limits are present in workspace attributes."
            )
            page_num = draw_five_plot_page(c, page_num, width, height, margin, "Session Scatterplots", scatterplots, scatter_note)

        if accepted_panels:
            accepted_note = (
                "Representative accepted micrographs are chosen only from accepted exposures and split into best, middle, and worst thirds by CTF fit. "
                "Pick overlays and extracted-particle examples use the picker active for each exposure (blob or template)."
            )
            page_num = draw_panel_pages(c, page_num, width, height, margin, "Representative Accepted Micrographs", accepted_panels, note=accepted_note)

        if rejected_panels:
            rejected_note = (
                "Rejected micrographs are shown separately as a deterministic random sample of up to 5 rejected exposures."
            )
            page_num = draw_panel_pages(c, page_num, width, height, margin, "Representative Rejected Micrographs", rejected_panels, note=rejected_note)

        if particle_panels:
            particle_note = (
                "Particles are lowpass filtered (25 Å), smoothed, and contrast-modulated "
                "for ease of viewing; color is inverted (lighter particles on darker background)"
            )
            page_num = draw_panel_pages(
                c,
                page_num,
                width,
                height,
                margin,
                "Example Extracted Particles",
                particle_panels,
                note=particle_note,
            )

        for i, classavg_img in enumerate(classavg_imgs, start=1):
            heading = f"2D Class Averages ({class_job_uid})"
            if len(classavg_imgs) > 1:
                heading += f" {i}/{len(classavg_imgs)}"
            if class_job_uid:
                class_note = (
                    f"The 2D class-average pages use the latest 2D job queued in the live workspace."
                )
            else:
                class_note = (
                    "No completed class_2D_new job was found automatically; the 2D class-average section may be omitted."
                )
            page_num = draw_single_image_page(c, page_num, width, height, margin, heading, classavg_img, note=class_note)

        c.save()
        
    except Exception as e:
        print(f"Error: failed while writing PDF: {e}")
        return 2
        
    print(f"Wrote: {pdf_name}")
    if classavg_mrc:
        print(f"Class-average job: {class_job_uid}")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Generate a CryoSPARC Live session PDF report.")
    parser.add_argument("project_dir", help="Path to CryoSPARC project directory")
    parser.add_argument("--session", default="S1", help="Session dir/uid (default: S1)")
    args = parser.parse_args()

    rc = build_report(
        args.project_dir,
        session_name=args.session,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()

