#!/usr/bin/env python3
"""Export the guitar/bass tone capture source file as a readable PDF."""

from __future__ import annotations

import argparse
import textwrap
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PAGE_SIZE = landscape(letter)
MARGIN_X = 0.45 * inch
MARGIN_Y = 0.42 * inch
HEADER_HEIGHT = 0.36 * inch
FOOTER_HEIGHT = 0.28 * inch
CODE_FONT = "Courier"
CODE_FONT_SIZE = 7.25
LINE_HEIGHT = 9.05
LINE_NUMBER_WIDTH = 0.42 * inch


def wrapped_code_lines(lines: list[str], available_width: float) -> list[tuple[int | None, str]]:
    char_width = stringWidth("M", CODE_FONT, CODE_FONT_SIZE)
    max_chars = max(40, int((available_width - LINE_NUMBER_WIDTH) / char_width))
    wrapped: list[tuple[int | None, str]] = []

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\n").replace("\t", "    ")
        if not line:
            wrapped.append((line_number, ""))
            continue

        chunks = textwrap.wrap(
            line,
            width=max_chars,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )

        for index, chunk in enumerate(chunks):
            wrapped.append((line_number if index == 0 else None, chunk))

    return wrapped


def draw_title_page(pdf: canvas.Canvas, source_path: Path, output_path: Path) -> None:
    width, height = PAGE_SIZE
    pdf.setFillColor(colors.HexColor("#121316"))
    pdf.rect(0, 0, width, height, fill=True, stroke=False)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(MARGIN_X, height - 1.28 * inch, "Guitar/Bass Amp Tone Capture Engine")

    pdf.setFont("Helvetica", 13)
    pdf.setFillColor(colors.HexColor("#ECE4DA"))
    pdf.drawString(MARGIN_X, height - 1.66 * inch, "Source code PDF for audio DSP portfolio upload")

    pdf.setStrokeColor(colors.HexColor("#D98A3A"))
    pdf.setLineWidth(2)
    pdf.line(MARGIN_X, height - 2.02 * inch, width - MARGIN_X, height - 2.02 * inch)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(MARGIN_X, height - 2.58 * inch, "Included File")

    pdf.setFont("Courier", 11)
    pdf.setFillColor(colors.HexColor("#FFF0DF"))
    pdf.drawString(MARGIN_X, height - 2.86 * inch, source_path.name)

    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.white)
    pdf.drawString(MARGIN_X, height - 3.38 * inch, "Project")

    pdf.setFont("Helvetica", 11)
    pdf.setFillColor(colors.HexColor("#FFF0DF"))
    pdf.drawString(MARGIN_X, height - 3.66 * inch, "Dynamic nonlinear guitar/bass tone capture with cabinet profile storage")

    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.white)
    pdf.drawString(MARGIN_X, height - 4.18 * inch, "Generated")

    pdf.setFont("Helvetica", 11)
    pdf.setFillColor(colors.HexColor("#FFF0DF"))
    pdf.drawString(MARGIN_X, height - 4.46 * inch, date.today().isoformat())

    pdf.setFont("Helvetica", 8.8)
    pdf.setFillColor(colors.HexColor("#C9B8A4"))
    pdf.drawRightString(width - MARGIN_X, MARGIN_Y, str(output_path))
    pdf.showPage()


def draw_code_header(pdf: canvas.Canvas, page_number: int, source_name: str) -> None:
    width, height = PAGE_SIZE
    pdf.setFillColor(colors.HexColor("#F8F4EF"))
    pdf.rect(0, height - HEADER_HEIGHT - 0.12 * inch, width, HEADER_HEIGHT + 0.12 * inch, fill=True, stroke=False)

    pdf.setFillColor(colors.HexColor("#252423"))
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(MARGIN_X, height - 0.34 * inch, source_name)

    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(colors.HexColor("#625A52"))
    pdf.drawRightString(width - MARGIN_X, height - 0.34 * inch, f"Page {page_number}")


def draw_code_footer(pdf: canvas.Canvas) -> None:
    width, _ = PAGE_SIZE
    pdf.setStrokeColor(colors.HexColor("#E2D6C9"))
    pdf.setLineWidth(0.5)
    pdf.line(MARGIN_X, FOOTER_HEIGHT, width - MARGIN_X, FOOTER_HEIGHT)

    pdf.setFillColor(colors.HexColor("#736B63"))
    pdf.setFont("Helvetica", 7.5)
    pdf.drawString(MARGIN_X, 0.16 * inch, "Generated from tone_capture_engine.py")


def export_code_pdf(source_path: Path, output_path: Path) -> None:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = PAGE_SIZE
    code_left = MARGIN_X
    code_right = width - MARGIN_X
    code_top = height - HEADER_HEIGHT - 0.18 * inch
    code_bottom = FOOTER_HEIGHT + 0.12 * inch
    available_width = code_right - code_left
    code_rows = wrapped_code_lines(lines, available_width)

    pdf = canvas.Canvas(str(output_path), pagesize=PAGE_SIZE)
    pdf.setTitle("Guitar/Bass Amp Tone Capture Engine Source Code")
    pdf.setAuthor("Codex")
    pdf.setSubject("Python source code for dynamic nonlinear guitar and bass tone capture")

    draw_title_page(pdf, source_path, output_path)

    page_number = 2
    y = code_top
    draw_code_header(pdf, page_number, source_path.name)

    for line_number, code in code_rows:
        if y < code_bottom:
            draw_code_footer(pdf)
            pdf.showPage()
            page_number += 1
            y = code_top
            draw_code_header(pdf, page_number, source_path.name)

        if line_number is not None:
            if line_number % 2 == 0:
                pdf.setFillColor(colors.HexColor("#FFFDFB"))
                pdf.rect(code_left - 0.05 * inch, y - 2.0, available_width + 0.1 * inch, LINE_HEIGHT, fill=True, stroke=False)

            pdf.setFillColor(colors.HexColor("#7B746D"))
            pdf.setFont(CODE_FONT, CODE_FONT_SIZE)
            pdf.drawRightString(code_left + LINE_NUMBER_WIDTH - 0.08 * inch, y, str(line_number))
        else:
            pdf.setFillColor(colors.HexColor("#AAA199"))
            pdf.setFont(CODE_FONT, CODE_FONT_SIZE)
            pdf.drawRightString(code_left + LINE_NUMBER_WIDTH - 0.08 * inch, y, ">")

        pdf.setFillColor(colors.HexColor("#111111"))
        pdf.setFont(CODE_FONT, CODE_FONT_SIZE)
        pdf.drawString(code_left + LINE_NUMBER_WIDTH, y, code)
        y -= LINE_HEIGHT

    draw_code_footer(pdf)
    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Python source code to a formatted PDF.")
    parser.add_argument("--source", type=Path, default=Path("tone_capture_engine.py"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/Guitar_Bass_Tone_Capture_Engine_Source_Code.pdf"),
    )
    args = parser.parse_args()

    export_code_pdf(args.source, args.output)
    print(f"Wrote PDF: {args.output}")


if __name__ == "__main__":
    main()
