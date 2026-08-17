#!/usr/bin/env python3
"""Create upload-ready JPEG images from the formatted source-code PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_jpeg_pages(render_dir: Path, prefix: str) -> list[Path]:
    pages = sorted(render_dir.glob(f"{prefix}-*.jpg"))
    if not pages:
        raise FileNotFoundError(f"No rendered JPEG pages found in {render_dir}")
    return pages


def combine_vertical(page_paths: list[Path], output_path: Path, gap: int = 36) -> None:
    images = [Image.open(path).convert("RGB") for path in page_paths]
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)

    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        x = (width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=92, optimize=True)


def make_contact_sheet(page_paths: list[Path], output_path: Path, columns: int = 2) -> None:
    thumbs: list[Image.Image] = []
    labels: list[str] = []
    for index, path in enumerate(page_paths, start=1):
        image = Image.open(path).convert("RGB")
        image.thumbnail((900, 700), Image.Resampling.LANCZOS)
        thumbs.append(image)
        labels.append(f"Page {index}")

    margin = 40
    gap = 28
    label_height = 34
    rows = (len(thumbs) + columns - 1) // columns
    cell_width = max(image.width for image in thumbs)
    cell_height = max(image.height for image in thumbs) + label_height
    width = margin * 2 + columns * cell_width + (columns - 1) * gap
    height = margin * 2 + rows * cell_height + (rows - 1) * gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for index, image in enumerate(thumbs):
        row = index // columns
        column = index % columns
        x = margin + column * (cell_width + gap) + (cell_width - image.width) // 2
        y = margin + row * (cell_height + gap)
        draw.text((margin + column * (cell_width + gap), y), labels[index], fill=(45, 55, 60), font=font)
        canvas.paste(image, (x, y + label_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=92, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine rendered PDF pages into upload-ready JPEGs.")
    parser.add_argument("--render-dir", type=Path, default=Path("tmp/pdfs"))
    parser.add_argument("--prefix", default="tone_capture_code")
    parser.add_argument(
        "--all-pages-output",
        type=Path,
        default=Path("output/jpeg/Guitar_Bass_Tone_Capture_Engine_Source_Code_ALL_PAGES.jpg"),
    )
    parser.add_argument(
        "--contact-sheet-output",
        type=Path,
        default=Path("output/jpeg/Guitar_Bass_Tone_Capture_Engine_Source_Code_CONTACT_SHEET.jpg"),
    )
    args = parser.parse_args()

    pages = load_jpeg_pages(args.render_dir, args.prefix)
    combine_vertical(pages, args.all_pages_output)
    make_contact_sheet(pages, args.contact_sheet_output)

    print(f"Wrote all-pages JPEG: {args.all_pages_output}")
    print(f"Wrote contact sheet JPEG: {args.contact_sheet_output}")


if __name__ == "__main__":
    main()
