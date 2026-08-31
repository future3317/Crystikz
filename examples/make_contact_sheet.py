"""Build a simple contact sheet from the gallery PNG outputs."""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

GALLERY_DIR = Path(__file__).parent.parent / "gallery"
OUTPUT = GALLERY_DIR / "contact_sheet.png"

PANELS = [
    ("01_rocksalt_ballstick.png", "NaCl rock-salt"),
    ("02_diamond_tetrahedral.png", "Diamond Si"),
    ("03_perovskite_octahedron.png", "BaTiO3 TiO6"),
    ("04_rutile_distorted.png", "Rutile TiO2"),
    ("05_wurtzite_hexagonal.png", "Wurtzite ZnO"),
    ("06_mos2_layered.png", "MoS2 layered"),
]


def main():
    images = []
    labels = []
    for fname, label in PANELS:
        path = GALLERY_DIR / fname
        if path.exists():
            images.append(Image.open(path).convert("RGBA"))
            labels.append(label)

    if not images:
        print("No gallery images found.")
        return

    # Normalize heights while preserving aspect ratios.
    target_height = 600
    resized = []
    for img in images:
        aspect = img.width / img.height
        w = int(target_height * aspect)
        resized.append(img.resize((w, target_height), Image.LANCZOS))

    cols = 3
    rows = (len(resized) + cols - 1) // cols
    pad = 30
    label_height = 50

    row_widths = []
    row_heights = []
    for r in range(rows):
        row_imgs = resized[r * cols:(r + 1) * cols]
        row_widths.append(sum(img.width for img in row_imgs) + pad * (len(row_imgs) - 1))
        row_heights.append(max(img.height for img in row_imgs) + label_height)

    total_width = max(row_widths) + pad * 2
    total_height = sum(row_heights) + pad * (rows + 1)

    sheet = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 28)
        except Exception:
            font = ImageFont.load_default()
    except Exception:
        draw = None
        font = None

    y = pad
    for r in range(rows):
        row_imgs = resized[r * cols:(r + 1) * cols]
        row_labels = labels[r * cols:(r + 1) * cols]
        row_w = sum(img.width for img in row_imgs) + pad * (len(row_imgs) - 1)
        x = (total_width - row_w) // 2
        for img, label in zip(row_imgs, row_labels, strict=False):
            sheet.paste(img, (x, y), img)
            if draw and font:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_x = x + (img.width - text_w) // 2
                draw.text((text_x, y + img.height + 8), label, fill=(40, 44, 52, 255), font=font)
            x += img.width + pad
        y += row_heights[r] + pad

    sheet.save(OUTPUT)
    print(f"Contact sheet saved to {OUTPUT}")


if __name__ == "__main__":
    main()
