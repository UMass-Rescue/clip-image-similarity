from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_rgb_image(path: Path) -> Image.Image:
    """Open an image with tolerant Pillow decoding and return an RGB copy."""
    with Image.open(path) as img:
        return img.convert("RGB")
