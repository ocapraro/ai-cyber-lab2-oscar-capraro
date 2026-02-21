from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    """Create a directory tree when it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Write a dictionary to disk as formatted JSON."""
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def load_json(input_path: Path) -> dict[str, Any]:
    """Read JSON file from disk."""
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


def write_png_rgb(image: list[list[tuple[int, int, int]]], output_path: Path) -> None:
    """Write a small RGB PNG image using only standard library."""
    height = len(image)
    width = len(image[0]) if height else 0
    if height == 0 or width == 0:
        raise ValueError("Cannot write empty image.")

    for row in image:
        if len(row) != width:
            raise ValueError("Inconsistent row widths in image data.")

    raw = bytearray()
    for row in image:
        raw.append(0)  # filter type 0
        for r, g, b in row:
            raw.extend((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=9)

    png_bytes = bytearray(b"\x89PNG\r\n\x1a\n")
    png_bytes.extend(_png_chunk(b"IHDR", ihdr))
    png_bytes.extend(_png_chunk(b"IDAT", idat))
    png_bytes.extend(_png_chunk(b"IEND", b""))

    ensure_dir(output_path.parent)
    with output_path.open("wb") as handle:
        handle.write(png_bytes)
