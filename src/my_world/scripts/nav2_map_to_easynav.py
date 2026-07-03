#!/usr/bin/env python3
"""Convert a Nav2 occupancy map YAML into an EasyNav SimpleMap file."""

import argparse
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an EasyNav SimpleMap .map file from a Nav2 map YAML "
            "and its referenced image."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input Nav2 map YAML file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output EasyNav .map file.",
    )
    parser.add_argument(
        "--unknown-value",
        choices=("free", "occupied", "unknown"),
        default="free",
        help=(
            "Value written when a pixel is between the free and occupied "
            "thresholds. The default reproduces the current EasyNav map."
        ),
    )
    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        help=(
            "Do not invert the image rows. By default the script flips Y to "
            "match the map origin convention used by the current EasyNav map."
        ),
    )
    return parser.parse_args()


def load_metadata(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)

    required_fields = [
        "image",
        "resolution",
        "origin",
        "occupied_thresh",
        "free_thresh",
    ]
    missing_fields = [field for field in required_fields if field not in metadata]
    if missing_fields:
        raise ValueError(
            f"Missing required field(s) in {yaml_path}: {', '.join(missing_fields)}"
        )

    return metadata


def occupancy_value(
    pixel_value: int,
    negate: bool,
    occupied_threshold: float,
    free_threshold: float,
    unknown_value: str,
) -> str:
    normalized_pixel = pixel_value / 255.0
    occupancy = normalized_pixel if negate else 1.0 - normalized_pixel

    if occupancy > occupied_threshold:
        return "1"
    if occupancy < free_threshold:
        return "0"
    if unknown_value == "occupied":
        return "1"
    if unknown_value == "unknown":
        return "-1"
    return "0"


def iter_rows(height: int, flip_y: bool) -> Iterable[int]:
    if flip_y:
        return range(height - 1, -1, -1)
    return range(height)


def convert_map(
    input_yaml: Path,
    output_map: Path,
    unknown_value: str,
    flip_y: bool,
) -> None:
    metadata = load_metadata(input_yaml)
    image_path = Path(metadata["image"])
    if not image_path.is_absolute():
        image_path = input_yaml.parent / image_path

    image = Image.open(image_path).convert("L")
    width, height = image.size
    origin = metadata["origin"]

    output_map.parent.mkdir(parents=True, exist_ok=True)
    with output_map.open("w", encoding="utf-8") as stream:
        stream.write(
            f"{width} {height} {metadata['resolution']} {origin[0]} {origin[1]}\n"
        )

        cells = []
        for y in iter_rows(height, flip_y):
            cells.extend(
                occupancy_value(
                    image.getpixel((x, y)),
                    bool(metadata.get("negate", 0)),
                    float(metadata["occupied_thresh"]),
                    float(metadata["free_thresh"]),
                    unknown_value,
                )
                for x in range(width)
            )
        stream.write(" ".join(cells))
        stream.write("\n")


def main() -> None:
    args = parse_args()
    convert_map(
        input_yaml=args.input,
        output_map=args.output,
        unknown_value=args.unknown_value,
        flip_y=not args.no_flip_y,
    )
    print(f"EasyNav map written to: {args.output}")


if __name__ == "__main__":
    main()
