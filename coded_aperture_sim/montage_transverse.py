"""Assemble the 24 saved normalized transverse PNG maps from a strip queue."""

import argparse
import json
from pathlib import Path, PureWindowsPath
import sys

if __package__ in (None, ""):
    sys.path.pop(0)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from scipy import ndimage


def _map_crop(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
    pixels = np.asarray(image)
    # shortcut: RGB spread plus largest component fits current viridis map PNGs; upgrade if plot styling changes.
    colored = pixels.max(axis=2).astype(int) - pixels.min(axis=2).astype(int) > 10
    components, count = ndimage.label(colored, structure=np.ones((3, 3), dtype=int))
    if not count:
        raise ValueError(f"no colored map found in {path}")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    y, x = np.where(components == sizes.argmax())
    return image.crop((x.min(), y.min(), x.max() + 1, y.max() + 1))


def create_montage(queue_dir):
    queue_dir = Path(queue_dir)
    manifest = json.loads((queue_dir / "queue.json").read_text(encoding="utf-8"))
    jobs = manifest.get("jobs")
    if (not isinstance(jobs, list) or len(jobs) != 24 or
            any(not isinstance(job, dict) or type(job.get("window")) is not int for job in jobs) or
            sorted(job["window"] for job in jobs) != list(range(24))):
        raise ValueError("queue.json must contain exactly one job for each window 0 through 23")
    if any(job.get("status") != "completed" or not isinstance(job.get("output_path"), str) for job in jobs):
        raise ValueError("all 24 queue jobs must be completed")

    maps = []
    for job in sorted(jobs, key=lambda item: item["window"]):
        run_name = PureWindowsPath(job["output_path"]).name
        run_png = queue_dir / run_name / "transverse_1250_normalized.png"
        maps.append(_map_crop(run_png))

    size = maps[0].size
    if any(image.size != size for image in maps):
        raise ValueError("cropped map dimensions do not match")
    gutter = 12
    montage = Image.new("RGB", (4 * size[0] + 5 * gutter, 6 * size[1] + 7 * gutter), "white")
    for index, image in enumerate(maps):
        x = gutter + (index % 4) * (size[0] + gutter)
        y = gutter + (index // 4) * (size[1] + gutter)
        montage.paste(image, (x, y))
    output = queue_dir / "transverse_1250_montage.png"
    montage.save(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_dir", type=Path, help="queue folder containing queue.json and map PNGs")
    args = parser.parse_args(argv)
    if not args.queue_dir.is_dir():
        parser.error(f"not a directory: {args.queue_dir}")
    print(create_montage(args.queue_dir))


if __name__ == "__main__":
    main()
