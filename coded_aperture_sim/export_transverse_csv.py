"""Export transverse slice CSVs from one saved run or every run under a folder."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.pop(0)  # Avoid shadowing the standard-library queue module with coded_aperture_sim/queue.py.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from coded_aperture_sim.preview import export_transverse_csvs
else:
    from .preview import export_transverse_csvs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="saved run directory or parent folder")
    args = parser.parse_args(argv)
    if not args.path.is_dir():
        parser.error(f"not a directory: {args.path}")
    volumes = [args.path / "volume.h5"] if (args.path / "volume.h5").is_file() else sorted(args.path.rglob("volume.h5"))
    if not volumes:
        parser.error(f"no volume.h5 found under {args.path}")
    incompatible = 0
    for volume in volumes:
        result = export_transverse_csvs(volume.parent)
        print(f"{volume.parent}: {result}")
        incompatible += result.startswith("incompatible:")
    print(f"Scanned {len(volumes)} run(s); {incompatible} incompatible.")
    return int(incompatible > 0)


if __name__ == "__main__":
    raise SystemExit(main())
