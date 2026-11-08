"""Command line entry point for coded_aperture_sim."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from . import model


def _check(scene):
    errors = []
    for module in ("numpy", "scipy", "matplotlib", "h5py", "openEMS", "CSXCAD"):
        try:
            __import__(module)
        except Exception as exc:
            errors.append(f"{module}: {exc}")
    from .paraview_view import find_pvpython
    pvpython = find_pvpython()
    if not pvpython:
        errors.append("ParaView pvpython executable was not found")
    try:
        from . import solver
        report = solver.preflight(scene)
    except Exception as exc:
        report = {"status": "failed", "error": str(exc)}
        errors.append(f"solver preflight: {exc}")
    print(json.dumps({"preflight": report, "pvpython": str(pvpython) if pvpython else None,
                      "import_errors": errors}, indent=2))
    return not errors and report.get("status", "ok") not in {"failed", "blocked"} and report.get("within_memory_limit", True)


def _new_run_path(parent):
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    for n in range(100):
        path = parent / (f"run_{stamp}" if n == 0 else f"run_{stamp}_{n}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise FileExistsError("could not reserve a unique run directory")


def _handoff_power(fields_path):
    import h5py
    import numpy as np

    with h5py.File(fields_path, "r") as h5:
        ix = int(h5.attrs["handoff_index"])
        x, y = np.asarray(h5["x_mm"], float), np.asarray(h5["y_mm"], float)
        e, h = h5["E"][ix], h5["H"][ix]
        if not np.isfinite(e).all() or not np.isfinite(h).all():
            raise ValueError("handoff fields contain non-finite values")
        sz = 0.5 * (e[..., 0] * np.conj(h[..., 1]) - e[..., 1] * np.conj(h[..., 0])).real
        flux = float(np.trapezoid(np.trapezoid(sz, y, axis=0), x, axis=0))
        return {"z_mm": float(h5["z_mm"][ix]), "integrated_signed_forward_flux_relative_mm2": flux,
                "finite_fields": True, "sample_count": int(sz.size),
                "units": "arbitrary source-spectrum-weighted field units times mm^2",
                "grid": {"nx": len(x), "ny": len(y), "x_mm": [float(x[0]), float(x[-1])],
                         "y_mm": [float(y[0]), float(y[-1])]}}


def run(scene, output_parent):
    """Reserve a unique run directory and retain partial output for diagnosis."""
    from . import solver
    from .export import export as export_run
    from .propagation import validate_handoff

    preflight = solver.preflight(scene)  # Must happen before allocating run output.
    allowed = preflight.get("resource_gate", preflight.get("within_memory_limit", False))
    if not allowed:
        raise MemoryError(
            "resource preflight blocked the run before output reservation: "
            f"estimate={preflight.get('estimated_peak_gib', 'unknown')} GiB, "
            f"configured_limit={scene['memory_limit_gib']} GiB, "
            f"currently_available={preflight.get('available_physical_memory_gib', 'unknown')} GiB"
        )
    final = _new_run_path(Path(output_parent))
    try:
        scene_path = final / "scene.json"
        scene_path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
        print(f"[run] preflight: {preflight['mesh_nodes']:,} nodes, est. peak {preflight['estimated_peak_gib']:.2f} GiB "
              f"(limit {scene['memory_limit_gib']}, available {preflight.get('available_physical_memory_gib')}); output: {final}", flush=True)
        t0 = time.perf_counter()
        print("[run] 1/3 pattern solve", flush=True)
        metadata = solver.solve(scene, final)
        t1 = time.perf_counter()
        reference_scene = dict(scene, mask_rows=["1111"] * 4)
        reference_dir = final / "reference"
        reference_dir.mkdir()
        print(f"[run] 2/3 all-open reference solve (pattern took {t1 - t0:.0f}s)", flush=True)
        ref_meta = solver.solve(reference_scene, reference_dir)
        t2 = time.perf_counter()
        print(f"[run] 3/3 validation + export (reference took {t2 - t1:.0f}s)", flush=True)
        timing = {"pattern_solve_s": t1 - t0, "reference_solve_s": t2 - t1}
        handoff = validate_handoff(final / "fields.h5", scene)
        ref_handoff = validate_handoff(reference_dir / "fields.h5", reference_scene)
        pattern_power = _handoff_power(final / "fields.h5")
        open_power = _handoff_power(reference_dir / "fields.h5")
        if pattern_power["z_mm"] != open_power["z_mm"] or pattern_power["grid"] != open_power["grid"]:
            raise ValueError("pattern and all-open reference do not share an identical handoff plane")
        ratio = (pattern_power["integrated_signed_forward_flux_relative_mm2"] /
                 open_power["integrated_signed_forward_flux_relative_mm2"]
                 if open_power["integrated_signed_forward_flux_relative_mm2"] != 0 else None)
        overall_status = "incomplete" if handoff["status"] == ref_handoff["status"] == "passed" else "failed"
        metadata = {"timing": timing, "solver": metadata, "all_open_reference": ref_meta,
                    "preflight": preflight, "validation": {
                        "handoff": handoff, "all_open_handoff": ref_handoff,
                        "common_plane_forward_flux": {"pattern": pattern_power, "all_open": open_power,
                                                      "relative_ratio": ratio},
                        "overall_status": overall_status,
                        "finite_capture": "not performed",
                        "mesh_refinement": "not performed"}}
        (final / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (final / "validation.json").write_text(json.dumps(metadata["validation"], indent=2) + "\n", encoding="utf-8")
        print(f"[run] handoff: pattern={handoff['status']}, all-open={ref_handoff['status']}, "
              f"flux ratio={ratio}; exporting volume...", flush=True)
        export_run(final)
        if overall_status == "failed":
            raise RuntimeError("handoff validation failed or is incomplete; see validation.json and exported view")
        print(f"timing: pattern {timing['pattern_solve_s']:.1f}s, reference {timing['reference_solve_s']:.1f}s, "
              f"total {time.perf_counter() - t0:.1f}s")
        print(final)
        return final
    except Exception as exc:
        (final / "failure.json").write_text(json.dumps({"error": str(exc), "run_dir": str(final)}, indent=2) + "\n", encoding="utf-8")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m coded_aperture_sim")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate scene and dependencies")
    check.add_argument("--config", type=Path)
    run_parser = sub.add_parser("run", help="run and export a fresh simulation")
    run_parser.add_argument("--config", type=Path)
    run_parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs")
    exp = sub.add_parser("export", help="rebuild visualizations from saved fields")
    exp.add_argument("run_dir", type=Path)
    smoke = sub.add_parser("smoke", help="run the explicitly requested tiny solver diagnostic")
    smoke.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "smoke")
    preview = sub.add_parser("preview", help="run an ideal scalar diffraction preview without openEMS")
    preview.add_argument("--config", type=Path, default=Path(__file__).with_name("scene_lowres.json"))
    preview.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "previews")
    preview.add_argument("--reflections", action="store_true", help="include one approximate reflected roundtrip")
    queue = sub.add_parser("preview-queue", help="run and resume a strip of ideal scalar previews")
    mode = queue.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--resume", type=Path)
    queue.add_argument("--output", type=Path)
    super_queue = sub.add_parser("preview-super-queue", help="run sequential preview queues from a saved super-queue config")
    super_mode = super_queue.add_mutually_exclusive_group(required=True)
    super_mode.add_argument("--config", type=Path)
    super_mode.add_argument("--resume", type=Path)
    super_queue.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            return 0 if _check(model.load_scene(args.config)) else 1
        if args.command == "run":
            scene = model.load_scene(args.config)
            run(scene, args.output)
        elif args.command == "export":
            from .export import export
            export(args.run_dir)
        elif args.command == "preview":
            from . import preview
            scene = model.load_scene(args.config)
            report = preview.preflight(scene)
            final = _new_run_path(args.output)
            print(f"[preview] preflight: {report['capture_shape_yx']} capture, {report['volume_shape_zyx']} volume; output: {final}", flush=True)
            preview.run(scene, final, reflections=args.reflections)
            print(final)
        elif args.command == "preview-queue":
            from .queue import run_queue
            return run_queue(config=args.config, output=args.output, resume=args.resume)
        elif args.command == "preview-super-queue":
            from .super_queue import run_super_queue
            return run_super_queue(config=args.config, output=args.output, resume=args.resume)
        else:
            from .solver import smoke_test
            args.output.mkdir(parents=True, exist_ok=False)
            print(json.dumps(smoke_test(args.output), indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
