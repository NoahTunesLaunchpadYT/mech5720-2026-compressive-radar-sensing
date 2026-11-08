"""Ideal scalar, two-screen diffraction preview; this does not run openEMS."""

import csv
import json
import tempfile
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import model, solver
from .paraview_view import find_pvpython
from .propagation import _prepare_plane, _propagate_prepared, propagate_plane


LABEL = "IDEAL SCALAR DIFFRACTION — NOT FULL-WAVE VALIDATED"
DEFAULT_DEPTHS_MM = [50., 250., 500.]


def _depths(scene):
    depths = scene.get("preview_depths_mm", DEFAULT_DEPTHS_MM)
    if (not isinstance(depths, list) or not depths or
            any(isinstance(d, bool) or not isinstance(d, (int, float)) or not np.isfinite(d) or d <= 0 or
                d > scene["volume_size_mm"][2] for d in depths) or len(set(depths)) != len(depths)):
        raise ValueError("preview_depths_mm must contain unique positive finite depths within the volume z extent")
    return tuple(depths)


def _count(size, step, name):
    cells = size / step
    if not np.isfinite(cells) or cells < 1 or not np.isclose(cells, round(cells), atol=1e-9, rtol=0):
        raise ValueError(f"{name} must be an integer multiple of its sampling step")
    return int(round(cells))


def _grid(size, step, name):
    n = _count(size, step, name)
    return (np.arange(n) - (n - 1) / 2) * step


def _screen_mask(x, y, scene, first=False):
    xx, yy = np.meshgrid(x, y)
    if first:
        return ((xx >= -scene["opening_size_mm"][0] / 2) & (xx < scene["opening_size_mm"][0] / 2) &
                (yy >= -scene["opening_size_mm"][1] / 2) & (yy < scene["opening_size_mm"][1] / 2))
    col = np.floor((xx + 2 * scene["pitch_mm"]) / scene["pitch_mm"]).astype(int)
    row = 3 - np.floor((yy + 2 * scene["pitch_mm"]) / scene["pitch_mm"]).astype(int)
    inside = (col >= 0) & (col < 4) & (row >= 0) & (row < 4)
    mask = np.zeros(xx.shape, dtype=bool)
    for r in range(4):
        for c in range(4):
            mask |= inside & (row == r) & (col == c) & (scene["mask_rows"][r][c] == "1")
    return mask


def _metal_mask(x, y, scene, prefix):
    """Project the finite metal faces named by prefix onto the capture plane."""
    xx, yy = np.meshgrid(x, y)
    metal = np.zeros(xx.shape, dtype=bool)
    for box in model.geometry(scene):
        if box["name"].startswith(prefix + "_"):
            x0, y0 = box["start"][:2]
            x1, y1 = box["stop"][:2]
            metal |= (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    return metal


def preflight(scene):
    """Validate preview sampling and bound the principal output arrays before reservation."""
    if not np.isfinite([scene["frequency_ghz"], scene["capture_step_mm"], scene["volume_step_mm"]]).all():
        raise ValueError("frequency and sampling steps must be finite")
    if scene["frequency_ghz"] <= 0 or scene["capture_step_mm"] <= 0 or scene["volume_step_mm"] <= 0:
        raise ValueError("frequency and sampling steps must be positive")
    _depths(scene)
    wavelength_mm = 299.792458 / scene["frequency_ghz"]
    if scene["capture_step_mm"] > wavelength_mm/2:
        raise ValueError("capture_step_mm must be at most half a wavelength for the forward hemisphere")
    cnx = _count(float(scene["capture_size_mm"][0]), float(scene["capture_step_mm"]), "capture width")
    cny = _count(float(scene["capture_size_mm"][1]), float(scene["capture_step_mm"]), "capture height")
    dims = scene["volume_size_mm"]
    counts = [_count(float(d), float(scene["volume_step_mm"]), f"volume axis {i}") for i, d in enumerate(dims)]
    if min(cnx, cny) < 2:
        raise ValueError("capture plane needs at least two samples per axis")
    needed_x = max(scene["opening_size_mm"][0], 4*scene["pitch_mm"])
    needed_y = max(scene["opening_size_mm"][1], 4*scene["pitch_mm"])
    cap_x0, cap_x1 = -scene["capture_size_mm"][0]/2+scene["capture_step_mm"]/2, scene["capture_size_mm"][0]/2-scene["capture_step_mm"]/2
    cap_y0, cap_y1 = -scene["capture_size_mm"][1]/2+scene["capture_step_mm"]/2, scene["capture_size_mm"][1]/2-scene["capture_step_mm"]/2
    if cap_x0 > -needed_x/2 or cap_x1 < needed_x/2 or cap_y0 > -needed_y/2 or cap_y1 < needed_y/2:
        raise ValueError("capture plane must cover the full opening and active mask grid")
    pad = scene["padding"]
    padded_base = (cnx*pad)*(cny*pad)
    padded_variant = (2*cnx*pad)*(2*cny*pad)
    # Keep the base spectrum cached while building a variant; 256 B/cell bounds FFT temporaries.
    estimated_peak = (padded_base+padded_variant)*256
    memory = solver._available_physical_memory()
    available = None if memory is None else memory[1]/1024**3
    if estimated_peak > scene["memory_limit_gib"]*1024**3:
        raise MemoryError("preview FFT workspace exceeds memory_limit_gib")
    # if available is not None and estimated_peak > max(0, available-1)*1024**3:
    #     raise MemoryError("preview FFT workspace exceeds available RAM after a 1 GiB reserve")
    if find_pvpython() is None:
        raise RuntimeError("ParaView pvpython executable was not found")
    return {"capture_shape_yx": [cny, cnx], "volume_shape_zyx": [counts[2], counts[1], counts[0]],
            "estimated_peak_bytes": estimated_peak, "available_physical_memory_gib": available,
            "memory_limit_gib": scene["memory_limit_gib"], "physical_memory_reserve_gib": 1,
            "capture_step_mm": scene["capture_step_mm"], "volume_step_mm": scene["volume_step_mm"]}


def _source(x, y, z_mm, frequency_ghz):
    xx, yy = np.meshgrid(x * 1e-3, y * 1e-3)
    z = z_mm * 1e-3
    r = np.sqrt(xx**2 + yy**2 + z**2)
    nx = xx / r
    k = 2 * np.pi * frequency_ghz * 1e9 / 299_792_458
    return np.exp(-1j * k * r) * (k*k/r * (1 - nx*nx) + (1/r**3 + 1j*k/r**2) * (3*nx*nx - 1))


def _prop(prepared, distance, x, y):
    return _propagate_prepared(prepared, distance, bandlimit=True, output_x_mm=x, output_y_mm=y)


def _chain(scene, capture_step, padding, reflections=False):
    c = model.coordinates(scene)
    if reflections:
        _say("  computing one-round-trip reflection correction")
    capx = _grid(float(scene["capture_size_mm"][0]), capture_step, "capture width")
    capy = _grid(float(scene["capture_size_mm"][1]), capture_step, "capture height")
    field = _source(capx, capy, c["plate_back"], scene["frequency_ghz"])
    field *= _screen_mask(capx, capy, scene, True)
    exit_field = propagate_plane(field, capture_step, c["mask_back"]-c["plate_back"], scene["frequency_ghz"],
                                 padding=padding, output_x_mm=capx, output_y_mm=capy)
    if reflections:
        mask_incident = propagate_plane(field, capture_step, c["mask_front"]-c["plate_back"], scene["frequency_ghz"],
                                        padding=padding, output_x_mm=capx, output_y_mm=capy)
        backward = propagate_plane(-(_metal_mask(capx, capy, scene, "mask") * mask_incident), capture_step,
                                   c["mask_front"]-c["plate_back"], scene["frequency_ghz"], padding=padding,
                                   output_x_mm=capx, output_y_mm=capy)
        plate_return = -(_metal_mask(capx, capy, scene, "plate") * backward)
        exit_field += propagate_plane(plate_return, capture_step, c["mask_back"]-c["plate_back"],
                                      scene["frequency_ghz"], padding=padding, output_x_mm=capx, output_y_mm=capy)
    exit_field *= _screen_mask(capx, capy, scene)
    prepared = _prepare_plane(exit_field, capture_step, scene["frequency_ghz"], padding=padding)
    return prepared


def _write_cut(path, coord, vals, coord_name, title, ylabel="Relative intensity", value_label="relative_intensity"):
    with path.with_name(path.name + ".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([coord_name, value_label]); w.writerows(zip(coord, vals))
    fig, ax = plt.subplots(figsize=(7, 3.5)); ax.plot(coord, vals)
    ax.set(xlabel=coord_name, ylabel=ylabel, title=f"{title}\n{LABEL}", ylim=(0, 1.05))
    fig.tight_layout(); fig.savefig(path.with_name(path.name + ".png"), dpi=140); plt.close(fig)


def _write_transverse_csvs(run_dir, depth, x, y, image, intensity_name="relative_intensity"):
    """Write full x/y slices as coordinate/value rows without replacing saved CSVs."""
    run_dir = Path(run_dir)
    peak = float(np.max(image, initial=0))
    normalized = image / peak if peak > 0 else np.zeros_like(image)
    created = skipped = 0
    for suffix, values, name in (("", image, intensity_name),
                                 ("_normalized", normalized, "peak_normalized_intensity")):
        path = run_dir / f"transverse_{depth:g}{suffix}.csv"
        if path.exists():
            skipped += 1
            continue
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x_mm", "y_mm", name])
            writer.writerows((xx, yy, values[iy, ix])
                             for iy, yy in enumerate(y) for ix, xx in enumerate(x))
        created += 1
    return created, skipped


def export_transverse_csvs(run_dir):
    """Export transverse CSVs from an existing complete preview or full-wave volume."""
    run_dir = Path(run_dir)
    scene_path, volume_path = run_dir / "scene.json", run_dir / "volume.h5"
    if not scene_path.is_file():
        return "incompatible: missing scene.json"
    if not volume_path.is_file():
        return "incompatible: missing volume.h5"
    try:
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        depths = _depths(scene)
        mask_back = model.coordinates(scene)["mask_back"]
        with h5py.File(volume_path, "r") as h5:
            if not bool(h5.attrs.get("complete", False)):
                return "incompatible: volume is not marked complete"
            dataset = next((key for key in ("relative_intensity", "relative_E2") if key in h5), None)
            if dataset is None:
                return "incompatible: missing relative_intensity or relative_E2"
            x, y, z = (np.asarray(h5[key], dtype=float) for key in ("x_mm", "y_mm", "z_mm"))
            values = h5[dataset]
            if (x.ndim != 1 or y.ndim != 1 or z.ndim != 1 or not x.size or not y.size or not z.size or
                    values.shape != (len(z), len(y), len(x)) or
                    not all(np.isfinite(axis).all() and np.all(np.diff(axis) > 0) for axis in (x, y, z))):
                return "incompatible: coordinate/data shapes or coordinates are invalid"
            created = skipped = 0
            for depth in depths:
                iz = int(np.argmin(np.abs(z - (mask_back + depth))))
                image = values[iz]
                if not np.isfinite(image).all():
                    return f"incompatible: non-finite {dataset} values at z index {iz}"
                new, existing = _write_transverse_csvs(run_dir, depth, x, y, image, dataset)
                created += new
                skipped += existing
        return f"{len(depths)} depth(s), {created} CSV(s) created, {skipped} existing CSV(s) skipped"
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return f"incompatible: {exc}"


def _write_beam_views(run_dir, normalized_only=False):
    """Write transverse planes and x/y cuts from an existing volume.h5."""
    run_dir = Path(run_dir)
    scene = json.loads((run_dir / "scene.json").read_text(encoding="utf-8"))
    c = model.coordinates(scene)
    with h5py.File(run_dir / "volume.h5", "r") as h5:
        x, y, z = h5["x_mm"][:], h5["y_mm"][:], h5["z_mm"][:]
        iy, ix = int(np.argmin(np.abs(y))), int(np.argmin(np.abs(x)))
        for depth in _depths(scene):
            iz = int(np.argmin(np.abs(z - (c["mask_back"] + depth))))
            image = h5["relative_intensity"][iz]
            if not normalized_only:
                _write_transverse_csvs(run_dir, depth, x, y, image)
            suffix = "_normalized" if normalized_only else ""
            if normalized_only:
                image_peak = float(np.max(image, initial=0))
                image = image / image_peak if image_peak > 0 else np.zeros_like(image)
            for axis, coord, vals, coord_name in (
                    ("x", x, image[iy, :], "x_mm"),
                    ("y", y, image[:, ix], "y_mm")):
                if normalized_only:
                    peak = float(np.max(vals, initial=0))
                    vals = vals / peak if peak > 0 else np.zeros_like(vals)
                    _write_cut(run_dir / f"cut_{axis}_{depth:g}{suffix}", coord, vals, coord_name,
                               f"Peak-normalized {axis} cut at z={z[iz]:g} mm", "Peak-normalized intensity",
                               "peak_normalized_intensity")
                else:
                    location = y[iy] if axis == "x" else x[ix]
                    _write_cut(run_dir / f"cut_{axis}_{depth:g}", coord, vals, coord_name,
                               f"{axis} cut at {'y' if axis == 'x' else 'x'}={location:g} mm, z={z[iz]:g} mm")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(image, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), vmin=0, vmax=1, aspect="equal")
            title = "Peak-normalized transverse intensity" if normalized_only else "Transverse relative intensity"
            label = "Peak-normalized intensity" if normalized_only else "Relative intensity"
            ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"{title} at z={z[iz]:g} mm\n{LABEL}")
            if normalized_only:
                ax.title.set_fontsize(9)
            fig.colorbar(ax.images[0], ax=ax, label=label); fig.tight_layout()
            save_options = {"dpi": 140, **({"bbox_inches": "tight"} if normalized_only else {})}
            fig.savefig(run_dir / f"transverse_{depth:g}{suffix}.png", **save_options)
            ax.invert_xaxis()
            ax.set_title(f"{title} at z={z[iz]:g} mm\nRadar view: looking along +z, +y up\n{LABEL}", fontsize=9)
            fig.tight_layout()
            fig.savefig(run_dir / f"transverse_{depth:g}{suffix}_radar.png", **save_options)
            plt.close(fig)


def _phi_index(phi, degrees):
    return int(np.argmin(np.abs(phi - np.deg2rad(degrees))))


def _farfield(prepared, scene, run_dir):
    theta = np.linspace(0, np.pi/2, 91); phi = np.arange(180) * 2*np.pi/180
    fx = -np.sin(theta)[:, None]*np.cos(phi)[None, :] / (299.792458/scene["frequency_ghz"])
    fy = -np.sin(theta)[:, None]*np.sin(phi)[None, :] / (299.792458/scene["frequency_ghz"])
    # Only spectrum magnitude contributes to scalar angular intensity; interpolation avoids phase-origin ambiguity.
    spec = np.abs(np.fft.fftshift(prepared["spectrum"], axes=(0, 1))[..., 0])**2
    from scipy.interpolate import RegularGridInterpolator
    fy_axis, fx_axis = np.fft.fftshift(prepared["fy"]), np.fft.fftshift(prepared["fx"])
    interp = RegularGridInterpolator((fy_axis, fx_axis), spec, bounds_error=False, fill_value=0)
    power = interp(np.column_stack((fy.ravel(), fx.ravel()))).reshape(fx.shape) * np.cos(theta)[:, None]**2
    peak = power.max(initial=0)
    if peak: power /= peak
    np.savez_compressed(run_dir / "farfield.npz", theta_rad=theta, phi_rad=phi, power=power)
    for angle, label in ((0, "phi0"), (90, "phi90")):
        phi_index = _phi_index(phi, angle)
        _write_cut(run_dir / f"farfield_{label}", np.rad2deg(theta), power[:, phi_index], "theta_deg",
                   f"Scalar angular intensity at phi={angle} deg")


def _say(msg):
    print(f"[preview] {msg}", flush=True)


def _every(i, n):
    """True on roughly every 10% of n steps and on the last one."""
    return (i + 1) % max(1, n // 10) == 0 or i + 1 == n


def run(scene, run_dir, reflections=False):
    """Generate finite two-screen scalar diffraction output in an already reserved directory."""
    run_dir = Path(run_dir); started = time.perf_counter()
    _say(f"one-round-trip reflections {'enabled' if reflections else 'disabled'}")
    (run_dir / "scene.json").write_text(json.dumps(scene, indent=2)+"\n", encoding="utf-8")
    c = model.coordinates(scene)
    x = _grid(float(scene["volume_size_mm"][0]), float(scene["volume_step_mm"]), "volume width")
    y = _grid(float(scene["volume_size_mm"][1]), float(scene["volume_step_mm"]), "volume height")
    z = c["mask_back"] + (np.arange(int(round(scene["volume_size_mm"][2]/scene["volume_step_mm"]))) + .5) * scene["volume_step_mm"]
    try:
        gate = preflight(scene)
        _say(f"volume: {scene['volume_size_mm'][0]:g} x {scene['volume_size_mm'][1]:g} x {scene['volume_size_mm'][2]:g} mm (x x y x z); sampling: {scene['volume_step_mm']:g} mm")
        _say("1/5 building propagation chain + reference planes")
        base = _chain(scene, scene["capture_step_mm"], scene["padding"], reflections)
        depths = _depths(scene)
        reference = {d: _prop(base, d, x, y) for d in depths}
        checks = [{"depth_from_mask_back_mm": d, "base_relative_l2": 0.0} for d in depths]
        _say("2/5 convergence checks")
        for name, ds, pad in (("double_padding", scene["capture_step_mm"], scene["padding"]*2),
                              ("half_capture_step", scene["capture_step_mm"]/2, scene["padding"])):
            _say(f"  convergence variant: {name}")
            variant = _chain(scene, ds, pad, reflections)
            for rec, depth in zip(checks, depths):
                other = _prop(variant, depth, x, y)
                ref_i, other_i = np.abs(reference[depth])**2, np.abs(other)**2
                denom = np.linalg.norm(ref_i); delta = np.linalg.norm(ref_i-other_i)
                rec[name+"_relative_l2"] = float(delta/denom) if denom else (0.0 if delta == 0 else None)
            del variant
        for rec, depth in zip(checks, depths):
            double_pad, half_step = (rec[k] for k in ("double_padding_relative_l2", "half_capture_step_relative_l2"))
            _say(f"  convergence at {depth:g} mm: double padding={double_pad:.3%}" if double_pad is not None else f"  convergence at {depth:g} mm: double padding=undefined")
            _say(f"    half step={half_step:.3%}" if half_step is not None else "    half step=undefined")
        for rec in checks:
            rec["within_5_percent"] = all(rec[k] is not None and rec[k] <= .05 for k in ("double_padding_relative_l2", "half_capture_step_relative_l2"))
        del reference
        nz, ny, nx = len(z), len(y), len(x)
        _say(f"3/5 propagating {nz} z-planes")
        partial = run_dir / ".volume.partial.h5"
        with h5py.File(partial, "w") as h5:
            h5["x_mm"], h5["y_mm"], h5["z_mm"] = x, y, z
            u = h5.create_dataset("U", (nz,ny,nx), dtype="complex64", chunks=(1,min(ny,128),min(nx,128)), compression="gzip", compression_opts=1)
            peak = 0.0
            for i, zz in enumerate(z):
                plane = _prop(base, zz-c["mask_back"], x, y)
                if not np.isfinite(plane).all(): raise ValueError(f"non-finite scalar field at z index {i}")
                u[i] = plane.astype(np.complex64); peak = max(peak, float(np.max(np.abs(plane)**2, initial=0)))
                if _every(i, nz): _say(f"  plane {i+1}/{nz} ({time.perf_counter()-started:.0f}s)")
            inten = h5.create_dataset("relative_intensity", u.shape, dtype="float32", chunks=u.chunks, compression="gzip", compression_opts=1)
            for i in range(nz): inten[i] = (np.abs(u[i])**2 / peak).astype("float32") if peak else 0
            h5.attrs["complete"] = True; h5.attrs["peak_intensity"] = peak
        partial.replace(run_dir / "volume.h5")
        export_started = time.perf_counter()
        _say("4/5 far field + cuts/images")
        _farfield(base, scene, run_dir)
        _write_beam_views(run_dir)
        _write_beam_views(run_dir, normalized_only=True)
        numeric_done = time.perf_counter()
        meta = {"label": LABEL, "status": "incomplete", "full_wave_validated": False, "source": "analytic x-directed Hertzian dipole at origin; arbitrary common scale",
                "reflections": {"enabled": bool(reflections), "round_trips": 1 if reflections else 0,
                                "electric_field_reflection_coefficient": -1 if reflections else None,
                                "plate_reflection_z_mm": c["plate_back"], "mask_reflection_z_mm": c["mask_front"],
                                "gap_mm": c["mask_front"]-c["plate_back"], "transmission_z_mm": c["mask_back"],
                                "approximation_notes": ["scalar single-round-trip correction to the exit-screen model",
                                                         "transmission remains at mask_back while reflection uses mask_front",
                                                         "screen thickness is approximated as free-space propagation",
                                                         "no internal enclosure or mutual vector coupling"]},
                "screen_z_mm": {"plate_back": c["plate_back"], "mask_back": c["mask_back"]}, "ignored_fd_parameters": ["enclosure", "materials", "finite-frame bypass"],
                "ignored_mesh_and_handoff_parameters": {k: scene[k] for k in ("mesh_max_mm", "mesh_detail_mm", "handoff_mm")},
                "assumptions": ["zero-thickness screens", "first opening only", "binary mask cells only", "all other locations opaque", "plate thickness affects its coordinate only", "scalar angular intensity is not calibrated power or NF2FF"],
                "convergence": checks, "converged_within_5_percent": all(q["within_5_percent"] for q in checks),
                "zero_transmission": peak == 0, "effective_preflight": gate,
                "timing_s": {"calculation_s": export_started-started, "numeric_export_s": numeric_done-export_started, "paraview_s": None, "total_s": numeric_done-started}}
        (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2)+"\n", encoding="utf-8")
        flat = [box for box in model.geometry(scene) if not box["name"].startswith("enclosure_")]
        for box in flat:
            plane = c["plate_back"] if box["name"].startswith("plate_") else c["mask_back"]
            box["start"][2] = box["stop"][2] = plane
        view_scene = dict(scene, preview_model=True, geometry=flat, validation_status=LABEL)
        from .paraview_view import write_paraview_files
        pv_started = time.perf_counter()
        _say("5/5 ParaView files")
        with tempfile.TemporaryDirectory(prefix=".preview_", dir=run_dir) as tmp:
            raw_dir = Path(tmp); vp = raw_dir / "scene.json"; vp.write_text(json.dumps(view_scene), encoding="utf-8")
            with h5py.File(run_dir / "volume.h5", "r") as h5:
                raw = np.memmap(raw_dir / "relative_intensity.f32", mode="w+", dtype=np.float32, shape=(nz,ny,nx))
                for i in range(nz): raw[i] = h5["relative_intensity"][i]
                raw.flush(); del raw
            write_paraview_files(run_dir, vp, raw_dir, (nx,ny,nz), (x[0],y[0],z[0]), (scene["volume_step_mm"],)*3)
        meta["timing_s"] = {"calculation_s": export_started-started, "numeric_export_s": numeric_done-export_started,
                            "paraview_s": time.perf_counter()-pv_started, "total_s": time.perf_counter()-started}
        (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2)+"\n", encoding="utf-8")
        _say(f"complete in {meta['timing_s']['total_s']:.1f}s")
        return run_dir
    except Exception as exc:
        if (run_dir / "metadata.json").exists():
            try:
                failed_meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
                failed_meta["status"] = "failed"; failed_meta["error"] = str(exc); failed_meta["timing_s"]["total_s"] = time.perf_counter()-started
                (run_dir / "metadata.json").write_text(json.dumps(failed_meta, indent=2)+"\n", encoding="utf-8")
            except Exception:
                pass
        else:
            (run_dir / "metadata.json").write_text(json.dumps({"label": LABEL, "status": "failed", "full_wave_validated": False, "error": str(exc)}, indent=2)+"\n", encoding="utf-8")
        (run_dir / "failure.json").write_text(json.dumps({"error":str(exc),"run_dir":str(run_dir)},indent=2)+"\n",encoding="utf-8")
        raise
