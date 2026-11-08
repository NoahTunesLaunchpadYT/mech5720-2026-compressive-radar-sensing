"""Rebuild numeric and ParaView exports from a saved production run."""

import csv
import json
import tempfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import model
from .paraview_view import write_paraview_files


def _cut(path, coord, value, coordinate_name, label, value_name="relative_E2", y_label="Relative |E|²"):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([coordinate_name, value_name])
        writer.writerows(zip(coord, value))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(coord, value)
    x_label = "Angle (degrees)" if coordinate_name.endswith("_deg") else "Position (mm)"
    ax.set(xlabel=x_label, ylabel=y_label, title=label, ylim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def _field_scalars(e, h, peak):
    """Return normalized intensity and signed +z source-weighted flux proxy."""
    e2 = np.sum(np.abs(e) ** 2, axis=-1)
    cross_z = e[..., 0] * np.conj(h[..., 1]) - e[..., 1] * np.conj(h[..., 0])
    return e2 / peak, 0.5 * cross_z.real


def _build_volume(fields_path, scene, volume_path, scene_path):
    """Stream direct and propagated fields into the requested target volume."""
    from .propagation import iter_volume

    step = float(scene["volume_step_mm"])
    nx, ny, nz = (int(round(d / step)) for d in scene["volume_size_mm"])
    origin = (-scene["volume_size_mm"][0] / 2 + step / 2,
              -scene["volume_size_mm"][1] / 2 + step / 2,
              model.coordinates(scene)["mask_back"] + step / 2)
    coords = (origin[0] + np.arange(nx) * step,
              origin[1] + np.arange(ny) * step,
              origin[2] + np.arange(nz) * step)
    partial_path = volume_path.with_name("volume.partial.h5")
    source_stat, scene_stat = fields_path.stat(), scene_path.stat()
    peak = 0.0
    with h5py.File(partial_path, "w") as out:
        for key, values in zip(("x_mm", "y_mm", "z_mm"), coords):
            out.create_dataset(key, data=values)
        shape = (nz, ny, nx, 3)
        chunks = (1, min(ny, 64), min(nx, 64), 3)
        e_out = out.create_dataset("E", shape, dtype="complex64", chunks=chunks, compression="gzip", compression_opts=1)
        h_out = out.create_dataset("H", shape, dtype="complex64", chunks=chunks, compression="gzip", compression_opts=1)
        out.attrs["units"] = "E/H: source-spectrum-weighted arbitrary units; coordinates: mm"
        out.attrs["Sz_relative"] = "signed +z flux proxy in arbitrary source-spectrum-weighted units"
        out.attrs["volume_origin_mm"] = origin
        out.attrs["volume_step_mm"] = step
        for iz, (_, e, h) in enumerate(iter_volume(fields_path, scene)):
            if not np.isfinite(e).all() or not np.isfinite(h).all():
                raise ValueError(f"propagated field slice {iz} contains non-finite values")
            e_out[iz] = e.astype(np.complex64)
            h_out[iz] = h.astype(np.complex64)
            peak = max(peak, float(np.sum(np.abs(e) ** 2, axis=-1).max(initial=0)))
            if (iz + 1) % max(1, nz // 10) == 0 or iz + 1 == nz:
                print(f"[export] volume plane {iz + 1}/{nz}", flush=True)
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError("propagated target volume has no finite nonzero electric field")
        relative = out.create_dataset("relative_E2", (nz, ny, nx), dtype="float32", chunks=(1, min(ny, 128), min(nx, 128)), compression="gzip", compression_opts=1)
        sz = out.create_dataset("Sz_relative", relative.shape, dtype="float32", chunks=relative.chunks, compression="gzip", compression_opts=1)
        for iz in range(nz):
            relative[iz], sz[iz] = _field_scalars(e_out[iz], h_out[iz], peak)
        out.attrs["peak_E2"] = peak
        out.attrs["complete"] = True
        out.attrs["source_fields_size"] = source_stat.st_size
        out.attrs["source_fields_mtime_ns"] = source_stat.st_mtime_ns
        out.attrs["scene_size"] = scene_stat.st_size
        out.attrs["scene_mtime_ns"] = scene_stat.st_mtime_ns
    partial_path.replace(volume_path)
    return volume_path


def export(run_dir):
    run_dir = Path(run_dir)
    fields_path = run_dir / "fields.h5"
    scene_path = run_dir / "scene.json"
    if not fields_path.is_file() or not scene_path.is_file():
        raise FileNotFoundError(f"{run_dir} is not a complete saved run: scene.json and fields.h5 are required")
    scene = model.load_scene(scene_path)
    with h5py.File(fields_path, "r") as h5:
        required = ("x_mm", "y_mm", "z_mm", "E", "H")
        missing = [key for key in required if key not in h5]
        if missing:
            raise ValueError(f"saved run is missing datasets: {', '.join(missing)}")
        x, y, z = (np.asarray(h5[k], dtype=float) for k in ("x_mm", "y_mm", "z_mm"))
        expected = (len(z), len(y), len(x), 3)
        if h5["E"].shape != expected or h5["H"].shape != expected:
            raise ValueError(f"field datasets must have shape {expected}; got E={h5['E'].shape}, H={h5['H'].shape}")
    volume_path = run_dir / "volume.h5"
    rebuild = not volume_path.is_file()
    if not rebuild:
        fields_stat, scene_stat = fields_path.stat(), scene_path.stat()
        try:
            with h5py.File(volume_path, "r") as saved:
                rebuild = not bool(saved.attrs.get("complete", False)) or any(
                    saved.attrs.get(key) != value for key, value in (
                        ("source_fields_size", fields_stat.st_size),
                        ("source_fields_mtime_ns", fields_stat.st_mtime_ns),
                        ("scene_size", scene_stat.st_size),
                        ("scene_mtime_ns", scene_stat.st_mtime_ns)))
        except OSError:
            rebuild = True
    if rebuild:
        _build_volume(fields_path, scene, volume_path, scene_path)
    with h5py.File(volume_path, "r") as volume:
        required_volume = ("x_mm", "y_mm", "z_mm", "E", "H", "relative_E2", "Sz_relative")
        missing = [key for key in required_volume if key not in volume]
        if missing:
            raise ValueError(f"saved target volume is missing datasets: {', '.join(missing)}")
        x, y, z = (np.asarray(volume[k], dtype=float) for k in ("x_mm", "y_mm", "z_mm"))
        shape = (len(z), len(y), len(x))
        if volume["E"].shape != shape + (3,) or volume["H"].shape != shape + (3,) or any(volume[k].shape != shape for k in ("relative_E2", "Sz_relative")):
            raise ValueError("saved target volume datasets do not share the expected z,y,x shape")
        if not all(np.isfinite(v).all() and np.all(np.diff(v) > 0) for v in (x, y, z)):
            raise ValueError("saved target volume coordinates must be finite and strictly increasing")
        if any(len(v) > 2 and not np.allclose(np.diff(v), np.diff(v)[0]) for v in (x, y, z)):
            raise ValueError("saved target volume coordinates must be uniformly spaced")
        with tempfile.TemporaryDirectory(prefix=".coded_aperture_volume_", dir=run_dir) as tmp:
            raw_dir = Path(tmp)
            for key, filename in (("relative_E2", "relative_E2.f32"), ("Sz_relative", "Sz_relative.f32")):
                raw = np.memmap(raw_dir / filename, mode="w+", dtype=np.float32, shape=shape)
                for zi in range(len(z)):
                    plane = volume[key][zi]
                    if not np.isfinite(plane).all() or not np.isfinite(volume["E"][zi]).all() or not np.isfinite(volume["H"][zi]).all():
                        raise ValueError(f"saved target volume contains non-finite fields at z index {zi}")
                    raw[zi] = plane
                raw.flush()
                del raw
            ff_path = run_dir / "farfield.npz"
            if not ff_path.is_file():
                raise FileNotFoundError("saved run is missing farfield.npz; export cannot invent far-field data")
            with np.load(ff_path) as ff:
                if not all(k in ff for k in ("theta_rad", "phi_rad", "power")):
                    raise ValueError("farfield.npz must contain theta_rad, phi_rad, and power")
                theta, phi, power = (np.asarray(ff[k], dtype=float) for k in ("theta_rad", "phi_rad", "power"))
                if power.shape != (len(theta), len(phi)):
                    raise ValueError("far-field power shape does not match its angular coordinates")
                if not all(np.isfinite(a).all() for a in (theta, phi, power)) or np.any(power < 0):
                    raise ValueError("far-field coordinates and power must be finite and power nonnegative")
                if power.max(initial=0) <= 0:
                    raise ValueError("saved far-field power is identically zero")
                normalized_power = power / float(power.max())
                wrap = lambda angles: (angles + np.pi) % (2 * np.pi) - np.pi
                phi0 = int(np.argmin(np.abs(wrap(phi))))
                phi90 = int(np.argmin(np.abs(wrap(phi - np.pi / 2))))
                _cut(run_dir / "farfield_phi0.csv", np.rad2deg(theta), normalized_power[:, phi0],
                     "theta_deg", f"Far-field angular cut at phi={np.rad2deg(phi[phi0]):g} deg",
                     "relative_radial_power_flux", "Relative radial power flux")
                _cut(run_dir / "farfield_phi90.csv", np.rad2deg(theta), normalized_power[:, phi90],
                     "theta_deg", f"Far-field angular cut at phi={np.rad2deg(phi[phi90]):g} deg",
                     "relative_radial_power_flux", "Relative radial power flux")
            # Center planes make orientation and the physical coordinates explicit.
            iz = int(np.argmin(np.abs(z - np.mean(z))))
            iy, ix = int(np.argmin(np.abs(y))), int(np.argmin(np.abs(x)))
            center_e2 = volume["relative_E2"][iz]
            _cut(run_dir / "cut_x.csv", x, center_e2[iy, :], "x_mm", f"x cut at y={y[iy]:g} mm, z={z[iz]:g} mm")
            _cut(run_dir / "cut_y.csv", y, center_e2[:, ix], "y_mm", f"y cut at x={x[ix]:g} mm, z={z[iz]:g} mm")
            view_scene = raw_dir / "scene.json"
            validation_path = run_dir / "validation.json"
            validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
            validation_status = validation.get("overall_status", validation.get("status", "not recorded"))
            if validation_status != "passed":
                validation_status = f"{validation_status} / UNVALIDATED"
            view_scene.write_text(json.dumps(dict(scene, geometry=model.geometry(scene),
                                                  validation_status=validation_status)), encoding="utf-8")
            write_paraview_files(run_dir, view_scene, raw_dir, (len(x), len(y), len(z)),
                                 (x[0], y[0], z[0]),
                                 (float(np.mean(np.diff(x))) if len(x)>1 else 1,
                                  float(np.mean(np.diff(y))) if len(y)>1 else 1,
                                  float(np.mean(np.diff(z))) if len(z)>1 else 1))
    from .preview import export_transverse_csvs
    print(f"[export] transverse CSVs: {export_transverse_csvs(run_dir)}", flush=True)
    return run_dir
