"""Complex vector-field propagation for the coded-aperture simulator."""

from __future__ import annotations

import h5py
import numpy as np
from scipy.ndimage import map_coordinates


def _prepare_plane(field, spacing_mm, frequency_ghz, padding=4):
    """Pad and cache the input spectrum and frequency geometry for reuse."""
    a = np.asarray(field)
    scalar = a.ndim == 2
    if scalar:
        a = a[..., None]
    if a.ndim != 3 or not np.iscomplexobj(a) or min(a.shape[:2]) < 1:
        raise ValueError("field must be a complex (ny,nx[,ncomp]) array")
    dx, dy = (float(spacing_mm),) * 2 if np.isscalar(spacing_mm) else map(float, spacing_mm)
    f = float(frequency_ghz)
    if not np.all(np.isfinite(a)) or not np.isfinite([dx, dy, f]).all():
        raise ValueError("field and parameters must be finite")
    if dx <= 0 or dy <= 0 or f <= 0:
        raise ValueError("spacing and frequency must be positive")
    if isinstance(padding, bool) or not isinstance(padding, (int, np.integer)) or padding < 1:
        raise ValueError("padding must be a positive integer")

    ny, nx, nc = a.shape
    py = ny + 2 * ((padding * ny - ny + 1) // 2)
    px = nx + 2 * ((padding * nx - nx + 1) // 2)
    before_y, before_x = (py - ny) // 2, (px - nx) // 2
    padded = np.zeros((py, px, nc), dtype=np.result_type(a.dtype, np.complex128))
    padded[before_y:before_y + ny, before_x:before_x + nx] = a
    # Input samples are centered about x=y=0, including even-sized arrays.
    fx, fy = np.fft.fftfreq(px, dx), np.fft.fftfreq(py, dy)
    FX, FY = np.meshgrid(fx, fy)
    k = 2 * np.pi * f * 1e9 / 299_792_458.0
    transverse = (2 * np.pi * FX * 1e3) ** 2 + (2 * np.pi * FY * 1e3) ** 2
    kz = np.sqrt((k * k - transverse).astype(np.complex128))
    kz = np.where(kz.real < 0, -kz, kz)
    kz = np.where((transverse > k * k) & (kz.imag > 0), -kz, kz)
    fx2, fy2 = (fx * 1e3)**2, (fy * 1e3)**2
    spectrum = np.fft.fft2(padded, axes=(0, 1))
    return {"spectrum": spectrum, "fx": fx, "fy": fy,
            "fx2": fx2, "fy2": fy2,
            "spacing_mm": (dx, dy), "original_shape": (ny, nx),
            "scalar": scalar, "k": k, "transverse": transverse,
            "kz": kz, "frequency_ghz": f, "padded_shape": (py, px)}


def _propagate_prepared(prepared, distance_mm, bandlimit=True, output_x_mm=None, output_y_mm=None):
    """Evaluate a distance from cached inputs without modifying their spectrum."""
    z = float(distance_mm)
    if not np.isfinite(z) or z < 0:
        raise ValueError("distance must be finite and nonnegative")
    dx, dy = prepared["spacing_mm"]
    ny, nx = prepared["original_shape"]
    py, px = prepared["padded_shape"]
    nc = prepared["spectrum"].shape[-1]
    fx, fy = prepared["fx"], prepared["fy"]
    kz, transverse, k = prepared["kz"], prepared["transverse"], prepared["k"]
    transfer = np.exp(-1j * kz * z * 1e-3)
    if bandlimit:
        wavelength = 299_792_458.0 / (prepared["frequency_ghz"] * 1e9)
        lx, ly = px * dx * 1e-3, py * dy * 1e-3
        fx2, fy2 = prepared["fx2"][None, :], prepared["fy2"][:, None]
        limit = 1 / wavelength**2
        propagating = transverse <= k * k
        safe = (
            fx2 * (1 + (2 * z * 1e-3 / lx)**2) + fy2 <= limit
        ) & (
            fx2 + fy2 * (1 + (2 * z * 1e-3 / ly)**2) <= limit
        )
        transfer *= (~propagating | safe)
    propagated = np.fft.ifft2(prepared["spectrum"] * transfer[..., None], axes=(0, 1))

    def coords(values, n, step):
        if values is None:
            return (np.arange(n) - (n - 1) / 2) * step
        c = np.asarray(values, dtype=float)
        if c.ndim != 1 or len(c) == 0 or not np.isfinite(c).all() or (len(c) > 1 and np.any(np.diff(c) <= 0)):
            raise ValueError("output coordinates must be finite increasing 1-D arrays")
        return c

    ox, oy = coords(output_x_mm, nx, dx), coords(output_y_mm, ny, dy)
    # Physical coordinate zero lies halfway between central samples for even grids.
    ix = ox / dx + (px - 1) / 2
    iy = oy / dy + (py - 1) / 2
    if ix.min() < 0 or ix.max() > px - 1 or iy.min() < 0 or iy.max() > py - 1:
        raise ValueError("requested output grid extends beyond padded propagation window")
    yy, xx = np.meshgrid(iy, ix, indexing="ij")
    out = np.stack([map_coordinates(propagated[..., c], (yy, xx), order=1, mode="nearest") for c in range(nc)], axis=-1)
    return out[..., 0] if prepared["scalar"] else out


def propagate_plane(
    field, spacing_mm, distance_mm, frequency_ghz, padding=4, bandlimit=True,
    output_x_mm=None, output_y_mm=None,
):
    """Propagate a scalar or Cartesian complex field on a uniform xy plane.

    Optional output coordinates are increasing 1-D mm arrays. The larger FFT
    domain is zero-extended from the measured capture plane; no output samples
    outside that domain are silently filled.
    """
    z = float(distance_mm)
    if not np.isfinite(z):
        raise ValueError("field and parameters must be finite")
    if z < 0:
        raise ValueError("spacing and frequency must be positive; distance must be nonnegative")
    return _propagate_prepared(
        _prepare_plane(field, spacing_mm, frequency_ghz, padding), z,
        bandlimit, output_x_mm, output_y_mm,
    )


def validate_handoff(fields_path, scene):
    """Compare propagated handoff fields with saved direct slices when present."""
    from .model import coordinates

    results = {"status": "not_run", "planes": []}
    with h5py.File(fields_path, "r") as h5:
        if not all(k in h5 for k in ("x_mm", "y_mm", "z_mm", "E", "H")):
            results["status"] = "missing_fields"
            return results
        z = np.asarray(h5["z_mm"])
        x, y = np.asarray(h5["x_mm"], dtype=float), np.asarray(h5["y_mm"], dtype=float)
        handoff = h5.attrs.get("handoff_index", -1)
        if isinstance(handoff, (bool, np.bool_)) or not isinstance(handoff, (int, np.integer)) or not 0 <= handoff < len(z):
            results["status"] = "missing_handoff_index"
            return results
        expected_handoff = coordinates(scene)["handoff_z"]
        if not np.isclose(z[int(handoff)], expected_handoff, atol=1e-6, rtol=0):
            results["status"] = "handoff_coordinate_mismatch"
            return results
        zi = float(z[int(handoff)])
        targets = tuple(coordinates(scene)["mask_back"] + offset for offset in (10.0, 20.0, 30.0))
        if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or not np.all(np.diff(a) > 0) for a in (x, y, z)):
            results["status"] = "invalid_coordinates"
            return results
        deltas = [np.diff(a) for a in (x, y)]
        if any(not np.allclose(d, d[0]) for d in deltas):
            results["status"] = "nonuniform_xy_coordinates"
            return results
        expected_shape = (len(z), len(y), len(x), 3)
        for key in ("E", "H"):
            if h5[key].shape != expected_shape or h5[key].dtype.kind != "c":
                results["status"] = f"invalid_{key}_dataset"
                return results
            if not np.isfinite(h5[key][int(handoff)]).all():
                results["status"] = f"nonfinite_{key}_handoff"
                return results
        spacing = (float(np.mean(np.diff(y))), float(np.mean(np.diff(x))))
        freq = float(scene["frequency_ghz"])
        base_padding = int(scene.get("padding", 4))
        comparison_padding = 2 * base_padding
        results["padding_comparison"] = [base_padding, comparison_padding]
        step = float(scene["volume_step_mm"])
        nx = int(round(scene["volume_size_mm"][0] / step))
        ny = int(round(scene["volume_size_mm"][1] / step))
        ox = -scene["volume_size_mm"][0]/2 + (np.arange(nx)+0.5)*step
        oy = -scene["volume_size_mm"][1]/2 + (np.arange(ny)+0.5)*step
        if x[0] > ox[0] or x[-1] < ox[-1] or y[0] > oy[0] or y[-1] < oy[-1]:
            results["status"] = "capture_does_not_cover_roi"
            return results
        roi_x, roi_y = (x >= ox[0]) & (x <= ox[-1]), (y >= oy[0]) & (y <= oy[-1])
        results["padding_checks"] = []
        results["required_planes"] = list(targets)
        for target_z in targets:
            distance = target_z - zi
            matches = np.flatnonzero(np.isclose(z, target_z, atol=1e-6, rtol=0))
            if not len(matches):
                results["planes"].append({"z_mm": target_z, "distance_mm": distance, "status": "missing_direct_plane"})
                continue
            i = int(matches[0])
            rec = {"z_mm": target_z, "distance_mm": distance, "status": "checked"}
            for key in ("E", "H"):
                handoff_field = h5[key][int(handoff)]
                pred = propagate_plane(handoff_field, spacing[::-1], distance, freq, padding=base_padding, output_x_mm=x, output_y_mm=y)
                pred2 = propagate_plane(handoff_field, spacing[::-1], distance, freq, padding=comparison_padding, output_x_mm=x, output_y_mm=y)
                direct = h5[key][i]
                pred, pred2, direct = pred[np.ix_(roi_y, roi_x)], pred2[np.ix_(roi_y, roi_x)], direct[np.ix_(roi_y, roi_x)]
                denom = np.linalg.norm(direct.ravel())
                rec[key + "_relative_l2"] = float(np.linalg.norm((pred - direct).ravel()) / denom) if denom else float(np.linalg.norm(pred))
                pdenom = np.linalg.norm(pred2.ravel())
                rec[key + "_padding_change"] = float(np.linalg.norm((pred2 - pred).ravel()) / pdenom) if pdenom else float(np.linalg.norm(pred))
            results["planes"].append(rec)
        for depth in (50.0, 250.0, 500.0):
            distance = coordinates(scene)["mask_back"] + depth - zi
            rec = {"depth_from_mask_back_mm": depth, "status": "checked"}
            for key in ("E", "H"):
                start = h5[key][int(handoff)]
                p4 = propagate_plane(start, spacing[::-1], distance, freq, padding=base_padding,
                                     output_x_mm=x, output_y_mm=y)[np.ix_(roi_y, roi_x)]
                p2 = propagate_plane(start, spacing[::-1], distance, freq, padding=comparison_padding,
                                     output_x_mm=x, output_y_mm=y)[np.ix_(roi_y, roi_x)]
                den = np.linalg.norm(p2.ravel())
                rec[key + "_padding_change"] = float(np.linalg.norm((p2-p4).ravel()) / den) if den else float(np.linalg.norm(p4))
            results["padding_checks"].append(rec)
    valid = [p for p in results["planes"] if p["status"] == "checked"]
    padding_ok = len(results.get("padding_checks", [])) == 3 and all(
        p[k] <= 0.05 for p in results["padding_checks"] for k in
        ("E_padding_change", "H_padding_change")
    )
    results["status"] = "passed" if len(valid) == 3 and padding_ok and all(
        p[k] <= 0.05 for p in valid for k in
        ("E_relative_l2", "H_relative_l2", "E_padding_change", "H_padding_change")
    ) else ("failed" if valid else "not_run")
    return results


def iter_volume(fields_path, scene):
    """Yield ``(z_mm, E, H)`` target-grid slices, using direct then propagated fields."""
    from .model import coordinates

    with h5py.File(fields_path, "r") as h5:
        x, y, z = (np.asarray(h5[k], dtype=float) for k in ("x_mm", "y_mm", "z_mm"))
        if len(x) < 2 or len(y) < 2 or len(z) < 2 or not all(np.isfinite(c).all() and np.all(np.diff(c) > 0) for c in (x, y, z)):
            raise ValueError("saved field coordinates must be strictly increasing")
        dx, dy = float(np.mean(np.diff(x))), float(np.mean(np.diff(y)))
        if not np.allclose(np.diff(x), dx) or not np.allclose(np.diff(y), dy):
            raise ValueError("saved field x/y coordinates must be uniformly sampled")
        raw_handoff = h5.attrs["handoff_index"]
        if isinstance(raw_handoff, (bool, np.bool_)) or not isinstance(raw_handoff, (int, np.integer)):
            raise ValueError("handoff_index must be an integer")
        handoff = int(raw_handoff)
        if not 0 <= handoff < len(z):
            raise ValueError("handoff_index is outside the saved z coordinates")
        handoff_z = z[handoff]
        if not np.isclose(handoff_z, coordinates(scene)["handoff_z"], atol=1e-6, rtol=0):
            raise ValueError("saved handoff coordinate does not match scene geometry")
        for key in ("E", "H"):
            if h5[key].shape != (len(z), len(y), len(x), 3) or h5[key].dtype.kind != "c":
                raise ValueError(f"saved {key} must be complex with shape (z,y,x,3)")
        dims = scene["volume_size_mm"]
        step = float(scene["volume_step_mm"])
        if not np.isfinite(step) or step <= 0 or len(dims) != 3 or not np.isfinite(dims).all() or any(d <= 0 for d in dims):
            raise ValueError("volume dimensions and step must be finite and positive")
        cells = [d/step for d in dims]
        if any(not np.isclose(n, round(n), atol=1e-9, rtol=0) for n in cells):
            raise ValueError("volume dimensions must be integer multiples of volume_step_mm")
        nx, ny, nz = (int(round(n)) for n in cells)
        ox = -dims[0]/2 + (np.arange(nx)+0.5)*step
        oy = -dims[1]/2 + (np.arange(ny)+0.5)*step
        oz = coordinates(scene)["mask_back"] + (np.arange(nz)+0.5)*step
        if ox[0] < x[0] or ox[-1] > x[-1] or oy[0] < y[0] or oy[-1] > y[-1]:
            raise ValueError("direct capture does not cover the requested target x/y grid")

        def resample(plane):
            ix = (ox-x[0])/dx
            iy = (oy-y[0])/dy
            yy, xx = np.meshgrid(iy, ix, indexing="ij")
            return np.stack([map_coordinates(plane[..., c], (yy, xx), order=1, mode="nearest")
                             for c in range(plane.shape[-1])], axis=-1)

        def direct_at(dataset, zq):
            if np.isclose(zq, z[0], atol=1e-9, rtol=0):
                return resample(dataset[0])
            if zq < z[0] or zq > z[-1]:
                raise ValueError(f"direct field does not cover target z={zq:g} mm")
            hi = int(np.searchsorted(z, zq, side="left"))
            if np.isclose(z[hi], zq, atol=1e-9, rtol=0):
                return resample(dataset[hi])
            lo = hi - 1
            weight = (zq-z[lo])/(z[hi]-z[lo])
            return resample(dataset[lo])*(1-weight) + resample(dataset[hi])*weight

        for target_z in oz:
            planes = []
            for key in ("E", "H"):
                if target_z < handoff_z:
                    plane = direct_at(h5[key], target_z)
                else:
                    start = h5[key][handoff]
                    propagated = propagate_plane(start, (dx, dy), target_z-handoff_z,
                                                 scene["frequency_ghz"], padding=scene.get("padding", 4),
                                                 output_x_mm=ox, output_y_mm=oy)
                    plane = propagated
                planes.append(plane)
            yield float(target_z), planes[0], planes[1]
