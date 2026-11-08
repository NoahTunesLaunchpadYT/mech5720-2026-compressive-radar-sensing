"""Lazy openEMS setup, bounded resource preflight, and simulation runner."""
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import ctypes
from ctypes import wintypes

import numpy as np
import h5py

try:
    from . import model
except ImportError:
    import model


def _mesh(scene):
    """Build a coordinate-aligned mesh; coordinates are in mm (grid unit 1 mm)."""
    c = model.coordinates(scene)
    boxes = model.geometry(scene)
    pec_x = max(max(abs(b["start"][0]), abs(b["stop"][0])) for b in boxes)
    pec_y = max(max(abs(b["start"][1]), abs(b["stop"][1])) for b in boxes)
    xhalf = max(scene["capture_size_mm"][0] / 2, scene["volume_size_mm"][0] / 2, pec_x+5) + 8 * scene["mesh_max_mm"]
    yhalf = max(scene["capture_size_mm"][1] / 2, scene["volume_size_mm"][1] / 2, pec_y+5) + 8 * scene["mesh_max_mm"]
    rear_nf_z = min(0, min(b["start"][2] for b in boxes)) - 5
    zmin, zmax = rear_nf_z - 5 - 8 * scene["mesh_max_mm"], c["mask_back"] + 30 + 8 * scene["mesh_max_mm"]
    d, fine = scene["mesh_max_mm"], scene["mesh_detail_mm"]
    def axis_mesh(low, high, edges, detail_edges=()):
        from CSXCAD.SmoothMeshLines import SmoothMeshLines

        anchors = [low, high, 0.0, *edges]
        detail_edges = np.asarray(detail_edges, dtype=float)
        refined = [float(v) for edge in detail_edges for v in np.arange(edge-1.0, edge+1.01*fine, fine)]
        anchors.extend(refined)
        anchors = np.unique(np.round(np.asarray(anchors), 8))
        anchors = anchors[(anchors >= low) & (anchors <= high)]
        lines = SmoothMeshLines(anchors, d, ratio=1.5)
        ratios = np.maximum(np.diff(lines)[1:] / np.diff(lines)[:-1], np.diff(lines)[:-1] / np.diff(lines)[1:])
        if len(ratios) and np.max(ratios) > 1.50001:
            raise ValueError(f"mesh spacing ratio {np.max(ratios):.3g} exceeds 1.5")
        return lines
    xedges = [v for b in boxes for v in (b["start"][0], b["stop"][0])]
    yedges = [v for b in boxes for v in (b["start"][1], b["stop"][1])]
    zedges = [v for b in boxes for v in (b["start"][2], b["stop"][2])]
    nf_x = max(max(abs(b["start"][0]), abs(b["stop"][0])) for b in boxes) + 5
    nf_y = max(max(abs(b["start"][1]), abs(b["stop"][1])) for b in boxes) + 5
    pitch = scene["pitch_mm"]
    grid_edge = np.arange(-2*pitch, 2*pitch + pitch/2, pitch)
    xedges += [-.25, .25, -nf_x, nf_x, *grid_edge]
    yedges += [0, -nf_y, nf_y, *grid_edge]
    zedges += [-.5, -.25, 0, .25, .5, rear_nf_z, c["mask_back"]+10, c["handoff_z"],
               c["mask_back"]+20, c["mask_back"]+30]
    return (axis_mesh(-xhalf, xhalf, xedges, xedges),
            axis_mesh(-yhalf, yhalf, yedges, yedges),
            axis_mesh(zmin, zmax, zedges, zedges))


def preflight(scene):
    """Count the generated local FDTD mesh and conservatively bound peak memory."""
    x, y, z = _mesh(scene)
    nodes = len(x) * len(y) * len(z)
    cells = (len(x)-1) * (len(y)-1) * (len(z)-1)
    # 128 B/node bounds solver state; FD E/H recording is restricted to the direct region.
    nx_fd = np.count_nonzero((x >= -scene["capture_size_mm"][0]/2-scene["mesh_max_mm"]) & (x <= scene["capture_size_mm"][0]/2+scene["mesh_max_mm"]))
    ny_fd = np.count_nonzero((y >= -scene["capture_size_mm"][1]/2-scene["mesh_max_mm"]) & (y <= scene["capture_size_mm"][1]/2+scene["mesh_max_mm"]))
    c = model.coordinates(scene)
    nz_fd = np.count_nonzero((z >= c["mask_back"]-scene["mesh_max_mm"]) & (z <= c["mask_back"]+30+scene["mesh_max_mm"]))
    fd_nodes = nx_fd * ny_fd * nz_fd
    capture_nx = math.ceil(scene["capture_size_mm"][0] / scene["capture_step_mm"])
    capture_ny = math.ceil(scene["capture_size_mm"][1] / scene["capture_step_mm"])
    fft_work = capture_nx * capture_ny * max(8, scene["padding"])**2 * 3 * 16 * 2
    peak = (nodes * 128 + fd_nodes * 72 + fft_work) / 1024**3
    ratios = {axis: float(np.max(np.maximum(np.diff(lines)[1:] / np.diff(lines)[:-1],
                                           np.diff(lines)[:-1] / np.diff(lines)[1:])))
              for axis, lines in zip("xyz", (x, y, z))}
    memory = _available_physical_memory()
    total_gib, available_gib = (None, None) if memory is None else tuple(v / 1024**3 for v in memory)
    available_gate = None if available_gib is None else bool(peak <= max(0, available_gib-1))
    configured_gate = peak <= scene["memory_limit_gib"]
    return {"mesh_lines": {"x": len(x), "y": len(y), "z": len(z)},
            "mesh_nodes": nodes, "mesh_cells": cells, "fd_dump_nodes": int(fd_nodes),
            "mesh_max_spacing_ratio": ratios,
            "estimated_peak_gib": float(peak), "fft_work_bytes": int(fft_work),
            "memory_limit_gib": scene["memory_limit_gib"],
            "physical_memory_total_gib": total_gib,
            "available_physical_memory_gib": available_gib,
            "physical_memory_reserve_gib": 1,
            "within_configured_limit": bool(configured_gate),
            "within_available_memory": None if available_gate is None else bool(available_gate),
            "resource_gate": bool(configured_gate and available_gate is not False),
            "within_memory_limit": bool(configured_gate and available_gate is not False),
            "mesh_bounds_mm": [[float(a[0]), float(a[-1])] for a in (x, y, z)]}


def _available_physical_memory():
    """Return total and currently available physical RAM on Windows, otherwise None."""
    if os.name != "nt":
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return status.ullTotalPhys, status.ullAvailPhys


def _bindings():
    try:
        if os.name == "nt" and Path(r"C:\openEMS").is_dir():
            _DLL_DIRECTORY = os.add_dll_directory(r"C:\openEMS")
        from CSXCAD import ContinuousStructure
        from openEMS import openEMS
        from openEMS.nf2ff import nf2ff
        from openEMS.utilities import HDF5Dump
    except ImportError as exc:
        raise RuntimeError("openEMS, CSXCAD and their Python bindings are required; install them before solving") from exc
    return ContinuousStructure, openEMS, nf2ff, HDF5Dump


def _save_fields(run_dir, e_file, h_file, frequency, handoff_z, validation_z,
                 capture_size_mm, capture_step_mm):
    """Resample openEMS complex FD fields onto common uniform x/y coordinates."""
    _, _, _, HDF5Dump = _bindings()
    from scipy.interpolate import RegularGridInterpolator
    with HDF5Dump(e_file) as ed, HDF5Dump(h_file) as hd:
        if not ed.IsFD() or not hd.IsFD():
            raise ValueError("field dumps must contain frequency-domain data")
        mesh = ed.GetMesh()
        hmesh = hd.GetMesh()
        if mesh["type"] != 0 or mesh["type"] != hmesh["type"] or any(
                a.shape != b.shape or not np.allclose(a, b) for a, b in zip(mesh["lines"], hmesh["lines"])):
            raise ValueError("electric and magnetic dumps must share a Cartesian mesh")
        native_x, native_y, native_z = (axis * 1000 for axis in mesh["lines"])
        x = np.arange(-capture_size_mm[0]/2, capture_size_mm[0]/2 + capture_step_mm/2, capture_step_mm)
        y = np.arange(-capture_size_mm[1]/2, capture_size_mm[1]/2 + capture_step_mm/2, capture_step_mm)
        if x[0] < native_x[0] or x[-1] > native_x[-1] or y[0] < native_y[0] or y[-1] > native_y[-1]:
            raise ValueError("native openEMS field does not cover the requested uniform capture plane")
        yy, xx = np.meshgrid(y, x, indexing="ij")
        query = np.column_stack((yy.ravel(), xx.ravel()))
        z = np.unique(np.r_[native_z, handoff_z, validation_z])
        zids = [int(np.argmin(np.abs(z-v))) for v in [handoff_z, *validation_z]]
        if any(abs(z[i] - v) > 1e-9 for i, v in zip(zids, [handoff_z, *validation_z])):
            raise ValueError("failed to insert handoff and validation coordinates")
        out = Path(run_dir) / "fields.h5"
        with h5py.File(out, "w") as f:
            f.create_dataset("x_mm", data=x)
            f.create_dataset("y_mm", data=y)
            f.create_dataset("z_mm", data=z)
            e_ds = f.create_dataset("E", (len(z), len(y), len(x), 3), dtype="complex64", chunks=(1, min(64,len(y)), min(64,len(x)),3))
            h_ds = f.create_dataset("H", e_ds.shape, dtype="complex64", chunks=e_ds.chunks)
            f.attrs["handoff_index"] = zids[0]
            f.attrs["validation_indices"] = zids[1:]
            f.attrs["frequency_hz"] = frequency
            f.attrs["phasor_convention"] = "openEMS frequency-domain +iwt"
            for iz, zpos in enumerate(z):
                upper = int(np.searchsorted(native_z, zpos))
                if upper < len(native_z) and abs(native_z[upper]-zpos) < 1e-9:
                    lower = upper
                    weight = 0.0
                else:
                    lower = upper-1
                    if lower < 0 or upper >= len(native_z):
                        raise ValueError("requested output plane is outside openEMS dump bounds")
                    weight = (zpos-native_z[lower])/(native_z[upper]-native_z[lower])
                fields = []
                for dump in (ed, hd):
                    dump.SetPlane("z", idx=lower)
                    lo = dump.GetFieldAtFrequency(frequency)
                    if upper != lower:
                        dump.SetPlane("z", idx=upper)
                        hi = dump.GetFieldAtFrequency(frequency)
                        lo = lo*(1-weight) + hi*weight
                    lo = np.moveaxis(lo, 0, -1)  # (x, y, component)
                    interp = RegularGridInterpolator((native_y, native_x),
                                                     np.transpose(lo, (1, 0, 2)),
                                                     bounds_error=True)
                    fields.append(interp(query).reshape(len(y), len(x), 3))
                e_ds[iz], h_ds[iz] = fields
    return str(out)


def solve(scene, run_dir, smoke=False):
    """Run openEMS in a log-capturing worker; enforce the configured RAM cap."""
    if smoke:
        return smoke_test(run_dir)
    report = preflight(scene)
    if not report["resource_gate"]:
        raise MemoryError(f"mesh estimate {report['estimated_peak_gib']:.2f} GiB exceeds configured/current RAM gate; see preflight report")
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    scene_path = run_dir / "scene.json"
    scene_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    log_path = run_dir / "solver.log"
    print(f"[solve] {run_dir.name}: {report['mesh_nodes']:,} nodes, est. peak {report['estimated_peak_gib']:.2f} GiB; log: {log_path}", flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(scene_path), str(run_dir)],
                                stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                proc.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                # Echo the newest openEMS progress line (or the last log line during mesh/operator setup).
                lines = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                step = [l for l in lines if "Timestep:" in l]
                print(f"[solve +{time.perf_counter() - start:.0f}s] {(step or lines or ['starting...'])[-1].strip()}", flush=True)
    print(f"[solve] worker finished in {time.perf_counter() - start:.0f}s, exit code {proc.returncode}", flush=True)
    if proc.returncode:
        raise RuntimeError(f"openEMS worker failed with exit code {proc.returncode}; see {log_path}")
    metadata_path = run_dir / "solver_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"end-criteria of ([^\s]+)dB reached after (\d+) timesteps", log_text)
    metadata["convergence"] = ({"status": "energy_threshold_reached", "reported_db": match.group(1),
                                "timesteps": int(match.group(2))} if match else
                               {"status": "not_confirmed", "reason": "solver log contains no energy-stop record"})
    metadata["solver_log"] = str(log_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not match:
        raise RuntimeError(f"openEMS returned without a confirmed energy stop; see {log_path}")
    return {"run_dir": str(run_dir), "preflight": report, "completed": True,
            "convergence": metadata["convergence"], "field_dump": metadata["field_export"],
            "farfield": str(run_dir / "farfield.npz")}


def _solve_worker(scene, run_dir):
    """Internal production simulation body executed by the log-capturing worker."""
    report = preflight(scene)
    if not report["resource_gate"]:
        raise MemoryError("worker mesh exceeds the configured memory limit")
    ContinuousStructure, OpenEMS, NF2FF, _ = _bindings()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    x, y, z = _mesh(scene)
    csx = ContinuousStructure()
    grid = csx.GetGrid()
    for axis, lines in zip("xyz", (x, y, z)):
        grid.SetLines(axis, lines)
    grid.SetDeltaUnit(1e-3)
    pec = csx.AddMetal("PEC")
    for item in model.geometry(scene):
        pec.AddBox(item["start"], item["stop"], priority=10)
    # The 0.5 mm x-polarised openEMS voltage excitation approximates an ideal electric dipole.
    source = csx.AddExcitation("x_dipole", 0, [1, 0, 0])
    source.AddBox([-0.25, 0, 0], [0.25, 0, 0], priority=20)
    fdtd = OpenEMS(NrTS=2_000_000, EndCriteria=1e-5)
    fdtd.SetCSX(csx)
    fdtd.SetGaussExcite(scene["frequency_ghz"] * 1e9, 5e9)
    fdtd.SetBoundaryCond(["PML_8"] * 6)
    c = model.coordinates(scene)
    capture = csx.AddDump("fields_E", dump_type=10, dump_mode=2, frequency=[scene["frequency_ghz"]*1e9], file_type=1)
    capture.AddBox([-scene["capture_size_mm"][0]/2-scene["mesh_max_mm"], -scene["capture_size_mm"][1]/2-scene["mesh_max_mm"], c["mask_back"]-scene["mesh_max_mm"]],
                   [scene["capture_size_mm"][0]/2+scene["mesh_max_mm"], scene["capture_size_mm"][1]/2+scene["mesh_max_mm"], c["mask_back"]+30+scene["mesh_max_mm"]])
    magnetic = csx.AddDump("fields_H", dump_type=11, dump_mode=2, frequency=[scene["frequency_ghz"]*1e9], file_type=1)
    magnetic.AddBox([-scene["capture_size_mm"][0]/2-scene["mesh_max_mm"], -scene["capture_size_mm"][1]/2-scene["mesh_max_mm"], c["mask_back"]-scene["mesh_max_mm"]],
                    [scene["capture_size_mm"][0]/2+scene["mesh_max_mm"], scene["capture_size_mm"][1]/2+scene["mesh_max_mm"], c["mask_back"]+30+scene["mesh_max_mm"]])
    boxes = model.geometry(scene)
    nf_x = max(max(abs(b["start"][0]), abs(b["stop"][0])) for b in boxes) + 5
    nf_y = max(max(abs(b["start"][1]), abs(b["stop"][1])) for b in boxes) + 5
    rear_nf_z = min(0, min(b["start"][2] for b in boxes)) - 5
    nfbox = fdtd.CreateNF2FFBox("nf2ff", start=[-nf_x, -nf_y, rear_nf_z],
                                stop=[nf_x, nf_y, c["mask_back"]+10],
                                frequency=[scene["frequency_ghz"]*1e9])
    fdtd.Run(str(run_dir), verbose=2, numThreads=max(1, (os.cpu_count() or 2)-1))
    field_file = _save_fields(run_dir, run_dir / "fields_E.h5", run_dir / "fields_H.h5",
                              scene["frequency_ghz"]*1e9, c["handoff_z"],
                              [c["mask_back"]+10, c["mask_back"]+20, c["mask_back"]+30],
                              scene["capture_size_mm"], scene["capture_step_mm"])
    theta = np.linspace(0, 180, 181)
    phi = np.linspace(0, 360, 361, endpoint=False)
    far = nfbox.CalcNF2FF(str(run_dir), scene["frequency_ghz"]*1e9, theta, phi, verbose=1, read_cached=False)
    np.savez_compressed(run_dir / "farfield.npz", theta_rad=far.theta, phi_rad=far.phi,
                        power=np.asarray(far.P_rad)[0], units="arbitrary source-weighted units")
    (run_dir / "solver_metadata.json").write_text(json.dumps({
        "scene": scene, "preflight": report, "solver": "openEMS", "python": sys.version,
        "platform": platform.platform(), "frequency_hz": scene["frequency_ghz"]*1e9,
        "boundary": ["PML_8"]*6, "source": "0.5 mm x-polarised openEMS voltage excitation approximating an ideal electric dipole",
        "convergence": {"status": "inspect solver.log"},
        "near_to_far_box": "calculated through openEMS nf2ff.CalcNF2FF",
        "field_export": field_file,
        "handoff_z_mm": c["handoff_z"], "validation_z_mm": [c["mask_back"]+10, c["mask_back"]+20, c["mask_back"]+30]
    }, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "preflight": report, "completed": True,
            "field_dump": field_file, "farfield": str(run_dir / "farfield.npz")}


def smoke_test(run_dir):
    """Run a separately labelled small-domain dipole diagnostic (never production geometry)."""
    ContinuousStructure, OpenEMS, NF2FF, _ = _bindings()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    csx = ContinuousStructure()
    grid = csx.GetGrid()
    xlines, ylines, zlines = np.linspace(-10, 10, 41), np.linspace(-8, 8, 33), np.linspace(-10, 10, 41)
    for axis, lines in zip("xyz", (xlines, ylines, zlines)):
        grid.SetLines(axis, lines)
    grid.SetDeltaUnit(1e-3)
    source = csx.AddExcitation("smoke_x_dipole", 0, [1, 0, 0])
    source.AddBox([-.25, 0, 0], [.25, 0, 0])
    freq = 60.5e9
    e = csx.AddDump("smoke_E", dump_type=10, dump_mode=2, frequency=[freq], file_type=1)
    h = csx.AddDump("smoke_H", dump_type=11, dump_mode=2, frequency=[freq], file_type=1)
    e.AddBox([-9.5, -8, -9.5], [9.5, 8, 9.5])
    h.AddBox([-9.5, -8, -9.5], [9.5, 8, 9.5])
    fdtd = OpenEMS(NrTS=2000, EndCriteria=1e-3)
    fdtd.SetCSX(csx)
    fdtd.SetGaussExcite(freq, 5e9)
    fdtd.SetBoundaryCond(["PML_8"] * 6)
    nfbox = fdtd.CreateNF2FFBox("smoke_nf2ff", frequency=[freq])
    fdtd.Run(str(run_dir), verbose=2, numThreads=1)
    field_file = _save_fields(run_dir, run_dir / "smoke_E.h5", run_dir / "smoke_H.h5",
                              freq, 1, [2, 3, 4], [18, 15], 0.5)
    theta = np.linspace(0, 180, 91)
    phi = np.linspace(0, 360, 181, endpoint=False)
    far = nfbox.CalcNF2FF(str(run_dir), freq, theta, phi, verbose=1, read_cached=False)
    np.savez_compressed(run_dir / "farfield.npz", theta_rad=far.theta, phi_rad=far.phi,
                        power=np.asarray(far.P_rad)[0], units="arbitrary source-weighted units")
    result = {"completed": True, "diagnostic_only": True, "domain_mm": [-10, 10],
              "production_geometry": False, "solver": "openEMS", "fields": field_file,
              "farfield": str(run_dir / "farfield.npz"), "iterations_capped": 2000}
    (run_dir / "smoke_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("solver.py is a library; worker mode is called by solve()")
    scene = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    _solve_worker(scene, sys.argv[3])
