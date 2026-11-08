"""Resumable sequential previews for 4-by-4 windows of a coded-aperture strip."""

import json
import os
import time
from pathlib import Path

from . import model, preview


def windows_from_rows(rows, windows=None):
    if not isinstance(rows, list) or len(rows) != 4 or any(not isinstance(r, str) or len(r) != 27 or set(r) - {"0", "1"} for r in rows):
        raise ValueError("strip_rows must be four 27-character binary strings")
    if windows is None:
        windows = list(range(24))
    if not isinstance(windows, list) or any(type(i) is not int or not 0 <= i <= 23 for i in windows):
        raise ValueError("windows must be a list of integers from 0 through 23")
    return [{"window": i, "mask_rows": [r[i:i + 4] for r in rows]} for i in windows]


def _save(path, manifest):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def _load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("queue config must be a JSON object")
    if set(config) - {"scene_config", "strip_rows", "windows", "reflections"}:
        raise ValueError("queue config contains unknown fields")
    if not isinstance(config.get("scene_config"), str) or not config["scene_config"]:
        raise ValueError("scene_config must be a relative or absolute path string")
    rows = config.get("strip_rows")
    windows = windows_from_rows(rows, config.get("windows"))
    if not windows:
        raise ValueError("windows must contain at least one window")
    if type(config.get("reflections", False)) is not bool:
        raise ValueError("reflections must be a boolean")
    scene_path = (path.parent / config["scene_config"]).resolve()
    scene = model.load_scene(scene_path)
    jobs = []
    for job in windows:
        scene_copy = dict(scene, mask_rows=job["mask_rows"])
        jobs.append({**job, "status": "pending", "scientific_status": None, "output_path": None,
                     "attempts": 0, "error": None, "scene": scene_copy})
    return {"version": 1, "config_path": str(path), "config": config, "scene_config": str(scene_path),
            "scene": scene, "reflections": config.get("reflections", False), "jobs": jobs}


def _read_manifest(queue_dir):
    path = Path(queue_dir) / "queue.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != 1 or not isinstance(manifest.get("jobs"), list) or not manifest["jobs"]:
        raise ValueError("unsupported or invalid queue manifest")
    if type(manifest.get("reflections")) is not bool:
        raise ValueError("invalid reflections setting in queue manifest")
    for job in manifest["jobs"]:
        if (not isinstance(job, dict) or job.get("status") not in {"pending", "running", "failed", "interrupted", "completed"}
                or type(job.get("window")) is not int or not 0 <= job["window"] <= 23
                or not isinstance(job.get("mask_rows"), list) or len(job["mask_rows"]) != 4
                or not isinstance(job.get("scene"), dict) or type(job.get("attempts")) is not int
                or job["attempts"] < 0
                or any(not isinstance(row, str) or len(row) != 4 or set(row) - {"0", "1"} for row in job["mask_rows"])
                or (job["status"] == "completed" and not isinstance(job.get("output_path"), str))):
            raise ValueError("invalid job in queue manifest")
    return path, manifest


def _export_sequence(queue_dir, manifest):
    completed = [(i, Path(job["output_path"]).resolve()) for i, job in enumerate(manifest["jobs"])
                 if job["status"] == "completed" and job.get("output_path")]
    if completed:
        print(f"[queue] exporting ParaView sequence ({len(completed)} completed frames)", flush=True)
        from .paraview_view import write_paraview_sequence
        write_paraview_sequence(queue_dir, completed)


def run_queue(config=None, output=None, resume=None):
    """Run validated jobs, retaining each run directory and a resumable manifest."""
    from . import __main__

    if resume is not None and output is not None:
        raise ValueError("--output cannot be used with --resume")
    if resume is None:
        manifest = _load_config(config)
        # All scene preflights happen before creating queue or run output.
        try:
            preview.preflight(manifest["jobs"][0]["scene"])
        except KeyboardInterrupt:
            print("[queue] interrupted during initial preflight; no queue output was created", flush=True)
            return 130
        parent = Path(output) if output else Path(__file__).resolve().parent / "preview_queues"
        queue_dir = __main__._new_run_path(parent)
        path = queue_dir / "queue.json"
        _save(path, manifest)
    else:
        queue_dir = Path(resume).resolve()
        path, manifest = _read_manifest(queue_dir)
    skipped = sum(job["status"] == "completed" for job in manifest["jobs"])
    print(f"[queue] directory {queue_dir}; {len(manifest['jobs'])} total; {skipped} resume skips", flush=True)
    if resume is not None:
        try:
            _export_sequence(queue_dir, manifest)
            manifest.pop("sequence_error", None); _save(path, manifest)
        except KeyboardInterrupt:
            manifest["sequence_error"] = "KeyboardInterrupt"; _save(path, manifest)
            print("[queue] interrupted during ParaView sequence export", flush=True)
            return 130
        except Exception as exc:
            manifest["sequence_error"] = str(exc); _save(path, manifest)
            print(f"[queue] sequence export failed: {exc}", flush=True)
            return 1
    for position, job in enumerate(manifest["jobs"]):
        if job["status"] == "completed":
            continue
        try:
            preview.preflight(job["scene"])
        except KeyboardInterrupt:
            job["status"] = "interrupted"; job["error"] = "KeyboardInterrupt"; _save(path, manifest)
            print("[queue] interrupted during preflight", flush=True)
            return 130
        except Exception as exc:
            job["status"] = "failed"; job["error"] = str(exc); _save(path, manifest)
            print(f"[queue] preflight failed: {exc}", flush=True)
            return 1
        run_dir = __main__._new_run_path(queue_dir)
        job["attempts"] += 1; job["status"] = "running"; job["output_path"] = str(run_dir.resolve()); job["error"] = None
        _save(path, manifest)
        start = time.perf_counter()
        print(f"[queue] {position + 1}/{len(manifest['jobs'])} window {job['window']} mask {''.join(job['mask_rows'])}; output {run_dir}", flush=True)
        try:
            preview.run(job["scene"], run_dir, reflections=manifest["reflections"])
            metadata_path = run_dir / "metadata.json"
            job["scientific_status"] = json.loads(metadata_path.read_text(encoding="utf-8")).get("status") if metadata_path.exists() else None
            job["status"] = "completed"  # Execution succeeded; scientific validation may still be incomplete.
            _save(path, manifest)
            print(f"[queue] completed in {time.perf_counter()-start:.1f}s; {sum(j['status']=='completed' for j in manifest['jobs'])}/{len(manifest['jobs'])} complete", flush=True)
        except KeyboardInterrupt:
            job["status"] = "interrupted"; job["error"] = "KeyboardInterrupt"; _save(path, manifest)
            print(f"[queue] {position + 1}/{len(manifest['jobs'])} interrupted after {time.perf_counter()-start:.1f}s; resume with --resume {queue_dir}", flush=True)
            return 130
        except Exception as exc:
            job["status"] = "failed"; job["error"] = str(exc); _save(path, manifest)
            print(f"[queue] failed after {time.perf_counter()-start:.1f}s: {exc}", flush=True)
            return 1
        try:
            _export_sequence(queue_dir, manifest)
            manifest.pop("sequence_error", None); _save(path, manifest)
        except KeyboardInterrupt:
            manifest["sequence_error"] = "KeyboardInterrupt"; _save(path, manifest)
            print("[queue] interrupted during ParaView sequence export", flush=True)
            return 130
        except Exception as exc:
            manifest["sequence_error"] = str(exc); _save(path, manifest)
            print(f"[queue] sequence export failed: {exc}", flush=True)
            return 1
    _save(path, manifest)
    print(f"[queue] complete: {sum(j['status']=='completed' for j in manifest['jobs'])}/{len(manifest['jobs'])} jobs; {queue_dir}", flush=True)
    return 0
