"""Sequential orchestration of saved preview-queue configurations."""

import json
import time
from pathlib import Path

from . import preview, queue


def _read(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("entries"), list) or not data["entries"]:
        raise ValueError("unsupported or invalid super-queue manifest")
    root = path.parent
    for i, entry in enumerate(data["entries"], 1):
        if not isinstance(entry, dict) or entry.get("status") not in {"pending", "running", "failed", "interrupted", "completed"}:
            raise ValueError("invalid super-queue entry")
        if entry.get("child_dir") != f"queue_{i:03d}" or (root / entry["child_dir"]).resolve().parent != root:
            raise ValueError("invalid child directory in super-queue manifest")
        if not isinstance(entry.get("config_path"), str) or not isinstance(entry.get("snapshot"), dict):
            raise ValueError("invalid config snapshot in super-queue manifest")
        _validate_snapshot(entry["snapshot"])
        if entry.get("status") == "completed" and entry.get("error") is not None:
            raise ValueError("completed super-queue entry has an error")
        if entry.get("status") in {"failed", "interrupted"} and not isinstance(entry.get("error"), str):
            raise ValueError("failed super-queue entry has no error")
    return path, data


def _validate_snapshot(snapshot):
    if (snapshot.get("version") != 1 or not isinstance(snapshot.get("jobs"), list) or not snapshot["jobs"]
            or type(snapshot.get("reflections")) is not bool):
        raise ValueError("invalid child queue snapshot")
    for job in snapshot["jobs"]:
        if (not isinstance(job, dict) or job.get("status") not in {"pending", "running", "failed", "interrupted", "completed"}
                or type(job.get("window")) is not int or not 0 <= job["window"] <= 23
                or not isinstance(job.get("mask_rows"), list) or len(job["mask_rows"]) != 4
                or any(not isinstance(row, str) or len(row) != 4 or set(row) - {"0", "1"} for row in job["mask_rows"])
                or not isinstance(job.get("scene"), dict) or type(job.get("attempts")) is not int or job["attempts"] < 0):
            raise ValueError("invalid job in child queue snapshot")


def _prepare(config):
    source = Path(config).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"queues"} or not isinstance(raw["queues"], list) or not raw["queues"] or any(not isinstance(p, str) or not p for p in raw["queues"]):
        raise ValueError('super-queue config must contain only a nonempty "queues" list of paths')
    prepared = []
    for i, item in enumerate(raw["queues"], 1):
        child_config = (source.parent / item).resolve()
        print(f"[super-queue] validating {i}/{len(raw['queues'])}: {child_config}", flush=True)
        snapshot = queue._load_config(child_config)
        preview.preflight(snapshot["jobs"][0]["scene"])
        prepared.append({"config_path": str(child_config), "snapshot": snapshot,
                         "child_dir": f"queue_{i:03d}", "status": "pending", "error": None})
    return prepared


def run_super_queue(config=None, output=None, resume=None):
    if config is not None and resume is not None:
        raise ValueError("--config and --resume are mutually exclusive")
    if resume is not None and output is not None:
        raise ValueError("--output cannot be used with --resume")
    if resume is None:
        try:
            prepared = _prepare(config)
        except KeyboardInterrupt:
            print("[super-queue] interrupted during validation; no queue was saved", flush=True)
            return 130
        parent = Path(output) if output else Path(__file__).resolve().parent / "preview_super_queues"
        from . import __main__
        parent = __main__._new_run_path(parent)
        manifest = {"version": 1, "entries": prepared}
        path = parent / "super_queue.json"
        try:
            queue._save(path, manifest)
        except KeyboardInterrupt:
            queue._save(path, manifest)
            print(f"[super-queue] interrupted; resume with: python -m coded_aperture_sim preview-super-queue --resume \"{parent}\"", flush=True)
            return 130
    else:
        path, manifest = _read(Path(resume) / "super_queue.json")
        parent = path.parent
    resume_cmd = f"python -m coded_aperture_sim preview-super-queue --resume \"{parent}\""
    skipped = sum(e["status"] == "completed" for e in manifest["entries"])
    print(f"[super-queue] directory {parent}; {len(manifest['entries'])} children; {skipped} resume skips; resume with: {resume_cmd}", flush=True)
    try:
        for i, entry in enumerate(manifest["entries"], 1):
            if entry["status"] == "completed":
                continue
            child = parent / entry["child_dir"]
            child.mkdir(exist_ok=True)
            child_manifest = child / "queue.json"
            if not child_manifest.exists():
                queue._save(child_manifest, entry["snapshot"])
            entry["status"], entry["error"] = "running", None
            queue._save(path, manifest)
            print(f"[super-queue] child {i}/{len(manifest['entries'])}: {entry['config_path']}", flush=True)
            started = time.perf_counter()
            result = queue.run_queue(resume=child)
            if result == 130:
                entry["status"], entry["error"] = "interrupted", "KeyboardInterrupt"
                queue._save(path, manifest)
                print(f"[super-queue] interrupted; resume with: {resume_cmd}", flush=True)
                return 130
            if result:
                entry["status"], entry["error"] = "failed", f"child queue returned {result}"
                queue._save(path, manifest)
                print(f"[super-queue] child {i}/{len(manifest['entries'])} failed: {entry['error']}; resume with: {resume_cmd}", flush=True)
                return 1
            entry["status"], entry["error"] = "completed", None
            queue._save(path, manifest)
            print(f"[super-queue] child {i}/{len(manifest['entries'])} completed in {time.perf_counter()-started:.1f}s", flush=True)
    except KeyboardInterrupt:
        current = next((e for e in manifest["entries"] if e["status"] == "running"), None)
        if current is not None:
            current["status"], current["error"] = "interrupted", "KeyboardInterrupt"
        queue._save(path, manifest)
        print(f"[super-queue] interrupted; resume with: {resume_cmd}", flush=True)
        return 130
    except Exception as exc:
        current = next((e for e in manifest["entries"] if e["status"] == "running"), None)
        if current is not None:
            current["status"], current["error"] = "failed", str(exc)
        queue._save(path, manifest)
        print(f"[super-queue] failed: {exc}; resume with: {resume_cmd}", flush=True)
        return 1
    completed = sum(e["status"] == "completed" for e in manifest["entries"])
    print(f"[super-queue] complete: {completed}/{len(manifest['entries'])} children; {parent}", flush=True)
    return 0

