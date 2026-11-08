import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coded_aperture_sim import queue


class QueueTests(unittest.TestCase):
    def test_slices_and_validates_strip(self):
        rows = ["012" * 9, "1" * 27, "0" * 27, "1" * 27]
        with self.assertRaises(ValueError):
            queue.windows_from_rows(rows)
        rows[0] = "010" * 9
        default = queue.windows_from_rows(rows)
        self.assertEqual(len(default), 24)
        self.assertEqual(default[0]["mask_rows"], [row[:4] for row in rows])
        self.assertEqual(default[-1]["mask_rows"], [row[23:27] for row in rows])
        self.assertEqual(default[0]["mask_rows"][0][1:], default[1]["mask_rows"][0][:3])
        got = queue.windows_from_rows(rows, [2, 0, 2])
        self.assertEqual([j["window"] for j in got], [2, 0, 2])
        self.assertEqual(got[0]["mask_rows"][0], "0010")
        for bad in ([24], [True], "1"):
            with self.assertRaises(ValueError):
                queue.windows_from_rows(rows, bad)

    def test_example_config_loads(self):
        example = Path(__file__).resolve().parents[1] / "queue_impulse_mask.json"
        with patch.object(queue.model, "load_scene", return_value={"mask_rows": ["1111"] * 4}):
            snapshot = queue._load_config(example)
        self.assertEqual(snapshot["scene_config"], str((example.parent / "scene_lowres.json").resolve()))
        self.assertEqual(len(snapshot["jobs"]), 24)

    def _config(self, root):
        scene = root / "scene.json"
        scene.write_text("{}", encoding="utf-8")
        cfg = root / "queue.json"
        cfg.write_text(json.dumps({"scene_config": "scene.json", "strip_rows": ["0" * 27] * 4,
                                   "windows": [3, 3], "reflections": False}), encoding="utf-8")
        return cfg

    def test_failure_then_resume_uses_snapshot_and_retries_stale_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg = self._config(root)
            calls = []
            def fake_run(scene, target, reflections=False):
                calls.append(scene["mask_rows"])
                if len(calls) == 1:
                    raise RuntimeError("expected")
                (target / "metadata.json").write_text('{"status":"incomplete"}')
            def pv(scene): return {}
            with patch.object(queue.model, "load_scene", return_value={"mask_rows": ["1111"]*4}), \
                 patch.object(queue.preview, "preflight", side_effect=pv), patch.object(queue.preview, "run", side_effect=fake_run), \
                 patch("coded_aperture_sim.paraview_view.write_paraview_sequence") as seq:
                self.assertEqual(queue.run_queue(config=cfg, output=root / "out"), 1)
                qdir = next((root / "out").iterdir()); mf = json.loads((qdir / "queue.json").read_text())
                self.assertEqual(mf["jobs"][0]["status"], "failed")
                self.assertEqual(mf["jobs"][0]["attempts"], 1)
                # A hard interruption after starting the second job leaves running state.
                mf["jobs"][1]["status"] = "running"; mf["jobs"][1]["attempts"] = 1
                (qdir / "queue.json").write_text(json.dumps(mf))
                cfg.unlink(); (root / "scene.json").unlink()
                self.assertEqual(queue.run_queue(resume=qdir), 0)
                mf = json.loads((qdir / "queue.json").read_text())
                self.assertEqual([j["status"] for j in mf["jobs"]], ["completed", "completed"])
                self.assertEqual(mf["jobs"][1]["attempts"], 2)
                self.assertEqual(len(calls), 3)
                self.assertGreaterEqual(seq.call_count, 2)

    def test_keyboard_interrupt_persists_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg = self._config(root)
            calls = []
            def fake_run(scene, target, reflections=False):
                calls.append(target)
                if len(calls) == 1:
                    raise KeyboardInterrupt
                (target / "metadata.json").write_text('{"status":"incomplete"}')
            with patch.object(queue.model, "load_scene", return_value={"mask_rows": ["1111"]*4}), \
                 patch.object(queue.preview, "preflight", return_value={}), patch.object(queue.preview, "run", side_effect=fake_run), \
                 patch("coded_aperture_sim.paraview_view.write_paraview_sequence"):
                self.assertEqual(queue.run_queue(config=cfg, output=root / "out"), 130)
            qdir = next((root / "out").iterdir())
            mf = json.loads((qdir / "queue.json").read_text())
            self.assertEqual(mf["jobs"][0]["status"], "interrupted")
            with patch.object(queue.preview, "preflight", return_value={}), \
                 patch.object(queue.preview, "run", side_effect=fake_run), \
                 patch("coded_aperture_sim.paraview_view.write_paraview_sequence"):
                self.assertEqual(queue.run_queue(resume=qdir), 0)
            mf = json.loads((qdir / "queue.json").read_text())
            self.assertEqual(mf["jobs"][0]["attempts"], 2)
            self.assertNotEqual(calls[0], calls[1])

    def test_keyboard_interrupt_during_sequence_keeps_completed_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg = self._config(root); calls = []
            def fake_run(scene, target, reflections=False):
                calls.append(target)
                (target / "metadata.json").write_text('{"status":"incomplete"}')
            with patch.object(queue.model, "load_scene", return_value={"mask_rows": ["1111"]*4}), \
                 patch.object(queue.preview, "preflight", return_value={}), patch.object(queue.preview, "run", side_effect=fake_run), \
                 patch("coded_aperture_sim.paraview_view.write_paraview_sequence", side_effect=[KeyboardInterrupt, None, None]):
                self.assertEqual(queue.run_queue(config=cfg, output=root / "out"), 130)
                qdir = next((root / "out").iterdir())
                manifest = json.loads((qdir / "queue.json").read_text())
                self.assertEqual(manifest["jobs"][0]["status"], "completed")
                self.assertEqual(queue.run_queue(resume=qdir), 0)
                manifest = json.loads((qdir / "queue.json").read_text())
                self.assertEqual(manifest["jobs"][0]["status"], "completed")
                self.assertEqual(len(calls), 2)

    def test_sequence_failure_keeps_completed_mask_for_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg = self._config(root); calls = []
            def fake_run(scene, target, reflections=False):
                calls.append(target)
                (target / "metadata.json").write_text('{"status":"incomplete"}')
            with patch.object(queue.model, "load_scene", return_value={"mask_rows": ["1111"]*4}), \
                 patch.object(queue.preview, "preflight", return_value={}), patch.object(queue.preview, "run", side_effect=fake_run), \
                 patch("coded_aperture_sim.paraview_view.write_paraview_sequence", side_effect=[RuntimeError("pv failed"), None, None]):
                self.assertEqual(queue.run_queue(config=cfg, output=root / "out"), 1)
                qdir = next((root / "out").iterdir())
                manifest = json.loads((qdir / "queue.json").read_text())
                self.assertEqual(manifest["jobs"][0]["status"], "completed")
                self.assertEqual(manifest["jobs"][0]["scientific_status"], "incomplete")
                self.assertEqual(queue.run_queue(resume=qdir), 0)
                self.assertEqual(len(calls), 2)
                manifest = json.loads((qdir / "queue.json").read_text())
                self.assertEqual(Path(manifest["jobs"][0]["output_path"]), calls[0])
                self.assertEqual(Path(manifest["jobs"][1]["output_path"]), calls[1])


if __name__ == "__main__":
    unittest.main()
