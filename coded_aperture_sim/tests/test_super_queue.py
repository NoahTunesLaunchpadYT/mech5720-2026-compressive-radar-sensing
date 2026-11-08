import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coded_aperture_sim import super_queue


class SuperQueueTests(unittest.TestCase):
    def _inputs(self, root, windows=(0,)):
        (root / "scene.json").write_text("{}", encoding="utf-8")
        children = []
        for name in ("a.json", "b.json"):
            (root / name).write_text(json.dumps({"scene_config": "scene.json", "strip_rows": ["0" * 27] * 4,
                                                 "windows": list(windows)}), encoding="utf-8")
            children.append(name)
        parent = root / "super.json"
        parent.write_text(json.dumps({"queues": children}), encoding="utf-8")
        return parent, children

    def _patches(self, run_side_effect=None):
        return (
            patch("coded_aperture_sim.queue.model.load_scene", return_value={"mask_rows": ["1111"] * 4}),
            patch("coded_aperture_sim.preview.preflight", return_value={}),
            patch("coded_aperture_sim.preview.run", side_effect=run_side_effect),
            patch("coded_aperture_sim.paraview_view.write_paraview_sequence"),
        )

    def test_order_duplicates_snapshots_and_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._inputs(root)
            config = root / "super.json"
            config.write_text(json.dumps({"queues": ["a.json", "b.json", "a.json"]}), encoding="utf-8")
            output = root / "out"
            stdout = io.StringIO()
            with contextlib.ExitStack() as stack:
                for p in self._patches(lambda scene, target, reflections=False: (target / "metadata.json").write_text('{"status":"incomplete"}')):
                    stack.enter_context(p)
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(super_queue.run_super_queue(config=config, output=output), 0)
            parent = next(output.iterdir())
            manifest = json.loads((parent / "super_queue.json").read_text())
            self.assertEqual([e["config_path"] for e in manifest["entries"]], [str((root / x).resolve()) for x in ("a.json", "b.json", "a.json")])
            self.assertEqual([e["child_dir"] for e in manifest["entries"]], ["queue_001", "queue_002", "queue_003"])
            self.assertTrue(all(e["status"] == "completed" for e in manifest["entries"]))
            text = stdout.getvalue()
            self.assertIn("child 1/3", text)
            self.assertIn("child 3/3", text)
            self.assertIn("--resume", text)

    def test_failure_stops_later_masks_and_children_then_resume_uses_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _ = self._inputs(root, windows=(0, 1))
            output = root / "out"
            n = 0

            def run(scene, target, reflections=False):
                nonlocal n
                n += 1
                if n == 1:
                    raise RuntimeError("expected")
                (target / "metadata.json").write_text('{"status":"incomplete"}', encoding="utf-8")

            with contextlib.ExitStack() as stack:
                for p in self._patches(run):
                    stack.enter_context(p)
                self.assertEqual(super_queue.run_super_queue(config=config, output=output), 1)
                parent = next(output.iterdir())
                manifest = json.loads((parent / "super_queue.json").read_text())
                self.assertEqual(manifest["entries"][0]["status"], "failed")
                self.assertEqual(manifest["entries"][1]["status"], "pending")
                child = parent / "queue_001"
                child_manifest = json.loads((child / "queue.json").read_text())
                self.assertEqual(child_manifest["jobs"][1]["attempts"], 0)
                config.unlink()
                (root / "a.json").unlink()
                (root / "b.json").unlink()
                self.assertEqual(super_queue.run_super_queue(resume=parent), 0)
                self.assertEqual(n, 5)
                manifest = json.loads((parent / "super_queue.json").read_text())
                self.assertTrue(all(e["status"] == "completed" for e in manifest["entries"]))

    def test_keyboard_interrupt_resume_skips_completed_masks_and_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _ = self._inputs(root, windows=(0, 1))
            output = root / "out"
            calls = 0

            def run(scene, target, reflections=False):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise KeyboardInterrupt
                (target / "metadata.json").write_text('{"status":"incomplete"}', encoding="utf-8")

            with contextlib.ExitStack() as stack:
                for p in self._patches(run):
                    stack.enter_context(p)
                self.assertEqual(super_queue.run_super_queue(config=config, output=output), 130)
                parent = next(output.iterdir())
                self.assertEqual(super_queue.run_super_queue(resume=parent), 0)
                self.assertEqual(calls, 5)  # completed first mask was not repeated
                manifest = json.loads((parent / "super_queue.json").read_text())
                self.assertTrue(all(e["status"] == "completed" for e in manifest["entries"]))
                self.assertEqual(super_queue.run_super_queue(resume=parent), 0)
                self.assertEqual(calls, 5)  # completed child queues were skipped

    def test_export_interrupt_retains_completed_mask_and_stale_parent_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _ = self._inputs(root, windows=(0,))
            output = root / "out"
            calls = []

            def run(scene, target, reflections=False):
                calls.append(target)
                (target / "metadata.json").write_text('{"status":"incomplete"}', encoding="utf-8")

            exports = 0

            def export(*args):
                nonlocal exports
                exports += 1
                if exports == 1:
                    raise KeyboardInterrupt

            with patch("coded_aperture_sim.queue.model.load_scene", return_value={"mask_rows": ["1111"] * 4}), \
                 patch("coded_aperture_sim.preview.preflight", return_value={}), \
                 patch("coded_aperture_sim.preview.run", side_effect=run), \
                 patch("coded_aperture_sim.paraview_view.write_paraview_sequence", side_effect=export):
                self.assertEqual(super_queue.run_super_queue(config=config, output=output), 130)
                parent = next(output.iterdir())
                child = parent / "queue_001"
                child_manifest = json.loads((child / "queue.json").read_text())
                self.assertEqual(child_manifest["jobs"][0]["status"], "completed")
                manifest_path = parent / "super_queue.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["entries"][0]["status"] = "running"  # stale parent state after child finished
                manifest["entries"][0]["error"] = None
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertEqual(super_queue.run_super_queue(resume=parent), 0)
                self.assertEqual(len(calls), 2)
                manifest = json.loads(manifest_path.read_text())
                manifest["entries"][0]["status"] = "running"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertEqual(super_queue.run_super_queue(resume=parent), 0)
                self.assertEqual(len(calls), 2)

    def test_bad_paths_and_queue_lists_fail_before_previews(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "super.json"
            for queues in ([], ["missing.json"], [3]):
                config.write_text(json.dumps({"queues": queues}), encoding="utf-8")
                with patch("coded_aperture_sim.preview.preflight") as preflight, self.assertRaises((ValueError, FileNotFoundError)):
                    super_queue.run_super_queue(config=config, output=root / "out")
                preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
