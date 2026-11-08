import builtins
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from coded_aperture_sim import model, preview


class ReflectionTests(unittest.TestCase):
    def test_chain_adds_one_coherent_roundtrip(self):
        scene = model.load_scene()
        scene["capture_size_mm"] = [2., 2.]
        scene["capture_step_mm"] = 2.
        distances = []
        alpha = .37

        def propagate(field, step, distance, frequency, **kwargs):
            distances.append(distance)
            return field * np.exp(-1j * alpha * distance)

        with patch.object(preview, "_source", return_value=np.ones((1, 1), complex)), \
             patch.object(preview, "_screen_mask", return_value=np.ones((1, 1), bool)), \
             patch.object(preview, "_metal_mask", return_value=np.ones((1, 1), bool)), \
             patch.object(preview, "propagate_plane", side_effect=propagate), \
             patch.object(preview, "_prepare_plane", side_effect=lambda field, *a, **k: field):
            direct = preview._chain(scene, 2., 1)
            direct_distances = distances[:]
            distances.clear()
            reflected = preview._chain(scene, 2., 1, reflections=True)

        c = model.coordinates(scene)
        gap = c["mask_front"] - c["plate_back"]
        separation = c["mask_back"] - c["plate_back"]
        roundtrip = np.exp(-2j * alpha * gap)
        self.assertEqual(direct_distances, [separation])
        self.assertEqual(distances, [separation, gap, gap, separation])
        np.testing.assert_allclose(direct, [[np.exp(-1j * alpha * separation)]])
        np.testing.assert_allclose(reflected / direct, [[1 + roundtrip]])

    def test_reflection_masks_follow_finite_metal_faces_and_holes(self):
        scene = model.load_scene()
        x = np.array([-49., -4., 0., 101.])
        y = np.array([0., 10., 0., 0.])
        plate = preview._metal_mask(x, y, scene, "plate")
        mask = preview._metal_mask(x, y, scene, "mask")
        self.assertTrue(plate[0, 0])       # Original plate/enclosure XY footprint.
        self.assertFalse(plate[0, 2])      # Plate opening.
        self.assertFalse(plate[0, 3])      # Outside the finite plate.
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])       # Open mask cell at row 1, column 1.
        self.assertFalse(mask[0, 2])       # Central aperture.
        self.assertFalse(mask[0, 3])

    def test_preview_cli_forwards_reflections_and_writes_metadata(self):
        scene = model.load_scene()
        scene.update(volume_size_mm=[4, 4, 4], volume_step_mm=2,
                     capture_size_mm=[40, 40], capture_step_mm=2, padding=1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root = Path(tmp)
            config = root / "scene.json"
            config.write_text(json.dumps(scene), encoding="utf-8")
            real_import = builtins.__import__

            def no_solver(name, *args, **kwargs):
                if name.split(".")[0] in {"openEMS", "CSXCAD"}:
                    raise AssertionError(f"preview tried importing {name}")
                return real_import(name, *args, **kwargs)

            from coded_aperture_sim.__main__ import main
            output = io.StringIO()
            with patch("builtins.__import__", side_effect=no_solver), \
                 patch("coded_aperture_sim.preview.find_pvpython", return_value=Path("pvpython")), \
                 patch("coded_aperture_sim.paraview_view.write_paraview_files"), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(main(["preview", "--config", str(config), "--output", str(root / "runs"), "--reflections"]), 0)
            run_dir = next((root / "runs").iterdir())
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["reflections"]["enabled"])
            self.assertEqual(metadata["reflections"]["round_trips"], 1)
            self.assertIn("one-round-trip reflections enabled", output.getvalue())
            self.assertIn("5/5 ParaView files", output.getvalue())
            self.assertIn("plane 2/2", output.getvalue())
            self.assertIn("complete in", output.getvalue())
            with h5py.File(run_dir / "volume.h5", "r") as h5:
                self.assertEqual(h5["U"].shape, (2, 2, 2))
                self.assertTrue(np.isfinite(h5["U"][:]).all())


if __name__ == "__main__":
    unittest.main()
