import json
import builtins
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import h5py

from coded_aperture_sim import model, preview
from coded_aperture_sim.export_transverse_csv import main as export_transverse_csv


class PreviewTests(unittest.TestCase):
    def test_transverse_csv_cli_exports_full_slices_and_preserves_existing_files(self):
        scene = model.load_scene()
        scene["preview_depths_mm"] = [25.5, 75]
        coords = model.coordinates(scene)
        values = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
            with h5py.File(run_dir / "volume.h5", "w") as h5:
                h5["x_mm"], h5["y_mm"] = [-1., 1., 3.], [-2., 2.]
                h5["z_mm"] = [coords["mask_back"] + 25.5, coords["mask_back"] + 74.5]
                h5.attrs["complete"] = True
                h5["relative_intensity"] = values
            self.assertEqual(export_transverse_csv([str(run_dir)]), 0)
            raw = np.genfromtxt(run_dir / "transverse_25.5.csv", delimiter=",", names=True)
            normalized = np.genfromtxt(run_dir / "transverse_25.5_normalized.csv", delimiter=",", names=True)
            self.assertEqual(raw.dtype.names, ("x_mm", "y_mm", "relative_intensity"))
            self.assertEqual(len(raw), 6)
            self.assertEqual(float(raw[0]["relative_intensity"]), 0.)
            self.assertEqual(float(normalized["peak_normalized_intensity"].max()), 1.)
            edge = np.genfromtxt(run_dir / "transverse_75.csv", delimiter=",", names=True)
            self.assertEqual(float(edge[0]["relative_intensity"]), 6.)
            saved = run_dir / "transverse_25.5_normalized.csv"
            saved.write_text("keep me\n", encoding="utf-8")
            self.assertEqual(export_transverse_csv([str(run_dir)]), 0)
            self.assertEqual(saved.read_text(encoding="utf-8"), "keep me\n")

    def test_saved_beam_views_peak_normalize_without_changing_source(self):
        scene = model.load_scene()
        scene["preview_depths_mm"] = [25.5, 75]
        coords = model.coordinates(scene)
        values = np.zeros((2, 3, 3), dtype=np.float32)
        values[0] = [[0, 1, 0], [2, 3, 4], [0, 1, 0]]
        x = y = np.array([-1., 0., 1.])
        z = coords["mask_back"] + np.array([25.5, 75.])
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
            with h5py.File(run_dir / "volume.h5", "w") as h5:
                h5["x_mm"], h5["y_mm"], h5["z_mm"] = x, y, z
                h5["relative_intensity"] = values
            with h5py.File(run_dir / "volume.h5", "r") as h5:
                source_before = h5["relative_intensity"][:]

            captured = []
            saved_views = {}
            imshow = preview.plt.Axes.imshow
            savefig = preview.plt.Figure.savefig
            def capture(self, image, *args, **kwargs):
                captured.append(np.array(image, copy=True))
                return imshow(self, image, *args, **kwargs)
            def capture_save(self, path, *args, **kwargs):
                ax = self.axes[0]
                saved_views[Path(path).name] = (ax.get_xlim(), ax.get_ylim(), ax.get_title())
                return savefig(self, path, *args, **kwargs)
            with patch.object(preview.plt.Axes, "imshow", autospec=True, side_effect=capture), \
                    patch.object(preview.plt.Figure, "savefig", autospec=True, side_effect=capture_save):
                preview._write_beam_views(run_dir)
                preview._write_beam_views(run_dir, normalized_only=True)

            self.assertTrue((run_dir / "transverse_25.5.png").exists())
            for name in ("transverse_25.5_radar.png", "transverse_75_radar.png",
                         "transverse_25.5_normalized_radar.png", "transverse_75_normalized_radar.png"):
                self.assertTrue((run_dir / name).exists())
            for name, (xlim, ylim, title) in saved_views.items():
                if name.endswith("_radar.png"):
                    self.assertGreater(xlim[0], xlim[1])
                    self.assertIn("Radar view", title)
                else:
                    self.assertLess(xlim[0], xlim[1])
                self.assertLess(ylim[0], ylim[1])
            self.assertTrue((run_dir / "cut_x_25.5.csv").exists())
            self.assertTrue((run_dir / "transverse_25.5_normalized.png").exists())
            self.assertTrue((run_dir / "cut_x_25.5_normalized.csv").exists())
            self.assertTrue((run_dir / "transverse_75.png").exists())
            normalized_x = np.genfromtxt(run_dir / "cut_x_25.5_normalized.csv", delimiter=",", skip_header=1)[:, 1]
            normalized_y = np.genfromtxt(run_dir / "cut_y_25.5_normalized.csv", delimiter=",", skip_header=1)[:, 1]
            self.assertEqual(float(normalized_x.max()), 1.)
            self.assertEqual(float(normalized_y.max()), 1.)
            self.assertTrue(np.isfinite(normalized_x).all())
            self.assertTrue(np.isfinite(normalized_y).all())
            for zero_cut in ("cut_x_75_normalized.csv", "cut_y_75_normalized.csv"):
                zero_values = np.genfromtxt(run_dir / zero_cut, delimiter=",", skip_header=1)[:, 1]
                self.assertTrue(np.isfinite(zero_values).all())
                self.assertEqual(float(zero_values.max()), 0.)
            self.assertAlmostEqual(float(captured[2].max()), 1.)
            self.assertTrue(np.isfinite(captured[3]).all())
            self.assertEqual(float(captured[3].max()), 0.)
            with h5py.File(run_dir / "volume.h5", "r") as h5:
                np.testing.assert_array_equal(h5["relative_intensity"][:], source_before)

    def test_hertzian_source_phase_is_outgoing_and_symmetric(self):
        x = np.array([-2., 2.]); y = np.array([0.])
        field = preview._source(x, y, 100., 60.5)[0]
        np.testing.assert_allclose(field[0], field[1])
        k = 2*np.pi*60.5e9/299_792_458
        near, far = preview._source(np.array([0.]), np.array([0.]), 100., 60.5)[0,0], preview._source(np.array([0.]), np.array([0.]), 101., 60.5)[0,0]
        self.assertAlmostEqual(np.angle(far/near), -k*.001, places=3)

    def test_mask_raster_uses_positive_y_for_row_zero(self):
        scene = model.load_scene()
        scene["mask_rows"] = ["1000", "0000", "0000", "0000"]
        x, y = np.array([-11.4, -3.8]), np.array([13.3, 5.7])
        mask = preview._screen_mask(x, y, scene)
        self.assertTrue(mask[0, 0]); self.assertFalse(mask[0, 1]); self.assertFalse(mask[1, 0])
        edge = preview._screen_mask(np.array([-13.25, 13.25]), np.array([0.]), scene, first=True)
        self.assertTrue(edge[0, 0]); self.assertFalse(edge[0, 1])
        phi = np.arange(180)*np.deg2rad(2)
        self.assertEqual(np.rad2deg(phi[preview._phi_index(phi, 90)]), 90.)

    def test_preview_depth_defaults_and_validation(self):
        scene = model.load_scene()
        self.assertEqual(preview._depths(scene), (50., 250., 500.))
        scene.pop("preview_depths_mm")
        self.assertEqual(preview._depths(scene), (50., 250., 500.))
        for depths in ([], [0], [-1], [501], [True], [float("nan")], [50, 50]):
            with self.subTest(depths=depths):
                scene["preview_depths_mm"] = depths
                with self.assertRaises(ValueError):
                    preview._depths(scene)

    def test_run_uses_custom_depths_for_convergence_and_reports_volume(self):
        scene = model.load_scene()
        scene.update(volume_size_mm=[4, 4, 2], volume_step_mm=2, capture_size_mm=[40, 40],
                     capture_step_mm=2, padding=1, preview_depths_mm=[0.5, 1.5])
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def prop(_prepared, depth, *_args):
                calls.append(depth)
                return np.ones((2, 2), dtype=complex)
            output = io.StringIO()
            with patch("coded_aperture_sim.preview.preflight", return_value={}), \
                 patch("coded_aperture_sim.preview._chain", side_effect=[object(), object(), object()]), \
                 patch("coded_aperture_sim.preview._prop", side_effect=prop), \
                 patch("coded_aperture_sim.preview._farfield"), \
                 patch("coded_aperture_sim.preview._write_beam_views"), \
                 patch("coded_aperture_sim.paraview_view.write_paraview_files"), \
                 contextlib.redirect_stdout(output):
                result = preview.run(scene, Path(tmp))
            metadata = json.loads((result / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual([row["depth_from_mask_back_mm"] for row in metadata["convergence"]], [0.5, 1.5])
        self.assertTrue(all(calls.count(depth) >= 3 for depth in (0.5, 1.5)))
        self.assertIn("[preview] volume: 4 x 4 x 2 mm (x x y x z); sampling: 2 mm", output.getvalue())

    def test_tiny_closed_mask_cli_without_solver_bindings(self):
        scene = model.load_scene()
        scene.update(volume_size_mm=[4, 4, 2], volume_step_mm=2, capture_size_mm=[40, 40], capture_step_mm=2,
                     mask_rows=["0000"]*4, padding=1, preview_depths_mm=[1, 2])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            cfg = Path(tmp)/"scene.json"; cfg.write_text(json.dumps(scene), encoding="utf-8")
            real_import = builtins.__import__
            def no_solver(name, *args, **kwargs):
                if name.split(".")[0] in {"openEMS", "CSXCAD"}:
                    raise AssertionError(f"preview tried importing {name}")
                return real_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=no_solver), patch("coded_aperture_sim.preview.find_pvpython", return_value=Path("pvpython")), patch("coded_aperture_sim.paraview_view.write_paraview_files") as writer:
                self.assertEqual(__import__("coded_aperture_sim.__main__", fromlist=["main"]).main(
                    ["preview", "--config", str(cfg), "--output", str(Path(tmp)/"runs")]), 0)
                run_dir = next((Path(tmp)/"runs").iterdir())
                with h5py.File(run_dir/"volume.h5", "r") as h5:
                    self.assertEqual(h5["U"].shape, (1,2,2))
                    self.assertTrue(np.isfinite(h5["relative_intensity"][:]).all())
                    self.assertEqual(float(h5["relative_intensity"][:].max()), 0.)
                    self.assertTrue(h5.attrs["complete"])
                self.assertTrue(json.loads((run_dir/"metadata.json").read_text())["zero_transmission"])
                scene["mask_rows"] = ["1000", "0000", "0000", "0000"]
                cfg.write_text(json.dumps(scene), encoding="utf-8")
                self.assertEqual(__import__("coded_aperture_sim.__main__", fromlist=["main"]).main(
                    ["preview", "--config", str(cfg), "--output", str(Path(tmp)/"open_runs")]), 0)
            opened = next((Path(tmp)/"open_runs").iterdir())
            with h5py.File(opened/"volume.h5", "r") as h5:
                self.assertEqual(h5["U"].shape, (1,2,2))
                np.testing.assert_allclose(h5["x_mm"][:], [-1., 1.])
                np.testing.assert_allclose(h5["y_mm"][:], [-1., 1.])
                self.assertAlmostEqual(float(h5["relative_intensity"][:].max()), 1., places=6)
            self.assertEqual(writer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
