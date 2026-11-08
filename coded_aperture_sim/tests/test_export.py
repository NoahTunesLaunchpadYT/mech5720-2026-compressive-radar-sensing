"""Small synthetic-field checks for export axis order and signed Sz."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coded_aperture_sim.__main__ import run
from coded_aperture_sim.export import _field_scalars, export
from coded_aperture_sim import model


class ExportScalarTests(unittest.TestCase):
    def test_live_resource_gate_blocks_before_run_directory_or_solver(self):
        scene = model.load_scene()
        with tempfile.TemporaryDirectory(prefix=".resource_gate_", dir=Path(__file__).resolve().parent) as tmp:
            output = Path(tmp) / "runs"
            with patch("coded_aperture_sim.solver.preflight", return_value={
                    "resource_gate": False, "within_memory_limit": False,
                    "estimated_peak_gib": 19.86, "available_physical_memory_gib": 5.0}), \
                    patch("coded_aperture_sim.solver.solve") as solve:
                with self.assertRaisesRegex(MemoryError, "before output reservation"):
                    run(scene, output)
                solve.assert_not_called()
            self.assertFalse(output.exists())

    def test_xyz_components_preserve_yx_plane_order_and_signed_sz(self):
        import numpy as np

        e = np.zeros((2, 3, 3), dtype=np.complex128)
        h = np.zeros_like(e)
        e[1, 2, 0] = 2 + 0j
        h[1, 2, 1] = 3 + 0j
        e[0, 0, 0] = 1j
        h[0, 0, 1] = 2 + 0j
        relative, sz = _field_scalars(e, h, 4.0)
        self.assertEqual(relative.shape, (2, 3))
        self.assertEqual(relative[1, 2], 1.0)
        self.assertEqual(sz[1, 2], 3.0)
        self.assertEqual(sz[0, 0], 0.0)

    def test_tiny_synthetic_run_exports_axes_and_binary_views(self):
        import h5py
        import numpy as np

        scene = model.load_scene()
        scene["volume_size_mm"] = [2, 2, 2]
        scene["volume_step_mm"] = 1
        with tempfile.TemporaryDirectory(prefix=".coded_aperture_synthetic_", dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            (root / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
            x = y = np.linspace(-2, 2, 5)
            z = np.array([49.9, 50.9, 51.9, 54.9, 60.0])
            e = np.zeros((len(z), len(y), len(x), 3), dtype=np.complex64)
            h = np.zeros_like(e)
            e[..., 0] = np.arange(len(x), dtype=np.float32)[None, None, :] + 1
            h[..., 1] = 2
            with h5py.File(root / "fields.h5", "w") as fields:
                for name, value in (("x_mm", x), ("y_mm", y), ("z_mm", z)):
                    fields.create_dataset(name, data=value)
                fields.create_dataset("E", data=e)
                fields.create_dataset("H", data=h)
                fields.attrs["handoff_index"] = 3
            theta, phi = np.linspace(0, np.pi, 5), np.linspace(0, 2*np.pi, 8, endpoint=False)
            np.savez(root / "farfield.npz", theta_rad=theta, phi_rad=phi, power=np.ones((len(theta), len(phi))))
            export(root)
            with h5py.File(root / "volume.h5", "r") as volume:
                self.assertEqual(volume["E"].shape, (2, 2, 2, 3))
                self.assertEqual(volume["relative_E2"].shape, (2, 2, 2))
                self.assertEqual(tuple(volume["x_mm"][:]), (-0.5, 0.5))
                self.assertGreater(float(volume["Sz_relative"][0, 0, 0]), 0)
            for filename in ("volume.vti", "geometry.vtm", "farfield.vtp", "view.pvsm", "view.png", "cut_x.csv", "cut_y.csv", "farfield_phi0.csv", "farfield_phi90.csv"):
                self.assertGreater((root / filename).stat().st_size, 0, filename)


if __name__ == "__main__":
    unittest.main()
