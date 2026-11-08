import unittest
import tempfile
import os
from pathlib import Path

import h5py
import numpy as np

from coded_aperture_sim.model import coordinates, load_scene
from coded_aperture_sim.propagation import _prepare_plane, _propagate_prepared, iter_volume, propagate_plane, validate_handoff


class PropagationTests(unittest.TestCase):
    def test_cached_propagation_matches_wrapper(self):
        rng = np.random.default_rng(41)
        for shape in ((5, 8), (6, 7)):
            for field in (rng.normal(size=shape) + 1j*rng.normal(size=shape),
                          rng.normal(size=shape + (3,)) + 1j*rng.normal(size=shape + (3,))):
                prepared = _prepare_plane(field, (0.7, 0.9), 60.5, padding=2)
                original_spectrum = prepared["spectrum"].copy()
                for distance in (0.0, 2.3):
                    cached = _propagate_prepared(prepared, distance)
                    wrapped = propagate_plane(field, (0.7, 0.9), distance, 60.5, padding=2)
                    relative_l2 = np.linalg.norm((cached-wrapped).ravel()) / np.linalg.norm(wrapped.ravel())
                    self.assertLessEqual(relative_l2, 1e-10)
                np.testing.assert_array_equal(prepared["spectrum"], original_spectrum)

    def test_zero_distance_is_identity(self):
        rng = np.random.default_rng(4)
        field = rng.normal(size=(7, 8, 3)) + 1j * rng.normal(size=(7, 8, 3))
        np.testing.assert_allclose(propagate_plane(field, 0.7, 0, 60.5, bandlimit=False), field, atol=1e-12)

    def test_forward_oblique_wave_phase(self):
        nx, ny, dx = 32, 24, 1.0
        freq = 60.5
        wavelength_mm = 299_792_458 / (freq * 1e9) * 1000
        fx, fy = 4 / (nx * dx), 3 / (ny * dx)
        kx, ky, k = 2*np.pi*fx, 2*np.pi*fy, 2*np.pi/wavelength_mm
        kz = np.sqrt(k*k-kx*kx-ky*ky)
        x = (np.arange(nx)-(nx-1)/2)*dx
        y = (np.arange(ny)-(ny-1)/2)*dx
        xx, yy = np.meshgrid(x, y)
        field = np.exp(-1j*(kx*xx+ky*yy))
        z = 2.3
        out = propagate_plane(field, dx, z, freq, padding=1, bandlimit=False)
        np.testing.assert_allclose(out, field*np.exp(-1j*kz*z), atol=1e-12)

    def test_diagonal_alias_mode_rejected_by_coupled_bandlimit(self):
        n, dx, z, freq = 64, 1.0, 20.0, 60.5
        fbin = 9/(n*dx)
        x = (np.arange(n)-(n-1)/2)*dx
        xx, yy = np.meshgrid(x, x)
        diagonal = np.exp(-2j*np.pi*fbin*(xx+yy))
        filtered = propagate_plane(diagonal, dx, z, freq, padding=1, bandlimit=True)
        unfiltered = propagate_plane(diagonal, dx, z, freq, padding=1, bandlimit=False)
        self.assertLess(np.max(np.abs(filtered)), 1e-12)
        np.testing.assert_allclose(np.abs(unfiltered), 1, atol=1e-12)

    def test_evanescent_wave_decays(self):
        n, dx, freq = 16, 20.0, 1.0
        x = np.arange(n)*dx
        fx = 5/(n*dx)
        field = np.exp(2j*np.pi*fx*x)[None, :]
        k = 2*np.pi*freq*1e9/299_792_458
        kx = 2*np.pi*fx*1e3
        kappa = np.sqrt(kx*kx-k*k)
        distance = 10.0
        out = propagate_plane(field, dx, distance, freq, padding=1, bandlimit=False)
        np.testing.assert_allclose(out, field*np.exp(-kappa*distance*1e-3), atol=1e-12)

    def test_e_and_h_components_keep_relative_phase(self):
        eta = 376.730313668
        e = np.zeros((9, 9, 3), dtype=complex)
        e[..., 0] = 1
        h = np.zeros_like(e)
        h[..., 1] = 1/eta
        pair = np.concatenate((e, h), axis=-1)
        out = propagate_plane(pair, 0.5, 3, 60.5, bandlimit=False)
        np.testing.assert_allclose(out[..., 4], out[..., 0]/eta, atol=1e-12)
        self.assertGreater(np.mean(np.real(np.cross(out[..., :3], np.conj(out[..., 3:]))[..., 2])), 0)

    def test_invalid_inputs_rejected(self):
        for kwargs in ({"spacing_mm": 0}, {"distance_mm": -1}, {"frequency_ghz": np.inf}, {"padding": 0}):
            args = dict(spacing_mm=1, distance_mm=1, frequency_ghz=60.5, padding=4)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                propagate_plane(np.ones((2, 2), complex), **args)
        with self.assertRaises(ValueError):
            propagate_plane(np.ones((2, 2)), 1, 0, 60.5)

    def _write_fields(self, path, scene, z, x, y, direct_phase_error=False):
        zi = coordinates(scene)["handoff_z"]
        handoff = int(np.flatnonzero(np.isclose(z, zi, atol=1e-9))[0])
        shape = (len(z), len(y), len(x), 3)
        with h5py.File(path, "w") as h5:
            for name, values in (("x_mm", x), ("y_mm", y), ("z_mm", z)):
                h5[name] = values
            h5.attrs["handoff_index"] = handoff
            h5.create_dataset("E", shape=shape, dtype=np.complex128)
            h5.create_dataset("H", shape=shape, dtype=np.complex128)
            if direct_phase_error:
                target = int(np.flatnonzero(np.isclose(z, coordinates(scene)["mask_back"]+10, atol=1e-9))[0])
                h5["E"][target, ..., 0] = 1j

    def test_handoff_validation_rejects_missing_plane_and_phase_error(self):
        scene = load_scene()
        scene["volume_size_mm"] = [4, 4, 20]
        x = np.arange(-9.5, 10, 1)
        y = x.copy()
        mask_back = coordinates(scene)["mask_back"]
        handoff_z = coordinates(scene)["handoff_z"]
        complete = np.arange(handoff_z-1, mask_back+30.01, 0.5)
        missing = complete[~np.isclose(complete, mask_back+20, atol=1e-9)]
        paths = []
        for _ in range(2):
            fd, filename = tempfile.mkstemp(suffix=".h5", dir=Path(__file__).parent)
            os.close(fd)
            paths.append(Path(filename))
        try:
            path = paths[0]
            self._write_fields(path, scene, missing, x, y)
            result = validate_handoff(path, scene)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any(p["status"] == "missing_direct_plane" for p in result["planes"]))
            path = paths[1]
            self._write_fields(path, scene, complete, x, y, direct_phase_error=True)
            result = validate_handoff(path, scene)
            self.assertEqual(result["status"], "failed")
            self.assertGreater(result["planes"][0]["E_relative_l2"], 0.05)
        finally:
            for path in paths:
                path.unlink(missing_ok=True)

    def test_iter_volume_streams_cell_centers_through_handoff(self):
        scene = load_scene()
        scene.update(source_to_plate_mm=41.5, handoff_mm=4.75, volume_size_mm=[2, 2, 6], volume_step_mm=0.5)
        x = np.arange(-2, 2.1, 1)
        y = x.copy()
        z = np.arange(49.5, 55.51, 0.25)
        fd, filename = tempfile.mkstemp(suffix=".h5", dir=Path(__file__).parent)
        os.close(fd)
        path = Path(filename)
        try:
            self._write_fields(path, scene, z, x, y)
            with h5py.File(path, "a") as h5:
                h5["E"][:, ..., 0] = 1
                h5["H"][:, ..., 1] = 1
            slices = list(iter_volume(path, scene))
            self.assertEqual(len(slices), 12)
            self.assertTrue(all(e.shape == (4, 4, 3) and h.shape == (4, 4, 3) for _, e, h in slices))
            handoff_plane = next((e, h) for z_mm, e, h in slices if np.isclose(z_mm, coordinates(scene)["handoff_z"]))
            np.testing.assert_allclose(handoff_plane[0][..., 0], 1)
            np.testing.assert_allclose(handoff_plane[1][..., 1], 1)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
