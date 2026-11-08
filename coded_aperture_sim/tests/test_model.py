import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model
import solver


class SceneTests(unittest.TestCase):
    def test_default_planes_and_mask_orientation(self):
        scene = model.load_scene()
        z = model.coordinates(scene)
        self.assertEqual((z["plate_front"], z["plate_back"], z["mask_front"], z["mask_back"], z["handoff_z"]),
                         (41.9, 43.9, 46.9, 49.9, 54.9))
        boxes = {b["name"]: b for b in model.geometry(scene)}
        self.assertNotIn("mask_r0c0", boxes)  # top-left is open
        self.assertIn("mask_r0c1", boxes)     # top row runs from +y, left to right
        self.assertNotIn("mask_r3c3", boxes)  # bottom-right is open

    def test_frame_and_joined_metal_cells(self):
        scene = model.load_scene()
        boxes = {b["name"]: b for b in model.geometry(scene)}
        self.assertEqual(boxes["mask_left"]["start"][:2], [-100.0, -21.5])
        # Adjacent closed cells are contiguous, with a shared edge and identical z span.
        a, b = boxes["mask_r2c2"], boxes["mask_r2c3"]
        self.assertAlmostEqual(a["stop"][0], b["start"][0])
        self.assertEqual(a["start"][2], b["start"][2])

    def test_rejects_bad_mask(self):
        path = Path(__file__).with_name("_invalid_scene.json")
        try:
            path.write_text(json.dumps({"mask_rows": ["10", "0000", "0000", "0000"]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                model.load_scene(path)
        finally:
            path.unlink(missing_ok=True)

    def test_all_open_reference_keeps_same_mesh(self):
        scene = model.load_scene()
        reference = dict(scene, mask_rows=["1111"] * 4)
        for actual, open_grid in zip(solver._mesh(scene), solver._mesh(reference)):
            self.assertTrue((actual == open_grid).all())

    def test_mesh_uses_configured_bulk_spacing_and_retains_fine_spacing(self):
        scene = model.load_scene()
        axes = solver._mesh(scene)
        spacings = [abs(line_b-line_a) for lines in axes for line_a, line_b in zip(lines[:-1], lines[1:])]
        self.assertGreater(max(spacings), 0.375)
        self.assertLessEqual(max(spacings), scene["mesh_max_mm"] + 1e-9)
        for lines in axes:
            delta = abs(lines[1:] - lines[:-1])
            self.assertTrue(all(value in lines for value in (-0.25, 0.0, 0.25)))
            self.assertTrue(any(abs(value-scene["mesh_detail_mm"]) < 1e-9 for value in delta))
            ratios = delta[1:] / delta[:-1]
            self.assertLessEqual(max(ratios.max(), (1 / ratios).max()), 1.50001)

    def test_lowres_mesh_spacing_and_rear_nf_anchor(self):
        scene = model.load_scene(Path(model.__file__).with_name("scene_lowres.json"))
        axes = solver._mesh(scene)
        for lines in axes:
            spacing = [b-a for a, b in zip(lines[:-1], lines[1:])]
            self.assertLessEqual(max(spacing), scene["mesh_max_mm"] + 1e-9)
            for a, b in zip(spacing[:-1], spacing[1:]):
                self.assertLessEqual(max(a/b, b/a), 1.50001)
        self.assertTrue(any(abs(value + 9.6) < 1e-8 for value in axes[2]))
        self.assertLess(axes[0][0], -scene["capture_size_mm"][0]/2)
        self.assertGreater(axes[0][-1], scene["capture_size_mm"][0]/2)
        self.assertLess(axes[1][0], -scene["capture_size_mm"][1]/2)
        self.assertGreater(axes[1][-1], scene["capture_size_mm"][1]/2)
        self.assertLess(axes[2][0], -9.6)
        self.assertGreater(axes[2][-1], model.coordinates(scene)["mask_back"] + 30)

    def test_default_enclosure_bounds_and_aperture(self):
        scene = model.load_scene()
        boxes = {b["name"]: b for b in model.geometry(scene)}
        for name, start, stop in (
            ("enclosure_left", [-50.0, -60.0], [-48.0, 40.0]),
            ("enclosure_right", [48.0, -60.0], [50.0, 40.0]),
            ("enclosure_bottom", [-48.0, -60.0], [48.0, -58.0]),
            ("enclosure_top", [-48.0, 38.0], [48.0, 40.0]),
        ):
            self.assertEqual(boxes[name]["start"][:2], start)
            self.assertEqual(boxes[name]["stop"][:2], stop)
            self.assertAlmostEqual(boxes[name]["start"][2], -3.1)
            self.assertAlmostEqual(boxes[name]["stop"][2], 41.9)
        self.assertEqual(boxes["enclosure_back"]["start"][:2], [-50.0, -60.0])
        self.assertEqual(boxes["enclosure_back"]["stop"][:2], [50.0, 40.0])
        self.assertAlmostEqual(boxes["enclosure_back"]["start"][2], -4.6)
        self.assertAlmostEqual(boxes["enclosure_back"]["stop"][2], -3.1)
        front = [boxes[f"plate_{side}"] for side in ("left", "right", "bottom", "top")]
        self.assertEqual([min(b["start"][axis] for b in front) for axis in range(2)], [-50.0, -60.0])
        self.assertEqual([max(b["stop"][axis] for b in front) for axis in range(2)], [50.0, 40.0])
        self.assertAlmostEqual(min(b["start"][2] for b in front), 41.9)
        self.assertAlmostEqual(max(b["stop"][2] for b in front), 43.9)
        self.assertFalse(any(box["start"][0] < 0 < box["stop"][0] and
                             box["start"][1] < 0 < box["stop"][1] and
                             box["start"][2] <= 42.9 <= box["stop"][2]
                             for box in front))

    def test_geometry_has_no_volumetric_overlaps_and_source_is_clear(self):
        boxes = model.geometry(model.load_scene())
        for index, a in enumerate(boxes):
            for b in boxes[index + 1:]:
                overlap = [min(a["stop"][axis], b["stop"][axis]) - max(a["start"][axis], b["start"][axis])
                           for axis in range(3)]
                self.assertFalse(all(width > 1e-9 for width in overlap), (a["name"], b["name"]))
        self.assertFalse(any(all(box["start"][axis] < 0 < box["stop"][axis] for axis in range(3))
                             for box in boxes))

    def test_enclosure_legacy_partial_and_null_config(self):
        default = model.load_scene()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "scene.json"
            legacy = {key: value for key, value in default.items() if key != "enclosure"}
            legacy["plate_size_mm"] = [43, 43]
            path.write_text(json.dumps(legacy), encoding="utf-8")
            legacy_boxes = {b["name"]: b for b in model.geometry(model.load_scene(path))}
            self.assertFalse(any(name.startswith("enclosure_") for name in legacy_boxes))
            self.assertEqual(legacy_boxes["plate_left"]["start"][:2], [-21.5, -21.5])
            self.assertEqual(legacy_boxes["plate_right"]["stop"][:2], [21.5, 21.5])

            path.write_text(json.dumps({"mesh_max_mm": 0.8}), encoding="utf-8")
            partial = model.load_scene(path)
            self.assertEqual(partial["mesh_max_mm"], 0.8)
            self.assertEqual(partial["enclosure"], default["enclosure"])

            path.write_text(json.dumps({"enclosure": None}), encoding="utf-8")
            self.assertFalse(any(b["name"].startswith("enclosure_") for b in model.geometry(model.load_scene(path))))

            thicker = dict(default["enclosure"], backplate_thickness_mm=10)
            path.write_text(json.dumps({"enclosure": thicker}), encoding="utf-8")
            thick_back = {b["name"]: b for b in model.geometry(model.load_scene(path))}["enclosure_back"]
            self.assertAlmostEqual(thick_back["start"][2], -13.1)

    def test_rejects_invalid_enclosure_dimensions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "scene.json"
            defaults = model.load_scene()["enclosure"]
            for override in ({"wall_thickness_mm": 0}, {"cavity_depth_mm": -1},
                             {"backplate_thickness_mm": 0}, {"center_xy_mm": [0]},
                             {"wall_thickness_mm": 50}, {"center_xy_mm": [0, 100]},
                             {"cavity_depth_mm": 40}):
                enclosure = dict(defaults, **override)
                path.write_text(json.dumps({"enclosure": enclosure}), encoding="utf-8")
                with self.subTest(enclosure=override), self.assertRaises(ValueError):
                    model.load_scene(path)


if __name__ == "__main__":
    unittest.main()
