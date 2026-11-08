import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from coded_aperture_sim.montage_transverse import create_montage


class TransverseMontageTests(unittest.TestCase):
    def test_moved_queue_is_cropped_and_ordered_by_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue_dir = Path(tmp)
            jobs = []
            for window in reversed(range(24)):
                run_name = f"run_{window:02d}"
                run_dir = queue_dir / run_name
                run_dir.mkdir()
                image = Image.new("RGB", (10, 8), "white")
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1, 7), fill=(0, 0, 0))
                draw.rectangle((3, 0, 6, 0), fill=(100, 100, 100))  # grayscale title
                color = (200, 20, 20 + window)
                draw.rectangle((3, 2, 6, 4), fill=color)
                draw.rectangle((8, 2, 8, 4), fill=(20, 200, 200))  # smaller colorbar
                image.save(run_dir / "transverse_1250_normalized.png")
                jobs.append({"window": window, "status": "completed",
                             "output_path": f"C:\\old\\queue\\{run_name}"})
            (queue_dir / "queue.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")

            output = create_montage(queue_dir)
            montage = Image.open(output)
            self.assertEqual(montage.size, (76, 102))
            for window in range(24):
                col, row = window % 4, window // 4
                x, y = 12 + col * 16 + 2, 12 + row * 15 + 1
                self.assertEqual(montage.getpixel((x, y)), (200, 20, 20 + window))


if __name__ == "__main__":
    unittest.main()
