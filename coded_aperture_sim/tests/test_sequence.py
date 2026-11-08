import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

from coded_aperture_sim.paraview_view import write_paraview_sequence


class SequenceTests(unittest.TestCase):
    def test_pvd_collections_are_ordered_relative_and_xml_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "queue"
            runs = [(4, root / "run & <four>"), (1, root / "run one")]
            with patch("coded_aperture_sim.paraview_view.find_pvpython", return_value=Path("pvpython")), \
                 patch.object(subprocess, "run"):
                write_paraview_sequence(queue, runs)
            for kind, dataset in (("volume", "volume.vti"), ("geometry", "geometry.vtm"), ("farfield", "farfield.vtp")):
                root_xml = ET.parse(queue / f"{kind}.pvd").getroot()
                entries = root_xml.findall("./Collection/DataSet")
                self.assertEqual([entry.attrib["timestep"] for entry in entries], ["1", "4"])
                self.assertEqual([entry.attrib["file"] for entry in entries], [
                    "../run one/" + dataset, "../run & <four>/" + dataset])

    def test_empty_sequence_does_nothing(self):
        with patch("coded_aperture_sim.paraview_view.find_pvpython") as find, \
             patch.object(subprocess, "run") as run:
            write_paraview_sequence(Path("unused"), [])
        find.assert_not_called()
        run.assert_not_called()

