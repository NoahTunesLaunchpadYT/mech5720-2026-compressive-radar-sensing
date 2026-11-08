"""Scene configuration and exact PEC geometry for the coded-aperture model."""
import json
import math
from pathlib import Path


DEFAULT_SCENE = Path(__file__).with_name("scene.json")
REQUIRED = {
    "frequency_ghz", "source_to_plate_mm", "plate_size_mm", "opening_size_mm",
    "plate_thickness_mm", "gap_mm", "mask_size_mm", "mask_thickness_mm",
    "pitch_mm", "mask_rows", "volume_size_mm", "volume_step_mm",
    "capture_size_mm", "capture_step_mm", "handoff_mm", "mesh_max_mm",
    "mesh_detail_mm", "memory_limit_gib", "padding",
}
ENCLOSURE_KEYS = {"center_xy_mm", "wall_thickness_mm", "cavity_depth_mm", "backplate_thickness_mm"}


def load_scene(path=None):
    """Load defaults, optionally overlaying a JSON file; reject incomplete/invalid values."""
    with DEFAULT_SCENE.open(encoding="utf-8") as stream:
        scene = json.load(stream)
    if path is not None:
        with Path(path).open(encoding="utf-8") as stream:
            override = json.load(stream)
        if not isinstance(override, dict):
            raise ValueError("scene override must be a JSON object")
        unknown = override.keys() - REQUIRED - {"enclosure", "preview_depths_mm"}
        if unknown:
            raise ValueError(f"unknown scene fields: {', '.join(sorted(unknown))}")
        scene.update(override)
        if REQUIRED <= override.keys() and "enclosure" not in override:
            scene["enclosure"] = None
    missing = REQUIRED - scene.keys()
    if missing:
        raise ValueError(f"missing scene fields: {', '.join(sorted(missing))}")
    positive = REQUIRED - {"mask_rows"}
    for key in positive - {"plate_size_mm", "opening_size_mm", "mask_size_mm", "volume_size_mm", "capture_size_mm", "padding"}:
        if isinstance(scene[key], bool) or not isinstance(scene[key], (int, float)) or not math.isfinite(scene[key]) or scene[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("plate_size_mm", "opening_size_mm", "mask_size_mm", "volume_size_mm", "capture_size_mm"):
        value = scene[key]
        if not isinstance(value, list) or len(value) != (3 if key == "volume_size_mm" else 2) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in value):
            raise ValueError(f"{key} must contain positive dimensions")
    rows = scene["mask_rows"]
    if not isinstance(rows, list) or len(rows) != 4 or any(not isinstance(row, str) or len(row) != 4 or set(row) - {"0", "1"} for row in rows):
        raise ValueError("mask_rows must contain four 4-character binary strings")
    if isinstance(scene["padding"], bool) or not isinstance(scene["padding"], int) or scene["padding"] < 1:
        raise ValueError("padding must be an integer of at least 1")
    if scene["opening_size_mm"][0] >= scene["plate_size_mm"][0] or scene["opening_size_mm"][1] >= scene["plate_size_mm"][1]:
        raise ValueError("opening must fit inside the plate")
    enclosure = scene.get("enclosure")
    if enclosure is not None:
        if not isinstance(enclosure, dict) or enclosure.keys() != ENCLOSURE_KEYS:
            raise ValueError(f"enclosure must contain exactly: {', '.join(sorted(ENCLOSURE_KEYS))}")
        for key in ("wall_thickness_mm", "cavity_depth_mm", "backplate_thickness_mm"):
            value = enclosure[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"enclosure.{key} must be positive")
        center = enclosure["center_xy_mm"]
        if not isinstance(center, list) or len(center) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in center):
            raise ValueError("enclosure.center_xy_mm must contain two finite coordinates")
        cx, cy = center
        t = enclosure["wall_thickness_mm"]
        cavity_x = (cx-scene["plate_size_mm"][0]/2+t, cx+scene["plate_size_mm"][0]/2-t)
        cavity_y = (cy-scene["plate_size_mm"][1]/2+t, cy+scene["plate_size_mm"][1]/2-t)
        opening = scene["opening_size_mm"]
        if not (cavity_x[0] < -opening[0]/2 and cavity_x[1] > opening[0]/2 and
                cavity_y[0] < -opening[1]/2 and cavity_y[1] > opening[1]/2):
            raise ValueError("opening must fit inside the enclosure cavity")
        rear_z = scene["source_to_plate_mm"] - enclosure["cavity_depth_mm"]
        if not (rear_z < 0 and scene["source_to_plate_mm"] > 0 and
                cavity_x[0] <= -0.25 and cavity_x[1] >= 0.25 and cavity_y[0] <= 0 <= cavity_y[1]):
            raise ValueError("enclosure must clear the source segment and extend behind and in front of it")
    if any(scene["pitch_mm"] * 4 > side for side in scene["mask_size_mm"]):
        raise ValueError("4x4 active grid must fit inside mask")
    if any(scene["capture_size_mm"][i] < scene["volume_size_mm"][i] for i in range(2)):
        raise ValueError("capture plane must cover the full volume cross-section")
    return scene


def coordinates(scene):
    plate_front = scene["source_to_plate_mm"]
    plate_back = plate_front + scene["plate_thickness_mm"]
    mask_front = plate_back + scene["gap_mm"]
    mask_back = mask_front + scene["mask_thickness_mm"]
    return {"plate_front": plate_front, "plate_back": plate_back, "mask_front": mask_front,
            "mask_back": mask_back, "handoff_z": mask_back + scene["handoff_mm"]}


def geometry(scene):
    """Return non-overlapping PEC boxes in millimetres, with mask rows +y to -y."""
    out, z = [], coordinates(scene)
    def box(name, x0, y0, x1, y1, za, zb):
        out.append({"name": name, "start": [x0, y0, za], "stop": [x1, y1, zb]})
    pw, ph = scene["plate_size_mm"]
    ow, oh = scene["opening_size_mm"]
    enclosure = scene.get("enclosure")
    cx, cy = enclosure["center_xy_mm"] if enclosure else (0, 0)
    xa, xb = cx - pw / 2, -ow / 2
    # Four bars leave one continuous square opening.
    box("plate_left", xa, cy-ph/2, xb, cy+ph/2, z["plate_front"], z["plate_back"])
    box("plate_right", -xb, cy-ph/2, cx+pw/2, cy+ph/2, z["plate_front"], z["plate_back"])
    box("plate_bottom", -ow/2, cy-ph/2, ow/2, -oh/2, z["plate_front"], z["plate_back"])
    box("plate_top", -ow/2, oh/2, ow/2, cy+ph/2, z["plate_front"], z["plate_back"])
    if enclosure:
        t = enclosure["wall_thickness_mm"]
        outer_x, outer_y = (cx-pw/2, cx+pw/2), (cy-ph/2, cy+ph/2)
        inner_x, inner_y = (outer_x[0]+t, outer_x[1]-t), (outer_y[0]+t, outer_y[1]-t)
        rear = z["plate_front"] - enclosure["cavity_depth_mm"]
        back = rear - enclosure["backplate_thickness_mm"]
        box("enclosure_left", outer_x[0], outer_y[0], inner_x[0], outer_y[1], rear, z["plate_front"])
        box("enclosure_right", inner_x[1], outer_y[0], outer_x[1], outer_y[1], rear, z["plate_front"])
        box("enclosure_bottom", inner_x[0], outer_y[0], inner_x[1], inner_y[0], rear, z["plate_front"])
        box("enclosure_top", inner_x[0], inner_y[1], inner_x[1], outer_y[1], rear, z["plate_front"])
        # shortcut: sealed rear omits openings and seams, upgrade for leakage studies.
        box("enclosure_back", outer_x[0], outer_y[0], outer_x[1], outer_y[1], back, rear)
    mw, mh = scene["mask_size_mm"]
    pitch = scene["pitch_mm"]
    grid_x0, grid_y0 = -2*pitch, -2*pitch
    # Full frame surrounds the active grid; metal cells join naturally when adjacent.
    box("mask_left", -mw/2, -mh/2, grid_x0, mh/2, z["mask_front"], z["mask_back"])
    box("mask_right", grid_x0 + 4*pitch, -mh/2, mw/2, mh/2, z["mask_front"], z["mask_back"])
    box("mask_bottom", grid_x0, -mh/2, grid_x0 + 4*pitch, grid_y0, z["mask_front"], z["mask_back"])
    box("mask_top", grid_x0, grid_y0 + 4*pitch, grid_x0 + 4*pitch, mh/2, z["mask_front"], z["mask_back"])
    for row, bits in enumerate(scene["mask_rows"]):
        y0 = grid_y0 + (3-row)*pitch
        for col, bit in enumerate(bits):
            if bit == "0":
                x0 = grid_x0 + col*pitch
                box(f"mask_r{row}c{col}", x0, y0, x0+pitch, y0+pitch, z["mask_front"], z["mask_back"])
    return out
