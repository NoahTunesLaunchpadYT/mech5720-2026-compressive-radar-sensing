"""
Continuous coded-mask capture with a camera (a copy of radar_capture_continuous.py plus
the camera features below): the carriage sweeps the whole mask at constant speed
while the radar streams, instead of stopping at the 24 mask positions. Each frame's
mask position comes from the Arduino's timestamped step count, corrected with the sync
model fitted by rail_sync_test.py (rail_tests/sync_model.json: latency, offset and
backlash), and the frames are averaged into BIN_MM bins of carriage position.

  "Capture sweep"  -> from the nearer end, sweep the mask at "Speed" there and back
                      (both directions averaged) while logging every radar frame and the
                      Arduino's P lines, then save captures_continuous/<scene>/sweep.npz.
                      Takes ~35 s at 10 mm/s. The slice plot fills in as it goes (a
                      preview from the step counts so far, shifted by the sync model's
                      latency and offset) with an orange line at the carriage.
                      Then checks the legs agree: each leg is binned on its own, its mean
                      over position removed (static clutter), and the two are
                      cross-correlated along carriage position. The best shift is how far
                      the back sweep's positions read from the forward sweep's; ~0 if the
                      sync model is right. Shown in the status and the slice title, with a
                      warning above ALIGN_WARN_MM or when the correlation is weak.
  Prev/Next, Mask 0..23 buttons, Calibrate, Reconnect, Celebrate, STOP: as in
                      radar_capture.py (they still use the 24 mask positions).
  Scene 00..88 / Ambient / Amb end -> as in radar_capture.py; grey = no sweep yet,
                      green = captured.
  "Subtract ambient" -> subtracts the chosen ambient sweep at the matching carriage
                      position (display only, saved sweeps stay raw). "start" uses each
                      sweep's own ambient.npz: the Ambient sweep current when it was
                      captured. So Ambient can be re-captured mid-calibration (e.g. after a
                      radar restart flips the near-field phase): sweeps from then on use
                      the new one, earlier ones keep theirs. When Ambient is re-captured,
                      each sweep on the old one (or with no ambient.npz yet) gets whichever
                      of the old and new cancels it best, so sweeps taken between a restart
                      and the re-capture move to the new one.
  Beam pattern     -> select a mask (0..23) or any carriage position (mm, slider or
                      entry) plus a radar range (mm): for each scene with a sweep, the
                      sum over that range of |slice - ambient| at that position, as a 3D
                      surface over the 9x9 grid. The 4x4 window shows the mask at that
                      position, including part-columns between mask positions.
  "Export beam heatmaps" -> the beam patterns at the 24 mask positions (as radar_capture.py)
                      plus beam_cube.npz: the whole 9 x 9 x position cube.
  "Demultiplexing..." -> radar_capture.py's demux window on sweeps: one row per carriage
                      position bin instead of one per mask (captures_continuous/demux/).
                      "Page" tabs: sets of the 8 scenarios plus their own demux Ambient
                      (page 1 in demux/, page n in demux/page<n>/), so a new demux ambient
                      can be measured on a new page without touching the others.
                      "Export figures" writes demux/<slot>/export_<time>/: summary.png (all
                      the plots below together) and each separately: profile, slice, and
                      per detection score_N, score_overlay_N (the camera snapshot at 50%
                      over the score map), signature_N; camera.png (the snapshot with the
                      estimated and true grid points) and scene3d.png (the reconstruction
                      with the snapshot on the calibration board).

Each sweep.npz holds:
  slice        complex (positions, range points): frames averaged per GRID_MM bin (mean
               over each frame's sweeps first); empty bins interpolated from neighbours
  counts       frames in each bin
  grid_mm      carriage distance from the switch of each bin centre
  profiles, frame_mm, frame_towards: every frame, its corrected position and direction
  frame_pc, frame_tick, pos_pc, pos_us, pos_um: the raw logs (as rail_sync_test.py),
               so the positions can be recomputed with a later sync model
  sync_model (JSON), speed_mm_s, distances_m, frame_errors, sensor_config, server_info
  align_shift_mm, align_rho: back-vs-forward shift and its correlation coefficient;
  align_shifts_mm, align_curve: the whole correlation curve
ambient.npz (next to a scene's or demux slot's sweep.npz): slice, time of the Ambient
sweep that was current when that sweep was captured

Camera ("Camera..." window): a phone streaming MJPEG over the network, e.g. the Android
app IP Webcam (http://<phone ip>:8080/video; a single-JPEG URL such as /shot.jpg is
polled instead). Keep the phone still and lock its exposure if the app allows.
  "Rotate 90°"       -> turns every frame a further 90° clockwise (remembered); clears
                      the grid corners, which no longer line up.
  "Set grid corners" -> click grid points 00 and 88 in the image. The phone faces the
                      grid square on, so image positions scale linearly to grid (row, col).
  Every sweep saves camera.jpg (taken as the sweep starts) and camera.json (corners,
  targets) next to sweep.npz. There is no automatic detection: in the demux window, click
  the camera snapshot to mark a target (right-click removes the nearest). The marks are
  saved in camera.json. "Camera back / up (mm)" (demux window) is where the camera is
  relative to the radar: behind it and above it. A mark is exact on the board, but a
  target nearer than the board is seen with parallax, so it is put on the camera's ray
  through the mark at the radar range of the estimate it best matches (camera_truth).
  That 3D point, seen from the radar, gives the true pixel on the heatmaps, the error
  column and "True pixels" (nearest grid points); the 3D scene puts the cut-out there.

Setup: as radar_capture.py; needs the 'G'/P-line firmware and a sync model.
  python radar_capture_camera.py [--arduino COM12] [--radar COM11] [--mock]
"""

import argparse
import json
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import serial
import tkinter as tk
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d
from PIL import Image, ImageChops, ImageTk

import demux
import radar_capture as rc
from radar_capture import (
    a121, AMBIENT, AMBIENT_END, AMBIENTS, ARDUINO_LINE, DEMUX_SLOTS,
    FrameErrors, MASK, MASK_OFF, MASK_ON, MAX_LAG_S, MOVE_TIMEOUT_S, NUM_POSITIONS,
    SCENE_FULL, SCENE_NONE, SCENES, SETTLE_S, ScanAborted, Settings, save_npz,
)
from rail_sync_test import (
    MAX_SWEEP_SPEED_MM, POS_LINE, RUNUP_MARGIN_MM, SWEEP_ACCEL_MM, TAIL_S, TARGET_RANGE_MM,
    to_pc_clock, unwrap_micros,
)

SAVE_DIR = Path(__file__).parent / "captures_continuous"
SETTINGS_FILE = Path(__file__).parent / "settings_continuous.json"
SYNC_MODEL_FILE = Path(__file__).parent / "rail_tests" / "sync_model.json"   # latest Analyse
SWEEP_FILE = "sweep.npz"
# Calibration sweeps (scenes and their ambients) use radar_capture.py's settings, to 2000 mm;
# demux sweeps the same out to 3000 mm. Same start and 10 mm step, so the first 181 bins
# line up: everything is held at the 3000 mm length, 2000 mm data NaN-padded (fit_range).
CAL_SENSOR_CONFIG = rc.SENSOR_CONFIG
SENSOR_CONFIG = a121.SensorConfig(
    start_point=rc.SENSOR_CONFIG.start_point,   # 200 mm
    num_points=281,                             # last point = 80 + 280 x 4 = 1200 -> 3000 mm
    step_length=rc.SENSOR_CONFIG.step_length,   # 10 mm
    profile=rc.SENSOR_CONFIG.profile,
    hwaas=rc.SENSOR_CONFIG.hwaas,
    sweeps_per_frame=rc.SENSOR_CONFIG.sweeps_per_frame,
    frame_rate=rc.SENSOR_CONFIG.frame_rate,
)
DISTANCES_M = (SENSOR_CONFIG.start_point
               + np.arange(SENSOR_CONFIG.num_points) * SENSOR_CONFIG.step_length) * 0.0025
AMBIENT_COPY = "ambient.npz"   # the ambient in effect for a sweep (see the docstring)
BIN_MM = 1.0
GRID_MM = np.arange(rc.START_OFFSET_MM, rc.mm_from_position(NUM_POSITIONS) + BIN_MM / 2, BIN_MM)
DEFAULT_SPEED_MM = 10.0
ALIGN_MAX_MM = 5.0       # forward/back check: shifts searched, +-
ALIGN_STEP_MM = 0.05
ALIGN_WARN_MM = 0.5      # warn when the legs disagree by more than this
ALIGN_MIN_RHO = 0.8      # or correlate worse than this
DEFAULT_CAMERA_URL = "http://192.168.1.2:8080/video"
CAMERA_VIEW_H = 600      # height of the image in the camera window
FPS_WINDOW_S = 2.0       # camera frame rate measured over this long
ROTATIONS = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
             270: Image.Transpose.ROTATE_90}   # clockwise degrees -> PIL (anticlockwise)
TARGET_R_CELLS = 0.6     # radius of a marked target (grid cells): the reflector is ~1 cell
BOARD_ALPHA = 0.35       # opacity of the snapshot board in the 3D scene (cut-outs are opaque)
CORNERS = ("00", "88")   # grid points clicked to scale the image to the grid


def mask_column(mm):
    """Continuous mask number at carriage distance mm: the CSV column at the window's
    left edge (mask m = columns m..m+3)."""
    pos = (mm - rc.START_OFFSET_MM) / rc.STEP_MM + 1
    return NUM_POSITIONS - pos if rc.MASK_REVERSED else pos - 1


def mask_mm(m):
    return rc.mm_from_position(rc.position_of_mask(m))


def grid_index(mm):
    return int(np.clip(np.round((mm - GRID_MM[0]) / BIN_MM), 0, len(GRID_MM) - 1))


def frame_positions(d, model):
    """(mm, towards) for each logged frame: the step count at the frame's time (both
    clocks mapped to the PC clock as in rail_sync_test.analyse), minus the sync model's
    offset + v x latency + backlash x (moving towards the switch). While stationary the
    direction of the last move is kept."""
    tel_t = to_pc_clock(unwrap_micros(d["pos_us"]) * 1e-6, d["pos_pc"])
    tel_mm = d["pos_um"] / 1000
    frame_t = to_pc_clock(d["frame_tick"], d["frame_pc"])
    x_tel = np.interp(frame_t, tel_t, tel_mm)
    v = np.interp(frame_t, tel_t, np.gradient(tel_mm, tel_t))
    last_move = np.maximum.accumulate(np.where(np.abs(v) > 0.5, np.arange(len(v)), 0))
    towards = (v < 0)[last_move]
    return (x_tel - model["offset_mm"] - v * model["latency_ms"] / 1000
            - model["backlash_mm"] * towards), towards


def bin_frames(mm, profiles):
    """(slice, counts): profiles averaged into the GRID_MM bins; empty bins linearly
    interpolated between the nearest filled ones."""
    k = np.round((mm - GRID_MM[0]) / BIN_MM).astype(int)
    ok = (k >= 0) & (k < len(GRID_MM))
    counts = np.bincount(k[ok], minlength=len(GRID_MM))
    sums = np.zeros((len(GRID_MM), profiles.shape[1]), complex)
    np.add.at(sums, k[ok], profiles[ok])
    have = np.flatnonzero(counts)
    slice_ = sums / np.maximum(counts, 1)[:, None]
    miss = np.flatnonzero(counts == 0)
    j = np.clip(np.searchsorted(have, miss), 1, len(have) - 1)
    a, b = have[j - 1], have[j]
    t = np.clip((miss - a) / (b - a), 0, 1)[:, None]
    slice_[miss] = (1 - t) * slice_[a] + t * slice_[b]
    return slice_, counts


def leg_alignment(mm, towards, profiles):
    """Cross-correlation of the two sweep directions along carriage position.

    Each leg is binned on GRID_MM and its mean over position removed, so only what the
    mask changes is compared. For each shift s the back leg is interpolated at x + s
    and correlated with the forward leg at x (normalised, over the overlap). Returns
    (shift, rho, shifts, curve): the best s, refined with a parabola through the peak,
    is how far the back sweep's positions read above the forward sweep's."""
    legs = []
    for leg in (~towards, towards):
        sl = bin_frames(mm[leg], profiles[leg])[0]
        legs.append(sl - sl.mean(axis=0))
    f, b = legs
    shifts = np.arange(-ALIGN_MAX_MM, ALIGN_MAX_MM + ALIGN_STEP_MM / 2, ALIGN_STEP_MM)
    curve = np.empty(len(shifts))
    for i, sh in enumerate(shifts):
        u = (GRID_MM - GRID_MM[0] + sh) / BIN_MM
        ok = (u >= 0) & (u <= len(GRID_MM) - 1)
        i0 = np.minimum(u[ok].astype(int), len(GRID_MM) - 2)
        t = (u[ok] - i0)[:, None]
        bs = (1 - t) * b[i0] + t * b[i0 + 1]
        fs = f[ok]
        curve[i] = np.vdot(fs, bs).real / (np.linalg.norm(fs) * np.linalg.norm(bs))
    k = int(np.clip(np.argmax(curve), 1, len(curve) - 2))
    c0, c1, c2 = curve[k - 1:k + 2]
    step = 0.5 * (c0 - c2) / (c0 - 2 * c1 + c2) if c0 - 2 * c1 + c2 < 0 else 0.0
    return shifts[k] + step * ALIGN_STEP_MM, float(curve[k]), shifts, curve


def alignment_text(shift, rho):
    return f"back vs forward {shift:+.2f} mm (correlation {rho:.2f})"


def load_alignment(folder):
    """(shift, rho) of the folder's sweep, or None. Sweeps saved before the check existed
    are checked from their saved frames."""
    f = folder / SWEEP_FILE
    if not f.exists():
        return None
    with np.load(f) as d:
        if "align_shift_mm" in d:
            return float(d["align_shift_mm"]), float(d["align_rho"])
        return leg_alignment(d["frame_mm"], d["frame_towards"], d["profiles"])[:2]


_slices = {}   # sweep file -> (mtime, slice), shared by the beam loads


def load_slice(folder, file=SWEEP_FILE):
    """The folder's sweep slice (positions x range, complex), or None if not captured."""
    f = folder / file
    if not f.exists():
        return None
    mtime = f.stat().st_mtime_ns
    hit = _slices.get(f)
    if hit is None or hit[0] != mtime:
        with np.load(f) as d:
            hit = _slices[f] = (mtime, fit_range(d["slice"]))
    return hit[1]


def fit_range(slice_):
    """A slice padded with NaN to len(DISTANCES_M) range bins (older sweeps stop at 2000 mm)."""
    pad = len(DISTANCES_M) - slice_.shape[-1]
    return np.pad(slice_, ((0, 0), (0, pad)), constant_values=np.nan) if pad else slice_


def sweep_ambient(folder, ambient):
    """The ambient slice to subtract from folder's sweep: for AMBIENT, the sweep's own copy
    if it has one; otherwise that ambient's current sweep (None if not captured)."""
    own = load_slice(folder, AMBIENT_COPY) if ambient == AMBIENT else None
    return own if own is not None else load_slice(SAVE_DIR / ambient)


def read_ambient(file):
    with np.load(file) as d:
        return fit_range(d["slice"]), float(d["time"])


def copy_ambient(folders, candidates):
    """Give each folder with a sweep, and no ambient.npz or a copy of one of the candidates,
    the candidate (slice, time) that cancels its sweep best (smallest mean
    |sweep - ambient|). Returns how many got each."""
    picks = [0] * len(candidates)
    times = [t for _, t in candidates]
    for folder in folders:
        f = folder / AMBIENT_COPY
        if (folder / SWEEP_FILE).exists() and (not f.exists() or read_ambient(f)[1] in times):
            sl = load_slice(folder)
            k = int(np.argmin([np.nanmean(np.abs(sl - a)) for a, _ in candidates]))
            np.savez(folder / AMBIENT_COPY, slice=candidates[k][0], time=candidates[k][1])
            picks[k] += 1
    return picks


def beam_cube(lo, hi, ambient):
    """9 x 9 x positions: per scene and carriage position, the sum over lo..hi mm of
    |slice - ambient slice| (NaN = scene not captured). ambient: AMBIENT (each scene's
    own, see sweep_ambient), AMBIENT_END or None. Returns (cube, subtracted): subtracted is
    False if no scene had an ambient to subtract."""
    in_range = (DISTANCES_M * 1000 >= lo) & (DISTANCES_M * 1000 <= hi)
    cube = np.full((9, 9, len(GRID_MM)), np.nan)
    subtracted = False
    for s in SCENES:
        sl = load_slice(SAVE_DIR / s)
        if sl is not None:
            amb = sweep_ambient(SAVE_DIR / s, ambient) if ambient else None
            subtracted |= amb is not None
            cube[int(s[0]), int(s[1])] = np.abs((sl - (0 if amb is None else amb))[:, in_range]).sum(axis=1)
    return cube, subtracted


def normalise_across_positions(cube):
    """Divide each scene by its mean over all carriage positions."""
    return cube / cube.mean(axis=2, keepdims=True)


def image_to_grid(corners, pts):
    """Image fractions -> grid (col, row). The camera faces the grid square on, so each
    axis is linear between the clicked points 00 and 88."""
    (x0, y0), (x8, y8) = corners
    return (np.asarray(pts) - (x0, y0)) / (x8 - x0, y8 - y0) * 8


def grid_to_image(corners, pts):
    (x0, y0), (x8, y8) = corners
    return np.asarray(pts) / 8 * (x8 - x0, y8 - y0) + (x0, y0)


def target_radius(corners, w):
    """Radius in pixels of a marked target on an image w pixels wide."""
    return TARGET_R_CELLS * abs(corners[1][0] - corners[0][0]) / 8 * w


def board_point(row, col):
    """(across, forward, up) mm of grid position (row, col) on the board."""
    return np.array([(col - demux.GRID_CENTRE) * demux.GRID_SPACING_MM,
                     np.full_like(np.asarray(col, float), demux.GRID_DISTANCE_MM),
                     (demux.GRID_CENTRE - row) * demux.GRID_SPACING_MM])


def camera_truth(corners, target, camera, results):
    """Where a target marked on the snapshot really is. camera = (across, forward, up) mm
    of the camera from the radar. The mark is where the camera's ray through the target
    meets the board; the target is on that ray at the radar range. Each estimate's range
    is tried and the one whose resulting radar pixel is nearest that estimate's pixel is
    used. Returns dict: xyz (mm), pixel (row, col) as seen from the radar, s (fraction of
    the camera-to-board distance; 1 = on the board, as with no estimates)."""
    col, row = image_to_grid(corners, (target["x"], target["y"]))
    d = board_point(row, col) - camera
    options = []
    for r in results:
        a, b, c = d @ d, camera @ d, camera @ camera - r["range_mm"] ** 2
        s = (-b + np.sqrt(b * b - a * c)) / a
        xyz = camera + s * d
        k = demux.GRID_DISTANCE_MM / xyz[1]   # on to the board from the radar
        pixel = (demux.GRID_CENTRE - xyz[2] * k / demux.GRID_SPACING_MM,
                 demux.GRID_CENTRE + xyz[0] * k / demux.GRID_SPACING_MM)
        options.append((np.hypot(pixel[0] - r["pixel"][0], pixel[1] - r["pixel"][1]),
                        {"xyz": xyz, "pixel": pixel, "s": s}))
    if not options:
        return {"xyz": camera + d, "pixel": (row, col), "s": 1.0}
    return min(options, key=lambda o: o[0])[1]


def photo_extent(corners):
    """imshow extent (left, right, bottom, top) placing a camera image in grid (col, row)
    units."""
    (c0, r0), (c1, r1) = image_to_grid(corners, [(0, 0), (1, 1)])
    return c0, c1, r1, r0


POLAR_W = 1.1 * demux.to_xyz(1.0, *demux.pixel_angles(demux.GRID_CENTRE, 8))[0]   # fixed half-width
GIF_STEP_MM = 0.5        # beam GIF: carriage step per frame (interpolated between 1 mm bins)
GIF_FRAME_MS = 20        # 50 fps (browsers don't go faster)
GIF_POLAR_FILL = 0.95    # beam GIF: fraction of its panel the largest beam fills
BEAM_GIF_CELLS = ((2, 6), (6, 2))   # beam GIF: heatmap cells (row, col) whose fingerprints are shown
GIF_STRIP_SPAN = 9       # mask strip panel: columns shown either side of the window


def plot_polar(ax, pattern, top, cmap, highlight=None, zoom=1.0, points=True):
    """Polar beam shape on a 3D ax: each direction (the grid pixel's bearing and elevation
    from the radar) out to a radius of intensity / top, radar at the origin. Fixed limits.
    highlight = (row, col) drawn red. Returns the artists."""
    have = np.isfinite(pattern)
    g = np.linspace(0, 8, rc.SMOOTH_PX)[::5]
    rows, cols = np.meshgrid(g, g, indexing="ij")
    r = np.clip(rc.smooth(pattern)[::5, ::5] / top, 0, 1)   # the spline can overshoot
    x, y, z = demux.to_xyz(r, *demux.pixel_angles(rows, cols))
    art = [ax.plot_surface(x, y, z, facecolors=cmap(np.nan_to_num(r)), shade=False,
                           edgecolor="#555", linewidth=0.1)]
    art.append(ax.scatter(0, 0, 0, marker="^", c="k", s=30))
    if points:
        rr, cc = np.nonzero(have)
        px, py, pz = demux.to_xyz(pattern[have] / top, *demux.pixel_angles(rr, cc))
        art.append(ax.scatter(px, py, pz, c="k", s=6))
        for i in range(len(px)):   # spokes from the radar
            art += ax.plot([0, px[i]], [0, py[i]], [0, pz[i]], color="#999", linewidth=0.3)
        if highlight and have[highlight]:
            i = np.flatnonzero((rr == highlight[0]) & (cc == highlight[1]))[0]
            art.append(ax.scatter(px[i], py[i], pz[i], c="red", s=40))
    w = POLAR_W
    ax.set_xlim(-w, w)
    ax.set_ylim(0, 1.05)
    ax.set_zlim(-w, w)
    ax.set_box_aspect((2 * w, 1.05, 2 * w), zoom=zoom)
    ax.set_xticks([-w * 0.8, 0, w * 0.8], ["", "0", ""])
    ax.set_zticks([-w * 0.8, 0, w * 0.8], ["", "0", ""])
    ax.set_xlabel("Across")
    ax.set_ylabel("Forward (intensity / max)")
    ax.set_zlabel("Up")
    return art


# Colour for maps that aren't radar intensity (match scores, correlations); intensity
# stays viridis (app.sequential)
SCORE_CMAP = matplotlib.colormaps["magma"].with_extremes(bad="#dddddd")
CONFUSE_PX = 3           # confusability: only points further apart than this (grid pixels) count
MATCH_FRAME_MS = 40      # matching GIF: 25 fps
MATCH_DRAW_FRAMES = 40   # frames drawing the fingerprint in
MATCH_HOLD_MS = 3000     # the result is held this long before the GIF loops
DATAFLOW_FRAMES = 100    # dataflow GIF: frames of the scan
DATAFLOW_FRAME_MS = 40
DATAFLOW_HOLD_MS = 4000  # the result is held this long before the GIF loops
DATAFLOW_DPI = 60


def mask_strip(ax):
    """Mask pattern strip with a fixed, outlined 4x4 window at x = 0..4 and the rest
    dimmed. Returns set(col): shows the strip at continuous mask number col."""
    rows_m, cols_m = MASK.shape
    strip = ax.imshow(MASK, cmap=matplotlib.colors.ListedColormap([MASK_OFF, MASK_ON]),
                      vmin=0, vmax=1, interpolation="nearest")
    span = GIF_STRIP_SPAN
    for x0, x1 in ((-span, 0), (4, 4 + span)):
        ax.axvspan(x0, x1, color="white", alpha=0.55, zorder=2, linewidth=0)
    ax.add_patch(matplotlib.patches.Rectangle((0, 0), 4, rows_m, fill=False,
                                              edgecolor="#e31a1c", linewidth=3, zorder=3))
    ax.set_xlim(-span, 4 + span)
    ax.set_ylim(rows_m, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return lambda col: strip.set_extent((-col, cols_m - col, rows_m, 0))


def scale_fonts(fig, k):
    """Every piece of text in fig k times its size: tick labels, titles, axis and colour
    bar labels, legends, annotations."""
    if k == 1:
        return
    fig.canvas.draw()   # creates the tick labels
    ticks = set()
    for ax in fig.axes:
        for axis in [ax.xaxis, ax.yaxis] + ([ax.zaxis] if hasattr(ax, "zaxis") else []):
            labels = axis.get_ticklabels(which="both")
            ticks.update(map(id, labels))
            if labels:
                axis.set_tick_params(which="both", labelsize=labels[0].get_fontsize() * k)
    for t in fig.findobj(matplotlib.text.Text):
        if id(t) not in ticks:
            t.set_fontsize(t.get_fontsize() * k)


class GifFrames:
    """GIF frames on one palette (from a frame with everything drawn), undithered, so
    fixed parts like colour bars don't flicker."""

    def __init__(self, palette_fig):
        self.palette = self.grab(palette_fig).quantize(256, dither=Image.Dither.NONE)
        self.frames, self.durations = [], []

    @staticmethod
    def grab(fig):
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])

    def add(self, fig, ms):
        self.frames.append(self.grab(fig).quantize(palette=self.palette, dither=Image.Dither.NONE))
        self.durations.append(ms)

    def save(self, path):
        self.frames[0].save(path, save_all=True, append_images=self.frames[1:],
                            duration=self.durations, loop=0)


def export_beams_bare(patterns, export_dir):
    """Presentation copies of the beam heatmaps: no title, labels, ticks, grid or colour
    bar. export_dir/beam_bare_<time>/: grid.png (6 x 4, each mask scaled to 0..1) and
    maskNN.png. Returns the folder."""
    out = export_dir / time.strftime("beam_bare_%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    cmap = matplotlib.colormaps["viridis"].with_extremes(bad="#dddddd")

    def draw(ax, p):
        if np.isfinite(p).sum() > 1:
            p = (p - np.nanmin(p)) / (np.nanmax(p) - np.nanmin(p))
        ax.imshow(rc.smooth(p), cmap=cmap, vmin=0, vmax=1, extent=(0, 8, 8, 0))
        ax.set_axis_off()

    fig = Figure(figsize=(8, 12), dpi=150)
    FigureCanvasAgg(fig)
    axes = fig.subplots(6, 4, gridspec_kw={"wspace": 0.05, "hspace": 0.05})
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    for p, ax in zip(patterns, axes.flat):
        draw(ax, p)
    fig.savefig(out / "grid.png")
    for m, p in enumerate(patterns):
        fig = Figure(figsize=(4, 4), dpi=150)
        FigureCanvasAgg(fig)
        draw(fig.add_axes((0, 0, 1, 1)), p)
        fig.savefig(out / f"mask{m:02d}.png")
    return out


def camera_dir():
    return SAVE_DIR / "camera"


def scene_colour(folder):
    return SCENE_FULL if (folder / SWEEP_FILE).exists() else SCENE_NONE


def demux_folder(slot, page=1):
    """A demux slot's folder: page 1 in demux/, page n in demux/page<n>/."""
    return SAVE_DIR / "demux" / ("" if page == 1 else f"page{page}") / slot


def demux_pages():
    pages = [int(f.name[4:]) for f in (SAVE_DIR / "demux").glob("page*") if f.name[4:].isdigit()]
    return [1] + sorted(pages)


class CameraStream:
    """Latest frame from an MJPEG stream (or a polled single-JPEG URL), read in a thread.
    jpeg = the raw bytes, image = decoded (PIL, RGB) and turned rotation degrees clockwise,
    time = when it arrived."""

    def __init__(self):
        self.gen = 0               # bumped to stop the current reader
        self.jpeg = self.image = None
        self.time = 0.0
        self.fps = 0.0
        self.arrivals = []         # frame times in the last FPS_WINDOW_S
        self.rotation = 0
        self.error = ""

    def start(self, url):
        if urllib.parse.urlparse(url).path in ("", "/"):
            url = url.rstrip("/") + "/video"   # IP Webcam's stream; the bare address is its web page
        self.gen += 1
        self.error = "Connecting..."
        threading.Thread(target=self.run, args=(self.gen, url), daemon=True).start()

    def stop(self):
        self.gen += 1
        self.jpeg = self.image = None
        self.error = ""

    def run(self, gen, url):
        while gen == self.gen:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    ctype = r.headers.get("Content-Type", "")
                    if "multipart" not in ctype:
                        self.set(gen, r.read(), ctype)
                        time.sleep(0.1)
                        continue
                    # Parts are separated by the boundary; each has headers, a blank line,
                    # then the JPEG (which may contain an EXIF thumbnail's own FFD8..FFD9)
                    boundary = ctype.split("boundary=")[1].split(";")[0].strip('"').encode()
                    if not boundary.startswith(b"--"):   # some cameras declare it with the dashes
                        boundary = b"--" + boundary
                    buf = b""
                    while gen == self.gen:
                        chunk = r.read1(65536)
                        if not chunk:
                            break
                        buf += chunk
                        parts = buf.split(boundary)
                        if len(parts) < 3:
                            continue
                        # parts[-1] is still arriving; parts[-2] is the newest complete one
                        body = parts[-2].split(b"\r\n\r\n", 1)[-1].rstrip(b"\r\n-")
                        buf = boundary + parts[-1]
                        try:
                            self.set(gen, body, ctype)
                        except ValueError as e:   # skip a bad frame; the next one clears it
                            self.error = f"Camera: {e}"
            except Exception as e:
                if gen == self.gen:
                    self.error = f"Camera: {e}"
                    time.sleep(1)

    def set(self, gen, jpeg, ctype):
        try:
            image = Image.open(BytesIO(jpeg)).convert("RGB")
        except Exception:
            raise ValueError(f"not an image (Content-Type {ctype!r}, starts {jpeg[:40]!r}) "
                             "- check the URL is the video stream, e.g. .../video")
        if self.rotation:
            image = image.transpose(ROTATIONS[self.rotation])
        if gen != self.gen:
            return
        now = time.monotonic()
        self.arrivals = [t for t in self.arrivals if t > now - FPS_WINDOW_S] + [now]
        self.fps = (len(self.arrivals) - 1) / max(now - self.arrivals[0], 1e-3)
        self.jpeg, self.image, self.time, self.error = jpeg, image, now, ""


class ContinuousApp(rc.CaptureApp):
    """radar_capture.CaptureApp with sweep captures: builds its own window, and reuses
    the parent's stepper, radar-reconnect, plotting-loop and live-plot methods."""

    def __init__(self, root, ser, client, open_arduino, open_radar):
        self.root = root
        self.settings = Settings(root, SETTINGS_FILE)
        self.ser = ser
        self.client = client
        self.open_arduino = open_arduino
        self.open_radar = open_radar
        self.radar_restart = False
        self.sensor_config = CAL_SENSOR_CONFIG   # wanted; radar_loop applies it on a restart
        self.session_config = None               # applied         # radar_loop exits (reconnect)
        self.session_restart = False       # radar_loop starts a fresh session
        self.session_started = threading.Event()
        self.latest = {}
        self.message = "Homing..."
        self.status = ""
        self.busy = False
        self.abort = False
        self.running = True
        self.ready = threading.Event()
        self.profile = None
        self.profile_dirty = False
        self.ymax = 1.0
        self.radar_frames = 0
        self.resyncs = 0
        self.drawn_frames = 0
        self.fps_time = time.time()
        self.frame_log = self.pos_log = None   # (pc, tick, profile) / (pc, us, um) during a sweep
        self.shown_mm = None
        self.live = None           # sweep in progress: {"model", "seen", "sums", "counts"}
        self.speed = tk.DoubleVar(value=DEFAULT_SPEED_MM)
        self.settings.bind("cont.speed", self.speed)
        self.scene = tk.StringVar(value=SCENES[0])
        self.settings.bind("cont.scene", self.scene)
        self.stack = load_slice(SAVE_DIR / self.scene.get())
        self.stack_name = rc.scene_title(self.scene.get())
        self.stack_align = load_alignment(SAVE_DIR / self.scene.get())
        self.scene_saved = False
        self.ambient_source = tk.StringVar(value=AMBIENT)
        self.settings.bind("cont.ambient_source", self.ambient_source)
        self.ambient = load_slice(SAVE_DIR / self.ambient_source.get())
        self.subtract = tk.BooleanVar(value=False)
        self.settings.bind("cont.subtract", self.subtract)
        if self.ambient is None:
            self.subtract.set(False)
        self.demux = None
        SAVE_DIR.mkdir(exist_ok=True)
        # Camera: stream and grid corners
        self.camera = CameraStream()
        self.cam_url = tk.StringVar(value=DEFAULT_CAMERA_URL)
        self.cam_rotation = tk.IntVar(value=0)
        self.cam_back_mm = tk.DoubleVar(value=0.0)   # camera behind the radar
        self.cam_up_mm = tk.DoubleVar(value=0.0)     # and above it
        self.photo_plane_mm = tk.DoubleVar(value=demux.GRID_DISTANCE_MM)   # 3D figures only
        self.grid_3d = tk.BooleanVar(value=True)   # calibration grid points in the 3D figures
        self.lines_3d = tk.BooleanVar(value=True)  # background grid lines in the 3D figures
        self.export_font = tk.DoubleVar(value=1.0)  # demux exports: text size multiplier
        for name in ("cam_url", "cam_rotation", "cam_back_mm", "cam_up_mm", "photo_plane_mm",
                     "grid_3d", "lines_3d", "export_font"):
            self.settings.bind(f"cont.{name}", getattr(self, name))
        self.camera.rotation = self.cam_rotation.get()
        f = camera_dir() / "corners.json"
        self.cam_corners = json.loads(f.read_text()) if f.exists() else None
        self.cam_window = None

        root.title("Radar Capture (continuous + camera)")
        root.geometry("1550x860")

        # ---- Left: position, mask, controls ----
        left = tk.Frame(root)
        left.pack(side="left", fill="y", padx=10, pady=6)
        self.lbl_pos = tk.Label(left, text="Mask --", font=("Helvetica", 28, "bold"))
        self.lbl_dist = tk.Label(left, text="-- mm", font=("Helvetica", 13))
        self.lbl_pos.pack()
        self.lbl_dist.pack()

        mrow = tk.Frame(left)
        mrow.pack(pady=4)
        self.cell = 32
        self.mask_canvas = tk.Canvas(mrow, width=4 * self.cell, height=4 * self.cell,
                                     highlightthickness=1, highlightbackground="#888")
        self.mask_canvas.pack(side="left", padx=(0, 10))
        grid = tk.Frame(mrow)
        grid.pack(side="left")
        self.goto_buttons = []
        for m in range(NUM_POSITIONS):
            b = tk.Button(grid, text=str(m), width=3,
                          command=lambda m=m: self.goto(rc.position_of_mask(m)))
            b.grid(row=m // 6, column=m % 6, padx=1, pady=1)
            self.goto_buttons.append(b)
        self.goto_bg = self.goto_buttons[0].cget("bg")

        # Fixed widths (~380 px) and wrapping, so long messages don't resize the window
        self.lbl_status = tk.Label(left, text="", font=("Helvetica", 12), width=42, wraplength=380)
        self.lbl_msg = tk.Label(left, text="", font=("Helvetica", 10), fg="#b00", width=47,
                                wraplength=380)
        self.lbl_errors = tk.Label(left, text="", font=("Helvetica", 9), width=54)
        self.lbl_model = tk.Label(left, text="", font=("Helvetica", 9), fg="#666", width=54,
                                  wraplength=380)
        for w in (self.lbl_status, self.lbl_msg, self.lbl_errors, self.lbl_model):
            w.pack()
        self.errors_shown = (0, 0)

        row = tk.Frame(left)
        row.pack(pady=3)
        self.btn_prev = tk.Button(row, text="◀ Prev", width=8, command=lambda: self.move("F"))
        self.btn_capture = tk.Button(row, text="Capture sweep", width=12, command=self.capture)
        self.btn_next = tk.Button(row, text="Next ▶", width=8, command=lambda: self.move("B"))
        for b in (self.btn_prev, self.btn_capture, self.btn_next):
            b.pack(side="left", padx=3)
        srow = tk.Frame(left)
        srow.pack()
        tk.Label(srow, text=f"Speed (mm/s, max {MAX_SWEEP_SPEED_MM:g}), there and back").pack(side="left")
        tk.Entry(srow, textvariable=self.speed, width=5).pack(side="left", padx=4)

        actions = tk.Frame(left)
        actions.pack(pady=3)
        self.btn_calibrate = tk.Button(actions, text="Calibrate", width=20, command=self.calibrate)
        self.btn_radar = tk.Button(actions, text="Reconnect radar", width=20,
                                   command=self.reconnect_radar)
        self.btn_arduino = tk.Button(actions, text="Reconnect Arduino", width=20,
                                     command=self.reconnect_arduino)
        btn_refresh = tk.Button(actions, text="Refresh plots", width=20, command=self.refresh_plots)
        self.btn_export = tk.Button(actions, text="Export beam heatmaps", width=20,
                                    command=self.export)
        self.btn_export_bare = tk.Button(actions, text="Export bare heatmaps", width=20,
                                         command=lambda: self.export(bare=True))
        self.btn_celebrate = tk.Button(actions, text="Celebrate!", width=20, fg="#a0a",
                                       command=lambda: self.send("T"))
        btn_demux = tk.Button(actions, text="Demultiplexing...", width=20, command=self.open_demux)
        btn_camera = tk.Button(actions, text="Camera...", width=20, command=self.open_camera)
        btn_stop = tk.Button(actions, text="STOP", width=20, fg="#b00", command=self.stop)
        for i, b in enumerate((self.btn_calibrate, self.btn_radar, self.btn_arduino, btn_refresh,
                               self.btn_export, self.btn_export_bare, self.btn_celebrate,
                               btn_demux, btn_camera, btn_stop)):
            b.grid(row=i // 2, column=i % 2, padx=2, pady=2)

        checks = tk.Frame(left)
        checks.pack()
        self.chk_subtract = tk.Checkbutton(checks, text="Subtract ambient (display only)",
                                           variable=self.subtract, command=self.on_subtract)
        self.chk_subtract.pack(side="left")
        self.ambient_radios = [
            tk.Radiobutton(checks, text=text, variable=self.ambient_source, value=a,
                           command=self.on_ambient_source)
            for text, a in (("start", AMBIENT), ("end", AMBIENT_END))]
        for b in self.ambient_radios:
            b.pack(side="left")
        tk.Label(left, text="C/Enter capture sweep  ·  ←/→ move  ·  Space stop",
                 font=("Helvetica", 9), fg="#666").pack(pady=2)

        tk.Label(left, text="Scene (save folder)", font=("Helvetica", 11, "bold")).pack(pady=(6, 0))
        sgrid = tk.Frame(left)
        sgrid.pack()
        self.scene_buttons = {}
        for i, s in enumerate(SCENES + list(AMBIENTS)):
            colour = scene_colour(SAVE_DIR / s)
            b = tk.Radiobutton(sgrid, text=s, variable=self.scene, value=s, indicatoron=False,
                               width=3, font=("Helvetica", 9), bg=colour, selectcolor=colour,
                               offrelief="raised", command=self.select_scene)
            if s in AMBIENTS:
                b.config(text="Ambient" if s == AMBIENT else "Amb end", wraplength=1, width=2)
                b.grid(row=0, column=9 + AMBIENTS.index(s), rowspan=9,
                       padx=(6 if s == AMBIENT else 1, 1), pady=1, sticky="ns")
            else:
                b.grid(row=i // 9, column=i % 9, padx=1, pady=1)
            self.scene_buttons[s] = b
        legend = tk.Frame(left)
        legend.pack(pady=2)
        for text, colour in (("no sweep", SCENE_NONE), ("captured", SCENE_FULL)):
            tk.Label(legend, text=text, bg=colour, font=("Helvetica", 9), padx=6).pack(side="left", padx=2)
        self.mark_selected_scene()

        self.sequential = matplotlib.colormaps["viridis"].with_extremes(bad="#dddddd")

        # ---- Far right: beam pattern at one carriage position across the 9x9 scenes ----
        beam = tk.Frame(root)
        beam.pack(side="right", fill="y", padx=6, pady=6)
        tk.Label(beam, text="Beam pattern", font=("Helvetica", 11, "bold")).pack()
        bctl = tk.Frame(beam)
        bctl.pack()
        self.beam_by = tk.StringVar(value="mask")      # select by "mask" or "position"
        self.beam_mask = tk.IntVar(value=0)
        self.beam_mm = tk.DoubleVar(value=mask_mm(0))
        self.beam_lo = tk.DoubleVar(value=900.0)
        self.beam_hi = tk.DoubleVar(value=1200.0)
        self.normalise = tk.BooleanVar(value=False)
        self.beam_polar = tk.BooleanVar(value=False)   # 3D polar beam shape, not a surface
        for name in ("beam_by", "beam_mask", "beam_mm", "beam_lo", "beam_hi", "normalise",
                     "beam_polar"):
            self.settings.bind(f"cont.{name}", getattr(self, name))
        tk.Radiobutton(bctl, text="Mask", variable=self.beam_by, value="mask",
                       command=self.draw_beam).grid(row=0, column=0, sticky="w")
        tk.Spinbox(bctl, from_=0, to=NUM_POSITIONS - 1, width=4, textvariable=self.beam_mask,
                   command=lambda: self.select_by("mask")).grid(row=0, column=1, sticky="w")
        tk.Radiobutton(bctl, text="Carriage (mm)", variable=self.beam_by, value="position",
                       command=self.draw_beam).grid(row=1, column=0, sticky="w")
        e = tk.Entry(bctl, textvariable=self.beam_mm, width=6)
        e.grid(row=1, column=1, sticky="w")
        e.bind("<Return>", lambda e: self.select_by("position"))
        tk.Scale(bctl, variable=self.beam_mm, from_=GRID_MM[0], to=GRID_MM[-1],
                 resolution=BIN_MM, orient="horizontal", length=240, showvalue=False,
                 command=lambda v: self.select_by("position")).grid(
            row=2, column=0, columnspan=5, sticky="we")
        tk.Label(bctl, text="Range (mm)").grid(row=3, column=0, sticky="w")
        for i, var in enumerate((self.beam_lo, self.beam_hi)):
            e = tk.Entry(bctl, textvariable=var, width=6)
            e.grid(row=3, column=1 + i, padx=1, sticky="w")
            e.bind("<Return>", lambda e: self.on_beam())
        tk.Checkbutton(bctl, text="Normalise each scene across positions",
                       variable=self.normalise, command=self.on_beam).grid(
            row=4, column=0, columnspan=5, sticky="w")
        tk.Checkbutton(bctl, text="Polar view (radius = intensity along each direction)",
                       variable=self.beam_polar, command=self.draw_beam).grid(
            row=5, column=0, columnspan=5, sticky="w")
        tk.Button(bctl, text="Export sweep GIF", command=self.export_beam_gif).grid(
            row=6, column=0, columnspan=5, sticky="w")
        bfig = Figure(figsize=(4.4, 4.4), dpi=100)
        self.beam_canvas = FigureCanvasTkAgg(bfig, master=beam)
        self.beam_canvas.get_tk_widget().pack()
        self.ax_beam = bfig.add_subplot(projection="3d")   # after the canvas, so orbiting hooks up
        self.beam_cols, self.beam_rows = np.meshgrid(range(9), range(9))
        self.beam_norm = matplotlib.colors.Normalize()
        self.beam_artists = []
        self.ax_beam.set_xlim(0, 8)
        self.ax_beam.set_ylim(8, 0)
        self.ax_beam.set_xlabel("Scene column")
        self.ax_beam.set_ylabel("Scene row")
        self.ax_beam.set_zlabel("Intensity")
        bfig.colorbar(matplotlib.cm.ScalarMappable(self.beam_norm, self.sequential),
                      ax=self.ax_beam, shrink=0.6, pad=0.18)
        self.throbber = tk.Canvas(beam, width=24, height=24, highlightthickness=0)
        self.throbber.pack(pady=(0, 6))
        self.throb_arc = self.throbber.create_arc(3, 3, 21, 21, start=0, extent=270,
                                                  style="arc", width=3, outline="#3a7bd5",
                                                  state="hidden")
        self.beam_loading = False
        self.throbbing = False
        self.throb_until = 0.0
        self.beam_mask_canvas = tk.Canvas(beam, width=4 * self.cell, height=4 * self.cell,
                                          highlightthickness=1, highlightbackground="#888")
        self.beam_mask_canvas.pack()
        self.beam_gen = 0
        self.beam_result = None
        self.beam_dirty = True
        self.cube = None                   # 9 x 9 x positions beam intensities
        self.cube_info = (0.0, 0.0, False)   # (lo, hi, normalised) of self.cube

        # ---- Right: live plots and the scene's sweep slice ----
        right = tk.Frame(root)
        right.pack(side="right", fill="both", expand=True)
        self.lbl_fps = tk.Label(right, text="", font=("Helvetica", 10), fg="#666")
        self.lbl_fps.pack(anchor="e", padx=10)
        fig = Figure(figsize=(7, 7.8), dpi=100)
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 0.6, 4], width_ratios=[40, 1])
        self.ax_line = fig.add_subplot(gs[0, 0])
        self.ax_color = fig.add_subplot(gs[1, 0], sharex=self.ax_line)
        self.ax_stack = fig.add_subplot(gs[2, 0], sharex=self.ax_line)
        for ax in (self.ax_line, self.ax_color):
            ax.tick_params(labelbottom=False)
        dist_mm = DISTANCES_M * 1000
        self.line, = self.ax_line.plot(dist_mm, np.zeros_like(dist_mm), animated=True)
        self.ax_line.set_ylabel("Intensity")
        self.ax_line.set_ylim(0, self.ymax)
        self.ax_line.set_title("Live radar: mean amplitude vs distance")
        self.ax_line.grid(alpha=0.3)
        half = (dist_mm[1] - dist_mm[0]) / 2
        self.img = self.ax_color.imshow(np.zeros((1, len(dist_mm))), aspect="auto",
                                        cmap="viridis", animated=True, vmin=0, vmax=self.ymax,
                                        extent=(dist_mm[0] - half, dist_mm[-1] + half, 0, 1))
        self.ax_color.set_yticks([])
        fig.colorbar(self.img, cax=fig.add_subplot(gs[1, 1]))
        self.stack_img = self.ax_stack.imshow(
            np.full((len(GRID_MM), len(dist_mm)), np.nan), aspect="auto", cmap=self.sequential,
            interpolation="nearest",
            extent=(dist_mm[0] - half, dist_mm[-1] + half,
                    GRID_MM[-1] + BIN_MM / 2, GRID_MM[0] - BIN_MM / 2))
        self.ax_stack.set_ylabel("Carriage (mm)")
        self.ax_stack.set_xlabel("Distance (mm)")
        self.ax_stack.set_title(" ", fontsize=10)
        fig.colorbar(self.stack_img, cax=fig.add_subplot(gs[2, 1]))
        self.stack_range = [self.ax_stack.axvline(x, color="red", linestyle="--", linewidth=1)
                            for x in (self.beam_lo.get(), self.beam_hi.get())]
        # calibration range only (to 2000 mm); demux sweeps beyond it are in the demux window
        self.ax_line.set_xlim(dist_mm[0] - half, dist_mm[CAL_SENSOR_CONFIG.num_points - 1] + half)
        self.stack_row = self.ax_stack.axhline(GRID_MM[0], color="white", alpha=0.7, linewidth=1)
        self.scan_line = self.ax_stack.axhline(GRID_MM[0], color="#ff9900", linewidth=2,
                                               visible=False)
        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.backgrounds = None
        self.canvas.mpl_connect("draw_event", self.on_full_draw)
        self.stack_dirty = True

        root.bind("<Left>", lambda e: self.move("F"))
        root.bind("<Right>", lambda e: self.move("B"))
        root.bind("<space>", lambda e: self.stop())
        root.bind("<Return>", lambda e: self.capture())
        root.bind("c", lambda e: self.capture())
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.reader_thread = threading.Thread(target=self.reader, daemon=True)
        self.reader_thread.start()
        threading.Thread(target=self.heartbeat, daemon=True).start()
        self.radar_thread = threading.Thread(target=self.radar_loop, daemon=True)
        self.radar_thread.start()
        self.tick()
        self.plot_tick()

    # ---- Display ----
    def draw_window(self, canvas, col):
        """The 4x4 mask window at continuous mask number col: part-columns at the edges."""
        canvas.delete("all")
        c = self.cell
        first = int(np.floor(col))
        for k in range(max(first, 0), min(first + 5, MASK.shape[1])):
            x0 = (k - col) * c
            for r in range(MASK.shape[0]):
                canvas.create_rectangle(x0, r * c, x0 + c, (r + 1) * c,
                                        fill=MASK_ON if MASK[r, k] else MASK_OFF, outline="#444")

    def on_ambient_source(self):
        self.ambient = load_slice(SAVE_DIR / self.ambient_source.get())
        if self.ambient is None:
            self.subtract.set(False)
        self.on_subtract()

    def selected_mm(self):
        return GRID_MM[grid_index(mask_mm(self.beam_mask.get()) if self.beam_by.get() == "mask"
                                  else self.beam_mm.get())]

    def select_by(self, mode):
        self.beam_by.set(mode)
        self.draw_beam()

    def start_beam(self):
        """Read the beam settings here (Tk thread) and build the cube in the background."""
        self.beam_gen += 1
        lo, hi = self.beam_lo.get(), self.beam_hi.get()
        threading.Thread(target=self.load_beam, daemon=True,
                         args=(self.beam_gen, lo, hi,
                               self.ambient_source.get() if self.subtract.get() else None,
                               self.normalise.get())).start()
        for line, x in zip(self.stack_range, (lo, hi)):
            line.set_xdata([x, x])
        self.stack_dirty = True
        self.throb_until = time.time() + rc.THROB_MIN_S
        self.beam_loading = True
        if not self.throbbing:
            self.throbbing = True
            self.throbber.itemconfig(self.throb_arc, state="normal")
            self.throb()

    def load_beam(self, gen, lo, hi, ambient, normalise):
        cube = np.full((9, 9, len(GRID_MM)), np.nan)
        try:
            cube = beam_cube(lo, hi, ambient)[0]
            if normalise:
                cube = normalise_across_positions(cube)
        except Exception as e:   # still deliver a result, so the throbber stops
            self.message = f"Beam pattern load error: {e}"
        if gen == self.beam_gen:
            self.beam_result = (cube, (lo, hi, normalise))

    def draw_beam(self, cube=None, info=None):
        """Show the cube at the selected position; a new cube comes from load_beam."""
        if cube is not None:
            self.cube, self.cube_info = cube, info
            self.beam_loading = False
        try:
            mm = self.selected_mm()
        except tk.TclError:
            return   # half-typed entry
        k = grid_index(mm)
        col = mask_column(mm)
        self.draw_window(self.beam_mask_canvas, col)
        self.stack_row.set_ydata([mm, mm])
        self.stack_dirty = True
        lo, hi, normalise = self.cube_info
        pattern = self.cube[:, :, k] if self.cube is not None else np.full((9, 9), np.nan)
        for a in self.beam_artists:
            a.remove()
        self.beam_artists = []
        have = ~np.isnan(pattern)
        if have.any() and self.beam_polar.get():
            self.draw_beam_polar(pattern, have)
        elif have.any():
            self.ax_beam.set_xlabel("Scene column")
            self.ax_beam.set_ylabel("Scene row")
            self.ax_beam.set_box_aspect(None)
            for axis in (self.ax_beam.xaxis, self.ax_beam.yaxis, self.ax_beam.zaxis):
                axis.set_major_locator(matplotlib.ticker.AutoLocator())
                axis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            self.beam_norm.vmin, self.beam_norm.vmax = np.nanmin(pattern), np.nanmax(pattern)
            self.beam_artists.append(self.ax_beam.plot_surface(
                self.beam_cols, self.beam_rows, pattern, cmap=self.sequential,
                norm=self.beam_norm, edgecolor="#555", linewidth=0.3))
            self.beam_artists.append(self.ax_beam.scatter(
                self.beam_cols[have], self.beam_rows[have], pattern[have], c="k", s=6))
            scene = self.scene.get()
            if scene not in AMBIENTS and have[int(scene[0]), int(scene[1])]:
                r, c = int(scene[0]), int(scene[1])
                self.beam_artists.append(self.ax_beam.scatter(c, r, pattern[r, c], c="red", s=40))
            self.ax_beam.set_xlim(0, 8)
            self.ax_beam.set_ylim(8, 0)
            self.ax_beam.set_zlim(0, self.beam_norm.vmax * 1.05)
        self.ax_beam.set_title(f"{mm:.1f} mm (mask {col:.2f}), {lo:.0f}-{hi:.0f} mm"
                               + (", normalised" if normalise else ""), fontsize=10)
        if not self.beam_polar.get():
            self.ax_beam.set_zlabel("x scene mean" if normalise else "Intensity")
        self.beam_canvas.draw()

    def draw_beam_polar(self, pattern, have):
        self.beam_norm.vmin, self.beam_norm.vmax = 0, 1
        scene = self.scene.get()
        self.beam_artists += plot_polar(self.ax_beam, pattern, np.nanmax(pattern), self.sequential,
                                        None if scene in AMBIENTS else (int(scene[0]), int(scene[1])))

    def export_beam_gif(self):
        """Beam GIF, sweeping the carriage over the whole grid in GIF_STEP_MM steps (linear
        between the 1 mm bins, smooth() across the grid), the mask strip moving right to
        left: on top the strip with the fixed 4x4 window in use; below, the polar beam shape
        (lobes half length, no axes, at the live view's angle) and the heatmap, both on
        one scale (the cube's maximum). Two heatmap cells (BEAM_GIF_CELLS) are outlined and
        joined to their fingerprints (intensity vs mask position, each scaled 0..1) at the
        bottom, where a cursor follows the mask. captures_continuous/exports/."""
        if self.cube is None:
            self.status = "No beam patterns loaded yet"
            return
        cube, (lo, hi, normalise) = self.cube, self.cube_info
        elev, azim = self.ax_beam.elev, self.ax_beam.azim
        cmap = self.sequential

        def job():
            with matplotlib.rc_context({"font.size": 2 * matplotlib.rcParams["font.size"]}):
                render()   # all the GIF's text twice the usual size

        def render():
            top = np.nanmax(cube)
            fig = Figure(figsize=(13, 12), dpi=80)
            FigureCanvasAgg(fig)
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 4.2, 1.9], width_ratios=[1.9, 1],
                                  left=0.06, right=0.95, bottom=0.08, top=0.9,
                                  hspace=0.2, wspace=0.08)
            set_strip = mask_strip(fig.add_subplot(gs[0, :]))
            ax_p = fig.add_subplot(gs[1, 0], projection="3d")
            ax_h = fig.add_subplot(gs[1, 1])
            # fingerprints under the beam; the leftmost cell's on top, so the joining
            # lines (to the strips' right ends) don't cross
            cells = sorted(BEAM_GIF_CELLS, key=lambda rc_: rc_[1])
            sub = gs[2, 0].subgridspec(len(cells), 1, hspace=0.5)
            ax_s = [fig.add_subplot(sub[i]) for i in range(len(cells))]
            img = ax_h.imshow(np.full((9, 9), np.nan), cmap=cmap, vmin=0, vmax=top,
                              extent=(0, 8, 8, 0))
            bar = fig.colorbar(img, ax=ax_h, shrink=0.8,
                               label="x scene mean" if normalise else "Intensity")
            bar.locator = matplotlib.ticker.MaxNLocator(4)
            bar.update_ticks()
            ax_h.set_xticks(range(0, 9, 4))
            ax_h.set_yticks(range(0, 9, 4))
            ax_h.set_xlabel("Scene column")
            ax_h.set_ylabel("Scene row")
            ax_h.set_title("Beam Pattern (Grid)")
            # the two cells' fingerprints, static; a cursor follows the mask
            x = mask_column(GRID_MM)
            order = np.argsort(x)
            x = x[order]
            cursors = []
            for i, ((r, c), ax) in enumerate(zip(cells, ax_s)):
                ax_h.add_patch(matplotlib.patches.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                                            ec="#e31a1c", lw=3))
                ax_h.text(c + 0.65, r - 0.55, str(i + 1), color="#e31a1c", fontweight="bold")
                v = cube[r, c][order]
                ax.imshow(((v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v)))[None], cmap=cmap,
                          aspect="auto", interpolation="nearest", extent=(x[0], x[-1], 0, 1))
                cursors.append(ax.axvline(x[0], color="k", lw=3))
                ax.set_yticks([])
                ax.set_ylabel(str(i + 1), rotation=0, color="#e31a1c", fontweight="bold",
                              va="center", labelpad=20)
                ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50 / rc.STEP_MM))
                ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                    lambda t, _: f"{t * rc.STEP_MM:.0f}"))
                ax.tick_params(labelbottom=ax is ax_s[-1])
                fig.add_artist(matplotlib.patches.ConnectionPatch(
                    (c, r + 0.5), (1, 0.5), coordsA=ax_h.transData, coordsB=ax.transAxes,
                    color="#e31a1c", lw=2))
            ax_s[-1].set_xlabel("Mask position (mm)")
            ax_s[0].set_title("Fingerprints")
            # mask column rises as the carriage distance falls: sweep from the far end
            steps = np.arange(GRID_MM[-1], GRID_MM[0] - 1e-9, -GIF_STEP_MM)
            # beam zoom: the largest that keeps every frame's beam inside the panel. Each
            # direction's maximum over the sweep bounds every frame's beam, so that
            # envelope is projected and the zoom bisected (perspective makes it nonlinear).
            env = np.nanmax(cube, axis=2)
            ax_p.view_init(elev=elev, azim=azim)
            plot_polar(ax_p, env, top, cmap, points=False)
            ax_p.set_axis_off()
            g = np.linspace(0, 8, rc.SMOOTH_PX)[::5]
            rows, cols = np.meshgrid(g, g, indexing="ij")
            r = np.clip(np.nan_to_num(rc.smooth(env)[::5, ::5]) / top, 0, 1)
            pts = [np.append(np.ravel(v), 0) for v in demux.to_xyz(r, *demux.pixel_angles(rows, cols))]

            def fits(zoom):
                ax_p.set_box_aspect((2 * POLAR_W, 0.525, 2 * POLAR_W), zoom=zoom)
                fig.canvas.draw()
                px, py, _ = proj3d.proj_transform(*pts, ax_p.get_proj())
                d = ax_p.transData.transform(np.column_stack([px, py]))
                box = ax_p.bbox
                mx, my = (1 - GIF_POLAR_FILL) / 2 * box.width, (1 - GIF_POLAR_FILL) / 2 * box.height
                return (d[:, 0].min() >= box.x0 + mx and d[:, 0].max() <= box.x1 - mx
                        and d[:, 1].min() >= box.y0 + my and d[:, 1].max() <= box.y1 - my)

            lo_z, hi_z = 0.5, 12.0
            for _ in range(14):
                mid = (lo_z + hi_z) / 2
                lo_z, hi_z = (mid, hi_z) if fits(mid) else (lo_z, mid)
            zoom = lo_z
            gif = None
            for n, mm in enumerate(steps):
                u = (mm - GRID_MM[0]) / BIN_MM
                k = min(int(u), len(GRID_MM) - 2)
                t = u - k
                pattern = (1 - t) * cube[:, :, k] + t * cube[:, :, k + 1]
                col = mask_column(mm)
                set_strip(col)
                for cur in cursors:
                    cur.set_xdata([col, col])
                ax_p.cla()
                ax_p.view_init(elev=elev, azim=azim)
                plot_polar(ax_p, pattern, top, cmap, points=False)
                ax_p.set_box_aspect((2 * POLAR_W, 0.525, 2 * POLAR_W), zoom=zoom)
                ax_p.set_axis_off()
                ax_p.set_title("Beam Pattern (Polar)")
                img.set_data(rc.smooth(pattern))
                fig.suptitle(f"Carriage {mm:.1f} mm (mask {col:.2f}), range "
                             f"{lo:.0f}-{hi:.0f} mm" + (", normalised" if normalise else ""))
                gif = gif or GifFrames(fig)   # one palette: the colour bar doesn't flicker
                gif.add(fig, GIF_FRAME_MS)
                self.status = f"Beam GIF: frame {n + 1}/{len(steps)}"
                if self.abort:
                    raise ScanAborted("Beam GIF cancelled")
            out = SAVE_DIR / "exports" / time.strftime("beam_sweep_%Y%m%d_%H%M%S.gif")
            out.parent.mkdir(parents=True, exist_ok=True)
            gif.save(out)
            self.status = f"Exported {out.relative_to(SAVE_DIR.parent)} ({len(steps)} frames)"
        self.run_job(job)

    def draw_profile(self):
        p = self.profile
        mm = self.latest.get("Distance_mm")
        if self.subtract.get() and self.ambient is not None and mm is not None:
            p = p - self.ambient[grid_index(mm)]   # nearest bin, e.g. parked in the run-up
        p = np.abs(p)
        self.line.set_ydata(p)
        self.img.set_data(p[None, :])
        peak = max(np.nanmax(p), 1.0)
        if peak > self.ymax or peak < 0.4 * self.ymax:
            self.ymax = peak * 1.25
            self.ax_line.set_ylim(0, self.ymax)
            self.img.set_clim(0, self.ymax)
            self.canvas.draw()
        elif self.backgrounds:
            self.blit()

    def draw_stack(self):
        if self.stack is None:
            shown = np.full((len(GRID_MM), len(DISTANCES_M)), np.nan)
        else:
            amb = (self.ambient if self.live is not None
                   else sweep_ambient(SAVE_DIR / self.scene.get(), self.ambient_source.get()))
            shown = np.abs(self.stack - (amb if self.subtract.get() and amb is not None else 0))
            if np.isfinite(shown).any():
                self.stack_img.set_clim(np.nanmin(shown), np.nanmax(shown))
        self.stack_img.set_data(shown)
        self.ax_stack.set_title(f"{self.stack_name}: "
                                + ("sweeping (preview)" if self.live is not None
                                   else "no sweep yet" if self.stack is None
                                   else f"sweep, {BIN_MM:g} mm bins"
                                        + (", " + alignment_text(*self.stack_align)
                                           if self.stack_align else "")), fontsize=10)
        self.canvas.draw()

    def select_scene_data(self):
        self.stack = load_slice(SAVE_DIR / self.scene.get())
        self.stack_align = load_alignment(SAVE_DIR / self.scene.get())
        self.stack_name = rc.scene_title(self.scene.get())
        self.stack_dirty = True

    def live_scan_step(self):
        """During a sweep: bin the frames logged so far into the slice plot and move the
        scan line to the carriage. Positions are the step counts at each frame's arrival
        time minus the latency, minus the offset: a preview; the saved slice uses the full
        clock mapping and model."""
        live, frames, positions = self.live, self.frame_log, self.pos_log
        if live is None or frames is None:
            return
        mm = self.latest.get("Distance_mm")
        if mm is not None:
            self.scan_line.set_ydata([mm, mm])
            self.scan_line.set_visible(True)
        self.stack_dirty = True
        if len(positions) < 2:
            return
        pos = np.array(positions[:])
        m = live["model"]
        new = frames[live["seen"]:]
        t = np.array([f[0] for f in new]) - m["latency_ms"] / 1000
        n = int(np.searchsorted(t, pos[-1, 0], side="right"))   # no extrapolating
        if not n:
            return
        x = np.interp(t[:n], pos[:, 0], pos[:, 2] / 1000) - m["offset_mm"]
        k = np.round((x - GRID_MM[0]) / BIN_MM).astype(int)
        ok = (k >= 0) & (k < len(GRID_MM))
        np.add.at(live["sums"], k[ok], np.array([f[2] for f in new[:n]])[ok])
        np.add.at(live["counts"], k[ok], 1)
        live["seen"] += n
        self.stack = np.where(live["counts"][:, None] > 0,
                              live["sums"] / np.maximum(live["counts"], 1)[:, None], np.nan)

    def tick(self):
        if not self.running:
            return
        self.live_scan_step()
        mm = self.latest.get("Distance_mm")
        if mm is not None:
            col = mask_column(mm)
            self.lbl_dist.config(text=f"{mm:.2f} mm from the switch")
            self.lbl_pos.config(text=f"Mask {col:.2f}" if -0.5 <= col <= NUM_POSITIONS - 0.5
                                else "Mask --")
            if self.shown_mm is None or abs(mm - self.shown_mm) > 0.05:
                self.shown_mm = mm
                self.draw_window(self.mask_canvas, col)
                for m, b in enumerate(self.goto_buttons):
                    b.config(bg=MASK_ON if abs(col - m) < 0.05 else self.goto_bg)
        if self.scene_saved:
            self.scene_saved = False
            for s, b in self.scene_buttons.items():
                colour = scene_colour(SAVE_DIR / s)
                b.config(bg=colour, selectcolor=colour)
        self.lbl_status.config(text=self.status)
        self.lbl_msg.config(text=self.message)
        errors = (FrameErrors.radar, FrameErrors.arduino)
        if errors != self.errors_shown:
            self.errors_shown = errors
            self.message = "Frame error - corrupted message discarded"
        self.lbl_errors.config(text=f"Frame errors: radar {errors[0]} · arduino {errors[1]}",
                               fg="#b00" if any(errors) else "#666")
        if SYNC_MODEL_FILE.exists():
            m = json.loads(SYNC_MODEL_FILE.read_text())
            self.lbl_model.config(text=f"Sync model: latency {m['latency_ms']:.1f} ms, backlash "
                                       f"{m['backlash_mm']:.3f} mm, offset {m['offset_mm']:.3f} mm")
        else:
            self.lbl_model.config(text="No sync model: run rail_sync_test.py Analyse first")
        state = "disabled" if self.busy else "normal"
        for b in (self.btn_capture, self.btn_calibrate, self.btn_radar, self.btn_arduino,
                  self.btn_export, self.btn_export_bare, self.btn_celebrate, self.btn_prev,
                  self.btn_next,
                  *self.goto_buttons, *self.scene_buttons.values()):
            b.config(state=state)
        self.chk_subtract.config(state="normal" if self.ambient is not None else "disabled")
        for b, a in zip(self.ambient_radios, AMBIENTS):
            have = (SAVE_DIR / a / SWEEP_FILE).exists()
            b.config(state="normal" if have and not self.busy else "disabled")
        self.root.after(rc.UI_MS, self.tick)

    # ---- Arduino and radar ----
    def reader(self):
        while self.running:
            try:
                raw = self.ser.readline()
            except serial.SerialException:
                self.message = "Arduino connection lost"
                return
            now = time.monotonic()
            line = raw.decode(errors="ignore").strip()
            if not line:
                continue
            m = POS_LINE.fullmatch(line)
            if m:
                us, um = int(m[1], 16), (int(m[2], 16) + 2 ** 31) % 2 ** 32 - 2 ** 31
                self.latest["Distance_mm"] = um / 1000
                log = self.pos_log
                if log is not None:
                    log.append((now, us, um))
            elif ARDUINO_LINE.fullmatch(line):
                self.latest["Distance_mm"] = float(line.partition(":")[2])
            elif ":" in line or not line.isprintable():
                FrameErrors.arduino += 1
            elif line.startswith("READY"):
                self.message = ""
                self.ready.set()
            else:
                self.message = line

    def radar_loop(self):
        """Streams continuously: feeds the live plot and logs frames during a sweep."""
        self.session_config = None   # a (re)connected client has open_radar's config
        try:
            while self.running and not self.radar_restart:
                if self.session_config is not self.sensor_config:
                    self.client.setup_session(self.sensor_config)
                    self.session_config = self.sensor_config
                self.client.start_session()
                self.session_started.clear()
                self.session_restart = False
                ref = None
                while self.running and not self.radar_restart and not self.session_restart:
                    result = self.client.get_next()
                    now = time.monotonic()
                    self.session_started.set()
                    # Restart when radar time lags the PC (backlog or drift), between sweeps
                    ref = ref or (result.tick_time, now)
                    lag = (now - ref[1]) - (result.tick_time - ref[0])
                    if abs(lag) > MAX_LAG_S and self.frame_log is None:
                        self.resyncs += 1
                        break
                    self.profile = fit_range(result.frame.mean(axis=0)[None])[0]
                    self.profile_dirty = True
                    self.radar_frames += 1
                    log = self.frame_log
                    if log is not None:
                        log.append((now, result.tick_time, self.profile))
                self.client.stop_session()
        except Exception as e:
            self.message = f"Radar stopped: {e}"

    def fresh_session(self):
        """Restart the radar session so a sweep starts with no frame backlog or drift."""
        self.session_restart = True
        while self.session_restart or not self.session_started.wait(0.1):
            if self.abort or not self.radar_thread.is_alive():
                raise ScanAborted
            time.sleep(0.01)

    def sweep_to(self, mm, speed=0.0):
        """Absolute move (firmware 'G'); speed 0 = the jog speed."""
        self.send(f"G{round(mm * 100)},{round(speed * 100)}\n")

    def wait_at(self, mm, settle=SETTLE_S, timeout=MOVE_TIMEOUT_S):
        deadline = time.time() + timeout
        settled_since = None
        while time.time() < deadline:
            if self.abort:
                raise ScanAborted
            if abs(self.latest.get("Distance_mm", -1) - mm) < rc.ARRIVE_TOLERANCE_MM:
                settled_since = settled_since or time.time()
                if time.time() - settled_since >= settle:
                    return
            else:
                settled_since = None
            time.sleep(0.02)
        raise ScanAborted(f"Timed out moving to {mm:.2f} mm")

    # ---- Jobs ----
    def capture(self):
        self.run_job(lambda: self.sweep_capture(SAVE_DIR / self.scene.get(),
                                                rc.scene_title(self.scene.get())))

    def sweep_capture(self, folder, name):
        """Sweep there and back at the set speed, then bin and save folder/sweep.npz."""
        if not SYNC_MODEL_FILE.exists():
            raise ScanAborted("No sync model: run rail_sync_test.py Analyse first")
        model = json.loads(SYNC_MODEL_FILE.read_text())
        self.message = ""
        speed = self.speed.get()
        if not 0 < speed <= MAX_SWEEP_SPEED_MM:
            raise ScanAborted(f"Speed must be between 0 and {MAX_SWEEP_SPEED_MM:g} mm/s")
        runup = speed ** 2 / (2 * SWEEP_ACCEL_MM) + RUNUP_MARGIN_MM
        ends = np.clip((GRID_MM[0] - runup, GRID_MM[-1] + runup), *TARGET_RANGE_MM)
        here = self.latest.get("Distance_mm", 0.0)
        start, end = ends if abs(here - ends[0]) <= abs(here - ends[1]) else ends[::-1]
        self.status = f"{name}: going to the start"
        self.sweep_to(start)
        self.wait_at(start)
        cfg = CAL_SENSOR_CONFIG if folder.parent == SAVE_DIR else SENSOR_CONFIG
        self.sensor_config = cfg
        self.fresh_session()
        errors_before = FrameErrors.radar
        cam_image = self.camera.image
        self.stack, self.stack_name, self.stack_align = None, name, None
        self.live = {"model": model, "seen": 0, "counts": np.zeros(len(GRID_MM), int),
                     "sums": np.zeros((len(GRID_MM), len(DISTANCES_M)), complex)}
        self.frame_log, self.pos_log = [], []
        try:
            for leg, (a, b) in enumerate(((start, end), (end, start))):
                self.status = f"{name}: sweep {leg + 1}/2 at {speed:g} mm/s"
                self.sweep_to(b, speed)
                self.wait_at(b, TAIL_S, abs(b - a) / speed + MOVE_TIMEOUT_S)
            frames, positions = self.frame_log, self.pos_log
        except BaseException:
            self.select_scene_data()   # back to the selected scene's saved sweep
            raise
        finally:
            self.frame_log = self.pos_log = self.live = None
            self.scan_line.set_visible(False)
            self.stack_dirty = True
        pc, tick, profiles = (np.array(x) for x in zip(*frames))
        profiles = profiles[:, :cfg.num_points]   # saved at the radar's own length
        pos_pc, us, um = np.array(positions).T
        raw = dict(frame_pc=pc, frame_tick=tick, pos_pc=pos_pc, pos_us=us.astype(np.int64),
                   pos_um=um.astype(np.int64))
        mm, towards = frame_positions(raw, model)
        slice_, counts = bin_frames(mm, profiles)
        shift, rho, shifts, curve = leg_alignment(mm, towards, profiles)
        folder.mkdir(parents=True, exist_ok=True)
        slice_ = fit_range(slice_)   # in memory at the full length, like loaded sweeps
        old_ambient = (read_ambient(folder / SWEEP_FILE)
                       if folder == SAVE_DIR / AMBIENT and (folder / SWEEP_FILE).exists() else None)
        save_npz(folder / SWEEP_FILE, slice=slice_[:, :cfg.num_points], counts=counts, grid_mm=GRID_MM,
                 distances_m=DISTANCES_M[:cfg.num_points], profiles=profiles, frame_mm=mm,
                 frame_towards=towards, speed_mm_s=speed, sync_model=json.dumps(model),
                 frame_errors=FrameErrors.radar - errors_before,
                 sensor_config=self.client.session_config.sensor_config.to_json(),
                 server_info=self.client.server_info.to_json(), time=time.time(),
                 align_shift_mm=shift, align_rho=rho, align_shifts_mm=shifts,
                 align_curve=curve, **raw)
        (folder / AMBIENT_COPY).unlink(missing_ok=True)
        if folder.name not in AMBIENTS and (SAVE_DIR / AMBIENT / SWEEP_FILE).exists():
            copy_ambient([folder], [read_ambient(SAVE_DIR / AMBIENT / SWEEP_FILE)])
        backfill = ""
        if old_ambient is not None:
            # sweeps on the old ambient (or none) get whichever of the old and new matches
            # them: a radar restart can flip the near-field phase before the re-capture
            n_old, n_new = copy_ambient([SAVE_DIR / s for s in SCENES]
                                        + [demux_folder(s, pg) for pg in demux_pages()
                                           for s in DEMUX_SLOTS if s != AMBIENT],
                                        [old_ambient, (slice_, time.time())])
            backfill = f". Sweeps on the old ambient: {n_old} kept, {n_new} moved to the new one"
        if folder == SAVE_DIR / self.ambient_source.get():
            self.ambient = slice_
        if folder == SAVE_DIR / self.scene.get():
            self.stack, self.stack_align = slice_, (shift, rho)
            self.stack_dirty = True
        else:
            self.select_scene_data()   # e.g. a demux sweep: back to the selected scene
        self.beam_dirty = True   # after the save, so the reload sees the new file
        self.scene_saved = True
        self.status = (f"Saved {folder.relative_to(SAVE_DIR).as_posix()}/{SWEEP_FILE}: "
                       f"{len(mm)} frames, {np.count_nonzero(counts)}/{len(GRID_MM)} bins filled, "
                       + alignment_text(shift, rho) + backfill)
        self.save_camera(folder, cam_image)
        warnings = []
        if abs(shift) > ALIGN_WARN_MM or rho < ALIGN_MIN_RHO:
            warnings.append(f"alignment {alignment_text(shift, rho)}: re-run the sync test "
                            "if this persists")
        if warnings:
            self.message = f"Check {name}: " + "; ".join(warnings)

    # ---- Camera ----
    def open_camera(self):
        if self.cam_window:
            self.cam_window.win.lift()
        else:
            self.cam_window = CameraWindow(self)

    def rotate_camera(self):
        """Turn the camera image a further 90 degrees clockwise. The grid corners belong
        to the old orientation, so they are cleared."""
        self.cam_rotation.set((self.cam_rotation.get() + 90) % 360)
        self.camera.rotation = self.cam_rotation.get()
        (camera_dir() / "corners.json").unlink(missing_ok=True)
        self.cam_corners = None

    def set_corners(self, points):
        camera_dir().mkdir(parents=True, exist_ok=True)
        (camera_dir() / "corners.json").write_text(json.dumps(points))
        self.cam_corners = points

    def save_camera(self, folder, image):
        """Save the snapshot with a sweep. Targets marked on an earlier snapshot of the same
        folder are kept (re-mark them in the demux window if the scene changed)."""
        if image is None:
            return
        f = folder / "camera.json"
        targets = json.loads(f.read_text()).get("targets", []) if f.exists() else []
        image.save(folder / "camera.jpg", quality=90)
        f.write_text(json.dumps({"targets": [{"x": t["x"], "y": t["y"]} for t in targets],
                                 "corners": self.cam_corners, "time": time.time()}, indent=1))

    def export(self, bare=False):
        lo, hi, subtract = self.beam_lo.get(), self.beam_hi.get(), self.subtract.get()
        normalise = self.normalise.get()
        ambient = self.ambient_source.get() if subtract else None

        def job():
            self.status = "Exporting beam heatmaps..."
            cube, sub = beam_cube(lo, hi, ambient)
            if normalise:
                cube = normalise_across_positions(cube)
            results = [(cube[:, :, grid_index(mask_mm(m))], sub) for m in range(NUM_POSITIONS)]
            if bare:
                out = export_beams_bare([p for p, _ in results], SAVE_DIR / "exports")
                self.status = f"Exported to {out.relative_to(SAVE_DIR.parent)}"
                return
            out = rc.export_beams(results, lo, hi, subtract, SAVE_DIR / "exports", normalise,
                                  rc.scene_title(ambient).lower() if ambient else "ambient")
            np.savez(out / "beam_cube.npz", cube=cube, grid_mm=GRID_MM, range_mm=(lo, hi),
                     normalised_across_positions=normalise, ambient_subtracted=sub)
            self.status = f"Exported to {out.relative_to(SAVE_DIR.parent)}"
        self.run_job(job)

    def open_demux(self):
        if self.demux:
            self.demux.win.lift()
        else:
            self.demux = ContinuousDemuxWindow(self)


class ContinuousDemuxWindow(rc.DemuxWindow):
    """radar_capture.DemuxWindow on sweeps: one row per carriage-position bin (GRID_MM)
    instead of one per mask. demux.py's CFAR and strategies A/B are unchanged."""

    def __init__(self, app):
        self.truths = []   # camera_truth of each marked target
        self.page = tk.IntVar(value=1)
        app.settings.bind("demux.page", self.page)
        if self.page.get() not in demux_pages():
            self.page.set(1)
        super().__init__(app)
        # Page tabs above the scenario buttons
        left = self.slot_buttons[AMBIENT].master.master
        self.tabs = tk.Frame(left)
        self.tabs.pack(before=left.winfo_children()[0], pady=(0, 4))
        self.draw_tabs()
        self.win.title("Demultiplexing (continuous)")
        self.btn_scan.config(text="Capture sweep")
        self.btn_capture.grid_remove()
        # Narrower left panel: no Margin column, slimmer columns and message
        self.table["displaycolumns"] = [c for c in self.table["columns"] if c != "Margin"]
        for c, w in (("#", 24), ("Range mm", 58), ("SNR dB", 46), ("Pixel", 38),
                     ("Bearing°", 54), ("Elev°", 44), ("Score", 46), ("Error px", 52)):
            self.table.column(c, width=w)
        self.lbl_msg.config(wraplength=320)
        # Own layout: top = range profile and slice | signature (narrow); bottom = score map
        # and 3D scene with the snapshot | camera snapshot (narrow, portrait)
        fig = self.canvas.figure
        fig.clear()
        fig.get_layout_engine().set(w_pad=0.06, wspace=0.08)   # room for colour bar labels
        upper, lower = fig.subfigures(2, 1, height_ratios=[1, 1.1])
        top, top_right = upper.subfigures(1, 2, width_ratios=[3, 1.4], wspace=0.04)
        bottom, bottom_right = lower.subfigures(1, 2, width_ratios=[3, 1.1], wspace=0.04)
        self.ax_prof, self.ax_img = top.subplots(2, 1, sharex=True,
                                                 gridspec_kw={"height_ratios": [2, 1.7]})
        self.ax_sig = top_right.subplots()
        gs = bottom.add_gridspec(1, 2, width_ratios=[1, 1.6])
        self.ax_score = bottom.add_subplot(gs[0])
        self.ax_3d = bottom.add_subplot(gs[1], projection="3d", computed_zorder=False)
        self.ax_3d.view_init(elev=20, azim=-60)
        self.ax_cam = bottom_right.subplots()
        dist_mm = DISTANCES_M * 1000
        half = (dist_mm[1] - dist_mm[0]) / 2
        self.mask_img = self.ax_img.imshow(
            np.full((len(GRID_MM), len(dist_mm)), np.nan), aspect="auto", cmap=app.sequential,
            interpolation="nearest", extent=(dist_mm[0] - half, dist_mm[-1] + half,
                                             GRID_MM[-1] + BIN_MM / 2, GRID_MM[0] - BIN_MM / 2))
        self.ax_img.set_ylabel("Carriage (mm)")
        self.ax_img.set_xlabel("Distance (mm)")
        top.colorbar(self.mask_img, ax=self.ax_img, label="|capture|")
        self.score_img = self.ax_score.imshow(np.full((9, 9), np.nan), cmap=SCORE_CMAP,
                                              interpolation="nearest")
        bottom.colorbar(self.score_img, ax=self.ax_score, shrink=0.6, pad=0.02)
        self.ax_score.set_xticks(range(9))
        self.ax_score.set_yticks(range(9))
        self.ax_score.tick_params(labelsize=7)
        self.overlays = []
        tk.Button(self.table.master, text="Export figures", command=self.export_figures).pack()
        tk.Button(self.table.master, text="Matching GIF (selected detection)",
                  command=self.export_matching_gif).pack()
        tk.Button(self.table.master, text="Dataflow GIF", command=self.export_dataflow_gif).pack()
        cam = tk.Frame(self.table.master)
        cam.pack(pady=2)
        plane = tk.Frame(self.table.master)
        plane.pack(pady=2)
        for frame, text, var in ((cam, "Camera back (mm)", app.cam_back_mm),
                                 (cam, "up (mm)", app.cam_up_mm),
                                 (plane, "Photo plane in 3D, forward (mm)", app.photo_plane_mm)):
            tk.Label(frame, text=text).pack(side="left", padx=(4, 2))
            tk.Entry(frame, textvariable=var, width=6).pack(side="left")
            var.trace_add("write", lambda *a: setattr(self, "dirty", True))
        boxes = tk.Frame(self.table.master)
        boxes.pack()
        for text, var in (("Calibration grid in 3D", app.grid_3d), ("3D grid lines", app.lines_3d)):
            tk.Checkbutton(boxes, text=text, variable=var,
                           command=lambda: setattr(self, "dirty", True)).pack(side="left", padx=4)
        fonts = tk.Frame(self.table.master)
        fonts.pack()
        tk.Label(fonts, text="Export text size (x)").pack(side="left", padx=(4, 2))
        tk.Entry(fonts, textvariable=app.export_font, width=5).pack(side="left")
        self.canvas.mpl_connect("button_press_event", self.on_cam_click)
        self.dirty = True

    def on_cam_click(self, event):
        """Left click on the camera snapshot marks a target, right click removes the nearest;
        True pixels follow the marks."""
        photo = self.photo()
        if event.inaxes is not self.ax_cam or not photo or event.button not in (1, 3):
            return
        img, corners, targets = photo
        h, w = img.shape[:2]
        x, y = event.xdata / w, event.ydata / h
        if event.button == 1:
            targets.append({"x": x, "y": y})
        elif targets:
            targets.remove(min(targets, key=lambda t: np.hypot((t["x"] - x) * w, (t["y"] - y) * h)))
        f = self.folder() / "camera.json"
        info = json.loads(f.read_text())
        info.update(targets=targets, corners=corners)
        f.write_text(json.dumps(info, indent=1))
        if not targets:
            self.truth.set("")
        self.save_info()   # compute() sets True pixels from the marks

    def camera(self):
        return np.array([0.0, -self.app.cam_back_mm.get(), self.app.cam_up_mm.get()])

    def truth_pixels(self):
        """The marked targets' true pixels (fractional) if any, else True pixels."""
        return [t["pixel"] for t in self.truths] if self.truths else super().truth_pixels()

    def folder(self, slot=None):
        return demux_folder(slot or self.slot.get(), self.page.get())

    def draw_tabs(self):
        for b in self.tabs.winfo_children():
            b.destroy()
        for pg in demux_pages():
            tk.Radiobutton(self.tabs, text=f"Page {pg}", variable=self.page, value=pg,
                           indicatoron=False, width=7, offrelief="raised",
                           command=self.select_page).pack(side="left", padx=1)
        tk.Button(self.tabs, text="+ New page", command=self.new_page).pack(side="left", padx=(6, 0))

    def new_page(self):
        pg = demux_pages()[-1] + 1
        demux_folder("", pg).mkdir(parents=True)
        self.page.set(pg)
        self.draw_tabs()
        self.select_page()

    def select_page(self):
        self.counts = {}
        self.select_slot()

    def select_slot(self):
        f = self.folder() / "info.json"
        info = json.loads(f.read_text()) if f.exists() else {}
        self.note.set(info.get("note", ""))
        self.truth.set(info.get("truth", ""))
        self.sel = 0
        self.dirty = True

    def save_info(self):
        folder = self.folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "info.json").write_text(json.dumps(
            {"note": self.note.get(), "truth": self.truth.get()}, indent=1))
        self.dirty = True

    def scan(self):
        slot, folder = self.slot.get(), self.folder()
        self.app.run_job(lambda: self.app.sweep_capture(folder, f"Demux p{self.page.get()} {slot}"))

    def compute(self):
        self.message = ""
        slot, strategy = self.slot.get(), self.strategy.get()
        lo, hi = self.r_lo.get(), self.r_hi.get()
        scale = 10 ** (self.thr_db.get() / 20)
        scn = load_slice(self.folder(slot))
        used = np.full(len(GRID_MM), scn is not None)
        if scn is None:
            scn = np.full((len(GRID_MM), len(DISTANCES_M)), np.nan)
        choice = self.scn_ambient.get()
        amb = None
        if choice != "none":
            amb = (load_slice(self.folder(AMBIENT)) if choice == "demux"
                   else sweep_ambient(self.folder(slot), choice))
            if amb is None and used.any() and slot != AMBIENT:
                self.message = "Scenario ambient not captured: not subtracted"
        if amb is not None:
            amb = np.where(np.isnan(amb), 0, amb)   # older ambients stop at 2000 mm
        self.y = np.abs(scn - (0 if amb is None else amb))   # positions x range bins
        self.y[:, np.isnan(self.y).all(axis=0)] = 0          # and older sweeps
        self.used = used
        dist_mm = DISTANCES_M * 1000
        self.profile = self.y.sum(axis=0) if used.any() else np.zeros(len(dist_mm))
        self.thr = demux.ca_cfar(self.profile, self.guard.get(), self.train.get(), scale)
        self.search = (dist_mm >= lo) & (dist_mm <= hi)
        self.floor = (self.min_peak.get() / 100 * self.profile[self.search].max()
                      if used.any() and self.search.any() else 0.0)
        peaks = (demux.detect(self.profile, self.thr, self.search, self.guard.get() + 1,
                              self.floor) if used.any() else [])

        # Calibration beams: one row per carriage position, one column per grid pixel
        cal_amb = self.cal_ambient.get()
        cube = beam_cube(self.cal_lo.get(), self.cal_hi.get(),
                         None if cal_amb == "none" else cal_amb)[0]
        B = cube.reshape(81, len(GRID_MM)).T
        self.B_cal = B   # every calibrated position, for the mask quality figures
        self.rows = used & np.isfinite(B).any(axis=1)
        self.factor = self.subpixel.get()
        self.B = demux.upsample_beams(B[self.rows], self.factor) if self.factor > 1 else B[self.rows]
        n = (demux.GRID_SIZE - 1) * self.factor + 1
        positions = np.array([divmod(i, n) for i in range(n * n)]) / self.factor
        self.results = []
        if peaks and not demux.valid_pixels(self.B).any():
            self.message += "No grid pixel has a calibration sweep - check the calibration range"
            peaks = []
        w = self.loc_bins.get()
        for k in peaks:
            y = self.y[self.rows, max(0, k - w):k + w + 1].sum(axis=1)
            r = demux.localise(self.B, y, strategy, positions)
            best = tuple(v / self.factor for v in divmod(r["best"], n))
            bearing, elevation = demux.pixel_angles(*best)
            r.update(bin=k, range_mm=dist_mm[k], peak=self.profile[k], y=y, pixel=best,
                     bearing=bearing, elevation=elevation,
                     xyz=demux.to_xyz(dist_mm[k], bearing, elevation),
                     snr_db=20 * np.log10(self.profile[k] * scale / self.thr[k]))
            self.results.append(r)
        self.sel = min(self.sel, max(len(self.results) - 1, 0))
        photo = self.photo()
        self.truths = ([camera_truth(photo[1], t, self.camera(), self.results) for t in photo[2]]
                       if photo else [])
        truths = self.truth_pixels()
        for r in self.results:
            r["error"] = min((np.hypot(r["pixel"][0] - t[0], r["pixel"][1] - t[1]) for t in truths),
                             default=np.nan)
        if self.truths:
            text = " ".join(dict.fromkeys(f"{min(max(round(r), 0), 8)}{min(max(round(c), 0), 8)}"
                                          for r, c in truths))
            if text != self.truth.get():
                self.truth.set(text)
                self.save_info()

        self.table.delete(*self.table.get_children())
        for i, r in enumerate(self.results):
            self.table.insert("", "end", iid=str(i), values=(
                i + 1, f"{r['range_mm']:.0f}", f"{r['snr_db']:.1f}",
                rc.pixel_label(r["pixel"]), f"{np.degrees(r['bearing']):.1f}",
                f"{np.degrees(r['elevation']):.1f}", f"{r['score'][r['best']]:.3f}",
                f"{r['margin']:.3f}" if strategy == "A" else f"{r['margin']:.2f}x",
                "" if np.isnan(r["error"]) else f"{r['error']:.1f}"))

    def draw(self):
        dist_mm = DISTANCES_M * 1000
        slot, strategy = self.slot.get(), self.strategy.get()
        for a in self.overlays:
            a.remove()
        self.overlays = []

        ax = self.ax_prof
        ax.cla()
        ax.plot(dist_mm, self.profile, label="profile")
        ax.plot(dist_mm, self.thr, "--", color="#d62728", linewidth=1, label="CFAR threshold")
        ax.axhline(self.floor, color="#d62728", linestyle=":", linewidth=1, label="min peak")
        lo, hi = self.r_lo.get(), self.r_hi.get()
        ax.axvspan(dist_mm[0], lo, color="#999", alpha=0.15, linewidth=0)
        ax.axvspan(hi, dist_mm[-1], color="#999", alpha=0.15, linewidth=0, label="not searched")
        for i, r in enumerate(self.results):
            ax.plot(r["range_mm"], r["peak"], "v", color="#d62728")
            ax.annotate(str(i + 1), (r["range_mm"], r["peak"]), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=9)
        ax.set_xlim(dist_mm[0], dist_mm[-1])
        ax.set_ylabel("Sum of |capture|")
        ax.set_title(f"{'Ambient' if slot == AMBIENT else slot}: "
                     f"{'sweep captured' if self.used.any() else 'no sweep yet'}, "
                     f"{len(self.results)} detections", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc=self.legend_loc())
        ax.tick_params(labelbottom=False)

        self.mask_img.set_data(self.y)
        if np.isfinite(self.y).any():
            self.mask_img.set_clim(np.nanmin(self.y), np.nanmax(self.y))
        for r in self.results:
            self.overlays.append(self.ax_img.axvline(r["range_mm"], color="white",
                                                     linestyle="--", linewidth=0.8))

        cmap = SCORE_CMAP if strategy == "A" else SCORE_CMAP.reversed()
        self.score_img.set_cmap(cmap)
        self.ax_sig.cla()
        truths = self.truth_pixels()
        if self.results:
            r = self.results[self.sel]
            n = (demux.GRID_SIZE - 1) * self.factor + 1
            score = r["score"].reshape(n, n)
            e = 0.5 / self.factor
            self.score_img.set_extent((-e, 8 + e, 8 + e, -e))
            self.ax_score.set_xlim(-e, 8 + e)
            self.ax_score.set_ylim(8 + e, -e)
            self.score_img.set_data(score)
            self.score_img.set_clim(np.nanmin(score), np.nanmax(score))
            self.overlays.append(self.ax_score.plot(r["pixel"][1], r["pixel"][0], "*",
                                                    color="white", markeredgecolor="k",
                                                    markersize=14)[0])
            self.ax_score.set_title(f"#{self.sel + 1} at {r['range_mm']:.0f} mm\n"
                                    + ("score" if strategy == "A" else "cost"), fontsize=9)
            mm = GRID_MM[self.rows]
            b = self.B[:, r["best"]]
            self.ax_sig.plot(mm, r["y"] / r["y"].mean(), lw=1, label="measured")
            self.ax_sig.plot(mm, b / b.mean(), lw=1, label=f"pixel {rc.pixel_label(r['pixel'])}")
            self.ax_sig.set_title("Signature", fontsize=9)
            self.ax_sig.set_ylabel("I / mean")
            self.ax_sig.set_xlabel("Carriage (mm)")
            self.ax_sig.legend(fontsize=7)
        else:
            self.score_img.set_data(np.full((9, 9), np.nan))
            self.ax_score.set_title("No detection", fontsize=9)
        for t in truths:
            self.overlays.append(self.ax_score.plot(t[1], t[0], "x", color="red",
                                                    markersize=10, mew=2)[0])

        photo = self.photo()
        self.plot_scene(self.ax_3d, photo)   # with the snapshot board and target cut-outs
        self.ax_cam.cla()
        if photo:
            self.plot_camera(self.ax_cam, photo)
            self.ax_cam.set_title("Click: mark target\nRight-click: remove", fontsize=8)
        else:
            self.ax_cam.text(0.5, 0.5, "No camera snapshot", ha="center", va="center",
                             transform=self.ax_cam.transAxes)
            self.ax_cam.set_axis_off()
        self.canvas.draw()

    # ---- Export ----
    def photo(self):
        """(image array, grid corners, camera targets) of this slot's camera snapshot, or
        None. The corners are those saved with it, else the current ones."""
        folder = self.folder()
        if not (folder / "camera.jpg").exists():
            return None
        info = json.loads((folder / "camera.json").read_text())
        corners = info.get("corners") or self.app.cam_corners
        return ((np.asarray(Image.open(folder / "camera.jpg")), corners, info.get("targets", []))
                if corners else None)

    def legend_loc(self):
        """Range profile legend: top middle when the last 700 mm has a peak reaching the
        top half (where the legend would sit), else top right."""
        dist_mm = DISTANCES_M * 1000
        late = self.profile[dist_mm >= dist_mm[-1] - 700]
        top = max(np.nanmax(self.profile), np.nanmax(self.thr), self.floor)
        return "upper center" if late.size and np.nanmax(late) > 0.5 * top else "upper right"

    def plot_profile(self, ax):
        dist_mm = DISTANCES_M * 1000
        lo, hi = self.r_lo.get(), self.r_hi.get()
        ax.plot(dist_mm, self.profile, label="profile")
        ax.plot(dist_mm, self.thr, "--", color="#d62728", linewidth=1, label="CFAR threshold")
        ax.axhline(self.floor, color="#d62728", linestyle=":", linewidth=1, label="min peak")
        ax.axvspan(dist_mm[0], lo, color="#999", alpha=0.15, linewidth=0)
        ax.axvspan(hi, dist_mm[-1], color="#999", alpha=0.15, linewidth=0, label="not searched")
        for i, r in enumerate(self.results):
            ax.plot(r["range_mm"], r["peak"], "v", color="#d62728")
            ax.annotate(str(i + 1), (r["range_mm"], r["peak"]), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=9)
        ax.set_xlim(dist_mm[0], dist_mm[-1])
        ax.set_xlabel("Distance (mm)")
        ax.set_ylabel("Sum of |capture|")
        ax.set_title(f"Range profile: {len(self.results)} detections", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc=self.legend_loc())

    def plot_slice(self, ax):
        dist_mm = DISTANCES_M * 1000
        half = (dist_mm[1] - dist_mm[0]) / 2
        img = ax.imshow(self.y, aspect="auto", cmap=self.app.sequential, interpolation="nearest",
                        extent=(dist_mm[0] - half, dist_mm[-1] + half,
                                GRID_MM[-1] + BIN_MM / 2, GRID_MM[0] - BIN_MM / 2))
        for r in self.results:
            ax.axvline(r["range_mm"], color="white", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Distance (mm)")
        ax.set_ylabel("Carriage (mm)")
        ax.set_title("Sweep slice", fontsize=10)
        ax.figure.colorbar(img, ax=ax, label="|capture|")

    def plot_score(self, ax, i, photo=None):
        """Score map of detection i; with photo, the camera image over it at 50%."""
        r, strategy = self.results[i], self.strategy.get()
        n = (demux.GRID_SIZE - 1) * self.factor + 1
        score = r["score"].reshape(n, n)
        e = 0.5 / self.factor
        cmap = SCORE_CMAP if strategy == "A" else SCORE_CMAP.reversed()
        img = ax.imshow(score, cmap=cmap, extent=(-e, 8 + e, 8 + e, -e),
                        interpolation="bicubic" if photo and np.isfinite(score).all() else "nearest")
        if photo:
            ax.imshow(photo[0], extent=photo_extent(photo[1]), alpha=0.5)
        ax.set_xlim(-e, 8 + e)
        ax.set_ylim(8 + e, -e)
        ax.plot(r["pixel"][1], r["pixel"][0], "*", color="white", markeredgecolor="k",
                markersize=14)
        for t in self.truth_pixels():
            ax.plot(t[1], t[0], "x", color="red", markersize=10, mew=2)
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.tick_params(labelsize=7)
        ax.set_title(f"#{i + 1} at {r['range_mm']:.0f} mm\n"
                     + ("score" if strategy == "A" else "cost")
                     + (" + camera" if photo else ""), fontsize=9)
        ax.figure.colorbar(img, ax=ax, shrink=0.8, pad=0.02)

    def plot_signature(self, ax, i):
        r = self.results[i]
        mm = GRID_MM[self.rows]
        b = self.B[:, r["best"]]
        ax.plot(mm, r["y"] / r["y"].mean(), lw=1, label="measured")
        ax.plot(mm, b / b.mean(), lw=1, label=f"pixel {rc.pixel_label(r['pixel'])}")
        ax.set_title(f"#{i + 1} signature", fontsize=9)
        ax.set_xlabel("Carriage (mm)")
        ax.set_ylabel("I / mean")
        ax.legend(fontsize=7)

    def plot_camera(self, ax, photo):
        """The snapshot with the grid points, the marked targets (circled) and the estimates
        (numbered)."""
        img, corners, targets = photo
        h, w = img.shape[:2]
        ax.imshow(img)
        grid = grid_to_image(corners, [(c, r) for r in range(9) for c in range(9)]) * (w, h)
        ax.scatter(grid[:, 0], grid[:, 1], s=4, color="#7fd4ff")
        r = target_radius(corners, w)
        for t in targets:
            ax.add_patch(matplotlib.patches.Circle((t["x"] * w, t["y"] * h), r, fill=False,
                                                   color="red", lw=2))
        for i, r in enumerate(self.results):
            x, y = grid_to_image(corners, (r["pixel"][1], r["pixel"][0])) * (w, h)
            ax.plot(x, y, "*", color="white", markeredgecolor="k", markersize=18)
            ax.annotate(f"#{i + 1} {r['range_mm']:.0f} mm", (x, y), textcoords="offset points",
                        xytext=(10, 6), color="white", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round", fc="k", alpha=0.5, lw=0))
        ax.set_axis_off()
        ax.set_title("Targets and estimates", fontsize=9)

    def plot_scene(self, ax, photo):
        """draw_scene's reconstruction on ax, with the snapshot on the calibration board.
        Each marked target is cut out of the board and moved along the camera's rays to
        its camera_truth position: scaled about the camera, so it covers the same angles
        from the camera as on the board. The snapshot itself is drawn at the "Photo plane"
        distance, likewise scaled about the camera (display only)."""
        self.ax_3d, live = ax, self.ax_3d
        ax.view_init(elev=live.elev, azim=live.azim)
        # with cut-outs, the true-direction rays end at them (drawn below) instead
        self.draw_scene([] if photo and self.truths else self.truth_pixels(), self.app.grid_3d.get())
        self.ax_3d = live
        ax.grid(self.app.lines_3d.get())
        spacing, centre, board = demux.GRID_SPACING_MM, demux.GRID_CENTRE, demux.GRID_DISTANCE_MM

        # Floor level with the bottom of the board image; shadows of the radar and its
        # estimates on it (the marked targets' shadows are added with their cut-outs)
        cam = self.camera()
        plane = self.app.photo_plane_mm.get()
        k_plane = (plane - cam[1]) / (board - cam[1])   # board -> photo plane, about the camera
        floor = cam[2] + k_plane * ((centre - 8.5) * spacing - cam[2])
        far = max(self.r_hi.get(), board, plane)
        half = centre * spacing * far / board   # as draw_scene's limits
        ax.set_xlim(-half, half)
        near = min(0.0, cam[1])
        z_lo, z_hi = min(floor, -half), max(half, cam[2])
        ax.set_ylim(near, far)
        ax.set_zlim(z_lo, z_hi)
        ax.set_box_aspect((2 * half, far - near, z_hi - z_lo))
        fx, fy = np.meshgrid([-half, half], [near, far])
        ax.plot_surface(fx, fy, np.full_like(fx, floor), color="k", alpha=0.08, shade=False,
                        linewidth=0, zorder=-2)

        def shadow(x, y, z, marker, size):
            ax.scatter(x, y, floor, marker=marker, s=size, color="k", alpha=0.35, linewidths=0,
                       depthshade=False, zorder=-1.5)
            ax.plot([x, x], [y, y], [z, floor], ":", color="k", alpha=0.3, linewidth=0.8, zorder=-1.5)

        shadow(0, 0, 0, "^", 50)
        if photo:
            ax.scatter(*cam, marker="s", color="#555", s=40, depthshade=False)
            ax.text(*cam, "   camera", fontsize=8, ha="left", va="center")
            shadow(*cam, "s", 40)
        for r in self.results:
            shadow(*r["xyz"], "o", 40)
        if not photo:
            return
        img, corners, targets = photo
        h, w = img.shape[:2]
        r = target_radius(corners, w)
        cut = [(t["x"] * w, t["y"] * h, r) for t in targets]   # circles in image pixels

        def surface(xy, scale, alpha, zorder):
            """Image pixels xy (n, n, 2) drawn at their board positions, scaled by scale
            about the camera."""
            col, row = np.moveaxis(image_to_grid(corners, xy / (w, h)), -1, 0)
            px = np.clip(xy[..., 0].astype(int), 0, w - 1)
            py = np.clip(xy[..., 1].astype(int), 0, h - 1)
            colours = np.concatenate([img[py, px, :3] / 255, alpha[..., None]], axis=-1)
            pts = cam + scale * (np.moveaxis(board_point(row, col), 0, -1) - cam)
            ax.plot_surface(*np.moveaxis(pts, -1, 0), facecolors=colours, rstride=1,
                            cstride=1, shade=False, linewidth=0, antialiased=False, zorder=zorder)

        g = np.linspace(-0.5, 8.5, 90)
        xy = grid_to_image(corners, np.stack(np.meshgrid(g, g), axis=-1)) * (w, h)
        alpha = np.full(xy.shape[:2], BOARD_ALPHA)
        for x, y, r in cut:   # holes where the targets were
            alpha[np.hypot(xy[..., 0] - x, xy[..., 1] - y) < r] = 0
        surface(xy, k_plane, alpha, -1)
        for (x, y, r), t, truth in zip(cut, targets, self.truths):
            d = np.linspace(-r, r, 30)
            dx, dy = np.meshgrid(d, d)
            surface(np.stack([x + dx, y + dy], axis=-1), truth["s"],
                    np.where(np.hypot(dx, dy) < r, 1.0, 0.0), 0)
            tx, ty, tz = truth["xyz"]
            mark = cam + k_plane * (board_point(*image_to_grid(corners, (t["x"], t["y"]))[::-1]) - cam)
            ax.plot(*np.transpose([cam, mark]), ":", color="#555", linewidth=0.8, zorder=1)
            shadow(tx, ty, tz, "s", 40)

    def plot_ranges(self, ax):
        """Radar, one semi-transparent range shell (over the grid's field of view) per
        detection, and on it the estimate and the camera truth at that range (the marked
        target whose radar pixel is nearest the estimate's), joined with the error in mm."""
        photo = self.photo()
        cam = self.camera()
        colours = matplotlib.colormaps["plasma"]
        lo, hi = self.r_lo.get(), self.r_hi.get()
        ax.scatter(0, 0, 0, marker="^", color="k", s=60, depthshade=False)
        ax.text(0, 0, 0, "\nradar", fontsize=8, ha="center", va="top")
        g = np.linspace(-0.5, 8.5, 30)
        rows, cols = np.meshgrid(g, g)
        bearing, elevation = demux.pixel_angles(rows, cols)
        far = max([r["range_mm"] for r in self.results] + [500.0])
        legend = []
        for i, r in enumerate(self.results):
            c = colours((r["range_mm"] - lo) / max(hi - lo, 1))
            ax.plot_surface(*demux.to_xyz(r["range_mm"], bearing, elevation), color=c, alpha=0.15,
                            shade=False, linewidth=0, zorder=0)
            e = np.asarray(r["xyz"])
            ax.scatter(*e, color=c, edgecolor="k", s=70, depthshade=False, zorder=2)
            legend.append(matplotlib.patches.Patch(color=c, alpha=0.5,
                                                   label=f"#{i + 1}: {r['range_mm']:.0f} mm"))
            if not photo or not photo[2]:
                continue
            truth = min((camera_truth(photo[1], t, cam, [r]) for t in photo[2]),
                        key=lambda t: np.hypot(t["pixel"][0] - r["pixel"][0],
                                               t["pixel"][1] - r["pixel"][1]))
            t = truth["xyz"]
            ax.scatter(*t, marker="X", color="red", s=70, depthshade=False, zorder=2)
            ax.plot(*np.transpose([e, t]), color="k", linewidth=1.2, zorder=1)
            ax.text(*(e + t) / 2, f" {np.linalg.norm(t - e):.0f} mm", fontsize=9,
                    fontweight="bold", zorder=3)
        half = demux.GRID_CENTRE * demux.GRID_SPACING_MM * far / demux.GRID_DISTANCE_MM
        ax.set_xlim(-half, half)
        ax.set_ylim(0, far)
        ax.set_zlim(-half, half)
        ax.set_box_aspect((2 * half, far, 2 * half))
        ax.view_init(elev=self.ax_3d.elev, azim=self.ax_3d.azim)
        ax.set_xlabel("Across (mm)", fontsize=8)
        ax.set_ylabel("Forward (mm)", fontsize=8)
        ax.set_zlabel("Up (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title("Estimates vs camera truth", fontsize=9)
        ax.grid(self.app.lines_3d.get())
        if legend:
            ax.legend(handles=legend, title="Range shell", fontsize=8, title_fontsize=8,
                      loc="upper left")

    def export_matching_gif(self):
        """GIF of how the selected detection's pixel is found: the fingerprint (intensity
        at its range vs mask position) draws in under the moving mask strip; then each
        calibrated grid point's signature is laid over it in turn, its score filling in
        the 9x9 map; last, the full (sub-pixel) map with the best pixel, its bearing and
        elevation, held. demux/<slot>/matching_<time>.gif."""
        if not self.results:
            self.message = "No detection to animate"
            return
        r, strategy, factor = self.results[self.sel], self.strategy.get(), self.factor
        B, sel = self.B, self.sel
        x = mask_column(GRID_MM[self.rows])   # mask position: rises as the carriage nears
        order = np.argsort(x)
        x, y = x[order], (r["y"] / r["y"].mean())[order]
        n = (demux.GRID_SIZE - 1) * factor + 1
        full = r["score"].reshape(n, n)
        coarse = full[::factor, ::factor]     # the 81 calibration points
        valid = demux.valid_pixels(B).reshape(n, n)[::factor, ::factor]
        cmap = SCORE_CMAP if strategy == "A" else SCORE_CMAP.reversed()
        norm = matplotlib.colors.Normalize(np.nanmin(full), np.nanmax(full))
        word = "score" if strategy == "A" else "cost"
        best = r["pixel"]
        out = self.folder() / time.strftime("matching_%Y%m%d_%H%M%S.gif")
        truths = self.truth_pixels()
        sig = B / B.mean(axis=0)
        font = self.app.export_font.get()

        def job():
            with matplotlib.rc_context({"font.size": 2 * matplotlib.rcParams["font.size"]}):
                render()

        def render():
            fig = Figure(figsize=(16, 9), dpi=70)
            FigureCanvasAgg(fig)
            gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 2, 2], width_ratios=[1.6, 1],
                                  left=0.07, right=0.94, bottom=0.09, top=0.9,
                                  hspace=0.45, wspace=0.18)
            set_strip = mask_strip(fig.add_subplot(gs[0, 0]))
            ax_f = fig.add_subplot(gs[1, 0])
            ax_c = fig.add_subplot(gs[2, 0], sharex=ax_f, sharey=ax_f)
            ax_s = fig.add_subplot(gs[:, 1])
            for ax in (ax_f, ax_c):
                ax.set_ylabel("I / mean")
                ax.grid(alpha=0.3)
            ax_f.set_xlim(x.min(), x.max())
            lo_y = min(np.nanmin(y), np.nanmin(sig))
            hi_y = max(np.nanmax(y), np.nanmax(sig))
            ax_f.set_ylim(lo_y - 0.05 * (hi_y - lo_y), hi_y + 0.05 * (hi_y - lo_y))
            ax_f.tick_params(labelbottom=False)
            ax_c.set_xlabel("Mask position")
            ax_f.set_title(f"Fingerprint at {r['range_mm']:.0f} mm")
            meas, = ax_f.plot([], [], color="k", lw=2)
            cursor = ax_f.axvline(x[0], color="#e31a1c", lw=2)
            ref, = ax_c.plot(x, y, color="#888", lw=2, label="measured")
            cand, = ax_c.plot([], [], color="#ff7f0e", lw=2, label="candidate")
            ax_c.legend(loc="upper right", fontsize="small")
            title_c = ax_c.set_title(" ")
            img = ax_s.imshow(np.full((9, 9), np.nan), cmap=cmap, norm=norm,
                              extent=(-0.5, 8.5, 8.5, -0.5), interpolation="nearest")
            bar = fig.colorbar(img, ax=ax_s, shrink=0.8, location="bottom", pad=0.1,
                               label=word)
            bar.locator = matplotlib.ticker.MaxNLocator(4)
            bar.update_ticks()
            ax_s.set_xticks(range(0, 9, 4))
            ax_s.set_yticks(range(0, 9, 4))
            ax_s.set_ylabel("Scene row")
            mark, = ax_s.plot([], [], "s", ms=26, mfc="none", mec="#ff7f0e", mew=3)
            star, = ax_s.plot([], [], "*", color="white", mec="k", ms=30)
            for t in truths:
                ax_s.plot(t[1], t[0], "x", color="red", ms=18, mew=3)
            title_s = ax_s.set_title("Score map")
            fig.suptitle(f"Detection #{sel + 1}: fingerprint matching")

            def final():
                meas.set_data(x, y)
                cursor.set_visible(False)
                set_strip(x[-1])
                b = sig[:, r["best"]]
                cand.set_data(x, b[order])
                title_c.set_text(f"Best: pixel {rc.pixel_label(best)}, "
                                 f"{word} {r['score'][r['best']]:.3f}")
                e = 0.5 / factor
                img.set_data(full)
                img.set_extent((-e, 8 + e, 8 + e, -e))
                img.set_interpolation("bicubic" if np.isfinite(full).all() else "nearest")
                mark.set_data([], [])
                star.set_data([best[1]], [best[0]])
                title_s.set_text(f"bearing {np.degrees(r['bearing']):+.1f}\N{DEGREE SIGN}\n"
                                 f"elevation {np.degrees(r['elevation']):+.1f}\N{DEGREE SIGN}")

            final()
            scale_fonts(fig, font)
            gif = GifFrames(fig)   # palette from the finished picture
            star.set_data([], [])
            title_s.set_text("Score map")
            img.set_interpolation("nearest")
            img.set_extent((-0.5, 8.5, 8.5, -0.5))
            img.set_data(np.full((9, 9), np.nan))
            cand.set_data([], [])
            title_c.set_text(" ")
            # 1: the fingerprint draws in under the moving strip
            cursor.set_visible(True)
            ref.set_visible(False)
            for f in range(1, MATCH_DRAW_FRAMES + 1):
                k = max(2, round(f / MATCH_DRAW_FRAMES * len(x)))
                meas.set_data(x[:k], y[:k])
                cursor.set_xdata([x[k - 1]] * 2)
                set_strip(x[k - 1])
                gif.add(fig, MATCH_FRAME_MS)
                self.app.status = f"Matching GIF: frame {len(gif.frames)}"
            cursor.set_visible(False)
            ref.set_visible(True)
            # 2: each calibration point in turn
            shown = np.full((9, 9), np.nan)
            for i in range(9):
                for j in range(9):
                    mark.set_data([j], [i])
                    if valid[i, j]:
                        cand.set_data(x, sig[:, (i * factor) * n + j * factor][order])
                        shown[i, j] = coarse[i, j]
                        title_c.set_text(f"Pixel {i}{j}: {word} {coarse[i, j]:.3f}")
                    else:
                        cand.set_data([], [])
                        title_c.set_text(f"Pixel {i}{j}: not calibrated")
                    img.set_data(shown)
                    gif.add(fig, MATCH_FRAME_MS)
                    self.app.status = f"Matching GIF: frame {len(gif.frames)}"
            # 3: the result
            final()
            gif.add(fig, MATCH_HOLD_MS)
            gif.save(out)
            self.app.status = f"Saved {out.relative_to(SAVE_DIR.parent).as_posix()}"
        self.app.run_job(job)

    def export_dataflow_gif(self):
        """Presentation GIF of the whole pipeline, 3 x 2 panels joined by arrows. Top left:
        the mask with the window sliding over it and, above it in the radar's cone, the
        scene (only the camera crops of the marked targets). Middle left: |capture| over
        the demux range vs mask position, drawn in under the window. Middle right: the
        range profile on its side, growing; at the end the detections A, B, ... Bottom
        left (at the end): each detection's measured fingerprint, a normalised strip. Bottom
        right (static): each direction's calibration fingerprint in a bearing / elevation
        grid, the best match of each detection outlined. Top right: the estimates (x) in
        the scene. demux/<slot>/dataflow_<time>.gif."""
        if not self.used.any():
            self.message = "No sweep to animate"
            return
        dist_mm = DISTANCES_M * 1000
        lo, hi = self.r_lo.get(), self.r_hi.get()
        band = (dist_mm >= lo) & (dist_mm <= hi)
        rng = dist_mm[band]
        x = mask_column(GRID_MM)
        order = np.argsort(x)
        x = x[order]
        Y = self.y[order][:, band]                 # positions x ranges
        B = self.B_cal[order]                      # positions x 81 pixels
        sig = B / B.mean(axis=0)
        cmap = self.app.sequential.with_extremes(bad="white")
        strategy, factor = self.strategy.get(), self.factor
        n = (demux.GRID_SIZE - 1) * factor + 1
        letters = [chr(65 + i) for i in range(len(self.results))]
        # each detection's calibration cell: of the (up to 4) cells around its sub-pixel
        # estimate, the best scoring. (The best cell overall can be far away: with
        # interpolated beams the estimate is chosen on the fine grid.)
        matches = {}                               # calibration pixel -> "AB"
        for r, ch in zip(self.results, letters):
            coarse = r["score"].reshape(n, n)[::factor, ::factor]
            near = [{int(np.floor(v)), int(np.ceil(v))} for v in r["pixel"]]
            sign = 1 if strategy == "A" else -1   # B: a cost, lower is better
            row, col = max(((i, j) for i in near[0] for j in near[1]),
                           key=lambda c: sign * np.nan_to_num(coarse[c], nan=-sign * np.inf))
            p = row * 9 + col
            matches[p] = matches.get(p, "") + ch
        xm = mask_column(GRID_MM[self.rows])       # the measured fingerprints' positions
        meas_order = np.argsort(xm)
        xm = xm[meas_order]

        def unit(v):
            return (v - np.min(v)) / (np.max(v) - np.min(v))

        photo = self.photo()
        out = self.folder() / time.strftime("dataflow_%Y%m%d_%H%M%S.gif")
        font = self.app.export_font.get()
        S, ZOOM = 6.0, 1.4           # scene width, crop enlargement
        cell = S / 9
        mid_y = 0.0                  # scene centre height: set below, once the targets are known
        # field of view: a sector cut flat where it meets the top of the 4-wide window
        half = np.radians(10) + np.arctan(0.25)              # half angle
        apex = 4 - 2 / np.tan(half)                          # where the edges would meet

        def scene_xy(col, row):
            return (np.asarray(col) - 4) * cell, mid_y - (np.asarray(row) - 4) * cell

        # each marked target labelled with the detection nearest it (from the radar)
        names = [min(zip(letters, self.results),
                     key=lambda lr: np.hypot(lr[1]["pixel"][0] - t["pixel"][0],
                                             lr[1]["pixel"][1] - t["pixel"][1]))[0]
                 if self.results else "" for t in self.truths]
        crops = []   # (rgba, extent relative to the scene centre)
        if photo:
            img, corners, targets = photo
            h, w = img.shape[:2]
            rad = target_radius(corners, w)
            for t in targets:
                cx, cy = t["x"] * w, t["y"] * h
                x0, x1 = int(max(cx - rad, 0)), int(min(cx + rad, w))
                y0, y1 = int(max(cy - rad, 0)), int(min(cy + rad, h))
                yy, xx = np.mgrid[y0:y1, x0:x1]
                alpha = (np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) < rad) * 255
                rgba = np.dstack([img[y0:y1, x0:x1, :3], alpha]).astype(np.uint8)
                (c0, r0), (c1, r1), (cc, rc_) = image_to_grid(
                    corners, [(x0 / w, y0 / h), (x1 / w, y1 / h), (t["x"], t["y"])])
                (l, r_), (t_, b) = scene_xy([c0, c1], [r0, r1])
                ox, oy = scene_xy(cc, rc_)
                crops.append((rgba, [ox + ZOOM * (l - ox), ox + ZOOM * (r_ - ox),
                                     oy + ZOOM * (b - oy), oy + ZOOM * (t_ - oy)]))

        # the scene as low as it goes with every target crop and estimate inside the sector;
        # the sector just reaches over them
        pts = [(c0, c1) for _, e in crops for c0 in e[:2] for c1 in e[2:]]
        for r in self.results:
            ex, ey = scene_xy(r["pixel"][1], r["pixel"][0])
            pts += [(ex + sx, ey + sy) for sx in (-0.4, 0.4) for sy in (-0.4, 0.4)]
        px, py = np.array(pts or [(0.0, 0.0)]).T
        tan_in = np.tan(half - np.radians(3))
        mid_y = next(c for c in np.arange(4.5, 40, 0.1)
                     if (py + c >= 4.4).all() and (np.abs(px) <= (py + c - apex) * tan_in).all())
        for _, e in crops:
            e[2] += mid_y
            e[3] += mid_y
        R = 1.05 * max(np.hypot(px, py + mid_y - apex).max(), 4 - apex + 2)
        a = np.linspace(np.pi / 2 + half, np.pi / 2 - half, 60)
        cone_xy = np.vstack([[-2, 4], np.column_stack([R * np.cos(a), apex + R * np.sin(a)]),
                             [2, 4]])
        y_top = apex + R + 1.3   # room for "Scene" over the sector

        def job():
            size = 2 * matplotlib.rcParams["font.size"]
            with matplotlib.rc_context({"font.size": size, "axes.titlesize": "medium",
                                        "axes.labelsize": 0.88 * size}):
                render()

        def render():
            # layout in inches; the left column's x axes all at u inches per mask position
            span = x[-1] - x[0]
            L, MW = 2.0, 9.0
            u = MW / span
            tl_h = (y_top + 0.3) * u
            RX, RW = 13.6, 6.8                           # right column
            line_h = matplotlib.rcParams["font.size"] * font / 72   # one line of text (in)
            G_SIZE, G_H = 6.3, 4.4                       # bearing / elevation grid: cells wide
            G_Y = 0.3 + 2.2 * line_h                     # room for its ticks and "Bearing"
            F_H = 3.6                                    # measured fingerprints height
            y_t, y_m, y_f = 13.1, 8.4, 2.6               # top, middle, bottom rows
            W, H = RX + RW + 1.1, y_t + tl_h + 0.4 + 1.5 * line_h   # room for the titles
            fig = Figure(figsize=(W, H), dpi=DATAFLOW_DPI)
            FigureCanvasAgg(fig)

            def axes(x0, y0, w, h, **kw):
                return fig.add_axes((x0 / W, y0 / H, w / W, h / H), **kw)

            ax_t = axes(L - 4.5 * u, y_t, (span + 11) * u, tl_h)
            ax_h = axes(L, y_m, MW, 3.6)
            ax_p = axes(RX, y_m, RW, 3.6, sharey=ax_h)
            ax_f = axes(L, y_f, MW, F_H, sharex=ax_h)
            ax_g = axes(RX + (RW - G_SIZE) / 2, G_Y, G_SIZE, G_H)
            tr_xlim = R * np.sin(half) + 0.5
            tr_top = apex + R + 0.3   # no "Scene" text here
            tr_w = min(RW, tl_h * 2 * tr_xlim / (tr_top - 3.7))
            tr_h = tr_w * (tr_top - 3.7) / (2 * tr_xlim)
            ax_r = axes(RX + (RW - tr_w) / 2, y_t + (tl_h - tr_h) / 2, tr_w, tr_h)
            rows_m, cols_m = MASK.shape
            mask_cmap = matplotlib.colors.ListedColormap([MASK_OFF, MASK_ON])

            def cone(ax):
                return ax.add_patch(matplotlib.patches.Polygon(
                    cone_xy, fc="#ffd54f", ec="#c9a000", alpha=0.35, lw=2, zorder=0))

            # top left: the window slides over the mask, the scene with it
            ax_t.imshow(MASK, cmap=mask_cmap, vmin=0, vmax=1, interpolation="nearest",
                        extent=(-2, cols_m - 2, 0, rows_m), zorder=2)
            dims = [ax_t.add_patch(matplotlib.patches.Rectangle((0, 0), 0, rows_m, fc="white",
                                                                alpha=0.55, lw=0, zorder=3))
                    for _ in range(2)]
            window = ax_t.add_patch(matplotlib.patches.Rectangle(
                (0, 0), 4, rows_m, fill=False, ec="#e31a1c", lw=3, zorder=4))
            fov = cone(ax_t)
            crop_art = [ax_t.imshow(rgba, extent=e, zorder=2) for rgba, e in crops]
            crop_txt = [ax_t.text(0, 0, name, color="#e31a1c", fontweight="bold", va="center",
                                  zorder=3) for name in names]
            scene_txt = ax_t.text(0, apex + R + 0.15, "Scene", ha="center", va="bottom")
            ax_t.text(-2.3, rows_m / 2, "Mask", rotation=90, ha="right", va="center",
                      fontsize="medium")   # full size, unlike the axis labels
            ax_t.set_xlim(x[0] - 4.5, x[-1] + 6.5)   # room for the cone at the end
            ax_t.set_ylim(-0.3, y_top)
            ax_t.set_axis_off()

            def set_top(m):
                dims[0].set_bounds(-2, 0, m, rows_m)
                dims[1].set_bounds(m + 2, 0, cols_m - 4 - m, rows_m)
                window.set_x(m - 2)
                fov.set_xy(cone_xy + (m, 0))
                scene_txt.set_x(m)
                for a, (_, e) in zip(crop_art, crops):
                    a.set_extent((e[0] + m, e[1] + m, e[2], e[3]))
                for t, (_, e) in zip(crop_txt, crops):
                    t.set_position((e[1] + m + 0.1, (e[2] + e[3]) / 2))

            # top right: the field of view with the estimates as x
            cone(ax_r)
            ax_r.set_xlim(-tr_xlim, tr_xlim)
            ax_r.set_ylim(3.7, tr_top)
            ax_r.set_axis_off()
            ax_r.set_title("3D Estimations")
            final = []
            for r, ch in zip(self.results, letters):
                ex, ey = scene_xy(r["pixel"][1], r["pixel"][0])
                final += ax_r.plot(ex, ey, "x", color="#e31a1c", ms=16, mew=4, zorder=5)
                final.append(ax_r.text(ex + 0.35, ey + 0.2, ch, color="#e31a1c",
                                       fontweight="bold", zorder=5))

            # middle left: |capture| over range vs mask position, drawn in
            dx = (x[1] - x[0]) / 2
            dr = (rng[1] - rng[0]) / 2
            full = Y.T
            heat = ax_h.imshow(np.full_like(full, np.nan), cmap=cmap, aspect="auto",
                               origin="lower", interpolation="nearest",
                               vmin=np.nanmin(full), vmax=np.nanmax(full),
                               extent=(x[0] - dx, x[-1] + dx, rng[0] - dr, rng[-1] + dr))
            cursor = ax_h.axvline(x[0], color="#e31a1c", lw=2)
            ax_h.set_xlim(x[0], x[-1])
            ax_h.set_ylim(rng[0], rng[-1])
            ax_h.set_ylabel("Range (m)")
            ax_h.set_title("Range Intensities")
            ax_h.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            ax_h.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v / 1000:g}"))
            ax_h.tick_params(labelbottom=False)

            # middle right: range profile on its side
            prof = Y.sum(axis=0)
            line, = ax_p.plot(np.zeros_like(rng), rng, color="k", lw=2)
            ax_p.set_xlim(0, 1.3 * prof.max())
            ax_p.set_xticks([])
            ax_p.tick_params(labelleft=False)
            ax_p.set_xlabel("Amplitude")
            ax_p.set_title("Range Profile")
            # the detections (known from the full sweep) marked from the start; the markers
            # ride on the growing profile
            peaks = []   # (column of Y, marker, label)
            for r, ch in zip(self.results, letters):
                fy = (y_m + (r["range_mm"] - rng[0]) / (rng[-1] - rng[0]) * 3.6) / H
                fig.add_artist(matplotlib.lines.Line2D(   # slice to profile
                    [L / W, (RX + RW) / W], [fy, fy], color="#e31a1c", ls="--", lw=2))
                mark, = ax_p.plot(r["peak"], r["range_mm"], "<", color="#e31a1c", ms=14)
                label = ax_p.annotate(ch, (r["peak"], r["range_mm"]), textcoords="offset points",
                                      xytext=(14, 0), va="center", color="#e31a1c",
                                      fontweight="bold")
                peaks.append((int(np.argmin(np.abs(rng - r["range_mm"]))), mark, label))

            # bottom left: each detection's measured fingerprint (its range slice), normalised
            if not self.results:
                final.append(ax_f.text(0.5, 0.5, "No detections", ha="center", va="center",
                                       transform=ax_f.transAxes))
            strips = []   # (normalised fingerprint, image): drawn in with the scan
            for k, (r, ch) in enumerate(zip(self.results, letters)):
                i = len(self.results) - 1 - k   # furthest at the top, as range above
                v = unit(r["y"][meas_order])
                strips.append((v, ax_f.imshow(v[None], cmap=cmap, aspect="auto",
                                              interpolation="nearest", vmin=0, vmax=1,
                                              extent=(xm[0] - dx, xm[-1] + dx, i + 0.32, i - 0.32))))
                ax_f.text(-0.015, i, ch, transform=ax_f.get_yaxis_transform(), ha="right",
                          va="center", color="#e31a1c", fontweight="bold")
            ax_f.set_xlim(x[0], x[-1])
            ax_f.set_ylim(max(len(self.results), 1) - 0.5, -0.5)
            ax_f.set_yticks([])
            ax_f.set_ylabel("Detection", labelpad=30 + 0.3 * line_h * 72)   # clear of the letters
            ax_f.set_title("Measured Fingerprints")
            ax_f.set_xlabel("Mask position (mm)")
            ax_f.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50 / rc.STEP_MM))
            ax_f.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v * rc.STEP_MM:.0f}"))
            for side in ("left", "right", "top"):
                ax_f.spines[side].set_visible(False)

            # bottom right: each direction's calibration fingerprint in a bearing / elevation
            # grid, the matched directions outlined in red
            for k in range(1, 9):
                ax_g.axhline(k - 0.5, color="#ddd", lw=1)
                ax_g.axvline(k - 0.5, color="#ddd", lw=1)
            for p in range(81):
                row, col = divmod(p, 9)
                ax_g.imshow(unit(sig[:, p])[None], cmap=cmap, aspect="auto",
                            interpolation="bilinear", vmin=0, vmax=1,
                            extent=(col - 0.46, col + 0.46, row + 0.42, row - 0.42))
                if p in matches:   # after the scan
                    final.append(ax_g.add_patch(matplotlib.patches.Rectangle(
                        (col - 0.5, row - 0.5), 1, 1, fill=False, ec="#e31a1c", lw=3, zorder=3)))
                    final.append(ax_g.text(
                        col, row, matches[p], ha="center", va="center", color="#e31a1c",
                        fontweight="bold", fontsize="small", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.85, lw=0)))
            ax_g.set_xlim(-0.5, 8.5)
            ax_g.set_ylim(8.5, -0.5)
            ends = [0, 4, 8]
            ax_g.set_xticks(ends, [f"{np.degrees(demux.pixel_angles(4, k)[0]):+.0f}\N{DEGREE SIGN}"
                                   for k in ends])
            ax_g.set_yticks(ends, [f"{np.degrees(demux.pixel_angles(k, 4)[1]):+.0f}\N{DEGREE SIGN}"
                                   for k in ends])
            ax_g.set_xlabel("Bearing")
            ax_g.set_ylabel("Elevation", labelpad=4 - 0.6 * line_h * 72)   # a character closer
            ax_g.set_title("Fingerprints by Direction")

            # dataflow arrows (inches)
            def arrow(*pts):
                fig.add_artist(matplotlib.patches.FancyArrowPatch(
                    path=matplotlib.path.Path(np.array(pts) / (W, H)), transform=fig.transFigure,
                    arrowstyle="-|>", mutation_scale=30, lw=3, color="#555"))

            g_x0, g_x1, g_mid = RX + (RW - G_SIZE) / 2, RX + (RW + G_SIZE) / 2, G_Y + G_H / 2

            # soft vertical grid lines through the left column, at the mask position ticks
            for t in np.arange(0, x[-1], 50 / rc.STEP_MM):
                if t >= x[0]:
                    fx = (L + (t - x[0]) * u) / W
                    fig.add_artist(matplotlib.lines.Line2D(
                        [fx, fx], [y_f / H, (y_t + 4.3 * u) / H], color="#bbb", lw=1,
                        alpha=0.6, zorder=3))
            t_mid = y_t + tl_h / 2
            bar = RX + RW + 0.7
            col_x = L + MW / 5   # left column arrows, clear of the centred titles
            arrow((col_x, y_t + 0.05), (col_x, y_m + 3.7))                 # scan -> captures
            chars = 1.2 * matplotlib.rcParams["font.size"] * font / 72   # ~2 characters (in)
            arrow((col_x - chars, y_m - 0.1), (col_x - chars, y_f + F_H + 0.1))   # -> fingerprints
            arrow((RX + 0.4, y_m - 0.15), (L + MW + 0.4, y_f + F_H + 0.2))  # ranges -> fingerprints
            arrow((L + MW + 0.4, y_f + 0.6 * F_H + 1.5 * line_h),
                  (g_x0 - 0.2 - 2.6 * line_h, y_f + 0.6 * F_H + 1.5 * line_h))   # short of the ticks  # -> directions
            arrow((g_x1 + 0.15, g_mid), (bar, g_mid), (bar, t_mid),
                  (RX + RW + 0.15, t_mid))                                 # directions -> estimates
            arrow((RX + RW / 2, y_m + 4.15),   # above the title
                  (RX + RW / 2, y_t + (tl_h - tr_h) / 2 + 0.2))   # range -> estimates

            def frame(k, end=False):
                m = x[k - 1]
                shown = np.full_like(full, np.nan)
                shown[:, :k] = full[:, :k]
                heat.set_data(shown)
                cursor.set_xdata([m, m])
                cursor.set_visible(not end)
                line.set_xdata(Y[:k].sum(axis=0))
                for j, mark, label in peaks:
                    v = Y[:k, j].sum()
                    mark.set_xdata([v])
                    label.xy = (v, label.xy[1])
                for v, im in strips:
                    im.set_data(np.where(xm <= m, v, np.nan)[None])
                set_top(m)
                for a in final:
                    a.set_visible(end)

            frame(len(x), end=True)
            scale_fonts(fig, font)
            gif = GifFrames(fig)   # palette from the finished picture
            for f in range(DATAFLOW_FRAMES):
                frame(max(1, round((f + 1) / DATAFLOW_FRAMES * len(x))))
                gif.add(fig, DATAFLOW_FRAME_MS)
                self.app.status = f"Dataflow GIF: frame {f + 1}/{DATAFLOW_FRAMES}"
            frame(len(x), end=True)
            gif.add(fig, DATAFLOW_HOLD_MS)
            gif.save(out)
            self.app.status = f"Saved {out.relative_to(SAVE_DIR.parent).as_posix()}"
        self.app.run_job(job)

    # ---- Mask quality (calibration only) ----
    def calibration(self):
        """(X, angles): the calibration signatures, positions x 81 pixels, over the carriage
        positions where every calibrated pixel has data (NaN columns = not calibrated), and
        (bearing, elevation) in degrees of each grid point, 9 x 9."""
        B = self.B_cal
        have = np.isfinite(B).any(axis=0)
        X = B[np.isfinite(B[:, have]).all(axis=1)]
        rows, cols = np.meshgrid(range(9), range(9), indexing="ij")
        return X, tuple(np.degrees(a) for a in demux.pixel_angles(rows, cols))

    def angle_map(self, ax, values, label, cmap):
        """values (9 x 9) over bearing / elevation, smooth when complete."""
        _, (be, el) = self.calibration()
        mesh = ax.pcolormesh(be, el, values, cmap=cmap,
                             shading="gouraud" if np.isfinite(values).all() else "nearest")
        ax.plot(be, el, ".", color="k", ms=2, alpha=0.4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Bearing (\N{DEGREE SIGN})")
        ax.set_ylabel("Elevation (\N{DEGREE SIGN})")
        ax.figure.colorbar(mesh, ax=ax, shrink=0.85, label=label)
        return mesh

    def plot_sensitivity(self, ax):
        """Total response of each calibration point: its signature summed over every mask
        position, relative to the strongest."""
        X, _ = self.calibration()
        total = X.sum(axis=0).reshape(9, 9)
        self.angle_map(ax, total / np.nanmax(total), "Relative response", self.app.sequential)
        ax.set_title(f"Sensitivity, {self.cal_lo.get():.0f}-{self.cal_hi.get():.0f} mm",
                     fontsize=10)

    def plot_confusability(self, ax_u, ax_c, ax_w=None):
        """How easily calibration points are mistaken for one another. Each signature has its
        mean over positions removed (only what the mask changes is compared) and is scaled
        to unit length, so the dot products are correlations. Neighbours share most of their
        beam, so being alike is expected and costs little; a mix-up with a far point is a
        real error. Left: confusability = correlation with the most similar point more than
        CONFUSE_PX grid pixels away (1 = a far twin; lower = a better mask). Right: every
        pair, titled with the largest far-pair correlation (the far-pair coherence).
        ax_w: the correlation matrix with only each point's worst far pair, rest white."""
        X, _ = self.calibration()
        Z = X - X.mean(axis=0)
        Z = Z / np.linalg.norm(Z, axis=0)
        C = Z.T @ Z                          # 81 x 81, NaN for uncalibrated points
        rows, cols = np.divmod(np.arange(81), 9)
        apart = np.hypot(rows[:, None] - rows, cols[:, None] - cols)
        far = np.where(apart > CONFUSE_PX, C, np.nan)
        with np.errstate(all="ignore"):
            confuse = np.nanmax(far, axis=1).reshape(9, 9)
        self.angle_map(ax_u, confuse, f"Max correlation > {CONFUSE_PX} px away", SCORE_CMAP)
        ax_u.set_title(f"Confusability" + chr(10) + f"mean {np.nanmean(confuse):.3f}, "
                       f"worst {np.nanmax(confuse):.3f}", fontsize=10)
        img = ax_c.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ticks = range(0, 81, 9)
        ax_c.set_xticks(ticks, [f"{k // 9}0" for k in ticks], fontsize=7)
        ax_c.set_yticks(ticks, [f"{k // 9}0" for k in ticks], fontsize=7)
        ax_c.set_xlabel("Pixel")
        ax_c.set_ylabel("Pixel")
        ax_c.figure.colorbar(img, ax=ax_c, shrink=0.85, label="Correlation")
        ax_c.set_title("Signature correlation" + chr(10)
                       + f"far-pair coherence {np.nanmax(np.abs(far)):.3f}", fontsize=10)
        if ax_w is None:
            return
        ok = np.isfinite(far).any(axis=1)
        i = np.flatnonzero(ok)
        j = np.nanargmax(far[ok], axis=1)
        W = np.full_like(C, np.nan)
        W[i, j] = far[i, j]
        img = ax_w.imshow(W, cmap=matplotlib.colormaps["RdBu_r"].with_extremes(bad="white"),
                          vmin=-1, vmax=1, interpolation="nearest")
        ax_w.set_xticks(ticks, [f"{k // 9}0" for k in ticks], fontsize=7)
        ax_w.set_yticks(ticks, [f"{k // 9}0" for k in ticks], fontsize=7)
        ax_w.set_xlabel("Pixel")
        ax_w.set_ylabel("Pixel")
        ax_w.figure.colorbar(img, ax=ax_w, shrink=0.85, label="Correlation")
        ax_w.set_title("Worst far pairs" + chr(10) + f"> {CONFUSE_PX} px apart", fontsize=10)

    def plot_mask(self, ax):
        """The mask in 3D, as seen from the radar: closed pixels (0s, foil kept) stand
        out of the plate, open pixels (1s, foil cut away) are flush with it. 7 mm pixels."""
        rows_m, cols_m = MASK.shape
        px = rc.STEP_MM
        for r in range(rows_m):
            for c in range(cols_m):
                closed = not MASK[r, c]
                ax.bar3d(c * px, 0, (rows_m - 1 - r) * px, px, 2.0 if closed else 0.6, px,
                         color=MASK_OFF if closed else MASK_ON, edgecolor="#555",
                         linewidth=0.3, shade=True)
        ax.set_box_aspect((cols_m * px, 8, rows_m * px), zoom=0.85)
        ax.set_xlim(0, cols_m * px)
        ax.set_ylim(-3, 5)
        ax.set_zlim(0, rows_m * px)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=-75)

    def export_figures(self):
        slot = self.slot.get()
        out = self.folder(slot) / time.strftime("export_%Y%m%d_%H%M%S")
        out.mkdir(parents=True)
        photo = self.photo()
        font = self.app.export_font.get()

        def placeholder(ax, text):
            ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        def save(name, size, draw, post=None, **kw):
            fig = Figure(figsize=size, dpi=150, layout="constrained")
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(**kw)
            draw(ax)
            scale_fonts(fig, font)
            if post:
                post(ax)
            # a 3D axes' labels can reach past the layout: crop to everything drawn
            fig.savefig(out / f"{name}.png", bbox_inches="tight" if post else None,
                        pad_inches=0.15)

        def sparse_3d(ax):
            """Lines as thick as the text is large, every other tick, and the axis labels
            clear of the tick numbers."""
            for line in ax.get_lines():
                line.set_linewidth(line.get_linewidth() * font)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                lo_, hi_ = sorted(axis.get_view_interval())
                ticks = axis.get_ticklocs()
                axis.set_ticks(ticks[(ticks >= lo_) & (ticks <= hi_)][::2])
                axis.labelpad = 5 * font

        save("profile", (9, 4), self.plot_profile)
        save("slice", (9, 4.5), self.plot_slice)
        for i in range(len(self.results)):
            save(f"score_{i + 1}", (5.5, 4.6), lambda ax: self.plot_score(ax, i))
            if photo:
                save(f"score_overlay_{i + 1}", (5.5, 4.6), lambda ax: self.plot_score(ax, i, photo))
            save(f"signature_{i + 1}", (6, 4), lambda ax: self.plot_signature(ax, i))
        if photo:
            save("camera", (6, 6 * photo[0].shape[0] / photo[0].shape[1]),
                 lambda ax: self.plot_camera(ax, photo))
        save("scene3d", (8, 7), lambda ax: self.plot_scene(ax, photo), sparse_3d,
             projection="3d", computed_zorder=False)   # the photo board behind the points
        save("ranges3d", (8, 7), self.plot_ranges, projection="3d", computed_zorder=False)
        save("sensitivity", (7, 5.8), self.plot_sensitivity)
        fig = Figure(figsize=(10, 10), dpi=200)   # 3D axes are square: crop to the mask
        FigureCanvasAgg(fig)
        self.plot_mask(fig.add_axes((0, 0, 1, 1), projection="3d"))
        img = GifFrames.grab(fig)
        box = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
        pad = 20
        img.crop((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)).save(out / "mask3d.png")
        fig = Figure(figsize=(21, 6), dpi=150, layout="constrained")
        FigureCanvasAgg(fig)
        self.plot_confusability(fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2),
                                fig.add_subplot(1, 3, 3))
        scale_fonts(fig, font)
        fig.savefig(out / "confusability.png")

        # Everything together; the detail plots are for the selected detection
        fig = Figure(figsize=(20, 19), dpi=120, layout="constrained")
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1.3, 1.3], width_ratios=[1, 1, 1, 1.2])
        self.plot_profile(fig.add_subplot(gs[0, :3]))
        self.plot_slice(fig.add_subplot(gs[1, :3]))
        ax = fig.add_subplot(gs[:2, 3])
        self.plot_camera(ax, photo) if photo else placeholder(ax, "No camera snapshot")
        axes = [fig.add_subplot(gs[2, k]) for k in range(3)]
        if self.results:
            self.plot_score(axes[0], self.sel)
            self.plot_score(axes[1], self.sel, photo) if photo else placeholder(axes[1], "No camera snapshot")
            self.plot_signature(axes[2], self.sel)
        else:
            for ax in axes:
                placeholder(ax, "No detection")
        self.plot_scene(fig.add_subplot(gs[2, 3], projection="3d", computed_zorder=False), photo)
        # bottom row: estimates vs truth, and the mask quality (calibration only)
        self.plot_ranges(fig.add_subplot(gs[3, 0], projection="3d", computed_zorder=False))
        self.plot_sensitivity(fig.add_subplot(gs[3, 1]))
        self.plot_confusability(fig.add_subplot(gs[3, 2]), fig.add_subplot(gs[3, 3]))
        title = f"Page {self.page.get()} " + ("Ambient" if slot == AMBIENT else slot)
        note = self.note.get()
        fig.suptitle(f"{title}{': ' + note if note else ''}\nstrategy {self.strategy.get()}, "
                     f"true pixels {self.truth.get() or 'not set'}", fontsize=13)
        scale_fonts(fig, font)
        fig.savefig(out / "summary.png")
        self.message = f"Exported {out.relative_to(SAVE_DIR.parent).as_posix()}"

    def save_results(self):
        slot = self.slot.get()
        out = self.folder(slot) / time.strftime("results_%Y%m%d_%H%M%S.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "scenario": slot, "page": self.page.get(), "note": self.note.get(), "truth": self.truth.get(),
            "strategy": self.strategy.get(),
            "search_mm": [self.r_lo.get(), self.r_hi.get()],
            "cfar": {"guard": self.guard.get(), "train": self.train.get(),
                     "threshold_db": self.thr_db.get(), "min_peak_percent": self.min_peak.get()},
            "scenario_ambient": self.scn_ambient.get(),
            "calibration": {"range_mm": [self.cal_lo.get(), self.cal_hi.get()],
                            "ambient": self.cal_ambient.get()},
            "localisation_window_bins": self.loc_bins.get(),
            "subpixel_factor": self.factor,
            "position_bin_mm": BIN_MM,
            "positions_used": int(self.rows.sum()),
            "detections": [{"range_mm": float(r["range_mm"]), "peak": float(r["peak"]),
                            "snr_db": float(r["snr_db"]),
                            "pixel": rc.pixel_label(r["pixel"]),
                            "row": float(r["pixel"][0]), "col": float(r["pixel"][1]),
                            "bearing_deg": float(np.degrees(r["bearing"])),
                            "elevation_deg": float(np.degrees(r["elevation"])),
                            "xyz_mm": {"across": float(r["xyz"][0]),
                                       "forward": float(r["xyz"][1]),
                                       "up": float(r["xyz"][2])},
                            "score": float(r["score"][r["best"]]),
                            "margin": r["margin"],
                            "error_px": None if np.isnan(r["error"]) else float(r["error"])}
                           for r in self.results],
        }, indent=1))
        self.message = f"Saved {out.relative_to(SAVE_DIR.parent)}"

    def poll(self):
        if not self.app.running or not self.win.winfo_exists():
            return
        try:
            state = "disabled" if self.app.busy else "normal"
            for b in (self.btn_scan, *self.slot_buttons.values()):
                b.config(state=state)
            stamps = {}
            for s, b in self.slot_buttons.items():
                f = self.folder(s) / SWEEP_FILE
                stamps[s] = f.stat().st_mtime_ns if f.exists() else None
                colour = scene_colour(self.folder(s))
                b.config(bg=colour, selectcolor=colour,
                         font=("Helvetica", 9, "bold" if s == self.slot.get() else "normal"))
            if stamps != self.counts:   # a new or re-captured sweep refreshes the results
                self.counts = stamps
                self.dirty = True
            if self.dirty:
                self.dirty = False
                self.compute()
                self.draw()
        except Exception as e:
            traceback.print_exc()
            self.message = f"Error: {e}"
        self.lbl_status.config(text=self.app.status)
        self.lbl_msg.config(text=self.message or self.app.message)
        self.win.after(rc.UI_MS, self.poll)


class CameraWindow:
    """Camera view: stream controls and grid corners."""

    def __init__(self, app):
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("Camera")
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        top = tk.Frame(self.win)
        top.pack(fill="x", padx=6, pady=4)
        tk.Label(top, text="Stream URL").pack(side="left")
        tk.Entry(top, textvariable=app.cam_url, width=36).pack(side="left", padx=4)
        tk.Button(top, text="Connect", command=lambda: app.camera.start(app.cam_url.get())).pack(side="left")
        tk.Button(top, text="Disconnect", command=app.camera.stop).pack(side="left", padx=4)
        tk.Button(top, text="Rotate 90°", command=app.rotate_camera).pack(side="left")
        self.lbl_fps = tk.Label(top, text="", fg="#666", width=14, anchor="w")
        self.lbl_fps.pack(side="left", padx=6)

        tk.Button(top, text="Set grid corners", command=self.pick_corners).pack(side="left", padx=4)

        self.canvas = tk.Canvas(self.win, width=338, height=CAMERA_VIEW_H, bg="#222",
                                highlightthickness=0)
        self.canvas.pack(padx=6, pady=4)
        self.canvas.bind("<Button-1>", self.on_click)
        self.lbl_info = tk.Label(self.win, text="", font=("Helvetica", 11), width=70, height=3,
                                 wraplength=620)
        self.lbl_info.pack(pady=(0, 6))
        self.picking = None        # corner points clicked so far, while picking
        self.photo = None
        self.size = (338, CAMERA_VIEW_H)
        self.poll()

    def pick_corners(self):
        self.picking = []

    def on_click(self, event):
        if self.picking is None:
            return
        w, h = self.size
        self.picking.append((event.x / w, event.y / h))
        if len(self.picking) == len(CORNERS):
            self.app.set_corners(self.picking)
            self.picking = None

    def poll(self):
        if not self.app.running or not self.win.winfo_exists():
            return
        try:
            self.draw()
        except Exception as e:
            traceback.print_exc()
            self.app.message = f"Camera window: {e}"
        self.win.after(150, self.poll)

    def draw(self):
        app, cam = self.app, self.app.camera
        self.lbl_fps.config(text=cam.error if cam.error == "Connecting..." else "Camera error"
                            if cam.error else f"{cam.fps:.1f} fps" if cam.image else "Not connected")
        c = self.canvas
        c.delete("all")
        if cam.image is not None:
            iw, ih = cam.image.size
            w, h = round(CAMERA_VIEW_H * iw / ih), CAMERA_VIEW_H
            if (w, h) != self.size:
                self.size = (w, h)
                c.config(width=w)
            self.photo = ImageTk.PhotoImage(cam.image.resize((w, h), Image.BILINEAR))
            c.create_image(0, 0, image=self.photo, anchor="nw")
        w, h = self.size
        if app.cam_corners:
            grid = [(col, row) for row in range(9) for col in range(9)]
            for (x, y), (col, row) in zip(grid_to_image(app.cam_corners, grid), grid):
                big = f"{row}{col}" in CORNERS
                r = 3 if big else 1.5
                c.create_oval(x * w - r, y * h - r, x * w + r, y * h + r,
                              fill="#7fd4ff" if big else "#ccc", outline="")
        if self.picking is not None:
            for x, y in self.picking:
                c.create_oval(x * w - 4, y * h - 4, x * w + 4, y * h + 4, outline="#ff0", width=2)
            c.create_text(10, 10, anchor="nw", fill="#ff0", font=("Helvetica", 13, "bold"),
                          text=f"Click grid point {CORNERS[len(self.picking)]}")

        if cam.error and cam.error != "Connecting...":
            info = cam.error
        elif not app.cam_corners:
            info = "Set the grid corners (click 00 then 88) so targets can be marked in the demux window"
        else:
            info = "Targets are marked on the snapshot in the demux window"
        self.lbl_info.config(text=info)

    def close(self):
        self.app.cam_window = None
        self.win.destroy()


class NoArduino:
    """Stands in for the Arduino when none is connected: writes go nowhere, nothing is
    read. Closing it ends the reader thread, as a real port does."""

    def __init__(self):
        self.closed = False

    def write(self, data):
        pass

    def readline(self):
        if self.closed:
            raise serial.SerialException("closed")
        time.sleep(0.1)
        return b""

    def close(self):
        self.closed = True


class NoRadar:
    """Stands in for the radar when none is connected: radar_loop stops at once."""

    def setup_session(self, config):
        pass

    def start_session(self):
        pass

    def stop_session(self):
        pass

    def close(self):
        pass

    def get_next(self):
        raise RuntimeError("not connected (Reconnect radar to retry)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arduino", help="Arduino serial port, e.g. COM12")
    ap.add_argument("--radar", help="Radar serial port, e.g. COM11 (default: auto-detect)")
    ap.add_argument("--mock", action="store_true", help="use simulated radar data")
    args = ap.parse_args()

    def open_radar():
        if args.mock:
            client = a121.Client.open(mock=True)
        else:
            client = a121.Client.open(serial_port=args.radar or rc.find_port("Radar", rc.RADAR_KEYWORDS),
                                      override_baudrate=rc.RADAR_BAUD)
        client.setup_session(CAL_SENSOR_CONFIG)
        return client

    def open_arduino():
        port = args.arduino or rc.find_port("Arduino", rc.ARDUINO_KEYWORDS)
        try:
            return serial.Serial(port, rc.BAUD, timeout=0.1)
        except serial.SerialException as e:
            raise RuntimeError(f"Couldn't open {port}: {e}\n"
                               "Is the Serial Monitor/Plotter still open?")

    # Whatever isn't connected is stood in for, so saved data can still be viewed,
    # demultiplexed and exported; the Reconnect buttons retry
    missing = []
    try:
        client = open_radar()
    except Exception as e:
        print(f"Radar: {e}")
        client = NoRadar()
        missing.append("radar")
    try:
        ser = open_arduino()
    except Exception as e:
        print(f"Arduino: {e}")
        ser = NoArduino()
        missing.append("Arduino")

    root = tk.Tk()
    app = ContinuousApp(root, ser, client, open_arduino, open_radar)
    if missing:
        text = (f"Not connected: {' and '.join(missing)} - viewing saved data only "
                "(use the Reconnect buttons)")
        root.after(500, lambda: setattr(app, "message", text))
    root.mainloop()


if __name__ == "__main__":
    main()
