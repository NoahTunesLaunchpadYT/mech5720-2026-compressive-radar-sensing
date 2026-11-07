"""
Continuous coded-mask capture: the carriage sweeps the whole mask at constant speed
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

Setup: as radar_capture.py; needs the 'G'/P-line firmware and a sync model.
  python radar_capture_continuous.py [--arduino COM12] [--radar COM11] [--mock]
"""

import argparse
import json
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import serial
import tkinter as tk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import demux
import radar_capture as rc
from radar_capture import (
    a121, AMBIENT, AMBIENT_END, AMBIENTS, ARDUINO_LINE, DEMUX_SLOTS, DISTANCES_M,
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
AMBIENT_COPY = "ambient.npz"   # the ambient in effect for a sweep (see the docstring)
BIN_MM = 1.0
GRID_MM = np.arange(rc.START_OFFSET_MM, rc.mm_from_position(NUM_POSITIONS) + BIN_MM / 2, BIN_MM)
DEFAULT_SPEED_MM = 10.0
ALIGN_MAX_MM = 5.0       # forward/back check: shifts searched, +-
ALIGN_STEP_MM = 0.05
ALIGN_WARN_MM = 0.5      # warn when the legs disagree by more than this
ALIGN_MIN_RHO = 0.8      # or correlate worse than this


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
            hit = _slices[f] = (mtime, d["slice"])
    return hit[1]


def sweep_ambient(folder, ambient):
    """The ambient slice to subtract from folder's sweep: for AMBIENT, the sweep's own copy
    if it has one; otherwise that ambient's current sweep (None if not captured)."""
    own = load_slice(folder, AMBIENT_COPY) if ambient == AMBIENT else None
    return own if own is not None else load_slice(SAVE_DIR / ambient)


def read_ambient(file):
    with np.load(file) as d:
        return d["slice"], float(d["time"])


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


def scene_colour(folder):
    return SCENE_FULL if (folder / SWEEP_FILE).exists() else SCENE_NONE


def demux_folder(slot):
    return SAVE_DIR / "demux" / slot


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
        self.radar_restart = False         # radar_loop exits (reconnect)
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

        root.title("Radar Capture (continuous)")
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
        self.btn_celebrate = tk.Button(actions, text="Celebrate!", width=20, fg="#a0a",
                                       command=lambda: self.send("T"))
        btn_demux = tk.Button(actions, text="Demultiplexing...", width=20, command=self.open_demux)
        btn_stop = tk.Button(actions, text="STOP", width=20, fg="#b00", command=self.stop)
        for i, b in enumerate((self.btn_calibrate, self.btn_radar, self.btn_arduino, btn_refresh,
                               self.btn_export, self.btn_celebrate, btn_demux, btn_stop)):
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
        for name in ("beam_by", "beam_mask", "beam_mm", "beam_lo", "beam_hi", "normalise"):
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
        if have.any():
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
        self.ax_beam.set_zlabel("x scene mean" if normalise else "Intensity")
        self.beam_canvas.draw()

    def draw_profile(self):
        p = self.profile
        mm = self.latest.get("Distance_mm")
        if self.subtract.get() and self.ambient is not None and mm is not None:
            p = p - self.ambient[grid_index(mm)]   # nearest bin, e.g. parked in the run-up
        p = np.abs(p)
        self.line.set_ydata(p)
        self.img.set_data(p[None, :])
        peak = max(p.max(), 1.0)
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
                  self.btn_export, self.btn_celebrate, self.btn_prev, self.btn_next,
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
        try:
            while self.running and not self.radar_restart:
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
                    self.profile = result.frame.mean(axis=0)
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
        self.fresh_session()
        errors_before = FrameErrors.radar
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
        pos_pc, us, um = np.array(positions).T
        raw = dict(frame_pc=pc, frame_tick=tick, pos_pc=pos_pc, pos_us=us.astype(np.int64),
                   pos_um=um.astype(np.int64))
        mm, towards = frame_positions(raw, model)
        slice_, counts = bin_frames(mm, profiles)
        shift, rho, shifts, curve = leg_alignment(mm, towards, profiles)
        folder.mkdir(parents=True, exist_ok=True)
        old_ambient = (read_ambient(folder / SWEEP_FILE)
                       if folder == SAVE_DIR / AMBIENT and (folder / SWEEP_FILE).exists() else None)
        save_npz(folder / SWEEP_FILE, slice=slice_, counts=counts, grid_mm=GRID_MM,
                 distances_m=DISTANCES_M, profiles=profiles, frame_mm=mm,
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
                                        + [demux_folder(s) for s in DEMUX_SLOTS if s != AMBIENT],
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
        if abs(shift) > ALIGN_WARN_MM or rho < ALIGN_MIN_RHO:
            self.message = (f"Check alignment of {name}: {alignment_text(shift, rho)}. "
                            "Re-run the sync test if this persists.")

    def export(self):
        lo, hi, subtract = self.beam_lo.get(), self.beam_hi.get(), self.subtract.get()
        normalise = self.normalise.get()
        ambient = self.ambient_source.get() if subtract else None

        def job():
            self.status = "Exporting beam heatmaps..."
            cube, sub = beam_cube(lo, hi, ambient)
            if normalise:
                cube = normalise_across_positions(cube)
            results = [(cube[:, :, grid_index(mask_mm(m))], sub) for m in range(NUM_POSITIONS)]
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
        super().__init__(app)
        self.win.title("Demultiplexing (continuous)")
        self.btn_scan.config(text="Capture sweep")
        self.btn_capture.grid_remove()
        dist_mm = DISTANCES_M * 1000
        half = (dist_mm[1] - dist_mm[0]) / 2
        self.mask_img.set_extent((dist_mm[0] - half, dist_mm[-1] + half,
                                  GRID_MM[-1] + BIN_MM / 2, GRID_MM[0] - BIN_MM / 2))
        self.ax_img.set_ylabel("Carriage (mm)")
        self.dirty = True

    def select_slot(self):
        f = demux_folder(self.slot.get()) / "info.json"
        info = json.loads(f.read_text()) if f.exists() else {}
        self.note.set(info.get("note", ""))
        self.truth.set(info.get("truth", ""))
        self.sel = 0
        self.dirty = True

    def save_info(self):
        folder = demux_folder(self.slot.get())
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "info.json").write_text(json.dumps(
            {"note": self.note.get(), "truth": self.truth.get()}, indent=1))
        self.dirty = True

    def scan(self):
        slot = self.slot.get()
        self.app.run_job(lambda: self.app.sweep_capture(demux_folder(slot), f"Demux {slot}"))

    def compute(self):
        self.message = ""
        slot, strategy = self.slot.get(), self.strategy.get()
        lo, hi = self.r_lo.get(), self.r_hi.get()
        scale = 10 ** (self.thr_db.get() / 20)
        scn = load_slice(demux_folder(slot))
        used = np.full(len(GRID_MM), scn is not None)
        if scn is None:
            scn = np.full((len(GRID_MM), len(DISTANCES_M)), np.nan)
        choice = self.scn_ambient.get()
        amb = None
        if choice != "none":
            amb = (load_slice(demux_folder(AMBIENT)) if choice == "demux"
                   else sweep_ambient(demux_folder(slot), choice))
            if amb is None and used.any() and slot != AMBIENT:
                self.message = "Scenario ambient not captured: not subtracted"
        self.y = np.abs(scn - (0 if amb is None else amb))   # positions x range bins
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
        truths = self.truth_pixels()
        for k in peaks:
            y = self.y[self.rows, max(0, k - w):k + w + 1].sum(axis=1)
            r = demux.localise(self.B, y, strategy, positions)
            best = tuple(v / self.factor for v in divmod(r["best"], n))
            bearing, elevation = demux.pixel_angles(*best)
            r.update(bin=k, range_mm=dist_mm[k], peak=self.profile[k], y=y, pixel=best,
                     bearing=bearing, elevation=elevation,
                     xyz=demux.to_xyz(dist_mm[k], bearing, elevation),
                     snr_db=20 * np.log10(self.profile[k] * scale / self.thr[k]),
                     error=min((np.hypot(best[0] - t[0], best[1] - t[1]) for t in truths),
                               default=np.nan))
            self.results.append(r)
        self.sel = min(self.sel, max(len(self.results) - 1, 0))

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
        ax.plot(dist_mm, self.profile, label="sum over positions")
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
        ax.set_ylabel("Sum of |capture - ambient|")
        ax.set_title(f"{'Ambient' if slot == AMBIENT else slot}: "
                     f"{'sweep captured' if self.used.any() else 'no sweep yet'}, "
                     f"{len(self.results)} detection(s)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelbottom=False)

        self.mask_img.set_data(self.y)
        if np.isfinite(self.y).any():
            self.mask_img.set_clim(np.nanmin(self.y), np.nanmax(self.y))
        for r in self.results:
            self.overlays.append(self.ax_img.axvline(r["range_mm"], color="white",
                                                     linestyle="--", linewidth=0.8))

        cmap = self.app.sequential if strategy == "A" else self.app.sequential.reversed()
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
                                    + ("cosine score" if strategy == "A" else "cost (low = good)"),
                                    fontsize=9)
            mm = GRID_MM[self.rows]
            b = self.B[:, r["best"]]
            self.ax_sig.plot(mm, r["y"] / r["y"].mean(), lw=1, label="measured")
            self.ax_sig.plot(mm, b / b.mean(), lw=1, label=f"pixel {rc.pixel_label(r['pixel'])}")
            self.ax_sig.set_title("Signature / mean over positions", fontsize=9)
            self.ax_sig.set_xlabel("Carriage (mm)")
            self.ax_sig.legend(fontsize=7)
        else:
            self.score_img.set_data(np.full((9, 9), np.nan))
            self.ax_score.set_title("No detection", fontsize=9)
        for t in truths:
            self.overlays.append(self.ax_score.plot(t[1], t[0], "x", color="red",
                                                    markersize=10, mew=2)[0])

        self.draw_scene(truths)
        self.canvas.draw()

    def save_results(self):
        slot = self.slot.get()
        out = demux_folder(slot) / time.strftime("results_%Y%m%d_%H%M%S.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "scenario": slot, "note": self.note.get(), "truth": self.truth.get(),
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
                f = demux_folder(s) / SWEEP_FILE
                stamps[s] = f.stat().st_mtime_ns if f.exists() else None
                colour = scene_colour(demux_folder(s))
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
        client.setup_session(rc.SENSOR_CONFIG)
        return client

    def open_arduino():
        port = args.arduino or rc.find_port("Arduino", rc.ARDUINO_KEYWORDS)
        try:
            return serial.Serial(port, rc.BAUD, timeout=0.1)
        except serial.SerialException as e:
            raise RuntimeError(f"Couldn't open {port}: {e}\n"
                               "Is the Serial Monitor/Plotter still open?")

    try:
        client = open_radar()
    except RuntimeError as e:
        sys.exit(str(e))
    try:
        ser = open_arduino()
    except RuntimeError as e:
        client.close()
        sys.exit(str(e))

    root = tk.Tk()
    ContinuousApp(root, ser, client, open_arduino, open_radar)
    root.mainloop()


if __name__ == "__main__":
    main()
