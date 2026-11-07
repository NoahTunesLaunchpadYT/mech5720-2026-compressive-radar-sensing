"""
Radar capture + mask stepping for the coded-mask rig.

The radar streams continuously and the plots show the live amplitude vs distance
(200 mm - 2 m). Captures take the next FRAMES_PER_CAPTURE frames from that stream.

Masks are numbered 0..23 (mask m = columns m..m+3 of mask_pattern.csv).

  C / Enter        -> capture FRAMES_PER_CAPTURE radar frames at the current mask
  Right arrow      -> move the carriage one position (7 mm) away from the switch
  Left arrow       -> move the carriage one position towards the switch
  Space            -> stop the stepper (and abort a scan or calibration)
  "Scan all 24"    -> go to the nearer end (position 1 or 24), then capture + step
                      through every position towards the other end
                      (with "Ambient" selected this is the ambient capture routine)
  "Advance after capture" moves to the next position once a single capture is saved.
  Mask 0..23 buttons -> go straight to that mask
  "Calibrate"      -> re-run the homing pass, then go to mask 23
  "Subtract ambient" -> subtracts each mask's complex ambient mean from the live plots,
                      the scene map and the beam pattern, then shows the magnitude.
                      Display only: saved captures stay raw. "start" / "end" picks which
                      ambient scan to use; each is enabled once its full scan exists
                      (captures/ambient/ or captures/ambient_end/ scene_map.npz).
  Scene 00..88 buttons -> choose the scene folder that captures and scans save to,
                      and show that scene's map. Colour: grey = nothing captured,
                      orange = some masks, green = all 24.
  "Ambient" button -> selected like a scene: captures and scans save to
                      captures/ambient/ (the start-of-calibration reference).
  "Amb end" button -> the same for captures/ambient_end/: a second ambient scan taken
                      at the end of calibration. Viewing one ambient while subtracting
                      the other shows how much the background drifted.
  Beam pattern     -> pick a mask and a range (mm): for each scene 00..88 that has a
                      capture of that mask, sums |capture| over the range bins (ambient
                      subtracted first if enabled) and plots it as a 3D surface over the
                      9x9 scene grid (drag to orbit). Red dot = selected scene.
  "Export beam heatmaps" -> writes every mask's beam pattern (current range and
                      subtraction setting) to captures/exports/beam_<time>/: grid.png
                      (all 24 on one shared scale), mask00..23.png (each on its own
                      scale) and beam_patterns.npz (24 x 9 x 9 array).
  "Normalise each scene across masks" -> divides each scene's beam value by its mean
                      over all 24 masks (same range), cancelling per-scene reflector
                      gain. Scenes missing any mask are left blank. Also applies to export.

Captures are saved to captures/<scene>/mask<NN>.npz (overwriting that mask's
previous capture); a completed scan also writes captures/<scene>/scene_map.npz,
the 24 x distance complex mean map with row m = mask m. Each capture file holds:
  frames       complex array (frames, sweeps_per_frame, num_points)
  distances_m  range of each point
  distance_mm  carriage distance from the limit switch when the capture started
  position     carriage position 1..24 (1 = nearest the limit switch)
  mask_index   mask number 0..23
  frame_errors corrupted radar messages discarded during this capture
  mask         the 4x4 mask in front of the sensor
  sensor_config, server_info (JSON strings)

Setup: pip install -r requirements.txt, then
  python radar_capture.py                          (auto-detects both devices)
  python radar_capture.py --arduino COM12 --radar COM11
  python radar_capture.py --mock                   (fake radar data, for testing)

Notes:
  - Close the Arduino Serial Monitor and the Exploration Tool app first.
  - Opening the Arduino port resets it, so it re-homes when this app starts.
  - Uses the arduino/arduino.ino protocol: 'F'/'B' = one 7 mm move, 'a'..'x' = go to
    position 1..24, 'C' = re-home, 'S' = stop, 'H' = heartbeat, 'T' = play the celebration tune.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.interpolate import RectBivariateSpline, griddata
from acconeer.exptool import a121

import demux
from acconeer.exptool._core.communication.message_stream import MessageStream


class FrameErrors:
    radar = 0
    arduino = 0


def robust_stream(self):
    """Replacement for MessageStream._get_stream that discards corrupted messages
    instead of disconnecting.

    Each message is a JSON header line, optionally followed by payload_size binary
    bytes. Payloads are read one byte long: in an intact stream that extra byte is
    the '{' opening the next header, so anything else means bytes were dropped.
    After an error, each line is searched for a '{"' that starts a valid header,
    since the next header usually arrives glued to leftover payload bytes.
    """
    carry = b""
    in_error = False
    while True:
        try:
            line = carry + self._link.recv_until(self.protocol.end_sequence)
        except Exception as e:
            self._error_callback(e)
        carry = b""
        try:
            start = 0
            while True:
                try:
                    header = json.loads(line[start:])
                    break
                except ValueError:
                    start = line.find(b'{"', start + 1)
                    if start < 0:
                        raise
            if start > 0 and not in_error:
                FrameErrors.radar += 1
            size = header.get("payload_size", 0)
            payload = b""
            if size:
                try:
                    data = self._link.recv(size + 1)
                except Exception as e:
                    self._error_callback(e)
                payload, carry = data[:-1], data[-1:]
                if carry != b"{":
                    carry = b""
                    raise ValueError("payload length mismatch")
            msg = self.protocol.parse_message(header, payload)
        except Exception:
            if not in_error:
                FrameErrors.radar += 1
            in_error = True
            continue
        in_error = False
        yield msg


MessageStream._get_stream = robust_stream

ARDUINO_LINE = re.compile(r"Distance_mm:-?\d+\.\d\d")

BAUD = 115200
RADAR_BAUD = 1_000_000   # XM125 over CH340: 2 Mbaud (the default) drops bytes
ARDUINO_KEYWORDS = ("arduino", "usbmodem", "ttyacm")
RADAR_KEYWORDS = ("ch340", "wch")
UI_MS = 300              # status/label refresh interval
HEARTBEAT_S = 0.1        # sent from its own thread; the Arduino stops after 300 ms of silence
PLOT_MS = 25             # poll for new radar frames faster than the 20 fps frame rate
MAX_LAG_S = 0.3          # restart the radar session when its timestamps lag the PC by this
THROB_MIN_S = 0.4       # shortest time the beam-pattern throbber is shown
FRAMES_PER_CAPTURE = 20
SAVE_DIR = Path(__file__).parent / "captures"
SETTINGS_FILE = Path(__file__).parent / "settings.json"   # inputs remembered between runs
AMBIENT = "ambient"                      # scene name of the start-of-calibration ambient
AMBIENT_END = "ambient_end"              # second ambient scan, taken at the end of calibration
AMBIENTS = (AMBIENT, AMBIENT_END)        # their mask files are the subtraction references
SCENES = [f"{r}{c}" for r in range(9) for c in range(9)]   # 00 .. 88
SCENE_NONE, SCENE_PARTIAL, SCENE_FULL = "#d9d9d9", "#ffc56b", "#8fd18f"
DEMUX_SLOTS = [f"D{i}" for i in range(1, 9)] + [AMBIENT]   # captures/demux/<slot>/

# Mask positions (must match arduino.ino)
START_OFFSET_MM = 3.9
STEP_MM = 7.0
NUM_POSITIONS = 24
ARRIVE_TOLERANCE_MM = 0.05
SETTLE_S = 0.3           # position must hold this long before a scan captures
MOVE_TIMEOUT_S = 60
HOMING_TIMEOUT_S = 90    # full 200 mm homing pass takes ~40 s
CALIBRATE_TO_MASK = 23   # where Calibrate leaves the carriage

# Mask pattern: 4 rows x 27 columns; mask m (0..23) is columns m .. m+3.
# MASK_REVERSED: carriage position 1 (nearest the switch) shows the last mask.
MASK = np.loadtxt(Path(__file__).parent / "mask_patterns" / "mask_impulse.csv",
                  delimiter=",", dtype=int)
MASK_REVERSED = True
MASK_ON = "#7fd4ff"      # 1s: light blue
MASK_OFF = "#000000"     # 0s: black

# Radar settings - tune for the experiment. 200 mm .. 2000 mm in 10 mm steps.
# Keep points x sweeps x frame_rate x 4 bytes well under ~80 kB/s at 1 Mbaud.
SENSOR_CONFIG = a121.SensorConfig(
    start_point=80,          # x 2.5 mm = 200 mm
    num_points=181,          # last point = 80 + 180 x 4 = 800 -> 2000 mm
    step_length=4,           # x 2.5 mm = 10 mm
    profile=a121.Profile.PROFILE_2,
    hwaas=16,
    sweeps_per_frame=4,
    frame_rate=15.0,
)
DISTANCES_M = (SENSOR_CONFIG.start_point
               + np.arange(SENSOR_CONFIG.num_points) * SENSOR_CONFIG.step_length) * 0.0025


def find_port(name, keywords):
    for p in serial.tools.list_ports.comports():
        text = f"{p.device} {p.description} {p.manufacturer}".lower()
        if any(k in text for k in keywords):
            return p.device
    names = ", ".join(p.device for p in serial.tools.list_ports.comports()) or "none found"
    raise RuntimeError(f"Couldn't identify the {name}. Use --{name.lower()} <port>\nPorts: {names}")


def save_npz(path, **data):
    """Write to a hidden temp file then rename, so readers never see a half-written file."""
    tmp = path.with_name("." + path.name)
    np.savez(tmp, **data)
    os.replace(tmp, path)


def position_from_mm(mm):
    return int(round((mm - START_OFFSET_MM) / STEP_MM)) + 1


def mm_from_position(pos):
    return START_OFFSET_MM + (pos - 1) * STEP_MM


def mask_index(pos):
    """Mask number 0..23 (CSV column order) shown at carriage position pos (1..24)."""
    return NUM_POSITIONS - pos if MASK_REVERSED else pos - 1


def position_of_mask(m):
    return NUM_POSITIONS - m if MASK_REVERSED else m + 1


def mask_at(pos):
    col = mask_index(pos)
    return MASK[:, col:col + 4]


def load_scene(folder):
    """Complex scene map from the mask<NN>.npz captures in folder (NaN rows = not captured)."""
    stack = np.full((NUM_POSITIONS, len(DISTANCES_M)), np.nan, dtype=complex)
    for f in folder.glob("mask*.npz"):
        with np.load(f) as d:
            stack[int(f.stem[4:])] = d["frames"].mean(axis=(0, 1))
    return stack


def scene_title(scene):
    return {AMBIENT: "Ambient (start)", AMBIENT_END: "Ambient (end)"}.get(scene, f"Scene {scene}")


def scene_colour(scene):
    n = len(list((SAVE_DIR / scene).glob("mask*.npz")))
    return SCENE_FULL if n >= NUM_POSITIONS else SCENE_PARTIAL if n else SCENE_NONE


_means = {}   # capture file -> (mtime, complex mean); shared by the beam loads


def capture_mean(f):
    """Complex mean over frames and sweeps of a capture, cached until the file changes."""
    mtime = f.stat().st_mtime_ns
    hit = _means.get(f)
    if hit is None or hit[0] != mtime:
        with np.load(f) as d:
            hit = _means[f] = (mtime, d["frames"].mean(axis=(0, 1)))
    return hit[1]


def normalise_across_masks(patterns):
    """Divide each scene by its mean over all masks (patterns: masks x 9 x 9). Scenes
    missing any mask become NaN, since a partial mean would bias them."""
    return patterns / patterns.mean(axis=0)


def beam_pattern(m, lo, hi, ambient):
    """9x9 per-scene sum over lo..hi mm of |capture - ambient| for mask m (NaN = not
    captured). ambient: AMBIENT, AMBIENT_END or None for no subtraction.
    Returns (pattern, subtracted): no subtraction when that mask has no ambient."""
    in_range = (DISTANCES_M * 1000 >= lo) & (DISTANCES_M * 1000 <= hi)
    f = SAVE_DIR / (ambient or AMBIENT) / f"mask{m:02d}.npz"
    subtracted = ambient is not None and f.exists()
    ambient = capture_mean(f) if subtracted else 0
    pattern = np.full((9, 9), np.nan)
    for s in SCENES:
        f = SAVE_DIR / s / f"mask{m:02d}.npz"
        if f.exists():
            pattern[int(s[0]), int(s[1])] = np.abs((capture_mean(f) - ambient)[in_range]).sum()
    return pattern, subtracted


SMOOTH_PX = 161   # samples across the grid in the exported heatmaps


def smooth(pattern):
    """The 9x9 pattern as a smooth surface over 0..8 (SMOOTH_PX square) through the
    scenes: a bicubic spline when all are captured, else piecewise cubic over the
    captured ones (NaN outside them)."""
    have = np.isfinite(pattern)
    g = np.linspace(0, 8, SMOOTH_PX)
    if have.all():
        return RectBivariateSpline(range(9), range(9), pattern)(g, g)
    if have.sum() < 3:
        return np.full((SMOOTH_PX, SMOOTH_PX), np.nan)
    rows, cols = np.nonzero(have)
    return griddata((rows, cols), pattern[have], tuple(np.meshgrid(g, g, indexing="ij")),
                    method="cubic")


def heatmap(ax, pattern, norm, title):
    img = ax.imshow(smooth(pattern), cmap=matplotlib.colormaps["viridis"].with_extremes(bad="#dddddd"),
                    norm=norm, extent=(0, 8, 8, 0))   # row 0 at the top, like the buttons
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)
    return img


def export_beams(results, lo, hi, subtract, export_dir, normalised=False,
                 ambient_label="ambient"):
    """Writes export_dir/beam_<time>/: all masks, each normalised to 0..1 (grid.png),
    each mask on its own scale (maskNN.png), and the numbers (beam_patterns.npz). The
    images are smooth surfaces through the scenes (smooth()).
    results[m] = (9x9 pattern, ambient subtracted) for mask m."""
    out = export_dir / time.strftime("beam_%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    patterns = np.array([p for p, _ in results])
    titles = [f"Mask {m} ({np.isfinite(p).sum()}/{len(SCENES)})"
              + ("\nno ambient" if subtract and not sub else "")
              for m, (p, sub) in enumerate(results)]
    caption = f"{lo:.0f}-{hi:.0f} mm, {ambient_label + ' subtracted' if subtract else 'raw'}"
    unit = "Relative to scene mean" if normalised else "Intensity"
    if normalised:
        caption += ", normalised across masks"

    fig = Figure(figsize=(10, 15), dpi=120, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(6, 4)
    norm = matplotlib.colors.Normalize(0, 1)
    for m, ax in enumerate(axes.flat):
        p = patterns[m]
        img = heatmap(ax, (p - np.nanmin(p)) / (np.nanmax(p) - np.nanmin(p))
                      if np.isfinite(p).sum() > 1 else p, norm, titles[m])
    fig.colorbar(img, ax=axes, shrink=0.6, label=f"{unit}, each mask scaled to 0-1")
    fig.suptitle(f"Beam patterns: {caption}")
    fig.savefig(out / "grid.png")

    for m in range(NUM_POSITIONS):
        fig = Figure(figsize=(5, 4.3), dpi=120, layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        p = patterns[m]
        norm = matplotlib.colors.Normalize(*(np.nanmin(p), np.nanmax(p))
                                           if np.isfinite(p).any() else (0, 1))
        sub = results[m][1]
        title = (f"Mask {m} ({np.isfinite(p).sum()}/{len(SCENES)}), {lo:.0f}-{hi:.0f} mm\n"
                 + ("ambient subtracted" if sub else "raw (no ambient captured)" if subtract
                    else "raw"))
        if normalised:
            title += ", normalised"
        fig.colorbar(heatmap(ax, p, norm, title), ax=ax, label=unit)
        ax.set_xlabel("Scene column")
        ax.set_ylabel("Scene row")
        fig.savefig(out / f"mask{m:02d}.png")

    np.savez(out / "beam_patterns.npz", patterns=patterns, range_mm=(lo, hi),
             normalised_across_masks=normalised,
             ambient_subtracted=[sub for _, sub in results])
    return out


class Settings:
    """Remembers tk variables between runs in SETTINGS_FILE, saved shortly after any
    change. bind() restores a saved value into the variable, then tracks it."""

    def __init__(self, root, path):
        self.root = root
        self.path = path
        self.values = json.loads(path.read_text()) if path.exists() else {}
        self.vars = {}
        self.pending = False

    def bind(self, name, var):
        if name in self.values:
            var.set(self.values[name])
        self.vars[name] = var
        var.trace_add("write", lambda *args: self.schedule())

    def schedule(self):
        if not self.pending:
            self.pending = True
            self.root.after(500, self.save)

    def save(self):
        self.pending = False
        for name, var in self.vars.items():
            try:
                self.values[name] = var.get()
            except tk.TclError:
                pass   # half-typed entry: keep the last valid value
        tmp = self.path.with_name("." + self.path.name)
        tmp.write_text(json.dumps(self.values, indent=1))
        os.replace(tmp, self.path)


class ScanAborted(Exception):
    pass


class CaptureApp:
    def __init__(self, root, ser, client, open_arduino, open_radar):
        self.root = root
        self.settings = Settings(root, SETTINGS_FILE)
        self.ser = ser
        self.client = client
        self.open_arduino = open_arduino   # used by the Reconnect buttons
        self.open_radar = open_radar
        self.radar_restart = False         # asks radar_loop to exit for a reconnect
        self.latest = {}
        self.message = "Homing..."
        self.status = ""
        self.busy = False          # capture, scan or calibration in progress
        self.abort = False
        self.running = True
        self.ready = threading.Event()   # set when the Arduino reports READY after homing
        self.profile = None        # latest complex sweep mean vs distance
        self.profile_dirty = False
        self.ymax = 1.0
        self.radar_frames = 0      # counters for the fps readout
        self.resyncs = 0           # radar session restarts to clear timestamp lag
        self.drawn_frames = 0
        self.fps_time = time.time()
        self.recording = None      # frames being collected by the radar thread
        self.record_done = threading.Event()
        self.shown_pos = None
        self.advance = tk.BooleanVar(value=False)
        self.settings.bind("main.advance", self.advance)
        # Scene map: one row per mask, each the complex mean of one capture
        self.scene = tk.StringVar(value=SCENES[0])
        self.settings.bind("main.scene", self.scene)
        self.stack = load_scene(SAVE_DIR / self.scene.get())
        self.stack_name = scene_title(self.scene.get())
        self.stack_dirty = False
        self.scene_saved = False   # a capture was saved; recolour the scene button
        # Ambient reference (complex, mask-indexed rows like the scene map), subtracted for display only
        self.ambient_source = tk.StringVar(value=AMBIENT)   # which ambient scan to subtract
        self.settings.bind("main.ambient_source", self.ambient_source)
        self.ambient = self.load_ambient()
        self.subtract = tk.BooleanVar(value=False)
        self.settings.bind("main.subtract", self.subtract)
        if self.ambient is None:
            self.subtract.set(False)   # the saved ambient may no longer exist
        self.demux = None          # DemuxWindow while it is open
        SAVE_DIR.mkdir(exist_ok=True)

        root.title("Radar Capture")
        root.geometry("1550x820")

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
        # Go-to buttons: one per mask, sends 'a'..'x' (absolute move to its position)
        grid = tk.Frame(mrow)
        grid.pack(side="left")
        self.goto_buttons = []
        for m in range(NUM_POSITIONS):
            b = tk.Button(grid, text=str(m), width=3,
                          command=lambda m=m: self.goto(position_of_mask(m)))
            b.grid(row=m // 6, column=m % 6, padx=1, pady=1)
            self.goto_buttons.append(b)
        self.goto_bg = self.goto_buttons[0].cget("bg")

        self.lbl_status = tk.Label(left, text="", font=("Helvetica", 12))
        self.lbl_msg = tk.Label(left, text="", font=("Helvetica", 10), fg="#b00", wraplength=380)
        self.lbl_errors = tk.Label(left, text="", font=("Helvetica", 9))
        self.lbl_status.pack()
        self.lbl_msg.pack()
        self.lbl_errors.pack()
        self.errors_shown = (0, 0)

        row = tk.Frame(left)
        row.pack(pady=3)
        self.btn_prev = tk.Button(row, text="◀ Prev", width=8, command=lambda: self.move("F"))
        self.btn_capture = tk.Button(row, text=f"Capture {FRAMES_PER_CAPTURE}", width=12,
                                     command=self.capture)
        self.btn_next = tk.Button(row, text="Next ▶", width=8, command=lambda: self.move("B"))
        for b in (self.btn_prev, self.btn_capture, self.btn_next):
            b.pack(side="left", padx=3)

        actions = tk.Frame(left)
        actions.pack(pady=3)
        self.btn_scan = tk.Button(actions, text=f"Scan all {NUM_POSITIONS}", width=20,
                                  command=self.scan)
        self.btn_calibrate = tk.Button(actions, text="Calibrate", width=20,
                                       command=self.calibrate)
        self.btn_radar = tk.Button(actions, text="Reconnect radar", width=20,
                                   command=self.reconnect_radar)
        self.btn_arduino = tk.Button(actions, text="Reconnect Arduino", width=20,
                                     command=self.reconnect_arduino)
        btn_refresh = tk.Button(actions, text="Refresh plots", width=20,
                                command=self.refresh_plots)
        self.btn_export = tk.Button(actions, text="Export beam heatmaps", width=20,
                                    command=self.export)
        # Arduino plays a tune on the stepper; disabled during jobs so captures stay still
        self.btn_celebrate = tk.Button(actions, text="Celebrate!", width=20, fg="#a0a",
                                       command=lambda: self.send("T"))
        btn_demux = tk.Button(actions, text="Demultiplexing...", width=20,
                              command=self.open_demux)
        btn_stop = tk.Button(actions, text="STOP", width=20, fg="#b00", command=self.stop)
        for i, b in enumerate((self.btn_scan, self.btn_calibrate, self.btn_radar,
                               self.btn_arduino, btn_refresh, self.btn_export,
                               self.btn_celebrate, btn_demux, btn_stop)):
            b.grid(row=i // 2, column=i % 2, padx=2, pady=2)

        checks = tk.Frame(left)
        checks.pack()
        tk.Checkbutton(checks, text="Advance after capture",
                       variable=self.advance).pack(side="left")
        self.chk_subtract = tk.Checkbutton(checks, text="Subtract ambient (display only)",
                                           variable=self.subtract, command=self.on_subtract)
        self.chk_subtract.pack(side="left")
        self.ambient_radios = [
            tk.Radiobutton(checks, text=text, variable=self.ambient_source, value=a,
                           command=self.on_ambient_source)
            for text, a in (("start", AMBIENT), ("end", AMBIENT_END))]
        for b in self.ambient_radios:
            b.pack(side="left")
        tk.Label(left, text="C/Enter capture  ·  ←/→ move  ·  Space stop",
                 font=("Helvetica", 9), fg="#666").pack(pady=2)

        # Scene selector: one folder per scene, coloured by how much has been captured
        tk.Label(left, text="Scene (save folder)", font=("Helvetica", 11, "bold")).pack(pady=(6, 0))
        sgrid = tk.Frame(left)
        sgrid.pack()
        self.scene_buttons = {}
        for i, s in enumerate(SCENES + list(AMBIENTS)):
            colour = scene_colour(s)
            b = tk.Radiobutton(sgrid, text=s, variable=self.scene, value=s, indicatoron=False,
                               width=3, font=("Helvetica", 9), bg=colour, selectcolor=colour,
                               offrelief="raised", command=self.select_scene)
            if s in AMBIENTS:   # tall buttons to the right of the 9x9 grid
                b.config(text="Ambient" if s == AMBIENT else "Amb end", wraplength=1, width=2)
                b.grid(row=0, column=9 + AMBIENTS.index(s), rowspan=9,
                       padx=(6 if s == AMBIENT else 1, 1), pady=1, sticky="ns")
            else:
                b.grid(row=i // 9, column=i % 9, padx=1, pady=1)
            self.scene_buttons[s] = b
        legend = tk.Frame(left)
        legend.pack(pady=2)
        for text, colour in (("none", SCENE_NONE), ("partial", SCENE_PARTIAL),
                             ("all 24", SCENE_FULL)):
            tk.Label(legend, text=text, bg=colour, font=("Helvetica", 9),
                     padx=6).pack(side="left", padx=2)
        self.mark_selected_scene()

        # grey = not captured yet
        self.sequential = matplotlib.colormaps["viridis"].with_extremes(bad="#dddddd")

        # ---- Far right: beam pattern of one mask across the 9x9 scenes ----
        beam = tk.Frame(root)
        beam.pack(side="right", fill="y", padx=6, pady=6)
        tk.Label(beam, text="Beam pattern", font=("Helvetica", 11, "bold")).pack()
        bctl = tk.Frame(beam)
        bctl.pack()
        self.beam_mask = tk.IntVar(value=0)
        self.beam_lo = tk.DoubleVar(value=900.0)
        self.beam_hi = tk.DoubleVar(value=1200.0)
        for name in ("beam_mask", "beam_lo", "beam_hi"):
            self.settings.bind(f"main.{name}", getattr(self, name))
        tk.Label(bctl, text="Mask").grid(row=0, column=0)
        tk.Spinbox(bctl, from_=0, to=NUM_POSITIONS - 1, width=4, textvariable=self.beam_mask,
                   command=self.on_beam).grid(row=0, column=1, padx=(0, 8))
        tk.Label(bctl, text="Range (mm)").grid(row=0, column=2)
        for i, var in enumerate((self.beam_lo, self.beam_hi)):
            e = tk.Entry(bctl, textvariable=var, width=6)
            e.grid(row=0, column=3 + i, padx=1)
            e.bind("<Return>", lambda e: self.on_beam())
        self.normalise = tk.BooleanVar(value=False)
        self.settings.bind("main.normalise", self.normalise)
        tk.Checkbutton(bctl, text="Normalise each scene across masks (complete scenes only)",
                       variable=self.normalise, command=self.on_beam).grid(
            row=1, column=0, columnspan=5, sticky="w")
        # 3D surface over the 9x9 scene grid; drag to orbit, right-drag to zoom
        bfig = Figure(figsize=(4.4, 4.4), dpi=100)
        self.beam_canvas = FigureCanvasTkAgg(bfig, master=beam)
        self.beam_canvas.get_tk_widget().pack()
        self.ax_beam = bfig.add_subplot(projection="3d")   # after the canvas, so orbiting hooks up
        self.beam_cols, self.beam_rows = np.meshgrid(range(9), range(9))
        self.beam_norm = matplotlib.colors.Normalize()
        self.beam_artists = []
        self.ax_beam.set_xlim(0, 8)
        self.ax_beam.set_ylim(8, 0)   # row 0 at the back, like the scene buttons
        self.ax_beam.set_xlabel("Scene column")
        self.ax_beam.set_ylabel("Scene row")
        self.ax_beam.set_zlabel("Intensity")
        bfig.colorbar(matplotlib.cm.ScalarMappable(self.beam_norm, self.sequential),
                      ax=self.ax_beam, shrink=0.6, pad=0.18)
        # Scene files load in a background thread; the throbber spins while they do
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
        self.beam_gen = 0          # latest request; older results are dropped
        self.beam_result = None
        self.beam_dirty = True

        # ---- Right: intensity plots ----
        right = tk.Frame(root)
        right.pack(side="right", fill="both", expand=True)
        self.lbl_fps = tk.Label(right, text="", font=("Helvetica", 10), fg="#666")
        self.lbl_fps.pack(anchor="e", padx=10)
        fig = Figure(figsize=(7, 7.8), dpi=100)
        # Right-hand column holds the colour bars, so the three plots stay x-aligned
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 0.6, 4], width_ratios=[40, 1])
        self.ax_line = fig.add_subplot(gs[0, 0])
        self.ax_color = fig.add_subplot(gs[1, 0], sharex=self.ax_line)
        self.ax_stack = fig.add_subplot(gs[2, 0], sharex=self.ax_line)
        for ax in (self.ax_line, self.ax_color):
            ax.tick_params(labelbottom=False)
        dist_mm = DISTANCES_M * 1000
        # animated artists are left out of full redraws and blitted on top each frame
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
            np.abs(self.stack), aspect="auto", cmap=self.sequential, interpolation="nearest",
            extent=(dist_mm[0] - half, dist_mm[-1] + half, NUM_POSITIONS - 0.5, -0.5))
        self.ax_stack.set_yticks([0, 4, 8, 12, 16, 20, 23])
        self.ax_stack.set_ylabel("Mask")
        self.ax_stack.set_xlabel("Distance (mm)")
        self.ax_stack.set_title(" ", fontsize=10)   # reserve space; text set in draw_stack
        fig.colorbar(self.stack_img, cax=fig.add_subplot(gs[2, 1]))
        # Beam pattern selection: range as dashed lines, mask as a faint row outline
        self.stack_range = [self.ax_stack.axvline(x, color="red", linestyle="--", linewidth=1)
                            for x in (self.beam_lo.get(), self.beam_hi.get())]
        self.stack_row = Rectangle((0, -0.5), 1, 1, transform=self.ax_stack.get_yaxis_transform(),
                                   fill=False, edgecolor="white", alpha=0.7, linewidth=1.2)
        self.ax_stack.add_patch(self.stack_row)
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
    def draw_mask(self, pos):
        self.mask_canvas.delete("all")
        if not 1 <= pos <= NUM_POSITIONS:
            return
        self.draw_pattern(self.mask_canvas, mask_at(pos))

    def draw_pattern(self, canvas, pattern):
        c = self.cell
        for r, row in enumerate(pattern):
            for k, v in enumerate(row):
                canvas.create_rectangle(k * c, r * c, (k + 1) * c, (r + 1) * c,
                                        fill=MASK_ON if v else MASK_OFF, outline="#444")

    def on_full_draw(self, event):
        # Axes, labels and grid were just redrawn: keep them as the blit background
        self.backgrounds = [self.canvas.copy_from_bbox(ax.bbox)
                            for ax in (self.ax_line, self.ax_color)]
        self.blit()

    def blit(self):
        for ax, bg, artist in zip((self.ax_line, self.ax_color), self.backgrounds,
                                  (self.line, self.img)):
            self.canvas.restore_region(bg)
            ax.draw_artist(artist)
            self.canvas.blit(ax.bbox)

    def load_ambient(self):
        """The chosen ambient scan, once a full scan of it exists (else None)."""
        folder = SAVE_DIR / self.ambient_source.get()
        return load_scene(folder) if (folder / "scene_map.npz").exists() else None

    def on_ambient_source(self):
        self.ambient = self.load_ambient()
        if self.ambient is None:
            self.subtract.set(False)
        self.on_subtract()

    def on_subtract(self):
        self.ymax = 0.0          # force a y-axis rescale on the next frame
        self.stack_dirty = True
        self.beam_dirty = True

    def on_beam(self):
        self.beam_dirty = True

    def start_beam(self):
        """Read the beam settings here (Tk thread) and load the scenes in the background."""
        self.beam_gen += 1
        m, lo, hi = self.beam_mask.get(), self.beam_lo.get(), self.beam_hi.get()
        threading.Thread(target=self.load_beam, daemon=True,
                         args=(self.beam_gen, m, lo, hi,
                               self.ambient_source.get() if self.subtract.get() else None,
                               self.normalise.get())).start()
        for line, x in zip(self.stack_range, (lo, hi)):
            line.set_xdata([x, x])
        self.stack_row.set_y(m - 0.5)
        self.stack_dirty = True
        # Loads are often near-instant: keep spinning long enough to be seen
        self.throb_until = time.time() + THROB_MIN_S
        self.beam_loading = True
        if not self.throbbing:
            self.throbbing = True
            self.throbber.itemconfig(self.throb_arc, state="normal")
            self.throb()

    def throb(self):
        if self.beam_loading or time.time() < self.throb_until:
            start = (float(self.throbber.itemcget(self.throb_arc, "start")) - 30) % 360
            self.throbber.itemconfig(self.throb_arc, start=start)
            self.root.after(50, self.throb)
        else:
            self.throbbing = False
            self.throbber.itemconfig(self.throb_arc, state="hidden")

    def load_beam(self, gen, m, lo, hi, ambient, normalise):
        pattern = np.full((9, 9), np.nan)
        try:
            if normalise:
                patterns = np.array([beam_pattern(k, lo, hi, ambient)[0]
                                     for k in range(NUM_POSITIONS)])
                pattern = normalise_across_masks(patterns)[m]
            else:
                pattern = beam_pattern(m, lo, hi, ambient)[0]
        except Exception as e:   # still deliver a result, so the throbber stops
            self.message = f"Beam pattern load error: {e}"
        if gen == self.beam_gen:
            self.beam_result = (m, lo, hi, normalise, pattern)

    def draw_beam(self, m, lo, hi, normalise, pattern):
        self.beam_loading = False
        self.beam_mask_canvas.delete("all")
        self.draw_pattern(self.beam_mask_canvas, MASK[:, m:m + 4])
        # Replace the plotted artists rather than clearing, so the orbit view is kept
        for a in self.beam_artists:
            a.remove()
        self.beam_artists = []
        have = ~np.isnan(pattern)
        if have.any():
            self.beam_norm.vmin, self.beam_norm.vmax = np.nanmin(pattern), np.nanmax(pattern)
            # NaN (uncaptured) cells leave holes in the surface
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
        self.ax_beam.set_title(f"Mask {m}, {lo:.0f}-{hi:.0f} mm"
                               + (", normalised" if normalise else ""), fontsize=10)
        self.ax_beam.set_zlabel("x scene mean" if normalise else "Intensity")
        self.beam_canvas.draw()

    def draw_profile(self):
        p = self.profile
        pos = self.shown_pos
        if self.subtract.get() and pos and 1 <= pos <= NUM_POSITIONS:
            p = p - np.nan_to_num(self.ambient[mask_index(pos)])
        p = np.abs(p)
        self.line.set_ydata(p)
        self.img.set_data(p[None, :])
        # Rescaling the y-axis needs a slow full redraw, so only do it when the
        # signal outgrows the axis or falls well below it. The strip shares that
        # scale, so its colour bar only changes on those full redraws.
        peak = max(p.max(), 1.0)
        if peak > self.ymax or peak < 0.4 * self.ymax:
            self.ymax = peak * 1.25
            self.ax_line.set_ylim(0, self.ymax)
            self.img.set_clim(0, self.ymax)
            self.canvas.draw()   # triggers on_full_draw, which blits the new data
        elif self.backgrounds:
            self.blit()

    def draw_stack(self):
        shown = self.stack
        if self.subtract.get():
            shown = shown - np.nan_to_num(self.ambient)
        shown = np.abs(shown)
        if not np.isnan(shown).all():
            self.stack_img.set_clim(np.nanmin(shown), np.nanmax(shown))
        self.stack_img.set_data(shown)
        self.ax_stack.set_title(f"{self.stack_name}: {FRAMES_PER_CAPTURE}-frame mean per mask",
                                fontsize=10)
        self.canvas.draw()   # full redraw; on_full_draw re-captures the blit backgrounds

    def select_scene(self):
        self.select_scene_data()
        self.mark_selected_scene()
        self.beam_dirty = True

    def mark_selected_scene(self):
        for s, b in self.scene_buttons.items():
            b.config(font=("Helvetica", 9, "bold" if s == self.scene.get() else "normal"))

    def plot_tick(self):
        if not self.running:
            return
        # One failed draw must not end the loop: it only reschedules itself here
        try:
            self.plot_step()
        except Exception as e:
            traceback.print_exc()
            self.message = f"Plot error: {e}"
        self.root.after(PLOT_MS, self.plot_tick)

    def refresh_plots(self):
        self.ymax = 0.0          # rescale the live plot on the next frame
        self.beam_dirty = True
        self.stack_dirty = True
        self.message = ""
        self.canvas.draw()

    def plot_step(self):
        if self.stack_dirty:
            self.stack_dirty = False
            self.draw_stack()
        if self.beam_dirty:
            self.beam_dirty = False
            self.start_beam()
        if self.beam_result:
            result, self.beam_result = self.beam_result, None
            self.draw_beam(*result)
        if self.profile_dirty:
            self.profile_dirty = False
            self.draw_profile()
            self.drawn_frames += 1
        elapsed = time.time() - self.fps_time
        if elapsed >= 1:
            self.lbl_fps.config(text=f"Radar {self.radar_frames / elapsed:.1f} fps  ·  "
                                     f"display {self.drawn_frames / elapsed:.1f} fps  ·  "
                                     f"radar resyncs {self.resyncs}")
            self.radar_frames = self.drawn_frames = 0
            self.fps_time = time.time()

    # ---- Stepper ----
    def send(self, cmd):
        try:
            self.ser.write(cmd.encode())
        except serial.SerialException:
            self.message = "Arduino connection lost"

    def move(self, cmd):
        if self.busy:
            self.message = "Wait for the capture to finish before moving"
            return
        self.send(cmd)

    def goto(self, pos):
        if self.busy:
            self.message = "Wait for the capture to finish before moving"
            return
        self.send(chr(ord("a") + pos - 1))

    def stop(self):
        self.abort = True
        self.send("S")

    def heartbeat(self):
        # Own thread so GUI stalls (window creation, full redraws) can't starve the Arduino
        while self.running:
            self.send("H")
            time.sleep(HEARTBEAT_S)

    def tick(self):
        if not self.running:
            return
        if "Distance_mm" in self.latest:
            mm = self.latest["Distance_mm"]
            pos = position_from_mm(mm)
            valid = 1 <= pos <= NUM_POSITIONS
            self.lbl_dist.config(text=f"Position {pos}  ·  {mm:.2f} mm")
            self.lbl_pos.config(text=f"Mask {mask_index(pos)}" if valid else "Mask --")
            if pos != self.shown_pos:
                self.shown_pos = pos
                self.draw_mask(pos)
                for m, b in enumerate(self.goto_buttons):
                    b.config(bg=MASK_ON if valid and m == mask_index(pos) else self.goto_bg)
        if self.scene_saved:
            self.scene_saved = False
            for s, b in self.scene_buttons.items():
                colour = scene_colour(s)
                b.config(bg=colour, selectcolor=colour)
        self.lbl_status.config(text=self.status)
        self.lbl_msg.config(text=self.message)
        errors = (FrameErrors.radar, FrameErrors.arduino)
        if errors != self.errors_shown:
            self.errors_shown = errors
            self.message = "Frame error - corrupted message discarded"
        self.lbl_errors.config(text=f"Frame errors: radar {errors[0]} · arduino {errors[1]}",
                               fg="#b00" if any(errors) else "#666")
        state = "disabled" if self.busy else "normal"
        for b in (self.btn_capture, self.btn_scan, self.btn_calibrate,
                  self.btn_radar, self.btn_arduino, self.btn_export, self.btn_celebrate,
                  self.btn_prev, self.btn_next, *self.goto_buttons,
                  *self.scene_buttons.values()):
            b.config(state=state)
        self.chk_subtract.config(state="normal" if self.ambient is not None else "disabled")
        for b, a in zip(self.ambient_radios, AMBIENTS):
            full = (SAVE_DIR / a / "scene_map.npz").exists()
            b.config(state="normal" if full and not self.busy else "disabled")
        self.root.after(UI_MS, self.tick)

    def reader(self):
        while self.running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
            except serial.SerialException:
                self.message = "Arduino connection lost"
                return
            if not line:
                continue
            if ARDUINO_LINE.fullmatch(line):
                for part in line.split(","):
                    name, _, value = part.partition(":")
                    self.latest[name] = float(value)
            elif ":" in line or not line.isprintable():
                FrameErrors.arduino += 1   # garbled telemetry line
            elif line.startswith("READY"):
                self.message = ""
                self.ready.set()
            else:
                self.message = line

    def wait_for_position(self, pos):
        target = mm_from_position(pos)
        deadline = time.time() + MOVE_TIMEOUT_S
        settled_since = None
        while time.time() < deadline:
            if self.abort:
                raise ScanAborted
            if abs(self.latest.get("Distance_mm", -1) - target) < ARRIVE_TOLERANCE_MM:
                settled_since = settled_since or time.time()
                if time.time() - settled_since >= SETTLE_S:
                    return
            else:
                settled_since = None
            time.sleep(0.05)
        raise ScanAborted(f"Timed out moving to position {pos}")

    # ---- Radar ----
    def radar_loop(self):
        """Streams continuously: feeds the live plot and any capture in progress."""
        self.client.start_session()
        ref = None   # (radar tick_time, wall time) at the first frame of the session
        while self.running and not self.radar_restart:
            try:
                result = self.client.get_next()
                # Radar time slowly falls behind the PC (frame backlog or clock drift).
                # Restarting the session flushes queued frames and re-zeroes the drift,
                # so do it between captures before it gets large.
                now = time.monotonic()
                ref = ref or (result.tick_time, now)
                lag = (now - ref[1]) - (result.tick_time - ref[0])
                if abs(lag) > MAX_LAG_S and self.recording is None:
                    self.client.stop_session()
                    self.client.start_session()
                    self.resyncs += 1
                    ref = None
                    continue
            except Exception as e:
                self.message = f"Radar stopped: {e}"
                return
            frame = result.frame
            self.profile = frame.mean(axis=0)
            self.profile_dirty = True
            self.radar_frames += 1
            rec = self.recording
            if rec is not None:
                rec.append(frame)
                if len(rec) == FRAMES_PER_CAPTURE:
                    self.record_done.set()

    def record(self, folder, pos):
        m = mask_index(pos)
        mm = self.latest.get("Distance_mm", float("nan"))
        errors_before = FrameErrors.radar
        self.record_done.clear()
        self.recording = []
        while not self.record_done.wait(0.1):
            self.status = f"Mask {m}: frame {len(self.recording)}/{FRAMES_PER_CAPTURE}"
            if self.abort or not self.radar_thread.is_alive():
                self.recording = None
                raise ScanAborted
        frames = np.stack(self.recording[:FRAMES_PER_CAPTURE])
        self.recording = None
        self.stack[m] = frames.mean(axis=(0, 1))
        self.stack_dirty = True
        if folder == SAVE_DIR / self.ambient_source.get() and self.ambient is not None:
            self.ambient[m] = self.stack[m]   # keep the reference matching the saved files
        folder.mkdir(parents=True, exist_ok=True)
        save_npz(
            folder / f"mask{m:02d}.npz",
            frames=frames,
            distances_m=DISTANCES_M,
            distance_mm=mm,
            position=pos,
            mask_index=m,
            frame_errors=FrameErrors.radar - errors_before,
            mask=mask_at(pos),
            sensor_config=self.client.session_config.sensor_config.to_json(),
            server_info=self.client.server_info.to_json(),
        )
        self.beam_dirty = True   # after the save, so the reload sees the new file
        self.scene_saved = True

    def run_job(self, job):
        if self.busy:
            return
        self.busy = True
        self.abort = False
        threading.Thread(target=self.job_wrapper, args=(job,), daemon=True).start()

    def job_wrapper(self, job):
        try:
            job()
        except ScanAborted as e:
            self.status = str(e) or "Aborted"
        except Exception as e:
            self.message = f"Failed: {e}"
            self.status = ""
        finally:
            self.busy = False

    def capture(self):
        self.run_job(self.capture_job)

    def capture_job(self):
        pos = position_from_mm(self.latest.get("Distance_mm", 0.0))
        if not 1 <= pos <= NUM_POSITIONS:
            self.message = "Not at a mask position - calibrate first"
            return
        self.record(SAVE_DIR / self.scene.get(), pos)
        self.status = f"Saved {self.scene.get()}/mask{mask_index(pos):02d}.npz"
        if self.advance.get():
            self.send("B")

    def calibrate(self):
        self.run_job(self.calibrate_job)

    def calibrate_job(self):
        self.ready.clear()
        self.send("C")
        self.status = "Calibrating (homing)..."
        deadline = time.time() + HOMING_TIMEOUT_S
        while not self.ready.wait(0.1):
            if self.abort or self.message.startswith("EMERGENCY"):
                raise ScanAborted("Calibration stopped")
            if time.time() > deadline:
                raise ScanAborted("Calibration timed out")
        pos = position_of_mask(CALIBRATE_TO_MASK)
        self.send(chr(ord("a") + pos - 1))
        self.status = f"Moving to mask {CALIBRATE_TO_MASK}"
        self.wait_for_position(pos)
        self.status = f"Calibrated - at mask {CALIBRATE_TO_MASK}"

    def scan(self):
        self.run_job(lambda: self.scan_job(SAVE_DIR / self.scene.get(),
                                           scene_title(self.scene.get())))

    def select_scene_data(self):
        """Back to showing the selected scene (thread-safe part of select_scene)."""
        self.stack = load_scene(SAVE_DIR / self.scene.get())
        self.stack_name = scene_title(self.scene.get())
        self.stack_dirty = True

    def scan_job(self, folder, name):
        self.stack = np.full((NUM_POSITIONS, len(DISTANCES_M)), np.nan, dtype=complex)
        self.stack_name = name
        self.stack_dirty = True
        # Start from whichever end is nearer, so back-to-back scans alternate direction
        here = position_from_mm(self.latest.get("Distance_mm", 0.0))
        order = list(range(1, NUM_POSITIONS + 1))
        if here > NUM_POSITIONS / 2:
            order.reverse()
        self.status = f"Moving to position {order[0]}"
        self.send(chr(ord("a") + order[0] - 1))   # absolute move to the first position
        for i, pos in enumerate(order):
            if i:
                self.send("B" if pos > order[i - 1] else "F")
            self.status = f"Moving to mask {mask_index(pos)}"
            self.wait_for_position(pos)
            self.record(folder, pos)
        save_npz(folder / "scene_map.npz", scene=self.stack, distances_m=DISTANCES_M)
        if folder == SAVE_DIR / self.ambient_source.get():
            self.ambient = self.stack.copy()   # full ambient scan: adopt as the reference
        self.status = f"Scan saved to {folder.name}"

    def export(self):
        # Read the settings here (Tk thread); the export itself runs as a job
        lo, hi, subtract = self.beam_lo.get(), self.beam_hi.get(), self.subtract.get()
        normalise = self.normalise.get()
        ambient = self.ambient_source.get() if subtract else None

        def job():
            self.status = "Exporting beam heatmaps..."
            results = [beam_pattern(m, lo, hi, ambient) for m in range(NUM_POSITIONS)]
            if normalise:
                patterns = normalise_across_masks(np.array([p for p, _ in results]))
                results = [(p, sub) for p, (_, sub) in zip(patterns, results)]
            out = export_beams(results, lo, hi, subtract, SAVE_DIR / "exports", normalise,
                               scene_title(ambient).lower() if ambient else "ambient")
            self.status = f"Exported to {out.relative_to(SAVE_DIR.parent)}"
        self.run_job(job)

    def reconnect_radar(self):
        self.run_job(self.reconnect_radar_job)

    def reconnect_radar_job(self):
        self.status = "Reconnecting radar..."
        self.radar_restart = True
        self.radar_thread.join(timeout=5)   # get_next times out if the link is dead
        self.radar_restart = False
        try:
            self.client.close()
        except Exception:
            pass   # the old connection is usually what failed
        self.client = self.open_radar()
        self.radar_thread = threading.Thread(target=self.radar_loop, daemon=True)
        self.radar_thread.start()
        self.message = ""
        self.status = "Radar reconnected"

    def reconnect_arduino(self):
        self.run_job(self.reconnect_arduino_job)

    def reconnect_arduino_job(self):
        self.status = "Reconnecting Arduino..."
        self.ser.close()                  # the reader thread exits on the closed port
        self.reader_thread.join(timeout=1)
        self.ready.clear()
        self.ser = self.open_arduino()    # opening the port resets it, so it re-homes
        self.reader_thread = threading.Thread(target=self.reader, daemon=True)
        self.reader_thread.start()
        self.message = "Homing..."
        self.status = "Arduino reconnected"

    def open_demux(self):
        if self.demux:
            self.demux.win.lift()
        else:
            self.demux = DemuxWindow(self)

    def close(self):
        self.settings.save()
        self.running = False
        self.abort = True
        self.send("S")
        self.ser.close()
        self.radar_thread.join(timeout=1)
        self.client.close()
        self.root.destroy()


def pixel_label(pixel):
    """'45' for a calibration grid point, '4.25,5.5' (row, col) between points."""
    row, col = pixel
    if row == int(row) and col == int(col):
        return f"{int(row)}{int(col)}"
    return f"{row:g},{col:g}"


def demux_folder(slot):
    return SAVE_DIR / "demux" / slot


def folder_colour(folder):
    n = len(list(folder.glob("mask*.npz")))
    return SCENE_FULL if n >= NUM_POSITIONS else SCENE_PARTIAL if n else SCENE_NONE


class DemuxWindow:
    """Second window: scan up to 8 scenarios through all 24 masks (plus their own
    ambient), find targets in range with CA-CFAR on the sum over masks, then localise
    each detection on the 9x9 calibration grid with strategy A or B (demux.py)."""

    def __init__(self, app):
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("Demultiplexing")
        self.win.geometry("1550x940")
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.slot = tk.StringVar(value=DEMUX_SLOTS[0])
        self.strategy = tk.StringVar(value="A")
        self.r_lo, self.r_hi = tk.DoubleVar(value=300.0), tk.DoubleVar(value=1400.0)
        self.guard, self.train = tk.IntVar(value=6), tk.IntVar(value=10)
        self.thr_db = tk.DoubleVar(value=6.0)
        self.min_peak = tk.DoubleVar(value=10.0)   # % of the strongest return in the search range
        self.loc_bins = tk.IntVar(value=2)
        self.subpixel = tk.IntVar(value=1)         # beam interpolation factor: 1 = off
        self.cal_lo = tk.DoubleVar(value=app.beam_lo.get())
        self.cal_hi = tk.DoubleVar(value=app.beam_hi.get())
        self.cal_ambient = tk.StringVar(value=AMBIENT)   # AMBIENT, AMBIENT_END or "none"
        self.scn_ambient = tk.StringVar(value="demux")   # "demux", AMBIENT, AMBIENT_END or "none"
        for name in ("slot", "strategy", "r_lo", "r_hi", "guard", "train", "thr_db",
                     "min_peak", "loc_bins", "subpixel", "cal_lo", "cal_hi", "cal_ambient",
                     "scn_ambient"):
            app.settings.bind(f"demux.{name}", getattr(self, name))
        self.note, self.truth = tk.StringVar(), tk.StringVar()
        self.results = []          # one dict per detection
        self.sel = 0               # detection shown in the detail plots
        self.counts = {}           # captures per slot, to notice new files
        self.dirty = True
        self.message = ""

        # ---- Left: scenario, capture, parameters, detections ----
        left = tk.Frame(self.win)
        left.pack(side="left", fill="y", padx=8, pady=6)
        tk.Label(left, text="Scenario", font=("Helvetica", 11, "bold")).pack()
        sg = tk.Frame(left)
        sg.pack()
        self.slot_buttons = {}
        for i, s in enumerate(DEMUX_SLOTS):
            b = tk.Radiobutton(sg, text="Ambient (no targets)" if s == AMBIENT else s,
                               variable=self.slot, value=s, indicatoron=False, width=6,
                               offrelief="raised", command=self.select_slot)
            if s == AMBIENT:
                b.grid(row=2, column=0, columnspan=4, sticky="ew", padx=1, pady=1)
            else:
                b.grid(row=i // 4, column=i % 4, padx=1, pady=1)
            self.slot_buttons[s] = b

        acts = tk.Frame(left)
        acts.pack(pady=4)
        self.btn_scan = tk.Button(acts, text=f"Scan all {NUM_POSITIONS}", width=16,
                                  command=self.scan)
        self.btn_capture = tk.Button(acts, text="Capture this mask", width=16,
                                     command=self.capture)
        btn_stop = tk.Button(acts, text="STOP", width=16, fg="#b00", command=app.stop)
        btn_recompute = tk.Button(acts, text="Recompute", width=16, command=self.mark_dirty)
        for i, b in enumerate((self.btn_scan, self.btn_capture, btn_stop, btn_recompute)):
            b.grid(row=i // 2, column=i % 2, padx=2, pady=2)
        self.lbl_status = tk.Label(left, text="", font=("Helvetica", 10))
        self.lbl_msg = tk.Label(left, text="", font=("Helvetica", 9), fg="#b00", wraplength=380)
        self.lbl_status.pack()
        self.lbl_msg.pack()

        info = tk.Frame(left)
        info.pack(pady=2)
        for r, (text, var) in enumerate((("Note", self.note),
                                         ("True pixels (e.g. 34 52)", self.truth))):
            tk.Label(info, text=text).grid(row=r, column=0, sticky="w")
            e = tk.Entry(info, textvariable=var, width=26)
            e.grid(row=r, column=1, padx=2)
            e.bind("<Return>", lambda e: self.save_info())
            e.bind("<FocusOut>", lambda e: self.save_info())

        def entries(frame, row, label, *variables):
            tk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            for k, var in enumerate(variables):
                e = tk.Entry(frame, textvariable=var, width=7)
                e.grid(row=row, column=1 + k, padx=1)
                e.bind("<Return>", lambda e: self.mark_dirty())

        def radios(frame, row, label, var, options):
            tk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            rf = tk.Frame(frame)
            rf.grid(row=row, column=1, columnspan=3, sticky="w")
            for text, value in options:
                tk.Radiobutton(rf, text=text, variable=var, value=value,
                               command=self.mark_dirty).pack(side="left")

        det = tk.LabelFrame(left, text="Detection (CA-CFAR on the sum over masks)")
        det.pack(fill="x", pady=2)
        entries(det, 0, "Search range (mm)", self.r_lo, self.r_hi)
        entries(det, 1, "Guard / training cells", self.guard, self.train)
        entries(det, 2, "Threshold (dB over local mean)", self.thr_db)
        entries(det, 3, "Min peak (% of strongest)", self.min_peak)
        radios(det, 4, "Scenario ambient", self.scn_ambient,
               (("demux", "demux"), ("cal start", AMBIENT), ("cal end", AMBIENT_END),
                ("none", "none")))

        loc = tk.LabelFrame(left, text="Localisation")
        loc.pack(fill="x", pady=2)
        radios(loc, 0, "Strategy", self.strategy,
               (("A: matched filter", "A"), ("B: normalised shapes", "B")))
        entries(loc, 1, "Signature window (± 10 mm bins around peak)", self.loc_bins)
        entries(loc, 2, "Calibration range (mm)", self.cal_lo, self.cal_hi)
        radios(loc, 3, "Calibration ambient", self.cal_ambient,
               (("start", AMBIENT), ("end", AMBIENT_END), ("none", "none")))
        radios(loc, 4, "Sub-pixel (interpolated beams)", self.subpixel,
               (("off", 1), ("x2", 2), ("x4", 4)))

        tk.Label(left, text="Detections (click one for details)",
                 font=("Helvetica", 10, "bold")).pack(pady=(6, 0))
        columns = (("#", 26), ("Range mm", 62), ("SNR dB", 50), ("Pixel", 40),
                   ("Bearing°", 58), ("Elev°", 48), ("Score", 50), ("Margin", 52),
                   ("Error px", 56))
        self.table = ttk.Treeview(left, columns=[c for c, _ in columns], show="headings",
                                  height=8)
        for c, w in columns:
            self.table.heading(c, text=c)
            self.table.column(c, width=w, anchor="center")
        self.table.pack()
        self.table.bind("<<TreeviewSelect>>", self.on_select)
        tk.Button(left, text="Save results (JSON)", command=self.save_results).pack(pady=4)

        # ---- Right: plots ----
        fig = Figure(figsize=(10.5, 9), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(fig, master=self.win)   # before the 3D axes, so orbiting hooks up
        # Sub-figures lay out separately, so the small square map gets its full width
        top, bottom = fig.subfigures(2, 1, height_ratios=[3.4, 2.9])
        self.ax_prof, self.ax_img = top.subplots(2, 1, sharex=True,
                                                 gridspec_kw={"height_ratios": [2, 1.7]})
        gs = bottom.add_gridspec(1, 3, width_ratios=[1.2, 1.4, 2.2])
        self.ax_score = bottom.add_subplot(gs[0])
        self.ax_sig = bottom.add_subplot(gs[1])
        self.ax_3d = bottom.add_subplot(gs[2], projection="3d")
        self.ax_3d.view_init(elev=20, azim=-60)
        dist_mm = DISTANCES_M * 1000
        half = (dist_mm[1] - dist_mm[0]) / 2
        self.mask_img = self.ax_img.imshow(
            np.full((NUM_POSITIONS, len(dist_mm)), np.nan), aspect="auto",
            cmap=app.sequential, interpolation="nearest",
            extent=(dist_mm[0] - half, dist_mm[-1] + half, NUM_POSITIONS - 0.5, -0.5))
        self.ax_img.set_ylabel("Mask")
        self.ax_img.set_xlabel("Distance (mm)")
        fig.colorbar(self.mask_img, ax=self.ax_img, label="|capture - ambient|")
        self.score_img = self.ax_score.imshow(np.full((9, 9), np.nan), cmap=app.sequential,
                                              interpolation="nearest")
        fig.colorbar(self.score_img, ax=self.ax_score, shrink=0.7, pad=0.02)
        self.ax_score.set_xticks(range(9))
        self.ax_score.set_yticks(range(9))
        self.ax_score.tick_params(labelsize=7)
        self.overlays = []         # detection lines and markers, replaced on each draw
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

        self.select_slot()
        self.poll()

    # ---- Slots, notes, capture ----
    def mark_dirty(self):
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
        self.dirty = True          # the error column depends on the truth

    def scan(self):
        slot = self.slot.get()

        def job():
            try:
                self.app.scan_job(demux_folder(slot), f"Demux {slot}")
            finally:
                self.app.select_scene_data()   # main window back to its own scene
        self.app.run_job(job)

    def capture(self):
        slot = self.slot.get()

        def job():
            pos = position_from_mm(self.app.latest.get("Distance_mm", 0.0))
            if not 1 <= pos <= NUM_POSITIONS:
                self.app.message = "Not at a mask position - calibrate first"
                return
            try:
                self.app.record(demux_folder(slot), pos)
            finally:
                self.app.select_scene_data()
            self.app.status = f"Saved demux/{slot}/mask{mask_index(pos):02d}.npz"
        self.app.run_job(job)

    def truth_pixels(self):
        return [(int(t[0]), int(t[1])) for t in self.truth.get().replace(",", " ").split()
                if len(t) == 2 and t.isdigit()]

    # ---- Processing ----
    def compute(self):
        self.message = ""
        slot, strategy = self.slot.get(), self.strategy.get()
        lo, hi = self.r_lo.get(), self.r_hi.get()
        scale = 10 ** (self.thr_db.get() / 20)    # amplitude ratio
        scn = load_scene(demux_folder(slot))
        used = np.isfinite(scn).all(axis=1)
        choice = self.scn_ambient.get()
        if choice == "none":
            amb = 0
        else:
            amb = load_scene(demux_folder(AMBIENT) if choice == "demux" else SAVE_DIR / choice)
            missing = int((np.isnan(amb).all(axis=1) & used).sum())
            if missing and slot != AMBIENT:
                self.message = f"Scenario ambient missing for {missing} mask(s): not subtracted there"
            amb = np.nan_to_num(amb)
        self.y = np.abs(scn - amb)                 # masks x range bins, NaN = not captured
        self.used = used
        dist_mm = DISTANCES_M * 1000
        self.profile = np.nansum(self.y[used], axis=0) if used.any() else np.zeros(len(dist_mm))
        self.thr = demux.ca_cfar(self.profile, self.guard.get(), self.train.get(), scale)
        self.search = (dist_mm >= lo) & (dist_mm <= hi)
        # Peaks within one guard width are one target; the floor stops CFAR firing on
        # tiny bumps where the local background is almost zero
        self.floor = (self.min_peak.get() / 100 * self.profile[self.search].max()
                      if used.any() and self.search.any() else 0.0)
        peaks = (demux.detect(self.profile, self.thr, self.search, self.guard.get() + 1,
                              self.floor) if used.any() else [])

        # Calibration beams: one row per mask, one column per grid pixel
        cal_amb = self.cal_ambient.get()
        B = np.array([beam_pattern(m, self.cal_lo.get(), self.cal_hi.get(),
                                   None if cal_amb == "none" else cal_amb)[0].ravel()
                      for m in range(NUM_POSITIONS)])
        self.rows = used & np.isfinite(B).any(axis=1)
        # Sub-pixel: candidates on a finer grid, beams linearly interpolated between points
        self.factor = self.subpixel.get()
        self.B = demux.upsample_beams(B[self.rows], self.factor) if self.factor > 1 else B[self.rows]
        n = (demux.GRID_SIZE - 1) * self.factor + 1
        positions = np.array([divmod(i, n) for i in range(n * n)]) / self.factor
        self.results = []
        if peaks and not demux.valid_pixels(self.B).any():
            self.message += ("No grid pixel is calibrated for every mask in this scenario - "
                            "check the calibration range and captures")
            peaks = []
        w = self.loc_bins.get()
        truths = self.truth_pixels()
        for k in peaks:
            y = self.y[self.rows, max(0, k - w):k + w + 1].sum(axis=1)
            r = demux.localise(self.B, y, strategy, positions)
            best = tuple(v / self.factor for v in divmod(r["best"], n))   # (row, col) on the 9x9 grid
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
                pixel_label(r["pixel"]), f"{np.degrees(r['bearing']):.1f}",
                f"{np.degrees(r['elevation']):.1f}", f"{r['score'][r['best']]:.3f}",
                f"{r['margin']:.3f}" if strategy == "A" else f"{r['margin']:.2f}x",
                "" if np.isnan(r["error"]) else f"{r['error']:.1f}"))

    def on_select(self, event):
        chosen = self.table.selection()
        if chosen and int(chosen[0]) != self.sel:
            self.sel = int(chosen[0])
            self.draw()

    def draw(self):
        dist_mm = DISTANCES_M * 1000
        slot, strategy = self.slot.get(), self.strategy.get()
        for a in self.overlays:
            a.remove()
        self.overlays = []

        ax = self.ax_prof
        ax.cla()
        ax.plot(dist_mm, self.profile, label="sum over masks")
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
        ax.set_title(f"{'Ambient' if slot == AMBIENT else slot}: {self.used.sum()}/"
                     f"{NUM_POSITIONS} masks captured, {len(self.results)} detection(s)",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelbottom=False)

        self.mask_img.set_data(self.y)
        if np.isfinite(self.y).any():
            self.mask_img.set_clim(np.nanmin(self.y), np.nanmax(self.y))
        for r in self.results:
            self.overlays.append(self.ax_img.axvline(r["range_mm"], color="white",
                                                     linestyle="--", linewidth=0.8))

        # Detail of the selected detection
        cmap = self.app.sequential if strategy == "A" else self.app.sequential.reversed()
        self.score_img.set_cmap(cmap)
        self.ax_sig.cla()
        truths = self.truth_pixels()
        if self.results:
            r = self.results[self.sel]
            n = (demux.GRID_SIZE - 1) * self.factor + 1
            score = r["score"].reshape(n, n)
            e = 0.5 / self.factor          # keep the axes in calibration-grid units
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
            masks = np.flatnonzero(self.rows)
            b = self.B[:, r["best"]]
            self.ax_sig.bar(masks - 0.2, r["y"] / r["y"].mean(), 0.4, label="measured")
            self.ax_sig.bar(masks + 0.2, b / b.mean(), 0.4,
                            label=f"pixel {pixel_label(r['pixel'])}")
            self.ax_sig.set_title("Signature / mean over masks", fontsize=9)
            self.ax_sig.set_xlabel("Mask")
            self.ax_sig.legend(fontsize=7)
        else:
            self.score_img.set_data(np.full((9, 9), np.nan))
            self.ax_score.set_title("No detection", fontsize=9)
        for t in truths:
            self.overlays.append(self.ax_score.plot(t[1], t[0], "x", color="red",
                                                    markersize=10, mew=2)[0])

        self.draw_scene(truths)
        self.canvas.draw()

    def draw_scene(self, truths, show_grid=True):
        """3D reconstruction: radar at the origin, each detection at its range along the
        direction of its estimated grid pixel. x = across, y = forward, z = up (mm)."""
        ax = self.ax_3d
        elev, azim = ax.elev, ax.azim      # cla() would reset the orbit
        ax.cla()
        ax.view_init(elev=elev, azim=azim)
        far = max(self.r_hi.get(), demux.GRID_DISTANCE_MM)
        half = demux.GRID_CENTRE * demux.GRID_SPACING_MM * far / demux.GRID_DISTANCE_MM
        ax.scatter(0, 0, 0, marker="^", color="k", s=50)
        ax.text(0, 0, 0, "\nradar", fontsize=8, ha="center", va="top")
        # Calibration grid on its board, for scale
        grid = np.array([demux.to_xyz(1.0, *demux.pixel_angles(r, c))
                         for r in range(9) for c in range(9)])
        board = grid * (demux.GRID_DISTANCE_MM / grid[:, 1:2])   # every point on the board plane
        if show_grid:
            ax.scatter(board[:, 0], board[:, 1], board[:, 2], color="#bbb", s=4)
        # True directions as rays (the truth has no range)
        for t in truths:
            u = np.array(demux.to_xyz(far, *demux.pixel_angles(*t)))
            ax.plot([0, u[0]], [0, u[1]], [0, u[2]], "--", color="red", linewidth=1)
        colours = matplotlib.colormaps["plasma"]
        lo, hi = self.r_lo.get(), self.r_hi.get()
        for i, r in enumerate(self.results):
            x, y, z = r["xyz"]
            c = colours((r["range_mm"] - lo) / max(hi - lo, 1))
            ax.plot([0, x], [0, y], [0, z], color=c, linewidth=0.8)
            ax.scatter(x, y, z, color=c, s=120 if i == self.sel else 50,
                       edgecolor="k" if i == self.sel else c, depthshade=False)
            ax.text(x, y, z, f"  {i + 1}", fontsize=9, fontweight="bold")
        ax.set_xlim(-half, half)
        ax.set_ylim(0, far)
        ax.set_zlim(-half, half)
        ax.set_box_aspect((2 * half, far, 2 * half))
        ax.set_xlabel("Across (mm)", fontsize=8)
        ax.set_ylabel("Forward (mm)", fontsize=8)
        ax.set_zlabel("Up (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title("Reconstructed scene", fontsize=9)

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
            "masks_used": np.flatnonzero(self.rows).tolist(),
            "detections": [{"range_mm": float(r["range_mm"]), "peak": float(r["peak"]),
                            "snr_db": float(r["snr_db"]),
                            "pixel": pixel_label(r["pixel"]),
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

    # ---- Loop ----
    def poll(self):
        if not self.app.running or not self.win.winfo_exists():
            return
        try:
            state = "disabled" if self.app.busy else "normal"
            for b in (self.btn_scan, self.btn_capture, *self.slot_buttons.values()):
                b.config(state=state)
            for s, b in self.slot_buttons.items():
                colour = folder_colour(demux_folder(s))
                b.config(bg=colour, selectcolor=colour,
                         font=("Helvetica", 9, "bold" if s == self.slot.get() else "normal"))
            # New captures (e.g. during a scan) refresh the results as they arrive
            counts = {s: len(list(demux_folder(s).glob("mask*.npz"))) for s in DEMUX_SLOTS}
            if counts != self.counts:
                self.counts = counts
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
        self.win.after(UI_MS, self.poll)

    def close(self):
        self.app.demux = None
        self.win.destroy()


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
            radar_port = args.radar or find_port("Radar", RADAR_KEYWORDS)
            client = a121.Client.open(serial_port=radar_port, override_baudrate=RADAR_BAUD)
        client.setup_session(SENSOR_CONFIG)
        return client

    def open_arduino():
        port = args.arduino or find_port("Arduino", ARDUINO_KEYWORDS)
        try:
            return serial.Serial(port, BAUD, timeout=0.1)
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
    CaptureApp(root, ser, client, open_arduino, open_radar)
    root.mainloop()


if __name__ == "__main__":
    main()
