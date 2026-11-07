"""
Rail position sync test: how accurately can radar frames taken while the carriage is
moving be matched to carriage position?

  "Stepped sweep"   -> stops at 93 references 1.75 mm apart (every 4th is one of the 24
                       mask positions, 3.9 .. 164.9 mm; 4 per gap = 23 x 4 + 1) and
                       captures FRAMES_PER_CAPTURE frames at each. Every reference is
                       approached from the switch side, so backlash is the same for all.
  "Sweep there and back" -> for each speed in "Speeds": a constant-speed sweep over the
                       same span, starting from the nearer end, then straight back at the
                       same speed (out = away from the switch). Logs every radar frame
                       and the Arduino's 100 Hz timestamped position reports (firmware 'G').
  "Run full test"   -> stepped sweep, then the sweeps.
  Recordings        -> the test's sweep recordings: tick the ones Analyse should use;
                       "Delete" moves one into the test's deleted/ folder (recoverable).
  "Analyse"         -> for every ticked sweep's frames:
      telemetry position: radar ticks and Arduino micros() are each mapped to the PC
          clock (fit to the earliest arrivals, so queueing delays drop out) and the
          Arduino positions are interpolated at the frame's time;
      radar position: where the frame best fits the stepped references, linearly
          interpolated between neighbouring references (complex or magnitude, inside
          the range window).
    error = telemetry - radar. Frames at constant speed are fitted with
          error = offset + v x latency + backlash x (moving towards the switch)
    (v signed, + away from the switch). The fit is saved to sync_model.json, so a later
    continuous scan can correct each frame:  position = telemetry - fitted error.
    Separating latency from backlash needs two or more speeds in both directions.

  Manual move: the arrow buttons (or keys) step the carriage by the chosen amount.
                       Inverted so the lit window moves the same way on the mask plot:
                       left = away from the switch, right = towards it.
  "Re-home"         -> re-runs the homing pass ('C'), e.g. after skipped steps.
  Live position: a particle filter (ParticleFilter) tracks the carriage from the radar.
      Each particle's state is (offset, offset rate): the true position is the step count
      (corrected by the latency, backlash and offset in sync_model.json once Analyse has
      run) plus the offset, so the step count's movement is the expected translation and
      a skip shows up as the offset changing. Predict: offset += rate dt, the rate decays
      towards 0, plus process noise with a 2x2 covariance (position, rate). Update: each
      frame is matched against all positions (references interpolated on a fine grid) and
      every particle is weighted by the match at its position. Particles start spread over
      all 93 references; after that nothing can teleport, so the estimate can only move
      as far as the transition model allows.
  Live row (bottom): the stepped sweep's mean amplitude (range window) at each mask
      position, with the latest frame's as a dashed line, above the full mask pattern with the 4 columns in view lit at the
      filtered position (yellow, continuous) and the step-count position (dashed); under
      it two heatmaps at the centre of the window: the measurement (RMS difference between
      the latest frame and the interpolated reference at every position, bright = close)
      and the particle filter posterior (log10, particles as white ticks); and the
      filtered (line), single-frame (grey dots), the same particle filter trusting the
      commands TRUST_LEVELS times more (likelihood widened, process noise shrunk) and open-loop (commands only: a trapezoid
      profile per G command from the firmware's speed and acceleration) position minus
      the step count over the last minute.
  Skip detector: when the median of the last SKIP_FRAMES filtered differences exceeds
      the threshold the motor has lost steps: it is flagged, and the motor stopped if
      "Stop on skip" is ticked. Works in manual moves and sweeps; off until the test has
      a stepped sweep. Analyse also counts sweep frames beyond the threshold.

Data: rail_tests/<test>/stepped.npz, sweep_<out|back>_<speed>.npz, sync_model.json
(the latest fit is also copied to rail_tests/sync_model.json).
Needs the arduino.ino with the 'G' command. Radar settings come from radar_capture.py.
  python rail_sync_test.py [--arduino COM12] [--radar COM11] [--mock]
"""

import argparse
import json
import re
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import serial
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.patches import Rectangle

from radar_capture import (
    a121, ARDUINO_LINE, ARDUINO_KEYWORDS, ARRIVE_TOLERANCE_MM, BAUD, DISTANCES_M,
    FRAMES_PER_CAPTURE, FrameErrors, HEARTBEAT_S, MASK, MASK_OFF, MASK_ON, MASK_REVERSED,
    MAX_LAG_S, MOVE_TIMEOUT_S, NUM_POSITIONS, RADAR_BAUD, RADAR_KEYWORDS, SENSOR_CONFIG, SETTINGS_FILE, SETTLE_S,
    START_OFFSET_MM, STEP_MM, UI_MS, ScanAborted, Settings, find_port, save_npz,
)

SAVE_DIR = Path(__file__).parent / "rail_tests"
SUBDIVISIONS = 4                                   # references per 7 mm mask step
REF_SPACING_MM = STEP_MM / SUBDIVISIONS            # 1.75 mm
REF_MM = START_OFFSET_MM + np.arange((NUM_POSITIONS - 1) * SUBDIVISIONS + 1) * REF_SPACING_MM
POS_LINE = re.compile(r"P([0-9A-F]{8}),([0-9A-F]{8})")   # firmware: P<micros>,<distance um> in hex
APPROACH_MM = 1.0        # references are approached from this far on the switch side
SWEEP_ACCEL_MM = 50.0    # firmware SWEEP_ACCEL_MM
MAX_SWEEP_SPEED_MM = 10.0
RUNUP_MARGIN_MM = 1.0    # extra travel beyond the ramp before the first reference
TARGET_RANGE_MM = (0.5, 174.9)   # firmware clamps 'G' targets to this
TAIL_S = 0.3             # keep logging this long after a sweep stops
STEADY = 0.1             # frames within 10 % of the sweep speed count as constant speed
JOG_STEPS_MM = (0.25, 1.75, 7.0)   # manual move sizes
SKIP_FRAMES = 5          # the skip detector uses the median of this many frames
LIVE_MS = 150            # live row: new frames are filtered this often
LIVE_DRAW_S = 0.4        # and the figure redrawn at most this often (a redraw is ~100 ms)
HISTORY_S = 60           # seconds of radar - step count difference shown
GRID_MM = 0.05           # all-positions match grid
PARTICLES_PER_REF = 10   # particle filter: starts with this many at each reference
MATCH_SD_MM = 0.5        # single-frame match sd: sets the likelihood width (tempering)
POS_NOISE = 0.3          # offset random walk, mm/sqrt(s)
RATE_NOISE = 1.0         # offset-rate random walk, mm^2/s^3 (sets the 2x2 process noise)
RATE_DECAY_S = 1.0       # offset rate relaxes to 0 with this time constant
SLIP_FRACTION = 0.05     # extra position noise per mm the step count moves
MAX_DT_S = 0.5           # longest prediction step (e.g. after a pause in frames)
TRUST_LEVELS = (3, 10, 30)   # extra filters trusting the commands this much more than blue
TRUST_COLOURS = ("#9467bd", "#8c564b", "#e377c2")
LABEL_CHARS = 46         # fixed width of the changing left-panel labels (else the window resizes)
LOG_P_FLOOR = -3         # posterior heatmap shows log10 down to this, relative to the peak
POST_BIN_MM = 0.25       # posterior heatmap bin width
JOG_SPEED_MM = 5.0       # firmware MAX_SPEED_MM (G with speed 0)
JOG_ACCEL_MM = 10.0      # firmware ACCEL_MM
SWEEP_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                 "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def to_pc_clock(src, pc):
    """Map device timestamps src (s) onto the PC clock, given each one's PC arrival time.
    Slope by least squares (clock rate difference), then the line is lowered to the 2nd
    percentile of the residuals: arrivals are only ever late, the earliest are truest."""
    s = src - src[0]
    b, a = np.polyfit(s, pc, 1)
    a += np.percentile(pc - (a + b * s), 2)
    return a + b * s


def trapezoid(t, x0, x1, v, a):
    """Position t s into a rest-to-rest move x0 -> x1 with top speed v and acceleration a."""
    d = abs(x1 - x0)
    ta = min(v / a, np.sqrt(d / a))
    v = a * ta
    tc = (d - a * ta ** 2) / v if v else 0.0
    t = min(max(t, 0.0), 2 * ta + tc)
    if t < ta:
        s = a * t ** 2 / 2
    elif t < ta + tc:
        s = a * ta ** 2 / 2 + v * (t - ta)
    else:
        s = d - a * (2 * ta + tc - t) ** 2 / 2
    return x0 + np.sign(x1 - x0) * s


def unwrap_micros(us):
    """micros() wraps every 71.6 minutes."""
    us = us.astype(float)
    return us + 2.0 ** 32 * np.concatenate(([0], np.cumsum(np.diff(us) < 0)))


def radar_positions(profiles, refs, ref_mm):
    """Position where each profile best fits the references (rows of refs at ref_mm),
    interpolating linearly between neighbours: for segment k the fit is
    refs[k] + t (refs[k+1] - refs[k]), t in [0, 1], least squares in closed form.
    Returns positions, inside (the best fit is not clamped at either end of the span)
    and residual (best squared distance / |profile|^2)."""
    R, D = refs[:-1], np.diff(refs, axis=0)
    pr = (profiles @ R.conj().T).real                          # Re<p, R_k>
    pd = (profiles @ D.conj().T).real                          # Re<p, D_k>
    rd = (R.conj() * D).sum(axis=1).real
    dd = (np.abs(D) ** 2).sum(axis=1)
    rr = (np.abs(R) ** 2).sum(axis=1)
    pp = (np.abs(profiles) ** 2).sum(axis=1)
    t = np.clip((pd - rd) / dd, 0, 1)
    cost = pp[:, None] - 2 * pr + rr - 2 * t * (pd - rd) + t ** 2 * dd
    k = cost.argmin(axis=1)
    i = np.arange(len(profiles))
    tb = t[i, k]
    inside = ~(((k == 0) & (tb == 0)) | ((k == len(D) - 1) & (tb == 1)))
    return ref_mm[k] + tb * np.diff(ref_mm)[k], inside, cost[i, k] / pp


def mask_column(mm):
    """First CSV column (continuous) of the 4 mask columns in view with the carriage at mm."""
    pos = (mm - START_OFFSET_MM) / STEP_MM + 1
    return NUM_POSITIONS - pos if MASK_REVERSED else pos - 1


def load_references(folder, lo_mm, hi_mm, magnitude):
    """Stepped references cut to the range window: (positions, refs, prep, scale).
    prep applies the same window (and magnitude) to other profiles. scale divides the
    squared mismatch to give the log likelihood. The bins aren't independent (drift,
    vibration, model error), so the frame noise alone makes the likelihood far too sharp;
    instead the scale is set so a mismatch of MATCH_SD_MM next to a reference costs 1/2:
    scale = 2 MATCH_SD_MM^2 x median |dref/dmm|^2."""
    with np.load(folder / "stepped.npz") as d:
        ref_mm = d["positions_mm"]
        window = (DISTANCES_M * 1000 >= lo_mm) & (DISTANCES_M * 1000 <= hi_mm)
        prep = (lambda x: np.abs(x[..., window])) if magnitude else (lambda x: x[..., window])
        frames = prep(d["frames"].mean(axis=2))            # refs x frames x bins
    refs = frames.mean(axis=1)
    slope = (np.abs(np.diff(refs, axis=0)) ** 2).sum(axis=1) / np.diff(ref_mm) ** 2
    return ref_mm, refs, prep, 2 * MATCH_SD_MM ** 2 * np.median(slope)


class ParticleFilter:
    """Particle filter for the carriage position.

    State per particle: (offset, rate); position = step count + offset. predict(dt, move):
    offset += rate dt and rate *= exp(-dt / RATE_DECAY_S), plus Gaussian noise with the
    constant-velocity covariance RATE_NOISE * [[dt^3/3, dt^2/2], [dt^2/2, dt]] plus
    POS_NOISE^2 dt and (SLIP_FRACTION x the step count's move)^2 on the position. update(profile, mm): the
    frame's squared mismatch against the references interpolated on a GRID_MM grid
    (all positions), turned into a likelihood with the tempered scale and
    looked up at each particle; then systematic resampling when the effective sample
    size drops below half. trust > 1 believes the step count (commands) more than the
    radar: the likelihood is widened by trust and the process noise shrunk by it."""

    def __init__(self, ref_mm, refs, scale, trust=1.0, rng=None):
        self.ref_mm = ref_mm
        self.x = np.arange(ref_mm[0], ref_mm[-1] + GRID_MM / 2, GRID_MM)
        self.k = np.minimum(np.searchsorted(ref_mm, self.x, side="right") - 1, len(ref_mm) - 2)
        self.t = (self.x - ref_mm[self.k]) / np.diff(ref_mm)[self.k]
        k, t = self.k, self.t
        rr = (np.abs(refs) ** 2).sum(axis=1)
        rx = (refs[:-1].conj() * refs[1:]).sum(axis=1).real
        self.ref_power = (1 - t) ** 2 * rr[k] + 2 * t * (1 - t) * rx[k] + t ** 2 * rr[k + 1]
        self.refs = refs
        self.scale = scale * trust ** 2
        self.trust = trust
        self.rng = rng or np.random.default_rng()
        self.particles = None
        self.loglik = self.cost = None

    def reset(self, mm):
        """Particles spread over every reference (jittered within half a spacing)."""
        spacing = np.diff(self.ref_mm).mean()
        x = (np.repeat(self.ref_mm, PARTICLES_PER_REF)
             + self.rng.uniform(-spacing / 2, spacing / 2, len(self.ref_mm) * PARTICLES_PER_REF))
        x = np.clip(x, self.ref_mm[0], self.ref_mm[-1])
        self.particles = np.column_stack([x - mm, np.zeros_like(x)])
        self.weights = np.full(len(x), 1 / len(x))
        self.last = mm

    def predict(self, dt, mm):
        if self.particles is None:
            self.reset(mm)
        move, self.last = mm - self.last, mm
        q = RATE_NOISE * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        q[0, 0] += POS_NOISE ** 2 * dt + (SLIP_FRACTION * move) ** 2
        q /= self.trust ** 2
        p = self.particles
        p[:, 0] += p[:, 1] * dt
        p[:, 1] *= np.exp(-dt / RATE_DECAY_S)
        p += self.rng.multivariate_normal([0, 0], q, len(p))

    def update(self, profile, mm):
        pr = (self.refs.conj() @ profile).real
        cost = ((np.abs(profile) ** 2).sum()
                - 2 * ((1 - self.t) * pr[self.k] + self.t * pr[self.k + 1]) + self.ref_power)
        self.cost = cost
        self.loglik = -(cost - cost.min()) / self.scale
        pos = mm + self.particles[:, 0]
        ll = np.interp(pos, self.x, self.loglik, left=self.loglik.min(), right=self.loglik.min())
        w = self.weights * np.exp(ll - ll.max())
        self.weights = w / w.sum()
        if 1 / (self.weights ** 2).sum() < len(w) / 2:
            u = (self.rng.random() + np.arange(len(w))) / len(w)
            idx = np.minimum(np.searchsorted(np.cumsum(self.weights), u), len(w) - 1)
            self.particles = self.particles[idx]
            self.weights = np.full(len(w), 1 / len(w))

    def estimate(self, mm):
        """(mean, sd) of the position."""
        pos = mm + self.particles[:, 0]
        mean = (self.weights * pos).sum()
        return mean, np.sqrt((self.weights * (pos - mean) ** 2).sum())


def analyse(folder, names, lo_mm, hi_mm, magnitude, skip_mm):
    """Fit the sync model to the named sweeps in folder. Returns (sweeps, model)."""
    ref_mm, refs, prep, _ = load_references(folder, lo_mm, hi_mm, magnitude)
    sweeps = []
    for f in sorted((folder / f"{n}.npz" for n in names), key=lambda f: f.stat().st_mtime):
        with np.load(f) as d:
            tel_t = to_pc_clock(unwrap_micros(d["pos_us"]) * 1e-6, d["pos_pc"])
            tel_mm = d["pos_um"] / 1000
            frame_t = to_pc_clock(d["frame_tick"], d["frame_pc"])
            x_tel = np.interp(frame_t, tel_t, tel_mm)
            v = np.interp(frame_t, tel_t, np.gradient(tel_mm, tel_t))
            x_rad, inside, resid = radar_positions(prep(d["profiles"]), refs, ref_mm)
            speed, towards = float(d["speed_mm_s"]), str(d["direction"]) == "back"
        v_nom = -speed if towards else speed
        steady = inside & (np.abs(v - v_nom) < STEADY * speed)
        err = x_tel - x_rad
        if steady.any():   # drop gross mismatches before fitting
            med = np.median(err[steady])
            mad = np.median(np.abs(err[steady] - med)) + 1e-6
            steady &= np.abs(err - med) < 6 * 1.4826 * mad
        sweeps.append({"name": f.stem, "speed": speed, "towards": towards, "x_tel": x_tel,
                       "x_rad": x_rad, "v": v, "err": err, "steady": steady, "inside": inside,
                       "resid": resid, "t": frame_t - frame_t[0]})

    # Least squares over all constant-speed frames: error = offset + v tau + beta towards
    v = np.concatenate([s["v"][s["steady"]] for s in sweeps])
    tw = np.concatenate([np.full(s["steady"].sum(), float(s["towards"])) for s in sweeps])
    err = np.concatenate([s["err"][s["steady"]] for s in sweeps])
    speeds = {s["speed"] for s in sweeps}
    both = len({s["towards"] for s in sweeps}) == 2
    if both and len(speeds) > 1:
        cols, note = ["offset", "latency", "backlash"], ""
    elif len(sweeps) > 1 and (len(speeds) > 1 or both):
        cols = ["offset", "latency"]
        note = ("one speed in both directions: backlash assumed 0, so it is folded into "
                "the latency" if both else "one direction only: backlash not measurable")
    else:
        cols, note = ["offset"], "one sweep: only a constant offset can be fitted"
    X = np.column_stack([np.ones_like(v), v, tw])[:, :len(cols)]
    coef = np.linalg.lstsq(X, err, rcond=None)[0]
    fit = dict(zip(["offset", "latency", "backlash"], [0.0, 0.0, 0.0]))
    fit.update(zip(cols, coef))
    for s in sweeps:
        s["model"] = fit["offset"] + s["v"] * fit["latency"] + fit["backlash"] * s["towards"]
        s["corrected"] = s["err"] - s["model"]
        s["beyond"] = int((s["inside"] & (np.abs(s["corrected"]) > skip_mm)).sum())
        # One frame per reference: the frame whose corrected position is nearest each one
        pos = s["x_tel"] - s["model"]
        ok = np.flatnonzero(s["steady"])
        picks = ok[np.abs(pos[ok][None, :] - ref_mm[:, None]).argmin(axis=1)] if len(ok) else ok
        s["picks"] = picks[np.abs(pos[picks] - ref_mm[:len(picks)]) < REF_SPACING_MM / 2]
    rms = lambda e: float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")
    model = {
        "offset_mm": float(fit["offset"]),
        "latency_ms": float(fit["latency"] * 1000),
        "backlash_mm": float(fit["backlash"]),
        "fitted": cols, "note": note,
        "rms_before_mm": rms(err),
        "rms_after_mm": rms(np.concatenate([s["corrected"][s["steady"]] for s in sweeps])),
        "frames": int(len(err)),
        "match": "magnitude" if magnitude else "complex", "range_mm": [lo_mm, hi_mm],
        "sweeps": [{"name": s["name"], "speed_mm_s": s["speed"],
                    "direction": "back" if s["towards"] else "out",
                    "frames": int(s["steady"].sum()),
                    "mean_error_mm": float(s["err"][s["steady"]].mean()),
                    "rms_after_mm": rms(s["corrected"][s["steady"]]),
                    "rms_picks_mm": rms(s["corrected"][s["picks"]]),
                    "frames_beyond_skip_threshold": s["beyond"]} for s in sweeps],
        "skip_threshold_mm": skip_mm,
        "test": folder.name, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "usage": "position_mm = telemetry_mm - (offset_mm + v_mm_per_s * latency_ms / 1000"
                 " + backlash_mm * moving_towards_switch)",
    }
    return sweeps, model


class RailTestApp:
    def __init__(self, root, ser, client):
        self.root = root
        self.settings = Settings(root, SETTINGS_FILE)
        self.ser = ser
        self.client = client
        self.send_lock = threading.Lock()
        self.latest = {}
        self.running = True
        self.busy = False
        self.abort = False
        self.status = "Idle"
        self.message = ""
        self.recording = None          # list of frames while a stepped capture runs
        self.record_done = threading.Event()
        self.frame_log = None          # (pc time, tick time, profile) per frame during a sweep
        self.pos_log = None            # (pc time, micros, um) per P line during a sweep
        self.radar_restart = False
        self.session_started = threading.Event()
        self.radar_frames = 0
        self.resyncs = 0
        self.live = deque(maxlen=300)          # (pc time, profile) of recent frames
        self.tel_hist = deque(maxlen=5000)     # (pc time, step-count mm)
        self.towards = False                   # last motion was towards the switch
        self.homing = False
        self.target = None                     # manual move target, mm
        self.live_seen = 0.0                   # pc time of the newest frame processed
        self.live_drawn = 0.0                  # monotonic time of the last live redraw
        self.recent = deque(maxlen=SKIP_FRAMES)  # radar - step count, latest frames
        self.deltas = deque(maxlen=3000)       # (pc time, filtered, single frame, open loop) - step count
        self.plan = None                       # open loop: (send time, from, to, speed, accel)
        self.x_radar = None                    # radar-matched position, mm
        self.tripped = False
        self.skips = 0
        self.refs_key = self.refs = None
        self.tracker = None                    # ParticleFilter for this test's references
        self.trusting = []                     # same, with TRUST_LEVELS
        self.last_frame_t = None
        self.x_sd = 0.0
        self.model_key, self.model = None, {}
        self.rec_key = None                    # (folder, sweep names) shown in Recordings
        self.rec_use = {}                      # (folder, name) -> BooleanVar: use in Analyse

        self.test = tk.StringVar(value="test1")
        self.speeds = tk.StringVar(value="1, 3")
        self.lo = tk.DoubleVar(value=200.0)
        self.hi = tk.DoubleVar(value=2000.0)
        self.magnitude = tk.BooleanVar(value=False)
        self.jog_mm = tk.DoubleVar(value=JOG_STEPS_MM[1])
        self.skip_mm = tk.DoubleVar(value=1.0)
        self.stop_on_skip = tk.BooleanVar(value=True)
        for name in ("test", "speeds", "lo", "hi", "magnitude", "jog_mm", "skip_mm",
                     "stop_on_skip"):
            self.settings.bind(f"rail.{name}", getattr(self, name))

        root.title("Rail position sync test")
        left = tk.Frame(root, padx=8, pady=8)
        left.pack(side="left", fill="y")
        self.lbl_pos = tk.Label(left, font=("Segoe UI", 14, "bold"), anchor="w")
        self.lbl_pos.pack(fill="x")
        self.lbl_status = tk.Label(left, anchor="w", justify="left", wraplength=300, width=LABEL_CHARS)
        self.lbl_status.pack(fill="x")
        self.lbl_msg = tk.Label(left, anchor="w", justify="left", wraplength=300, fg="#b00",
                                width=LABEL_CHARS)
        self.lbl_msg.pack(fill="x")
        self.lbl_radar = tk.Label(left, anchor="w", justify="left", fg="#666", wraplength=300,
                                  width=LABEL_CHARS)
        self.lbl_radar.pack(fill="x")

        form = tk.Frame(left, pady=6)
        form.pack(fill="x")
        rows = (("Test name", self.test, 14), ("Speeds (mm/s, comma separated)", self.speeds, 14),
                ("Match range from (mm)", self.lo, 8), ("Match range to (mm)", self.hi, 8))
        for i, (text, var, width) in enumerate(rows):
            tk.Label(form, text=text, anchor="w").grid(row=i, column=0, sticky="w")
            tk.Entry(form, textvariable=var, width=width).grid(row=i, column=1, sticky="w")
        tk.Checkbutton(form, text="Match on magnitude (not complex)",
                       variable=self.magnitude).grid(row=len(rows), column=0, columnspan=2,
                                                     sticky="w")

        self.btn_stepped = tk.Button(left, text=f"Stepped sweep ({len(REF_MM)} references)",
                                     command=lambda: self.run_job(self.stepped_job))
        self.btn_sweeps = tk.Button(left, text="Sweep there and back (each speed)",
                                    command=lambda: self.run_job(self.sweeps_job))
        self.btn_full = tk.Button(left, text="Run full test",
                                  command=lambda: self.run_job(self.full_job))
        self.btn_analyse = tk.Button(left, text="Analyse", command=self.analyse)
        for b in (self.btn_stepped, self.btn_sweeps, self.btn_full, self.btn_analyse):
            b.pack(fill="x", pady=2)
        tk.Button(left, text="STOP", bg="#c33", fg="white", font=("Segoe UI", 11, "bold"),
                  command=self.stop).pack(fill="x", pady=(8, 2))

        man = tk.LabelFrame(left, text="Manual move (left = away from switch)", padx=4, pady=4)
        man.pack(fill="x", pady=(8, 0))
        self.btn_left = tk.Button(man, text="\u25c0", width=4, command=lambda: self.jog(+1))
        self.btn_right = tk.Button(man, text="\u25b6", width=4, command=lambda: self.jog(-1))
        self.btn_left.grid(row=0, column=0)
        self.btn_right.grid(row=0, column=1, padx=(2, 8))
        for i, mm in enumerate(JOG_STEPS_MM):
            tk.Radiobutton(man, text=f"{mm:g} mm", variable=self.jog_mm,
                           value=mm).grid(row=0, column=2 + i)
        self.btn_home = tk.Button(man, text="Re-home", command=self.rehome)
        self.btn_home.grid(row=1, column=0, columnspan=2, sticky="we", pady=(4, 0))

        skip = tk.LabelFrame(left, text="Skip detector", padx=4, pady=4)
        skip.pack(fill="x", pady=(8, 0))
        tk.Label(skip, text="Threshold (mm)").grid(row=0, column=0, sticky="w")
        tk.Entry(skip, textvariable=self.skip_mm, width=6).grid(row=0, column=1, sticky="w")
        tk.Checkbutton(skip, text="Stop on skip", variable=self.stop_on_skip).grid(
            row=0, column=2, sticky="w")
        self.lbl_skip = tk.Label(skip, anchor="w", justify="left", wraplength=300, width=LABEL_CHARS)
        self.lbl_skip.grid(row=1, column=0, columnspan=3, sticky="w")
        self.rec_frame = tk.LabelFrame(left, text="Recordings (ticked = used by Analyse)",
                                       padx=4, pady=4)
        self.rec_frame.pack(fill="x", pady=(8, 0))
        self.rec_frame.columnconfigure(0, weight=1)
        self.buttons = (self.btn_stepped, self.btn_sweeps, self.btn_full, self.btn_analyse,
                        self.btn_left, self.btn_right, self.btn_home)
        self.lbl_result = tk.Label(left, anchor="w", justify="left", font=("Consolas", 9),
                                   wraplength=320)
        self.lbl_result.pack(fill="x", pady=(10, 0))

        right = tk.Frame(root)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(11, 6.4), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.build_live(right)

        root.bind("<space>", lambda e: self.stop())
        for key, direction in (("<Left>", +1), ("<Right>", -1)):
            root.bind(key, lambda e, d=direction: None if isinstance(e.widget, tk.Entry)
                      else self.jog(d))
        root.protocol("WM_DELETE_WINDOW", self.close)
        for target in (self.reader, self.heartbeat, self.radar_loop):
            threading.Thread(target=target, daemon=True).start()
        self.tick()
        self.live_tick()

    def build_live(self, parent):
        self.live_fig = Figure(figsize=(11, 4))   # laid out on resize only: redraws stay cheap
        gs = self.live_fig.add_gridspec(4, 2, width_ratios=(3, 2), height_ratios=(2, 4, 1, 1))
        ax_mask = self.live_fig.add_subplot(gs[1, 0])
        ax_amp = self.live_fig.add_subplot(gs[0, 0], sharex=ax_mask)
        ax_meas = self.live_fig.add_subplot(gs[2, 0], sharex=ax_mask)
        ax_prob = self.live_fig.add_subplot(gs[3, 0], sharex=ax_mask)
        self.amp_line, = ax_amp.plot([], [], "-", lw=1.2)
        self.amp_now = ax_amp.axhline(0, color="#d62728", ls="--", lw=1, visible=False)   # latest frame
        ax_amp.set_ylabel("Stepped\namplitude", fontsize=8)
        ax_amp.tick_params(labelbottom=False, labelsize=7)
        ax_amp.grid(alpha=0.3)
        self.ax_amp = ax_amp
        ax_delta = self.live_fig.add_subplot(gs[:, 1])
        ax_mask.imshow(MASK, cmap=ListedColormap([MASK_OFF, MASK_ON]), vmin=0, vmax=1)
        ax_mask.set_yticks([])
        ax_mask.tick_params(labelbottom=False)
        self.meas_img = ax_meas.imshow(np.zeros((1, 2)), cmap="magma_r", aspect="auto",
                                       extent=(-0.5, MASK.shape[1] - 0.5, 0, 1))
        ax_meas.set(facecolor="black", yticks=[])
        ax_meas.tick_params(labelbottom=False)
        ax_meas.set_ylabel("Frame -\nref",rotation=0, ha="right", va="center", fontsize=8)
        self.prob_img = ax_prob.imshow(np.full((1, 2), LOG_P_FLOOR), cmap="magma",
                                       vmin=LOG_P_FLOOR, vmax=0, aspect="auto",
                                       extent=(-0.5, MASK.shape[1] - 0.5, 0, 1))
        ax_prob.set(facecolor="black", yticks=[], xlim=(-0.5, MASK.shape[1] - 0.5),
                    xticks=range(0, MASK.shape[1], 2), xlabel="Mask column")
        ax_prob.set_ylabel("Posterior", rotation=0, ha="right", va="center", fontsize=8)
        self.particle_marks, = ax_prob.plot([], [], "|", color="white", ms=8, alpha=0.3)
        self.rect_radar = Rectangle((0, -0.5), 4, MASK.shape[0], fc=(1, 0.9, 0, 0.35),
                                    ec="#ffd400", lw=2.5, visible=False)
        self.rect_tel = Rectangle((0, -0.5), 4, MASK.shape[0], fill=False, ec="white",
                                  ls="--", lw=1.5, visible=False)
        ax_mask.add_patch(self.rect_radar)
        ax_mask.add_patch(self.rect_tel)
        self.ax_mask = ax_mask
        self.raw_dots, = ax_delta.plot([], [], ".", ms=2, color="#bbb", label="single frame")
        self.delta_line, = ax_delta.plot([], [], "-", lw=1.2, label="filtered")
        self.open_line, = ax_delta.plot([], [], "-", lw=1.2, color="#2ca02c", label="open loop (commands)")
        self.trust_lines = [ax_delta.plot([], [], "-", lw=1, color=c, label=f"filter, trust x{k:g}")[0]
                            for k, c in zip(TRUST_LEVELS, TRUST_COLOURS)]
        ax_delta.legend(fontsize=7, loc="upper left")
        self.thr_lines = [ax_delta.axhline(sign, color="#c33", ls="--", lw=0.8)
                          for sign in (1, -1)]
        ax_delta.axhline(0, color="#999", lw=0.5)
        ax_delta.set(xlim=(-HISTORY_S, 0), xlabel="Seconds ago", ylabel="mm")
        # titles as plain text: Axes titles are re-positioned (slowly) on every draw
        ax_delta.text(0.5, 1.02, "Radar - step count (dashed: skip threshold)",
                      transform=ax_delta.transAxes, ha="center", va="bottom")
        self.live_title = ax_amp.text(0.5, 1.04, "", transform=ax_amp.transAxes,
                                      ha="center", va="bottom", fontsize=9)
        self.ax_delta = ax_delta
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, parent)
        self.live_canvas.get_tk_widget().pack(side="top", fill="x")
        self.live_canvas.get_tk_widget().bind(
            "<Configure>", lambda e: self.root.after_idle(self.layout_live), add="+")

    def layout_live(self):
        ConstrainedLayoutEngine().execute(self.live_fig)
        self.live_canvas.draw_idle()

    # ---- Arduino ----
    def send(self, cmd):
        with self.send_lock:   # a 'G' line must not be split by a heartbeat
            self.ser.write(cmd.encode())

    def heartbeat(self):
        while self.running:
            self.send("H")
            time.sleep(HEARTBEAT_S)

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
                self.track(now, um / 1000)
                log = self.pos_log
                if log is not None:
                    log.append((now, us, um))
            elif ARDUINO_LINE.fullmatch(line):
                self.track(now, float(line.partition(":")[2]))
            elif ":" in line or not line.isprintable():
                FrameErrors.arduino += 1
            elif line.startswith("READY"):
                self.homing = False
                self.plan = None   # re-anchor the open loop on the step count after homing
            else:
                self.message = line

    def track(self, now, mm):
        prev = self.latest.get("Distance_mm")
        if prev is not None and abs(mm - prev) > 0.01:   # P lines (um) vs 2-decimal lines
            self.towards = mm < prev
        self.latest["Distance_mm"] = mm
        self.tel_hist.append((now, mm))

    def jog(self, direction):
        """Manual move by the chosen step: +1 = away from the switch."""
        if self.busy or self.homing:
            return
        base = self.target if self.target is not None else self.latest.get("Distance_mm", REF_MM[0])
        self.target = float(np.clip(base + direction * self.jog_mm.get(), REF_MM[0], REF_MM[-1]))
        self.move(self.target)

    def rehome(self):
        self.homing = True
        self.target = None
        self.plan = None
        self.tripped = False
        self.recent.clear()
        for tr in [self.tracker, *self.trusting] if self.tracker is not None else []:
            tr.particles = None   # re-spread over all references once homed
        self.send("C")

    def move(self, mm, speed=0.0):
        """Absolute move (firmware 'G'); speed 0 = the jog speed."""
        now = time.monotonic()
        x0 = self.open_loop(now)
        if x0 is None:
            x0 = self.latest.get("Distance_mm", mm)
        self.plan = (now, x0, float(np.clip(mm, *TARGET_RANGE_MM)),
                     min(speed, MAX_SWEEP_SPEED_MM) if speed else JOG_SPEED_MM,
                     SWEEP_ACCEL_MM if speed else JOG_ACCEL_MM)
        self.send(f"G{round(mm * 100)},{round(speed * 100)}\n")

    def open_loop(self, t):
        """Position from the commands alone (None before the first move after homing)."""
        if self.plan is None:
            return None
        t0, x0, x1, v, a = self.plan
        return trapezoid(t - t0, x0, x1, v, a)

    def wait_at(self, mm, settle=SETTLE_S, timeout=MOVE_TIMEOUT_S):
        deadline = time.time() + timeout
        settled_since = None
        while time.time() < deadline:
            if self.abort:
                raise ScanAborted
            if abs(self.latest.get("Distance_mm", -1) - mm) < ARRIVE_TOLERANCE_MM:
                settled_since = settled_since or time.time()
                if time.time() - settled_since >= settle:
                    return
            else:
                settled_since = None
            time.sleep(0.02)
        raise ScanAborted(f"Timed out moving to {mm:.2f} mm")

    # ---- Radar ----
    def radar_loop(self):
        try:
            while self.running:
                self.client.start_session()
                self.radar_restart = False
                ref = None
                while self.running and not self.radar_restart:
                    result = self.client.get_next()
                    now = time.monotonic()
                    self.session_started.set()
                    ref = ref or (result.tick_time, now)
                    lag = (now - ref[1]) - (result.tick_time - ref[0])
                    if abs(lag) > MAX_LAG_S and self.recording is None and self.frame_log is None:
                        self.resyncs += 1
                        break
                    self.radar_frames += 1
                    self.live.append((now, result.frame.mean(axis=0)))
                    log = self.frame_log
                    if log is not None:
                        log.append((now, result.tick_time, result.frame.mean(axis=0)))
                    rec = self.recording
                    if rec is not None:
                        rec.append(result.frame)
                        if len(rec) >= FRAMES_PER_CAPTURE:
                            self.record_done.set()
                self.client.stop_session()
        except Exception as e:
            self.message = f"Radar stopped: {e}"

    def capture(self):
        self.record_done.clear()
        self.recording = []
        while not self.record_done.wait(0.1):
            if self.abort:
                raise ScanAborted
        frames, self.recording = self.recording, None
        return np.array(frames[:FRAMES_PER_CAPTURE])

    def fresh_session(self):
        """Restart the radar session so a sweep starts with no frame backlog or drift."""
        self.session_started.clear()
        self.radar_restart = True
        while not self.session_started.wait(0.1):
            if self.abort:
                raise ScanAborted

    # ---- Jobs ----
    def folder(self):
        return SAVE_DIR / self.test.get().strip()

    def refresh_recordings(self):
        """Rebuild the Recordings list when the test's sweep files change."""
        folder = self.folder()
        names = sorted(f.stem for f in folder.glob("sweep_*.npz"))
        if (folder, names) == self.rec_key:
            return
        self.rec_key = (folder, names)
        for w in self.rec_frame.winfo_children():
            w.destroy()
        for i, n in enumerate(names):
            var = self.rec_use.setdefault((folder, n), tk.BooleanVar(value=True))
            tk.Checkbutton(self.rec_frame, text=n, variable=var).grid(row=i, column=0, sticky="w")
            tk.Button(self.rec_frame, text="Delete", padx=4, pady=0,
                      command=lambda n=n: self.delete_recording(n)).grid(row=i, column=1)
        if not names:
            tk.Label(self.rec_frame, text="No sweeps recorded for this test", fg="#666").grid(
                row=0, column=0, sticky="w")

    def selected_recordings(self):
        folder, names = self.rec_key
        return [n for n in names if self.rec_use[(folder, n)].get()]

    def delete_recording(self, name):
        folder = self.folder()
        if not messagebox.askyesno("Delete recording",
                                   f"Move {name} to {folder / 'deleted'}?", parent=self.root):
            return
        (folder / "deleted").mkdir(exist_ok=True)
        (folder / f"{name}.npz").replace(folder / "deleted" / f"{name}.npz")
        self.refresh_recordings()

    def speed_list(self):
        speeds = [float(s) for s in self.speeds.get().replace(",", " ").split()]
        bad = [s for s in speeds if not 0 < s <= MAX_SWEEP_SPEED_MM]
        if bad or not speeds:
            raise ValueError(f"Speeds must be between 0 and {MAX_SWEEP_SPEED_MM:g} mm/s")
        return speeds

    def run_job(self, job, args=()):
        if self.busy:
            return
        self.busy = True
        self.abort = False
        threading.Thread(target=self.job_wrapper, args=(job, args), daemon=True).start()

    def job_wrapper(self, job, args):
        try:
            job(*args)
            self.status = "Done"
        except ScanAborted as e:
            self.status = f"Stopped{': ' + str(e) if str(e) else ''}"
        except Exception as e:
            traceback.print_exc()
            self.status = f"Failed: {e}"
        finally:
            self.recording = self.frame_log = self.pos_log = None
            self.target = None
            self.busy = False

    def stepped_job(self):
        folder = self.folder()
        self.status = "Stepped: approaching the first reference"
        self.move(REF_MM[0] - APPROACH_MM)
        self.wait_at(REF_MM[0] - APPROACH_MM, settle=0)
        frames, measured = [], []
        for i, mm in enumerate(REF_MM):
            self.status = f"Stepped: reference {i + 1}/{len(REF_MM)} at {mm:.2f} mm"
            self.move(mm)
            self.wait_at(mm)
            measured.append(self.latest["Distance_mm"])
            frames.append(self.capture())
        folder.mkdir(parents=True, exist_ok=True)
        save_npz(folder / "stepped.npz", positions_mm=REF_MM, measured_mm=np.array(measured),
                 frames=np.array(frames), distances_m=DISTANCES_M, time=time.time())

    def sweep_job(self, direction, speed):
        folder = self.folder()
        runup = speed ** 2 / (2 * SWEEP_ACCEL_MM) + RUNUP_MARGIN_MM
        ends = (REF_MM[0] - runup, REF_MM[-1] + runup)
        start, end = np.clip(ends if direction == "out" else ends[::-1], *TARGET_RANGE_MM)
        label = f"Sweep {direction} at {speed:g} mm/s"
        self.status = f"{label}: going to the start"
        self.move(start)
        self.wait_at(start)
        self.fresh_session()
        self.pos_log, self.frame_log = [], []
        self.status = label
        self.move(end, speed)
        self.wait_at(end, settle=TAIL_S, timeout=abs(end - start) / speed + MOVE_TIMEOUT_S)
        frames, positions = self.frame_log, self.pos_log
        self.frame_log = self.pos_log = None
        pc, tick, profiles = zip(*frames)
        pos_pc, us, um = np.array(positions).T
        folder.mkdir(parents=True, exist_ok=True)
        save_npz(folder / f"sweep_{direction}_{speed:g}.npz", speed_mm_s=speed,
                 direction=direction, frame_pc=np.array(pc), frame_tick=np.array(tick),
                 profiles=np.array(profiles), pos_pc=pos_pc, pos_us=us.astype(np.int64),
                 pos_um=um.astype(np.int64), distances_m=DISTANCES_M, time=time.time())

    def sweeps_job(self):
        """Each speed there and back, starting from the nearer end (so every pair ends
        where it started and the next speed needs no repositioning)."""
        speeds = self.speed_list()
        far = self.latest.get("Distance_mm", 0) > (REF_MM[0] + REF_MM[-1]) / 2
        order = ("back", "out") if far else ("out", "back")
        for speed in speeds:
            for direction in order:
                self.sweep_job(direction, speed)

    def full_job(self):
        self.speed_list()                  # check the speeds before the long stepped sweep
        self.stepped_job()
        self.sweeps_job()

    def stop(self):
        self.abort = True
        self.target = None
        x = self.open_loop(time.monotonic())
        if x is not None:
            self.plan = (0.0, x, x, 1.0, 1.0)
        self.send("S")

    # ---- Analysis ----
    def analyse(self):
        folder = self.folder()
        self.refresh_recordings()
        names = self.selected_recordings()
        if not names:
            self.message = "Analysis: tick at least one recording"
            return
        try:
            sweeps, model = analyse(folder, names, self.lo.get(), self.hi.get(),
                                    self.magnitude.get(), self.skip_mm.get())
        except Exception as e:
            traceback.print_exc()
            self.message = f"Analysis failed: {e}"
            return
        text = json.dumps(model, indent=1)
        (folder / "sync_model.json").write_text(text)
        (SAVE_DIR / "sync_model.json").write_text(text)
        self.message = ""
        self.draw(sweeps, model)

    def draw(self, sweeps, model):
        fig = self.fig
        fig.clear()
        ax_err, ax_bias, ax_corr, ax_v = fig.subplots(2, 2).ravel()
        for i, s in enumerate(sweeps):
            c = SWEEP_COLOURS[i % len(SWEEP_COLOURS)]
            ok = s["steady"]
            ax_err.plot(s["x_tel"][ok], s["err"][ok], ".", ms=2, color=c, label=s["name"])
            ax_corr.plot(s["x_tel"][ok], s["corrected"][ok], ".", ms=2, color=c, alpha=0.4)
            ax_corr.plot(s["x_tel"][s["picks"]], s["corrected"][s["picks"]], "o", ms=4,
                         color=c, mec="k", mew=0.5)
            ax_bias.plot(np.median(s["v"][ok]), np.median(s["err"][ok]), "o", color=c)
            ax_v.plot(s["x_tel"], s["v"], "-", lw=0.8, color=c)
        vmax = max(s["speed"] for s in sweeps) * 1.15
        v = np.linspace(-vmax, vmax, 50)
        for towards, ls, text in ((False, "-", "model, moving out"), (True, "--", "model, moving back")):
            ax_bias.plot(v, model["offset_mm"] + v * model["latency_ms"] / 1000
                         + model["backlash_mm"] * towards, "k", ls=ls, lw=0.8, label=text)
        ax_err.set(title="Telemetry - radar position (constant-speed frames)",
                   xlabel="Telemetry position (mm)", ylabel="Error (mm)")
        ax_err.legend(fontsize=7, markerscale=4)
        ax_bias.set(title="Median error per sweep vs speed", xlabel="Speed (mm/s, + = out)",
                    ylabel="Error (mm)")
        ax_bias.axhline(0, color="#999", lw=0.5)
        ax_bias.axvline(0, color="#999", lw=0.5)
        ax_bias.legend(fontsize=7)
        ax_corr.set(title="After correction (circles: one frame per reference)",
                    xlabel="Telemetry position (mm)", ylabel="Residual (mm)")
        ax_corr.axhline(0, color="#999", lw=0.5)
        ax_v.set(title="Carriage speed at each frame (from telemetry)",
                 xlabel="Telemetry position (mm)", ylabel="Speed (mm/s)")
        for ax in (ax_err, ax_corr, ax_v):
            for mm in REF_MM[::SUBDIVISIONS]:
                ax.axvline(mm, color="#ddd", lw=0.5, zorder=0)
        self.canvas.draw_idle()

        lines = [f"Latency  {model['latency_ms']:8.1f} ms",
                 f"Backlash {model['backlash_mm']:8.3f} mm",
                 f"Offset   {model['offset_mm']:8.3f} mm",
                 f"RMS {model['rms_before_mm']:.3f} -> {model['rms_after_mm']:.3f} mm "
                 f"({model['frames']} frames)"]
        if model["note"]:
            lines.append(f"Note: {model['note']}")
        for s in model["sweeps"]:
            lines.append(f"{s['name']:>14}: mean {s['mean_error_mm']:+.3f}, "
                         f"rms after {s['rms_after_mm']:.3f}, picks {s['rms_picks_mm']:.3f}, "
                         f"beyond threshold {s['frames_beyond_skip_threshold']}")
        self.lbl_result.config(text="\n".join(lines))

    # ---- Live row + skip detector ----
    def live_tick(self):
        if not self.running:
            return
        try:
            self.live_step()
        except Exception as e:   # keep the loop alive, e.g. on a half-typed range entry
            self.message = f"Live display: {e}"
        self.root.after(LIVE_MS, self.live_tick)

    def load_live_inputs(self):
        """(Re)load this test's stepped references and sync model when they change."""
        f = self.folder() / "stepped.npz"
        key = (f, f.stat().st_mtime_ns if f.exists() else None,
               self.lo.get(), self.hi.get(), self.magnitude.get())
        if key != self.refs_key:
            self.refs_key = key
            self.refs = self.tracker = None
            if key[1] is not None:
                self.ref_mm, self.refs, self.prep, scale = load_references(f.parent, *key[2:])
                self.tracker = ParticleFilter(self.ref_mm, self.refs, scale)
                self.trusting = [ParticleFilter(self.ref_mm, self.refs, scale, k) for k in TRUST_LEVELS]
                c = mask_column(self.tracker.x[[0, -1]]) + 1.5   # window centre
                self.meas_img.set_extent((c[0], c[1], 0, 1))
                self.prob_img.set_extent((c[0], c[1], 0, 1))
                amp = np.abs(self.refs).mean(axis=1)
                self.amp_line.set_data(mask_column(self.ref_mm) + 1.5, amp)
                pad = 0.05 * (amp.max() - amp.min())
                self.ax_amp.set_ylim(amp.min() - pad, amp.max() + pad)
            else:
                self.amp_line.set_data([], [])
            self.recent.clear()
        f = self.folder() / "sync_model.json"
        key = (f, f.stat().st_mtime_ns if f.exists() else None)
        if key != self.model_key:
            self.model_key = key
            self.model = json.loads(f.read_text()) if key[1] is not None else {}

    def live_step(self):
        self.load_live_inputs()
        new = [f for f in list(self.live) if f[0] > self.live_seen]
        if new:
            self.live_seen = new[-1][0]
        if new and self.refs is not None:
            self.amp_now.set_ydata([np.abs(self.prep(new[-1][1])).mean()] * 2)
            self.amp_now.set_visible(True)
        if new and self.refs is not None and len(self.tel_hist) > 1 and not self.homing:
            t = np.array([f[0] for f in new])
            x_rad, inside, _ = radar_positions(self.prep(np.array([f[1] for f in new])),
                                               self.refs, self.ref_mm)   # single-frame, for comparison
            hist = np.array(self.tel_hist)
            m = self.model
            x_tel = (np.interp(t - m.get("latency_ms", 0) / 1000, hist[:, 0], hist[:, 1])
                     - m.get("offset_mm", 0) - m.get("backlash_mm", 0) * self.towards)
            raw = np.where(inside, x_rad - x_tel, np.nan)
            for ti, profile, xt, r in zip(t, self.prep(np.array([f[1] for f in new])), x_tel, raw):
                dt = min(ti - self.last_frame_t, MAX_DT_S) if self.last_frame_t else 0.0
                self.last_frame_t = ti
                for tr in [self.tracker, *self.trusting]:
                    tr.predict(dt, xt)
                if not self.ref_mm[0] <= xt <= self.ref_mm[-1]:
                    continue    # off the references: predict only (the step count carries it)
                for tr in [self.tracker, *self.trusting]:
                    tr.update(profile, xt)
                self.x_radar, self.x_sd = self.tracker.estimate(xt)
                self.recent.append(self.x_radar - xt)
                ol = self.open_loop(ti - m.get("latency_ms", 0) / 1000)   # same timing and
                if ol is not None:                                         # corrections as xt
                    ol -= m.get("offset_mm", 0) + m.get("backlash_mm", 0) * self.towards
                self.deltas.append((ti, self.x_radar - xt, r, np.nan if ol is None else ol - xt,
                                    *[tr.estimate(xt)[0] - xt for tr in self.trusting]))
            self.check_skip()
        if time.monotonic() - self.live_drawn > LIVE_DRAW_S:
            self.live_drawn = time.monotonic()
            self.draw_live()

    def check_skip(self):
        if len(self.recent) < SKIP_FRAMES:
            return
        delta, thr = np.median(self.recent), self.skip_mm.get()
        if abs(delta) > thr and not self.tripped:
            self.tripped = True
            self.skips += 1
            self.message = (f"Skip detected: radar is {delta:+.2f} mm from the step count. "
                            "Re-home before trusting positions.")
            if self.stop_on_skip.get():
                self.stop()
        elif abs(delta) < thr / 2:
            self.tripped = False

    def draw_live(self):
        mm = self.latest.get("Distance_mm")
        title = []
        if mm is not None:
            self.rect_tel.set(x=mask_column(mm) - 0.5, visible=True)
            title.append(f"step count {mm:.2f} mm (dashed)")
        if self.x_radar is not None:
            self.rect_radar.set(x=mask_column(self.x_radar) - 0.5, visible=True)
            title.insert(0, f"radar {self.x_radar:.2f} \u00b1 {self.x_sd:.2f} mm, "
                            f"mask {mask_column(self.x_radar):.2f}")
        tr = self.tracker
        if tr is not None and tr.cost is not None:
            diff = np.sqrt(np.maximum(tr.cost, 0) / tr.refs.shape[1])   # RMS frame - ref
            self.meas_img.set_data(diff[None])
            self.meas_img.set_clim(diff.min(), np.percentile(diff, 90))
            pos = tr.last + tr.particles[:, 0]
            edges = np.arange(tr.x[0], tr.x[-1] + POST_BIN_MM, POST_BIN_MM)
            post = np.histogram(np.clip(pos, edges[0], edges[-1]), edges, weights=tr.weights)[0]
            self.prob_img.set_data(np.maximum(np.log10(post / post.max() + 1e-12), LOG_P_FLOOR)[None])
            self.particle_marks.set_data(mask_column(pos) + 1.5, np.full(len(pos), 0.5))
        self.live_title.set_text(" · ".join(title) or "No position yet")
        thr = self.skip_mm.get()
        now = time.monotonic()
        d = np.array([x for x in self.deltas if x[0] > now - HISTORY_S]).reshape(-1, 4 + len(TRUST_LEVELS))
        for i, line in enumerate(self.trust_lines):
            line.set_data(d[:, 0] - now, d[:, 4 + i])
        self.delta_line.set_data(d[:, 0] - now, d[:, 1])
        self.open_line.set_data(d[:, 0] - now, d[:, 3])
        self.raw_dots.set_data(d[:, 0] - now, d[:, 2])
        for line, sign in zip(self.thr_lines, (1, -1)):
            line.set_ydata([sign * thr] * 2)
        top = max(1.5 * thr, 1.1 * np.abs(d[:, 1]).max()) if len(d) else 1.5 * thr
        top = min(top, 10.0)   # wild single-frame matches would squash the filtered line
        self.ax_delta.set_ylim(-top, top)
        self.live_canvas.draw_idle()

    # ---- UI ----
    def tick(self):
        if not self.running:
            return
        mm = self.latest.get("Distance_mm")
        self.lbl_pos.config(text=f"{mm:.3f} mm" if mm is not None else "-- mm")
        self.lbl_status.config(text=self.status)
        self.lbl_msg.config(text=self.message)
        self.refresh_recordings()
        if self.refs is None:
            skip_text, colour = "Off: this test has no stepped sweep yet", "#666"
        else:
            delta = np.median(self.recent) if self.recent else 0.0
            skip_text = (f"Radar - step count {delta:+.2f} mm (\u00b1 {self.x_sd:.2f}) · "
                         f"skips detected: {self.skips}")
            colour = "#b00" if self.tripped else "#060"
        self.lbl_skip.config(text=skip_text, fg=colour)
        self.lbl_radar.config(text=f"Radar frames {self.radar_frames} · resyncs {self.resyncs}"
                                   f" · frame errors {FrameErrors.radar}/{FrameErrors.arduino}")
        for b in self.buttons:
            b.config(state="disabled" if self.busy else "normal")
        self.root.after(UI_MS, self.tick)

    def close(self):
        self.settings.save()
        self.running = False
        self.send("S")
        self.root.after(200, self.root.destroy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arduino", help="Arduino serial port, e.g. COM12")
    ap.add_argument("--radar", help="Radar serial port, e.g. COM11 (default: auto-detect)")
    ap.add_argument("--mock", action="store_true", help="use simulated radar data")
    args = ap.parse_args()
    try:
        if args.mock:
            client = a121.Client.open(mock=True)
        else:
            client = a121.Client.open(serial_port=args.radar or find_port("Radar", RADAR_KEYWORDS),
                                      override_baudrate=RADAR_BAUD)
        client.setup_session(SENSOR_CONFIG)
        ser = serial.Serial(args.arduino or find_port("Arduino", ARDUINO_KEYWORDS), BAUD,
                            timeout=0.1)
    except (RuntimeError, serial.SerialException) as e:
        sys.exit(str(e))
    root = tk.Tk()
    RailTestApp(root, ser, client)
    root.mainloop()


if __name__ == "__main__":
    main()
