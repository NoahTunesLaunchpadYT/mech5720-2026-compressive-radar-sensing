"""
Single-mask capture: hold the carriage at one mask and capture the corner reflector
at every scene position 00..88 (plus an ambient capture with no reflector).

  Mask 0..23 buttons -> choose the working mask; the carriage moves to it
  Scene 00..88 buttons -> where the reflector is now; "Ambient" = no reflector
  C / Enter        -> capture FRAMES_PER_CAPTURE frames for the selected scene
  "Advance after capture" selects the next scene (00, 01, .. 08, 10, ..) once saved.
  Space            -> stop the stepper
  "Subtract ambient" -> subtracts this mask's complex ambient from the live plot and
                      the beam pattern, then shows the magnitude. Display only.
  Beam pattern     -> sums |capture| over the chosen range for each scene and plots it
                      as a 3D surface over the 9x9 grid (drag to orbit).
  "Export beam heatmaps" -> writes every mask's beam pattern (current range and
                      subtraction setting) to mask_captures/exports/beam_<time>/: grid.png
                      (all 24 on one shared scale), mask00..23.png (each on its own
                      scale) and beam_patterns.npz (24 x 9 x 9 array).

Captures are saved to mask_captures/mask<NN>/scene<rc>.npz and
mask_captures/mask<NN>/ambient.npz (overwriting any previous capture). Each holds
the same fields as radar_capture.py's captures, plus scene.

Radar settings, ports and mask positions come from radar_capture.py.
  python mask_capture.py [--arduino COM12] [--radar COM11] [--mock]
"""

import argparse
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

from radar_capture import (
    a121, ARDUINO_LINE, ARDUINO_KEYWORDS, BAUD, DISTANCES_M, FRAMES_PER_CAPTURE,
    FrameErrors, HEARTBEAT_S, MASK, MASK_OFF, MASK_ON, MAX_LAG_S, NUM_POSITIONS,
    PLOT_MS, RADAR_BAUD, RADAR_KEYWORDS, SCENE_FULL, SCENE_NONE, SCENES, SENSOR_CONFIG,
    UI_MS, export_beams, find_port, mask_index, position_from_mm, position_of_mask,
    save_npz,
)

SAVE_DIR = Path(__file__).parent / "mask_captures"
AMBIENT = "ambient"


def capture_file(m, scene):
    name = AMBIENT if scene == AMBIENT else f"scene{scene}"
    return SAVE_DIR / f"mask{m:02d}" / f"{name}.npz"


def load_mean(f):
    """Complex mean over frames and sweeps of one capture file."""
    with np.load(f) as d:
        return d["frames"].mean(axis=(0, 1))


def beam_pattern(m, lo, hi, subtract):
    """9x9 per-scene sum over lo..hi mm of |capture - ambient| (NaN = not captured).
    Returns (pattern, subtracted): no subtraction when the mask has no ambient."""
    in_range = (DISTANCES_M * 1000 >= lo) & (DISTANCES_M * 1000 <= hi)
    f = capture_file(m, AMBIENT)
    subtracted = subtract and f.exists()
    ambient = load_mean(f) if subtracted else 0
    pattern = np.full((9, 9), np.nan)
    for s in SCENES:
        f = capture_file(m, s)
        if f.exists():
            pattern[int(s[0]), int(s[1])] = np.abs((load_mean(f) - ambient)[in_range]).sum()
    return pattern, subtracted


class MaskCaptureApp:
    def __init__(self, root, ser, client, open_arduino, open_radar):
        self.root = root
        self.ser = ser
        self.client = client
        self.open_arduino = open_arduino
        self.open_radar = open_radar
        self.latest = {}
        self.message = "Homing..."
        self.status = ""
        self.busy = False
        self.running = True
        self.radar_restart = False
        self.profile = None
        self.profile_dirty = False
        self.ymax = 1.0
        self.recording = None
        self.record_done = threading.Event()
        self.mask = tk.IntVar(value=0)
        self.scene = tk.StringVar(value=SCENES[0])
        self.advance = tk.BooleanVar(value=True)
        self.subtract = tk.BooleanVar(value=False)
        self.ambient = None        # complex ambient of the working mask
        self.beam_dirty = True
        SAVE_DIR.mkdir(exist_ok=True)

        root.title("Single-Mask Capture")
        root.geometry("1250x780")

        # ---- Left: mask, scenes, controls ----
        left = tk.Frame(root)
        left.pack(side="left", fill="y", padx=10, pady=6)
        self.lbl_mask = tk.Label(left, text="", font=("Helvetica", 22, "bold"))
        self.lbl_dist = tk.Label(left, text="-- mm", font=("Helvetica", 12))
        self.lbl_mask.pack()
        self.lbl_dist.pack()

        mrow = tk.Frame(left)
        mrow.pack(pady=4)
        self.cell = 28
        self.mask_canvas = tk.Canvas(mrow, width=4 * self.cell, height=4 * self.cell,
                                     highlightthickness=1, highlightbackground="#888")
        self.mask_canvas.pack(side="left", padx=(0, 10))
        grid = tk.Frame(mrow)
        grid.pack(side="left")
        self.mask_buttons = []
        for m in range(NUM_POSITIONS):
            b = tk.Radiobutton(grid, text=str(m), variable=self.mask, value=m,
                               indicatoron=False, width=3, offrelief="raised",
                               selectcolor=MASK_ON, command=self.select_mask)
            b.grid(row=m // 6, column=m % 6, padx=1, pady=1)
            self.mask_buttons.append(b)

        self.lbl_status = tk.Label(left, text="", font=("Helvetica", 12))
        self.lbl_msg = tk.Label(left, text="", font=("Helvetica", 10), fg="#b00", wraplength=360)
        self.lbl_status.pack()
        self.lbl_msg.pack()

        actions = tk.Frame(left)
        actions.pack(pady=3)
        self.btn_capture = tk.Button(actions, text=f"Capture {FRAMES_PER_CAPTURE}", width=18,
                                     command=self.capture)
        btn_stop = tk.Button(actions, text="STOP", width=18, fg="#b00", command=self.stop)
        self.btn_radar = tk.Button(actions, text="Reconnect radar", width=18,
                                   command=self.reconnect_radar)
        self.btn_arduino = tk.Button(actions, text="Reconnect Arduino", width=18,
                                     command=self.reconnect_arduino)
        btn_refresh = tk.Button(actions, text="Refresh plots", width=18,
                                command=self.refresh_plots)
        self.btn_export = tk.Button(actions, text="Export beam heatmaps", width=18,
                                    command=self.export)
        for i, b in enumerate((self.btn_capture, btn_stop, self.btn_radar, self.btn_arduino,
                               btn_refresh, self.btn_export)):
            b.grid(row=i // 2, column=i % 2, padx=2, pady=2)

        checks = tk.Frame(left)
        checks.pack()
        tk.Checkbutton(checks, text="Advance after capture",
                       variable=self.advance).pack(side="left")
        self.chk_subtract = tk.Checkbutton(checks, text="Subtract ambient",
                                           variable=self.subtract, command=self.on_change)
        self.chk_subtract.pack(side="left")
        tk.Label(left, text="C/Enter capture  ·  Space stop",
                 font=("Helvetica", 9), fg="#666").pack(pady=2)

        # Scene grid for the working mask: green = captured
        tk.Label(left, text="Reflector scene", font=("Helvetica", 11, "bold")).pack(pady=(6, 0))
        sgrid = tk.Frame(left)
        sgrid.pack()
        self.scene_buttons = {}
        for i, s in enumerate(SCENES + [AMBIENT]):
            b = tk.Radiobutton(sgrid, text=s, variable=self.scene, value=s, indicatoron=False,
                               width=3, font=("Helvetica", 9), offrelief="raised",
                               command=self.mark_selected_scene)
            if s == AMBIENT:   # tall button to the right of the 9x9 grid
                b.config(text="Ambient", wraplength=1, width=2)
                b.grid(row=0, column=9, rowspan=9, padx=(6, 1), pady=1, sticky="ns")
            else:
                b.grid(row=i // 9, column=i % 9, padx=1, pady=1)
            self.scene_buttons[s] = b
        self.lbl_count = tk.Label(left, text="", font=("Helvetica", 10))
        self.lbl_count.pack(pady=2)

        # ---- Right: live profile above, beam pattern below ----
        right = tk.Frame(root)
        right.pack(side="right", fill="both", expand=True)
        self.lbl_fps = tk.Label(right, text="", font=("Helvetica", 10), fg="#666")
        self.lbl_fps.pack(anchor="e", padx=10)
        fig = Figure(figsize=(7, 2.6), dpi=100)
        self.ax_line = fig.subplots()
        dist_mm = DISTANCES_M * 1000
        self.line, = self.ax_line.plot(dist_mm, np.zeros_like(dist_mm), animated=True)
        self.ax_line.set_ylim(0, self.ymax)
        self.ax_line.set_xlabel("Distance (mm)")
        self.ax_line.set_ylabel("Intensity")
        self.ax_line.set_title("Live radar: mean amplitude vs distance", fontsize=10)
        self.ax_line.grid(alpha=0.3)
        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.get_tk_widget().pack(fill="x")
        self.background = None
        self.canvas.mpl_connect("draw_event", self.on_full_draw)

        bctl = tk.Frame(right)
        bctl.pack(pady=(6, 0))
        self.beam_lo = tk.DoubleVar(value=900.0)
        self.beam_hi = tk.DoubleVar(value=1200.0)
        tk.Label(bctl, text="Beam pattern range (mm)").pack(side="left")
        for var in (self.beam_lo, self.beam_hi):
            e = tk.Entry(bctl, textvariable=var, width=6)
            e.pack(side="left", padx=2)
            e.bind("<Return>", lambda e: self.on_change())
        bfig = Figure(figsize=(7, 4.6), dpi=100)
        self.beam_canvas = FigureCanvasTkAgg(bfig, master=right)
        self.beam_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.ax_beam = bfig.add_subplot(projection="3d")   # after the canvas, so orbiting hooks up
        self.beam_cols, self.beam_rows = np.meshgrid(range(9), range(9))
        self.sequential = matplotlib.colormaps["viridis"]
        self.beam_norm = matplotlib.colors.Normalize()
        self.beam_artists = []
        self.ax_beam.set_xlabel("Scene column")
        self.ax_beam.set_ylabel("Scene row")
        self.ax_beam.set_zlabel("Intensity")
        bfig.colorbar(matplotlib.cm.ScalarMappable(self.beam_norm, self.sequential),
                      ax=self.ax_beam, shrink=0.6, pad=0.1)

        root.bind("<space>", lambda e: self.stop())
        root.bind("<Return>", lambda e: self.capture())
        root.bind("c", lambda e: self.capture())
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.reader_thread = threading.Thread(target=self.reader, daemon=True)
        self.reader_thread.start()
        threading.Thread(target=self.heartbeat, daemon=True).start()
        self.radar_thread = threading.Thread(target=self.radar_loop, daemon=True)
        self.radar_thread.start()
        self.load_mask()
        self.tick()
        self.plot_tick()

    # ---- Display ----
    def load_mask(self):
        """Refresh everything that depends on the working mask."""
        m = self.mask.get()
        f = capture_file(m, AMBIENT)
        self.ambient = load_mean(f) if f.exists() else None
        if self.ambient is None:
            self.subtract.set(False)
        self.mask_canvas.delete("all")
        c = self.cell
        for r, row in enumerate(MASK[:, m:m + 4]):
            for k, v in enumerate(row):
                self.mask_canvas.create_rectangle(k * c, r * c, (k + 1) * c, (r + 1) * c,
                                                  fill=MASK_ON if v else MASK_OFF,
                                                  outline="#444")
        self.colour_scenes()
        self.mark_selected_scene()
        self.on_change()

    def colour_scenes(self):
        m = self.mask.get()
        for s, b in self.scene_buttons.items():
            colour = SCENE_FULL if capture_file(m, s).exists() else SCENE_NONE
            b.config(bg=colour, selectcolor=colour)
        n = sum(capture_file(m, s).exists() for s in SCENES)
        self.lbl_count.config(text=f"Mask {m}: {n}/{len(SCENES)} scenes captured")

    def mark_selected_scene(self):
        for s, b in self.scene_buttons.items():
            b.config(font=("Helvetica", 9, "bold" if s == self.scene.get() else "normal"))
        self.beam_dirty = True   # moves the red marker

    def on_change(self):
        self.ymax = 0.0          # force a y-axis rescale on the next frame
        self.beam_dirty = True

    def on_full_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax_line.bbox)
        self.blit()

    def blit(self):
        self.canvas.restore_region(self.background)
        self.ax_line.draw_artist(self.line)
        self.canvas.blit(self.ax_line.bbox)

    def draw_profile(self):
        p = self.profile
        if self.subtract.get():
            p = p - self.ambient
        p = np.abs(p)
        self.line.set_ydata(p)
        peak = max(p.max(), 1.0)
        if peak > self.ymax or peak < 0.4 * self.ymax:
            self.ymax = peak * 1.25
            self.ax_line.set_ylim(0, self.ymax)
            self.canvas.draw()   # triggers on_full_draw, which blits the new data
        elif self.background:
            self.blit()

    def draw_beam(self):
        """Per scene: sum over the range bins of |capture (- ambient)|."""
        m = self.mask.get()
        lo, hi = self.beam_lo.get(), self.beam_hi.get()
        pattern, _ = beam_pattern(m, lo, hi, self.subtract.get())
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
            if scene != AMBIENT and have[int(scene[0]), int(scene[1])]:
                r, c = int(scene[0]), int(scene[1])
                self.beam_artists.append(self.ax_beam.scatter(c, r, pattern[r, c], c="red", s=40))
            self.ax_beam.set_zlim(0, self.beam_norm.vmax * 1.05)
        self.ax_beam.set_xlim(0, 8)
        self.ax_beam.set_ylim(8, 0)   # row 0 at the back, like the scene buttons
        self.ax_beam.set_title(f"Mask {m}, {lo:.0f}-{hi:.0f} mm", fontsize=10)
        self.beam_canvas.draw()

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
        self.message = ""
        self.canvas.draw()

    def plot_step(self):
        if self.beam_dirty:
            self.beam_dirty = False
            self.draw_beam()
        if self.profile_dirty:
            self.profile_dirty = False
            self.draw_profile()

    def tick(self):
        if not self.running:
            return
        if "Distance_mm" in self.latest:
            mm = self.latest["Distance_mm"]
            pos = position_from_mm(mm)
            at = f"at mask {mask_index(pos)}" if 1 <= pos <= NUM_POSITIONS else "between masks"
            self.lbl_dist.config(text=f"Carriage {at}  ·  {mm:.2f} mm")
        self.lbl_mask.config(text=f"Mask {self.mask.get()}  ·  "
                                  f"{'Ambient' if self.scene.get() == AMBIENT else 'Scene ' + self.scene.get()}")
        self.lbl_status.config(text=self.status)
        self.lbl_msg.config(text=self.message)
        state = "disabled" if self.busy else "normal"
        for b in (self.btn_capture, self.btn_radar, self.btn_arduino, self.btn_export,
                  *self.mask_buttons):
            b.config(state=state)
        self.chk_subtract.config(state="normal" if self.ambient is not None else "disabled")
        self.root.after(UI_MS, self.tick)

    # ---- Arduino ----
    def send(self, cmd):
        try:
            self.ser.write(cmd.encode())
        except serial.SerialException:
            self.message = "Arduino connection lost"

    def select_mask(self):
        self.send(chr(ord("a") + position_of_mask(self.mask.get()) - 1))
        self.load_mask()

    def stop(self):
        self.send("S")

    def heartbeat(self):
        while self.running:
            self.send("H")
            time.sleep(HEARTBEAT_S)

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
                FrameErrors.arduino += 1
            elif line.startswith("READY"):
                self.message = ""
            else:
                self.message = line

    # ---- Radar ----
    def radar_loop(self):
        """Streams continuously: feeds the live plot and any capture in progress."""
        self.client.start_session()
        ref = None   # (radar tick_time, wall time) at the first frame of the session
        while self.running and not self.radar_restart:
            try:
                result = self.client.get_next()
                # Restart the session between captures when radar time lags the PC
                now = time.monotonic()
                ref = ref or (result.tick_time, now)
                if (abs((now - ref[1]) - (result.tick_time - ref[0])) > MAX_LAG_S
                        and self.recording is None):
                    self.client.stop_session()
                    self.client.start_session()
                    ref = None
                    continue
            except Exception as e:
                self.message = f"Radar stopped: {e}"
                return
            frame = result.frame
            self.profile = frame.mean(axis=0)
            self.profile_dirty = True
            rec = self.recording
            if rec is not None:
                rec.append(frame)
                if len(rec) == FRAMES_PER_CAPTURE:
                    self.record_done.set()

    # ---- Jobs ----
    def run_job(self, job):
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self.job_wrapper, args=(job,), daemon=True).start()

    def job_wrapper(self, job):
        try:
            job()
        except Exception as e:
            self.message = f"Failed: {e}"
            self.status = ""
        finally:
            self.busy = False

    def capture(self):
        self.run_job(self.capture_job)

    def capture_job(self):
        m, scene = self.mask.get(), self.scene.get()
        mm = self.latest.get("Distance_mm", float("nan"))
        pos = position_from_mm(mm)
        if not 1 <= pos <= NUM_POSITIONS or mask_index(pos) != m:
            self.message = f"Carriage isn't at mask {m} - wait for it to arrive"
            return
        errors_before = FrameErrors.radar
        self.record_done.clear()
        self.recording = []
        while not self.record_done.wait(0.1):
            self.status = f"{scene}: frame {len(self.recording)}/{FRAMES_PER_CAPTURE}"
            if not self.radar_thread.is_alive():
                self.recording = None
                raise RuntimeError("radar stopped")
        frames = np.stack(self.recording[:FRAMES_PER_CAPTURE])
        self.recording = None
        f = capture_file(m, scene)
        f.parent.mkdir(parents=True, exist_ok=True)
        save_npz(
            f,
            frames=frames,
            distances_m=DISTANCES_M,
            distance_mm=mm,
            position=pos,
            mask_index=m,
            scene=scene,
            frame_errors=FrameErrors.radar - errors_before,
            mask=MASK[:, m:m + 4],
            sensor_config=self.client.session_config.sensor_config.to_json(),
            server_info=self.client.server_info.to_json(),
        )
        self.status = f"Saved {f.parent.name}/{f.name}"
        self.message = ""
        self.root.after(0, self.after_capture)

    def after_capture(self):
        """Tk-thread follow-up: reload ambient, recolour, and step to the next scene."""
        scene = self.scene.get()
        if scene == AMBIENT:
            self.ambient = load_mean(capture_file(self.mask.get(), AMBIENT))
        elif self.advance.get() and scene != SCENES[-1]:
            self.scene.set(SCENES[SCENES.index(scene) + 1])
        self.colour_scenes()
        self.mark_selected_scene()
        self.on_change()

    def export(self):
        # Read the settings here (Tk thread); the export itself runs as a job
        lo, hi, subtract = self.beam_lo.get(), self.beam_hi.get(), self.subtract.get()

        def job():
            self.status = "Exporting beam heatmaps..."
            results = [beam_pattern(m, lo, hi, subtract) for m in range(NUM_POSITIONS)]
            out = export_beams(results, lo, hi, subtract, SAVE_DIR / "exports")
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
        self.ser = self.open_arduino()    # opening the port resets it, so it re-homes
        self.reader_thread = threading.Thread(target=self.reader, daemon=True)
        self.reader_thread.start()
        self.message = "Homing..."
        self.status = "Arduino reconnected"

    def close(self):
        self.running = False
        self.send("S")
        self.ser.close()
        self.radar_thread.join(timeout=1)
        self.client.close()
        self.root.destroy()


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
    MaskCaptureApp(root, ser, client, open_arduino, open_radar)
    root.mainloop()


if __name__ == "__main__":
    main()
