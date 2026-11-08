"""
Arrow-key step control for the stepper sketch.

  Left / Right arrow press -> queue one 7 mm move; holding never repeats
  Space                    -> stop and clear all queued moves
  The window also shows distance from the limit switch, the voltage on A5,
  and homing / emergency-stop messages from the Arduino.

Setup:
  pip install pyserial
  python main.py            (auto-detects the Arduino)
  python main.py COM3       (or name the port: COM3, /dev/ttyACM0, /dev/cu.usbmodem...)

Notes:
  - Close the Arduino Serial Monitor / Plotter first; only one program can use the port.
  - Opening the port resets the Arduino, so it re-homes when this app starts.
  - The window must have focus for the keys to work.
"""

import sys
import threading

import serial
import serial.tools.list_ports
import tkinter as tk

BAUD = 115200
SEND_INTERVAL_MS = 100   # keep-alive rate; the Arduino stops after 300 ms of silence

# Forward = towards the limit switch. Flip this if the arrows feel backwards.
LEFT_IS_FORWARD = True


def find_port():
    if len(sys.argv) > 1:
        return sys.argv[1]
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        text = f"{p.device} {p.description} {p.manufacturer}".lower()
        if any(k in text for k in ("arduino", "ch340", "usbmodem", "ttyacm", "wch")):
            return p.device
    if len(ports) == 1:
        return ports[0].device
    names = ", ".join(p.device for p in ports) or "none found"
    sys.exit(f"Couldn't identify the Arduino. Run: python main.py <port>\nPorts: {names}")


class JogApp:
    def __init__(self, root, ser):
        self.root = root
        self.ser = ser
        self.held = set()          # arrow keys currently held
        self.release_jobs = {}     # pending key releases (filters OS key auto-repeat)
        self.latest = {}           # latest values from the Arduino
        self.message = "Connecting..."
        self.running = True

        root.title("Stepper Steps")
        root.geometry("360x260")
        font_big = ("Helvetica", 58, "bold")
        font_mid = ("Helvetica", 36)

        self.lbl_state = tk.Label(root, text="7 mm / press", font=font_mid)
        self.lbl_dist = tk.Label(root, text="-- mm", font=font_big)
        self.lbl_volt = tk.Label(root, text="-- V", font=font_mid)
        self.lbl_msg = tk.Label(root, text="", font=("Helvetica", 12), fg="#b00")
        self.lbl_help = tk.Label(root, text="Tap ← / →: 7 mm · Space: stop + clear",
                                 font=("Helvetica", 11), fg="#666")
        for w in (self.lbl_state, self.lbl_dist, self.lbl_volt, self.lbl_msg, self.lbl_help):
            w.pack(pady=4)

        root.bind("<KeyPress-Left>", self.on_press)
        root.bind("<KeyPress-Right>", self.on_press)
        root.bind("<KeyRelease-Left>", self.on_release)
        root.bind("<KeyRelease-Right>", self.on_release)
        root.bind("<space>", self.stop_all)
        root.bind("<FocusOut>", self.stop_all)  # never leave it running if focus is lost
        root.protocol("WM_DELETE_WINDOW", self.close)

        threading.Thread(target=self.reader, daemon=True).start()
        self.tick()

    # ---- Keys ----
    def on_press(self, event):
        key = event.keysym
        job = self.release_jobs.pop(key, None)
        if job:
            self.root.after_cancel(job)
        if key not in self.held:
            self.held.add(key)
            forward = (key == "Left") == LEFT_IS_FORWARD
            self.send_command("F" if forward else "B")  # one command per distinct press

    def on_release(self, event):
        # Linux sends release+press pairs while a key auto-repeats; wait briefly
        # so those don't register as real releases.
        key = event.keysym
        self.release_jobs[key] = self.root.after(40, self.finish_release, key)

    def finish_release(self, key):
        self.release_jobs.pop(key, None)
        self.held.discard(key)

    def stop_all(self, event=None):
        # Keep held keys latched so auto-repeat cannot restart motion after Space.
        self.send_command("S")

    # ---- Serial out ----
    def send_command(self, cmd):
        try:
            self.ser.write(cmd.encode())
        except serial.SerialException:
            self.message = "Serial connection lost"

    def tick(self):
        if not self.running:
            return
        self.send_command("H")  # keep-alive; never queues movement
        if "Distance_mm" in self.latest:
            self.lbl_dist.config(text=f"{self.latest['Distance_mm']:.2f} mm")
        if "Voltage" in self.latest:
            self.lbl_volt.config(text=f"{self.latest['Voltage']:.2f} V")
        self.lbl_msg.config(text=self.message)
        self.root.after(SEND_INTERVAL_MS, self.tick)

    # ---- Serial in ----
    def reader(self):
        while self.running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
            except serial.SerialException:
                self.message = "Serial connection lost"
                return
            if not line:
                continue
            if ":" in line:
                for part in line.split(","):
                    name, _, value = part.partition(":")
                    try:
                        self.latest[name] = float(value)
                    except ValueError:
                        pass
            elif line == "HOMING":
                self.message = "Homing... (arrow keys = emergency stop)"
            elif line == "HOMED":
                self.message = ""
            else:
                self.message = line  # e.g. EMERGENCY STOP message

    def close(self):
        self.running = False
        try:
            self.ser.write(b"S")
            self.ser.close()
        except serial.SerialException:
            pass
        self.root.destroy()


def main():
    port = find_port()
    try:
        ser = serial.Serial(port, BAUD, timeout=0.1)
    except serial.SerialException as e:
        sys.exit(f"Couldn't open {port}: {e}\nIs the Serial Monitor/Plotter still open?")
    root = tk.Tk()
    JogApp(root, ser)
    root.mainloop()


if __name__ == "__main__":
    main()
