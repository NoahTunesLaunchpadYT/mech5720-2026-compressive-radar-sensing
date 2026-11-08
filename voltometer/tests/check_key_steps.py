"""Headless key checks; no Tk window, serial port, firmware build, or motor access."""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def check():
    serial = ModuleType("serial")
    serial.SerialException = OSError
    serial.tools = ModuleType("serial.tools")
    serial.tools.list_ports = ModuleType("serial.tools.list_ports")
    modules = {
        "serial": serial,
        "serial.tools": serial.tools,
        "serial.tools.list_ports": serial.tools.list_ports,
        "tkinter": ModuleType("tkinter"),
    }
    spec = importlib.util.spec_from_file_location(
        "step_keys", Path(__file__).resolve().parents[2] / "main.py"
    )
    app_module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", modules):
        spec.loader.exec_module(app_module)

    sent, jobs = [], {}

    def after(delay, callback, *args):
        job = object()
        jobs[job] = lambda: callback(*args)
        return job

    app = app_module.JogApp.__new__(app_module.JogApp)
    app.root = SimpleNamespace(after=after, after_cancel=jobs.pop)
    app.ser = SimpleNamespace(write=sent.append)
    app.held, app.release_jobs, app.latest = set(), {}, {}
    app.message, app.running = "", True
    for name in ("lbl_dist", "lbl_volt", "lbl_msg"):
        setattr(app, name, SimpleNamespace(config=lambda **kwargs: None))
    left, right = SimpleNamespace(keysym="Left"), SimpleNamespace(keysym="Right")

    app.on_press(left)
    app.on_press(left)
    app.tick()
    assert sent == [b"F", b"H"]  # Holding and heartbeat cannot queue extra moves.
    app.on_release(left)
    app.on_press(left)  # Linux repeat is a release/press pair; cancel its release.
    assert sent == [b"F", b"H"]
    app.on_release(left)
    jobs.pop(app.release_jobs["Left"])()
    assert sent == [b"F", b"H"]  # A real release does not send stop.
    app.on_press(left)
    app.on_press(right)
    assert sent[-2:] == [b"F", b"B"]
    app.stop_all()
    app.on_press(left)
    app.on_press(right)
    assert sent[-1] == b"S"  # Held-key repeat cannot restart after Space.
    app.on_release(right)
    jobs.pop(app.release_jobs["Right"])()
    app.on_press(right)
    assert sent[-2:] == [b"S", b"B"]  # A fresh press may start again.
    print("PASS distinct presses, repeat filtering, heartbeat, release, stop, restart")


if __name__ == "__main__":
    check()
