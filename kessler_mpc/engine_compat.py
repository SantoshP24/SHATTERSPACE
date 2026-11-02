"""Import kesslergame headlessly (the engine imports tkinter at module load even with NoGraphics)."""
import sys
import types


def import_engine():
    try:
        import tkinter  # noqa: F401
    except ImportError:
        m = types.ModuleType("tkinter")
        m.__all__ = []
        m.Tk = object
        m.Canvas = object
        m.NW = "nw"
        sys.modules["tkinter"] = m
    try:
        import inputs  # noqa: F401
    except Exception:
        inp = types.ModuleType("inputs")
        inp.get_gamepad = lambda: []
        inp.devices = types.SimpleNamespace(gamepads=[])
        sys.modules["inputs"] = inp
    import kesslergame
    return kesslergame
