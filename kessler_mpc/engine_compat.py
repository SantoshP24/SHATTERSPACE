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
    import kesslergame
    return kesslergame
