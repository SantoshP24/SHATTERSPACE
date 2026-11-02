"""Survival-first, threat-driven MPC agent for ThalesGroup/kessler-game (verified against kesslergame 2.4.0)."""
from .config import CONFIG, make_config

__all__ = ["CONFIG", "make_config", "KesslerMPCController"]


def __getattr__(name):  # lazy: importing the controller pulls in kesslergame (and tkinter via its graphics module)
    if name == "KesslerMPCController":
        from .controller import KesslerMPCController
        return KesslerMPCController
    raise AttributeError(name)
