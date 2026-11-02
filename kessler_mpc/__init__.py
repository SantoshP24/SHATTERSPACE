"""Survival-first, threat-driven MPC agent and interactive survival game."""
from .config import CONFIG, make_config

__all__ = [
    "CONFIG",
    "make_config",
    "KesslerMPCController",
    "KesslerAIController",
    "AStarPlanner",
    "SafeZone",
    "PlayerShip",
    "Spaceship",
    "DebrisManager",
    "PhysicsEngine",
    "GameSettings",
    "KesslerGameApp",
]


def __getattr__(name):
    if name == "KesslerMPCController":
        from .controller import KesslerMPCController
        return KesslerMPCController
    if name == "KesslerAIController":
        from .kessler_ai import KesslerAIController
        return KesslerAIController
    if name == "AStarPlanner":
        from .pathfinding import AStarPlanner
        return AStarPlanner
    if name == "SafeZone":
        from .safe_zone import SafeZone
        return SafeZone
    if name == "PlayerShip":
        from .player import PlayerShip
        return PlayerShip
    if name == "Spaceship":
        from .spaceship import Spaceship
        return Spaceship
    if name == "DebrisManager":
        from .debris import DebrisManager
        return DebrisManager
    if name == "PhysicsEngine":
        from .physics import PhysicsEngine
        return PhysicsEngine
    if name == "GameSettings":
        from .settings import GameSettings
        return GameSettings
    if name == "KesslerGameApp":
        from .game import KesslerGameApp
        return KesslerGameApp
    raise AttributeError(name)
