"""Safe Zone objective and arrival detection for Kessler AI survival game.

Represents an orbital sanctuary / deep-space docking station that the player and
autonomous AI must navigate toward to achieve victory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class DockingState:
    ship_id: int
    is_inside: bool
    dock_time: float = 0.0
    dock_progress: float = 0.0
    is_docked: bool = False


class SafeZone:
    """Designated safe zone sanctuary with docking mechanics and defense field."""

    def __init__(
        self,
        center: Tuple[float, float] = (850.0, 150.0),
        radius: float = 80.0,
        required_dock_time: float = 3.0,
        deflect_debris: bool = True,
    ):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.required_dock_time = float(required_dock_time)
        self.deflect_debris = deflect_debris
        self.pulse_phase: float = 0.0
        self.docking_states: Dict[int, DockingState] = {}

    @property
    def x(self) -> float:
        return float(self.center[0])

    @property
    def y(self) -> float:
        return float(self.center[1])

    def distance_to_perimeter(self, pos: Tuple[float, float] | np.ndarray) -> float:
        p = np.asarray(pos, dtype=float)
        dist = float(np.linalg.norm(p - self.center))
        return max(0.0, dist - self.radius)

    def distance_to_center(self, pos: Tuple[float, float] | np.ndarray) -> float:
        p = np.asarray(pos, dtype=float)
        return float(np.linalg.norm(p - self.center))

    def contains_point(self, pos: Tuple[float, float] | np.ndarray, margin: float = 0.0) -> bool:
        p = np.asarray(pos, dtype=float)
        return float(np.linalg.norm(p - self.center)) <= (self.radius + margin)

    def update(self, dt: float, ship_states: Dict[int, Tuple[float, float]]) -> None:
        """Update pulse animation and evaluate docking status for all ships."""
        self.pulse_phase = (self.pulse_phase + dt * 2.0) % (2.0 * math.pi)

        for ship_id, pos in ship_states.items():
            if ship_id not in self.docking_states:
                self.docking_states[ship_id] = DockingState(ship_id=ship_id, is_inside=False)

            state = self.docking_states[ship_id]
            is_inside = self.contains_point(pos)
            state.is_inside = is_inside

            if is_inside and not state.is_docked:
                state.dock_time += dt
                state.dock_progress = min(1.0, state.dock_time / self.required_dock_time)
                if state.dock_progress >= 1.0:
                    state.is_docked = True
            elif not is_inside and not state.is_docked:
                # Gradual decay if ship drifts out
                state.dock_time = max(0.0, state.dock_time - dt * 0.5)
                state.dock_progress = state.dock_time / self.required_dock_time

    def is_ship_docked(self, ship_id: int) -> bool:
        return self.docking_states.get(ship_id, DockingState(ship_id, False)).is_docked

    def get_dock_progress(self, ship_id: int) -> float:
        return self.docking_states.get(ship_id, DockingState(ship_id, False)).dock_progress

    def apply_debris_deflection(self, asteroid_positions: np.ndarray, asteroid_velocities: np.ndarray) -> None:
        """Sanctuary shield barrier deflects incoming debris away from station core."""
        if not self.deflect_debris or len(asteroid_positions) == 0:
            return

        diff = asteroid_positions - self.center[None, :]
        dist = np.linalg.norm(diff, axis=1)
        # Inside or touching shield perimeter
        mask = (dist < self.radius * 1.05) & (dist > 1e-4)

        if np.any(mask):
            normal = diff[mask] / dist[mask, None]
            # Reflect velocity outward along normal if heading inward
            radial_vel = np.sum(asteroid_velocities[mask] * normal, axis=1)
            inward = radial_vel < 0.0
            if np.any(inward):
                idx = np.where(mask)[0][inward]
                norm_inward = normal[inward]
                asteroid_velocities[idx] -= 1.8 * radial_vel[inward, None] * norm_inward
