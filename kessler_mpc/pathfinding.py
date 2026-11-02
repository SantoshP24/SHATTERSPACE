"""A* Global Route Planning for Kessler AI safe-zone navigation.

Provides:
- AStarGrid: Discretized occupancy and hazard-cost grid over the game arena.
- AStarPlanner: Computes optimal, collision-free global paths through moving debris fields toward the safe zone.
- PathSmoother: String-pulling / line-of-sight path smoothing.
- WaypointNavigator: Follows global waypoints and provides target points for local MPC trajectory tracking.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass(order=True)
class _PriorityNode:
    priority: float
    cost: float = field(compare=False)
    x: int = field(compare=False)
    y: int = field(compare=False)


class AStarGrid:
    """Discretized grid with static obstacles and dynamic hazard penalties."""

    def __init__(self, map_w: float = 1000.0, map_h: float = 800.0, cell_size: float = 25.0):
        self.map_w = float(map_w)
        self.map_h = float(map_h)
        self.cell_size = float(cell_size)
        self.cols = int(math.ceil(self.map_w / self.cell_size))
        self.rows = int(math.ceil(self.map_h / self.cell_size))
        # Obstacle mask: True = impassable
        self.blocked = np.zeros((self.rows, self.cols), dtype=bool)
        # Hazard cost: extra penalty for crossing near hazards or their trajectories
        self.hazard_cost = np.zeros((self.rows, self.cols), dtype=np.float32)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        c = int(np.clip(x / self.cell_size, 0, self.cols - 1))
        r = int(np.clip(y / self.cell_size, 0, self.rows - 1))
        return c, r

    def grid_to_world(self, c: int, r: int) -> Tuple[float, float]:
        x = (c + 0.5) * self.cell_size
        y = (r + 0.5) * self.cell_size
        return float(min(x, self.map_w)), float(min(y, self.map_h))

    def update_hazards(
        self,
        asteroid_positions: np.ndarray,
        asteroid_velocities: np.ndarray,
        asteroid_radii: np.ndarray,
        ship_radius: float = 20.0,
        safety_multiplier: float = 1.3,
        lookahead_s: float = 1.5,
    ) -> None:
        """Populate blocked cells and hazard cost gradients based on live debris."""
        self.blocked.fill(False)
        self.hazard_cost.fill(0.0)

        n_ast = len(asteroid_radii)
        if n_ast == 0:
            return

        # Precompute coordinate grids for vectorised distance checks
        xs = (np.arange(self.cols) + 0.5) * self.cell_size
        ys = (np.arange(self.rows) + 0.5) * self.cell_size
        grid_x, grid_y = np.meshgrid(xs, ys)

        for i in range(n_ast):
            pos = asteroid_positions[i]
            vel = asteroid_velocities[i]
            rad = float(asteroid_radii[i])
            safe_rad = (rad + ship_radius) * safety_multiplier

            # Bounding box for current asteroid footprint
            min_c = max(0, int((pos[0] - safe_rad) / self.cell_size))
            max_c = min(self.cols - 1, int((pos[0] + safe_rad) / self.cell_size))
            min_r = max(0, int((pos[1] - safe_rad) / self.cell_size))
            max_r = min(self.rows - 1, int((pos[1] + safe_rad) / self.cell_size))

            if min_c <= max_c and min_r <= max_r:
                sub_x = grid_x[min_r : max_r + 1, min_c : max_c + 1]
                sub_y = grid_y[min_r : max_r + 1, min_c : max_c + 1]
                dist_sq = (sub_x - pos[0]) ** 2 + (sub_y - pos[1]) ** 2
                self.blocked[min_r : max_r + 1, min_c : max_c + 1] |= dist_sq <= (safe_rad**2)

            # Trajectory hazard corridor: predict positions over lookahead
            speed = float(np.linalg.norm(vel))
            if speed > 10.0:
                steps = 4
                for step in range(1, steps + 1):
                    t = (step / steps) * lookahead_s
                    pred_pos = pos + vel * t
                    # Wrapped or clamped position
                    pred_x = float(pred_pos[0] % self.map_w)
                    pred_y = float(pred_pos[1] % self.map_h)
                    p_min_c = max(0, int((pred_x - safe_rad * 1.2) / self.cell_size))
                    p_max_c = min(self.cols - 1, int((pred_x + safe_rad * 1.2) / self.cell_size))
                    p_min_r = max(0, int((pred_y - safe_rad * 1.2) / self.cell_size))
                    p_max_r = min(self.rows - 1, int((pred_y + safe_rad * 1.2) / self.cell_size))

                    if p_min_c <= p_max_c and p_min_r <= p_max_r:
                        sub_x = grid_x[p_min_r : p_max_r + 1, p_min_c : p_max_c + 1]
                        sub_y = grid_y[p_min_r : p_max_r + 1, p_min_c : p_max_c + 1]
                        dist_sq = (sub_x - pred_x) ** 2 + (sub_y - pred_y) ** 2
                        corridor = dist_sq <= ((safe_rad * 1.2) ** 2)
                        # Hazard penalty decays with prediction time
                        decay = (1.0 - (step / steps) * 0.5) * 50.0
                        self.hazard_cost[p_min_r : p_max_r + 1, p_min_c : p_max_c + 1] += np.where(corridor, decay, 0.0)


class AStarPlanner:
    """A* pathfinder with 8-directional movement, diagonal cost, and dynamic hazard weighting."""

    def __init__(self, map_w: float = 1000.0, map_h: float = 800.0, cell_size: float = 25.0):
        self.grid = AStarGrid(map_w=map_w, map_h=map_h, cell_size=cell_size)
        self.last_path: List[Tuple[float, float]] = []

    def plan_path(
        self,
        start_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        asteroid_positions: np.ndarray,
        asteroid_velocities: np.ndarray,
        asteroid_radii: np.ndarray,
        ship_radius: float = 20.0,
    ) -> List[Tuple[float, float]]:
        """Compute smooth global path from start_pos to goal_pos.

        Returns list of (x, y) waypoints in world coordinates.
        """
        # Update obstacle grid with live asteroid state
        self.grid.update_hazards(
            asteroid_positions=asteroid_positions,
            asteroid_velocities=asteroid_velocities,
            asteroid_radii=asteroid_radii,
            ship_radius=ship_radius,
        )

        start_c, start_r = self.grid.world_to_grid(start_pos[0], start_pos[1])
        goal_c, goal_r = self.grid.world_to_grid(goal_pos[0], goal_pos[1])

        # If start is inside a blocked cell, unblock start cell temporarily so planning can escape
        self.grid.blocked[start_r, start_c] = False
        # If goal is blocked, unblock goal cell
        self.grid.blocked[goal_r, goal_c] = False

        if (start_c, start_r) == (goal_c, goal_r):
            return [goal_pos]

        # 8-connected neighbors: (dc, dr, base_cost)
        sqrt2 = math.sqrt(2.0)
        neighbors = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, sqrt2),
            (1, -1, sqrt2),
            (-1, 1, sqrt2),
            (-1, -1, sqrt2),
        ]

        def heuristic(c: int, r: int) -> float:
            # Octile distance heuristic
            dx = abs(c - goal_c)
            dy = abs(r - goal_r)
            return (dx + dy) + (sqrt2 - 2.0) * min(dx, dy)

        open_set: List[_PriorityNode] = []
        heapq.heappush(open_set, _PriorityNode(priority=heuristic(start_c, start_r), cost=0.0, x=start_c, y=start_r))

        g_score = { (start_c, start_r): 0.0 }
        came_from = {}
        closed_set = set()

        found = False
        max_iterations = 3000
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)
            curr_coord = (current.x, current.y)

            if curr_coord in closed_set:
                continue
            closed_set.add(curr_coord)

            if curr_coord == (goal_c, goal_r):
                found = True
                break

            curr_cost = g_score[curr_coord]

            for dc, dr, move_dist in neighbors:
                nc, nr = current.x + dc, current.y + dr
                if 0 <= nc < self.grid.cols and 0 <= nr < self.grid.rows:
                    if self.grid.blocked[nr, nc]:
                        continue

                    # Diagonal corner cutting check
                    if dc != 0 and dr != 0:
                        if self.grid.blocked[current.y, nc] and self.grid.blocked[nr, current.x]:
                            continue

                    hazard = float(self.grid.hazard_cost[nr, nc])
                    tentative_g = curr_cost + move_dist * self.grid.cell_size + hazard

                    neighbor_coord = (nc, nr)
                    if neighbor_coord not in g_score or tentative_g < g_score[neighbor_coord]:
                        came_from[neighbor_coord] = curr_coord
                        g_score[neighbor_coord] = tentative_g
                        h = heuristic(nc, nr) * self.grid.cell_size
                        f = tentative_g + h
                        heapq.heappush(open_set, _PriorityNode(priority=f, cost=tentative_g, x=nc, y=nr))

        if not found:
            # Fallback: find node closest to goal explored so far
            if closed_set:
                best_coord = min(closed_set, key=lambda p: (p[0] - goal_c) ** 2 + (p[1] - goal_r) ** 2)
            else:
                best_coord = (start_c, start_r)
        else:
            best_coord = (goal_c, goal_r)

        # Reconstruct path
        path_cells = [best_coord]
        curr = best_coord
        while curr in came_from:
            curr = came_from[curr]
            path_cells.append(curr)
        path_cells.reverse()

        # Convert to world coordinates
        world_path = [self.grid.grid_to_world(c, r) for c, r in path_cells]
        # Replace first with start_pos and last with goal_pos
        world_path[0] = (float(start_pos[0]), float(start_pos[1]))
        if found:
            world_path[-1] = (float(goal_pos[0]), float(goal_pos[1]))

        # Smooth path with line-of-sight checks
        smoothed = self._smooth_path(world_path)
        self.last_path = smoothed
        return smoothed

    def _smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Remove intermediate waypoints where a straight line is unblocked."""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # Look as far ahead as possible
            furthest_visible = current_idx + 1
            for next_idx in range(len(path) - 1, current_idx + 1, -1):
                if self._line_of_sight(path[current_idx], path[next_idx]):
                    furthest_visible = next_idx
                    break
            smoothed.append(path[furthest_visible])
            current_idx = furthest_visible

        return smoothed

    def _line_of_sight(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Bresenham-like raycast checking if segment between p1 and p2 crosses blocked cells."""
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(2, int(dist / (self.grid.cell_size * 0.5)))
        for s in range(1, steps):
            t = s / steps
            x = p1[0] + (p2[0] - p1[0]) * t
            y = p1[1] + (p2[1] - p1[1]) * t
            c, r = self.grid.world_to_grid(x, y)
            if self.grid.blocked[r, c]:
                return False
        return True


class WaypointNavigator:
    """Tracks progression along a path and supplies the current navigation target."""

    def __init__(self, waypoint_reach_radius: float = 50.0):
        self.reach_radius = waypoint_reach_radius
        self.path: List[Tuple[float, float]] = []
        self.target_idx: int = 0

    def set_path(self, path: List[Tuple[float, float]]) -> None:
        self.path = list(path)
        self.target_idx = 0

    def get_current_target(self, ship_pos: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if not self.path:
            return None

        # Advance waypoints that have been reached
        while self.target_idx < len(self.path):
            wp = self.path[self.target_idx]
            dist = math.hypot(wp[0] - ship_pos[0], wp[1] - ship_pos[1])
            if dist <= self.reach_radius and self.target_idx < len(self.path) - 1:
                self.target_idx += 1
            else:
                break

        return self.path[self.target_idx]

    @property
    def is_finished(self) -> bool:
        return self.target_idx >= len(self.path) - 1
