"""Kessler AI Controller with Hierarchical A* Global Navigation,
Model Predictive Control (MPC) Local Trajectory Planning, and
Collision-Safety Override for Autonomous Asteroid Survival.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from .config import CONFIG, make_config
from .dynamics import ShipParams, speed_step, step
from .events import bullet_events
from .firecontrol import FireControl
from .geometry import aim_point_bearing_deg, wrap_angle_deg
from .pathfinding import AStarPlanner, WaypointNavigator
from .planner import MPCPlanner, Plan
from .risk import AdaptiveRisk, RiskState
from .safe_zone import SafeZone
from .targeting import Target, TargetSelector
from .threat import Threats, assess
from .threat_assessment import DebrisThreatInfo, ThreatEvaluator
from .world import World


class AIBehaviorState(Enum):
    ATTACK = "ATTACK"
    EVADE = "EVADE"
    REPOSITION = "REPOSITION"
    SURVIVE = "SURVIVE"


@dataclass
class AIDebugTelemetry:
    state: AIBehaviorState = AIBehaviorState.REPOSITION
    target_asteroid_idx: Optional[int] = None
    target_pos: Optional[Tuple[float, float]] = None
    threat_level: float = 0.0
    highest_threat_idx: Optional[int] = None
    emergency_override: bool = False
    risk_level: int = 0
    risk_multiplier: float = 1.0
    mpc_cost: float = 0.0
    mpc_best_collides: bool = False
    mpc_trajectory: List[Tuple[float, float]] = field(default_factory=list)
    astar_path: List[Tuple[float, float]] = field(default_factory=list)
    active_waypoint: Optional[Tuple[float, float]] = None
    critical_threats_count: int = 0
    top_threats: List[DebrisThreatInfo] = field(default_factory=list)


class KesslerAIController:
    """Fully autonomous controller using Kessler MPC, A* Global Planning, and Collision Override."""

    def __init__(
        self,
        config: Optional[dict] = None,
        map_size: Tuple[float, float] = (1000.0, 800.0),
        safe_zone: Optional[SafeZone] = None,
        **overrides,
    ):
        merged_overrides = {"wrap_boundaries": False, **overrides}
        self.cfg = make_config(**{**(config or {}), **merged_overrides})
        self.map_w, self.map_h = map_size
        self.safe_zone = safe_zone

        # Core modules
        self.threat_eval = ThreatEvaluator(self.cfg)
        self.risk = AdaptiveRisk(self.cfg)
        self.selector = TargetSelector(self.cfg)
        self.planner = MPCPlanner(self.cfg)
        self.fire = FireControl(self.cfg)
        self.astar = AStarPlanner(map_w=self.map_w, map_h=self.map_h, cell_size=25.0)
        self.navigator = WaypointNavigator(waypoint_reach_radius=60.0)

        # State tracking
        self.telemetry = AIDebugTelemetry()
        self.last_astar_replan_time = -999.0
        self.astar_replan_interval = 0.6  # Replan global path periodically or on emergency
        self.step_count = 0

    def update_safe_zone(self, safe_zone: Optional[SafeZone]) -> None:
        self.safe_zone = safe_zone

    def get_actions(
        self,
        ship_state: dict,
        game_state: dict,
    ) -> Tuple[float, float, bool, bool]:
        """Compute (thrust, turn, fire, boost) for live autonomous game loop."""
        self.step_count += 1
        world = self._build_world(ship_state, game_state)
        return self._decide(world)

    def _build_world(self, ship_state: dict, game_state: dict) -> World:
        """Construct a World instance compatible with existing kessler_mpc modules."""
        map_w, map_h = float(game_state.get("map_size", (self.map_w, self.map_h))[0]), float(
            game_state.get("map_size", (self.map_w, self.map_h))[1]
        )
        dt = 1.0 / float(game_state.get("frame_rate", 60.0))
        tr = ship_state.get("thrust_range", (-480.0, 480.0))
        tu = ship_state.get("turn_rate_range", (-180.0, 180.0))

        params = ShipParams(
            thrust_min=float(tr[0]),
            thrust_max=float(tr[1]),
            turn_max=float(tu[1]),
            max_speed=float(ship_state.get("max_speed", 240.0)),
            drag=float(ship_state.get("drag", 80.0)),
            radius=float(ship_state.get("radius", 20.0)),
            map_w=map_w,
            map_h=map_h,
            frame_dt=dt,
        )

        asteroids = game_state.get("asteroids", [])
        n_ast = len(asteroids)

        if n_ast > 0:
            ast_pos = np.array([a["position"] for a in asteroids], dtype=float)
            ast_vel = np.array([a["velocity"] for a in asteroids], dtype=float)
            ast_size = np.array([a.get("size", 1) for a in asteroids], dtype=float)
            ast_rad = np.array([a.get("radius", 20.0) for a in asteroids], dtype=float)
            ast_mass = np.array([a.get("mass", 100.0) for a in asteroids], dtype=float)
            ast_is_pen = np.array([bool(a.get("is_penalty", False)) for a in asteroids], dtype=bool)
        else:
            ast_pos = np.zeros((0, 2), dtype=float)
            ast_vel = np.zeros((0, 2), dtype=float)
            ast_size = np.zeros(0, dtype=float)
            ast_rad = np.zeros(0, dtype=float)
            ast_mass = np.zeros(0, dtype=float)
            ast_is_pen = np.zeros(0, dtype=bool)

        pos = np.array(ship_state.get("position", (0.0, 0.0)), dtype=float)
        vel = np.array(ship_state.get("velocity", (0.0, 0.0)), dtype=float)
        speed = float(ship_state.get("speed", np.linalg.norm(vel)))
        heading = float(ship_state.get("heading", 0.0))

        return World(
            params=params,
            pos=pos,
            vel=vel,
            speed=speed,
            heading=heading,
            lives=int(ship_state.get("lives_remaining", ship_state.get("lives", 3))),
            respawning=bool(ship_state.get("is_respawning", False)),
            respawn_left=float(ship_state.get("respawn_time_left", 0.0)),
            can_fire=bool(ship_state.get("can_fire", True)),
            bullets_remaining=int(ship_state.get("bullets_remaining", 250)),
            time=float(game_state.get("time", 0.0)),
            ast_pos=ast_pos,
            ast_vel=ast_vel,
            ast_size=ast_size,
            ast_rad=ast_rad,
            ast_mass=ast_mass,
            other_pos=np.zeros((0, 2), dtype=float),
            other_vel=np.zeros((0, 2), dtype=float),
            other_rad=np.zeros(0, dtype=float),
            bul_pos=np.zeros((0, 2), dtype=float),
            bul_vel=np.zeros((0, 2), dtype=float),
            bul_mass=np.zeros(0, dtype=float),
            random_splits=False,
            health=float(ship_state.get("health", 100.0)),
            shields=float(ship_state.get("shields", 100.0)),
            ast_is_penalty=ast_is_pen,
        )

    def _find_clearest_sector(self, world: World) -> Tuple[float, float]:
        """Find the point in the arena that maximizes clearance from all active asteroids."""
        if world.n_ast == 0:
            return (self.map_w * 0.5, self.map_h * 0.5)

        # 4x4 candidate grid in the interior
        xs = np.linspace(self.map_w * 0.15, self.map_w * 0.85, 4)
        ys = np.linspace(self.map_h * 0.15, self.map_h * 0.85, 4)
        candidates = [(float(x), float(y)) for x in xs for y in ys]

        best_pt = candidates[0]
        max_min_dist = -1.0

        for cx, cy in candidates:
            c_pos = np.array([cx, cy])
            # Minimum distance from candidate to any asteroid
            dists = np.linalg.norm(world.ast_pos - c_pos[None, :], axis=1) - world.ast_rad
            min_d = float(np.min(dists))
            if min_d > max_min_dist:
                max_min_dist = min_d
                best_pt = (cx, cy)

        return best_pt

    def _decide(self, world: World) -> Tuple[float, float, bool, bool]:
        """Hierarchical autonomous decision pipeline evaluating ATTACK, EVADE, REPOSITION, and SURVIVE."""
        p = world.params
        risk = self.risk.update(world.lives, world.health)
        raw_threats, threat_list = self.threat_eval.assess_threats(world)

        # 1. Threat Classification & Emergency Override Check
        critical_threats = [t for t in threat_list if t.is_critical or (t.is_urgent and t.collision_probability > 0.45)]
        emergency_override = len(critical_threats) > 0

        # Update telemetry
        self.telemetry.risk_level = risk.level
        self.telemetry.risk_multiplier = risk.multiplier
        self.telemetry.top_threats = threat_list[:5]
        self.telemetry.critical_threats_count = len(critical_threats)
        self.telemetry.emergency_override = emergency_override
        self.telemetry.threat_level = threat_list[0].overall_threat_score if threat_list else 0.0
        self.telemetry.highest_threat_idx = threat_list[0].asteroid_index if threat_list else None

        # 2. Target selection for plasma cannon firing
        target = self.selector.select(world, raw_threats)
        events = bullet_events(world)

        if target is not None:
            self.telemetry.target_asteroid_idx = target.idx
            self.telemetry.target_pos = (float(world.ast_pos[target.idx][0]), float(world.ast_pos[target.idx][1]))
        else:
            self.telemetry.target_asteroid_idx = None
            self.telemetry.target_pos = None

        # 3. Global A* Path Planning
        now = world.time
        need_replan = (
            (now - self.last_astar_replan_time >= self.astar_replan_interval)
            or not self.navigator.path
            or emergency_override
        )
        if need_replan:
            if self.safe_zone is not None:
                goal_pos = (self.safe_zone.x, self.safe_zone.y)
            else:
                goal_pos = self._find_clearest_sector(world)

            path = self.astar.plan_path(
                start_pos=(world.pos[0], world.pos[1]),
                goal_pos=goal_pos,
                asteroid_positions=world.ast_pos,
                asteroid_velocities=world.ast_vel,
                asteroid_radii=world.ast_rad,
                ship_radius=p.radius,
            )
            self.navigator.set_path(path)
            self.last_astar_replan_time = now
            self.telemetry.astar_path = list(path)

        active_waypoint = self.navigator.get_current_target((world.pos[0], world.pos[1]))
        self.telemetry.active_waypoint = active_waypoint

        # 4. Local MPC Trajectory Optimization
        plan = self.planner.plan(world, raw_threats, target, risk, events)
        self.telemetry.mpc_cost = plan.cost
        self.telemetry.mpc_best_collides = plan.best_collides
        self.telemetry.mpc_trajectory = [(float(f[0]), float(f[1])) for f in plan.ship_frames]

        # 5. Boundary Awareness & Corner Escape Checks
        center = np.array([self.map_w * 0.5, self.map_h * 0.5])
        to_center = center - world.pos
        center_dist = float(np.linalg.norm(to_center))
        center_dir = to_center / (center_dist + 1e-6)
        center_heading = math.degrees(math.atan2(to_center[1], to_center[0])) % 360.0

        corner_margin = 135.0
        wall_margin = 75.0
        near_left = world.pos[0] < corner_margin
        near_right = world.pos[0] > self.map_w - corner_margin
        near_top = world.pos[1] < corner_margin
        near_bottom = world.pos[1] > self.map_h - corner_margin
        is_in_corner = (near_left or near_right) and (near_top or near_bottom)

        rad_now = math.radians(world.heading)
        facing_wall = (
            (world.pos[0] < wall_margin and math.cos(rad_now) < 0) or
            (world.pos[0] > self.map_w - wall_margin and math.cos(rad_now) > 0) or
            (world.pos[1] < wall_margin and math.sin(rad_now) < 0) or
            (world.pos[1] > self.map_h - wall_margin and math.sin(rad_now) > 0)
        )

        # 6. Behavior Mode Arbitration: EVADE, SURVIVE, ATTACK, or REPOSITION
        boost_needed = False

        # --- A. EVADE (Emergency Collision Override) ---
        if emergency_override:
            self.telemetry.state = AIBehaviorState.EVADE
            most_critical = critical_threats[0]

            # Compute tangential evasion vector perpendicular to oncoming threat
            rel_v = most_critical.relative_velocity
            rel_spd = max(1.0, float(np.linalg.norm(rel_v)))
            tan_u = np.array([-rel_v[1], rel_v[0]]) / rel_spd

            # Choose direction moving away from collision center
            if np.dot(tan_u, world.pos - most_critical.position) < 0:
                tan_u = -tan_u

            # Boundary awareness: deflect evasion if it leads toward wall or corner
            evade_future = world.pos + tan_u * 80.0
            if (evade_future[0] < wall_margin or evade_future[0] > self.map_w - wall_margin or
                evade_future[1] < wall_margin or evade_future[1] > self.map_h - wall_margin):
                rev_future = world.pos - tan_u * 80.0
                if (rev_future[0] > wall_margin and rev_future[0] < self.map_w - wall_margin and
                    rev_future[1] > wall_margin and rev_future[1] < self.map_h - wall_margin):
                    tan_u = -tan_u
                else:
                    tan_u = 0.5 * tan_u + 0.9 * center_dir
                    tan_u = tan_u / (np.linalg.norm(tan_u) + 1e-6)

            desired_evade_angle = math.degrees(math.atan2(tan_u[1], tan_u[0])) % 360.0
            turn_err = wrap_angle_deg(desired_evade_angle - world.heading)
            turn = float(np.clip(turn_err / p.frame_dt, -p.turn_max, p.turn_max))

            if abs(turn_err) < 45.0:
                thrust = p.thrust_max
            elif abs(turn_err) > 135.0:
                thrust = p.thrust_min
            else:
                thrust = p.thrust_max * 0.5

            if most_critical.time_to_collision < 0.45:
                boost_needed = True

            fire = self.fire.decide(world, thrust, turn, plan, risk, self.planner, raw_threats, target, events)
            return float(thrust), float(turn), bool(fire), boost_needed

        # --- CORNER ESCAPE OVERRIDE ---
        if is_in_corner:
            self.telemetry.state = AIBehaviorState.REPOSITION
            turn_err = wrap_angle_deg(center_heading - world.heading)
            turn = float(np.clip(turn_err / p.frame_dt, -p.turn_max, p.turn_max))
            thrust = p.thrust_max if abs(turn_err) < 60.0 else 0.0
            fire = self.fire.decide(world, thrust, turn, plan, risk, self.planner, raw_threats, target, events)
            return float(thrust), float(turn), bool(fire), False

        # --- WALL TURNAROUND OVERRIDE ---
        if facing_wall and plan.best_collides:
            self.telemetry.state = AIBehaviorState.REPOSITION
            turn_err = wrap_angle_deg(center_heading - world.heading)
            turn = float(np.clip(turn_err / p.frame_dt, -p.turn_max, p.turn_max))
            thrust = p.thrust_max if abs(turn_err) < 60.0 else 0.0
            fire = self.fire.decide(world, thrust, turn, plan, risk, self.planner, raw_threats, target, events)
            return float(thrust), float(turn), bool(fire), False

        # --- B. SURVIVE (Low Health or Critical Hazard Density) ---
        is_low_health = world.health <= 40.0 or risk.level >= 2
        threat_density_high = len(threat_list) >= 4 and threat_list[0].overall_threat_score > 0.40

        if is_low_health or threat_density_high:
            self.telemetry.state = AIBehaviorState.SURVIVE
            # Prioritize maintaining safety clearance and moving to clear space
            if threat_list:
                nearest_threat = threat_list[0]
                if nearest_threat.distance < 180.0:
                    # Steer directly away from closest hazard
                    away = world.pos - nearest_threat.position
                    away_angle = math.degrees(math.atan2(away[1], away[0])) % 360.0
                    turn_err = wrap_angle_deg(away_angle - world.heading)
                    turn = float(np.clip(turn_err / p.frame_dt, -p.turn_max, p.turn_max))
                    thrust = p.thrust_max if abs(turn_err) < 60.0 else 0.0
                else:
                    thrust, turn = self.planner.resolve_action(world, plan.thrust_idx, plan.turn_mode, None)
            else:
                thrust, turn = self.planner.resolve_action(world, plan.thrust_idx, plan.turn_mode, None)

            # In SURVIVE mode, only fire if a shot is safe and won't spawn hazardous children nearby
            fire = self.fire.decide(world, thrust, turn, plan, risk, self.planner, raw_threats, target, events)
            return float(thrust), float(turn), bool(fire), False

        # --- C. ATTACK (Target Selected and Safe Conditions) ---
        if target is not None and target.score > 0.25 and not plan.best_collides:
            self.telemetry.state = AIBehaviorState.ATTACK
            # Resolve aim and forward intercept via MPC
            thrust, turn = self.planner.resolve_action(world, plan.thrust_idx, plan.turn_mode, target)
            fire = self.fire.decide(world, thrust, turn, plan, risk, self.planner, raw_threats, target, events)
            return float(thrust), float(turn), bool(fire), False

        # --- D. REPOSITION (Relocating via A* Waypoints to Open Space) ---
        self.telemetry.state = AIBehaviorState.REPOSITION
        if active_waypoint is not None:
            to_wp = np.array(active_waypoint) - world.pos
            desired_heading = math.degrees(math.atan2(to_wp[1], to_wp[0])) % 360.0
            turn_err = wrap_angle_deg(desired_heading - world.heading)
            turn = float(np.clip(turn_err / p.frame_dt, -p.turn_max, p.turn_max))

            if not plan.best_collides and abs(turn_err) < 60.0:
                thrust = p.thrust_max
            else:
                thrust, mpc_turn = self.planner.resolve_action(world, plan.thrust_idx, plan.turn_mode, None)
                if plan.best_collides:
                    turn = mpc_turn
        else:
            thrust, turn = self.planner.resolve_action(world, plan.thrust_idx, plan.turn_mode, None)

        fire = self.fire.decide(world, thrust, turn, plan, risk, self.planner, raw_threats, target, events)
        return float(thrust), float(turn), bool(fire), False
