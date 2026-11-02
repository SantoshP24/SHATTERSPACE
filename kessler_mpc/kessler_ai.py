"""Kessler AI Controller with Hierarchical A* Global Navigation,
Model Predictive Control (MPC) Local Trajectory Planning, and
Collision-Safety Override for Live Gameplay.
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
    NAVIGATING_TO_SAFE_ZONE = "Navigating to Safe Zone"
    ENGAGING_THREAT = "Engaging Threat"
    EMERGENCY_AVOIDANCE = "EMERGENCY COLLISION AVOIDANCE"
    DOCKING = "Docking in Safe Zone"


@dataclass
class AIDebugTelemetry:
    state: AIBehaviorState = AIBehaviorState.NAVIGATING_TO_SAFE_ZONE
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
    """Live interactive controller using Kessler MPC, A* Global Planning, and Collision Override."""

    def __init__(
        self,
        config: Optional[dict] = None,
        map_size: Tuple[float, float] = (1000.0, 800.0),
        safe_zone: Optional[SafeZone] = None,
        **overrides,
    ):
        self.cfg = make_config(**{**(config or {}), **overrides})
        self.map_w, self.map_h = map_size
        self.safe_zone = safe_zone or SafeZone(center=(self.map_w - 150.0, 150.0), radius=80.0)

        # Core components
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
        self.astar_replan_interval = 0.5  # Replan global path every 0.5s or on emergency
        self.step_count = 0

    def update_safe_zone(self, safe_zone: SafeZone) -> None:
        self.safe_zone = safe_zone

    def get_actions(
        self,
        ship_state: dict,
        game_state: dict,
    ) -> Tuple[float, float, bool, bool]:
        """Compute (thrust, turn, fire, boost) for live game loop."""
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
        else:
            ast_pos = np.zeros((0, 2), dtype=float)
            ast_vel = np.zeros((0, 2), dtype=float)
            ast_size = np.zeros(0, dtype=float)
            ast_rad = np.zeros(0, dtype=float)
            ast_mass = np.zeros(0, dtype=float)

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
            lives=int(ship_state.get("lives", 3)),
            respawning=bool(ship_state.get("is_respawning", False)),
            respawn_left=float(ship_state.get("respawn_time_left", 0.0)),
            can_fire=bool(ship_state.get("can_fire", True)),
            bullets_remaining=int(ship_state.get("bullets_remaining", 100)),
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
        )

    def _decide(self, world: World) -> Tuple[float, float, bool, bool]:
        """Main hierarchical decision pipeline."""
        p = world.params
        risk = self.risk.update(world.lives)
        raw_threats, threat_list = self.threat_eval.assess_threats(world)

        # 1. Evaluate Safe Zone status
        in_safe_zone = self.safe_zone.contains_point(world.pos)
        is_docked = self.safe_zone.is_ship_docked(0)

        # 2. Check Emergency Collision-Safety Override
        critical_threats = [t for t in threat_list if t.is_critical or (t.is_urgent and t.collision_probability > 0.6)]
        emergency_override = len(critical_threats) > 0

        # Update telemetry
        self.telemetry.risk_level = risk.level
        self.telemetry.risk_multiplier = risk.multiplier
        self.telemetry.top_threats = threat_list[:5]
        self.telemetry.critical_threats_count = len(critical_threats)
        self.telemetry.emergency_override = emergency_override
        self.telemetry.threat_level = threat_list[0].overall_threat_score if threat_list else 0.0
        self.telemetry.highest_threat_idx = threat_list[0].asteroid_index if threat_list else None

        # 3. Target selection for weapon firing
        target = self.selector.select(world, raw_threats)
        events = bullet_events(world)

        if target is not None:
            self.telemetry.target_asteroid_idx = target.idx
            self.telemetry.target_pos = (float(world.ast_pos[target.idx][0]), float(world.ast_pos[target.idx][1]))
        else:
            self.telemetry.target_asteroid_idx = None
            self.telemetry.target_pos = None

        # 4. Global A* Path Planning toward Safe Zone
        now = world.time
        need_replan = (
            (now - self.last_astar_replan_time >= self.astar_replan_interval)
            or not self.navigator.path
            or emergency_override
        )
        if need_replan:
            goal_pos = (self.safe_zone.x, self.safe_zone.y)
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

        # 5. Local MPC Planning & Trajectory Evaluation
        plan = self.planner.plan(world, raw_threats, target, risk, events)
        self.telemetry.mpc_cost = plan.cost
        self.telemetry.mpc_best_collides = plan.best_collides
        self.telemetry.mpc_trajectory = [(float(f[0]), float(f[1])) for f in plan.ship_frames]

        # 6. Behavior State & Action Execution
        boost_needed = False

        if emergency_override:
            # === COLLISION-SAFETY OVERRIDE ===
            self.telemetry.state = AIBehaviorState.EMERGENCY_AVOIDANCE
            most_critical = critical_threats[0]

            # Emergency evasive maneuver: calculate normal evasion vector
            rel_v = most_critical.relative_velocity
            rel_spd = max(1.0, float(np.linalg.norm(rel_v)))
            # Tangent vector perpendicular to oncoming threat
            tan_u = np.array([-rel_v[1], rel_v[0]]) / rel_spd

            # Determine side that aligns best with escaping
            side_test = np.dot(tan_u, world.pos - most_critical.position)
            if side_test < 0:
                tan_u = -tan_u

            desired_evade_angle = math.degrees(math.atan2(tan_u[1], tan_u[0])) % 360.0
            turn_err = wrap_angle_deg(desired_evade_angle - world.heading)
            turn = float(np.clip(turn_err / p.frame_dt, -p.turn_max, p.turn_max))

            # If facing close to evasive heading, thrust maximum; if head-on imminent, boost!
            if abs(turn_err) < 45.0:
                thrust = p.thrust_max
            elif abs(turn_err) > 135.0:
                thrust = p.thrust_min
            else:
                thrust = p.thrust_max * 0.5

            if most_critical.time_to_collision < 0.5:
                boost_needed = True

            fire = self.fire.decide(world, thrust, turn, plan, risk, self.planner, raw_threats, target, events)
            return float(thrust), float(turn), bool(fire), boost_needed

        elif in_safe_zone:
            # === DOCKING IN SAFE ZONE ===
            self.telemetry.state = AIBehaviorState.DOCKING
            # Decelerate / station-keep inside safe zone
            dist_to_center = float(np.linalg.norm(world.pos - self.safe_zone.center))
            if dist_to_center > self.safe_zone.radius * 0.4:
                # Steer gently toward center
                to_center = self.safe_zone.center - world.pos
                desired_heading = math.degrees(math.atan2(to_center[1], to_center[0])) % 360.0
                turn_err = wrap_angle_deg(desired_heading - world.heading)
                turn = float(np.clip(turn_err / p.frame_dt, -p.turn_max * 0.5, p.turn_max * 0.5))
                thrust = p.thrust_max * 0.25 if abs(turn_err) < 30.0 else 0.0
            else:
                # Brake to halt
                turn = 0.0
                thrust = p.thrust_min if world.speed > 10.0 else 0.0

            fire = self.fire.decide(world, thrust, turn, plan, risk, self.planner, raw_threats, target, events)
            return float(thrust), float(turn), bool(fire), False

        else:
            # === SAFE ZONE NAVIGATION WITH MPC ===
            self.telemetry.state = AIBehaviorState.NAVIGATING_TO_SAFE_ZONE

            if target is not None and target.score > 0.4 and not plan.best_collides:
                # Engage target if favorable
                self.telemetry.state = AIBehaviorState.ENGAGING_THREAT
                thrust, turn = self.planner.resolve_action(world, plan.thrust_idx, plan.turn_mode, target)
            else:
                # Steer toward active A* waypoint
                if active_waypoint is not None:
                    to_wp = np.array(active_waypoint) - world.pos
                    desired_heading = math.degrees(math.atan2(to_wp[1], to_wp[0])) % 360.0
                    turn_err = wrap_angle_deg(desired_heading - world.heading)
                    turn = float(np.clip(turn_err / p.frame_dt, -p.turn_max, p.turn_max))

                    # Combine waypoint heading with MPC safe thrust
                    if not plan.best_collides and abs(turn_err) < 60.0:
                        thrust = p.thrust_max
                    else:
                        thrust, mpc_turn = self.planner.resolve_action(world, plan.thrust_idx, plan.turn_mode, None)
                        # Blend turn when hazard is near
                        if plan.best_collides:
                            turn = mpc_turn
                else:
                    thrust, turn = self.planner.resolve_action(world, plan.thrust_idx, plan.turn_mode, None)

            fire = self.fire.decide(world, thrust, turn, plan, risk, self.planner, raw_threats, target, events)
            return float(thrust), float(turn), bool(fire), False
