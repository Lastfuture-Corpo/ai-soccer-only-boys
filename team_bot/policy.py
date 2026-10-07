from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


DIRECTION_VECTORS = {
    "STAY": (0, 0),
    "UP": (0, 1),
    "UP_RIGHT": (1, 1),
    "RIGHT": (1, 0),
    "DOWN_RIGHT": (1, -1),
    "DOWN": (0, -1),
    "DOWN_LEFT": (-1, -1),
    "LEFT": (-1, 0),
    "UP_LEFT": (-1, 1),
}
MOVES = list(DIRECTION_VECTORS)
KICK_DIRECTIONS = [name for name in MOVES if name != "STAY"]
ACTION_COUNT = len(MOVES) + len(KICK_DIRECTIONS) * 3


def direction_toward(dx: float, dy: float, dead_zone: float = 1.0) -> str:
    horizontal = "" if abs(dx) <= dead_zone else ("RIGHT" if dx > 0 else "LEFT")
    vertical = "" if abs(dy) <= dead_zone else ("UP" if dy > 0 else "DOWN")
    return f"{vertical}_{horizontal}" if vertical and horizontal else vertical or horizontal or "STAY"


def _bucket(value: float, limits: tuple[float, ...]) -> int:
    return next((index for index, limit in enumerate(limits) if value < limit), len(limits))


def _sector(dx: float, dy: float) -> int:
    if abs(dx) + abs(dy) < 1e-9:
        return 8
    return int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8


def _move_is_safe(observation: dict[str, Any], move: str) -> bool:
    if move == "STAY":
        return True
    state = observation["state"]
    me = state["players"][observation["player_id"]]
    field = state["field"]
    speed = float(field.get("player_speed", 4.0))
    radius = float(field.get("player_radius", 3.0))
    vx, vy = DIRECTION_VECTORS[move]
    length = math.hypot(vx, vy) or 1.0
    x = me["x"] + vx / length * speed
    y = me["y"] + vy / length * speed
    if x - radius < 0 or x + radius > field["width"] or y - radius < 0 or y + radius > field["height"]:
        return False
    for obstacle in state.get("obstacles", []):
        closest_x = min(max(x, obstacle["x"]), obstacle["x"] + obstacle["width"])
        closest_y = min(max(y, obstacle["y"]), obstacle["y"] + obstacle["height"])
        if math.hypot(x - closest_x, y - closest_y) < radius + 0.25:
            return False
    return True


def _safe_move(observation: dict[str, Any], preferred: str) -> str:
    preferred_vector = DIRECTION_VECTORS[preferred]
    choices = [move for move in MOVES if move != "STAY" and _move_is_safe(observation, move)]
    if not choices:
        return "STAY"
    return max(
        choices,
        key=lambda move: (
            DIRECTION_VECTORS[move][0] * preferred_vector[0]
            + DIRECTION_VECTORS[move][1] * preferred_vector[1],
            move == preferred,
        ),
    )


def _segment_hits_box(
    x1: float, y1: float, x2: float, y2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> bool:
    """Liang-Barsky 2D ray-box intersection."""
    dx = x2 - x1
    dy = y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - bx1, bx2 - x1, y1 - by1, by2 - y1]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0.0:
                return False
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return False
                if t < u2:
                    u2 = t
    return u1 <= u2


def is_kick_path_blocked(
    x1: float, y1: float, x2: float, y2: float,
    obstacles: list[dict[str, Any]],
    opponent: dict[str, Any],
    ball_radius: float = 1.5,
) -> bool:
    margin = ball_radius + 1.0
    for obs in obstacles:
        bx1 = obs["x"] - margin
        by1 = obs["y"] - margin
        bx2 = obs["x"] + obs["width"] + margin
        by2 = obs["y"] + obs["height"] + margin
        if _segment_hits_box(x1, y1, x2, y2, bx1, by1, bx2, by2):
            return True

    ox, oy = opponent["x"], opponent["y"]
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq > 1e-6:
        t = max(0.0, min(1.0, ((ox - x1) * dx + (oy - y1) * dy) / seg_len_sq))
        px = x1 + t * dx
        py = y1 + t * dy
        if (ox - px) ** 2 + (oy - py) ** 2 < (4.75 ** 2):
            return True

    return False


def tactical_action(observation: dict[str, Any], kick_power: int = 3) -> dict[str, Any]:
    player_id = observation["player_id"]
    opponent_id = observation["opponent_id"]
    state = observation["state"]
    field = state["field"]
    me = state["players"][player_id]
    opponent = state["players"][opponent_id]
    ball = state["ball"]
    attack = observation["attack_direction"]
    sign = 1 if attack == "UP" else -1

    if ball["possession"] == player_id:
        distance_to_goal = field["height"] - me["y"] if sign > 0 else me["y"]
        possession_steps = int(ball.get("possession_steps", 0))
        opponent_distance = math.hypot(opponent["x"] - me["x"], opponent["y"] - me["y"])
        opponent_ahead = (
            sign * (opponent["y"] - me["y"]) > 0
            and abs(opponent["x"] - me["x"]) < 12
            and opponent_distance < 50
        )
        side = "LEFT" if opponent["x"] >= me["x"] else "RIGHT"
        dribble = f"{attack}_{side}" if opponent_ahead else attack
        move = _safe_move(observation, dribble)

        goal_y = field["height"] if sign > 0 else 0.0
        obstacles = state.get("obstacles", [])

        if opponent["x"] >= field["width"] / 2:
            candidate_kicks = [f"{attack}_LEFT", attack, f"{attack}_RIGHT"]
        else:
            candidate_kicks = [f"{attack}_RIGHT", attack, f"{attack}_LEFT"]

        best_shot: str | None = None
        best_power: int = 3

        for cand in candidate_kicks:
            vx, vy = DIRECTION_VECTORS[cand]
            if vy == 0 or (vy > 0 and sign < 0) or (vy < 0 and sign > 0):
                continue

            t_goal = (goal_y - me["y"]) / vy
            if t_goal <= 0:
                continue

            x_hit = me["x"] + vx * t_goal
            if 34.0 <= x_hit <= 66.0:
                hit_x = x_hit
                hit_y = goal_y
                shot_distance = math.hypot(hit_x - me["x"], hit_y - me["y"])
                if shot_distance <= 96.0:
                    if not is_kick_path_blocked(me["x"], me["y"], hit_x, hit_y, obstacles, opponent):
                        best_shot = cand
                        best_power = 3 if shot_distance > 32.0 else 1
                        break

        if best_shot is not None:
            if distance_to_goal <= 65.0 or possession_steps >= 2:
                return {"move": move, "kick": {"direction": best_shot, "power": best_power}}

        if possession_steps < 3:
            return {"move": move}

        fallback_dir = best_shot
        if fallback_dir is None:
            safe_cands = []
            for cand in candidate_kicks:
                vx, vy = DIRECTION_VECTORS[cand]
                tx = me["x"] + vx * 96.0
                ty = me["y"] + vy * 96.0
                if not is_kick_path_blocked(me["x"], me["y"], tx, ty, obstacles, opponent):
                    safe_cands.append(cand)
            fallback_dir = safe_cands[0] if safe_cands else (dribble if opponent_ahead else attack)

        powers = observation["action_space"]["kick"]["power"]
        power = min(max(powers), max(min(powers), kick_power))
        return {"move": move, "kick": {"direction": fallback_dir, "power": power}}

    target_x, target_y = ball["x"], ball["y"]
    if ball["status"] == "moving":
        velocity = ball.get("velocity", {})
        target_x += 1.25 * float(velocity.get("x", 0.0))
        target_y += 1.25 * float(velocity.get("y", 0.0))
    elif ball["possession"] == opponent_id:
        defend_sign = -1 if attack == "UP" else 1
        target_y += defend_sign * 3.0
        target_x += -3.0 if opponent["x"] > field["width"] / 2 else 3.0
    preferred = direction_toward(target_x - me["x"], target_y - me["y"], dead_zone=0.6)
    return {"move": _safe_move(observation, preferred)}


class Policy:
    """Standalone inference policy for the sparse RL models made by train_bot.py."""

    def __init__(
        self,
        q_table: dict[str, list[float]] | None = None,
        visits: dict[str, int] | None = None,
        seed: int = 0,
        evaluation_epsilon: float = 0.03,
        safety_margin: float = 1.0,
        kick_power: int = 3,
    ) -> None:
        self.q_table = q_table or {}
        self.visits = visits or {}
        self.model_seed = seed
        self.evaluation_epsilon = evaluation_epsilon
        self.safety_margin = safety_margin
        self.kick_power = kick_power
        self.random = random.Random(seed)
        self.match_seeded = False

    @classmethod
    def load(cls, path: Path) -> "Policy":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("format") == 3 and raw.get("action_count") == ACTION_COUNT:
            q_table: dict[str, list[float]] = {}
            for key, sparse in raw.get("q_table", {}).items():
                values = [0.0] * ACTION_COUNT
                for index, value in sparse.items():
                    values[int(index)] = float(value)
                q_table[key] = values
            return cls(
                q_table=q_table,
                visits={key: int(value) for key, value in raw.get("visits", {}).items()},
                seed=int(raw.get("seed", 0)),
                evaluation_epsilon=float(raw.get("evaluation_epsilon", 0.03)),
                safety_margin=max(1.0, float(raw.get("safety_margin", 0.6))),
            )
        return cls(kick_power=int(raw.get("kick_power", 3)))

    @staticmethod
    def state_key(observation: dict[str, Any]) -> str:
        player_id = observation["player_id"]
        opponent_id = observation["opponent_id"]
        state = observation["state"]
        field = state["field"]
        me = state["players"][player_id]
        opponent = state["players"][opponent_id]
        ball = state["ball"]
        width, height = float(field["width"]), float(field["height"])
        sign = 1.0 if observation["attack_direction"] == "UP" else -1.0
        attack_y = me["y"] if sign > 0 else height - me["y"]
        ball_dx, ball_dy = ball["x"] - me["x"], sign * (ball["y"] - me["y"])
        opponent_dx = opponent["x"] - me["x"]
        opponent_dy = sign * (opponent["y"] - me["y"])
        velocity = ball.get("velocity", {})
        nearest = (999.0, 8)
        for obstacle in state.get("obstacles", []):
            dx = obstacle["x"] + obstacle["width"] / 2 - me["x"]
            dy = sign * (obstacle["y"] + obstacle["height"] / 2 - me["y"])
            distance = math.hypot(dx, dy)
            if distance < nearest[0]:
                nearest = (distance, _sector(dx, dy))
        possession = "S" if ball["possession"] == player_id else "O" if ball["possession"] == opponent_id else "F"
        values = (
            min(4, int(me["x"] / width * 5)),
            min(6, int(attack_y / height * 7)),
            _sector(ball_dx, ball_dy),
            _bucket(math.hypot(ball_dx, ball_dy), (7, 16, 32, 60)),
            _sector(opponent_dx, opponent_dy),
            _bucket(math.hypot(opponent_dx, opponent_dy), (8, 18, 38, 70)),
            possession,
            ball["status"][0].upper(),
            _sector(float(velocity.get("x", 0.0)), sign * float(velocity.get("y", 0.0))),
            _bucket(float(ball.get("remaining_kick_distance", 0.0)), (1, 25, 60)),
            _bucket(int(ball.get("possession_steps", 0)), (1, 3, 6)),
            nearest[1] if nearest[0] < 24 else 8,
            _bucket(nearest[0], (8, 16, 24)),
        )
        return "|".join(map(str, values))

    @staticmethod
    def action_from_index(index: int) -> dict[str, Any]:
        if index < len(MOVES):
            return {"move": MOVES[index]}
        kick_index = index - len(MOVES)
        direction = KICK_DIRECTIONS[kick_index // 3]
        return {"move": direction, "kick": {"direction": direction, "power": kick_index % 3 + 1}}

    @staticmethod
    def index_from_action(action: dict[str, Any]) -> int:
        kick = action.get("kick")
        if isinstance(kick, dict) and kick.get("direction") in KICK_DIRECTIONS:
            power = max(1, min(3, int(kick.get("power", 1))))
            return len(MOVES) + KICK_DIRECTIONS.index(kick["direction"]) * 3 + power - 1
        return MOVES.index(action.get("move", "STAY"))

    @staticmethod
    def valid_indices(observation: dict[str, Any]) -> list[int]:
        moves = [index for index in range(1, len(MOVES)) if _move_is_safe(observation, MOVES[index])]
        ball = observation["state"]["ball"]
        if ball["possession"] != observation["player_id"]:
            return moves or [0]
        attack = observation["attack_direction"]
        forward = [attack, f"{attack}_LEFT", f"{attack}_RIGHT"]
        if int(ball.get("possession_steps", 0)) < 2:
            controlled = [MOVES.index(move) for move in forward if _move_is_safe(observation, move)]
            return controlled or moves or [0]
        kicks = [
            len(MOVES) + KICK_DIRECTIONS.index(direction) * 3 + power
            for direction in forward
            for power in range(3)
        ]
        return moves + kicks

    def choose_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        if not self.match_seeded:
            state_seed = int(observation["state"].get("seed", 0))
            side_seed = 17 if observation["player_id"] == "player_1" else 31
            self.random.seed(self.model_seed ^ (state_seed * 1_000_003) ^ side_seed)
            self.match_seeded = True
        fallback_action = tactical_action(observation, self.kick_power)
        valid = self.valid_indices(observation)
        values = self.q_table.get(self.state_key(observation))
        if values is None or self.visits.get(self.state_key(observation), 0) < 2:
            return fallback_action
        if self.random.random() < self.evaluation_epsilon:
            return self.action_from_index(self.random.choice(valid))
        best = max(values[index] for index in valid)
        fallback = self.index_from_action(fallback_action)
        if fallback in valid and best - values[fallback] < self.safety_margin:
            return fallback_action
        choices = [index for index in valid if abs(values[index] - best) < 1e-9]
        return self.action_from_index(self.random.choice(choices))