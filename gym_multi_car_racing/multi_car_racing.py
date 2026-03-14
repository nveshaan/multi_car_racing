import math
import colorsys

import numpy as np

import gymnasium as gym
import gymnasium.envs.box2d.car_dynamics as car_dynamics
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` '
        'followed by `pip install "gymnasium[box2d]"`'
    ) from e

try:
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e

from shapely.geometry import Point, Polygon


STATE_W = 96
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0
TRACK_RAD = 900 / SCALE
PLAYFIELD = 2000 / SCALE
FPS = 50
ZOOM = 2.7
ZOOM_FOLLOW = True

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)
MAX_TRACK_GEN_ATTEMPTS = 50
MAX_DEST_SEARCH_ROUNDS = 64
MAX_BETA_ADJUST_ITERS = 128

ROAD_COLOR = np.array([102, 102, 102])
BG_COLOR = np.array([102, 204, 102])
GRASS_COLOR = np.array([102, 230, 102])

CAR_COLORS = [
    (0.8, 0.0, 0.0),
    (0.0, 0.0, 0.8),
    (0.0, 0.8, 0.0),
    (0.0, 0.8, 0.8),
    (0.8, 0.8, 0.8),
    (0.0, 0.0, 0.0),
    (0.8, 0.0, 0.8),
    (0.8, 0.8, 0.0),
]

EGO_COLOR = (0.85, 0.10, 0.10)
TEAMMATE_COLOR = (0.10, 0.80, 0.10)
OPPONENT_COLOR = (0.10, 0.10, 0.85)

LINE_SPACING = 5
LATERAL_SPACING = 3

BACKWARD_THRESHOLD = np.pi / 2
K_BACKWARD = 0


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return

        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited[obj.car_id]:
                tile.road_visited[obj.car_id] = True
                self.env.tile_visited_count[obj.car_id] += 1

                car_team = self.env.team_ids[obj.car_id]
                team_mask = self.env.team_ids == car_team
                team_size = int(team_mask.sum())
                num_teammates = max(0, team_size - 1)
                num_opponents = self.env.num_agents - team_size

                past_teammates = sum(
                    tile.road_visited[i]
                    for i in range(self.env.num_agents)
                    if team_mask[i]
                ) - 1
                past_opponents = sum(
                    tile.road_visited[i]
                    for i in range(self.env.num_agents)
                    if not team_mask[i]
                )

                teammate_term = 0.0
                if num_teammates > 0:
                    teammate_term = (
                        self.env.teammate_reward_scale
                        * (past_teammates / num_teammates)
                    )

                # Opponents reduce reward; teammate impact is configurable.
                if num_opponents > 0:
                    reward_factor = 1.0 - (past_opponents / num_opponents) + teammate_term
                else:
                    reward_factor = 1.0 + teammate_term
                reward_factor = max(0.0, reward_factor)
                self.env.reward[obj.car_id] += (
                    reward_factor * 1000.0 / len(self.env.track)
                )

                if (
                    tile.idx == 0
                    and self.env.tile_visited_count[obj.car_id] / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap[obj.car_id] = True
        else:
            obj.tiles.discard(tile)


class MultiCarRacing(gym.Env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "state_pixels"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        num_agents: int = 2,
        verbose: bool = False,
        direction: str = "CCW",
        use_random_direction: bool = True,
        backwards_flag: bool = True,
        h_ratio: float = 0.75,
        use_ego_color: bool = False,
        human_show_team_colors: bool = False,
        continuous: bool = True,
        discrete_actions: np.ndarray | None = None,
        render_mode: str | None = None,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        team_ids: list[int] | None = None,
        teammate_reward_scale: float = 0.0,
    ):
        if team_ids is None:
            team_ids = list(range(num_agents))
        if len(team_ids) != num_agents:
            raise ValueError(
                f"team_ids must have length num_agents ({num_agents}), got {len(team_ids)}"
            )
        EzPickle.__init__(
            self,
            num_agents,
            verbose,
            direction,
            use_random_direction,
            backwards_flag,
            h_ratio,
            use_ego_color,
            human_show_team_colors,
            continuous,
            discrete_actions,
            render_mode,
            lap_complete_percent,
            domain_randomize,
            team_ids,
            teammate_reward_scale,
        )

        self.num_agents = num_agents
        self.verbose = verbose
        self.continuous = continuous
        self.render_mode = render_mode
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self.use_random_direction = use_random_direction
        self.episode_direction = direction
        self.backwards_flag = backwards_flag
        self.h_ratio = h_ratio
        self.use_ego_color = use_ego_color
        self.human_show_team_colors = human_show_team_colors
        self.team_ids = np.array(team_ids, dtype=np.int32)
        self.teammate_reward_scale = float(teammate_reward_scale)
        self.team_color_map = self._build_team_color_map()

        self.np_random = np.random.default_rng()
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self, lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)

        self.screen = [None] * num_agents
        self.display_screen = None
        self.clock = None
        self.isopen = True

        self._grid_cols = None
        self._grid_rows = None
        self._grid_viewport_w = None
        self._grid_viewport_h = None

        self.road = None
        self.road_poly = []
        self.road_poly_shapely = []
        self.track = []
        self.cars = [None] * num_agents
        self.car_order = None

        self.reward = np.zeros(num_agents, dtype=np.float32)
        self.prev_reward = np.zeros(num_agents, dtype=np.float32)
        self.tile_visited_count = np.zeros(num_agents, dtype=np.int32)
        self.new_lap = np.zeros(num_agents, dtype=bool)
        self.driving_backward = np.zeros(num_agents, dtype=bool)
        self.driving_on_grass = np.zeros(num_agents, dtype=bool)

        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        if discrete_actions is None:
            discrete_actions = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.8],
                ],
                dtype=np.float32,
            )
        self.discrete_actions = np.asarray(discrete_actions, dtype=np.float32)
        if self.discrete_actions.ndim != 2 or self.discrete_actions.shape[1] != 3:
            raise ValueError("discrete_actions must have shape (n_actions, 3)")

        action_low = np.tile(np.array([-1.0, 0.0, 0.0], dtype=np.float32), num_agents)
        action_high = np.tile(np.array([1.0, 1.0, 1.0], dtype=np.float32), num_agents)
        if self.continuous:
            self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)
        else:
            n_actions = int(self.discrete_actions.shape[0])
            if self.num_agents == 1:
                self.action_space = spaces.Discrete(n_actions)
            else:
                self.action_space = spaces.MultiDiscrete(
                    np.full(self.num_agents, n_actions)
                )

        obs_shape = (STATE_H, STATE_W, 3)
        if self.num_agents > 1:
            obs_shape = (self.num_agents, STATE_H, STATE_W, 3)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )

    def _build_team_color_map(self):
        unique_teams = np.unique(self.team_ids)
        team_color_map = {}

        if len(unique_teams) <= len(CAR_COLORS):
            for idx, team_id in enumerate(unique_teams):
                team_color_map[int(team_id)] = CAR_COLORS[idx]
            return team_color_map

        for idx, team_id in enumerate(unique_teams):
            hue = idx / len(unique_teams)
            team_color_map[int(team_id)] = colorsys.hsv_to_rgb(hue, 0.8, 0.85)
        return team_color_map

    def _init_colors(self):
        if self.domain_randomize:
            self.road_color = self.np_random.uniform(0, 210, size=3)
            self.bg_color = self.np_random.uniform(0, 210, size=3)
            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] = min(self.grass_color[idx] + 20, 255)
        else:
            self.road_color = ROAD_COLOR.copy()
            self.bg_color = BG_COLOR.copy()
            self.grass_color = GRASS_COLOR.copy()

    def _reinit_colors(self, randomize: bool):
        if not self.domain_randomize:
            self._init_colors()
            return

        if randomize:
            self.road_color = self.np_random.uniform(0, 210, size=3)
            self.bg_color = self.np_random.uniform(0, 210, size=3)
            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] = min(self.grass_color[idx] + 20, 255)

    def _destroy(self):
        if not self.road:
            return
        for tile in self.road:
            self.world.DestroyBody(tile)
        self.road = []

        for car in self.cars:
            if car is not None:
                car.destroy()
        self.cars = [None] * self.num_agents

    def _format_observation(self, observation):
        observation = np.asarray(observation, dtype=self.observation_space.dtype)
        if observation.shape != self.observation_space.shape:
            raise ValueError(
                f"Observation shape {observation.shape} does not match observation "
                f"space {self.observation_space.shape}"
            )
        return observation

    def _create_track(self):
        checkpoints_count = 12
        checkpoints = []
        for checkpoint_id in range(checkpoints_count):
            noise = self.np_random.uniform(0, 2 * math.pi / checkpoints_count)
            alpha = 2 * math.pi * checkpoint_id / checkpoints_count + noise
            radius = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if checkpoint_id == 0:
                alpha = 0
                radius = 1.5 * TRACK_RAD
            if checkpoint_id == checkpoints_count - 1:
                alpha = 2 * math.pi * checkpoint_id / checkpoints_count
                self.start_alpha = 2 * math.pi * (-0.5) / checkpoints_count
                radius = 1.5 * TRACK_RAD

            checkpoints.append(
                (alpha, radius * math.cos(alpha), radius * math.sin(alpha))
            )

        self.road = []
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False

        while True:
            if not np.isfinite([x, y, beta]).all():
                return False
            alpha = math.atan2(y, x)
            if not np.isfinite(alpha):
                return False

            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            dest_search_rounds = 0
            while True:
                dest_search_rounds += 1
                if dest_search_rounds > MAX_DEST_SEARCH_ROUNDS:
                    return False

                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x
            dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy
            if not np.isfinite(proj) or not np.isfinite(beta - alpha):
                return False

            beta_adjust_iters = 0
            while beta - alpha > 1.5 * math.pi:
                beta_adjust_iters += 1
                if beta_adjust_iters > MAX_BETA_ADJUST_ITERS:
                    return False
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta_adjust_iters += 1
                if beta_adjust_iters > MAX_BETA_ADJUST_ITERS:
                    return False
                beta += 2 * math.pi

            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))

            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))

            if laps > 4:
                break

            no_freeze -= 1
            if no_freeze == 0:
                return False

        i1, i2 = -1, -1
        index = len(track)
        while True:
            index -= 1
            if index == 0:
                return False
            pass_through_start = (
                track[index][0] > self.start_alpha
                and track[index - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = index
            elif pass_through_start and i1 == -1:
                i1 = index
                break

        if self.verbose:
            print(f"Track generation: {i1}..{i2} -> {i2 - i1}-tiles track")

        track = track[i1 : i2 - 1]
        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        border = [False] * len(track)
        for index in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[index - neg - 0][1]
                beta2 = track[index - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[index] = good
        for index in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[index - neg] |= border[index]

        for index in range(len(track)):
            _, beta1, x1, y1 = track[index]
            _, beta2, x2, y2 = track[index - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            tile = self.world.CreateStaticBody(fixtures=self.fd_tile)
            tile.userData = tile
            color_shift = 0.01 * (index % 3) * 255
            tile.color = np.clip(self.road_color + color_shift, 0, 255)
            tile.road_visited = [False] * self.num_agents
            tile.road_friction = 1.0
            tile.idx = index
            tile.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], tile.color))
            self.road.append(tile)

            if border[index]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if index % 2 == 0 else (255, 0, 0),
                    )
                )

        self.track = track
        self.road_poly_shapely = [Polygon(poly) for poly, _ in self.road_poly]
        return True

    def _spawn_cars(self):
        _, pos_x, pos_y = self.track[0][1:4]
        ids = list(range(self.num_agents))
        shuffled_ids = self.np_random.choice(ids, size=self.num_agents, replace=False)
        self.car_order = {car_id: shuffled_ids[car_id] for car_id in ids}

        for car_id in range(self.num_agents):
            line_number = math.floor(self.car_order[car_id] / 2)
            side = (2 * (self.car_order[car_id] % 2)) - 1

            dx = self.track[-line_number * LINE_SPACING][2] - pos_x
            dy = self.track[-line_number * LINE_SPACING][3] - pos_y

            angle = self.track[-line_number * LINE_SPACING][1]
            if self.episode_direction == "CW":
                angle -= np.pi

            norm_theta = angle - np.pi / 2
            new_x = pos_x + dx + (LATERAL_SPACING * np.sin(norm_theta) * side)
            new_y = pos_y + dy + (LATERAL_SPACING * np.cos(norm_theta) * side)

            car = car_dynamics.Car(self.world, angle, new_x, new_y)
            car_team = int(self.team_ids[car_id])
            car.hull.color = self.team_color_map[car_team]
            self.cars[car_id] = car

            for wheel in car.wheels:
                wheel.car_id = car_id

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround

        self.reward = np.zeros(self.num_agents, dtype=np.float32)
        self.prev_reward = np.zeros(self.num_agents, dtype=np.float32)
        self.tile_visited_count = np.zeros(self.num_agents, dtype=np.int32)
        self.new_lap = np.zeros(self.num_agents, dtype=bool)
        self.driving_backward = np.zeros(self.num_agents, dtype=bool)
        self.driving_on_grass = np.zeros(self.num_agents, dtype=bool)
        self.t = 0.0
        self.isopen = True
        self.road_poly = []
        self.road_poly_shapely = []

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict) and "randomize" in options:
                randomize = bool(options["randomize"])
            self._reinit_colors(randomize)
        else:
            self._init_colors()

        if self.use_random_direction:
            self.episode_direction = self.np_random.choice(["CW", "CCW"])

        success = False
        for attempt in range(1, MAX_TRACK_GEN_ATTEMPTS + 1):
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(f"retry to generate track ({attempt}/{MAX_TRACK_GEN_ATTEMPTS})")

        if not success:
            raise RuntimeError(
                "Track generation failed repeatedly. "
                f"Tried {MAX_TRACK_GEN_ATTEMPTS} attempts without success."
            )

        self._spawn_cars()
        observation = self._format_observation(self.step(None)[0])
        return observation, {}

    def _decode_action(self, action):
        if self.continuous:
            action = np.asarray(action, dtype=np.float32)
            if action.size != self.num_agents * 3:
                raise InvalidAction(
                    f"Continuous action must contain {self.num_agents * 3} values, got {action.size}"
                )
            return np.reshape(action, (self.num_agents, 3))

        if self.num_agents == 1:
            action_scalar = int(np.asarray(action).item())
            if not self.action_space.contains(action_scalar):
                raise InvalidAction(
                    f"you passed the invalid action `{action}`. "
                    f"The supported action_space is `{self.action_space}`"
                )
            action_indices = np.array([action_scalar], dtype=np.int64)
        else:
            action_indices = np.asarray(action, dtype=np.int64).reshape(self.num_agents)
            if not self.action_space.contains(action_indices):
                raise InvalidAction(
                    f"you passed the invalid action `{action}`. "
                    f"The supported action_space is `{self.action_space}`"
                )
        return self.discrete_actions[action_indices]

    def _update_driving_flags(self, step_reward):
        track_xy = np.array(self.track)[:, 2:]
        for car_id, car in enumerate(self.cars):
            velocity = car.hull.linearVelocity
            if np.linalg.norm(velocity) > 0.5:
                car_angle = -math.atan2(velocity[0], velocity[1])
            else:
                car_angle = car.hull.angle
            car_angle = (car_angle + 2 * np.pi) % (2 * np.pi)

            car_pos = np.array(car.hull.position).reshape((1, 2))
            car_point = Point((float(car_pos[0, 0]), float(car_pos[0, 1])))

            distance_to_tiles = np.linalg.norm(car_pos - track_xy, ord=2, axis=1)
            track_index = int(np.argmin(distance_to_tiles))

            on_grass = not np.array(
                [car_point.within(polygon) for polygon in self.road_poly_shapely]
            ).any()
            self.driving_on_grass[car_id] = on_grass

            desired_angle = self.track[track_index][1]
            if self.episode_direction == "CW":
                desired_angle += np.pi
            desired_angle = (desired_angle + 2 * np.pi) % (2 * np.pi)

            angle_diff = abs(desired_angle - car_angle)
            if angle_diff > np.pi:
                angle_diff = abs(angle_diff - 2 * np.pi)

            if angle_diff > BACKWARD_THRESHOLD:
                self.driving_backward[car_id] = True
                step_reward[car_id] -= K_BACKWARD * angle_diff
            else:
                self.driving_backward[car_id] = False

    def step(self, action):
        if action is not None:
            decoded_action = self._decode_action(action)
            for car_id, car in enumerate(self.cars):
                car.steer(-float(decoded_action[car_id][0]))
                car.gas(float(decoded_action[car_id][1]))
                car.brake(float(decoded_action[car_id][2]))

        for car in self.cars:
            car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render_frames("state_pixels")

        step_reward = np.zeros(self.num_agents, dtype=np.float32)
        terminated = False
        truncated = False
        info = {}

        if action is not None:
            self.reward -= 0.1
            for car in self.cars:
                car.fuel_spent = 0.0

            step_reward = self.reward - self.prev_reward
            self._update_driving_flags(step_reward)
            self.prev_reward = self.reward.copy()

            lap_finished_agents = np.logical_or(
                self.new_lap,
                np.equal(self.tile_visited_count, len(self.track)),
            )
            if np.any(lap_finished_agents):
                terminated = True
                info["lap_finished"] = True
                info["lap_finished_agents"] = lap_finished_agents.copy()
                info["winner"] = int(np.flatnonzero(lap_finished_agents)[0])

            out_of_bounds_agents = np.zeros(self.num_agents, dtype=bool)
            for car_id, car in enumerate(self.cars):
                x, y = car.hull.position
                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                    terminated = True
                    step_reward[car_id] = -100.0
                    out_of_bounds_agents[car_id] = True

            if np.any(out_of_bounds_agents):
                info["lap_finished"] = False
                info["out_of_bounds_agents"] = out_of_bounds_agents.copy()

            unique_teams = np.unique(self.team_ids)
            info["team_rewards"] = {
                int(t): float(step_reward[self.team_ids == t].sum())
                for t in unique_teams
            }

        if not self.isopen:
            terminated = True

        observation = self._format_observation(self.state)
        if self.num_agents == 1:
            reward = float(step_reward[0])
        else:
            reward = step_reward.astype(np.float32)

        return observation, reward, terminated, truncated, info

    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode or "rgb_array"
        assert mode in self.metadata["render_modes"]

        if mode == "human":
            for car_id in range(self.num_agents):
                self._render_car_view(car_id, "human")
            self._render_split_screen()
            return None

        return self._render_frames(mode)

    def _render_frames(self, mode: str):
        assert mode in ["state_pixels", "rgb_array"]
        frames = [self._render_car_view(car_id, mode) for car_id in range(self.num_agents)]
        frames = np.stack(frames, axis=0)
        if self.num_agents == 1:
            return frames[0]
        return frames

    def _render_car_view(self, car_id: int, mode: str):
        assert mode in self.metadata["render_modes"]

        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return None

        surf = pygame.Surface((WINDOW_W, WINDOW_H))

        focus_car = self.cars[car_id]
        angle = -focus_car.hull.angle
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(focus_car.hull.position[0]) * zoom
        scroll_y = -(focus_car.hull.position[1]) * zoom
        translation = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        translation = (
            WINDOW_W / 2 + translation[0],
            WINDOW_H * (1.0 - self.h_ratio) + translation[1],
        )

        self._render_road(surf, zoom, translation, angle)

        original_colors = None
        apply_role_colors = self.use_ego_color and not (
            mode == "human" and self.human_show_team_colors
        )
        if apply_role_colors:
            original_colors = [tuple(car.hull.color) for car in self.cars]
            ego_team = int(self.team_ids[car_id])
            for idx, car in enumerate(self.cars):
                if idx == car_id:
                    car.hull.color = EGO_COLOR
                elif int(self.team_ids[idx]) == ego_team:
                    car.hull.color = TEAMMATE_COLOR
                else:
                    car.hull.color = OPPONENT_COLOR

        for car in self.cars:
            car.draw(
                surf,
                zoom,
                translation,
                angle,
                mode not in ["state_pixels_list", "state_pixels"],
            )

        if original_colors is not None:
            for car, color in zip(self.cars, original_colors):
                car.hull.color = color

        surf = pygame.transform.flip(surf, False, True)
        self._render_indicators(surf, car_id, WINDOW_W, WINDOW_H)

        if mode == "human":
            self.screen[car_id] = surf
            return self.isopen
        if mode == "rgb_array":
            return self._create_image_array(surf, (VIDEO_W, VIDEO_H))
        if mode == "state_pixels":
            return self._create_image_array(surf, (STATE_W, STATE_H))
        return self.isopen

    def _render_split_screen(self):
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        if self._grid_cols is None or self.display_screen is None:
            if pygame.display.get_surface() is None:
                pygame.display.init()

            try:
                display_info = pygame.display.Info()
                current_w = getattr(display_info, "current_w", 0) or 0
                current_h = getattr(display_info, "current_h", 0) or 0
            except pygame.error:
                current_w = 0
                current_h = 0

            max_display_w = current_w - 50 if current_w > 100 else 1920
            max_display_h = current_h - 100 if current_h > 100 else 1080
            preferred_scale = 0.6
            preferred_viewport_w = int(WINDOW_W * preferred_scale)
            preferred_viewport_h = int(WINDOW_H * preferred_scale)

            max_cols = (
                max(1, max_display_w // preferred_viewport_w)
                if preferred_viewport_w > 0
                else 1
            )
            self._grid_cols = min(self.num_agents, max_cols)
            self._grid_rows = math.ceil(self.num_agents / self._grid_cols)

            viewport_w = preferred_viewport_w
            viewport_h = preferred_viewport_h
            total_w = viewport_w * self._grid_cols
            total_h = viewport_h * self._grid_rows
            if total_w > max_display_w or total_h > max_display_h:
                scale_w = max_display_w / total_w if total_w > 0 else 1.0
                scale_h = max_display_h / total_h if total_h > 0 else 1.0
                scale_factor = min(scale_w, scale_h)
                viewport_w = int(viewport_w * scale_factor)
                viewport_h = int(viewport_h * scale_factor)
                total_w = viewport_w * self._grid_cols
                total_h = viewport_h * self._grid_rows

            total_w = max(total_w, 100)
            total_h = max(total_h, 100)
            self._grid_viewport_w = max(viewport_w, 50)
            self._grid_viewport_h = max(viewport_h, 50)

            pygame.display.set_caption(f"Multi-Car Racing - {self.num_agents} Players")
            self.display_screen = pygame.display.set_mode((total_w, total_h))

        self.display_screen.fill((30, 30, 30))

        cols = self._grid_cols
        viewport_w = self._grid_viewport_w
        viewport_h = self._grid_viewport_h
        for car_id in range(self.num_agents):
            if self.screen[car_id] is None:
                continue

            row = car_id // cols
            col = car_id % cols
            scaled_surf = pygame.transform.scale(
                self.screen[car_id], (viewport_w, viewport_h)
            )
            x_pos = col * viewport_w
            y_pos = row * viewport_h
            self.display_screen.blit(scaled_surf, (x_pos, y_pos))

            if col > 0:
                pygame.draw.line(
                    self.display_screen,
                    (255, 255, 255),
                    (x_pos, y_pos),
                    (x_pos, y_pos + viewport_h),
                    2,
                )
            if row > 0:
                pygame.draw.line(
                    self.display_screen,
                    (255, 255, 255),
                    (x_pos, y_pos),
                    (x_pos + viewport_w, y_pos),
                    2,
                )

            font = pygame.font.Font(None, 36)
            label = font.render(f"Player {car_id + 1}", True, (255, 255, 255))
            self.display_screen.blit(label, (x_pos + 10, y_pos + 10))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit(0)

        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(self.metadata["render_fps"])

    def _render_road(self, surface, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]
        self._draw_colored_polygon(
            surface, field, self.bg_color, zoom, translation, angle, clip=False
        )

        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                poly = [
                    (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                    (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                    (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                    (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                ]
                self._draw_colored_polygon(
                    surface, poly, self.grass_color, zoom, translation, angle
                )

        for poly, color in self.road_poly:
            self._draw_colored_polygon(
                surface,
                [(point[0], point[1]) for point in poly],
                [int(channel) for channel in color],
                zoom,
                translation,
                angle,
            )

    def _render_indicators(self, surface, agent_id: int, width: int, height: int):
        scale_x = width / 40.0
        scale_y = height / 40.0
        pygame.draw.rect(surface, (0, 0, 0), (0, height - 5 * scale_y, width, 5 * scale_y))

        def vertical_ind(place, val, color):
            val = max(0, min(val, 4))
            rect = pygame.Rect(
                int(place * scale_x),
                int(height - scale_y - scale_y * val),
                int(scale_x),
                int(scale_y * val),
            )
            pygame.draw.rect(surface, color, rect)

        def horiz_ind(place, val, color):
            x_pos = int(place * scale_x)
            rect_width = int(val * scale_x)
            if rect_width < 0:
                x_pos += rect_width
                rect_width = -rect_width
            rect = pygame.Rect(x_pos, int(height - 4 * scale_y), rect_width, int(2 * scale_y))
            pygame.draw.rect(surface, color, rect)

        car = self.cars[agent_id]
        true_speed = np.sqrt(
            np.square(car.hull.linearVelocity[0])
            + np.square(car.hull.linearVelocity[1])
        )

        vertical_ind(5, 0.02 * true_speed, (255, 255, 255))
        vertical_ind(7, 0.01 * car.wheels[0].omega, (0, 0, 255))
        vertical_ind(8, 0.01 * car.wheels[1].omega, (0, 0, 255))
        vertical_ind(9, 0.01 * car.wheels[2].omega, (51, 0, 255))
        vertical_ind(10, 0.01 * car.wheels[3].omega, (51, 0, 255))
        horiz_ind(20, -10.0 * car.wheels[0].joint.angle, (0, 255, 0))
        horiz_ind(30, -0.8 * car.hull.angularVelocity, (255, 0, 0))

        font = pygame.font.Font(None, 48)
        score_text = font.render(f"{int(self.reward[agent_id]):04d}", True, (255, 255, 255))
        surface.blit(score_text, (20, int(height - 2.5 * scale_y - 18)))

        if self.driving_backward[agent_id] and self.backwards_flag:
            pygame.draw.polygon(
                surface,
                (0, 0, 255),
                [(width - 100, height - 30), (width - 75, height - 70), (width - 50, height - 30)],
            )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        rotated = [pygame.math.Vector2(point).rotate_rad(angle) for point in poly]
        transformed = [
            (point[0] * zoom + translation[0], point[1] * zoom + translation[1])
            for point in rotated
        ]
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in transformed
        ):
            gfxdraw.aapolygon(surface, transformed, color)
            gfxdraw.filled_polygon(surface, transformed, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.display_screen is not None or any(screen is not None for screen in self.screen):
            pygame.display.quit()
            pygame.quit()

        self.screen = [None] * self.num_agents
        self.display_screen = None
        self._grid_cols = None
        self._grid_rows = None
        self._grid_viewport_w = None
        self._grid_viewport_h = None
        self.isopen = False


if __name__ == "__main__":
    num_cars = 2
    car_control_keys = [
        [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN],
        [pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_s],
    ]

    actions = np.zeros((num_cars, 3), dtype=np.float32)
    env = MultiCarRacing(num_cars, render_mode="human")
    observation, info = env.reset()

    running = True
    restart = False
    total_reward = np.zeros(num_cars, dtype=np.float32)
    steps = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_RETURN:
                    restart = True

        keys = pygame.key.get_pressed()
        for car_id in range(min(len(car_control_keys), num_cars)):
            if keys[car_control_keys[car_id][0]]:
                actions[car_id][0] = -1.0
            elif keys[car_control_keys[car_id][1]]:
                actions[car_id][0] = 1.0
            else:
                actions[car_id][0] = 0.0

            actions[car_id][1] = 1.0 if keys[car_control_keys[car_id][2]] else 0.0
            actions[car_id][2] = 0.8 if keys[car_control_keys[car_id][3]] else 0.0

        observation, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        if steps % 200 == 0 or done:
            print("\nActions: " + " ".join([f"Car {idx}: {actions[idx]}" for idx in range(num_cars)]))
            print(f"Step {steps} Total_reward {total_reward}")

        env.render()

        if done or restart:
            observation, info = env.reset()
            total_reward = np.zeros(num_cars, dtype=np.float32)
            steps = 0
            restart = False

    env.close()
