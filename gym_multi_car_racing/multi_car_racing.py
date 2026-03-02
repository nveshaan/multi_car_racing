import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gymnasium as gym
import gymnasium.envs.box2d.car_dynamics as car_dynamics
from gymnasium import spaces
from gymnasium.utils import colorize, seeding, EzPickle

import pygame
from shapely.geometry import Point, Polygon

# Easiest continuous control task to learn from pixels, a top-down racing environment.
# Discrete control is reasonable in this environment as well, on/off discretization is
# fine.
#
# State consists of STATE_W x STATE_H pixels.
#
# Reward is -0.1 every frame and +1000/N for every track tile visited, where N is
# the total number of tiles visited in the track. For example, if you have finished in 732 frames,
# your reward is 1000 - 0.1*732 = 926.8 points.
#
# Game is solved when agent consistently gets 900+ points. Track generated is random every episode.
#
# Episode finishes when all tiles are visited. Car also can go outside of PLAYFIELD, that
# is far off the track, then it will get -100 and die.
#
# Some indicators shown at the bottom of the window and the state RGB buffer. From
# left to right: true speed, four ABS sensors, steering wheel position and gyroscope.
#
# To play yourself (it's rather fast for humans), type:
#
# python gym/envs/box2d/car_racing.py
#
# Remember it's powerful rear-wheel drive car, don't press accelerator and turn at the
# same time.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

STATE_W = 96   # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE       = 6.0        # Track scale
TRACK_RAD   = 900/SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD   = 2000/SCALE # Game over boundary
FPS         = 50         # Frames per second
ZOOM        = 2.7        # Camera zoom
ZOOM_FOLLOW = True       # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40/SCALE
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]

# Specify different car colors
CAR_COLORS = [(0.8, 0.0, 0.0), (0.0, 0.0, 0.8),
              (0.0, 0.8, 0.0), (0.0, 0.8, 0.8),
              (0.8, 0.8, 0.8), (0.0, 0.0, 0.0),
              (0.8, 0.0, 0.8), (0.8, 0.8, 0.0)]

# Distance between cars
LINE_SPACING = 5     # Starting distance between each pair of cars
LATERAL_SPACING = 3  # Starting side distance between pairs of cars

# Penalizing backwards driving
BACKWARD_THRESHOLD = np.pi/2
K_BACKWARD = 0  # Penalty weight: backwards_penalty = K_BACKWARD * angle_diff  (if angle_diff > BACKWARD_THRESHOLD)

class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
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
            obj  = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj  = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]

        # This check seems to implicitly make sure that we only look at wheels as the tiles 
        # attribute is only set for wheels in car_dynamics.py.
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            # print tile.road_friction, "ADD", len(obj.tiles)
            if not tile.road_visited[obj.car_id]:
                tile.road_visited[obj.car_id] = True
                self.env.tile_visited_count[obj.car_id] += 1

                # The reward is dampened on tiles that have been visited already.
                past_visitors = sum(tile.road_visited)-1
                reward_factor = 1 - (past_visitors / self.env.num_agents)
                self.env.reward[obj.car_id] += reward_factor * 1000.0/len(self.env.track)
        else:
            obj.tiles.remove(tile)
            # print tile.road_friction, "DEL", len(obj.tiles) -- should delete to zero when on grass (this works)

class MultiCarRacing(gym.Env, EzPickle):
    metadata = {
        'render_modes': ['human', 'rgb_array', 'state_pixels'],
        'render_fps': FPS
    }

    def __init__(self, num_agents=2, verbose=1, direction='CCW',
                 use_random_direction=True, backwards_flag=True, h_ratio=0.25,
                 use_ego_color=False, render_mode=None):
        EzPickle.__init__(self)
        self.np_random = None  # Will be set in reset()
        self.num_agents = num_agents
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0,0), contactListener=self.contactListener_keepref)
        self.screen = [None] * num_agents  # Pygame surfaces for each car's view
        self.display_screen = None  # Main pygame display window (for split-screen)
        self.clock = None  # Pygame clock for FPS control
        self.render_mode = render_mode
        # Cached grid layout to prevent continuous resizing
        self._grid_cols = None
        self._grid_rows = None
        self._grid_viewport_w = None
        self._grid_viewport_h = None
        self.road = None
        self.cars = [None] * num_agents
        self.car_order = None  # Determines starting positions of cars
        self.reward = np.zeros(num_agents)
        self.prev_reward = np.zeros(num_agents)
        self.tile_visited_count = [0]*num_agents
        self.verbose = verbose
        self.fd_tile = fixtureDef(
                shape = polygonShape(vertices=
                    [(0, 0),(1, 0),(1, -1),(0, -1)]))
        self.driving_backward = np.zeros(num_agents, dtype=bool)
        self.driving_on_grass = np.zeros(num_agents, dtype=bool)
        self.use_random_direction = use_random_direction  # Whether to select direction randomly
        self.episode_direction = direction  # Choose 'CCW' (default) or 'CW' (flipped)
        # Note: if use_random_direction is True, direction is set in reset()
        self.backwards_flag = backwards_flag  # Boolean for rendering backwards driving flag
        self.h_ratio = h_ratio  # Configures vertical location of car within rendered window
        self.use_ego_color = use_ego_color  # Whether to make ego car always render as the same color
        
        self.surf = None  # Main rendering surface for pygame
        self.isopen = True  # Track if window is open

        self.action_lb = np.tile(np.array([-1,+0,+0]), 1)  # self.num_agents)
        self.action_ub = np.tile(np.array([+1,+1,+1]), 1)  # self.num_agents)

        self.action_space = spaces.Box( self.action_lb, self.action_ub, dtype=np.float32)  # (steer, gas, brake) x N
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []

        for car in self.cars:
            car.destroy()

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2*math.pi*c/CHECKPOINTS + self.np_random.uniform(0, 2*math.pi*1/CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD/3, TRACK_RAD)
            if c==0:
                alpha = 0
                rad = 1.5*TRACK_RAD
            if c==CHECKPOINTS-1:
                alpha = 2*math.pi*c/CHECKPOINTS
                self.start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
                rad = 1.5*TRACK_RAD
            checkpoints.append( (alpha, rad*math.cos(alpha), rad*math.sin(alpha)) )

        # print "\n".join(str(h) for h in checkpoints)
        # self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5*TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2*math.pi
            while True: # Find destination from checkpoints
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
                alpha -= 2*math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
            while beta - alpha >  1.5*math.pi:
                 beta -= 2*math.pi
            while beta - alpha < -1.5*math.pi:
                 beta += 2*math.pi
            prev_beta = beta
            proj *= SCALE
            if proj >  0.3:
                 beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
            if proj < -0.3:
                 beta += min(TRACK_TURN_RATE, abs(0.001*proj))
            x += p1x*TRACK_DETAIL_STEP
            y += p1y*TRACK_DETAIL_STEP
            track.append( (alpha,prev_beta*0.5 + beta*0.5,x,y) )
            if laps > 4:
                 break
            no_freeze -= 1
            if no_freeze==0:
                 break
        # print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i==0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha \
                and track[i-1][0] <= self.start_alpha
            if pass_through_start and i2==-1:
                i2 = i
            elif pass_through_start and i1==-1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
        assert i1!=-1
        assert i2!=-1

        track = track[i1:i2-1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square( first_perp_x*(track[0][2] - track[-1][2]) ) +
            np.square( first_perp_y*(track[0][3] - track[-1][3]) ))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False]*len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i-neg-0][1]
                beta2 = track[i-neg-1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE*0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i-neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            _, beta1, x1, y1 = track[i]
            _, beta2, x2, y2 = track[i-1]
            road1_l = (x1 - TRACK_WIDTH*math.cos(beta1), y1 - TRACK_WIDTH*math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH*math.cos(beta1), y1 + TRACK_WIDTH*math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH*math.cos(beta2), y2 - TRACK_WIDTH*math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH*math.cos(beta2), y2 + TRACK_WIDTH*math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01*(i%3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = [False]*self.num_agents
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(( [road1_l, road1_r, road2_r, road2_l], t.color ))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side* TRACK_WIDTH        *math.cos(beta1), y1 + side* TRACK_WIDTH        *math.sin(beta1))
                b1_r = (x1 + side*(TRACK_WIDTH+BORDER)*math.cos(beta1), y1 + side*(TRACK_WIDTH+BORDER)*math.sin(beta1))
                b2_l = (x2 + side* TRACK_WIDTH        *math.cos(beta2), y2 + side* TRACK_WIDTH        *math.sin(beta2))
                b2_r = (x2 + side*(TRACK_WIDTH+BORDER)*math.cos(beta2), y2 + side*(TRACK_WIDTH+BORDER)*math.sin(beta2))
                self.road_poly.append(( [b1_l, b1_r, b2_r, b2_l], (1,1,1) if i%2==0 else (1,0,0) ))
        self.track = track
        self.road_poly_shapely = [Polygon(self.road_poly[i][0]) for i in
                                  range(len(self.road_poly))]
        return True

    def reset(self, seed=None, options=None):
        # Handle seeding (Gymnasium API)
        if seed is not None:
            self.np_random, seed = seeding.np_random(seed)
        elif self.np_random is None:
            self.np_random, seed = seeding.np_random(None)
        
        self._destroy()
        self.reward = np.zeros(self.num_agents)
        self.prev_reward = np.zeros(self.num_agents)
        self.tile_visited_count = [0]*self.num_agents
        self.t = 0.0
        self.road_poly = []

        # Reset driving backwards/on-grass states and track direction
        self.driving_backward = np.zeros(self.num_agents, dtype=bool)
        self.driving_on_grass = np.zeros(self.num_agents, dtype=bool)
        if self.use_random_direction:  # Choose direction randomly
            self.episode_direction = self.np_random.choice(['CW', 'CCW'])

        # Set positions of cars randomly
        ids = [i for i in range(self.num_agents)]
        shuffle_ids = self.np_random.choice(ids, size=self.num_agents, replace=False)
        self.car_order = {i: shuffle_ids[i] for i in range(self.num_agents)}

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print("retry to generate track (normal if there are not many of this messages)")

        (angle, pos_x, pos_y) = self.track[0][1:4]
        car_width = car_dynamics.SIZE * (car_dynamics.WHEEL_W * 2 \
            + (car_dynamics.WHEELPOS[1][0]-car_dynamics.WHEELPOS[1][0]))
        for car_id in range(self.num_agents):

            # Specify line and lateral separation between cars
            line_spacing = LINE_SPACING
            lateral_spacing = LATERAL_SPACING

            #index into positions using modulo and pairs
            line_number = math.floor(self.car_order[car_id] / 2)  # Starts at 0
            side = (2 * (self.car_order[car_id] % 2)) - 1  # either {-1, 1}

            # Compute offsets from start (this should be zero for first pair of cars)
            dx = self.track[-line_number * line_spacing][2] - pos_x  # x offset
            dy = self.track[-line_number * line_spacing][3] - pos_y  # y offset

            # Compute angle based off of track index for car
            angle = self.track[-line_number * line_spacing][1]
            if self.episode_direction == 'CW':  # CW direction indicates reversed
                angle -= np.pi  # Flip direction is either 0 or pi

            # Compute offset angle (normal to angle of track)
            norm_theta = angle - np.pi/2

            # Compute offsets from position of original starting line
            new_x = pos_x + dx + (lateral_spacing * np.sin(norm_theta) * side)
            new_y = pos_y + dy + (lateral_spacing * np.cos(norm_theta) * side)

            # Display spawn locations of cars.
            # print(f"Spawning car {car_id} at {new_x:.0f}x{new_y:.0f} with "
            #       f"orientation {angle}")

            # Create car at location with given angle
            self.cars[car_id] = car_dynamics.Car(self.world, angle, new_x,
                                                 new_y)
            self.cars[car_id].hull.color = CAR_COLORS[car_id % len(CAR_COLORS)]

            # This will be used to identify the car that touches a particular tile.
            for wheel in self.cars[car_id].wheels:
                wheel.car_id = car_id

        # Get initial observation via step with no action
        obs = self.step(None)[0]
        info = {}
        return obs, info

    def step(self, action):
        """ Run environment for one timestep. 
        
        Parameters:
            action(np.ndarray): Numpy array of shape (num_agents,3) containing the
                commands for each car. Each command is of the shape (steer, gas, brake).
        """

        if action is not None:
            # NOTE: re-shape action as input action is flattened
            action = np.reshape(action, (self.num_agents, -1))
            for car_id, car in enumerate(self.cars):
                car.steer(-action[car_id][0])
                car.gas(action[car_id][1])
                car.brake(action[car_id][2])

        for car in self.cars:
            car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        self.state = self.render("state_pixels")

        step_reward = np.zeros(self.num_agents)
        done = False
        if action is not None: # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER

            # NOTE(IG): Probably not relevant. Seems not to be used anywhere. Commented it out.
            # self.cars[0].fuel_spent = 0.0

            step_reward = self.reward - self.prev_reward

            # Add penalty for driving backward
            for car_id, car in enumerate(self.cars):  # Enumerate through cars

                # Get car speed
                vel = car.hull.linearVelocity
                if np.linalg.norm(vel) > 0.5:  # If fast, compute angle with v
                    car_angle = -math.atan2(vel[0], vel[1])
                else:  # If slow, compute with hull
                    car_angle = car.hull.angle

                # Map angle to [0, 2pi] interval
                car_angle = (car_angle + (2 * np.pi)) % (2 * np.pi)

                # Retrieve car position
                car_pos = np.array(car.hull.position).reshape((1, 2))
                car_pos_as_point = Point((float(car_pos[0, 0]),
                                          float(car_pos[0, 1]))) # FIXME: 0 vs :


                # Compute closest point on track to car position (l2 norm)
                distance_to_tiles = np.linalg.norm(
                    car_pos - np.array(self.track)[:, 2:], ord=2, axis=1)
                track_index = np.argmin(distance_to_tiles)

                # Check if car is driving on grass by checking inside polygons
                on_grass = not np.array([car_pos_as_point.within(polygon)
                                   for polygon in self.road_poly_shapely]).any()
                self.driving_on_grass[car_id] = on_grass

                # Find track angle of closest point
                desired_angle = self.track[track_index][1]

                # If track direction reversed, reverse desired angle
                if self.episode_direction == 'CW':  # CW direction indicates reversed
                    desired_angle += np.pi

                # Map angle to [0, 2pi] interval
                desired_angle = (desired_angle + (2 * np.pi)) % (2 * np.pi)

                # Compute smallest angle difference between desired and car
                angle_diff = abs(desired_angle - car_angle)
                if angle_diff > np.pi:
                    angle_diff = abs(angle_diff - 2 * np.pi)

                # If car is driving backward and not on grass, penalize car. The
                # backwards flag is set even if it is driving on grass.
                if angle_diff > BACKWARD_THRESHOLD:
                    self.driving_backward[car_id] = True
                    step_reward[car_id] -= K_BACKWARD * angle_diff
                else:
                    self.driving_backward[car_id] = False

            self.prev_reward = self.reward.copy()
            if len(self.track) in self.tile_visited_count:
                done = True

            # The car that leaves the field experiences a reward of -100 
            # and the episode is terminated subsequently.
            for car_id, car in enumerate(self.cars):
                x, y = car.hull.position
                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                    done = True
                    step_reward[car_id] = -100

        return self.state, step_reward, done, False, {}

    def render(self, mode='human'):
        if mode is None:
            mode = self.render_mode
        assert mode in ['human', 'state_pixels', 'rgb_array']

        # Render each car's view to its own surface
        result = []
        for cur_car_id in range(self.num_agents):
            frame = self._render_window(cur_car_id, mode)
            # Only collect frames for array-returning modes
            if mode != 'human':
                result.append(frame)
        
        # For human mode, create a split-screen display and return None
        if mode == 'human':
            self._render_split_screen()
            return None
        
        return np.stack(result, axis=0)

    def _render_split_screen(self):
        """ Render all car views in a 2D grid, scaling to fit display if needed. """
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        
        # Only calculate grid layout once (or when display_screen is None)
        if self._grid_cols is None or self.display_screen is None:
            # Ensure display is initialized before getting display info
            if pygame.display.get_surface() is None:
                pygame.display.init()
            
            # Get display dimensions safely (handles headless / error cases)
            try:
                display_info = pygame.display.Info()
                current_w = getattr(display_info, "current_w", 0) or 0
                current_h = getattr(display_info, "current_h", 0) or 0
            except pygame.error:
                current_w = 0
                current_h = 0
            except Exception:
                current_w = 0
                current_h = 0
            
            # Use sensible defaults if display info returns invalid values
            max_display_w = current_w - 50 if current_w > 100 else 1920
            max_display_h = current_h - 100 if current_h > 100 else 1080
            
            # Preferred viewport scale
            preferred_scale = 0.6
            
            # Calculate optimal grid layout (cols x rows)
            # Try to fit as many columns as possible at preferred scale
            preferred_viewport_w = int(WINDOW_W * preferred_scale)
            preferred_viewport_h = int(WINDOW_H * preferred_scale)
            
            # Calculate how many columns fit at preferred scale
            max_cols = max(1, max_display_w // preferred_viewport_w) if preferred_viewport_w > 0 else 1
            self._grid_cols = min(self.num_agents, max_cols)
            self._grid_rows = math.ceil(self.num_agents / self._grid_cols)
            
            # Calculate viewport size that fits within display
            # First, try with preferred scale
            viewport_w = preferred_viewport_w
            viewport_h = preferred_viewport_h
            total_w = viewport_w * self._grid_cols
            total_h = viewport_h * self._grid_rows
            
            # If total size exceeds display, scale down while maintaining aspect ratio
            if total_w > max_display_w or total_h > max_display_h:
                # Calculate scale factors for width and height constraints
                scale_w = max_display_w / total_w if total_w > 0 else 1.0
                scale_h = max_display_h / total_h if total_h > 0 else 1.0
                scale_factor = min(scale_w, scale_h)
                
                viewport_w = int(viewport_w * scale_factor)
                viewport_h = int(viewport_h * scale_factor)
                total_w = viewport_w * self._grid_cols
                total_h = viewport_h * self._grid_rows
            
            # Ensure minimum size
            total_w = max(total_w, 100)
            total_h = max(total_h, 100)
            self._grid_viewport_w = max(viewport_w, 50)
            self._grid_viewport_h = max(viewport_h, 50)
            
            # Create the display window
            pygame.display.set_caption(f"Multi-Car Racing - {self.num_agents} Players")
            self.display_screen = pygame.display.set_mode((total_w, total_h))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Use cached values
        cols = self._grid_cols
        rows = self._grid_rows
        viewport_w = self._grid_viewport_w
        viewport_h = self._grid_viewport_h
        
        # Fill background
        self.display_screen.fill((30, 30, 30))
        
        # Blit each car's surface onto the main display in a grid
        for car_id in range(self.num_agents):
            if self.screen[car_id] is not None:
                # Calculate grid position (row, col)
                row = car_id // cols
                col = car_id % cols
                
                # Scale the car's surface to fit the viewport (maintains aspect ratio)
                scaled_surf = pygame.transform.scale(self.screen[car_id], (viewport_w, viewport_h))
                
                # Position in grid
                x_pos = col * viewport_w
                y_pos = row * viewport_h
                self.display_screen.blit(scaled_surf, (x_pos, y_pos))
                
                # Draw divider lines
                # Vertical divider (left edge of viewport, except first column)
                if col > 0:
                    pygame.draw.line(self.display_screen, (255, 255, 255), 
                                     (x_pos, y_pos), (x_pos, y_pos + viewport_h), 2)
                # Horizontal divider (top edge of viewport, except first row)
                if row > 0:
                    pygame.draw.line(self.display_screen, (255, 255, 255), 
                                     (x_pos, y_pos), (x_pos + viewport_w, y_pos), 2)
                
                # Draw player label
                font = pygame.font.Font(None, 36)
                label = font.render(f"Player {car_id + 1}", True, (255, 255, 255))
                self.display_screen.blit(label, (x_pos + 10, y_pos + 10))
        
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(FPS)

    def _render_window(self, car_id, mode):
        """ Performs the actual rendering for each car individually using Pygame. 
        
        Parameters:
            car_id(int): Numerical id of car for which the corresponding window
                will be rendered.
            mode(str): Rendering mode.
        """
        
        # Initialize pygame if needed
        if not pygame.get_init():
            pygame.init()
        
        # Create a surface for this car's view
        if self.screen[car_id] is None:
            self.screen[car_id] = pygame.Surface((WINDOW_W, WINDOW_H))
        
        if "t" not in self.__dict__:
            return None  # reset() not called yet

        # Create a surface to render on
        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))
        
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = self.cars[car_id].hull.position[0]
        scroll_y = self.cars[car_id].hull.position[1]
        angle = -self.cars[car_id].hull.angle
        vel = self.cars[car_id].hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])

        # Render the road
        self._render_road_pygame(zoom, scroll_x, scroll_y, angle)
        
        # Render cars
        for id, car in enumerate(self.cars):
            if self.use_ego_color:
                car.hull.color = (0.0, 0.0, 0.8)
                if id == car_id:
                    car.hull.color = (0.8, 0.0, 0.0)
            self._render_car_pygame(car, zoom, scroll_x, scroll_y, angle)

        # Render indicators
        self._render_indicators_pygame(car_id, WINDOW_W, WINDOW_H)

        self.screen[car_id].blit(self.surf, (0, 0))
        
        # For human mode, the display is handled in _render_split_screen
        if mode == 'human':
            return self.isopen

        # Return pixel array for rgb_array or state_pixels modes
        if mode == 'rgb_array':
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == 'state_pixels':
            return self._create_image_array(self.surf, (STATE_W, STATE_H))

    def _world_to_screen(self, x, y, zoom, scroll_x, scroll_y, angle):
        """ Transform world coordinates to screen coordinates. """
        # Translate relative to car
        dx = x - scroll_x
        dy = y - scroll_y
        
        # Rotate
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        
        # Scale and translate to screen
        screen_x = int(WINDOW_W / 2 + rx * zoom)
        screen_y = int(WINDOW_H * self.h_ratio - ry * zoom)
        
        return screen_x, screen_y

    def _render_road_pygame(self, zoom, scroll_x, scroll_y, angle):
        """ Render the road using Pygame. """
        # Fill with grass color
        self.surf.fill((102, 204, 102))
        
        # Draw grass pattern
        k = PLAYFIELD / 20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                poly = [
                    self._world_to_screen(k*x + k, k*y + 0, zoom, scroll_x, scroll_y, angle),
                    self._world_to_screen(k*x + 0, k*y + 0, zoom, scroll_x, scroll_y, angle),
                    self._world_to_screen(k*x + 0, k*y + k, zoom, scroll_x, scroll_y, angle),
                    self._world_to_screen(k*x + k, k*y + k, zoom, scroll_x, scroll_y, angle),
                ]
                pygame.draw.polygon(self.surf, (102, 230, 102), poly)
        
        # Draw road polygons
        for poly, color in self.road_poly:
            screen_poly = [
                self._world_to_screen(p[0], p[1], zoom, scroll_x, scroll_y, angle)
                for p in poly
            ]
            color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            pygame.draw.polygon(self.surf, color_rgb, screen_poly)

    def _render_car_pygame(self, car, zoom, scroll_x, scroll_y, angle):
        """ Render a car using Pygame. """
        # Get car body vertices
        for fixture in car.hull.fixtures:
            shape = fixture.shape
            vertices = [(car.hull.transform * v) for v in shape.vertices]
            screen_vertices = [
                self._world_to_screen(v[0], v[1], zoom, scroll_x, scroll_y, angle)
                for v in vertices
            ]
            color = car.hull.color
            color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            pygame.draw.polygon(self.surf, color_rgb, screen_vertices)
        
        # Draw wheels
        for wheel in car.wheels:
            for fixture in wheel.fixtures:
                shape = fixture.shape
                vertices = [(wheel.transform * v) for v in shape.vertices]
                screen_vertices = [
                    self._world_to_screen(v[0], v[1], zoom, scroll_x, scroll_y, angle)
                    for v in vertices
                ]
                pygame.draw.polygon(self.surf, (50, 50, 50), screen_vertices)

    def _render_indicators_pygame(self, agent_id, W, H):
        """ Render HUD indicators using Pygame. """
        s = W / 40.0
        h = H / 40.0
        
        # Draw black bar at bottom
        pygame.draw.rect(self.surf, (0, 0, 0), (0, H - 5*h, W, 5*h))
        
        def vertical_ind(place, val, color):
            val = max(0, min(val, 4))  # Clamp to prevent overflow
            rect = pygame.Rect(int(place * s), int(H - h - h * val), int(s), int(h * val))
            pygame.draw.rect(self.surf, color, rect)
        
        def horiz_ind(place, val, color):
            # Handle negative values by adjusting x position and using absolute width
            x = int(place * s)
            width = int(val * s)
            if width < 0:
                x = x + width  # Shift x left by the width amount
                width = -width  # Make width positive
            rect = pygame.Rect(x, int(H - 4*h), width, int(2*h))
            pygame.draw.rect(self.surf, color, rect)
        
        true_speed = np.sqrt(
            np.square(self.cars[agent_id].hull.linearVelocity[0]) +
            np.square(self.cars[agent_id].hull.linearVelocity[1])
        )
        
        vertical_ind(5, 0.02 * true_speed, (255, 255, 255))
        vertical_ind(7, 0.01 * self.cars[agent_id].wheels[0].omega, (0, 0, 255))
        vertical_ind(8, 0.01 * self.cars[agent_id].wheels[1].omega, (0, 0, 255))
        vertical_ind(9, 0.01 * self.cars[agent_id].wheels[2].omega, (51, 0, 255))
        vertical_ind(10, 0.01 * self.cars[agent_id].wheels[3].omega, (51, 0, 255))
        horiz_ind(20, -10.0 * self.cars[agent_id].wheels[0].joint.angle, (0, 255, 0))
        horiz_ind(30, -0.8 * self.cars[agent_id].hull.angularVelocity, (255, 0, 0))
        
        # Draw score
        font = pygame.font.Font(None, 48)
        score_text = font.render(f"{int(self.reward[agent_id]):04d}", True, (255, 255, 255))
        self.surf.blit(score_text, (20, int(H - 2.5*h - 18)))
        
        # Render backwards flag
        if self.driving_backward[agent_id] and self.backwards_flag:
            pygame.draw.polygon(
                self.surf, (0, 0, 255),
                [(W - 100, H - 30), (W - 75, H - 70), (W - 50, H - 30)]
            )

    def _create_image_array(self, surface, size):
        """ Create a numpy array from a pygame surface at the given size. """
        scaled = pygame.transform.scale(surface, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled)), axes=(1, 0, 2)
        )

    def close(self):
        if any(screen is not None for screen in self.screen) or self.display_screen is not None:
            pygame.quit()
        self.screen = [None] * self.num_agents
        self.display_screen = None
        self._grid_cols = None
        self._grid_rows = None
        self._grid_viewport_w = None
        self._grid_viewport_h = None
        self.isopen = False


if __name__ == "__main__":
    NUM_CARS = 2  # Supports key control of two cars, but can simulate as many as needed

    # Specify key controls for cars (using pygame key codes)
    CAR_CONTROL_KEYS = [
        [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN],
        [pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_s]
    ]

    a = np.zeros((NUM_CARS, 3))
    env = MultiCarRacing(NUM_CARS, render_mode='human')
    
    record_video = False
    if record_video:
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(env, '/tmp/video-test', episode_trigger=lambda x: True)
    
    obs, info = env.reset()
    
    running = True
    restart = False
    total_reward = np.zeros(NUM_CARS)
    steps = 0
    
    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_RETURN:
                    restart = True
        
        # Get current key states
        keys = pygame.key.get_pressed()
        for i in range(min(len(CAR_CONTROL_KEYS), NUM_CARS)):
            # Steering: -1 for left, +1 for right, 0 for none
            if keys[CAR_CONTROL_KEYS[i][0]]:  # Left
                a[i][0] = -1.0
            elif keys[CAR_CONTROL_KEYS[i][1]]:  # Right
                a[i][0] = +1.0
            else:
                a[i][0] = 0
            
            # Gas
            a[i][1] = 1.0 if keys[CAR_CONTROL_KEYS[i][2]] else 0
            
            # Brake
            a[i][2] = 0.8 if keys[CAR_CONTROL_KEYS[i][3]] else 0
        
        s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        total_reward += r
        steps += 1
        
        if steps % 200 == 0 or done:
            print("\nActions: " + " ".join([f"Car {x}: {a[x]}" for x in range(NUM_CARS)]))
            print(f"Step {steps} Total_reward {total_reward}")
        
        env.render()
        
        if done or restart:
            obs, info = env.reset()
            total_reward = np.zeros(NUM_CARS)
            steps = 0
            restart = False
    
    env.close()
