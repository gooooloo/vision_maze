
import gym
from gym import spaces
import numpy as np

class VisionMazeEnv(gym.Env):
    def __init__(self, room_length=5, num_rooms_per_side=5):
        assert room_length % 2 == 1, "room_length must be odd"
        assert room_length >= 3, "room_length must be greater than 3"
        assert num_rooms_per_side >= 1, "must have at least 1 room"

        self.room_length = room_length
        self.num_rooms_per_side = num_rooms_per_side
        # 0 = up, 1 = right, 2 = down, 3 = left
        self.action_space = spaces.Discrete(4)
        self.max_pos = room_length * num_rooms_per_side - 1
        obs_space = (self.max_pos + 1, self.max_pos + 1, 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_space)
        self.goal_reward = 1
        self.not_moved_penalty = -1
        self.wrong_room_penalty = -1
        self.goal_state = [self.max_pos, self.max_pos]
        self._obs = np.zeros(obs_space)
        self.max_steps = 10
        self._reset()

    def _get_obs(self):
        self._obs.fill(0)
        self._obs[self.state[0], self.state[1], :] = 1
        return self._obs

    def _reset(self):
        # start in random state in the maze
        x = np.random.randint(self.max_pos)
        y = np.random.randint(self.max_pos)
        self.state = np.array([x, y])

        self.original_room_x = room_x = x // self.room_length
        self.original_room_y = room_y = y // self.room_length

        self.target_room_x, self.target_room_y = room_x + 1, room_y  # rightside room
        if self.target_room_x == self.num_rooms_per_side:
            # left then down
            self.target_room_x -= 1
            self.target_room_y -= 1
        if self.target_room_y == -1:
            # up then left
            self.target_room_y += 1
            self.target_room_x -= 1

        return self._get_obs()

    def _step(self, a):
        assert self.action_space.contains(a)
        ox, oy = x, y = self.state

        # up
        if a == 0:
            y = self._step_up(x, y)
        # right
        elif a == 1:
            x = self._step_right(x, y)
        # down
        elif a == 2:
            y = self._step_down(x, y)
        # left
        else:
            x = self._step_left(x, y)

        room_x = x // self.room_length
        room_y = y // self.room_length

        r, done = 0, False
        if room_x == self.target_room_x and room_y == self.target_room_y:
            r, done = self.goal_reward, True
        elif room_x != self.original_room_x or room_y != self.original_room_y:
            r, done = self.wrong_room_penalty, True
        elif x == ox and y == oy:
            r = self.not_moved_penalty
        else:
            pass

        self.state = np.array([x, y])
        return self._get_obs(), r, done, {'succ':done}

    def _step_up(self, x, y):
        ny = y + 1

        # convert to single room format
        local_ny = ny % self.room_length

        # this condition True indicates passing through wall
        if local_ny == 0:

            # this is only allowed if passing through doorway
            if not (x % self.room_length == self.room_length // 2):
                ny = y

        ny = min(ny, self.max_pos)
        return ny

    def _step_right(self, x, y):
        nx = x + 1

        # convert to single room format
        local_nx = nx % self.room_length

        # this condition True indicates passing through wall
        if local_nx == 0:

            # this is only allowed if passing through doorway
            if not (y % self.room_length == self.room_length // 2):
                nx = x

        nx = min(nx, self.max_pos)
        return nx

    def _step_down(self, x, y):
        ny = y - 1

        # convert to single room format
        local_ny = ny % self.room_length

        # this condition True indicates passing through wall
        if local_ny == self.room_length - 1:

            # this is only allowed if passing through doorway
            if not (x % self.room_length == self.room_length // 2):
                ny = y

        ny = max(0, ny)
        return ny

    def _step_left(self, x, y):
        nx = x - 1

        # convert to single room format
        local_nx = nx % self.room_length

        # this condition True indicates passing through wall
        if local_nx == self.room_length - 1:

            # this is only allowed if passing through doorway
            if not (y % self.room_length == self.room_length // 2):
                nx = x

        nx = max(0, nx)
        return nx
