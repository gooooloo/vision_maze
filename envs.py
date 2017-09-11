import gym
import logging
from collections import deque
import numpy as np
import vision_maze
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_ENV_ID = 'Breakout-v0'
PS_IP = '127.0.0.1'
PS_PORT = 12222
TB_PORT = 12345
NUM_GLOBAL_STEPS = 30000000
SAVE_MODEL_SECS = 30
SAVE_SUMMARIES_SECS = 30
LOG_DIR = './log/'
NUM_WORKER = 4
VISUALISED_WORKERS = []  # e.g. [0] or [1,2]

_N_AVERAGE = 100

VSTR = 'fun.V2'

class feudal_config():
    alpha = .5
    vf_hidden_size = 128
    k = 16 #Dimensionality of w
    g_dim = 256
    c = 10
    beta_start = .01
    beta_end = .001
    decay_steps = 50000000


class MyEnv:
    def __init__(self, env):
        self._env = env
        self._ep_count = 0
        self._ep_steps = 0
        self._ep_rewards = []

        self._steps_last_n_eps = deque(maxlen=_N_AVERAGE)
        self._rewards_last_n_eps = deque(maxlen=_N_AVERAGE)

        self.spec = env.spec
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = env.metadata

    def reset(self):
        self._ep_count += 1
        self._ep_steps = 0
        self._ep_rewards = []
        s = self._env.reset()
        return s

    def step(self, a):
        s, r, t, _ = self._env.step(a)

        self._ep_steps += 1
        self._ep_rewards.append(r)

        if self._ep_steps >= self._env.max_steps: t = True

        i = {}
        if t:
            self._steps_last_n_eps.append(self._ep_steps)
            self._rewards_last_n_eps.append(np.sum(self._ep_rewards))

            i['{}/ep_count'.format(VSTR)] = self._ep_count
            i['{}/ep_steps'.format(VSTR)] = self._steps_last_n_eps[-1]
            i['{}/ep_rewards'.format(VSTR)] = self._rewards_last_n_eps[-1]
            i['{}/aver_steps_{}'.format(VSTR, _N_AVERAGE)] = np.average(self._steps_last_n_eps)
            i['{}/aver_rewards_{}'.format(VSTR, _N_AVERAGE)] = np.average(self._rewards_last_n_eps)

            print(i)

        return s, r, t, i


def create_env(task_id):
    env = vision_maze.VisionMazeEnv()
    return MyEnv(env)
