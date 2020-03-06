import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)

STEP_SIZE = 60 # In seconds
DAYS_SIMULATED = 2
NUM_SESSIONS_PER_DAY = 288 # Number of mutual trading sessiong
NUM_SESSIONS = DAYS_SIMULATED*NUM_SESSIONS_PER_DAY # Number of trading sessions
NUM_TIMESTEPS = (NUM_SESSIONS*5*60)/STEP_SIZE

class TradingSession(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TradingSession, self).__init__()
        # Definition of action space:
        self.action_space = spaces.MultiDiscrete([2]*NUM_SESSIONS)
        # Definition of observation space:
        self.observation_space = spaces.Dict({'session_open': spaces.Box(low=0, high=1, shape=(NUM_SESSIONS,), dtype=np.int16),
                                              'session_prices': spaces.Box(low=0.0, high=10.0, shape=(NUM_SESSIONS,), dtype=np.float32),
                                              'current_step': spaces.Discrete(NUM_TIMESTEPS - 1)})

    def step(self, action):
        '''
        Executes one time step in the env.
        '''
        # Execute one time step within the environment
        self._take_action(action)
        self._match_orders()

        obs, reward, done = self._next_observation(), self._get_reward(), self._check_if_done()

        self.current_step += 1
        return obs, reward, done, {}

    def reset(self):
        '''
        Reset env to initial conditions.
        '''
        self.current_step = 0
        self.reward = 0
        self.session_prices = np.zeros(NUM_SESSIONS)
        self.session_open = np.array([1] * NUM_SESSIONS_PER_DAY + [0] * (NUM_SESSIONS-NUM_SESSIONS_PER_DAY))
        #self._update_session_open()

    def _take_action(self, action):
        '''
        Place agent's order.
        '''
        self.reward = np.sum(np.multiply(np.multiply(action, self.session_prices), self.session_open))

    def _place_other_orders(self):
        '''
        Place other agents' orders.
        '''
        pass

    def _match_orders(self):
        '''
        Match orders using online algorithm and update trading session prices.
        '''
        self._update_session_prices()

    def _get_reward(self):
        '''
        Get reward.
        '''
        return self.reward

    def _next_observation(self):
        '''
        Update env and returns formated version of next observation.
        '''
        self._update_session_open()
        self._place_other_orders()
        obs = self.observation_space.sample()
        return obs

    def _update_session_open(self):
        '''
        Update the session_open array based on value of current timestep.
        '''
        if (self.current_step % (300/STEP_SIZE)) == 0:
            session_end_idx = int(self.current_step/(300/STEP_SIZE))
            session_start_idx = session_end_idx + (NUM_SESSIONS_PER_DAY - 1)
            self.session_open[session_end_idx] = 0
            if session_start_idx < NUM_SESSIONS:
                self.session_open[session_start_idx] = 1


    def _update_session_prices(self):
        '''
        Update the price of open trading sessions.
        '''
        self.session_prices = np.multiply(np.random.rand(NUM_SESSIONS), self.session_open)

    def _check_if_done(self):
        '''
        Check if episode is done.
        '''
        return self.current_step >= (NUM_TIMESTEPS - 1)

    def render(self, mode='human', close=False):
        '''
        Render the environment to the screen
        '''
        print('Step: {} - Reward: {}'.format(self.current_step, self.reward))
        print('{}'.format(self.session_open))
