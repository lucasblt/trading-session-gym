import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)

STEP_SIZE = 10 # In seconds
SESSION_DURATION = 24*60*60/STEP_SIZE # In steps
NUM_SIMULATED_SESSIONS = 10*288 # Used to get the done
NUM_MUTUAL_SESSIONS = 288 # Number of mutual trading sessions
PRODUCT_DURATION = 5*60/STEP_SIZE # In steps

class TradingSession(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TradingSession, self).__init__()
        # Definition of action space:
        self.action_space = spaces.Box(low=0, high=1, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16)
        # Definition of observation space:
        self.observation_space = spaces.Dict({'session_steps_left': spaces.Box(low=1, high=SESSION_DURATION, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int32),
                                              'session_prices': spaces.Box(low=0.0, high=1.0, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.float32),
                                              'sessions_available': spaces.Box(low=0, high=1, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16)})


    def step(self, action):
        '''
        Executes one time step in the env.
        '''
        # Execute one time step within the environment
        self._take_action(action)
        obs, reward, done = self._next_observation(action), self._get_reward(), self._check_if_done()
        self.current_step += 1
        return obs, reward, done, {}

    def reset(self):
        '''
        Reset env to initial conditions.
        '''
        self.sessions_completed = 0
        self.current_step = 0
        self.reward = 0
        self.session_prices = np.zeros(NUM_MUTUAL_SESSIONS)
        self.sessions_available = np.ones(NUM_MUTUAL_SESSIONS)
        self.session_steps_left = np.linspace(SESSION_DURATION, PRODUCT_DURATION, num=NUM_MUTUAL_SESSIONS)

    def _take_action(self, action):
        '''
        Place agent's order.
        '''
        numerator =  np.sum(np.multiply(action, self.sessions_available))
        denominator = np.sum(np.multiply(np.multiply(action, self.sessions_available), self.session_prices))
        if denominator == 0 or numerator == 0:
            self.reward = 0
        else:
            self.reward = numerator/denominator

    def _next_observation(self, action):
        '''
        Update env and returns formated version of next observation.
        '''
        self._update_session_available(action)
        self._update_session_prices()
        self._update_session_steps_left()
        obs = {'session_steps_left': self.session_steps_left,
               'session_prices': self.session_prices,
               'sessions_available': self.sessions_available}
        return obs

    def _update_session_available(self, action):
        '''
        Update trading sessions that the agent placed orders.
        '''
        self.sessions_available = np.subtract(self.sessions_available, np.multiply(action, self.sessions_available))

    def _update_session_prices(self):
        '''
        Update the price of trading sessions.
        '''
        self.session_prices += np.random.normal(0, 0.1, NUM_MUTUAL_SESSIONS)
        neg_prices_idx = np.argwhere(self.session_prices < 0)
        max_prices_idx = np.argwhere(self.session_prices >= 1)

        for idx in neg_prices_idx:
            self.session_prices[neg_prices_idx] = 0
        for idx in max_prices_idx:
            self.session_prices[max_prices_idx] = 1

    def _update_session_steps_left(self):
        '''
        Update the progress of trading sessions.
        '''
        self.session_steps_left -= 1
        completed_idx = np.argwhere(self.session_steps_left == 0)
        self.session_steps_left[completed_idx] = SESSION_DURATION
        self.sessions_available[completed_idx] = 1
        self.sessions_completed += len(completed_idx)


    def _get_reward(self):
        '''
        Get reward.
        '''
        return self.reward

    def _check_if_done(self):
        '''
        Check if episode is done.
        '''
        return self.sessions_completed >= NUM_SIMULATED_SESSIONS

    def render(self, mode='human', close=False):
        '''
        Render the environment to the screen
        '''
        pass
