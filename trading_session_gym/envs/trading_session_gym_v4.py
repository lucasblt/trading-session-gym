import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)

STEP_SIZE = 60 # In seconds
SESSION_DURATION = 24*60*60/STEP_SIZE # In steps
NUM_MUTUAL_SESSIONS = 12 # Number of mutual trading sessions
NUM_SIMULATED_SESSIONS = 10*NUM_MUTUAL_SESSIONS # Used to get the done
PRODUCT_DURATION = 5*60/STEP_SIZE # In steps
MAX_SESSION_QUANTITY = 7000
MAX_SESSION_PRICE = 100
BOUNDARY = 3300

class TradingSession(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TradingSession, self).__init__()
        # Definition of action space:
        self.action_space = spaces.Box(low=-275, high=275, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16)
        # Definition of observation space:
        self.observation_space = spaces.Dict({'session_steps_left': spaces.Box(low=1, high=SESSION_DURATION, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16),
                                              'session_prices': spaces.Box(low=0, high=MAX_SESSION_PRICE, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16),
                                              'session_quantities': spaces.Box(low=-MAX_SESSION_QUANTITY, high=MAX_SESSION_QUANTITY, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16),
                                              'holdings_quantity': spaces.Box(low=0, high=MAX_SESSION_QUANTITY, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16),
                                              'holdings_cash': spaces.Box(low=0, high=MAX_SESSION_PRICE*MAX_SESSION_QUANTITY, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16)})


    def step(self, action):
        '''
        Executes one time step in the env.
        '''
        # Execute one time step within the environment
        self.reward = 0
        self._take_action(action)
        reward = self._compute_reward()
        obs = self._next_observation()
        done = self._check_if_done()
        self.current_step += 1
        return obs, reward, done, {}

    def reset(self):
        '''
        Reset env to initial conditions.
        '''
        self.sessions_completed = 0
        self.current_step = 0
        self.reward = 0
        self.session_prices = np.round(MAX_SESSION_PRICE * np.random.rand(NUM_MUTUAL_SESSIONS))
        self.session_quantities = np.full(NUM_MUTUAL_SESSIONS, MAX_SESSION_QUANTITY)
        self.session_steps_left = np.arange(SESSION_DURATION, (SESSION_DURATION-NUM_MUTUAL_SESSIONS*PRODUCT_DURATION), -PRODUCT_DURATION)
        self.holdings_quantity = np.zeros(NUM_MUTUAL_SESSIONS)
        self.holdings_cash = np.zeros(NUM_MUTUAL_SESSIONS)
        self.boundary = BOUNDARY
        self.multiplier = 0

    def _take_action(self, action):
        '''
        Place agent's order and update holdings
        '''
        self.holdings_quantity_previous = self.holdings_quantity.copy()
        self.holdings_cash_previous = self.holdings_cash.copy()

        for idx in range(NUM_MUTUAL_SESSIONS):
            if (action[idx] > 0) and (action[idx] > self.session_quantities[idx]):
                    action[idx] = self.session_quantities[idx]
            elif action[idx] < 0 and (action[idx] > self.session_quantities[idx]):
                    action[idx] = self.session_quantities[idx]

        self.holdings_quantity += action
        self.holdings_cash += np.multiply(action, self.session_prices)
        self.session_quantities -= action

    def _next_observation(self):
        '''
        Update env and returns formated version of next observation.
        '''
        self._update_session_prices()
        self._update_session_quantities()
        self._update_session_steps_left()
        obs = {'session_steps_left': self.session_steps_left,
               'session_prices': self.session_prices,
               'session_quantities': self.session_quantities,
               'holdings_quantity': self.holdings_quantity,
               'holdings_cash': self.holdings_cash}
        return obs

    def _update_session_prices(self):
        '''
        Update the price of trading sessions.
        '''
        self.session_prices += np.round(np.random.normal(0, 0.01*MAX_SESSION_PRICE, NUM_MUTUAL_SESSIONS))
        neg_prices_idx = np.argwhere(self.session_prices < 0)
        max_prices_idx = np.argwhere(self.session_prices >= MAX_SESSION_PRICE)

        for idx in neg_prices_idx:
            self.session_prices[neg_prices_idx] = 0
        for idx in max_prices_idx:
            self.session_prices[max_prices_idx] = MAX_SESSION_PRICE

    def _update_session_quantities(self):
        '''
        Update available quantities of trading sessions.
        '''
        pass


    def _update_session_steps_left(self):
        '''
        Update the progress of trading sessions.
        '''
        self.session_steps_left -= 1

        completed_idx = np.argwhere(self.session_steps_left == 0)

        for idx in completed_idx:
            self._complete_session(idx)

    def _complete_session(self, idx):
        self.sessions_completed +=1
        self.session_steps_left[idx] = SESSION_DURATION
        self.session_quantities[idx] = MAX_SESSION_QUANTITY
        self.holdings_cash[idx] = 0
        self.holdings_cash_previous[idx] = 0
        self.holdings_quantity[idx] = 0
        self.holdings_quantity_previous[idx] = 0

    def _compute_reward(self):

        delta_forecast_previous = np.absolute(self.boundary - np.sum(self.holdings_quantity_previous))
        delta_forecast_updated = np.absolute(self.boundary - np.sum(self.holdings_quantity))

        if delta_forecast_previous == delta_forecast_updated:
            self.reward = 0
            return 0

        elif delta_forecast_previous > delta_forecast_updated:
            multiplier = 1
        elif delta_forecast_previous < delta_forecast_updated:
            multiplier = -1

        delta_holdings = np.subtract(self.holdings_quantity, self.holdings_quantity_previous)
        delta_cash = np.subtract(self.holdings_cash, self.holdings_cash_previous)

        quantity_over_cash = np.sum(np.nan_to_num(np.divide(delta_holdings, delta_cash), copy=True, nan=0, posinf=0, neginf=0))
        self.reward = multiplier*quantity_over_cash
        return self.reward

    def get_reward(self):
        return self.reward

    def get_prices(self):
        return self.session_prices

    def get_holdings_quantity(self):
        return self.holdings_quantity

    def get_sessions_completed(self):
        return self.sessions_completed

    def _check_if_done(self):
        '''
        Check if episode is done.
        '''
        return self.sessions_completed >= NUM_SIMULATED_SESSIONS

    def render(self, mode='human', close=False):
        '''
        Render the environment to the screen
        '''
        #print("Reward: {}".format(round(self.reward, 3)))
        pass
