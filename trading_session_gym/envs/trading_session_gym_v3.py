import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)

STEP_SIZE = 60 # In seconds
SESSION_DURATION = 24*60*60/STEP_SIZE # In steps
NUM_SIMULATED_SESSIONS = 2*288 # Used to get the done
NUM_MUTUAL_SESSIONS = 288 # Number of mutual trading sessions
PRODUCT_DURATION = 5*60/STEP_SIZE # In steps
MAX_SESSION_QUANTITY = 10000
MAX_SESSION_PRICE = 1000
MAX_FORECAST = 1000

class TradingSession(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TradingSession, self).__init__()
        # Definition of action space:
        self.action_space = spaces.Box(low=0, high=5, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16)
        # Definition of observation space:
        self.observation_space = spaces.Dict({'session_steps_left': spaces.Box(low=1, high=SESSION_DURATION, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16),
                                              'session_prices': spaces.Box(low=0, high=MAX_SESSION_PRICE, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16),
                                              'session_quantities': spaces.Box(low=0, high=MAX_SESSION_QUANTITY, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16),
                                              'holdings_quantity': spaces.Box(low=0, high=MAX_SESSION_QUANTITY, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16),
                                              'holdings_cash': spaces.Box(low=0, high=MAX_SESSION_PRICE*MAX_SESSION_QUANTITY, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16),
                                              'forecast_quantity': spaces.Box(low=0, high=MAX_FORECAST, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16)})


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
        self.session_steps_left = np.linspace(SESSION_DURATION, PRODUCT_DURATION, num=NUM_MUTUAL_SESSIONS)
        self.holdings_quantity = np.zeros(NUM_MUTUAL_SESSIONS)
        self.holdings_quantity_previous = np.zeros(NUM_MUTUAL_SESSIONS)
        self.holdings_cash = np.zeros(NUM_MUTUAL_SESSIONS)
        self.forecast_quantity = np.full(NUM_MUTUAL_SESSIONS, 0.5 * MAX_FORECAST)

    def _take_action(self, action):
        '''
        Place agent's order and update holdings
        '''
        self.holdings_quantity_previous = self.holdings_quantity.copy()
        self.holdings_cash_previous = self.holdings_cash.copy()

        greater_idx = np.argwhere(np.greater(action, self.session_quantities))
        for idx in greater_idx:
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
               'holdings_cash': self.holdings_cash,
               'forecast_quantity': self.forecast_quantity}
        return obs

    def _update_session_prices(self):
        '''
        Update the price of trading sessions.
        '''
        self.session_prices += np.round(np.random.normal(0, 10, NUM_MUTUAL_SESSIONS))
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
        delta_forecast_previous = np.absolute(self.forecast_quantity - self.holdings_quantity_previous)
        delta_forecast_updated = np.absolute(self.forecast_quantity - self.holdings_quantity)

        multiplier_array = np.zeros(NUM_MUTUAL_SESSIONS)

        for idx in range(NUM_MUTUAL_SESSIONS):
            if delta_forecast_previous[idx] > delta_forecast_updated[idx]:
                multiplier_array[idx] = 1
            elif delta_forecast_previous[idx] < delta_forecast_updated[idx]:
                multiplier_array[idx] = -1

        delta_holdings = np.subtract(self.holdings_quantity, self.holdings_quantity_previous)
        delta_cash = np.subtract(self.holdings_cash, self.holdings_cash_previous)

        quantity_over_cash = np.nan_to_num(np.divide(delta_holdings, delta_cash), copy=True, nan=0, posinf=0, neginf=0)
        self.reward =np.sum(np.multiply(multiplier_array, quantity_over_cash))

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
        print("Reward: {}".format(round(self.reward, 3)))
        pass
