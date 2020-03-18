import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

STEP_SIZE = 5*60 # In seconds
SESSION_DURATION = 24*60*60/STEP_SIZE # In steps
NUM_MUTUAL_SESSIONS = 12 # Number of mutual trading sessions
NUM_SIMULATED_SESSIONS = 1*NUM_MUTUAL_SESSIONS # Used to get the done
PRODUCT_DURATION = 5*60/STEP_SIZE # In steps
MAX_SESSION_QUANTITY = 7
MAX_SESSION_PRICE = 101
MIN_SESSION_PRICE = 1
BOUNDARY = 3.3
CONSTANT_ORDER = 0.2/100


class TradingSession(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, action_space_config = 'continous'):
        super(TradingSession, self).__init__()
        self.action_space_config = action_space_config
        # Definition of action space:
        if self.action_space_config  == 'continous':
            self.action_space = spaces.Box(low=0, high=1, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.float32)
        elif self.action_space_config  == 'discrete':
            self.constant_order = CONSTANT_ORDER
            self.action_space = spaces.Discrete(NUM_MUTUAL_SESSIONS + 1)
        # Definition of observation space:
        low_space = np.array([0]*NUM_MUTUAL_SESSIONS + [0])
        high_space = np.array([1]*NUM_MUTUAL_SESSIONS + [MAX_SESSION_QUANTITY*NUM_MUTUAL_SESSIONS/BOUNDARY])
        self.observation_space = spaces.Box(low=low_space, high=high_space, dtype=np.float32)

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
        self.session_prices = np.full(NUM_MUTUAL_SESSIONS, (MAX_SESSION_PRICE+MIN_SESSION_PRICE)/2)
        self.tendency = self.session_prices[0]
        self.session_quantities = np.full(NUM_MUTUAL_SESSIONS, MAX_SESSION_QUANTITY, dtype='float')
        self.session_steps_left = np.arange(SESSION_DURATION, (SESSION_DURATION-NUM_MUTUAL_SESSIONS*PRODUCT_DURATION), -PRODUCT_DURATION)
        self.holdings_quantity = np.zeros(NUM_MUTUAL_SESSIONS, dtype='float')
        self.holdings_quantity_total = 0.0
        self.holdings_cash = np.zeros(NUM_MUTUAL_SESSIONS, dtype='float')
        self.boundary = BOUNDARY
        self.multiplier = 0
        return np.hstack([self.session_prices/self.session_prices.max(), self.holdings_quantity_total/self.boundary])

    def _take_action(self, action):
        '''
        Place agent's order and update holdings
        '''
        if self.action_space_config == 'discrete':
            idx = action
            action = np.zeros(len(self.session_prices))

            if idx < len(self.session_prices):
                action[idx] = self.constant_order

        self.holdings_quantity_previous = self.holdings_quantity.copy()
        self.holdings_cash_previous = self.holdings_cash.copy()

        action_times_quantity = np.multiply(action, self.session_quantities)

        self.holdings_quantity += action_times_quantity
        self.holdings_cash += np.multiply(action_times_quantity, self.session_prices)

        self.holdings_quantity_total = np.sum(self.holdings_quantity)

        for idx in range(NUM_MUTUAL_SESSIONS):
            if action_times_quantity[idx] > 0:
                self.session_quantities[idx] -= action_times_quantity[idx]
            elif action_times_quantity[idx] < 0:
                self.session_quantities[idx] += action_times_quantity[idx]

    def _next_observation(self):
        '''
        Update env and returns formated version of next observation.
        '''
        self._update_session_prices()
        self._update_session_steps_left()
        obs = np.hstack([self.session_prices/self.session_prices.max(), self.holdings_quantity_total/self.boundary])
        return obs

    def _update_session_prices(self):
        '''
        Update the price of trading sessions.
        '''
        self._sim_price_dynamics()


        neg_prices_idx = np.argwhere(self.session_prices < MIN_SESSION_PRICE)
        max_prices_idx = np.argwhere(self.session_prices >= MAX_SESSION_PRICE)

        for idx in neg_prices_idx:
            self.session_prices[neg_prices_idx] = MIN_SESSION_PRICE
        for idx in max_prices_idx:
            self.session_prices[max_prices_idx] = MAX_SESSION_PRICE

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
        self.session_steps_left[idx] = 0
        self.session_quantities[idx] = 0

    def _compute_reward(self):

        delta_forecast_previous = np.absolute(self.boundary - np.sum(self.holdings_quantity_previous))
        delta_forecast_updated = np.absolute(self.boundary - np.sum(self.holdings_quantity))

        if delta_forecast_previous == delta_forecast_updated:
            self.reward = 0
            return 0

        elif delta_forecast_previous > delta_forecast_updated:
            multiplier = 1000
        elif delta_forecast_previous < delta_forecast_updated:
            multiplier = -1000

        delta_holdings = np.subtract(self.holdings_quantity, self.holdings_quantity_previous)
        delta_cash = np.subtract(self.holdings_cash, self.holdings_cash_previous)

        quantity_over_cash = np.sum(np.divide(delta_holdings, self.session_prices))

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

    def get_current_state(self):
        current_state = np.hstack([self.session_prices, self.holdings_quantity])
        return current_state

    def get_boundary(self):
        return self.boundary

    def _check_if_done(self):
        '''
        Check if episode is done.
        '''
        return self.sessions_completed >= NUM_SIMULATED_SESSIONS

    def _sim_price_dynamics(self):
        self.tendency += self._ou_process(self.tendency,
                                          ou_lambda = 1e-5,
                                          ou_mu = self.tendency,
                                          ou_sigma = 0.3)
        for i in range(len(self.session_prices)):
            if i == 0:
                self.session_prices[i] += self._ou_process(self.session_prices[i],
                                                           ou_lambda = 10e-3,
                                                           ou_mu = self.tendency,
                                                           ou_sigma = 0.1)
            else:
                self.session_prices[i] += self._ou_process(self.session_prices[i],
                                                           ou_lambda = 10e-3,
                                                           ou_mu = self.session_prices[i-1],
                                                           ou_sigma = 0.1)

    def _ou_process(self, price, ou_lambda, ou_mu, ou_sigma):
        return (ou_lambda * (ou_mu - price) + ou_sigma * np.random.randn())

    def render(self, mode='human', close=False):
        '''
        Render the environment to the screen
        '''
        pass
