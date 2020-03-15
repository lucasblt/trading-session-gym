import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

STEP_SIZE = 60 # In seconds
SESSION_DURATION = 24*60*60/STEP_SIZE # In steps
NUM_MUTUAL_SESSIONS = 12 # Number of mutual trading sessions
NUM_SIMULATED_SESSIONS = 1*NUM_MUTUAL_SESSIONS # Used to get the done
PRODUCT_DURATION = 5*60/STEP_SIZE # In steps
MAX_SESSION_QUANTITY = 7
MAX_SESSION_PRICE = 101
MIN_SESSION_PRICE = 1
BOUNDARY = 3.3


class TradingSession(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TradingSession, self).__init__()
        self.visualizer = None
        # Definition of action space:
        self.action_space = spaces.Box(low=0, high=1, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.float32)
        # Definition of observation space:
        self.observation_space = spaces.Dict({'session_steps_left': spaces.Box(low=1, high=SESSION_DURATION, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.int16),
                                              'session_prices': spaces.Box(low=MIN_SESSION_PRICE, high=MAX_SESSION_PRICE, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.float32),
                                              'session_quantities': spaces.Box(low=-MAX_SESSION_QUANTITY, high=MAX_SESSION_QUANTITY, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.float32),
                                              'holdings_quantity': spaces.Box(low=-MAX_SESSION_QUANTITY, high=MAX_SESSION_QUANTITY, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.float32),
                                              'holdings_cash': spaces.Box(low=-MAX_SESSION_PRICE*MAX_SESSION_QUANTITY, high=MAX_SESSION_PRICE*MAX_SESSION_QUANTITY, shape=(NUM_MUTUAL_SESSIONS,), dtype=np.float32)})

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
        #self.session_prices = MAX_SESSION_PRICE * np.random.rand(NUM_MUTUAL_SESSIONS)
        self.session_quantities = np.full(NUM_MUTUAL_SESSIONS, MAX_SESSION_QUANTITY, dtype='float')
        self.session_steps_left = np.arange(SESSION_DURATION, (SESSION_DURATION-NUM_MUTUAL_SESSIONS*PRODUCT_DURATION), -PRODUCT_DURATION)
        self.holdings_quantity = np.zeros(NUM_MUTUAL_SESSIONS, dtype='float')
        self.holdings_cash = np.zeros(NUM_MUTUAL_SESSIONS, dtype='float')
        self.boundary = BOUNDARY
        self.multiplier = 0


    def _take_action(self, action):
        '''
        Place agent's order and update holdings
        '''
        self.holdings_quantity_previous = self.holdings_quantity.copy()
        self.holdings_cash_previous = self.holdings_cash.copy()

        action_times_quantity = np.multiply(action, self.session_quantities)

        self.holdings_quantity += action_times_quantity
        self.holdings_cash += np.multiply(action_times_quantity, self.session_prices)

        for idx in range(NUM_MUTUAL_SESSIONS):
            if action_times_quantity[idx] > 0:
                self.session_quantities[idx] -= action_times_quantity[idx]
            elif action_times_quantity[idx] > 0:
                self.session_quantities[idx] += action_times_quantity[idx]

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
        self.session_prices += np.random.normal(0, 0.005*MAX_SESSION_PRICE, NUM_MUTUAL_SESSIONS)
        neg_prices_idx = np.argwhere(self.session_prices < MIN_SESSION_PRICE)
        max_prices_idx = np.argwhere(self.session_prices >= MAX_SESSION_PRICE)

        for idx in neg_prices_idx:
            self.session_prices[neg_prices_idx] = MIN_SESSION_PRICE
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
        #self.session_steps_left[idx] = SESSION_DURATION
        #self.session_quantities[idx] = MAX_SESSION_QUANTITY
        self.session_steps_left[idx] = 0
        self.session_quantities[idx] = 0
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
            multiplier = 100
        elif delta_forecast_previous < delta_forecast_updated:
            multiplier = -100

        delta_holdings = np.subtract(self.holdings_quantity, self.holdings_quantity_previous)
        delta_cash = np.subtract(self.holdings_cash, self.holdings_cash_previous)

        #quantity_over_cash = np.sum(np.nan_to_num(np.divide(delta_holdings, delta_cash), copy=True, nan=0, posinf=0, neginf=0))

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
        current_state = {'session_steps_left': self.session_steps_left,
                         'session_prices': self.session_prices,
                         'session_quantities': self.session_quantities,
                         'holdings_quantity': self.holdings_quantity,
                         'holdings_cash': self.holdings_cash}
        return current_state

    def get_boundary(self):
        return self.boundary

    def _check_if_done(self):
        '''
        Check if episode is done.
        '''
        return self.sessions_completed >= NUM_SIMULATED_SESSIONS

    def render(self, mode='human', close=False):
        '''
        Render the environment to the screen
        '''
        if mode == 'real-time':
            if self.visualizer == None:
                self.visualizer = EnvironmentVisualizer()
            self.visualizer._render_reward(self.reward)
            self.visualizer._render_prices(self.session_prices)
            self.visualizer._resize_x()

            plt.pause(0.001)
        pass

class EnvironmentVisualizer:
    def __init__(self, limit_x = 100):
        plt.ion()
        self.fig, self.axs = plt.subplots(13, figsize=(7,7), sharex=True)
        self.fig.subplots_adjust(hspace=0)
        self.lines = []
        self.limit_x = limit_x
        self.rewards_buffer = np.zeros(1)
        self.session_prices_buffer =np.zeros((1, 12))
        for ax in self.axs:
            line, = ax.plot(0, 0, alpha=0.8)
            self.lines.append(line)

        self.axs[-1].set_ylim([0, 0.1])

        plt.show()

    def _render_reward(self, reward):
        self.rewards_buffer = np.append(self.rewards_buffer, reward)

        x_vec = np.arange(len(self.rewards_buffer))
        y_vec = 100*self.rewards_buffer

        self.lines[-1].set_data(x_vec, y_vec)

        if np.min(y_vec)*1.30 <= self.lines[-1].axes.get_ylim()[0] or np.max(y_vec)*1.30 >= self.lines[-1].axes.get_ylim()[1]:
            self.axs[-1].set_ylim([np.min(y_vec)*1.30, np.max(y_vec)*1.30])

    def _render_prices(self, prices):
        self.session_prices_buffer = np.vstack((self.session_prices_buffer, np.reshape(prices, (1, 12))))

        for idx in range(len(prices)):
            x_vec = np.arange(len(self.session_prices_buffer))
            y_vec = self.session_prices_buffer[:, idx]

            self.lines[idx].set_data(x_vec, y_vec)

            if np.min(y_vec)*1.30 <= self.lines[idx].axes.get_ylim()[0] or np.max(y_vec)*1.30 >= self.lines[idx].axes.get_ylim()[1]:
                self.axs[idx].set_ylim([np.min(y_vec)*1.30, np.max(y_vec)*1.30])

    def _resize_x(self):
        #if self.lines[-1].axes.get_xlim()[1] >= self.limit_x:
        #    self.axs[0].set_xlim([self.lines[-1].axes.get_xlim()[0] + 1, self.lines[-1].axes.get_xlim()[1] + 1])
        #else:
            self.axs[0].set_xlim([0, self.lines[-1].axes.get_xlim()[1] + 1])
