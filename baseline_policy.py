import numpy as np
import random

class BaselinePolicy:
    """
    Policy that selects action based in current environment state.
    If holdings_quantity does not exceed the boundary, place order with constant value in session with min price.
    If holdings_quantity exceeds the boundary, do not place any order.
    Inputs:
        - current_state: dictionary with current state of the environment
        - constant_order: constant value of the order whem holdings_quantity do not exceed the boundary
        - boundary: boundary of the environment
    """
    def __init__(self, mode, constant_order, boundary, action_space_config='discrete'):
        self.constant_order = constant_order
        self.boundary = boundary
        self.mode = mode
        self.action_space_config = action_space_config

    def select_action(self, env):
        self.session_prices = env.get_prices()
        self.holdings_quantity = env.get_holdings_quantity()


        if np.sum(self.holdings_quantity) > self.boundary:
            if self.action_space_config == 'continous':
                action = np.zeros(len(self.holdings_quantity))
                return action
            elif self.action_space_config == 'discrete':
                return np.array([12])

        else:
            if self.mode == 'min_price':
                idx_min_price = np.argmin(self.session_prices)
                if self.action_space_config == 'continous':
                    action[idx_min_price] = self.constant_order
                    return action
                elif self.action_space_config == 'discrete':
                    return np.array([idx_min_price])

            elif self.mode == 'random':
                idx_random = random.randint(0,len(self.session_prices)-1)
                if self.action_space_config == 'continous':
                    action[idx_random] = self.constant_order
                    return action
                elif self.action_space_config == 'discrete':
                    return np.array([idx_random])
