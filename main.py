import random
import numpy as np
from trading_session_gym.envs.trading_session_gym import TradingSession

env = TradingSession()
env.reset()
env.render()
done = False

proability_of_zero = 0.7

while done == False:
    action = env.action_space.sample()
    action = np.where(np.random.random(action.shape) < proability_of_zero, 0, action)

    obs, reward, done, _ = env.step(action)
    env.render()
