import random
import numpy as np
import matplotlib.pyplot as plt
from trading_session_gym.envs.trading_session_gym import TradingSession
from baseline_policy import BaselinePolicy

env = TradingSession()
env.reset()
env.render()

policy = BaselinePolicy(mode = 'min_price', constant_order = 0.03/100, boundary = env.get_boundary())

done = False
i = 0
while done == False:
    action = policy.select_action(env)
    obs, reward, done, _ = env.step(action)
    if i % 10 == 0:
        env.render(mode = 'real-time')
    i +=1
