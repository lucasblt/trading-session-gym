from trading_session_gym.envs.trading_session_gym import TradingSession

env = TradingSession()
env.reset()
env.render()
done = False

while done == False:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    env.render()
