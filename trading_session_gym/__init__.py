from gym.envs.registration import register

register(
    id='trading-session-gym-v0',
    entry_point='trading_session_gym.envs:TradingSession',
)
