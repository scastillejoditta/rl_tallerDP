# test_env.py
from rl_tallerdp.envs.milan_taxi import MilanTaxiEnv

env = MilanTaxiEnv()
state, info = env.reset()
print("Estado inicial:", state)

next_state, reward, terminated, truncated, info = env.step(0)  # SOUTH
print("Siguiente estado:", next_state, "reward:", reward)