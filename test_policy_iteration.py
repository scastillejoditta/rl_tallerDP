# test_policy_iteration.py
import time
import numpy as np
from rl_tallerdp.envs.milan_taxi import MilanTaxiEnv
from rl_tallerdp.models.build_mdp import build_model, decode_state
from rl_tallerdp.agents.dynamic_programming import policy_iteration

env = MilanTaxiEnv()
P, R = build_model(env)

GAMMA = 0.9

t0 = time.time()
V, pi, n_iter = policy_iteration(P, R, GAMMA)
t1 = time.time()

print(f"Policy Iteration convergió en {n_iter} iteraciones ({t1-t0:.3f}s)")
print(f"V.shape={V.shape}, pi.shape={pi.shape}")

# Verifica que pi sea una política determinista válida (one-hot por fila)
row_sums = pi.sum(axis=1)
assert np.allclose(row_sums, 1.0), "Alguna fila de pi no suma 1"
print("✓ pi es una distribución válida en cada estado")

# Inspecciona la acción óptima en un par de estados concretos
ACCIONES = ["SOUTH", "NORTH", "EAST", "WEST", "PICKUP", "DROPOFF"]

# Ejemplo: taxi en (0,0), pasajero en LOCS[0]=(0,0), destino=2 -> ¿debería recomendar PICKUP?
from rl_tallerdp.models.build_mdp import encode_state
s = encode_state(0, 0, 0, 2)
a_opt = np.argmax(pi[s])
print(f"\nEstado {s} {decode_state(s)}: acción óptima = {ACCIONES[a_opt]}")
print(f"V(s) = {V[s]:.3f}")