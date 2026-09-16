import numpy as np
from rl_tallerdp.envs.milan_taxi import MilanTaxiEnv
from rl_tallerdp.models.build_mdp import build_model, encode_state, decode_state

env = MilanTaxiEnv()
P, R = build_model(env)

print("Forma de P:", P.shape)   # esperado: (500, 6, 500)
print("Forma de R:", R.shape)   # esperado: (500, 6)

# Cada fila de P[s,a,:] debe sumar 1 (es una distribución de probabilidad)
sums = P.sum(axis=2)
assert np.allclose(sums, 1.0), f"Filas que no suman 1: {np.where(~np.isclose(sums, 1.0))}"
print("✓ Todas las filas de P suman 1")

# Caso PICKUP exitoso
s = encode_state(0, 0, 0, 2)
print("\nCaso PICKUP exitoso, P[s, PICKUP]:")
print(np.nonzero(P[s, 4]), R[s, 4])

# Taxi en (0,0), pero pasajero está en LOCS[1]=(0,4), no en LOCS[0]
s_fail = encode_state(0, 0, 1, 2)
print("\nCaso PICKUP fallido:")
idx = np.nonzero(P[s_fail, 4])[0]
print("Estados destino:", idx, "reward:", R[s_fail, 4])

# Taxi en LOCS[2]=(4,0), pasajero en el taxi (pass_idx=4), destino=2 → LOCS[2]=(4,0)
s_dropoff = encode_state(4, 0, 4, 2)
print("\nCaso DROPOFF exitoso (debe dar 12 ramas):")
idx = np.nonzero(P[s_dropoff, 5])[0]
print("Número de estados destino distintos:", len(idx))  # esperado: 12
print("Suma de probabilidades:", P[s_dropoff, 5].sum())    # esperado: 1.0
print("Reward:", R[s_dropoff, 5])                          # esperado: 20

# Movimiento libre con resbale 0.2: dos ramas (avanzar 0.8, quedarse 0.2)
s_move = encode_state(2, 2, 0, 1)
idx_move = np.nonzero(P[s_move, 2])[0]  # EAST
print("\nCaso EAST con resbale (debe dar 2 ramas):")
print("Número de estados destino:", len(idx_move))
print("Probabilidades:", sorted(P[s_move, 2, idx_move]))
assert len(idx_move) == 2
assert np.allclose(sorted(P[s_move, 2, idx_move]), [0.2, 0.8])

env0 = MilanTaxiEnv(prob_resbale=0.0)
P0, _ = build_model(env0)
idx0 = np.nonzero(P0[s_move, 2])[0]
assert len(idx0) == 1
print("✓ Sin resbale, EAST tiene una sola rama")