import numpy as np
from rl_tallerdp.envs.milan_taxi import MOVEMENT_ACTIONS, NUM_COLS, NUM_ROWS

N_STATES = NUM_ROWS * NUM_COLS * 5 * 4   # 500
N_ACTIONS = 6


def encode_state(row, col, pass_idx, dest_idx):
    """Convierte (row, col, pass_idx, dest_idx) a un único entero."""
    return ((row * NUM_COLS + col) * 5 + pass_idx) * 4 + dest_idx


def decode_state(s):
    dest_idx = s % 4
    s //= 4
    pass_idx = s % 5
    s //= 5
    col = s % NUM_COLS
    row = s // NUM_COLS
    return row, col, pass_idx, dest_idx


def build_model(env):
    """Devuelve (P, R) como arrays densos, siguiendo la convención del notebook."""
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS))

    todos_los_pares = [(p, d) for p in range(4) for d in range(4) if p != d]
    prob_por_par = 1 / len(todos_los_pares)
    prob_resbale = getattr(env, "prob_resbale", 0.0)

    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            for pass_idx in range(5):
                for dest_idx in range(4):
                    s = encode_state(row, col, pass_idx, dest_idx)
                    s_stay = s
                    for a in range(N_ACTIONS):
                        new_row, new_col, new_pass_idx, _, reward, dest_reached = \
                            env._transitions(row, col, pass_idx, dest_idx, a)

                        if dest_reached:
                            for (p, d) in todos_los_pares:
                                s2 = encode_state(new_row, new_col, p, d)
                                P[s, a, s2] += prob_por_par
                        else:
                            s2 = encode_state(new_row, new_col, new_pass_idx, dest_idx)
                            if (
                                a in MOVEMENT_ACTIONS
                                and prob_resbale > 0.0
                                and s2 != s_stay
                            ):
                                P[s, a, s2] += 1.0 - prob_resbale
                                P[s, a, s_stay] += prob_resbale
                            else:
                                P[s, a, s2] += 1.0

                        R[s, a] = reward
    return P, R
