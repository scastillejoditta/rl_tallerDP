"""Tests de Policy Iteration y Value Iteration sobre MilanRainTaxi."""
import time

import numpy as np

from rl_tallerdp.agents.dynamic_programming import policy_iteration, value_iteration
from rl_tallerdp.envs import MilanRainTaxiEnv
from rl_tallerdp.utils.mdp import build_sparse_mdp

GAMMA = 0.99
PICKUP = 4

env = MilanRainTaxiEnv()                    # lluvia p=0.2
env_seco = MilanRainTaxiEnv(slip_prob=0.0)

P, R = build_sparse_mdp(env)
t0 = time.time()
V_pi, pi_pi, n_iter = policy_iteration(P, R, GAMMA)
print(f"Policy Iteration convergió en {n_iter} iteraciones ({time.time() - t0:.3f}s)")
V_vi, pi_vi, n_sweeps, deltas = value_iteration(P, R, GAMMA)

P0, R0 = build_sparse_mdp(env_seco)
V_seco, pi_seco, _, _ = value_iteration(P0, R0, GAMMA)


def _seguir_politica(e, pi, s0, max_steps=100):
    """Ejecuta la política desde s0; devuelve (retorno, pasos, estrellas)."""
    e.reset(seed=0)
    e.s = s0
    total = 0.0
    for pasos in range(1, max_steps + 1):
        _, r, terminado, _, info = e.step(int(pi[e.s].argmax()))
        total += r
        if terminado:
            return total, pasos, info["stars"]
    raise AssertionError("la política no entregó al pasajero")


def test_formas_y_politica_valida():
    assert V_pi.shape == (env.nS,)
    assert pi_pi.shape == (env.nS, env.nA)
    assert np.allclose(pi_pi.sum(axis=1), 1.0)           # una acción por estado
    assert set(np.unique(pi_pi)) <= {0.0, 1.0}           # determinista (one-hot)


def test_pi_y_vi_dan_el_mismo_valor():
    assert np.max(np.abs(V_pi - V_vi)) < 1e-6
    assert n_iter < 50          # PI converge en pocas iteraciones
    assert deltas[-1] < 1e-9    # VI terminó por tolerancia


def test_en_la_parada_con_el_pasajero_la_accion_optima_es_recoger():
    s = env.encode(0, 0, 0, 2)  # taxi y pasajero en la parada 0, destino parada 2
    assert pi_pi[s].argmax() == PICKUP
    assert pi_vi[s].argmax() == PICKUP


def test_sin_lluvia_la_ruta_optima_es_la_mas_corta_y_da_5_estrellas():
    s0 = env_seco.encode(0, 0, 0, 2)
    retorno, pasos, estrellas = _seguir_politica(env_seco, pi_seco, s0)
    # recoger (1) + 6 movimientos de (0,0) a (3,3) + dejar (1)
    assert pasos == 8
    assert estrellas == 5
    assert retorno == -1 - 6 + 20 + 10


def test_la_lluvia_reduce_el_valor_esperado():
    inicio = env.initial_state_distrib > 0
    assert V_vi[inicio].mean() < V_seco[inicio].mean()
