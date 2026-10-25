"""Tests de las matrices (P, R) construidas con build_sparse_mdp."""
import numpy as np

from rl_tallerdp.envs import MilanRainTaxiEnv
from rl_tallerdp.utils.mdp import build_sparse_mdp

SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF = range(6)

env = MilanRainTaxiEnv()
P, R = build_sparse_mdp(env)
nS, nA = env.nS, env.nA


def fila(s, a, P=P):
    """Sucesores de (s, a) como dict {estado_siguiente: probabilidad}."""
    row = P[s * nA + a]
    return dict(zip(row.indices.tolist(), row.data.tolist()))


def test_formas():
    assert P.shape == (nS * nA, nS)   # (73728, 12288)
    assert R.shape == (nS, nA)        # (12288, 6)


def test_filas_suman_1_salvo_terminales():
    sumas = np.asarray(P.sum(axis=1)).ravel()
    terminales = sumas < 0.5
    # Terminales: DROPOFF correcto = una parada de destino x todos los valores de b
    assert terminales.sum() == env.n_stops * (env.b_max + 1)
    assert np.allclose(sumas[~terminales], 1.0)
    assert np.allclose(sumas[terminales], 0.0)


def test_pickup_exitoso():
    # Taxi en la parada 0 (0,0), pasajero en la parada 0, destino parada 2 (3,3)
    s = env.encode(0, 0, 0, 2)
    b0 = 6 + env.slack  # distancia Manhattan (0,0)-(3,3) + slack
    assert fila(s, PICKUP) == {env.encode(0, 0, 6, 2, b0): 1.0}
    assert R[s, PICKUP] == -1.0


def test_pickup_fallido():
    # Taxi en (0,0) pero el pasajero espera en la parada 1 (0,7)
    s = env.encode(0, 0, 1, 2)
    assert fila(s, PICKUP) == {s: 1.0}
    assert R[s, PICKUP] == -10.0


def test_dropoff_exitoso_es_terminal_y_paga_segun_estrellas():
    # Taxi en la parada 2 (3,3), pasajero a bordo con destino 2
    casos = {10: 30.0,   # 5 estrellas: 20 + 10
             7: 25.0,    # 4 estrellas: 20 + 5
             4: 20.0,    # 3 estrellas: 20 + 0
             1: 15.0,    # 2 estrellas: 20 - 5
             0: 10.0}    # 1 estrella:  20 - 10
    for b, recompensa in casos.items():
        s = env.encode(3, 3, 6, 2, b)
        assert fila(s, DROPOFF) == {}, "el estado terminal no debe tener masa en P"
        assert R[s, DROPOFF] == recompensa, f"b={b}"


def test_dropoff_fallido_gasta_tiempo():
    s = env.encode(0, 0, 6, 2, 10)  # a bordo pero lejos del destino
    assert fila(s, DROPOFF) == {env.encode(0, 0, 6, 2, 9): 1.0}
    assert R[s, DROPOFF] == -10.0


def test_movimiento_seco_y_mojado():
    s_seco = env.encode(2, 2, 0, 1)
    assert list(fila(s_seco, EAST).values()) == [1.0]

    s_mojado = env.encode(4, 2, 0, 1)
    ramas = fila(s_mojado, EAST)
    assert len(ramas) == 3
    assert np.allclose(sorted(ramas.values()), [0.1, 0.1, 0.8])
    assert env.encode(4, 3, 0, 1) in ramas and np.isclose(ramas[env.encode(4, 3, 0, 1)], 0.8)
    assert R[s_mojado, EAST] == -1.0


def test_sin_lluvia_todo_es_determinista():
    env0 = MilanRainTaxiEnv(slip_prob=0.0)
    P0, _ = build_sparse_mdp(env0)
    s_mojado = env0.encode(4, 2, 0, 1)
    assert list(fila(s_mojado, EAST, P0).values()) == [1.0]
    # P tiene a lo sumo un sucesor por (s, a)
    assert np.diff(P0.indptr).max() <= 1


def test_presupuesto_baja_un_paso_y_no_pasa_de_cero():
    s = env.encode(2, 2, 6, 2, 10)
    assert fila(s, EAST) == {env.encode(2, 3, 6, 2, 9): 1.0}
    s0 = env.encode(2, 2, 6, 2, 0)
    assert fila(s0, EAST) == {env.encode(2, 3, 6, 2, 0): 1.0}
