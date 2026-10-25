"""Tests del entorno MilanRainTaxi (grilla 8x8, lluvia en filas 4-5, calificación)."""
from rl_tallerdp.envs import MilanRainTaxiEnv

SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF = range(6)

env = MilanRainTaxiEnv()                    # lluvia p=0.2
env_seco = MilanRainTaxiEnv(slip_prob=0.0)  # sin lluvia


def _ramas(e, r, c, a, p=0, d=1, b=0):
    """Probabilidades (ordenadas) de los sucesores de la acción `a` en (r, c)."""
    s = e.encode(r, c, p, d, b)
    return sorted(round(prob, 6) for prob, *_ in e.P[s][a])


def _celda_siguiente(e, r, c, a):
    s = e.encode(r, c, 0, 1, 0)
    prob, ns, *_ = e.P[s][a][0]
    return e.decode(ns)[:2]


def test_dimensiones():
    assert env.nS == 12288
    assert env.nA == 6
    assert env.observation_space.n == env.nS
    assert env.action_space.n == 6


def test_reset_deja_al_pasajero_esperando():
    s, _ = env.reset(seed=0)
    r, c, p, d, b = env.decode(s)
    assert 0 <= r < env.size and 0 <= c < env.size
    assert p < env.n_stops and p != d and b == 0


def test_reset_con_semilla_es_reproducible():
    s1, _ = env.reset(seed=123)
    s2, _ = env.reset(seed=123)
    assert s1 == s2


def test_step_devuelve_formato_gymnasium():
    env.reset(seed=0)
    s, reward, terminated, truncated, info = env.step(SOUTH)
    assert 0 <= s < env.nS
    assert reward == -1.0
    assert terminated is False and truncated is False


def test_resbalar_solo_en_zona_mojada():
    assert _ramas(env, 2, 2, EAST) == [1.0]                # fila seca
    assert _ramas(env, 4, 2, EAST) == [0.1, 0.1, 0.8]      # fila mojada
    assert _ramas(env_seco, 4, 2, EAST) == [1.0]           # sin lluvia


def test_muros_y_bordes_bloquean():
    assert _celda_siguiente(env, 0, 3, EAST) == (0, 3)   # muro entre (0,3) y (0,4)
    assert _celda_siguiente(env, 0, 4, WEST) == (0, 4)   # el mismo muro, desde el otro lado
    assert _celda_siguiente(env, 0, 0, NORTH) == (0, 0)  # borde
    assert _celda_siguiente(env, 7, 7, EAST) == (7, 7)   # borde


def test_estrellas_segun_presupuesto():
    esperado = {10: 5, 9: 4, 7: 4, 6: 3, 4: 3, 3: 2, 1: 2, 0: 1}
    for b, estrellas in esperado.items():
        assert env.stars(b) == estrellas, f"b={b}"


def test_entrega_termina_el_episodio_con_propina():
    env.reset(seed=0)
    env.s = env.encode(3, 3, 6, 2, 10)  # taxi en la parada 2, pasajero a bordo, b=10
    _, reward, terminated, _, info = env.step(DROPOFF)
    assert terminated is True
    assert info["stars"] == 5
    assert reward == 20.0 + 10.0
