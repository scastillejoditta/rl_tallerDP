"""Evaluación de políticas por simulación y comparación de políticas."""
import numpy as np


def random_policy(n_states, n_actions):
    """Política uniforme aleatoria como matriz (nS, nA)."""
    return np.full((n_states, n_actions), 1.0 / n_actions)


def evaluate_policy(env, pi, gamma, n_episodes, max_steps=300, seed=0):
    """Simula `n_episodes` episodios con la política `pi` (matriz nS x nA).

    Un episodio termina al entregar al pasajero o al llegar a `max_steps`.
    Devuelve el retorno medio descontado, la tasa de entrega, los pasos medios
    (solo de episodios con entrega), las estrellas medias y la fracción de
    entregas con 1..5 estrellas.
    """
    n_actions = pi.shape[1]
    cdf = np.cumsum(pi, axis=1)
    rng = np.random.default_rng(seed)

    returns, steps, stars = [], [], []
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed if ep == 0 else None)
        G, discount = 0.0, 1.0
        for t in range(max_steps):
            a = min(int(np.searchsorted(cdf[s], rng.random(), side="right")), n_actions - 1)
            s, r, terminated, _, info = env.step(a)
            G += discount * r
            discount *= gamma
            if terminated:
                steps.append(t + 1)
                stars.append(info["stars"])
                break
        returns.append(G)

    n_deliv = len(stars)
    out = {
        "retorno_medio": float(np.mean(returns)),
        "tasa_entrega": n_deliv / n_episodes,
        "pasos_medios": float(np.mean(steps)) if n_deliv else None,
        "estrellas_medias": float(np.mean(stars)) if n_deliv else None,
    }
    for k in range(1, 6):
        out[f"frac_{k}_estrellas"] = (stars.count(k) / n_deliv) if n_deliv else 0.0
    return out


def contar_diferencias_reales(Q, pi_a, pi_b, tol=1e-8):
    """Estados donde pi_a y pi_b eligen acciones con Q distinto (más que `tol`).

    Comparar argmax directamente cuenta como "diferencia" los empates entre
    acciones igual de buenas; aquí solo se cuentan las diferencias que cambian
    el valor esperado.
    """
    idx = np.arange(Q.shape[0])
    a, b = pi_a.argmax(axis=1), pi_b.argmax(axis=1)
    return int(np.sum(np.abs(Q[idx, a] - Q[idx, b]) > tol))


def contar_estados_suboptimos(Q, pi, tol=1e-8):
    """Estados donde la acción de `pi` no es óptima bajo la función Q dada."""
    idx = np.arange(Q.shape[0])
    a = pi.argmax(axis=1)
    return int(np.sum(Q.max(axis=1) - Q[idx, a] > tol))
