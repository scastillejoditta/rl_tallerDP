import numpy as np

from rl_tallerdp.envs.milan_taxi import HORIZON, MilanTaxiEnv
from rl_tallerdp.models.build_mdp import N_ACTIONS, N_STATES, encode_state


def random_policy(n_states=N_STATES, n_actions=N_ACTIONS):
    return np.ones((n_states, n_actions)) / n_actions


def choose_action(pi, s, rng):
    return int(rng.choice(pi.shape[1], p=pi[s]))


def evaluate_policy(pi, prob_resbale, gamma, n_episodios=60, seed=0, horizon=HORIZON):
    rng = np.random.default_rng(seed)
    env = MilanTaxiEnv(prob_resbale=prob_resbale)
    retornos = []
    entregas = []

    for _ in range(n_episodios):
        ep_seed = int(rng.integers(1_000_000_000))
        state, _ = env.reset(seed=ep_seed)
        retorno = 0.0
        descuento = 1.0
        while True:
            s = encode_state(*state)
            action = choose_action(pi, s, rng)
            state, reward, terminated, truncated, info = env.step(action)
            retorno += descuento * reward
            descuento *= gamma
            if terminated or truncated or env.time_step >= horizon:
                retornos.append(retorno)
                entregas.append(info["delivered_passengers"])
                break

    retornos = np.asarray(retornos)
    entregas = np.asarray(entregas)
    return {
        "retorno_medio": float(retornos.mean()),
        "retorno_std": float(retornos.std(ddof=1)) if n_episodios > 1 else 0.0,
        "entregas_medias": float(entregas.mean()),
        "n_episodios": n_episodios,
    }
