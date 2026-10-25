"""Punto de entrada: resuelve el MDP con Policy Iteration y Value Iteration,
evalúa las políticas por simulación y registra todo en MLflow.

Uso:
    uv run python run_experiment.py
    uv run python run_experiment.py env.slip_prob=0.3
    uv run python run_experiment.py -m env.slip_prob=0.0,0.1,0.2,0.3
    uv run python run_experiment.py hydra.verbose=rl_tallerdp     # logs DEBUG
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
import logging
import time
from pathlib import Path

import hydra
import mlflow
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from rl_tallerdp.agents.dynamic_programming import (
    policy_iteration,
    q_from_v,
    value_iteration,
)
from rl_tallerdp.envs import MilanRainTaxiEnv
from rl_tallerdp.evaluation import (
    contar_diferencias_reales,
    contar_estados_suboptimos,
    evaluate_policy,
    random_policy,
)
from rl_tallerdp.utils.mdp import build_sparse_mdp
from rl_tallerdp.utils.visualization import plot_vi_convergence

log = logging.getLogger(__name__)


def _verificar_P(env):
    """Cada (s, a) debe tener probabilidades que sumen 1."""
    for s in env.P:
        for a in env.P[s]:
            total = sum(t[0] for t in env.P[s][a])
            if abs(total - 1.0) > 1e-9:
                raise RuntimeError(f"P mal construida: (s={s}, a={a}) suma {total}")


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def resolver(env, env_base, cfg):
    """Resuelve `env` con PI y VI, evalúa y compara con el entorno base.

    `cfg` solo se lee con cfg["clave"], así que sirve un DictConfig o un dict.
    Devuelve (resultados_json, deltas_vi, deltas_base, V_vi, pi_vi).
    """
    gamma, seed = cfg["gamma"], cfg["seed"]
    n_ep, max_steps = cfg["n_episodios"], cfg["max_steps"]
    c_pi, c_vi = cfg["policy_iteration"], cfg["value_iteration"]

    _verificar_P(env)
    P, R = build_sparse_mdp(env)
    log.info("MDP: %d estados, %d acciones, %d transiciones no nulas", env.nS, env.nA, P.nnz)

    t0 = time.perf_counter()
    V_pi, pi_pi, n_pi = policy_iteration(
        P, R, gamma=c_pi["gamma"], tol=c_pi["tol"], max_iter=c_pi["max_iter"]
    )
    t_pi = time.perf_counter() - t0
    if n_pi >= c_pi["max_iter"]:
        log.warning("Policy Iteration alcanzó max_iter=%d sin converger", n_pi)
    log.info("PI: %d iteraciones en %.2fs", n_pi, t_pi)

    t0 = time.perf_counter()
    V_vi, pi_vi, n_vi, deltas_vi = value_iteration(
        P, R, gamma=c_vi["gamma"], tol=c_vi["tol"], max_iter=c_vi["max_iter"]
    )
    t_vi = time.perf_counter() - t0
    if n_vi >= c_vi["max_iter"]:
        log.warning("Value Iteration alcanzó max_iter=%d sin converger", n_vi)
    log.info("VI: %d barridos en %.2fs (delta final %.2e)", n_vi, t_vi, deltas_vi[-1])

    # Entorno base (p. ej. sin lluvia): ¿qué tanto cambia la política óptima?
    P_b, R_b = build_sparse_mdp(env_base)
    _, pi_base, _, deltas_base = value_iteration(
        P_b, R_b, gamma=c_vi["gamma"], tol=c_vi["tol"], max_iter=c_vi["max_iter"]
    )
    Q = q_from_v(P, R, V_vi, c_vi["gamma"])

    log.info("Evaluando políticas (%d episodios, máx. %d pasos)", n_ep, max_steps)
    ev_rnd = evaluate_policy(env, random_policy(env.nS, env.nA), gamma, n_ep, max_steps, seed)
    ev_pi = evaluate_policy(env, pi_pi, gamma, n_ep, max_steps, seed + 1)
    ev_vi = evaluate_policy(env, pi_vi, gamma, n_ep, max_steps, seed + 2)

    resultados = {
        "slip_prob": env.slip_prob,
        "n_estados": env.nS,
        "n_acciones": env.nA,
        "n_transiciones_no_nulas": int(P.nnz),
        "policy_iteration": {
            "iteraciones": n_pi,
            "tiempo_s": t_pi,
            "V_media": float(V_pi.mean()),
            "V_inicial": float(env.initial_state_distrib @ V_pi),
            "partidas": ev_pi,
        },
        "value_iteration": {
            "sweeps": n_vi,
            "tiempo_s": t_vi,
            "delta_final": float(deltas_vi[-1]),
            "V_media": float(V_vi.mean()),
            "V_inicial": float(env.initial_state_distrib @ V_vi),
            "partidas": ev_vi,
        },
        "aleatoria": ev_rnd,
        "max_abs_V_PI_menos_VI": float(np.max(np.abs(V_pi - V_vi))),
        "estados_con_diferencia_real_PI_VI": contar_diferencias_reales(Q, pi_pi, pi_vi),
        "estados_con_accion_distinta_PI_VI": int(
            np.sum(pi_pi.argmax(axis=1) != pi_vi.argmax(axis=1))
        ),
        "estados_con_mas_de_una_accion_optima": int(
            np.sum((Q.max(axis=1, keepdims=True) - Q < 1e-8).sum(axis=1) > 1)
        ),
        "baseline_slip_prob": env_base.slip_prob,
        "estados_donde_politica_base_es_suboptima": contar_estados_suboptimos(Q, pi_base),
    }
    return resultados, deltas_vi, deltas_base, V_vi, pi_vi


def _resumen(r):
    for nombre in ("policy_iteration", "value_iteration"):
        b = r[nombre]
        log.info(
            "%s: V inicial %.2f | retorno %.2f | estrellas %.2f | pasos %.1f | entrega %.0f%%",
            nombre, b["V_inicial"], b["partidas"]["retorno_medio"],
            b["partidas"]["estrellas_medias"] or float("nan"),
            b["partidas"]["pasos_medios"] or float("nan"),
            100 * b["partidas"]["tasa_entrega"],
        )
    log.info("aleatoria: retorno %.2f | entrega %.0f%%",
             r["aleatoria"]["retorno_medio"], 100 * r["aleatoria"]["tasa_entrega"])
    log.info("PI vs VI: max|dV|=%.2e | diferencias reales=%d | empates (acción distinta)=%d",
             r["max_abs_V_PI_menos_VI"], r["estados_con_diferencia_real_PI_VI"],
             r["estados_con_accion_distinta_PI_VI"])
    log.info("Estados donde la política del entorno base es subóptima: %d",
             r["estados_donde_politica_base_es_suboptima"])


@hydra.main(version_base=None, config_path="configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
    log.info("Configuración:\n%s", OmegaConf.to_yaml(cfg))
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    env = MilanRainTaxiEnv(**env_kwargs)
    env_base = MilanRainTaxiEnv(**{**env_kwargs, "slip_prob": cfg.baseline_slip_prob})

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"slip_{cfg.env.slip_prob}"):
        mlflow.log_params(_flatten(OmegaConf.to_container(cfg, resolve=True)))

        resultados, deltas_vi, deltas_base, V_vi, pi_vi = resolver(env, env_base, cfg)
        _resumen(resultados)

        # Métricas escalares y curva de convergencia
        metricas = {
            k: v for k, v in _flatten(resultados).items()
            if isinstance(v, (int, float)) and v is not None
        }
        mlflow.log_metrics(metricas)
        for i, delta in enumerate(deltas_vi):
            mlflow.log_metric("vi_delta", float(delta), step=i)

        # Artefactos
        fig = out_dir / "convergencia_value_iteration.png"
        plot_vi_convergence(
            {
                f"p={cfg.env.slip_prob}": [float(x) for x in deltas_vi],
                f"p={cfg.baseline_slip_prob}": [float(x) for x in deltas_base],
            },
            fig,
        )
        (out_dir / "resultados.json").write_text(
            json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        np.save(out_dir / "V_vi.npy", V_vi)
        np.save(out_dir / "politica_vi.npy", pi_vi.argmax(axis=1).astype(np.int8))
        for nombre in ("convergencia_value_iteration.png", "resultados.json",
                       "V_vi.npy", "politica_vi.npy"):
            mlflow.log_artifact(str(out_dir / nombre))

    log.info("Resultados guardados en %s", out_dir)


if __name__ == "__main__":
    main()
