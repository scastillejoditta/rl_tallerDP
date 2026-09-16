import json
from pathlib import Path

import numpy as np
import yaml

from rl_tallerdp.agents.dynamic_programming import policy_iteration, value_iteration
from rl_tallerdp.envs.milan_taxi import MilanTaxiEnv
from rl_tallerdp.evaluation import evaluate_policy, random_policy
from rl_tallerdp.models.build_mdp import N_ACTIONS, N_STATES, build_model
from rl_tallerdp.utils.visualization import plot_vi_convergence

RAIZ = Path(__file__).resolve().parent
RUNS = RAIZ / "runs"


def _leer_yaml(nombre):
    with open(RAIZ / "configs" / nombre, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _un_ambiente(prob_resbale, cfg_pi, cfg_vi, n_episodios):
    env = MilanTaxiEnv(prob_resbale=prob_resbale)
    P, R = build_model(env)
    if not np.allclose(P.sum(axis=2), 1.0):
        raise RuntimeError("P está mal: alguna fila no suma 1")

    V_pi, pi_pi, n_pi = policy_iteration(
        P, R, gamma=cfg_pi["gamma"], tol=cfg_pi["tol"], max_iter=cfg_pi["max_iter"]
    )
    V_vi, pi_vi, n_vi, deltas = value_iteration(
        P, R, gamma=cfg_vi["gamma"], tol=cfg_vi["tol"], max_iter=cfg_vi["max_iter"]
    )

    gamma = cfg_vi["gamma"]
    ev_rnd = evaluate_policy(random_policy(), prob_resbale, gamma, n_episodios, seed=0)
    ev_pi = evaluate_policy(pi_pi, prob_resbale, gamma, n_episodios, seed=1)
    ev_vi = evaluate_policy(pi_vi, prob_resbale, gamma, n_episodios, seed=2)

    return {
        "prob_resbale": prob_resbale,
        "n_estados": N_STATES,
        "n_acciones": N_ACTIONS,
        "policy_iteration": {
            "iteraciones": n_pi,
            "V_media": float(V_pi.mean()),
            "partidas": ev_pi,
        },
        "value_iteration": {
            "sweeps": n_vi,
            "delta_final": float(deltas[-1]),
            "deltas": [float(x) for x in deltas],
            "V_media": float(V_vi.mean()),
            "partidas": ev_vi,
        },
        "aleatoria": ev_rnd,
        "pi_de_PI_igual_a_VI": bool(np.array_equal(pi_pi, pi_vi)),
        "estados_con_accion_distinta_PI_VI": int(
            np.sum(np.argmax(pi_pi, axis=1) != np.argmax(pi_vi, axis=1))
        ),
        "max_abs_V_PI_menos_VI": float(np.max(np.abs(V_pi - V_vi))),
        "_pi": pi_pi,
        "_V": V_pi,
    }


def _imprimir(r, n_pi_distinta_entre_ambientes):
    print("Estados:", N_STATES, "| acciones:", N_ACTIONS)
    print()
    for nombre, b in r.items():
        if not isinstance(b, dict):
            continue
        print(f"=== {nombre} (resbale={b['prob_resbale']}) ===")
        print(
            "  PI:",
            b["policy_iteration"]["iteraciones"],
            "iteraciones | V media",
            round(b["policy_iteration"]["V_media"], 2),
            "| retorno",
            round(b["policy_iteration"]["partidas"]["retorno_medio"], 2),
            "| entregas",
            round(b["policy_iteration"]["partidas"]["entregas_medias"], 2),
        )
        print(
            "  VI:",
            b["value_iteration"]["sweeps"],
            "sweeps | V media",
            round(b["value_iteration"]["V_media"], 2),
            "| retorno",
            round(b["value_iteration"]["partidas"]["retorno_medio"], 2),
            "| entregas",
            round(b["value_iteration"]["partidas"]["entregas_medias"], 2),
        )
        print(
            "  aleatoria: retorno",
            round(b["aleatoria"]["retorno_medio"], 2),
            "| entregas",
            round(b["aleatoria"]["entregas_medias"], 2),
        )
        print(
            "  ¿misma pi PI vs VI?",
            b["pi_de_PI_igual_a_VI"],
            "| acciones distintas",
            b["estados_con_accion_distinta_PI_VI"],
            "| max |V_PI - V_VI|",
            f"{b['max_abs_V_PI_menos_VI']:.4f}",
        )
        print()
    print("Estados con π* distinta (original vs resbale):", n_pi_distinta_entre_ambientes)


def main():
    cfg_pi = _leer_yaml("policy_iteration.yaml")
    cfg_vi = _leer_yaml("value_iteration.yaml")
    cfg_exp = _leer_yaml("experiment.yaml")
    n_episodios = cfg_exp["n_episodios"]

    RUNS.mkdir(exist_ok=True)
    resultados = {
        "sin_resbale": _un_ambiente(0.0, cfg_pi, cfg_vi, n_episodios),
        "con_resbale": _un_ambiente(cfg_exp["prob_resbale"], cfg_pi, cfg_vi, n_episodios),
    }

    n_pi_distinta = int(
        np.sum(
            np.argmax(resultados["sin_resbale"]["_pi"], axis=1)
            != np.argmax(resultados["con_resbale"]["_pi"], axis=1)
        )
    )

    plot_vi_convergence(
        {
            "sin resbale": resultados["sin_resbale"]["value_iteration"]["deltas"],
            "con resbale 0.2": resultados["con_resbale"]["value_iteration"]["deltas"],
        },
        RUNS / "convergencia_value_iteration.png",
    )

    para_json = {}
    for k, v in resultados.items():
        copia = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        copia["value_iteration"] = dict(copia["value_iteration"])
        copia["value_iteration"].pop("deltas")
        para_json[k] = copia
    para_json["estados_con_pi_distinta_original_vs_resbale"] = n_pi_distinta

    (RUNS / "resultados.json").write_text(
        json.dumps(para_json, indent=2), encoding="utf-8"
    )
    _imprimir(para_json, n_pi_distinta)
    print(f"Guardado en {RUNS / 'resultados.json'}")
    print(f"Curva en {RUNS / 'convergencia_value_iteration.png'}")
    return para_json


if __name__ == "__main__":
    main()
