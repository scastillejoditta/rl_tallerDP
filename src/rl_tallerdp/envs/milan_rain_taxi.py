"""Milan Rain Taxi: variante de Taxi-v3 para Programación Dinámica.

Modificaciones respecto al Taxi original:
  1. Grilla 8x8 con 6 paradas (en vez de 5x5 con 4).
  2. Lluvia: en las celdas de la zona mojada ("Navigli") el taxi se desliza:
     con probabilidad `slip_prob` se desvía a una de las dos direcciones
     perpendiculares (p/2 cada una). Fuera de la zona, las transiciones son
     deterministas.
  3. Calificación del pasajero por velocidad: al subir, el taxi recibe un
     "presupuesto de tiempo" b = distancia_manhattan(origen, destino) + slack.
     Cada paso dentro del taxi lo reduce en 1 (mínimo 0). Al bajar al pasajero
     en su destino, las estrellas dependen de b:
         b >= t5 -> 5 estrellas, b >= t4 -> 4, b >= t3 -> 3, b >= t2 -> 2, si no 1.
     Cada calificación suma una propina (o penalización) a la recompensa.
     El presupuesto b hace parte del estado para que el MDP siga siendo Markov.

Acciones: 0 sur, 1 norte, 2 este, 3 oeste, 4 recoger, 5 dejar.
Estado: (fila, col, pasajero, destino, b)
    pasajero in 0..n_stops-1 -> esperando en esa parada (b = 0)
    pasajero == n_stops      -> dentro del taxi (b = presupuesto restante)
Expone `P[s][a] = [(prob, s', recompensa, terminado), ...]` como Taxi-v3.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Direcciones de movimiento (fila, columna) para S, N, E, W
_DELTAS = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
# Direcciones perpendiculares a cada movimiento (las posibles al resbalar)
_PERP = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}

DEFAULT_STOPS = ((0, 0), (0, 7), (3, 3), (5, 2), (7, 0), (7, 7))
# (r, c) en el conjunto: muro entre (r, c) y (r, c + 1)
DEFAULT_WALLS = frozenset({(0, 3), (1, 3), (6, 3), (7, 3)})


class MilanRainTaxiEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        size: int = 8,
        stops=DEFAULT_STOPS,
        walls=DEFAULT_WALLS,
        slippery_rows=(4, 5),
        slip_prob: float = 0.2,
        step_reward: float = -1.0,
        illegal_reward: float = -10.0,
        delivery_reward: float = 20.0,
        slack: int = 12,
        star_thresholds=(10, 7, 4, 1),  # b mínimo para 5, 4, 3 y 2 estrellas
        tips=(-10.0, -5.0, 0.0, 5.0, 10.0),  # propina para 1..5 estrellas
        render_mode=None,
    ):
        super().__init__()
        assert 0.0 <= slip_prob <= 1.0
        self.size = size
        self.stops = tuple(tuple(s) for s in stops)
        self.walls = frozenset(walls)
        self.slippery_rows = frozenset(slippery_rows)
        self.slip_prob = slip_prob
        self.step_reward = step_reward
        self.illegal_reward = illegal_reward
        self.delivery_reward = delivery_reward
        self.slack = slack
        self.star_thresholds = tuple(star_thresholds)
        self.tips = tuple(tips)
        self.render_mode = render_mode
        self.n_stops = len(self.stops)

        self._build_states()
        self._build_P()

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(6)

        # Inicio: taxi en cualquier celda, pasajero esperando en una parada,
        # con un destino distinto.
        self.initial_state_distrib = np.zeros(self.nS)
        for i, (_, _, p, _, _) in enumerate(self._states):
            if p < self.n_stops:
                self.initial_state_distrib[i] = 1.0
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.s = None

    # ------------------------------------------------------------------ #
    # Construcción del MDP
    # ------------------------------------------------------------------ #
    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _build_states(self):
        n = self.n_stops
        max_dist = max(self._manhattan(a, b) for a in self.stops for b in self.stops)
        self.b_max = max_dist + self.slack
        self._states, self._idx = [], {}

        def add(key):
            self._idx[key] = len(self._states)
            self._states.append(key)

        for r in range(self.size):
            for c in range(self.size):
                for p in range(n):
                    for d in range(n):
                        if p != d:
                            add((r, c, p, d, 0))
        for r in range(self.size):
            for c in range(self.size):
                for d in range(n):
                    for b in range(self.b_max + 1):
                        add((r, c, n, d, b))
        self.nS, self.nA = len(self._states), 6

    def decode(self, s):
        return self._states[s]

    def encode(self, r, c, p, d, b=0):
        return self._idx[(r, c, p, d, b)]

    def stars(self, b: int) -> int:
        t5, t4, t3, t2 = self.star_thresholds
        if b >= t5:
            return 5
        if b >= t4:
            return 4
        if b >= t3:
            return 3
        if b >= t2:
            return 2
        return 1

    def _blocked(self, r, c, nr, nc):
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return True
        if nc == c + 1 and (r, c) in self.walls:
            return True
        if nc == c - 1 and (r, nc) in self.walls:
            return True
        return False

    def _move_outcomes(self, r, c, a):
        """Lista (prob, (nueva_fila, nueva_col)) para una acción de movimiento."""
        if r in self.slippery_rows and self.slip_prob > 0:
            dirs = [(a, 1 - self.slip_prob)] + [(q, self.slip_prob / 2) for q in _PERP[a]]
        else:
            dirs = [(a, 1.0)]
        out = []
        for direction, prob in dirs:
            dr, dc = _DELTAS[direction]
            nr, nc = r + dr, c + dc
            if self._blocked(r, c, nr, nc):
                nr, nc = r, c
            out.append((prob, (nr, nc)))
        return out

    def _build_P(self):
        n = self.n_stops
        self.P = {}
        for s, (r, c, p, d, b) in enumerate(self._states):
            in_taxi = p == n
            b_next = max(b - 1, 0) if in_taxi else 0
            self.P[s] = {}
            for a in range(6):
                acc = {}  # (s', r, done) -> prob

                def put(prob, ns, rew, done):
                    acc[(ns, rew, done)] = acc.get((ns, rew, done), 0.0) + prob

                if a < 4:
                    for prob, (nr, nc) in self._move_outcomes(r, c, a):
                        put(prob, self._idx[(nr, nc, p, d, b_next)], self.step_reward, False)
                elif a == 4:  # recoger
                    if (not in_taxi) and (r, c) == self.stops[p]:
                        nb = self._manhattan(self.stops[p], self.stops[d]) + self.slack
                        put(1.0, self._idx[(r, c, n, d, nb)], self.step_reward, False)
                    else:
                        put(1.0, self._idx[(r, c, p, d, b_next)], self.illegal_reward, False)
                else:  # dejar
                    if in_taxi and (r, c) == self.stops[d]:
                        tip = self.tips[self.stars(b) - 1]
                        put(1.0, s, self.delivery_reward + tip, True)
                    else:
                        put(1.0, self._idx[(r, c, p, d, b_next)], self.illegal_reward, False)
                self.P[s][a] = [(pr, ns, rw, dn) for (ns, rw, dn), pr in acc.items()]

    def to_arrays(self):
        """Arreglos (nS, nA, K) con K = máximo de sucesores, para DP vectorizada.

        Devuelve probs, next_states, rewards, dones. Las posiciones de relleno
        tienen prob = 0.
        """
        K = max(len(self.P[s][a]) for s in self.P for a in self.P[s])
        probs = np.zeros((self.nS, self.nA, K))
        nxt = np.zeros((self.nS, self.nA, K), dtype=np.int64)
        rew = np.zeros((self.nS, self.nA, K))
        done = np.zeros((self.nS, self.nA, K), dtype=bool)
        for s in range(self.nS):
            for a in range(self.nA):
                for k, (pr, ns, rw, dn) in enumerate(self.P[s][a]):
                    probs[s, a, k], nxt[s, a, k], rew[s, a, k], done[s, a, k] = pr, ns, rw, dn
        return probs, nxt, rew, done

    # ------------------------------------------------------------------ #
    # API de Gymnasium
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = int(self.np_random.choice(self.nS, p=self.initial_state_distrib))
        return self.s, {"prob": 1.0}

    def step(self, a):
        trans = self.P[self.s][int(a)]
        i = int(self.np_random.choice(len(trans), p=[t[0] for t in trans]))
        prob, ns, rew, done = trans[i]
        info = {"prob": prob}
        if done:
            info["stars"] = self.stars(self.decode(self.s)[4])
        self.s = ns
        return int(ns), rew, bool(done), False, info

    def render(self):
        r, c, p, d, b = self.decode(self.s)
        grid = [["~" if i in self.slippery_rows else "." for _ in range(self.size)]
                for i in range(self.size)]
        for k, (sr, sc) in enumerate(self.stops):
            grid[sr][sc] = str(k)
        grid[r][c] = "T"
        text = "\n".join(" ".join(row) for row in grid)
        where = "en taxi" if p == self.n_stops else f"esperando en {p}"
        return f"{text}\npasajero {where} -> destino {d}, presupuesto b={b}"
