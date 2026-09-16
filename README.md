# MilanTaxi — Programación Dinámica (Taller RL)

Proyecto del taller de Aprendizaje por Refuerzo: resolver el ambiente **MilanTaxi** mediante métodos de Programación Dinámica (Policy Iteration y Value Iteration), partiendo de un MDP completamente conocido.

**Modificación del ambiente:** al moverse (`NORTH/SOUTH/EAST/WEST`), con probabilidad **0.2** el taxi resbala y se queda en el sitio. Recoger, dejar y el spawn del siguiente pasajero no cambian.

**Estudiantes:** Sara Castillejo, Stefany Mojica y Alexander Pineda.

---

## 1. Definición del problema

### 1.1 El problema, en términos concretos

Un taxi se mueve en una grilla de **5×5** celdas. En cuatro esquinas de la grilla hay puntos de recogida/entrega (`LOCS`). En cada momento, hay un pasajero esperando en una de esas esquinas, que quiere ir a otra de ellas. El taxi debe:

1. Moverse hasta la celda donde está el pasajero.
2. Ejecutar `PICKUP` para recogerlo.
3. Moverse hasta la celda de destino del pasajero.
4. Ejecutar `DROPOFF` para entregarlo.

Al entregarlo con éxito, aparece un nuevo pasajero (con un nuevo par recogida→destino, elegido al azar), y el proceso se repite indefinidamente — **es una tarea continua, sin un estado terminal real** (solo se trunca la simulación después de `HORIZON=100` pasos, que es un límite de simulación, no del problema en sí).

Hay paredes internas en la grilla (`INTERNAL_WALLS`) que bloquean ciertos movimientos horizontales entre celdas adyacentes.

### 1.2 El MDP formal

El problema se modela como un MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

**Estados** — cada estado es una tupla $s = (r, c, p, d)$, donde en el código (`build_mdp.py`) esto corresponde a `(row, col, pass_idx, dest_idx)`:
- $r, c \in \{0, \dots, 4\}$ (`row, col`): posición del taxi en la grilla.
- $p \in \{0,1,2,3,4\}$ (`pass_idx`): en cuál de las 4 esquinas está el pasajero esperando, o `4` si ya está dentro del taxi.
- $d \in \{0,1,2,3\}$ (`dest_idx`): esquina de destino del pasajero actual.

Total: $5 \times 5 \times 5 \times 4 = 500$ estados. Se codifican como un único entero mediante:

$$s = ((r \cdot 5 + c) \cdot 5 + p) \cdot 4 + d$$

**Acciones** — $\mathcal{A} = \{\text{SOUTH}, \text{NORTH}, \text{EAST}, \text{WEST}, \text{PICKUP}, \text{DROPOFF}\}$ (6 acciones discretas).

**Recompensas** — $R(s,a)$:
- $-1$ por cualquier movimiento (o intento de acción).
- $-10$ por un `PICKUP` o `DROPOFF` inválido (taxi no está en la celda correcta, o no hay pasajero que recoger/entregar).
- $+20$ por un `DROPOFF` exitoso.

**Dinámica de transición** — $P(s' \mid s, a)$:
- Un movimiento (`NORTH/SOUTH/EAST/WEST`) **sin resbale** (o contra pared/borde) es determinista: una sola transición con probabilidad 1.
- Un movimiento **con resbale 0.2** (si la casilla prevista es distinta de quedarse) tiene dos ramas: $0.8$ ir a la casilla prevista y $0.2$ quedarse. En ambos casos la recompensa es $-1$.
- `PICKUP` y `DROPOFF` **inválidos** son deterministas.
- Un `DROPOFF` **exitoso** es estocástico: el siguiente pasajero se elige uniforme entre las $4 \times 3 = 12$ combinaciones con recogida ≠ destino, cada una con probabilidad $1/12$. El resbale no aplica aquí.

**Ecuación de Bellman (forma general, con transiciones estocásticas):**

$$V^{\star}(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^{\star}(s') \right]$$

En el ambiente original, para la mayoría de $(s,a)$ la suma colapsa a un solo término, excepto el `DROPOFF` exitoso (12 términos). Con la modificación, cada movimiento libre suma **dos** términos.

**Descuento**: $\gamma = 0.95$ en los experimentos — necesario porque la tarea es continua (sin fin natural).

El ambiente por defecto es el modificado (`MilanTaxiEnv()`, `prob_resbale=0.2`). El original se obtiene con `MilanTaxiEnv(prob_resbale=0.0)`.

---

## 2. Estado actual del proyecto

```
rl_tallerDP/
├── configs/
│   ├── policy_iteration.yaml
│   ├── value_iteration.yaml
│   └── experiment.yaml
├── runs/
│   ├── resultados.json
│   └── convergencia_value_iteration.png
├── src/rl_tallerdp/
│   ├── envs/
│   │   └── milan_taxi.py              # MilanTaxi + resbale 0.2
│   ├── models/
│   │   └── build_mdp.py               # build_model(env) -> (P, R)
│   ├── agents/
│   │   └── dynamic_programming.py     # PI, VI, evaluación, Q, greedy
│   ├── evaluation.py                  # partidas: retorno y entregas
│   └── utils/
│       └── visualization.py           # slice de V y curva de VI
├── test_env.py
├── test_build_p.py
├── test_policy_iteration.py
├── run_experiment.py                  # original vs resbale, PI y VI
└── pyproject.toml
```

---

## 3. Cómo correrlo

Requiere [`uv`](https://docs.astral.sh/uv/) instalado.

```bash
uv sync

uv run python test_env.py
uv run python test_build_p.py
uv run python test_policy_iteration.py

# Compara original (resbale=0) vs modificado (resbale=0.2)
uv run python run_experiment.py
```

`run_experiment.py` escribe `runs/resultados.json` y `runs/convergencia_value_iteration.png`.
Los números de la sección 4 salen de esa corrida ($\gamma = 0.95$, horizonte 100, 60 episodios).

---

## 4. Preguntas guía del taller

### Pregunta 1 — ¿Cuál es exactamente el MDP que están resolviendo?

Ver la sección [1.2](#12-el-mdp-formal): 500 estados, 6 acciones, recompensas $-1 / -10 / +20$, y transiciones que mezclan el dropoff estocástico de fábrica con el **resbale 0.2** en los movimientos.

`build_model` lee `env.prob_resbale` y devuelve $P$ de forma `(500, 6, 500)` y $R$ de forma `(500, 6)`. El `step` sortea la misma moneda, así que DP y las partidas usan las mismas reglas.

### Pregunta 2 — ¿Qué cambia cuando las transiciones dejan de ser deterministas?

Matemáticamente, la ecuación de Bellman pasa de una forma "de un solo término":

$$V(s) = \max_a \left[ R(s,a) + \gamma\, V(s'_{a}) \right]$$

a la forma general con **esperanza sobre transiciones**:

$$V(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s, a)\, V(s') \right]$$

El backup ya no consulta un único sucesor: promedia. El taxi original ya tenía 12 ramas en un `DROPOFF` exitoso. La modificación agrega una segunda fuente de azar **en el movimiento**:

- $0.8$ · (casilla prevista, $-1$)
- $0.2$ · (quedarse, $-1$)

Si pared o borde hacen que “avanzar” = quedarse, la fila sigue siendo determinista (probabilidad 1). `PICKUP`/`DROPOFF` no resbalan.

El algoritmo (PI / VI) no cambia: opera sobre $P$ y $R$. Lo que cambia es la tabla $P$. Efecto medido: $V_*$ media baja de **7.37 a 2.07**.

### Pregunta 3 — Diferencia fundamental entre Policy Iteration y Value Iteration

No es solo "uno actualiza políticas y el otro valores" — la diferencia está en **qué tan exactamente se evalúa la política en cada paso**:

- **Value Iteration**: en cada iteración aplica el operador de Bellman de *optimalidad* (con el `max`) una sola vez por estado. No mantiene una política dentro del loop; se extrae al final con un greedy sobre $Q(\cdot; V^*)$.
- **Policy Iteration**: primero **evalúa exactamente** $V^\pi$ (sistema lineal $V = (I - \gamma P^\pi)^{-1} R^\pi$) y *después* mejora con greedy respecto a ese $V^\pi$ ya exacto.

Números sobre MilanTaxi (500 estados, $\gamma = 0.95$):

| | PI (iteraciones externas) | VI (sweeps) | $\max |V_{\mathrm{PI}}-V_{\mathrm{VI}}|$ |
|---|---|---|---|
| sin resbale | 11 | 277 | $< 10^{-5}$ |
| con resbale 0.2 | 11 | 247 | $< 10^{-4}$ |

Pocas iteraciones caras (PI) vs. muchos sweeps baratos (VI). Las $V_*$ coinciden. Las $\pi$ **no** son el mismo vector: hay empates (36 estados sin resbale, 59 con resbale). Eso no es un fallo — varias acciones son óptimas a la vez.

### Pregunta 4 — ¿Cómo saben que el algoritmo convergió y que la política es adecuada?

**Convergencia (números internos).**

- VI: $\|V_{k+1} - V_k\|_\infty < 10^{-6}$ (curva en `runs/convergencia_value_iteration.png`).
- PI: $\|V^{\pi_{k+1}} - V^{\pi_k}\|_\infty < 10^{-9}$. Parar por igualdad bit a bit de $\pi$ no sirve: hay acciones empatadas y el `argmax` puede oscilar. $V$ sí se estabiliza.
- Los dos métodos dan la misma $V_*$.

**Que la política sirve (el juego).** Eso no lo dice $\Delta$. `evaluation.py` simula 60 episodios de 100 pasos y compara contra una política aleatoria:

| | retorno medio (desc. $\gamma$) | entregas / 100 pasos |
|---|---|---|
| aleatoria, sin resbale | −79.6 | 0.03 |
| PI / VI, sin resbale | ~2.7 / ~2.9 | ~6.8 / ~7.0 |
| aleatoria, con resbale | −79.7 | 0.03 |
| PI / VI, con resbale | ~−1.6 / ~−2.8 | ~5.7 |

La $\pi_*$ entrega dos órdenes de magnitud más que el azar. El retorno de las partidas no es igual a $V_*$ media: $V$ es horizonte infinito descontado; las partidas cortan a 100 pasos.

### Pregunta 5 — ¿Qué efecto tiene la modificación sobre la dificultad?

El resbale **no** agranda el espacio de estados (siguen siendo 500). Cambia las **transiciones** y, con ellas, $v_*$ y lo difícil que es entregar.

- $V_*$ media: 7.37 → 2.07 (el futuro vale menos: a veces no avanzas).
- Entregas en 100 pasos: ~7.0 → ~5.7.
- Retorno en partidas: ~2.8 → ~−2.
- Bellman de un movimiento libre: 1 sucesor → 2.
- $\pi^*$ cambia en **67** de 500 estados (original vs resbale).
- PI sigue en 11 iteraciones; VI baja de 277 a 247 sweeps (el $\gamma$-contracción es la misma, el $P$ distinto mueve el camino al punto fijo).

La política óptima sigue existiendo; el mundo es más ruidoso, entonces el valor óptimo baja. Eso es “más difícil”, no “PI/VI dejan de funcionar”.

---

## 5. Cómo contribuir (para el equipo)

1. Clonar el repo y correr `uv sync`.
2. Los scripts de prueba y el experimento están en la raíz (`test_*.py`, `run_experiment.py`).
3. Cualquier cambio a las reglas del taxi tiene que verse en `step` **y** en `build_model` (si no, DP y las partidas no coinciden).
