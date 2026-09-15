# MilanTaxi — Programación Dinámica (Taller RL)

Proyecto del taller de Aprendizaje por Refuerzo: resolver el ambiente **MilanTaxi** mediante métodos de Programación Dinámica (Policy Iteration y Value Iteration), partiendo de un MDP completamente conocido.

Estudiantes: Sara Castillejo, Stefany Mojica y Alexander Pineda. 

Universidad del Rosario
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
- Los movimientos (`NORTH/SOUTH/EAST/WEST`) y los `PICKUP`/`DROPOFF` **inválidos o fallidos** son **deterministas**: una sola transición con probabilidad 1.
- Un `DROPOFF` **exitoso** es **estocástico**: dispara la aparición de un nuevo pasajero elegido uniformemente entre las $4 \times 3 = 12$ combinaciones posibles de (recogida, destino) con recogida ≠ destino, cada una con probabilidad $1/12$.

**Ecuación de Bellman (forma general, con transiciones estocásticas):**

$$V^*(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^*(s') \right]$$

En este ambiente, para la mayoría de $(s,a)$ la suma colapsa a un solo término (transición determinista), excepto para los pares $(s, \text{DROPOFF})$ donde el dropoff es exitoso, donde la suma tiene 12 términos igualmente ponderados.

**Descuento**: $\gamma \in (0,1)$ — necesario porque la tarea es continua (sin fin natural), así que un horizonte infinito sin descuento no converge.

---

## 2. Estado actual del proyecto

```
rl_TallerDP/
├── configs/                     # (pendiente) YAMLs de configuración por algoritmo
├── notebooks/                   # (opcional) notebooks de exploración
├── runs/                        # (pendiente) resultados de cada ejecución
├── src/rl_tallerdp/
│   ├── envs/
│   │   └── milan_taxi.py        # ✅ Ambiente MilanTaxi (gym.Env)
│   ├── models/
│   │   └── build_mdp.py         # ✅ build_model(env) -> (P, R) como arrays densos
│   ├── agents/
│   │   └── dynamic_programming.py   # ✅ policy_matrices, solve_policy_direct,
│   │                                 #    iterative_policy_evaluation, q_from_v,
│   │                                 #    greedy_policy, policy_iteration, value_iteration
│   ├── policies.py               # (pendiente)
│   └── utils/                    # (pendiente) visualización, tracking, configs
├── test_env.py                   # ✅ prueba manual de reset()/step()
├── test_build_p.py               # ✅ verifica P y R (formas, suma de probabilidades, casos concretos)
├── test_policy_iteration.py      # ✅ corre policy_iteration end-to-end
└── pyproject.toml
```

### Qué falta por hacer
- [ ] Modificar el ambiente (tamaño de grilla, recompensas, o transiciones no deterministas adicionales — pregunta guía 5).
- [ ] `evaluation.py` con métricas de calidad de política (simulación de episodios, recompensa promedio, etc.).
- [ ] `configs/*.yaml` + Hydra para parametrizar experimentos.
- [ ] Tracking de experimentos (MLflow) — comparar ambiente original vs modificado.
- [ ] Visualización de $V^*$ y la política (slices de la grilla por `pass_idx`/`dest_idx`, ya que el estado no es una simple posición 2D).
- [ ] `cli.py` para correr experimentos desde línea de comandos.

---

## 3. Cómo correrlo

Requiere [`uv`](https://docs.astral.sh/uv/) instalado.

```bash
# Instalar dependencias (crea el entorno virtual automáticamente)
uv sync

# Probar que el ambiente corre
uv run python test_env.py

# Construir el MDP (P, R) y verificar su consistencia
uv run python test_build_p.py

# Correr Policy Iteration completo sobre el MDP
uv run python test_policy_iteration.py
```

Todos los scripts de prueba están en la raíz del proyecto (mismo nivel que `pyproject.toml`).

---

## 4. Preguntas guía del taller

### ✅ Pregunta 1 — ¿Cuál es exactamente el MDP que están resolviendo?

Ver la sección [1.2](#12-el-mdp-formal) arriba: estados, acciones, recompensas y dinámica de transición quedan completamente definidos ahí. El MDP se construye explícitamente en `models/build_mdp.py::build_model`, que devuelve $P$ como un array `(500, 6, 500)` y $R$ como `(500, 6)`, enumerando los 500 estados posibles y usando la lógica interna `env._transitions()` del ambiente para calcular cada transición.

### ✅ Pregunta 2 — ¿Qué cambia cuando las transiciones dejan de ser deterministas?

Matemáticamente, la ecuación de Bellman pasa de una forma "de un solo término":

$$V(s) = \max_a \left[ R(s,a) + \gamma\, V(s'_{a}) \right] \quad \text{(un único } s'_a \text{ por acción)}$$

a la forma general con **esperanza sobre transiciones**:

$$V(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s, a)\, V(s') \right]$$

En la práctica, esto significa que el "backup" de Bellman para cada $(s,a)$ ya no es una simple consulta a un único estado siguiente, sino un **promedio ponderado** sobre todos los estados siguientes posibles. Nuestro ambiente MilanTaxi **ya tiene esto integrado de fábrica**: cuando un `DROPOFF` es exitoso, el siguiente estado es una de 12 posibilidades equiprobables (nuevo pasajero al azar), así que `build_model` ya construye $P$ con sumas de 12 términos para esos casos, y con un único término (probabilidad 1) para el resto.

### ✅ Pregunta 3 — Diferencia fundamental entre Policy Iteration y Value Iteration

No es solo "uno actualiza políticas y el otro valores" — la diferencia real está en **qué tan exactamente se evalúa la política en cada paso**:

- **Value Iteration**: en cada iteración, aplica el operador de Bellman de *optimalidad* (con el `max`) una sola vez por estado, sobre una estimación de $V$ que aún no es exacta para ninguna política. Nunca mantiene una política explícita hasta el final — la política solo se extrae al terminar.
- **Policy Iteration**: en cada iteración, primero **evalúa exactamente** $V^\pi$ para la política actual (resolviendo el sistema lineal $V = (I - \gamma P^\pi)^{-1} R^\pi$, sin ningún `max` de por medio), y solo *después* mejora la política tomando la acción greedy respecto a ese $V^\pi$ ya exacto.

En nuestros experimentos sobre MilanTaxi (500 estados): **Policy Iteration convergió en 12 iteraciones externas** (~0.2s), mientras que Value Iteration típicamente necesita muchas más iteraciones (cada una más barata) para alcanzar la misma precisión — el clásico trade-off de "pocas iteraciones caras" vs. "muchas iteraciones baratas".

### ✅ Pregunta 4 — ¿Cómo saben que el algoritmo convergió y que la política es adecuada?

**Para Value Iteration / evaluación iterativa**: convergencia cuando $\|V_{k+1} - V_k\|_\infty < \text{tol}$ (contracción $\gamma$ garantiza convergencia a un único punto fijo).

**Para Policy Iteration** — aquí encontramos un matiz importante en la práctica: nuestra primera implementación comparaba la política nueva contra la anterior bit a bit (`np.array_equal(pi_new, pi)`), y **no convergía nunca** (llegaba a las 1000 iteraciones máximas). Diagnosticamos que $V$ ya se había estabilizado exactamente (`max|V_k - V_{k-1}| = 0.0`) desde la iteración 11, pero la política seguía "cambiando" porque **existen estados con más de una acción óptima empatada exactamente** (ej. moverse al sur o al oeste llevan al mismo valor esperado, por simetría de la grilla) — y ruido de punto flotante de `np.linalg.solve` hacía que el desempate fuera inconsistente entre iteraciones.

**Corrección**: el criterio de convergencia correcto para Policy Iteration es sobre la **estabilización de $V$**, no sobre la igualdad exacta de $\pi$:

$$\|V^{\pi_{k+1}} - V^{\pi_k}\|_\infty < \text{tol} \implies \text{convergencia}$$

Esto es consistente con el teorema de mejora de política: $V^{\pi_{k+1}} \ge V^{\pi_k}$ siempre, con igualdad si y solo si ya se alcanzó el óptimo — sin importar cuál acción empatada haya elegido el desempate.

**Sobre la calidad de la política** (pendiente de implementar en `evaluation.py`): simular $N$ episodios con la política encontrada y medir recompensa promedio por episodio, tasa de entregas exitosas, y comparar contra una política aleatoria como baseline.

### ⬜ Pregunta 5 — ¿Qué efecto tiene la modificación del ambiente sobre la dificultad del problema?

*(Pendiente — aún no se ha decidido/implementado la modificación del ambiente.)*

Puntos a cubrir una vez se implemente:
- ¿Cambia el número de estados? (ej. si se modifica el tamaño de la grilla)
- ¿Cómo cambia la estructura de $P$? (ej. si se introduce no-determinismo adicional en el movimiento, más pares $(s,a)$ tendrán más de una rama en `P[s,a]`)
- ¿Cambia la política óptima? (comparar $\pi^*$ del ambiente original vs. modificado, estado por estado)
- ¿Cambia el número de iteraciones hasta convergencia de Policy Iteration / Value Iteration?
- ¿Cambia $V^*$ de forma esperable? (ej. más recompensa negativa acumulada si el ambiente es "más difícil")

---

## 5. Cómo contribuir (para el equipo)

1. Clonar el repo y correr `uv sync`.
2. Revisar la sección 2 ("qué falta por hacer") y tomar un ítem.
3. Cada nuevo módulo debería tener su propio script de prueba en la raíz (siguiendo el patrón `test_*.py` ya usado), antes de integrarse a `train.py`/`experiment.py`.
4. Documentar cualquier decisión de diseño (ej. la modificación elegida del ambiente) directamente en este README, en la sección de la Pregunta 5.

1. Clonar el repo y correr `uv sync`.
2. Revisar la sección 2 ("qué falta por hacer") y tomar un ítem.
3. Cada nuevo módulo debería tener su propio script de prueba en la raíz (siguiendo el patrón `test_*.py` ya usado), antes de integrarse a `train.py`/`experiment.py`.
4. Documentar cualquier decisión de diseño (ej. la modificación elegida del ambiente) directamente en este README, en la sección de la Pregunta 5.