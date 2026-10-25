# Milan Rain Taxi: Programación Dinámica con Policy Iteration y Value Iteration

**Estudiantes:** Sara Castillejo, Stefany Mojica y Alexander Pineda.

Un taxi recorre una grilla de Milán bajo la lluvia, recoge a un pasajero y lo lleva a su destino. Este proyecto resuelve ese problema como un Proceso de Decisión de Markov (MDP) finito con **Policy Iteration** y **Value Iteration**, sobre una versión modificada del entorno Milan Taxi (basado en Taxi-v3 de Gymnasium), y registra los experimentos con **Hydra**, **logging** y **MLflow**.

## Contenido

1. [Cómo reproducir el ejercicio](#1-cómo-reproducir-el-ejercicio)
2. [Descripción del ejercicio](#2-descripción-del-ejercicio)
3. [Definición del MDP](#3-definición-del-mdp)
4. [Modificación del entorno original](#4-modificación-del-entorno-original)
5. [Policy Iteration vs Value Iteration](#5-policy-iteration-vs-value-iteration)
6. [Convergencia y evaluación de la política](#6-convergencia-y-evaluación-de-la-política)
7. [Herramientas: Hydra, logging y MLflow](#7-herramientas-hydra-logging-y-mlflow)
8. [Resultados](#8-resultados)

---

## 1. Cómo reproducir el ejercicio

### Requisitos

- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes y entornos de Python). El proyecto fija la versión de Python en `.python-version` y las dependencias exactas en `uv.lock`; `uv` se encarga de instalar ambas.

Instalar `uv`:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>

# 2. Crear el entorno virtual e instalar las dependencias fijadas en uv.lock
uv sync

# 3. (Opcional) Verificar la instalación: deben pasar los 22 tests
uv run pytest -q

# 4. Ejecutar el experimento con la configuración por defecto
uv run python run_experiment.py

# 5. Abrir la interfaz de MLflow para ver y comparar las ejecuciones
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# luego abrir http://127.0.0.1:5000 en el navegador
```

### Variar parámetros (Hydra)

Cualquier parámetro de `configs/` se puede sobrescribir desde la línea de comandos:

```bash
# Otra probabilidad de resbalar
uv run python run_experiment.py env.slip_prob=0.3

# Otro factor de descuento
uv run python run_experiment.py gamma=0.95

# Barrido (una ejecución por valor): comparar entornos con distinta lluvia
uv run python run_experiment.py -m env.slip_prob=0.0,0.1,0.2,0.3,0.5

# Barrido de dos parámetros (producto cartesiano)
uv run python run_experiment.py -m env.slip_prob=0.0,0.2 gamma=0.9,0.95,0.99

# Más detalle en los logs (nivel DEBUG para el paquete del proyecto)
uv run python run_experiment.py hydra.verbose=rl_tallerdp
```

### Qué produce cada ejecución

- `outputs/<fecha>/<hora>/` (o `multirun/...` en barridos) con:
  - `resultados.json`: todas las métricas de la ejecución.
  - `convergencia_value_iteration.png`: curva del delta de Value Iteration.
  - `V_vi.npy` y `politica_vi.npy`: función de valor y política óptima.
  - `run_experiment.log` y `.hydra/`: el log y la configuración exacta usada.
- Una ejecución registrada en MLflow (`mlflow.db`) con parámetros, métricas y esos mismos archivos como artefactos.

### Estructura del proyecto

```
configs/
  experiment.yaml            # configuración principal (gamma, episodios, MLflow, ...)
  env/milan_rain.yaml        # parámetros del entorno (lluvia, recompensas, calificación)
  solver/
    policy_iteration.yaml    # tolerancia y máximo de iteraciones de PI
    value_iteration.yaml     # tolerancia y máximo de iteraciones de VI
src/rl_tallerdp/
  envs/milan_rain_taxi.py    # el entorno (MDP con P[s][a], API de Gymnasium)
  agents/dynamic_programming.py  # Policy Iteration, Value Iteration
  utils/mdp.py               # construye P (dispersa) y R a partir del entorno
  utils/visualization.py     # gráficas
  evaluation.py              # evaluación por simulación y comparación de políticas
run_experiment.py            # punto de entrada (Hydra + logging + MLflow)
test_env.py, test_build_p.py, test_policy_iteration.py   # tests (pytest)
```

---

## 2. Descripción del ejercicio

Se trata de un problema de **planificación** en un MDP finito con modelo conocido: se conocen las probabilidades de transición `P(s'|s,a)` y las recompensas `R(s,a)`, por lo que no hay aprendizaje por interacción; la política óptima se calcula con Programación Dinámica (DP). 

El agente es un taxi en una grilla de 8×8 que representa la ciudad de Milán. Este debe ir hasta la celda donde le espera un pasajero, recogerle y dejarle en alguno de los 6 destinos turísticos disponibles. El entorno simula un día lluvioso en el que las calles de una franja de la ciudad están mojadas, haciendo que el taxi pueda resbalar. Además, la dinámica incluye el feedback del pasajero, quien califica al conductor de 1 a 5 estrellas según qué tan rápido lo llevó a su destino. Esta calificación se traduce en una propina o una penalización al llegar al destino y tiene la intención de simular el afán de los turistas cuando empieza a llover. 

El objetivo es, entonces, encontrar la política que maximiza el retorno esperado descontado, usando Policy Iteration y Value Iteration, y comparar cómo se comportan ambos algoritmos al cambiar las características del entorno.

---

## 3. Definición del MDP

### Mundo

Grilla de 8×8 (filas y columnas de 0 a 7), con 6 paradas numeradas de 0 a 5 en las posiciones (fila, columna):

| Parada | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Posición | (0,0) | (0,7) | (3,3) | (5,2) | (7,0) | (7,7) |

Hay cuatro muros verticales que bloquean el paso entre las columnas 3 y 4 en las filas 0, 1, 6 y 7. Las **filas 4 y 5** forman la zona mojada ("Navigli"), donde el taxi puede resbalar.

### Estados

Un estado es una tupla `(fila, columna, pasajero, destino, b)`:

- `fila`, `columna`: posición del taxi (64 celdas).
- `pasajero`: 0 a 5 si espera en alguna de las 6 paradas (origen del viaje) y 6 si está dentro del taxi.
- `destino`: 0 a 5, cada una de las 6 paradas.
- `b`: **presupuesto de tiempo** del viaje, de 0 a 26. Vale 0 mientras el pasajero espera. Al recogerlo se fija en `distancia_Manhattan(origen, destino) + 12` y baja en 1 con cada paso dentro del taxi (sin pasar de 0). Se incluye en el estado para que la calificación dependa solo del estado actual y el proceso siga siendo Markov.

Total: **12.288 estados** 
- Esperando: 64 posiciones del taxi × 6 paradas × 5 destinos = 1.920.
- A bordo: 64 posiciones × 6 destinos × 27 valores de b = 10.368.

La distribución inicial es uniforme sobre los 1.920 estados con el pasajero esperando. El estado terminal se alcanza al dejar al pasajero en su destino.

### Acciones

Seis (6) acciones discretas: `0` sur, `1` norte, `2` este, `3` oeste, `4` recoger, `5` dejar.

### Recompensas

| Evento | Recompensa | Razonamiento |
|---|---|---|
| Cada paso (movimiento o recoger correcto) | −1 | hace que el tiempo cueste, así el taxi prefiere rutas cortas.|
| Recoger o dejar en un lugar incorrecto | −10 |es diez veces el costo de un paso, de modo que una acción ilegal es claramente peor que perder un paso. Así el taxi no prueba “recoger” al azar.|
| Dejar al pasajero en su destino | +20 más la propina según las estrellas | tiene que superar el costo típico de un viaje para que completar la entrega valga la pena.|

Las estrellas dependen del presupuesto `b` en el momento de dejar al pasajero, es decir, del retraso del viaje respecto a la distancia mínima. Un `b` alto significa un bajo retraso y, como resultado, una calificación alta (5★) que se traduce en una propina alta también:

| `b` al dejar | Retraso (pasos extra sobre la distancia Manhattan) | Estrellas | Propina |
|---|---|---|---|
| ≥ 10 | ≤ 2 | 5 | +10 |
| 7 a 9 | 3 a 5 | 4 | +5 |
| 4 a 6 | 6 a 8 | 3 | 0 |
| 1 a 3 | 9 a 11 | 2 | −5 |
| 0 | ≥ 12 | 1 | −10 |

Razonamiento sobre el valor de -10 a +10 de la propina: la propina máxima es la mitad de la recompensa base, con lo cual la calificación influye en la política sin dominar sobre completar la entrega. Por el otro lado, la peor calificación (1★) sigue dejando la entrega en +10, no en negativo.

### Dinámica de transición

- **Fuera de la zona mojada** (filas 0 a 3 y 6 a 7), los movimientos son deterministas. Si el movimiento choca con un muro o con el borde, el taxi se queda en su celda.
- **En la zona mojada** (filas 4 y 5), un movimiento sale como se pidió con probabilidad `1 − p` y se desvía a cada una de las dos direcciones perpendiculares con probabilidad `p/2`. Con `p = 0.2`: 0.8 / 0.1 / 0.1. El deslizamiento depende de la celda donde está el taxi al moverse.
- **Recoger** (acción 4) funciona solo si el taxi está en la parada del pasajero; el pasajero pasa a estar a bordo y `b` se inicializa. Si no, cuesta −10 y no cambia nada.
- **Dejar** (acción 5) termina el episodio solo si el taxi está en la parada de destino con el pasajero a bordo. Si no, cuesta −10 y no cambia nada.
- Todo paso con el pasajero a bordo (incluidas las acciones fallidas) reduce `b` en 1.
- Factor de descuento por defecto: `γ = 0.99`.

---

## 4. Modificación del entorno original

### En qué consistió

El entorno original (Milan Taxi, derivado de Taxi-v3) es una grilla de 5×5 con 4 paradas, 500 estados y 6 acciones. Se hicieron tres cambios, uno por cada aspecto que pedía el taller (tamaño, transiciones, recompensas):

1. **Tamaño de la grilla:** de 5×5 con 4 paradas a 8×8 con 6 paradas y muros propios.
2. **Transiciones no deterministas ("día lluvioso"):** en las filas 4 y 5 el taxi resbala con probabilidad `p` hacia los lados.
3. **Estructura de recompensas con calificación del pasajero:** una propina o penalización según las estrellas que el pasajero da por la velocidad del viaje. Para que siga siendo un MDP, el estado se amplió con el presupuesto de tiempo `b`.

### Efectos sobre el espacio de estados y la dificultad

| | Original | Modificado |
|---|---|---|
| Estados | 500 | 12.288 (≈ 25 veces más) |
| Celdas / paradas | 25 / 4 | 64 / 6 |
| Pares (estado, acción) sucesores | 1 (determinista) | hasta 3 |
| Transiciones no nulas de `P` | alrededor de 3.000 (500 estados × 6 acciones) | 73.566 con `p = 0`; 98.142 con `p = 0.2` |
| Recompensa | solo −1 / −10 / +20 | −1 / −10 / +20 más propina según estrellas |

- **Espacio de estados.** El crecimiento viene de tres lados: más celdas, más paradas y la variable `b`, que multiplica por 27 los estados con el pasajero a bordo. Una matriz `P` densa de forma (estados, acciones, estados) ocuparía unos 7,2 GB, y la matriz de transición de una política, de 12.288 × 12.288, unos 1,2 GB. Por eso `P` se construye como **matriz dispersa** (`scipy.sparse`) y los sistemas lineales de Policy Iteration se resuelven con un solver disperso. Una matriz dispersa es una estructura de datos que solo guarda las posiciones de los valores que no son cero, ahorrando drásticamente memoria y tiempo de procesamiento.
- **Transiciones.** Con `p = 0` cada fila de `P` tiene un solo sucesor, y el problema se parece a buscar caminos más cortos en un grafo. Con lluvia, las filas de la zona mojada tienen hasta 3 sucesores, de modo que cada actualización pasa a ser un promedio ponderado y el costo de un barrido crece con el número de transiciones no nulas de `P`.
- **Recompensas.** La propina hace que la recompensa terminal no sea constante: depende de `b`, así que el taxi tiene un incentivo adicional para evitar retrasos, y los deslizamientos tienen un costo extra por las estrellas que se pierden además de los pasos que se añaden.
- **Política óptima.** Deja de ser simplemente el camino más corto. Debe ponderar el costo esperado de resbalar (más pasos y menos estrellas) frente a lo que ahorra una ruta más corta. En la ejecución por defecto (`p = 0.2`), la política óptima del entorno sin lluvia deja de ser óptima en **1.343 estados**. También baja el valor esperado de los estados iniciales.

### Qué cambia en los algoritmos cuando las transiciones dejan de ser deterministas

**Matemáticamente, en la ecuación de Bellman.** Con transiciones deterministas, el siguiente estado es una función `s' = f(s, a)` y la ecuación de optimalidad es:

$$V^*(s) = \max_a \Big[\, r(s,a) + \gamma\, V^*\big(f(s,a)\big) \Big]$$

Con transiciones estocásticas, el siguiente estado es una variable aleatoria y aparece una **esperanza** sobre los posibles sucesores:

$$V^*(s) = \max_a \Big[\, R(s,a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^*(s') \Big], \qquad R(s,a) = \sum_{s'} P(s' \mid s,a)\, r(s,a,s')$$

Por ejemplo, para moverse al este desde una celda mojada con `p = 0.2`:

$$Q(s,\text{este}) = -1 + \gamma\,\big[\,0.8\, V(s_{\text{este}}) + 0.1\, V(s_{\text{norte}}) + 0.1\, V(s_{\text{sur}})\,\big]$$

El `max` queda fuera de la esperanza: el agente elige la acción, pero no el resultado. Por eso la política debe ser robusta al azar y no puede limitarse a seguir la ruta más corta.

**En el comportamiento de los algoritmos:**

- Cada actualización de Bellman es un promedio de hasta 3 valores en lugar de una consulta única, por lo que el trabajo por barrido es proporcional al número de transiciones no nulas.
- La convergencia se mantiene garantizada: con `γ < 1` el operador de Bellman sigue siendo una contracción en la norma del máximo y existe un único `V*`. Lo que cambia es la velocidad: la información de valor ahora se propaga mezclada con probabilidades, y Value Iteration necesita más barridos para alcanzar la misma tolerancia.
- En Policy Iteration, la evaluación de una política exige resolver `(I − γ P_π) V = R_π`, donde `P_π` ya no es una matriz de ceros y unos (un grafo funcional) sino una matriz de probabilidades. Sigue siendo dispersa, así que se resuelve de forma exacta con un solver disperso.
- Pueden aparecer más empates entre acciones (varias acciones con el mismo `Q`), por lo que dos políticas óptimas pueden diferir en acciones sin diferir en valor.

---

## 5. Policy Iteration vs Value Iteration

Ambos resuelven la ecuación de optimalidad de Bellman y convergen al mismo `V*`, pero con estrategias distintas.

### Policy Iteration (PI)

Alterna dos pasos sobre una política explícita `π` (inicialmente uniforme aleatoria):

1. **Evaluación de la política.** Calcula `V^π` de forma **exacta** resolviendo el sistema lineal

   $$(I - \gamma P_\pi)\, V^\pi = R_\pi$$

   con un solver disperso (no hay iteración interna).
2. **Mejora de la política.** Calcula `Q(s,a) = R(s,a) + γ Σ P(s'|s,a) V^π(s')` y toma la política voraz (_greedy_ en la literatura) `π'(s) = argmax_a Q(s,a)`.
3. **Parada.** Termina cuando `V` deja de cambiar entre iteraciones (diferencia máxima menor que la tolerancia), lo que significa que la política es estable y por lo tanto óptima.

Cada iteración es **cara** (un sistema lineal de 12.288 incógnitas), pero hay **pocas**: el teorema de mejora de políticas garantiza que cada paso mejora la política y, como hay un número finito de políticas deterministas, converge en un número finito de pasos.

### Value Iteration (VI)

No mantiene una política explícita. Parte de `V₀ = 0` y en cada **barrido** aplica el operador de optimalidad de Bellman a todos los estados:

$$V_{k+1}(s) = \max_a \Big[ R(s,a) + \gamma \sum_{s'} P(s'|s,a)\, V_k(s') \Big]$$

1. **Barrido.** Calcula `Q` con la `V` actual y toma el máximo por estado, sin resolver ningún sistema lineal.
2. **Parada.** Termina cuando `max_s |V_{k+1}(s) − V_k(s)|` queda por debajo de la tolerancia.
3. **Extracción de la política.** Solo al final se calcula la política voraz respecto a `V`.

Cada barrido es **barato** (una multiplicación dispersa), pero la convergencia es lineal con razón `γ`, por lo que necesita **muchos** barridos.

### Comparación

| | Policy Iteration | Value Iteration |
|---|---|---|
| Qué mantiene | política `π` y su valor `V^π` | solo `V` (la política es implícita) |
| Cada iteración | evaluar `π` (sistema lineal) + mejorar | un backup de Bellman con `max` |
| Evaluación | exacta (a cada paso `V = V^π`) | truncada a un solo paso |
| Costo por iteración | alto | bajo |
| Número de iteraciones | pocas (finitas) | muchas (convergencia lineal de razón `γ`) |
| Criterio de parada | `V` estable entre iteraciones | delta máximo `< tol` |
| Política final | exacta | voraz respecto a un `V` aproximado |

Value Iteration equivale a una Policy Iteration en la que la evaluación se trunca tras un único barrido. Ambos son casos de la Generalized Policy Iteration (GPI) que se explica en el capítulo 4 de _Reinforcement Learning: An Introduction_ como una dinámica de competencia y cooperación entre el proceso que ajusta la función de valor hacia el valor real de la política actual ($v \to v_\pi$) y el que ajusta la política para que sea voraz (_greedy_) respecto a la función de valor actual ($\pi \to \text{greedy}(v)$). 

---

## 6. Convergencia y evaluación de la política

### ¿Cómo sabemos que el algoritmo convergió?

- **Value Iteration:** se guarda el delta máximo de cada barrido (`deltas`) y se detiene cuando baja de la tolerancia (`configs/solver/value_iteration.yaml`). La curva de deltas se grafica (`convergencia_value_iteration.png`) y se registra en MLflow como métrica `vi_delta` por barrido. Con `γ < 1`, el error respecto a `V*` queda acotado por `γ/(1−γ) · delta`.
- **Policy Iteration:** se detiene cuando `V` deja de cambiar entre iteraciones, es decir, cuando la política voraz es estable.
- **Contra falsa convergencia:** si se alcanza `max_iter` sin cumplir la tolerancia, el programa emite un `WARNING` en el log.
- **Validación cruzada de los dos métodos:** la diferencia máxima `max |V_PI − V_VI|` se calcula y se registra en cada ejecución (por defecto del orden de 1e-10). Que dos algoritmos distintos coincidan es evidencia independiente de que ambos alcanzaron el único punto fijo `V*`.
- **Empates:** `estados_con_diferencia_real_PI_VI` cuenta solo los estados donde las acciones de PI y VI tienen distinto `Q`; los empates (acciones distintas con el mismo valor) no se consideran diferencias.

### ¿Cómo sabemos que la política es adecuada?

1. **Simulación (Monte Carlo).** Se ejecuta la política en el entorno durante `n_episodios` (1.000 por defecto, con tope de 300 pasos por episodio) y se miden:
   - retorno medio descontado,
   - tasa de entrega,
   - pasos medios por episodio (solo episodios con entrega),
   - estrellas medias y fracción de entregas con 1 a 5 estrellas.
2. **Consistencia con la teoría.** El retorno medio simulado debe coincidir con el valor `V` de los estados iniciales (`V_inicial`, promedio de `V` bajo la distribución inicial). Si no coinciden, hay un error en el modelo `P`, en `R` o en la evaluación.
3. **Línea base.** Se compara contra una política aleatoria uniforme, para comprobar que lo aprendido es mucho mejor que actuar al azar.
4. **Efecto del entorno.** `estados_donde_politica_base_es_suboptima` mide en cuántos estados la política óptima del entorno sin lluvia deja de ser óptima con lluvia.
5. **Tests automáticos** (22, con `pytest`): las probabilidades de cada `(s, a)` suman 1; las filas terminales no tienen masa en `P`; recoger y dejar pagan lo correcto; las estrellas salen según `b`; PI y VI dan el mismo valor; sin lluvia la ruta óptima desde (0,0) hasta la parada 2 toma 8 pasos y da 5 estrellas; la lluvia reduce el valor esperado.

Nota: dos ejecuciones con semillas distintas pueden dar promedios que difieren en centésimas por puro ruido de muestreo; las comparaciones entre políticas deben hacerse con la misma semilla o con intervalos de variación.

---

## 7. Herramientas: Hydra, logging y MLflow

### Hydra: configuración y barridos

Toda la configuración vive en `configs/` y no en el código. `configs/experiment.yaml` compone los grupos `env/` (parámetros del entorno) y `solver/` (tolerancias de PI y VI). Hydra permite sobrescribir cualquier valor desde la terminal y lanzar barridos con `-m`, y guarda la configuración exacta de cada ejecución en `.hydra/`, lo que hace los experimentos reproducibles.

### Logging

Se usa el módulo estándar `logging` de Python en `run_experiment.py`, en lugar de `print`. Hydra lo configura automáticamente: los mensajes salen por consola y quedan en `run_experiment.log` dentro de la carpeta de cada ejecución. Se registran el tamaño del MDP, las iteraciones y tiempos de PI y VI, avisos de no convergencia (`WARNING`) y el resumen de la evaluación. Con `hydra.verbose=rl_tallerdp` se activa el nivel DEBUG del paquete.

### MLflow

Cada ejecución queda registrada como un *run* en el experimento `milan_rain_taxi` (almacenado en `mlflow.db`), con:

- **Parámetros:** toda la configuración de Hydra (lluvia, `γ`, tolerancias, semilla, etc.).
- **Métricas:** iteraciones y tiempos de PI y VI, `V_inicial`, estrellas, pasos, retorno y tasa de entrega de cada política, diferencias entre PI y VI, y la curva `vi_delta` por barrido.
- **Artefactos:** `resultados.json`, la gráfica de convergencia, `V_vi.npy` y `politica_vi.npy`.

La interfaz (`mlflow ui`) permite ordenar y graficar las ejecuciones de un barrido (por ejemplo, por `env.slip_prob`) sin copiar números a mano.

---

## 8. Resultados

_Pendiente de completar._
