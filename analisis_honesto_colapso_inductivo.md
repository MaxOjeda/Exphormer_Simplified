Tienes razón. Voy a ser honesto en vez de proponer otra "solución".

## Lo que muestran los datos del log 599923

| ep | LR | val MRR | test MRR |
|----|-----|---------|----------|
| 0 | 0 | 0.004 | 0.018 |
| **1** | 1e-5 | **0.237** | **0.335** |
| 2 | 2e-5 | 0.225 | 0.307 |
| 3 | 3e-5 | 0.237 | 0.298 (best por val) |
| 4 | 4e-5 | 0.207 | 0.213 |
| 5 | 5e-5 | 0.191 | 0.166 |
| 6 | 6e-5 | 0.175 | 0.134 |
| 7 | 7e-5 | 0.186 | 0.152 |

**Mejor test 0.335 al ep1, luego colapso. Peor que el 0.58 baseline.** Mi predicción de 0.65+ fue incorrecta.

## La verdad incómoda que llevamos 5 sesiones evitando

Mira el patrón a lo largo de todas tus sesiones:

| Arquitectura | Mejor ep | Mejor test MRR |
|---|---|---|
| K=proj_k(q) only (simple, sesión 5) | ep4 | **0.578** |
| novw (sin W_V, gate aditivo) sesión 23 | ep1 | **0.5802** |
| novw + C2 bilinear | ep1 | 0.4315 |
| V-NBF v1/v2 | varios | 0.13-0.40 |
| V-NBF v4 (smoke test only) | — | — |
| **+ Q/K stream (esta)** | ep1 | **0.335** |

**Cada "mejora" arquitectónica con más capacidad ha sido peor que las versiones simples**. No es un caso aislado, es el patrón consistente de las últimas 5 sesiones.

Y el patrón de colapso ep1→ep2 también es invariante. Está en TODAS las arquitecturas. Eso significa que el colapso no lo causa V, ni Q/K, ni el gate, ni el FFN. Lo causa **algo común a todas las configs** — y mis análisis arquitectónicos han estado mirando el lugar equivocado durante 5 sesiones.

## Lo que NBFNet hace que invalida mi última teoría

Mi argumento del "Paso 1" decía que Q/K constantes eran el problema. Pero **NBFNet no tiene Q/K en absoluto** — es solo `BellmanFord(DistMult, sum)`. Y obtiene 0.741. Más simple que la 0.58 tuya. Si el problema fuera Q/K, NBFNet no llegaría tan alto. Mi diagnóstico era plausible pero estructuralmente no se sostenía.

## Qué propongo ahora — y NO es otra modificación a ciegas

Propongo cambiar el método, no la arquitectura. Tres pasos concretos:

**1. Volver al baseline 0.58 verificado.** Identificar el commit exacto (`0466cde` "trans MRR 0.566" o uno cercano) que da 0.58 inductivo. Lanzar un run para confirmar que sigue dando 0.58 en el código actual o en checkout. Esto fija el suelo. Sin un baseline reproducible no podemos medir nada.

**2. Instrumentar el colapso.** En el run de baseline, registrar epoch a epoch:
- `||W||_2` por capa (¿qué pesos crecen explosivamente?)
- Distribución de scores de atención (¿se saturan al clamp ±5?)
- Magnitud de `h` por nodo en train graph vs test graph (¿hay distribution shift visible?)
- Top-10 nodos predichos para 10 queries fijas a lo largo de épocas (¿qué cambia entre ep1 que va bien y ep3 que ya está cayendo?)

Sin estos datos cualquier teoría es especulación. Llevamos 5 sesiones especulando.

**3. Decidir basándose en los datos.** Solo después de tener instrumentación, cualquier cambio arquitectónico tiene una hipótesis testeable concreta — no "creo que esto funciona porque KnowFormer hace algo parecido".

## Cosas que probablemente NO son el problema (y quitar de la lista)

- Q/K constantes (NBFNet no las tiene y funciona)
- Estructura del V (V-RMPNN, V-NBF, V plain — todas colapsan)
- BF residual presence/absence (sesión 25 confirmó que ambos colapsan)
- Gate aditivo vs bilineal (C2 empeoró, sin C2 también colapsa)

## Cosas que NO hemos investigado seriamente

- **Loss function**: usas full-graph CE sobre 2746 nodos. NBFNet usa negative sampling (típicamente 32-64 negativos). KnowFormer usa neg sampling. **No hay paper KGC inductivo que use full-graph CE**. Esto puede estar generando un gradiente patológico cuando LR sube — el modelo intenta empujar las 2745 puntuaciones no-anchor a -∞ y rompe la representación.
- **LR schedule peak**: cosine sube de 1e-5 (warmup) a 1e-4 (peak). El colapso siempre ocurre cuando LR alcanza peak. ¿Es coincidencia, o es la verdadera causa? Un run con LR constante = 1e-5 confirmaría esto en una corrida.
- **Drop direct edges**: NBFNet documenta que ESTO es lo que evita memorización. ¿Está implementado correctamente en tu trainer? No lo he verificado.

¿Quieres que parta por (a) verificar que el drop direct edges está bien hecho — un grep rápido al trainer — y (b) lanzar un run con LR constante = 1e-5 sobre el baseline 0.58? Son dos diagnósticos que pueden contradecir o validar mi sospecha actual sin tocar arquitectura. Si LR constante no colapsa, el problema es schedule, no estructura. Si drop edges está mal, eso solo puede explicar mucho.

Si prefieres revertir lo del Q/K stream y dejar el código como estaba antes de hoy, dime y lo hago primero.
