# Papers Distilled — Exphormer-KGC Project

Resumen técnico de los 4 papers/documentos relevantes al proyecto, con énfasis en ecuaciones, decisiones arquitectónicas, resultados numéricos, y la importancia específica de cada uno para este trabajo. Los PDFs originales están en el raíz del proyecto para consulta puntual.

---

## 0. Manuscrito de Candidatura (tesis doctoral — Mauricio Ojeda, PUC Chile)

**Rol en el proyecto**: define el marco completo de la tesis. Sin este documento, ninguna decisión técnica tiene sentido estratégico.

### Objetivo general

Diseñar y validar una arquitectura **Graph Transformer con atención dispersa basada en grafos expander**, que integre **codificación relacional composicional y transferible** (estilo ULTRA), capaz de **link prediction con generalización zero-shot** en KGs no vistos.

### Hipótesis centrales

- **H1**: integrar representaciones relacionales transferibles dentro de atención dispersa expander permitirá capturar dependencias globales en complejidad O(N) sin sacrificar poder expresivo, superando a GNNs basadas en message-passing en link prediction.
- **H2**: un GT disperso con codificación relacional composicional generalizará zero-shot a relaciones/entidades no vistas, **superando a NBFNet** en benchmarks KG no vistos.

### Preguntas de investigación

- **P1**: ¿cómo modificar la atención dispersa basada en expander para inyectar codificación relacional transferible?
- **P2**: ¿cómo afecta la dispersión al poder expresivo vs GTs densos o GNNs basadas en caminos?
- **P3**: ¿cuál es la estrategia óptima para integrar lógica relacional composicional en un GT disperso?
- **P4**: ¿qué componentes arquitectónicos son estrictamente necesarios para el equilibrio O(N) ↔ poder expresivo (modelo fundacional)?

### Plan en 3 etapas

| Etapa | Foco | Entregable | Target |
|-------|------|-----------|--------|
| **1. Adaptación Exphormer → KGC** | atención dispersa condicionada a query (u, q), entidades sin embedding propio, solo embeddings relacionales | modelo inductivo competitivo con NBFNet en WN18RR/FB15k-237 (v1-v4) | ICLR / LOG |
| **2. Codificación relacional transferible** | representaciones composicionales de relaciones inspiradas en ULTRA, grafo de interacción entre relaciones + expander de relaciones | generalización a KGs con relaciones distintas | (tentative) |
| **3. Integración + zero-shot** | entrenamiento multi-grafo (FB15k-237, WN18RR, NELL-995), evaluación zero-shot en KGs no vistos | prototipo de modelo fundacional, análisis de componentes mínimos | NeurIPS / ICML / ICLR |

**Estado actual (2026-04-20)**: en Etapa 1. Transductivo ya supera NBFNet en WN18RR (0.550 vs 0.551 MRR). Inductivo v1 en 0.565 (vs NBFNet 0.741, KnowFormer 0.752) — gap abierto.

### Implicancias para decisiones técnicas

- La arquitectura debe ser **única para transductivo e inductivo** (solo hiperparámetros difieren entre settings) — requisito del manuscrito.
- Mecanismo zero-shot = relaciones como funciones de posición estructural en el grafo de relaciones, no como vectores propios.
- El expander debe escalar entre grafos de distintos tamaños (estudio teórico sobre grado del expander en contexto multi-grafo es una contribución esperada).

---

## 1. NBFNet (Zhu et al., NeurIPS 2021)

**Rol en el proyecto**: baseline técnico principal. La arquitectura actual **ya replica varias ideas clave de NBFNet** (boundary condition, sum aggregation, DistMult messages, drop-edge on query). El objetivo de Etapa 1 es **igualar o superar** su MRR inductivo.

### Idea central

Link prediction como **path formulation**: `h_q(u,v) = ⊕_{P∈P_uv} h_q(P)` donde `h_q(P) = ⊗ w_q(e_i)` (producto generalizado de embeddings de aristas). Se resuelve eficientemente con **Bellman-Ford generalizado** (DFS sobre caminos → programación dinámica).

### Arquitectura (3 componentes neurales)

Parametriza el BF generalizado con funciones aprendibles:

1. **INDICATOR**: boundary condition. `h^(0)_v ← 1(u=v) * q` donde `q` es el embedding de la relación query.
   - Única fuente de identidad: **solo el nodo source recibe `rel_emb[r_q]`**, el resto queda en cero.
   - Esta es la base del "anchor" en el proyecto.

2. **MESSAGE**: operador multiplicación generalizado. `m_{(x,r,v)} = MESSAGE(h_x^{(t-1)}, w_q(x,r,v))`.
   - Instanciaciones: TransE (suma), **DistMult (producto Hadamard — el óptimo)**, RotatE.
   - `w_q(x,r,v) = W_r * q + b_r` — edge representation depende de la query.

3. **AGGREGATE**: permutation-invariant sobre el set de mensajes. Sum, mean, max, **PNA** (best en ablation).
   - `h^(t)_v ← AGGREGATE({m_{(x,r,v)} : (x,r,v) ∈ E(v)} ∪ {h^(0)_v})`
   - Después: Linear + ReLU + **LayerNorm**.

### Ecuación clave (BF iteration)

```
h_q^(t)(u,v) = [⊕_{(x,r,v)∈E(v)} h_q^{(t-1)}(u,x) ⊗ w_q(x,r,v)] ⊕ h_q^(0)(u,v)
```

### Detalles de implementación importantes

- **6 capas** (T=6) es óptimo — performance satura después.
- **hidden dim = 32** (MLP head = 64).
- **Edge representations diferentes por capa** (T conjuntos): permite distinguir orden de caminos (mother's father ≠ father's mother).
- **Reciprocal augmentation**: cada triplet (u,q,v) se acompaña de (v, q⁻¹, u).
- **Drop edges que conectan el par query directamente** durante training → fuerza al modelo a usar caminos largos, previene overfitting.
- Loss: negative log-likelihood con negativos PCA (Partial Completeness Assumption).
- Residual short-cut + LayerNorm después de AGGREGATE.
- **Fused message passing**: reduce memoria de O(|E|d) a O(|V|d).
- Solo **3M parámetros** en FB15k-237 (TransE usa 30M).

### Resultados clave (inductive H@10)

| Split | FB15k-237 | WN18RR |
|-------|-----------|--------|
| NBFNet v1 | 0.834 | **0.948** |
| NBFNet v2 | 0.949 | 0.905 |
| NBFNet v3 | 0.951 | 0.893 |
| NBFNet v4 | 0.960 | 0.890 |

**Inductive MRR WN18RR v1: 0.741** (KnowFormer reporta 0.752 con el mismo setup).

### Ablaciones importantes para el proyecto

- **MESSAGE**: DistMult > RotatE > TransE (cuando AGGREGATE=sum).
- **AGGREGATE**: PNA (0.415) > Sum (0.388) > Max (0.374) > Mean (0.384). Sum satisface semiring con DistMult.
- Combinations que satisfacen el semiring (TransE+max, DistMult+sum) son localmente óptimas.

### Importancia para este proyecto

1. **Fuente del anchor-based boundary condition**: KGCNodeEncoder implementa `x_h = rel_emb[r], x_v = 0` copiando esta idea.
2. **Justifica sum aggregation**: el cambio `wV / Z` → `wV` en el proyecto dio +0.23 MRR (sesión 3) — confirma que normalización es anti-NBF.
3. **Justifica DistMult V**: intento `v_src = h_src ⊙ z_r` (fc_v_expand con init Kaiming) se basa directamente en NBFNet.
4. **Justifica drop direct edges**: NBFNet lo usa para prevenir overfitting — nuestro trainer hace esto también.
5. **Techo a superar**: 0.741 MRR (WN18RR v1 inductive). Sin esto, la contribución de la tesis es débil.

---

## 2. Exphormer (Shirzad et al., ICML 2023)

**Rol en el proyecto**: esqueleto arquitectónico base. Todo el código parte de la atención dispersa sobre expander + local + virtual nodes de Exphormer. Las modificaciones KGC se construyen sobre esta atención.

### Idea central

Transformer disperso con atención **O(|V|+|E|)**. Se construye un "interaction graph" H que define qué pares de nodos se atienden, combinando tres tipos de aristas:

1. **Local neighborhood** (`E`): aristas del grafo original + sus inversas.
2. **Expander graph**: grafo `d`-regular aleatorio (típicamente `d=3`) sobre los mismos nodos.
3. **Global attention**: virtual nodes conectados a todos los demás.

### Ecuación de atención

Con edge features (estilo SAN):

```
ATTN_H(X)_{:,i} = x_i + Σ_j W_O^j W_V^j X_{N_H(i)} · σ(W_E^j E_{N_H(i)} ⊙ (W_K^j X_{N_H(i)})^T W_Q^j x_i)
```

Donde:
- `σ` es softmax (o escalado con `√d`).
- `W_E^j E_{N_H(i)} ⊙ (W_K^j X)^T`: producto elemento-a-elemento para inyectar edge features en la matriz de atención.
- `N_H(i)`: vecinos en el interaction graph H (NO en G).

### Propiedades teóricas clave del expander

El expander `d`-regular `ε`-expander aproxima espectralmente al grafo completo:

```
(1−ε)(1/n)L_K ⪯ (1/d)L_G ⪯ (1+ε)(1/n)L_K
```

Esto preserva **cuts, expansión de vértices, mixing de random walks**. Lemma 4.2: después de `t ≥ (1/(2(1−ε)))·log(n/δ²)` pasos, el random walk en el expander converge a distribución uniforme. Corolario 4.4: **apilando O(log n) capas, se modelan todas las interacciones pares**.

### Generación del expander

Friedman (2003) — random `d`-regular near-Ramanujan:
1. Elegir `d/2` permutaciones `π_1, ..., π_{d/2}` uniformemente aleatorias.
2. `E' = {(i, π_j(i)), (i, π_j^{-1}(i)) : j ∈ [d/2], i ∈ [n]}`.
3. Rechazar si no cumple el threshold near-Ramanujan (evento de baja probabilidad).

En el proyecto esto se hace **una sola vez** al cargar el dataset (expander estático sobre el grafo completo); luego el trainer lo "tila" (`_tile_expander`) para batches.

### Resultados relevantes

| Dataset | Exphormer | GraphGPS full | Sparse (BigBird/Performer) |
|---------|-----------|---------------|---------------------------|
| CIFAR10 | 74.69 | 72.30 | ~70 |
| MalNet-Tiny | 94.02 | 93.50 | 92.34 |
| MNIST | 98.55 | 98.05 | — |
| CLUSTER | 78.07 | 78.02 | — |

Exphormer supera o iguala al full transformer en la mayoría de casos, con complejidad lineal.

### Detalles de implementación relevantes

- **Edge features aprendibles** para los tres tipos de aristas (expander, global, local).
- **Un solo mecanismo de atención** compartido entre tipos de aristas — se distingue solo por edge features.
- Se puede omitir expander o global nodes según el dataset (ablation en apéndice D).
- Con virtual nodes, cada nodo queda a distancia 2 (vía virtual) de cualquier otro.

### Importancia para este proyecto

1. **Base arquitectónica completa**: el código parte de Exphormer + GraphGPS.
2. **Justifica teóricamente por qué expander funciona**: propiedades espectrales y mixing dan cobertura global con O(|V|) aristas adicionales.
3. **La ablation de sesión 3 mostró que expander NO es el cuello de botella en KGC inductivo** (0.252 vs 0.256 con/sin expander). Esto contradice parcialmente el paper original — el cuello de botella en KGC es relacional, no topológico.
4. **`exp_deg = 3`** es el default razonable. El paper usa `d=3` ampliamente.
5. **Virtual nodes DESHABILITADOS en KGC**: `num_virt_node=0`. Con 40K+ nodos en FB15k-237 y un virtual node conectado a todos, la atención se satura.
6. **Una contribución potencial del proyecto**: demostrar empíricamente qué pasa cuando el expander interactúa con el mecanismo relacional de NBFNet — terreno no explorado por ningún paper previo.

---

## 3. KnowFormer (Liu et al., ICML 2024)

**Rol en el proyecto**: competidor técnico SOTA en inductivo. Sus dos ideas clave (redefinición de atención + estructuras-aware Q/V) son la inspiración para **query conditioning** en el proyecto, pero la arquitectura dual-stream (Q-RMPNN + V-RMPNN) resultó no transferir directamente (ver sesión 14).

### Idea central

Redefinir la atención como **agregación pesada de información pareada basada en la plausibilidad de pares como prototipos de la query**. No es atención secuencial-style; se construye específicamente para KGs.

### Definición de atención (ecuación 1-2)

Atención estándar (Tsai et al., Chen et al.):

```
Attn(x_u) = Σ_v [κ(f_q(x_u), f_q(x_v)) / Σ_w κ(f_q(x_u), f_q(x_w))] · f_v(x_v)
```

Donde `κ` es un kernel positivo-definido. Relation-aware sería O(|R|·|V|²), **inviable**. Truco de KnowFormer: reducir a O(|V|²) usando **query prototypes** — pares (u,v) son prototipo de `r_q` si ambos pueden ser cabezas de `r_q`.

### Dos streams estructurales

Reemplaza `f_q` y `f_v` por redes RMPNN especializadas (ambas son NBF internas):

#### Q-RMPNN (ecuación 5)

Codifica cada nodo `u` según su contexto k-hop, con **inyección de ruido** para distinguir fuentes:

```
z_u^{(0)} = [x_u, ε]                    donde ε ~ N(0, I)
m_{u|v,r}^{(l-1)} = z_v^{(l-1)} ⊙ r̂    donde r̂ = R[r_q]·W_r + b_r
z_u^{(l)} = Φ^{(l)}(α^{(l)} z_u^{(l-1)} + Σ_{r(v,u)∈E} m_{u|v,r}^{(l-1)})
```

Usa `eL` capas internas (típicamente 2). Output: `z̃_u` structure-aware, query-conditioned.

#### V-RMPNN (ecuación 6)

Similar pero con **head labeling** (one-hot del nodo cabeza):

```
ẑ_u^{(0)} = [x_u, I_{u=h} ⊙ 1]    → 1 en la posición del cabeza, 0 en el resto
ẑ_u^{(l)} = Ψ^{(l)}(β^{(l)} ẑ_u^{(l-1)} + Σ_{r(v,u)∈E} m̂_{u|v,r}^{(l-1)})
```

`bL` capas internas (típicamente 2). Output: `ẑ_u` pairwise-aware.

### Kernel lineal (ecuación 8-10)

Aproximación de Taylor de primer orden del kernel exponencial:

```
κ(z̃_u, z̃_v) ≈ 1 + ⟨z̃_u W_1, z̃_v W_2⟩    (normalizado por Frobenius)
```

Esto permite computar primero `K^T V` y luego `Q · (K^T V)` → **complejidad lineal O(|V|)**. Con el **término de bypass** `+ 1^T V + v·|V|` que estabiliza cuando Q es ruidoso.

Fórmula final (ec. 10):

```
Q = z̃W_1 / ||z̃W_1||_F
K = z̃W_2 / ||z̃W_2||_F  
V = ẑ
D = diag(1 + Q(K^T 1) + |V|) / |V|
Z = D^{-1} [V + 1^T V + Q(K^T V) / |V|]
```

### Arquitectura completa (sección 4.2)

```
A^(l) = LayerNorm1(X^(l-1) + Attn(X^(l-1), R))
X^(l) = LayerNorm2(A^(l) + FFN(A^(l)))
```

- `X^(0) = 0` (all-zero entity features).
- `R` es aprendible, inicializada aleatoriamente.
- L capas apiladas (L=2-6 según dataset).
- Loss: negative sampling `L = -log σ(t|h,r) - Σ log(1 - σ(t'|h,r))`.

### Resultados (MRR)

| Método | FB15k-237 | WN18RR | NELL-995 | YAGO3-10 |
|--------|-----------|--------|----------|----------|
| NBFNet | 0.415 | 0.551 | 0.525 | 0.563 |
| **KnowFormer** | **0.430** | **0.579** | **0.566** | **0.615** |

Inductive (v1 MRR): **KnowFormer 0.752 vs NBFNet 0.741** en WN18RR v1.

### Ablaciones relevantes (Tabla 3, FB15k-237)

| Variante | MRR | H@10 |
|----------|-----|------|
| Full KnowFormer | 0.430 | 60.8 |
| w/o attention (= NBFNet) | 0.417 | 58.8 |
| w/o query function | 0.422 | 59.2 |
| **w/o value function** | **0.367** | **48.7** |  ← el más crítico
| RF-based kernel | 0.419 | 57.6 |

**Lección**: el V-RMPNN (stream pairwise con head labeling) es el componente dominante. Sin él, cae más de 6 puntos MRR.

### Importancia para este proyecto

1. **Inspiración del query conditioning**: en `ExphormerAttention`, `proj_q/k/e(shared_rel_emb[r_q])` viene directamente de KnowFormer — conditionar Q/K/E por la relación query.
2. **Inyección de ruido para symmetry-breaking**: `ε ~ N(0, I)` en `z^(0)_u` es la próxima intervención propuesta (sesión 14).
3. **Head labeling one-hot**: nuestra arquitectura usa `rel_emb[r_q]` en el anchor en vez de one-hot; es una variante más informativa pero pierde la identidad espacial.
4. **V bypass FRACASÓ** (sesión 14 — documentado en SESSION_NOTES): el término `+ v·|V|` del kernel NO transfiere a nuestra arquitectura de stream único. Razón: KnowFormer tiene 2 capas NBF internas que filtran `V` antes del bypass. En stream único, el bypass es un atajo que el optimizer explota y rompe el refinamiento iterativo.
5. **Ablation w/o value function = -6.3 MRR**: sugiere que la expresividad de V es crítica. Nuestra DistMult V (`h ⊙ z_r`) es más débil que V-RMPNN. Potencial camino futuro.
6. **Kernel lineal Taylor**: técnica conocida, no es la fuente de su ventaja (ver ablation RF-based kernel). La ventaja viene de los RMPNN estructura-aware.
7. **LayerNorm tras atención Y tras FFN**: diferente del Exphormer puro. Nuestro código sigue este patrón.

### Lo que KnowFormer confirma sobre nuestra dirección

- El mecanismo **relacional en V** es más importante que el mecanismo de atención en sí (Q/K).
- Redefinir atención para KGs es válido — no hay que usar atención "clásica" (Q·K^T con scaling).
- Los streams separados (Q-stream y V-stream) son su ventaja clave — **nuestra stream única es una simplificación que paga costo**.

---

## Tabla maestra de comparación

| Aspecto | NBFNet | Exphormer | KnowFormer | **Proyecto actual** |
|---------|--------|-----------|------------|--------------------|
| **Setting** | KGC (trans + ind) | graph/node class. | KGC (trans + ind) | KGC (trans + ind) |
| **Complejidad** | O((\|V\|+\|E\|)d) | O(\|V\|+\|E\|) | O(\|V\|+\|E\|) | O(\|V\|+\|E\|) |
| **Message passing** | BF iterativo con DistMult | atención dispersa global | atención linear con RMPNN | atención dispersa + anchor BF residual |
| **Boundary cond.** | 1(u=v)·q | ninguno (grafo completo disperso) | `[x_u, ε]` o `[x_u, I_{u=h}]` | `x_h = rel_emb[r_q], x_v=0` (NBF-style) |
| **Expander** | no | **sí** (d-regular) | no | **sí** (d=3) |
| **V** | DistMult puro | W_V(h) estándar | V-RMPNN (2 capas NBF) | DistMult vía fc_v_expand |
| **MRR inductive v1** | **0.741** | — | **0.752** | **0.565** |
| **MRR transductive WN18RR** | 0.551 | — | 0.579 | **0.550** |
| **Capas** | T=6 | 3-5 | L=2-6 | 3 |
| **Dim** | 32 | 64-96 | 64-256 | 64 |

---

## Decisiones de diseño justificadas desde estos papers

1. **Sum aggregation** (no wV/Z): NBFNet §3 + ablation AGGREGATE. Implementado sesión 3.
2. **DistMult messages**: NBFNet Tabla 2 + 6a; KnowFormer ec. 4. Implementado como `fc_v_expand`.
3. **Anchor boundary condition**: NBFNet ec. 3 (INDICATOR). Implementado en `KGCNodeEncoder`.
4. **BF residual (`h += x0`)**: NBFNet ec. 4 (término `⊕ h^(0)_q(u,v)`). Implementado en `MultiLayer.forward`.
5. **Drop direct query edges**: NBFNet §4.1 "to encourage the model to capture longer paths". Implementado en trainer.
6. **Reciprocal triplets**: NBFNet §4.1 "augment each triplet (u,q,v) with (v,q⁻¹,u)". Implementado en KGCDataset.
7. **Expander degree = 3**: Exphormer default. Exp ablation (sesión 3) mostró que `exp=False` da MRR equivalente — expander no es bottleneck pero tampoco daña.
8. **Query conditioning Q/K/E**: KnowFormer §4.1 (query prototype). Implementado como `proj_q/k/e(shared_rel_emb[r_q])`.
9. **Inductive routing (K = proj_k(r_q) solo)**: intervención del proyecto — elimina dependencia de `W_K(h)` que memoriza topología del train graph. No viene de ningún paper; es contribución original. +0.05 MRR.
10. **LayerNorm post-attn y post-FFN**: KnowFormer §4.2 + Exphormer. Implementado en MultiLayer.

## Direcciones futuras sugeridas por los papers

| Dirección | Fuente | Estado |
|-----------|--------|--------|
| **Inyección de ruido Gaussian** al anchor | KnowFormer ec. 5 (`[x_u, ε]`) | próximo — sesión 15 |
| **Head labeling one-hot** en V | KnowFormer ec. 6 | sin probar (probablemente peor que rel_emb) |
| **Edge representations diferentes por capa** | NBFNet §3.2 | sin probar — agrega params |
| **FiLM conditioning del FFN** | no viene de papers; propuesto en CLAUDE.md | sin probar, único componente sin query-cond |
| **PNA aggregation** | NBFNet Tabla 6a (best) | **PROBADO, CATASTRÓFICO** (0.228) — mean reintroduce normalización Z |
| **RMPNN dual-stream (Q-RMPNN + V-RMPNN)** | KnowFormer §4.1 | **PROBADO parcial (V-RMPNN), 0.513** — no mejora por stream-único |
| **V bypass** (KnowFormer kernel term `+v|V|`) | KnowFormer ec. 10 | **PROBADO, FALLÓ** (sesión 14) — atajo rompe refinamiento |
| **Increase T hasta 6** | NBFNet sección 4.3 | **PROBADO, peor** (L=5 ya overfit más rápido) |

Ver `SESSION_NOTES.md` para el detalle cronológico de cada experimento.
