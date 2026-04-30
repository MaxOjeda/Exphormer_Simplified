# Session Notes

> **Highlights históricos (sesiones 1-9, pre-2026-04-14)** — resumen ejecutivo. Detalle completo en `SESSION_NOTES_ARCHIVE.md` (cargar solo bajo demanda).
>
> - **Sum aggregation (NBFNet-style)**: `h_out = batch.wV` (sin `/Z`). +0.23 MRR en inductivo v1 (0.252→0.482). Descubrimiento más grande de la fase 1.
> - **LR tuning (lr=1e-5 + warmup=5)**: +0.033 MRR. Baja LR es crítica en inductivo pequeño; cosine schedule estándar.
> - **V_gate SIN sigmoid**: cambio clave que dio +0.08 MRR muy temprano. `gate = batch.E_gate`, no `sigmoid(...)`. No volver a agregar.
> - **Mejor inductivo v1 pre-sesión 12**: **0.565 test MRR** @ ep4 (arquitectura pre-refactor con W_V+gate).
> - **FALLIDOS probados (no repetir)**: V-RMPNN (0.513), use_relational_v (0.384), use_nbf_v xavier (0.420), PNA con mean (0.228), MLP scorer (0.458), L=5 (0.494), dim=128 L=5 (0.501), tie_layers (0.513), constlr (0.563 = igual), Tucker W_r (sesión 9).
> - **Regla metodológica**: todos los experimentos nuevos usan `ckpt_monitor_split: val` (val es entity-specific al train graph en inductivo pero es la selección legítima académicamente).

---

## Estado actual — 2026-04-29 (sesión 24): C2 (gate bilinear) implementado, **falla aislado** — re-diagnóstico

### Marco metodológico (cambio de proceso)

Tras dos sesiones de parches (warmup, lr, label smoothing) sin cierre del gap inductivo, se hizo un análisis estructural completo cruzando QC-Exphormer con KnowFormer. El resultado quedó en `diagnostico_solucion_inductivo.md` como referencia permanente:

**Regla estructural única**: una arquitectura inductiva-por-construcción no puede tener `Linear(h_acumulada)` en ningún path que defina ROUTING o GATING. Solo puede tener `Linear` sobre cantidades relacionales (x0, query_emb, edge_emb).

Componentes que violan la regla en la arquitectura actual:
- `K = W_K(h) + proj_k(q)` ❌ (memoriza routing de train)
- `FFN(h)` ❌ (memoriza distribución de h en train)
- `score = exp(Q*K*E)` ❌ (a través de K)

Más: el gate `W_g(emb_uv) + proj_vg(q)` es aditivo → solo rango-1 en el cross `(r, q)`. KnowFormer usa bilinear vía `fc_z(q).reshape(R, d)[r]`.

**Cuatro cambios propuestos** (todos justificados por la misma regla):
- C1: K relacional pura (`K = K(x0) + proj_k(q)`)
- C2: gate bilinear `M_q[r]` en lugar de aditivo
- C3: pre-LayerNorm sobre V para acotar magnitud (estabilidad del decay)
- C4: eliminar FFN o condicionalo por query (FiLM-FFN)

### Implementación de C2

Reemplazado en `layer/exphormer.py` el gate aditivo por bilinear:

```python
# C2 — Bilinear V gate.
# gate(r, q) = gate_base[r] + M_q[r], with M_q = fc_zq(query_emb).view(R+1, d_out).
# Indexed by batch.edge_rel_idx (sentinel = num_relations for expander edges).
gate_base : nn.Parameter (R+1, d_out), std=1.0   # per-relation baseline
fc_zq     : Linear(d_in, (R+1)*d_out), std=0.01  # query-modulated cross-term
```

Threading `cfg.dataset.num_relations` desde MultiModel → MultiLayer → GlobalModel → ExphormerAttention. Asserts: incompatible con `use_virt_nodes=True`. Optimización: `M_q.view(-1, d_out)` + `index_select(0, flat_idx)` con `flat_idx = edge_graph * (R+1) + edge_rel` (1D scatter en backward, 12× más rápido que indexing 2D fancy).

**Params**: 234,113 (novw baseline) → **588,353** (novw + C2). Δ = +354,240 (≈ 5 capas × 70K = gate_base 1,216 + fc_zq 77,824 cada una).

### Resultados — Job 597009 (`wn18rr_ind_v1_novw_c2.yaml`, T=5, lr=8e-4, wu=3, 30 ep)

| Epoch | LR | val MRR | test MRR | h@10 test |
|-------|-----|---------|----------|-----------|
| 0 | 0.0 | 0.0120 | 0.0175 | 0.032 |
| **1** | **2.7e-4** | **0.3550** | **0.4315** | **0.718** |
| 2 | 5.3e-4 | 0.2640 | 0.2943 | 0.601 |
| 3 | 8.0e-4 (peak) | 0.2266 | 0.1591 | 0.410 |
| 4 | 7.97e-4 | 0.2206 | 0.1187 | 0.354 |
| 5 | 7.89e-4 | 0.1699 | 0.0866 | 0.287 |
| 6 | 7.76e-4 | 0.1659 | 0.0836 | 0.255 |

**`time_iter = 0.047s`** (vs 0.149s baseline) — **C2 es 3.2× MÁS RÁPIDO** que la baseline novw, porque eliminó dos Linear per-edge (`V_gate(edge_attr)`, `proj_vg(shared_edge)`) por uno per-graph (`fc_zq(query_emb)`) + gather.

### Comparación directa con baseline (mismo schedule)

| Run | best val | best test | best ep | observación |
|-----|----------|-----------|---------|-------------|
| **596512** (novw, gate aditivo) | 0.4803 | **0.5802** | ep1 | colapso ep2+ |
| **597009** (novw + C2 bilinear) | 0.3550 | **0.4315** | ep1 | colapso ep2+ |
| Δ | −0.125 | **−0.149** | — | **peor** |

**C2 aislado EMPEORA la métrica en 0.149 MRR**. Y colapsa al menos tan fuerte como el baseline.

### Re-diagnóstico

El error de razonamiento fue tratar C1, C2, C3, C4 como independientes y testeables individualmente. Re-leyendo `diagnostico_solucion_inductivo.md`:

La regla dice que TODA función que toque h debe ser content-only o relational-conditioned. Si dejo `W_K(h)` activo (driver de routing entity-específico, **VIOLACIÓN ACTIVA**) y le agrego C2 (más capacidad expresiva al gate), el optimizer encuentra **más dimensiones para combinar el routing memorizado con el gate bilinear** y crear shortcuts train-específicos. La capacidad adicional NO sirve sin antes bloquear el path entity-contaminante.

**Conclusión**: **C2 sin C1 es contraproducente**. El bilinear gate solo paga su valor cuando el routing ya es relacional puro.

Esto cambia la metodología de validación:

- ❌ **NO testear los 4 cambios uno por uno aislados**. Cada uno aislado puede empeorar el modelo si los demás violadores siguen activos.
- ✅ **Testear acumulativamente con C1 como base** (eliminar el path de routing entity-contaminante), luego sumar C2 encima, luego C3, luego C4.

### Próximo paso propuesto

**Cancelar 597009** (trayectoria ya clara) y lanzar:

**Experimento siguiente: C1 solo** (`K = K(x0) + proj_k(q)`, sin C2 todavía).

Razón: aislar el efecto de eliminar el routing entity-específico. El test crítico:
- Si C1 solo sube de 0.58 → ~0.62-0.65: el routing era el cuello principal. **Después** sumar C2 para capturar el cross-term.
- Si C1 solo NO mueve nada o baja: el diagnóstico está mal y hay que repensar la regla estructural antes de seguir.
- Si C1 solo ya rompe transductivo gravemente: revisar tradeoff vs el K solo-query (sesión 5, gave 0.578 inductive).

**Después de validar C1**: lanzar **C1 + C2** sobre la base estabilizada.

### Estado del job 597009

- Corriendo en GPU (compute-gpu-3-1), ep6 al cierre
- Plan: cancelar después de confirmar trayectoria de colapso (probablemente cancelar ahora; los datos son suficientes)
- Log: `logs/wn18rr_ind_v1_novw_c2_597009.out`

### Archivos relevantes de esta sesión

- `diagnostico_solucion_inductivo.md` — diagnóstico estructural completo y solución (4 cambios C1-C4 con justificación)
- `configs/Exphormer/wn18rr_ind_v1_novw_c2.yaml` — config del experimento C2
- `sbatch_wn18rr_ind_v1_novw_c2.sh` — sbatch correspondiente
- `layer/exphormer.py` — implementación de C2 (commit pendiente)
- `network/model.py` — threading de num_relations a través de MultiModel/MultiLayer/GlobalModel

---

## Estado anterior — 2026-04-29 (sesión 23): Cambio 1 (sin W_V) — resultados inductivos y diagnóstico de inestabilidad

### Jobs lanzados

| Job | Config | LR | Estado |
|-----|--------|----|--------|
| 596512 | novw (ind v1, 1-GPU) | 8e-4 | Corriendo (30 épocas) |
| 596513 | transduct_novw (1-GPU verificación) | 2e-4 | Corriendo (30 épocas) |
| 596522 | novw_lr2e4 (ind v1, 1-GPU) | 2e-4 | Corriendo (30 épocas) |
| pendiente | transduct_novw_4gpu (4-GPU, 100 épocas) | 8e-4 (×4GPU) | Script listo, no lanzado aún |

### Resultados del Cambio 1 — inductivo v1

**Job 596512 (lr=8e-4)**

| Epoch | LR | val MRR | test MRR |
|-------|-----|---------|----------|
| 1 | 2.7e-4 | **0.4803** | **0.5802** |
| 2 | 5.3e-4 | 0.4230 | 0.3016 |
| 3 | 8.0e-4 (peak) | 0.1953 | 0.2186 |
| 4-7 | ↓ cosine | ~0.15 | ~0.09 |

Checkpoint guardado: epoch 1, test=**0.5802** (iguala el baseline anterior 0.578).

**Job 596522 (lr=2e-4)**

| Epoch | LR | val MRR | test MRR |
|-------|-----|---------|----------|
| 1 | 6.7e-5 | 0.3941 | 0.4822 |
| 2 | 1.3e-4 | **0.4374** | **0.5377** |
| 3 | 2.0e-4 (peak) | 0.4285 | 0.2510 |
| 4-5 | ↓ cosine | ~0.43 → 0.37 | ~0.17 |

Checkpoint guardado: epoch 2, test=0.5377. Peor que 596512.

### Diagnóstico de inestabilidad

El patrón es idéntico en ambos runs: pico durante warmup, colapso exactamente al llegar al peak LR. El colapso ocurre independientemente del valor del peak (8e-4 o 2e-4), solo varía en qué epoch ocurre.

**Causa probable**: sin W_V, los gradientes del ranking loss fluyen directamente por `V = h → query_rel_emb + encoder` sin la amortiguación que daba la proyección lineal. El peak LR produce updates que rompen los patrones de propagación relacional aprendidos en los primeros pasos de warmup. La inestabilidad es estructural a la arquitectura V=h con BF residual + LR schedule actual.

**Mejor resultado disponible (Change 1)**: test MRR = **0.5802** @ ep1 del job 596512 (val-selected via ckpt_best).

### Opciones para estabilizar (pendiente de implementar)

1. **Label smoothing 0.1**: reduce sharpness del gradiente de ranking loss
2. **Warmup más largo** (e.g., 10 épocas): permite más pasos a LR bajo antes del peak
3. **Escalar V=h**: `V = h * alpha` con alpha learnable inicializado pequeño (amortigua el gradiente directo)
4. **LR aún más bajo** (e.g., 5e-5): probablemente no resuelve el problema estructural

### Cambio 1 — impacto en params

```
Antes (con W_V): 254,593 params
Después (sin W_V): 234,113 params  (−20,480 = 5 capas × 64×64)
```

### Estado del cluster al cierre de sesión

- 596512, 596513, 596522 corriendo en H100s
- Script `sbatch_wn18rr_transduct_novw_4gpu.sh` listo para lanzar manualmente (4-GPU, 100 épocas, verifica que Change 1 no regresa transductivo desde 0.566)

---

## Estado actual — 2026-04-28 (sesión 22): análisis arquitectural profundo — por qué KnowFormer generaliza a inductivo y nosotros no

### Experimentos de ruido gaussiano — DESCARTADOS

Jobs 596498/596499/596500 (noise_std 0.5/1.0/2.0, wn18rr ind v1, 1×H100 cada uno).

| noise_std | ep5 val MRR | ep5 test MRR | Tendencia |
|-----------|-------------|-------------|-----------|
| 0.5 | 0.0216 | 0.0243 | casi plano |
| 1.0 | 0.0170 | 0.0256 | plano desde ep2 |
| 2.0 | 0.0153 | 0.0200 | declinando |

**Causa del fallo**: el ruido se agrega a `batch.x` → que se copia en `batch.x0` → que se re-suma en CADA capa vía residual BF `h = h + batch.x0`. Con T=5 capas y noise_std=1.0, la señal del anchor (`query_emb` con std≈0.01) queda aplastada por ruido acumulado de magnitud ~5. KnowFormer NO tiene residual BF — su ruido es fresco en cada capa, no se acumula. La intervención estaba mal adaptada a nuestra arquitectura.

### Eliminaciones de esta sesión

- FiLM (`use_film_e`, `proj_e`) eliminado del codebase completo. 255,105 params (−20,480 = 5 capas × 64×64).
- Noise injection descartada como dirección en la forma actual.

### Diagnóstico de los logs de ruido

Los logs son catastróficos — val MRR ~0.015-0.02 en los tres, prácticamente aleatorio. Antes de analizar por qué, hay algo importante en la implementación que los destruye, lo explico abajo.

### Diagnóstico de los logs

| noise_std | ep5 val MRR | ep5 test MRR | Tendencia |
|-----------|-------------|-------------|-----------|
| 0.5 | 0.0216 | 0.0243 | casi plano |
| 1.0 | 0.0170 | 0.0256 | plano desde ep2 |
| 2.0 | 0.0153 | 0.0200 | declinando |

**El ruido destruyó el entrenamiento.** La causa está en el BF residual:

```python
# MultiModel.forward()
batch.x0 = batch.x          # ← x0 captura el ruido

# MultiLayer.forward() en cada capa
h = h + batch.x0            # ← re-agrega x0 CADA capa
```

El ruido de los nodos no-anchor se suma **T veces** a través de todas las capas. Con T=5 y noise_std=1.0, la señal efectiva del anchor (`query_emb` con std=0.01) queda aplastada por ruido acumulado de magnitud ~5. KnowFormer NO tiene residual BF — su ruido se aplica solo al primer paso del QK-stream y no se propaga. La intervención está mal adaptada a nuestra arquitectura.

### Por qué KnowFormer funciona para inductivo (análisis profundo)

Leyendo el código completo de `Knowformer/src/model.py`, la diferencia arquitectónica es mucho más profunda de lo que pensábamos.

#### KnowFormer no es un transformer sobre un grafo — es NBFNet dentro de un transformer

Dentro de cada `KnowformerLayer.forward()`:

```python
# Q/K stream: propaga NBF desde ceros, pesos = fc_qk_z(query_emb)
qk_x = zeros(B, N, 1).normal_(0, 4)                    # simetría rota
qk_x = fc_qk_x(cat([x, qk_x]))                         # proyectar al espacio d
for layer in self.qk_layers:                            # num_qk_layer=2 pasos NBF
    qk_x = layer(qk_x, qk_z, graph)                    # generalized_rspmm: Σ z[r] ⊙ h[u]

# V stream: propaga NBF desde one-hot del head, pesos = fc_z(query_emb)
v_x = zeros(B, N, d)
v_x[:, h_index] = 1                                     # head labeling (no es rel_emb, es one-hot)
v_x = fc_v_x(cat([x, v_x]))
for layer in self.v_layers:                             # num_v_layer=2 pasos NBF
    v_x = layer(v_x, z, r_index, graph)                 # generalized_rspmm

q, k = fc_to_qk(qk_x).chunk(2)
v = v_x
x = x + attn(q, k, v)
```

**La operación `generalized_rspmm`** es exactamente la iteración de NBFNet DistMult:
```
output[v] = Σ_{(u,r,v)} z[b,r] ⊙ qk_x[b,u]
```
donde `z[b,r]` viene del query embedding y `qk_x[b,u]` es el estado actual del nodo. Esto es la misma semántica que `h_x ⊙ (W_r * q)` de NBFNet.

#### Las diferencias críticas

**1. Q y K no leen de `x` — tienen su propio NBF interno**

En KnowFormer: Q/K provienen de un NBF separado (`qk_stream`) que empieza de ceros con ruido, propaga con pesos relacionales, y NUNCA lee `x`. Es puramente relacional.

En el nuestro: `Q = W_Q(x0)`, `K = W_K(h^{t-1})`. Estas leen la representación acumulada que se contamina con el contexto de entidades del grafo de train.

**2. V tiene su propio NBF con one-hot head labeling**

En KnowFormer: `V` viene de un NBF que propaga desde `v_x[h_index] = 1` (one-hot, no rel_emb) con pesos relacionales. V codifica "qué caminos relacionales llevan desde el head hasta cada nodo". Esto es transferible: los caminos relacionales existen igual en train y test.

En el nuestro: `V = W_V(h)` donde `h` acumula contexto de entidades. `W_V` aprende a proyectar la distribución del grafo de train — no transfiere.

**3. Los Q/K/V se recomputan desde cero en cada capa**

Dentro de cada `KnowformerLayer`, se lanza un NBF fresh desde ceros. Nunca hay contaminación de información de entidades del grafo anterior. El noise en qk_x tampoco se acumula porque se recomputa en cada forward.

**4. La atención es secundaria — el trabajo lo hace el NBF**

El mecanismo de atención linear en KnowFormer agrega Q/K/V que ya son puramente relacionales. La atención combina "¿qué tan bueno es el camino desde el anchor hasta v?" (Q/K stream) con "¿qué tan bueno es el camino desde el head hasta v?" (V stream). Es la intuición de path-based KGC.

#### Por qué el nuestro no transfiere

| Componente | Nuestro | KnowFormer |
|---|---|---|
| Q | `W_Q(x0)` + bias query | NBF 2-capas desde ceros + ruido |
| K | `W_K(h^{t-1})` + bias query | mismo NBF que Q |
| V | `W_V(h^{t-1})` * gate | NBF 2-capas desde one-hot head |
| `h^{t-1}` | acumula contexto de entidades | nunca entra a Q/K/V |

La representación `h^{t-1}` en nuestro modelo acumula contexto de qué entidades son los vecinos en el grafo de train. `W_K(h)` aprende a identificar "buenos sources de mensajes" según ese contexto. En el grafo inductivo de test, los mismos vectores de h contienen contexto de entidades completamente distintas — `W_K` proyecta a las mismas direcciones que aprendió para entidades del train.

`W_V(h)` tiene el mismo problema: aprende a proyectar la distribución de representaciones del train graph. En test, esa distribución es distinta.

#### Lo que necesita cambiar (dirección, sin implementar aún)

La solución consistente con la arquitectura de KnowFormer sería que **Q/K/V no lean de `h` sino de propagaciones relacionales propias**. Específicamente:

- **V debe ser una función de `(r_uv, r_q)` aplicada a la representación que viaja, NO una proyección lineal de h** — en el límite, `V = h ⊙ z_r(r_q)` donde `z_r` no tiene componente `W_K/W_V` que aprenda distribución de entidades.
- Alternativamente: eliminar `W_V` completamente y usar `V = h * gate` (gate = función de arista × query) — esto se aproxima al DistMult de NBFNet donde el "V" es directamente h sin proyección adicional.
- El insight sobre el ruido no estaba equivocado en principio, pero el ruido no ataca el problema real: el problema no es simetría inicial sino que `W_K` y `W_V` son funciones de la representación acumulada, y esa representación memoriza el grafo de entrenamiento.

---

## Estado actual — 2026-04-28 (sesión 21): ablación FiLM transductivo + clarificación arquitectura

### Pregunta respondida: ¿FiLM se aplica solo sobre E?

**Sí.** En el código actual (`layer/exphormer.py`), los tres mecanismos de query conditioning son distintos:

| Componente | Forma | ¿Es FiLM? |
|---|---|---|
| E (edge features) | `E = E * (1.0 + proj_e(shared_edge))` | **Sí** — multiplicativo con residual identidad |
| Q (query) | `Q_h = Q_h + proj_q(shared_node)` | No — bias aditivo |
| K (key) | `K_h = K_h + proj_k(shared_node)` | No — bias aditivo |
| V gate | `V_h * (V_gate(edge_attr) + proj_vg(shared_edge))` | No — gate multiplicativo sin `1+` |

FiLM en sentido estricto (`x * (1 + scale)`) solo se aplica a E. El V-gate es multiplicativo pero sin el residual identidad, por eso no colapsa a cero cuando el gate ≈ 0 solo bajo condiciones extremas, pero tampoco tiene la propiedad de preservar la señal cuando scale→0. Q y K reciben bias aditivo (más estable para el scoring).

### Experimento lanzado: ablación use_film_e

**Job 594125** — `transduct_nofilme` (4 H100, 4 GPUs)

- Script: `sbatch_wn18rr_transduct_nofilme.sh`
- Config: `configs/Exphormer/wn18rr_transduct_nofilme.yaml`
- Log: `logs/wn18rr_transduct_nofilme_594125.out` (en NFS)
- **Única diferencia vs. job 593831**: `use_film_e: False`
- Resto idéntico: L=5, d=64, n_heads=4, lr=0.0008, 100 épocas, train_steps_per_epoch=10854

**Baseline de referencia (job 593831)**: val=0.5629, test=0.5628 @ ep13, `time_iter≈2.82s`

**Qué buscamos**: si MRR cae respecto a 0.5628 → FiLM en E contribuye; si se mantiene o sube → se puede eliminar sin pérdida (y reduce un Linear).

### Estado del cluster al cierre de sesión

- Job 594125 en cola/corriendo (4 H100, transduct_nofilme)
- Sin otros jobs activos de mojeda_imfd (solo bash interactivo 594120)

### Pendiente (no cambia de sesión 20)

1. **Regresión de velocidad 2.7×** — sigue sin resolverse. Ver análisis completo en sesión 20 abajo.
2. **Ruido gaussiano en inductivo** — cuando velocidad esté resuelta o en 1 GPU.
3. **Interpretar job 594125** — comparar MRR final vs. 0.5628.

---

## Estado actual — 2026-04-28 (sesión 20): diagnóstico refinado de la regresión de velocidad 2.7×

### Observación del usuario que cambió el diagnóstico

El usuario reportó que **antes de unificar las tablas de embedding (commit 0466cde) el run se demoraba ~50 min/época**, y **después de la unificación se duplicó a ~2h/época**. Esto contradice aparentemente las notas de sesión 17, que decían:

> Option B: cada `ExphormerAttention` con su `nn.Embedding` independiente (como `0466cde`) | `time_iter = 2.795s`. **No es eso.**

### Por qué Option B no detectó la causa

Releyendo sesión 17 con cuidado: Option B fue un **revert parcial**. Solo restauró las tablas de embedding **dentro de `ExphormerAttention`** (per-layer), pero NO revirtió:

- `KGCNodeEncoder` (sigue leyendo `batch.query_emb`)
- `KGCHead` (sigue leyendo `batch.query_emb`)
- `ExpanderEdgeFixer` (sigue computando `proj_exp_edge(batch.query_emb[graph_idx])`)

Por lo tanto, la regresión podría estar en cualquiera de estos tres componentes, que NO se testearon revirtiendo. La intuición del usuario es consistente: la unificación COMPLETA causó la regresión, y solo se testeó revirtiendo una parte.

### Cambio computacional real en `ExpanderEdgeFixer`

Hay un cambio que NO se ha testeado revertir y que sí tiene impacto computacional real:

**0466cde** (`encoder/exp_edge_fixer.py`):
```python
self.exp_edge_query_emb = nn.Embedding(R, dim_edge)
return self.exp_edge_query_emb(query_rel)   # gather: O(E_exp × d)
```

**Actual**:
```python
self.proj_exp_edge = nn.Linear(dim_hidden, dim_edge, bias=False)
return self.proj_exp_edge(batch.query_emb[graph_idx])  # gather + matmul: O(E_exp × d × d)
```

Con E_exp ≈ 3.9M edges (16 grafos × 245,658 expander edges), d=64:
- **Original**: ~3.9M × 64 = 250M ops (solo gather)
- **Actual**: 250M (gather) + ~16B FLOPs (matmul)

Y este matmul corre en cada forward antes del loop de capas. Su backward retiene `batch.query_emb[graph_idx]` (~1GB de activación, shape (3.9M, 64) float32) durante todo el loop checkpoint. Esa activación no se libera entre capas porque viene del encoder, antes del loop.

### Doble convergencia de gradiente sobre `query_rel_emb`

Combinado con que `batch.x0` también tiene ruta de gradiente a `query_rel_emb` (vía `KGCNodeEncoder` que ahora lee `batch.query_emb`), tenemos:

```
batch.x0[anchor] ──────────┐
batch.expander_edge_attr ──┼──→ MISMO query_rel_emb.weight
qemb (en _run × L capas) ──┘
```

**Triple convergencia** sobre el mismo parámetro durante el backward del checkpoint. Crucialmente, `batch.x0` y `batch.expander_edge_attr` se computan **antes** del loop de checkpoint, y se acceden **dentro** vía closure. Con `use_reentrant=False`, durante la recomputation del backward, PyTorch tiene que resolver este grafo en cada una de las L capas, y los gradientes de las L capas se acumulan en `query_rel_emb.weight` con dependencias que no existían en el original.

En 0466cde no había overlap entre los caminos:
```
batch.x0[anchor] ←── KGCNodeEncoder.rel_emb        (tabla independiente)
batch.expander_edge_attr ←── exp_edge_query_emb    (tabla independiente)
shared_rel_emb_table en cada capa                  (tabla independiente)
```
Cero overlap entre los caminos de gradiente. Tres tablas distintas → tres rutas de gradiente independientes hacia tres parámetros distintos.

### Test propuesto (concreto y dirigido)

Revertir SOLO `ExpanderEdgeFixer` y `KGCNodeEncoder` (y opcionalmente `KGCHead`) para que tengan tablas propias (como en 0466cde), manteniendo el resto del código actual. Esto es un revert parcial pero dirigido a los dos componentes que NO se testearon en sesión 17.

**Cambios a aplicar**:

1. `encoder/node_encoders.py` (`KGCNodeEncoder`):
   - Restaurar `self.rel_emb = nn.Embedding(num_relations, dim_emb)`
   - Forward: `r_emb = self.rel_emb(batch.query_relation); h[anchor_global] = r_emb`

2. `encoder/exp_edge_fixer.py` (`ExpanderEdgeFixer`):
   - Reemplazar `self.proj_exp_edge` por `self.exp_edge_query_emb = nn.Embedding(num_relations, dim_edge)`
   - `_exp_attr`: `query_rel = batch.query_relation[graph_idx]; return self.exp_edge_query_emb(query_rel)`

3. `network/heads.py` (`KGCHead`) — opcional:
   - Restaurar `self.rel_emb` propia, no leer `batch.query_emb`

Si esto baja `time_iter` a ~1s → confirmado el diagnóstico: la causa es que el grafo de gradiente actual converge sobre `query_rel_emb` desde múltiples puntos del encoder + activación grande retenida del matmul `proj_exp_edge`.

### Alternativa más quirúrgica (1 línea)

Antes de aplicar el revert completo, probar primero:

```python
# network/model.py línea 368
batch.x0 = batch.x.detach()  # rompe ruta de gradiente x0 → query_rel_emb
```

Esto rompe UNA de las tres rutas de gradiente. Si baja la velocidad parcialmente, la hipótesis se confirma parcialmente. Si no cambia nada, la causa es el matmul de `proj_exp_edge` y/o la retención de activación de `batch.query_emb[graph_idx]`. Trade-off: pierde el gradiente vía residual BF, pero el gradiente principal sigue fluyendo por `qemb` y por `batch.expander_edge_attr`.

### Pendiente

Aplicar y testear en 1 GPU. El usuario interrumpió la aplicación del detach para hacer notar que su intuición ya descarta partes del diagnóstico. Pendiente: decidir entre el revert dirigido (más probable que arregle) o el detach (más quirúrgico).

---

## Estado actual — 2026-04-26 (sesión 19): verificación de equivalencia vs. 0466cde + job pendiente

### Comparación arquitectura actual vs. commit 0466cde (0.566 MRR)

Pregunta respondida: ¿en qué difiere el código actual del commit que logró 0.566 MRR transductivo?

**Diferencia principal — embedding de relaciones:**

| Componente | `0466cde` | Actual |
|---|---|---|
| NodeEncoder | `rel_emb: Embedding(R, d)` propia | usa `batch.query_emb` (de MultiModel) |
| KGCHead | `rel_emb: Embedding(R, d)` propia | usa `batch.query_emb` (de MultiModel) |
| ExpEdgeFixer | `exp_edge_query_emb: Embedding(R, d_e)` | `proj_exp_edge: Linear(d_h, d_e)` sobre `batch.query_emb` |
| ExphormerAttention | `shared_rel_emb_table: Embedding(R, d)` por capa | ninguna — usa `batch.query_emb` directamente |
| MultiModel | sin tabla compartida | `query_rel_emb: Embedding(R, d)` — **única fuente** |

**Equivalencia matemática confirmada**: para los configs de producción (`use_query_conditioning=True`, `use_film_e=True`), Q/K/E/gate/scoring son idénticos. FiLM estaba activo en `0466cde` (proj_e siempre creado bajo `use_query_conditioning`); el flag `use_edge_gating` controlaba el V-gate. Actual: `use_film_e=True` (default) + V-gate siempre activo bajo `use_query_conditioning` → comportamiento idéntico.

**Flags experimentales eliminados** (no en producción en 0466cde): `use_nbf_v`, `use_pna`, `use_distmult_v`, `use_rel_matrix_v`, `use_vrmpnn`, `inductive_routing`, `use_alpha_mix_qk`, `gate_rel_mult`, `use_film_ffn`, `ffn_type`.

**Params**: 281,345 (0466cde) → 275,585 (actual). Reducción por L tablas redundantes eliminadas.

### Smoke test del código actual (transductivo)

```
params: 275,585 — exit 0
```

El código actual corre sin errores con `wn18rr_transduct_best.yaml`.

### Estado de jobs

- **Job 593799** (4 GPUs, transductivo): cancelado. Causa: solo 1 GPU libre en el cluster. Job 593660 de nschiaffino ocupa 6 GPUs con ~6 días restantes de wall time.
- **Script listo**: `sbatch_wn18rr_transduct_2gpu.sh` (2 GPUs, `wandb.use True`). Comando para cuando liberen recursos:

```bash
cd /local_scratch/mojeda_imfd/Doctorado/Exphormer_Max
sbatch sbatch_wn18rr_transduct_2gpu.sh
```

- Config: `wn18rr_transduct_best.yaml`, `optim.base_lr 0.0004` (2e-4 × 2 GPUs), `train.auto_resume True`, `wandb.use True`. `train_steps_per_epoch` sin cambio (el trainer divide por `world_size` internamente).
- La sesión interactiva (job 593750, 1 GPU) será cerrada manualmente por el usuario para liberar 1 GPU adicional.

---

## Estado actual — 2026-04-26 (sesión 18): unificación correcta de tablas de embeddings

### Diagnóstico del estado heredado de sesión 17

Al revisar el código al inicio de la sesión, se encontró que el estado real del código **no coincidía** con lo descrito en el plan de sesión 15/16. La sesión 17 había revertido la unificación para intentar solucionar la regresión de velocidad, pero dejó el código con:

- **L+3 tablas independientes de r_q** (L por capa en `ExphormerAttention` + 1 en `KGCNodeEncoder` + 1 en `ExpanderEdgeFixer` + 1 en `KGCHead`)
- El comentario en `MultiModel.__init__` decía "single canonical embedding" pero el código hacía lo contrario
- La nota "Cambio #3 ✅ COMPLETADO" en el plan de sesión 15 estaba desactualizada

### Unificación implementada (sesión 18)

**Diseño final** — 2 tablas de embedding en total:

| Tabla | Módulo | Shape | Rol |
|-------|--------|-------|-----|
| `query_rel_emb` | `MultiModel` | `(R, d)`, std=0.01 | **única tabla r_q** — lookup una vez por forward |
| `emb` | `RelationEmbeddingEncoder` | `(R, d)` | r_uv (features de aristas del KG) — legítimamente separada |

**Cambios por archivo:**

- `network/model.py`: añadido `self.query_rel_emb = nn.Embedding(R, d)` con std=0.01 en `MultiModel.__init__`. Eliminados `_num_rel` y `num_relations` de la cadena `MultiLayer` → `GlobalModel`. Forward: `query_emb = self.query_rel_emb(batch.query_relation)` antes del encoder.
- `encoder/node_encoders.py` (`KGCNodeEncoder`): eliminado `self.rel_emb`. Usa `batch.query_emb` directamente para la boundary condition del anchor.
- `encoder/exp_edge_fixer.py` (`ExpanderEdgeFixer`): eliminado `self.exp_edge_query_emb`. Añadido `self.proj_exp_edge = nn.Linear(dim_hidden, dim_edge, bias=False)` (std=0.01). Expander edge features = `proj_exp_edge(batch.query_emb[graph_idx])`.
- `layer/exphormer.py` (`ExphormerAttention`): eliminados `num_relations` y `self.shared_rel_emb_table`. Q/K/E/gate usan `batch.query_emb[batch.batch]` (per-node) y `batch.query_emb[edge_graph]` (per-edge).
- `network/heads.py` (`KGCHead`): eliminado `self.rel_emb` y `num_relations`. Usa `batch.query_emb` directamente para el scoring.

**Params resultantes**: 275,073 (ind v1, L=5, R=18) / 275,585 (trans, L=5, R=22). Ambos smoke tests exit 0.

### Fix del path de gradient checkpoint (KnowFormer-style)

Revisando el código de KnowFormer (`Knowformer/src/model.py`), se confirmó que el patrón correcto es pasar `z` (equivalente a `query_emb`) como **argumento posicional explícito** al loop de capas, NO adjunto al objeto batch. KnowFormer:

```python
z = self.query_embedding(r_index)   # (B, d) — una vez
for layer in self.layers:
    x = layer(..., z, ...)          # z como argumento, no en batch
```

Se actualizó `MultiModel.forward()` para seguir este patrón en el path `grad_checkpoint=True`:

```python
# Antes del loop: sacar query_emb del batch
del batch.query_emb
# En cada checkpoint: pasar como arg explícito
x_out, ea_out = checkpoint(_run, x_in, ea_in, query_emb, use_reentrant=False)
# Dentro de _run: _batch.query_emb = qemb; ...; del _batch.query_emb
# Después del loop: restaurar para KGCHead
batch.query_emb = query_emb
```

PyTorch trata los args posicionales de `checkpoint()` como "entradas guardadas" (no como activaciones retenidas), evitando que un tensor `requires_grad=True` se serialice a través de los L replays de backward via el closure del batch.

El path sin grad_checkpoint (producción actual con 4 GPUs) no cambia: `batch.query_emb` se setea antes del encoder y persiste hasta `post_mp`.

Smoke tests: exit 0 con `grad_checkpoint=False` y `grad_checkpoint=True`.

### Estado de la regresión de velocidad

La regresión 2.7× de sesión 17 **sigue sin diagnóstico definitivo**. Los descartes anteriores mostraron que la causa no está en el embedding ni en la unificación. La causa raíz está en otro lugar de los 5 archivos refactorizados. Pendiente: hacer bench de 1 GPU con el código actual y comparar con commit `0466cde`.

---

## Estado actual — 2026-04-24 (sesión 17): regresión de performance del refactor

### Síntoma

- Run histórico **578765** (commit `0466cde`, código pre-refactor): **~50 min/época**, `time_iter ≈ 1.034s` (4 GPU), 281,345 params, alcanzó **0.566 MRR test** transductivo en WN18RR.
- Job **592218** (post-refactor sesión 16, mismo `wn18rr_transduct_best.yaml`): **~2h 7min/época**, `time_iter ≈ 2.808s` (4 GPU), 271,489 params. Mismo MRR esperado (entrenando), pero **2.7× más lento** pese a tener menos parámetros.

### Descartes (lo que NO es la causa)

| Hipótesis | Test | Resultado |
|---|---|---|
| DDP allreduce overhead con embedding compartido | Bench 1 GPU (sin DDP), código NEW | `time_iter = 2.79s` (idéntico a 4-GPU). **No es DDP.** |
| Cross-checkpoint capture de `batch.query_emb` | Option A: layer hace lookup propio adentro, manteniendo embedding compartido | `time_iter = 2.796s`. **No es eso.** |
| Embedding compartido entre L capas (sesión 15) | Option B: cada `ExphormerAttention` con su `nn.Embedding` independiente (como `0466cde`) | `time_iter = 2.795s`. **No es eso.** |
| Versión de PyTorch / CUDA / driver | Bench 1 GPU del **código exacto** del commit `0466cde` (mismo entorno actual) | `time_iter = 0.96s`. **No es ambiental — la regresión está en el refactor.** |
| `grad_checkpoint=False` para aislar interacción | Test bloqueado por OOM (5 capas × 40K nodos no caben sin checkpoint) | No testeable directamente. |

### Conclusión hasta ahora

La regresión **vive en los 5 archivos del refactor** (`layer/exphormer.py`, `network/model.py`, `network/heads.py`, `encoder/node_encoders.py`, `encoder/exp_edge_fixer.py`), pero **no en los componentes obvios**:
- No es la centralización del lookup (Option A revierte sin efecto).
- No es la unificación del embedding (Option B revierte sin efecto).
- El compute en `propagate_attention` y en `forward` es funcionalmente idéntico OLD vs NEW (mismas Linears, mismos índices, mismo scatter).

### Hipótesis activa al cierre de sesión

`batch.query_emb` (tensor con `requires_grad=True` producido por `MultiModel.query_rel_emb(...)`) queda **adjunto al objeto Batch** que después entra al `torch.utils.checkpoint.checkpoint(_run, ...)` loop. El closure de `_run` captura `_batch` por referencia. Aunque las capas de Option B **no leen** `batch.query_emb`, está colgado en la jerarquía de atributos del Batch y PyTorch (con `use_reentrant=False`) lo retiene como activación a través de los L replays de backward.

**Test propuesto y en curso**: `del batch.query_emb` antes del loop de capas, reasignar después para que `KGCHead` siga teniéndolo. Si baja a ~1s/iter → confirmado. Si no → la regresión es otra cosa más sutil que requiere más bisect.

### Job 592218 en estado

- Cancelado tras varios intentos de fix fallidos (Option A y mi primer fix de "proyectar a B-scale antes de broadcast" que causó deadlock NCCL en backward).
- Estado del código actual al cerrar la sesión: NEW + Option B (per-layer independent embeddings) + del/restore de `batch.query_emb` alrededor del loop.
- **Pendiente**: confirmar resultado del bench 592876, decidir siguiente paso (correr 4-GPU si baja, o seguir bisectando si no).

### Lecciones metodológicas

- **Bench de 1 GPU es el discriminador correcto** para aislar regresiones de DDP vs código. Hacerlo antes de quemar 8h en 4 GPUs.
- **El refactor "estético" de unificación de embeddings no es benigno bajo gradient checkpoint** — la interacción con `use_reentrant=False` y el closure-capture de Batch es no-trivial. Cualquier futuro refactor que mueva tensores `requires_grad=True` al objeto Batch necesita un bench inmediato.
- `git fsck --no-reflog` recupera stashes dropeados — no perder esta capacidad cuando se hace bisect destructivo.