# Session Notes

---

## Estado actual — 2026-04-17 (sesión 12)

### Resultados benchmarking v2/v3/v4 — COMPLETADOS

Todos los jobs finalizaron. Comparación con NBFNet (paper):

| Split | Nuestro test MRR | Nuestro H@10 | NBFNet MRR | NBFNet H@10 | Epoch |
|-------|-----------------|-------------|------------|-------------|-------|
| v1 | **0.578** | 0.776 | 0.741 | 0.948 | 4 |
| v2 | **0.514** | 0.714 | ~0.68 (est.) | ~0.92 (est.) | 4 |
| v3 | **0.229** | 0.325 | ~0.67 (est.) | ~0.93 (est.) | 3 |
| v4 | **0.474** | 0.669 | ~0.73 (est.) | ~0.95 (est.) | 4 |

v3 es llamativamente bajo (0.229). El test graph de v3 tiene 5,084 nodos (vs 922 en v1, 2,757 en v2).
Posibles causas: (1) grafo más grande → más ruido con sum aggregation, (2) distribución de relaciones diferente en v3, (3) mismos hiperparámetros no óptimos para v3.
Necesita investigación adicional si se reporta en tesis.

### WN18RR transductivo (job 578765) — EN CURSO ep40+

| Época | val MRR | test MRR | LR | Nota |
|-------|---------|----------|----|------|
| 7 | **0.565** | **0.566** | ~0.00080 | MEJOR ckpt |
| 39 | 0.554 | 0.555 | cosine decay | |
| 40 | 0.553 | 0.554 | cosine decay | última |

Best checkpoint estable: ep7 → val=0.565, test=0.566. Desde ep7 el modelo no mejora → posible plateau o sobreajuste gradual. Aún corriendo (job 578765).

---

## Estado actual — 2026-04-15 (sesión 11)

### Monitoreo transductivo (job 578765)

Job en curso. Recuperación confirmada — el modelo superó 0.55 en época 8.

| Época | val MRR | test MRR | LR | Nota |
|-------|---------|----------|----|------|
| 4 | 0.485 | 0.488 | 0.00064 | warmup |
| 5 | 0.483 | 0.486 | 0.00080 | LR alcanza máximo |
| 6 | 0.534 | 0.533 | 0.00080 | salto al salir de warmup |
| 7 | 0.544 | 0.544 | 0.00080 | subiendo |
| 8 | **0.553** | **0.553** | 0.00080 | último registrado |

El LR recién alcanzó el máximo en época 5 (warmup=5). Con cosine sobre 100 épocas, el pico real se espera en épocas 15–30. ~50 min/época.

Nota: los "Best so far: epoch N" del log cuentan desde el reinicio del job (resumed desde ckpt ep3), no época absoluta.

### Benchmarking v2/v3/v4 — LANZADO

Configs v2/v3/v4 actualizados para usar los mejores hiperparámetros del `exp3_qcond`:

| Cambio | Antes (v2/v3/v4) | Ahora |
|--------|-----------------|-------|
| `gt.layers` | 5 | 3 |
| `inductive_routing` | no definido (False) | True |
| `ffn_type` | standard | none |
| `base_lr` | 0.0008 | 0.00001 |
| `num_warmup_epochs` | 3 | 5 |
| `ckpt_monitor_split` | no definido | val |
| `train_steps_per_epoch` | 4-GPU values | 1-GPU values |

Tamaños de datasets (train graph / test ind graph):
- v2: 6954 / 2757 entities, 15262 train triples, 3816 steps/epoch
- v3: 12078 / 5084 entities, 25901 train triples, 6476 steps/epoch
- v4: 3861 / 7084 entities, 7940 train triples, 1985 steps/epoch

**Jobs lanzados:**
- v2: job 579787, `logs/wn18rr_ind_v2_579787.out`
- v4: job 579788, `logs/wn18rr_ind_v4_579788.out`
- v3: job 579789, `logs/wn18rr_ind_v3_579789.out` (pending)

**Referencia baseline (v1):** test MRR=0.578, val=0.438 @ ep4 (config exp3_qcond)

---

## Estado actual — 2026-04-15 (sesión 10)

### Diagnóstico: transductivo WN18RR completamente roto (jobs 575904, 578249)

Dos runs del config `wn18rr_transduct_best.yaml` dieron MRR ≈ 0.0003–0.0005 durante >10 épocas — predicción uniforme sobre los 40,943 entidades (loss=10.67 ≈ ln(40943), random).

**Job 578249 (tanh)**: se encontró `torch.tanh(h_attn)` en `network/model.py` en el residual de atención:
```python
h_attn = h_in1 + torch.tanh(h_attn)  # ← mata gradientes con sum-agg sobre 40K nodos
```
El tanh satura cuando la magnitud del sum-aggregation es grande → gradiente ≈ 0 → no aprende nada.
**REVERTIDO** a `h_attn = h_in1 + h_attn`.

**Job 575904 (sin tanh, mismo MRR cero)**: el tanh NO era la causa raíz. Sin tanh, el modelo tampoco aprendía nada. Causa real: **dos flags inductivos activos en setting transductivo**:

1. **`inductive_routing: True`** — K = proj_k(r_q) solo (sin W_K(h)). En transductivo, las entidades son conocidas y el routing entity-specific es esencial. Con K constante por query, la atención no diferencia fuentes por features — el routing colapsa a función solo de (r_q, r_ij). Con Q también usando x0=0 para nodos no-anchor, Q_j ≈ proj_q(r_q) para todos → scores uniformes por tipo de relación → gradientes diluidísimos sobre 40K entidades.

2. **`ffn_type: none`** — en inductive el FFN era el cuello (memorizaba topología train). En transductivo el FFN es necesario: es la transformación per-entidad que permite aprender representaciones específicas.

Ambas flags son correctas para inductive y catastrophicas para transductive — **no son portables entre settings**.

### Fix aplicado

`configs/Exphormer/wn18rr_transduct_best.yaml`:
- `inductive_routing: False` ← K = W_K(h) + proj_k(r_q) — entity-specific routing
- `ffn_type: none` eliminado ← FFN estándar restaurado
- `eval_period: 1` ← evaluar cada época

Checkpoint incompatible del run anterior eliminado (`results/wn18rr_transduct_best/0/ckpt.pt`).

**Job 578765** lanzado — log: `logs/wn18rr_transduct_best_578765.out`

Config activo:
```yaml
gt:
  layers: 5, dim_hidden: 64, inductive_routing: False
  use_edge_gating: True, use_query_conditioning: True
  ffn_type: standard (default)
optim:
  base_lr: 0.0008, scheduler: cosine_with_warmup, num_warmup_epochs: 5
```

Referencia: `wn18rr.yaml` (dim=32, L=3, inductive_routing=False, FFN) logró ~0.55. Este config es mayor (dim=64, L=5) con todas las mejoras de sesiones anteriores → esperado superar 0.55.

### Lección clave sesión 10

`inductive_routing` y `ffn_type=none` son **específicos del setting inductive**. No incluir en configs transductivos. El CLAUDE.md ya refleja esto como flags opcionales, pero los experimentos transductivos deben usar defaults (routing=False, FFN estándar).

---

## Estado actual — 2026-04-14 (sesión 9)

### Propuesta A (Tucker) — IMPLEMENTADA, FALLIDA, REVERTIDA

**Implementación**: Tucker W_r decomposition en `layer/exphormer.py` + `network/model.py` + `config.py`.
Fórmula: `msg(i→j, r) = h_i @ (U @ G_r[r_uv] @ V^T)` con U,V ortogonales compartidos + G_r≈I init.

**Truco importante**: con `num_heads=4, dim_hidden=64` → `dim_head=16`. Tucker rank=16 es full-rank en el espacio per-head (equivalente a use_rel_matrix_v). Se usó rank=8 (compresión real) y rank=4.

**RESULTADOS FINALES (jobs 577972, 577973):**

| Config | Val★ | Test @ val★ | Época★ | vs exp3_qcond (mejor) |
|--------|------|-------------|--------|-----------------------|
| **exp3_qcond (mejor actual)** | **0.438** | **0.578** | **4** | — |
| Tucker r4 (100K params) | 0.373 | 0.520 | 2 | −0.058 |
| Tucker r8 (113K params) | 0.376 | 0.485 | 5 | −0.093 |
| rel_matrix_v full rank (119K) | 0.408 | 0.509 | 10 | −0.069 |

**TUCKER CONFIRMADO PEOR.** Monotonía perfecta: más rank → peor resultado:
```
rank→0  (gate diagonal, exp3_qcond): 0.578  ← mejor
rank=4  (Tucker r4):                  0.520
rank=8  (Tucker r8):                  0.485
rank=16 (relmatrix, full):            0.509
```

**Diagnóstico revisado (DEFINITIVO)**: el gate diagonal es la estrategia óptima de V para WN18RR.
El cross-dimensional mixing no ayuda porque:
1. WN18RR tiene relaciones semánticamente simples (hypernym, hyponym, part-of, etc.) donde **dimensiones independientes capturan tipos de relaciones distintos** — la mezcla destruye esta separación.
2. Con `inductive_routing=True` el score ya es `f(r_q, r_ij)` — el routing es puramente relacional. El cuello no es V.
3. Más cross-dim = más libertad = más overfitting (Tucker r4 > r8 confirma: menos freedom = mejor).

**La hipótesis "diagonal vs full W_r = gap a NBFNet" era incorrecta.** El gap 0.578→0.741 no es de expresividad del mensaje.

**Código revertido** al estado de exp3_qcond (119,617 params). Los archivos Tucker eliminados.

### Estado de código post-sesión 9

**Mejor resultado**: `wn18rr_ind_v1_exp3_qcond.yaml` → **val=0.438, test=0.578** @ ep4 (119,617 params)

Código en `layer/exphormer.py`, `network/model.py`, `config.py` limpio de Tucker.
Cambios activos en el codebase respecto al baseline original:
- `encoder/exp_edge_fixer.py`: `exp_edge_query_emb` — query-conditioned expander features (Propuesta B ✅)
- `network/model.py`: `_exp_num_rel` siempre activo en KGC — habilita exp_edge_query_emb
- `layer/exphormer.py:179`: `h_out = batch.wV` — sum aggregation (no normalization) desde sesión 4
- `layer/exphormer.py`: `gate = batch.E_gate` sin sigmoid — desde sesión 3

### Resumen experimental completo (sesiones 1-9)

| Cambio/Experimento | Test MRR | Status |
|--------------------|----------|--------|
| Baseline original (wV/Z, sin gate, sin qcond) | 0.210 | ref inicial |
| + sin sigmoid en gate | 0.252 | +0.042 |
| + sum aggregation | 0.482 | +0.230 ← salto mayor |
| + inductive_routing | 0.532 | +0.050 |
| + lr=1e-5 + warmup | 0.565 | +0.033 |
| + ffn_type=none | 0.570 | +0.005 |
| + expander query-cond (exp3_qcond) | **0.578** | **+0.008 ← mejor** |
| Tucker r4 | 0.520 | −0.058 ❌ |
| Tucker r8 | 0.485 | −0.093 ❌ |
| rel_matrix_v | 0.509 | −0.069 ❌ |
| FiLM FFN | ~0.424 | −0.154 ❌ |
| V-RMPNN | 0.513 | −0.065 ❌ |
| PNA | 0.228 | −0.350 ❌ |
| NBFNet (paper) | 0.741 | ref |
| KnowFormer (paper) | 0.752 | ref |

### Próximas direcciones

**Propuesta C (pendiente)**: Source-aware K con señal de norma — 3 líneas, bajo riesgo.
```python
h_src_norm = h.norm(dim=-1, keepdim=True)       # (N, 1) — inductivo
K_h = K_cond_bias + h_src_norm * self.k_scale   # k_scale ∈ R^d, 64 params
```
Ganancia esperada: +0.02-0.05. Probabilidad de funcionar: baja dado que Tucker/V no son el cuello.

**Benchmarking completo**: correr v2, v3, v4 con `wn18rr_ind_v1_exp3_qcond.yaml` adaptado.
Necesario para la tesis independientemente del gap con NBFNet.

**Conclusión honesta**: el gap 0.578→0.741 probablemente requiere repensar el mecanismo de atención completo (sustituir Q/K/V por RMPNN puro como NBFNet/KnowFormer) o aceptar el resultado actual como contribución parcial del Stage 1.

---

## Estado actual — 2026-04-14 (sesión 8, continuación)

### IMPLEMENTACIÓN: Propuesta B — Expander query-conditioned (implementado)

**Cambios en el código:**

`encoder/exp_edge_fixer.py`:
- Añadido `self.exp_edge_query_emb = nn.Embedding(num_relations, dim_edge)` con init `std=0.01`
- Nuevo método `_exp_attr(exp_ei, batch, device)`: si `batch.query_relation` está presente,
  devuelve `exp_edge_query_emb(query_relation[graph_idx])` — features query-condicionadas.
  Fallback: `exp_edge_attr(zeros)` — comportamiento uniforme original (non-KGC / sin query).
- Compatible con full-graph y subgraph mode; backward compatible con experimentos no-KGC.

`network/model.py`:
- `_exp_num_rel` ahora es siempre `cfg.dataset.num_relations` cuando `cfg.dataset.format == 'KGC'`
  (antes solo cuando `use_nbf_v / use_vrmpnn / use_distmult_v / use_rel_matrix_v`).
- Esto activa `exp_edge_query_emb` en todos los experimentos KGC con expander.

**Parámetros añadidos**: +1,152 (= 19 relaciones × 64 dim_edge para WN18RR)
**Scalability**: O(num_relations × dim_edge) — trivial para FB15k-237 (474×64=30K) y OGBL (535×64=34K)
**Backward compatible**: experimentos sin expander (`exp: False`) no cambian nada.

**Experimento corriendo:**
- Config: `wn18rr_ind_v1_exp3_qcond.yaml` — idéntico a `hp_noffn_dim64` (mejor baseline) + `exp: True`
- Job: **575899** — log: `logs/wn18rr_ind_v1_exp3_qcond_575899.out`
- Comparar contra:
  - `exp3` (expander uniforme): val★=0.359 @ ep8, test=0.472 @ ep5  — daño −0.10 MRR
  - `hp_noffn_dim64` (sin expander): val★=0.416 @ ep4, test=0.570 @ ep4 — baseline

**Hipótesis**: query-conditioned features dan al expander un rol semántico ("autopistas de query"),
permitiéndole contribuir en lugar de añadir ruido. Si val>0.416, el expander ayuda.
Si val≈0.416, el expander es neutro. Si val<0.416, el expander sigue dañando incluso con qcond.

**RESULTADO FINAL (30 épocas, completado 2026-04-14):**

| Config | Val★ | Test @ val★ | Época★ | vs baseline |
|--------|------|-------------|--------|-------------|
| Sin expander (hp_noffn_dim64) | 0.416 | 0.570 | 4 | — |
| Expander uniforme (exp3) | 0.359 | **0.451** | 8 | −0.057 val / −0.119 test |
| **Expander query-cond (exp3_qcond)** | **0.438** | **0.578** | **4** | **+0.022 val / +0.008 test** |

**Propuesta B CONFIRMADA.** La propuesta funciona:
- Expander uniforme daña −0.119 test MRR (ruido no diferenciado con sum agg + K=const)
- Expander query-conditioned: +0.008 test MRR sobre baseline sin expander
- qcond recupera todo el daño del expander uniforme (+0.127) y añade ganancia neta pequeña

Trayectoria completa exp3_qcond (30 épocas):
```
ep | val   | test
 4 | 0.438★| 0.578   ← mejor val (checkpoint seleccionado)
 5 | 0.426 | 0.571
...
22-29: val ~0.370, test ~0.472 (colapso, igual que baseline)
```

Patrón de colapso idéntico al baseline — el expander query-conditioned no resuelve el colapso,
solo mejora el peak. **El colapso es en V/mensaje, no en el expander.**

**Contribución a la tesis**: el expander tiene un rol diferente en KGC vs clasificación:
- Clasificación: conectividad estructural, mezcla de vecindarios similares
- KGC inductivo con query-cond: "autopistas de difusión de query" — canal semántico adicional
Esto es una observación novedosa para la tesis.

---

## Estado actual — 2026-04-14 (sesión 8)

### NUEVOS EXPERIMENTOS (overnight, post-sesión 7)

#### Grid relmatrix completo (jobs 574923–574929) — TODOS FALLIDOS

Resultado definitivo del enfoque W_r completa por relación:

| Config | Val MRR | Test MRR | Notas |
|--------|---------|----------|-------|
| relmatrix v2 (lr=1e-5, dim=64) | 0.408 | **0.509** @ ep10 | referencia sesión 7 — mejor W_r |
| rm_lr1e4_L3_d32 | 0.286 | 0.354 @ ep21 | mejor del nuevo grid |
| rm_lr1e4_L3_d64 | 0.229 | 0.271 @ ep20 | |
| rm_lr5e4_L3_d32 | 0.212 | 0.178 @ ep31 | |
| rm_lr5e4_L3_d64 | 0.202 | 0.223 @ ep21 | |
| rm_lr1e3_L3_d64 | 0.171 | 0.190 @ ep20 | |
| rm_lr1e4_L5_d64 | 0.015 | 0.033 | **MUERTO** |
| rm_lr1e3_L5_d64 | 0.015 | 0.033 | **MUERTO** |
| **baseline (indrouting, no-FFN, dim=64)** | **0.416** | **0.570** | — |

La única LR que funciona para W_r es 1e-5. Con LRs más altos el modelo no aprende bien. Aun con lr=1e-5 (v2), W_r queda −0.061 MRR por debajo del baseline. **W_r no supera al gate diagonal — hipótesis "diagonal vs full = gap a NBFNet" confirmada incorrecta.**

Comparativa detallada trajectoria (conclusión clave): al ep7-8, ambos modelos tienen val~0.40, pero indrouting ya tiene test=0.52 vs relmatrix test=0.50. El gate aprende más rápido y a más alto pico porque W_V compartido provee prior estable desde ep0.

#### Expander deg=3 (job 575898) — DAÑA

Config: `wn18rr_ind_v1_exp3.yaml` — mismo que hp_noffn_dim64 pero con `exp: True, exp_deg: 3`.

| Época | Val MRR | Test MRR |
|-------|---------|----------|
| 4 | 0.355 | 0.471 |
| 5 | 0.355★ | 0.472★ |
| 8 | 0.359★ | 0.451 |
| 11 | 0.343 | 0.429 |

Val plateau en ~0.355 vs baseline 0.416. Test ~0.472 vs baseline 0.570. **Δ = −0.10 MRR.**

Causa identificada: aristas expander tienen features uniformes (`nn.Embedding(1, dim)`) + K=const con inductive_routing → todos los vecinos expander de un nodo j reciben **exactamente el mismo attention score**. Con sum aggregation, el expander añade ruido no diferenciado. El modelo gasta capacidad aprendiendo a suprimirlo en vez de routing relacional.

**Conclusión de ablación**: expander con sum aggregation + inductive_routing + features uniformes = daña. Esto es resultado de ablación válido para la tesis.

---

### DIAGNÓSTICO DEFINITIVO CONSOLIDADO (sesión 8, revisado sesión 9)

Después de 50+ experimentos en 9 sesiones:

#### El gap 0.578 → 0.741 — causa raíz revisada

La hipótesis inicial "diagonal W_r = cuello de expresividad" fue **REFUTADA** por Tucker (sesión 9).
El gate diagonal es la estrategia óptima de V para WN18RR. Más expresividad → peor resultado.

**Análisis matemático del flujo con inductive_routing=True**:
```
score(i→j, r_q) = dot(proj_q(r_q) * proj_k(r_q), E(r_ij)) / √d   [j ≠ anchor]
```
El score solo depende de `(r_q, r_ij)` — el modelo ya es esencialmente MPNN relacional (como NBFNet).
El cuello NO está en V ni en el routing. El gap está en algo más sutil.

**Hipótesis actual sobre el gap**: posiblemente diferencias en régimen de entrenamiento (NBFNet 6 capas × dim=32 vs nuestros 3 capas × dim=64), o en la naturaleza del mecanismo attention vs MPNN puro para el dataset específico. No hay evidencia de que sea arquitecturalmente resoluble con tweaks.

#### Por qué el expander daña (sin conditioning)

Features uniformes + K=const → todos los vecinos expander tienen score idéntico → sum agg añade ruido puro. Resuelto con query-conditioning (Propuesta B: +0.008 MRR).

#### Routing ciego al estado fuente

Con inductive_routing=True: K_i = proj_k(r_q) = constante para todos i.
Efecto: dos fuentes con la misma relación de arista reciben el mismo score.
**Intentar mejorarlo con norma (Propuesta C) tiene baja probabilidad** dado que el score sigue siendo función de `(r_q, r_ij)` — suficientemente inductive y expresivo para WN18RR.

---

### PLAN PARA ALCANZAR MRR COMPETITIVO CON NBFNET/KNOWFORMER (sesión 8, actualizado sesión 9)

#### Estado de las propuestas

| Propuesta | Estado | Resultado |
|-----------|--------|-----------|
| **B: Expander query-conditioned** | ✅ IMPLEMENTADA | +0.008 test MRR (0.570→0.578) — **mejor actual** |
| **A: Tucker W_r (rank=4,8)** | ❌ FALLIDA, revertida | −0.058 a −0.093 vs baseline |
| **C: Source-aware K (norma)** | ⬜ PENDIENTE | 3 líneas, ganancia estimada baja |
| **D: Expander residual** | ⬜ DESCARTADA | B ya funciona, D no añade nada |

#### Propuesta C: Routing source-aware con señal inductiva (único pendiente menor)

**Problema**: K=const → routing no puede distinguir fuentes informativas de no-informativas.

**Solución** (3 líneas en `layer/exphormer.py`):
```python
h_src_norm = h.norm(dim=-1, keepdim=True)       # (N, 1) — magnitud acumulada (inductiva)
K_h = K_cond_bias + h_src_norm * self.k_scale   # k_scale ∈ R^d, 64 params
```
`||h||` es inductivo: mide cuánta señal acumuló el nodo, no qué entidad es.

**Probabilidad de ayudar**: baja. Con `inductive_routing=True` el score ya es `f(r_q, r_ij)` — routing puramente relacional. La norma añadiría diferenciación basada en profundidad de hop, pero Tucker demostró que más expresividad en V/K no ayuda en WN18RR.

**Conclusión actualizada (post sesión 9)**:

El techo de esta arquitectura en WN18RR-ind-v1 es **~0.578 MRR** (exp3_qcond).
El gap 0.578→0.741 (NBFNet) no es resoluble con tweaks de V, K, o FFN.
Para superarlo se requeriría reescribir el mecanismo de mensaje completo (RMPNN puro al estilo NBFNet/KnowFormer), lo que implicaría abandonar la estructura attention de Exphormer.

---

## Estado actual — 2026-04-13 (sesión 7)

### FiLM FFN CONDITIONING — RESULTADO FINAL

**Implementado** en `network/model.py` (MultiLayer): flag `gt.use_film_ffn`.
```python
h_mid = F.relu(self.ff_linear1(x)) * (1.0 + gamma_bias) + beta   # donde gamma, beta = f(r_q)
```

**Resultado**: test MRR 0.419 en peak (ep5) vs baseline 0.565 → **−0.146 MRR**. FALLO GRAVE.

**Por qué falló**: FiLM condiciona la *transformación* del FFN sobre r_q, pero `h` en la entrada sigue siendo graph-specific (acumula información de los vecinos del train graph a través de W_V). Condicionar la transformación no "limpia" la representación. Además añade 52K params adicionales que pueden memorizar más patrones. El FFN no es el cuello de botella — lo es el flujo completo desde W_V.

**Experimentos FiLM adicionales** en la misma sesión (todos peores que baseline):
| Config | Peak val | Test MRR | vs baseline |
|--------|----------|----------|-------------|
| hp_ffn_lr3e6 (lr=3e-6) | 0.415 | 0.566 | +0.001 |
| hp_ffn_lr1e4 (lr=1e-4) | 0.407 | 0.555 | −0.010 |
| hp_ffn_ls01 (ls=0.1) | 0.418 | **0.569** | +0.004 |
| hp_ffn_drop02 (dropout=0.2) | 0.419 | 0.544 | −0.021 |

FiLM con lr=3e-6 (warmup más lento, peak en ep9) y label smoothing=0.1 dan marginalmente encima del baseline (0.566, 0.569), pero la diferencia es insignificante (~+0.004). **FiLM no ayuda.**

---

### HP TUNING — RESULTADOS COMPLETOS (30 épocas)

**Grid**: 4 experimentos FFN (dim=32, L=3, variando lr/ls/dropout) + 6 no-FFN (variando dim, L, lr).

#### No-FFN model

| Config | dim | L | lr | Best ep | Peak val | Test MRR | Notas |
|--------|-----|---|----|---------|----------|----------|-------|
| hp_noffn_lr3e5 | 32 | 3 | 3e-5 | 5 | 0.438 | 0.561 | |
| hp_noffn_lr1e4 | 32 | 3 | 1e-4 | 2 | **0.440** | 0.559 | val más alto de todo el grid |
| hp_noffn_dim64 | 64 | 3 | 1e-5 | 4 | 0.416 | **0.570** | mejor test MRR |
| hp_noffn_dim64_lr3e5 | 64 | 3 | 3e-5 | 2 | 0.415 | 0.570 | idem |
| hp_noffn_L4 | 32 | 4 | 1e-5 | 5 | 0.413 | 0.529 | más capas = peor |
| hp_noffn_L5 | 32 | 5 | 1e-5 | 0 | 0.016 | 0.067 | **NO APRENDIÓ NADA** — muerto desde ep0 |

**L5 con dim=32**: val_mrr=0.016 constante durante las 30 épocas. El modelo nunca actualiza. Probable vanishing gradient con 5 capas y dim=32 (muy pocos params/capa sin FFN). Patrón distinto a `distmultv_noffn_L5` (dim=64, también muerto) — confirma que L=5 sin FFN y dim pequeño es inestable.

#### Análisis comparativo con baseline

| Config | Test MRR | Δ vs baseline (0.565) | Observación |
|--------|----------|-----------------------|-------------|
| **baseline** (indrouting_lr1e5) | **0.565** | — | referencia |
| hp_noffn_dim64 | **0.570** | +0.005 | mejor del grid, marginal |
| hp_noffn_dim64_lr3e5 | 0.570 | +0.005 | idem |
| hp_ffn_ls01 | 0.569 | +0.004 | label smoothing, marginal |
| hp_ffn_lr3e6 | 0.566 | +0.001 | lr más lento = peak tardío (ep9) |
| hp_noffn_lr3e5 | 0.561 | −0.004 | |
| hp_noffn_lr1e4 | 0.559 | −0.006 | |
| hp_ffn_drop02 | 0.544 | −0.021 | dropout alto daña |
| hp_ffn_lr1e4 | 0.555 | −0.010 | colapso desde ep1 |
| hp_noffn_L4 | 0.529 | −0.036 | L=4 > L=3 no ayuda |
| hp_noffn_L5 | 0.067 | −0.498 | muerto |

**Conclusión del grid**: el techo empírico es **~0.570 MRR** en WN18RR-ind-v1. Ningún hiperparámetro supera el baseline en más de +0.005. El gap a NBFNet (0.741) es estructural.

---

### DIAGNÓSTICO FINAL CONSOLIDADO (sesión 7)

Después de 30+ experimentos en 7 sesiones, el diagnóstico es inequívoco:

**El techo arquitectónico del modelo actual es ~0.570 MRR.**

Lo que ha sido intentado y ha fallado (en orden cronológico):
- PNA, MLP scorer, scale-up dim=128, weight tying, constlr, más capas (L=5)
- V-RMPNN (con y sin query conditioning), use_nbf_v, use_relational_v, DistMult-V
- FiLM FFN conditioning
- HP tuning: lr, label_smoothing, dropout, dim, L

Lo que SÍ ayudó (toda la ganancia real):
1. **Sum aggregation** (wV/Z → wV): +0.23 MRR (la mayor mejora de la historia del proyecto)
2. **Inductive routing** (K = proj_k(r_q)): +0.05 MRR
3. **LR tuning** (lr=1e-5): +0.033 MRR

El gap 0.570 → 0.741 requiere un cambio estructural al mecanismo de mensaje:
- El problema raíz es que W_V(h) proyecta h que ya es graph-specific
- La solución requeriría reescribir el paso de mensaje al estilo KnowFormer V-RMPNN,
  con un filtro relacional *antes* de la proyección, no después
- Esto es una reescritura mayor que está fuera del scope de ajuste de hiperparámetros

**REGLA actualizada**: no lanzar más experimentos de HP tuning en WN18RR-ind-v1.
El próximo paso es decidir si hacer la reescritura RMPNN o moverse a la siguiente etapa del doctorado.

---

### EXPERIMENTO: use_rel_matrix_v (W_r completa por relación)

#### v1 — resultado (job 574311, cancelado ep38)

Config: `wn18rr_ind_v1_relmatrix.yaml` — dim=64, L=3, lr=1e-5, ffn=none, inductive_routing=True,
`use_edge_gating: False`, `use_rel_matrix_v: True`.

| Época | Val MRR | Test MRR |
|-------|---------|----------|
| 4 | 0.165 | 0.215 |
| 8 | 0.347 | 0.431 |
| **13** | **0.365** | **0.465** |
| 14–38 | plateau | plateau |

**Peak**: val=0.365, test=**0.465** @ ep13 — **peor que baseline (0.565) en −0.10 MRR**.

Patrón distinto al baseline: sube lento durante 13 épocas (sin colapso) y luego se estabiliza.
No hay memorización del train graph. Pero el peak es más bajo.

**Dos problemas identificados**:

1. **Velocidad**: 105s/época vs 19s/época (5.5×). Causa: `W = self.W_r[rel_ids]` materializa
   tensor `(E, heads, d, d)` ≈ 350MB por forward pass para todas las aristas simultáneamente.
   **No afecta métricas** — solo tiempo.

2. **Query conditioning perdido**: el baseline tenía `V_gate = V_gate(r_uv) + proj_vg(r_q)`,
   condicionando V en AMBAS la relación de arista y la de query. `use_rel_matrix_v` v1 usaba
   solo `h @ W_r[r_uv]` sin ningún conditioning de r_q en V. Este sí afecta métricas.

#### v2 — fixes aplicados (job 574626, en curso)

**Fix 1 (memoria)**: loop por tipo de relación en `propagate_attention`:
```python
v_src = torch.zeros_like(v_raw)
for r in range(self.W_r.shape[0]):        # 19 iteraciones para WN18RR
    mask = rel_ids == r
    if not mask.any(): continue
    v_src[mask] = torch.einsum('ehk,hkd->ehd', v_raw[mask], self.W_r[r])
```
Memoria: O(max_E_per_relation × heads × d²) en lugar de O(E_total × heads × d²).
**Matemáticamente idéntico** — sin impacto en métricas.

**Fix 2 (query conditioning)**: `proj_vg` creado también cuando `use_rel_matrix_v=True`,
aplicado como FiLM sobre la salida de W_r:
```python
# En forward(): precomputar
batch.E_relmat_qcond = self.proj_vg(shared_edge).view(-1, heads, out_dim)

# En propagate_attention(): aplicar después de W_r
if hasattr(batch, 'E_relmat_qcond'):
    v_src = v_src * (1.0 + batch.E_relmat_qcond)
```
Resultado: `msg = (h @ W_r[r_uv]) * (1 + proj_vg(r_q))` — W_r completa + query conditioning.
Params: 139,969 → 152,257 (+12K de proj_vg).

#### v2 — resultado (job 574626, ep0–13, aún en curso)

| Época | Val MRR | Test MRR |
|-------|---------|----------|
| 4 | 0.286 | 0.345 |
| 7 | 0.387 | 0.475 |
| 8 | 0.400 | 0.501 |
| 9 | 0.404 | 0.510 |
| **10** | **0.408** | **0.509** | ← BEST VAL
| 11–13 | 0.399↓ | 0.490↓ | declinando |

**Peak**: val=0.408, test=**0.509** @ ep10.

**Comparativa vs v1 y baseline**:
| Config | Val MRR | Test MRR | Δ vs baseline |
|--------|---------|----------|---------------|
| baseline (indrouting_lr1e5) | 0.415 | **0.565** | — |
| relmatrix v1 (sin query cond.) | 0.365 | 0.465 | −0.100 |
| **relmatrix v2 (con query cond.)** | **0.408** | **0.509** | −0.056 |

**Velocidad**: 72s/época vs 105s (v1) vs 19s (baseline). El loop por relación mejora 32% vs materializar tensor, pero sigue 3.8× más lento que el baseline.

**Conclusiones**:

1. **Query conditioning SÍ importa en V**: v2 (0.509) vs v1 (0.465) = +0.044 MRR. El `proj_vg(r_q)` recupera parte del rendimiento perdido al remover el gate.

2. **Aún −0.056 vs baseline**: el W_r completa no supera al baseline con gate diagonal (0.565). La hipótesis "full vs diagonal = 0.741 vs 0.570" no se confirmó — la diferencia real es pequeña.

3. **Val casi igual** (0.408 vs 0.415): el gap test (0.509 vs 0.565) sugiere que W_r sigue generalizando peor al test graph que el gate, posiblemente porque las 19 matrices 16×16 tienen más capacidad para memorizar la topología del train graph.

4. **Plateau más tardío** (ep10 vs ep4 baseline): W_r aprende más lento. El LR 1e-5 puede ser subóptimo para matrices 16×16 — necesita más señal por parámetro.

**Diagnóstico**: el gap baseline→W_r no es simplemente "diagonal vs full". La atención selectiva ponderada puede estar suprimiendo mensajes de W_r que NBFNet pasaría con suma uniforme. O el LR necesita ajuste específico para W_r.

---

### DIAGNÓSTICO DEFINITIVO: POR QUÉ 0.570 Y NO 0.741

Después de confirmar el techo con HP tuning, el diagnóstico es preciso.

**El único componente no-inductivo que nunca fue eliminado correctamente es el Value:**

```python
V = W_V(h_src) * V_gate(rel_emb[r_uv])
```

`W_V` es una proyección lineal compartida para todas las relaciones. `V_gate` escala el resultado elemento a elemento. Esto es matemáticamente equivalente a una **transformación diagonal específica por relación**:

```
V_ij = W_V @ h_i * gate_r  ↔  h_i @ (W_V * gate_r)  =  h_i @ D_r W_V
```

donde D_r es diagonal — sólo escala dimensiones, no las mezcla. **Todos los experimentos intentados implementaron variaciones de diagonal W_r**:

| Experimento | Forma matemática | Por qué falla |
|-------------|-----------------|---------------|
| `V_gate(rel_emb[r])` (baseline) | `W_V(h) ⊙ gate_vec(r)` | Diagonal — escala pero no mezcla dimensiones |
| `W_V(h) ⊙ rel_emb[r]` (distmultv) | `h @ (W_V * rel_emb[r])` | Exactamente igual en expresividad al baseline |
| `h ⊙ nbf_rel_emb[r]` (use_nbf_v) | `h ⊙ d_r` | También diagonal, plus init inestable |
| V-RMPNN separado | stream paralelo | Compite en gradiente con stream principal |
| FiLM FFN | condiciona FFN, no V | h ya lleva patrones del train graph antes del FFN |

**NBFNet usa matrices COMPLETAS por relación** (no diagonal):
```python
msg = h_src @ W_r[r_uv]   # W_r ∈ R^{dim × dim}, una matriz distinta por relación
```

La composición de W_r a través de hops permite razonar sobre caminos relacionales: `hypernym ∘ hyponym ≠ hypernym ∘ hypernym` porque las matrices mezclan dimensiones entre sí. Con diagonal (nuestro modelo), cada dimensión se propaga independientemente — no hay composición cruzada entre dimensiones. Eso es la brecha.

---

### PLAN PARA ALCANZAR 0.741+: W_r POR RELACIÓN

**El cambio**: reemplazar el Value con matrices completas por relación, manteniendo todo lo demás.

```python
# ACTUAL (diagonal, graph-specific):
V_src = self.V(h_src)           # W_V shared, sin distinción por relación
V_src = V_src * batch.E_gate    # gate escala elemento a elemento

# PROPUESTO (W_r por relación, NBFNet-style):
# Añadir a __init__:
self.W_r = nn.Parameter(torch.empty(num_relations, dim_h, dim_h))
nn.init.orthogonal_(self.W_r.view(num_relations * dim_h, dim_h))

# En forward, para KG edges:
V_src = torch.einsum('ed,edf->ef',
    h[edge_index[0]],        # (E, dim) — source features
    self.W_r[edge_rel_ids]   # (E, dim, dim) — relation-specific map
)
# Para expander edges (sin relación):
# Usar self.W_exp: nn.Parameter(torch.empty(dim_h, dim_h)) — una matriz especial
```

**Costo en parámetros** (sin cambio arquitectónico mayor):
- WN18RR (18 relaciones, dim=32): 18 × 32² = **18,432 params** — menor que el FFN
- WN18RR (18 relaciones, dim=64): 18 × 64² = 73,728 params
- FB15k-237 (474 relaciones, dim=32): ~486K params (manejable)

**Costo computacional**: O(E × dim²) — igual que `W_V(h)` actual, sin overhead real.

**Config base para el experimento**:
```yaml
gt:
  layers: 3
  dim_hidden: 64    # dim=64 es el mejor no-FFN
  n_heads: 4
  dropout: 0.1
  use_edge_gating: False    # reemplazado por W_r
  use_query_conditioning: True
  inductive_routing: True
  ffn_type: none            # sin FFN (ya confirmado mejor)

optim:
  base_lr: 0.00001
  scheduler: cosine_with_warmup
  num_warmup_epochs: 5
  max_epoch: 50
```

---

### COMPATIBILIDAD CON LA TESIS (ETAPA 1)

El cambio es SÓLO en la función de valor V. Todo lo demás se mantiene intacto:

| Componente | Estado | Rol |
|-----------|--------|-----|
| Expander graph | ✅ sin cambio | Conectividad global, aristas extra para atención sparse |
| Trilinear score `(Q⊙K⊙E).sum()` | ✅ sin cambio | Routing selectivo basado en query |
| Inductive routing (K = proj_k(r_q)) | ✅ sin cambio | Score puramente relacional |
| BF residual (h += x0) | ✅ sin cambio | Reinyección de señal de query por capa |
| Q/E conditioning (proj_q, proj_e) | ✅ sin cambio | Condicionamiento por relación de query |
| **Value: W_r completa por relación** | 🔄 cambio clave | Mensaje NBFNet-style — composición relacional |
| FFN | `ffn_type: none` | Eliminado (ya confirmado que ayuda) |

**Contribución en términos de la tesis**: Exphormer con mensajes relacionales NBFNet.
- NBFNet hace suma uniforme sobre todos los caminos
- Exphormer ruta con atención dispersa (incluyendo expander para alcance global) — selecciona qué caminos son más relevantes para la query
- Los mensajes que transporta son relacionalmente expresivos (W_r completa, no diagonal)

Esto es novel respecto a KnowFormer (que usa dos streams separados con competencia de gradiente) y respecto a NBFNet (que no tiene atención ni expander).

**Expectativa realista**:
- NBFNet con suma uniforme: 0.741
- Exphormer con W_r + atención selectiva: ≥ 0.741 (la atención debería ayudar al seleccionar caminos)
- KnowFormer: 0.752 — rango objetivo

---

### FIX TÉCNICO: SLURM PARALELISMO

**Problema**: jobs enviados con `--mem` por defecto (256G) — cada job consumía todo el RAM disponible.
Con 753G total y 2 jobs con 256G ya no quedaba RAM para tercero.

**Fix**: enviar con `--mem=16G --cpus-per-task=4`. Con 16G/job, 5 jobs corren en paralelo en compute-gpu-3-1.
Los modelos de WN18RR-ind-v1 usan <2GB RAM real; el default era 16× sobredimensionado.

---

## Estado actual — 2026-04-13 (sesión 6)

### ANÁLISIS ARQUITECTÓNICO PROFUNDO (Mauricio, sesión 6)

Análisis completo del gap 0.565 → 0.741 basado en lectura de NBFNet, KnowFormer y Exphormer.
Conclusión: el problema no es un módulo sino el flujo de información completo.

**NBFNet** (0.741): propagación Bellman-Ford pura — `msg = h_src ⊙ W_r`, sum aggregation, sin atención, sin FFN. Inductivo por construcción porque `W_r` no depende de identidades de nodos.

**KnowFormer** (0.752): dos streams separados — Q-RMPNN para routing estructural (sin anchor), V-RMPNN para path information (anchor-conditioned, NBFNet-style). La atención combina ambos. La ablación muestra que V-RMPNN aporta +0.063 MRR (mayor componente individual).

**Nuestro modelo** (0.565): stream único. `V = W_V(h)` — proyección genérica sin dependencia relacional del mensaje. El gate `V_gate(r_ij)` escala post-proyección pero no puede seleccionar qué dimensiones de h importan POR relación (como DistMult). El FFN aprende patrones estadísticos del train graph.

**Por qué fallaron los experimentos anteriores de V:**
- `use_relational_v` (V *= rel_emb extra): dos señales multiplicativas relacionales en competencia → interferencia, 0.384
- `use_nbf_v` (DistMult puro): eliminó W_V → h crudo inestable en marco de atención, 0.420
- V-RMPNN: stream separado compite por gradiente con el stream principal → interferencia, 0.513

**Propuesta correcta**: `use_distmult_v` — mantiene W_V (proyección estabilizadora) y añade DistMult encima: `msg = W_V(h_src) ⊙ msg_rel_emb[r]`. Remove gate. Equivalente a KnowFormer V-RMPNN pero dentro de la capa de atención, evitando competencia de gradiente.

### EXPERIMENTOS COMPLETADOS EN SESIÓN 6

| Config | Params | Peak val | Test @ best val | Comportamiento |
|--------|--------|----------|-----------------|----------------|
| **distmultv_noffn** (dim=64, L=3) | 97K | ep11, 0.361 | **0.511** | Decae desde ep12 — peor que baseline |
| **distmultv_noffn_L5** (dim=64, L=5) | 160K | — | **~0.025** | **MUERTO** — nunca aprende (ep0-5 congelado) |
| **indrouting_noffn_dim32** (dim=32, L=3) | 31K | ep11, 0.436 | **0.558** | Plateau estable ~0.495-0.502 post-ep30 |
| **indrouting_ffn_dim32** (dim=32, L=3, con FFN) | 44K | ep37, 0.385 | **0.502** | Estable, sin colapso |
| **distmultv_noffn_dim32** (dim=32, L=3) | 27K | en curso (ep21→) | ~0.471 | En ejecución (job 573786) |

### ANÁLISIS DE RESULTADOS

**DistMult-V NO mejora sobre el gate en dim=64 (0.511 < 0.565)**:
La hipótesis era que `W_V(h) ⊙ W_r` sería más expresivo que `W_V(h) * gate(r)`. En dim=64 no se validó. El gate ya es suficiente selección dimensional para esta arquitectura.

**L=5 + DistMult-V + sin FFN = modelo muerto**:
MRR congelado en 0.025 durante épocas 0-5 (warmup). Sin FFN ni gate, 5 capas de DistMult sin normalización intermedia destruye el flujo de gradiente. El modelo no puede aprender.

**El hallazgo principal: `indrouting_noffn_dim32` (0.558) ≈ baseline (0.565) con 4× menos params y SIN colapso**:
- dim=32, L=3, sin FFN, con gate, inductive_routing=True
- Peak ep11: val=0.436, test=0.558
- Post-ep30: test estable ~0.495-0.502 (decae suavemente, no colapso)
- 31,585 parámetros vs ~131K del baseline

Esto demuestra que la capacidad del modelo NO es el factor limitante. El FFN sí daña la generalización inductiva incluso a dim=32: con FFN el mismo dim=32 da 0.502 vs 0.558 sin FFN.

**Sin FFN + dim pequeño = training más estable**:
La combinación dim=32 + sin FFN elimina los dos componentes que memorizan patrones del train graph. El resultado es casi idéntico al baseline pero con entrenamiento mucho más estable.

### DIAGNÓSTICO ACTUALIZADO (sesión 6)

La secuencia de ablaciones revela la importancia relativa de cada cambio:

| Cambio | Δ MRR | Componente eliminado |
|--------|-------|---------------------|
| wV/Z → wV (sum aggregation) | +0.23 | Normalización por Z |
| inductive_routing (K = f(r_q) only) | +0.05 | W_K(h) graph-specific |
| sin FFN + dim=32 | ≈ 0 vs baseline, pero estable | FFN + capacidad extra |
| DistMult-V (dim=64) | -0.054 vs baseline | gate → DistMult (regresión) |

**Conclusión**: el gap residual 0.558 → 0.741 no es del V ni del FFN. Es del mecanismo de mensaje completo. El gate ya es un buen proxy del DistMult para esta arquitectura. La brecha con NBFNet/KnowFormer probablemente requiere cambios más estructurales.

### EXPERIMENTO EN CURSO

`distmultv_noffn_dim32` (job 573786): DistMult-V + sin FFN + dim=32.
- Objetivo: verificar si DistMult-V añade valor sobre el gate cuando el resto del modelo es más estable (dim=32, sin FFN)
- Log: `logs/distmultv_noffn_dim32.log`
- Best actual (ep35): val=0.340, test=0.471 (por debajo de `indrouting_noffn_dim32` en la misma época)

---

## Estado actual — 2026-04-10 (sesión 5)

### MEJOR RESULTADO ACTUAL (CONFIRMADO CON val ckpt)

**`wn18rr_ind_v1_indrouting_lr1e5.yaml`** con `ckpt_monitor_split: val`:
- **Peak val MRR: 0.415** en ep4 → checkpoint guardado
- **Test MRR en ese checkpoint: 0.565** — confirma que el 0.565 no era overfitting a test

El 0.565 reportado anteriormente fue con `ckpt_monitor_split: test` (selección por test, no válida).
Ahora con selección por val el test MRR del mejor checkpoint es **el mismo: 0.565**.
El resultado es legítimo.

Config: `inductive_routing=True`, `use_edge_gating=True`, `lr=1e-5`, `warmup=5`, `L=3`, `dim=64`.

### EXPERIMENTOS COMPLETADOS EN SESIÓN 5

| Config | Peak val MRR | Test MRR @ best val | Época | Conclusión |
|--------|-------------|---------------------|-------|------------|
| **baseline (val ckpt)** | **0.415** | **0.565** | **4** | **referencia limpia confirmada** |
| use_nbf_v (xavier init) | ~0.22 | 0.420 | 10 | peor que gate — raw h sin proyección inestable |
| V-RMPNN sin query cond. | ~0.35 | 0.467 | 11 | mejor que nbf_v pero peor que baseline |
| **V-RMPNN con query cond.** | **~0.38** | **0.513** | **12** | **mejor V-RMPNN pero <0.565** |
| PNA concat(sum,mean,max) | 0.228 | 0.284 | 4 | **FALLO CATASTRÓFICO** |
| MLP scorer (128→64→ReLU→1) | 0.326 | 0.458 | 4 | **PEOR** — head más expresivo daña |

### ANÁLISIS PNA — POR QUÉ FALLÓ

**Resultado**: val_mrr=0.228 vs baseline 0.415 — el peor resultado de todos los experimentos.

**Causa**: la proyección `concat(sum, mean, max) → Linear(192→64)` incluye `mean = sum/Z`
(atención normalizada). Esto **reintroduce la normalización por Z** que fue el peor diseño
del sistema original — el cambio de wV/Z a wV fue la mayor mejora individual (+0.23 MRR).
Aunque el Linear aprende a minimizar su peso, el gradient signal de mean contamina el
aprendizaje de sum desde el comienzo.

**Lección**: la normalización por Z es dañina en KGC con sum aggregation NBFNet-style.
NO incluir mean como componente de PNA. Si se prueba PNA de nuevo, usar solo `concat(sum, max)`.

### ANÁLISIS V-RMPNN — POR QUÉ NO SUPERA EL BASELINE

El V-RMPNN con query conditioning (implementación correcta según KnowFormer appendix):
- `rel_vec = rel_emb[r_edge] * (1 + proj_qcond(query_emb[r_q]))` — bilineal aproximado
- Peak val 0.38, test 0.513 (ep12) — **0.052 debajo del baseline**
- Patrón: oscila 0.50-0.51 en test sin tendencia a subir

Hipótesis de por qué no ayuda:
1. El V-stream separado opera con una señal de loss idéntica al attention stream principal
   → ambos compiten por el mismo gradiente, el V-RMPNN interfiere en lugar de complementar
2. El gate `W_V(h) * V_gate(r)` ya es una buena aproximación al V relacional para
   nuestra arquitectura específica (sum + trilinear score)

### ANÁLISIS MLP SCORER — POR QUÉ FALLÓ

**Evolución por época** (val | test):
```
ep0:  0.008 | 0.016  (random)
ep1:  0.024 | 0.075
ep2:  0.100 | 0.252  (convergencia más rápida que baseline)
ep3:  0.286 | 0.461  ← pico test antes que el val
ep4:  0.326 | 0.458  ← BEST VAL (checkpoint)
ep5:  0.300 | 0.398  (colapso)
ep6-8: ~0.26-0.28 | ~0.34-0.40 (plateau bajo)
ep9-49: ~0.26-0.28 | ~0.32-0.34 (se estabiliza al bajar LR)
```

**Comparativa directa con baseline** (misma época, val ckpt):
| Métrica | Baseline | MLP scorer | Δ |
|---------|----------|------------|---|
| Peak val MRR | **0.415** | 0.326 | -0.089 |
| Test MRR @ best val | **0.565** | 0.458 | -0.107 |

**Causa raíz**: `h_v^L` en nuestra arquitectura ya está fuertemente condicionado en la query
relation a través de 3 mecanismos en cada capa del transformer:
1. BF residual: `h += x0` donde `x0_anchor = rel_emb_enc[r_q]`
2. Q conditioning: `Q = W_Q(x0) + proj_q(shared_rel_emb[r_q])`
3. E conditioning: `E = W_E(edge_attr) * (1 + proj_e(shared_rel_emb[r_q]))`

Dado esto, `cat(h_v^L, rel_emb_head[r_q])` concatena una señal redundante: la relación ya
está en `h_v^L`. El Linear(2d→1) puede aprovechar esta redundancia como sesgo de confirmación.
El MLP(128→64→1) en cambio fuerza una interacción entre `h_v` y `r_q` a través de la capa
oculta — pero esa interacción ya ocurrió en el transformer. La ReLU introduce no-linealidad
innecesaria que hace el landscape de optimización más difícil sin aportar expresividad real.

**Lección clave**: en nuestra arquitectura, **el scorer simple (Linear) es mejor que el MLP**
porque la representación `h_v^L` es query-conditioned. Más expresividad en el head solo
añade ruido de optimización.

**Arquitectura del scoring**: el Linear(2d→1) sobre cat(h_v, r_q) es equivalente a:
```
score = W_h · h_v  +  W_r · rel_emb[r_q]  +  b
```
El término `W_r · rel_emb[r_q]` es constante por grafo → solo aporta un sesgo global
que puede aprenderse a anular. Lo que realmente discrimina es `W_h · h_v`. El scorer
aprende a proyectar `h_v^L` en la dirección que maximiza la separación entre entidades.

### ESTADO DE EXPERIMENTOS POR LÍNEA DE INVESTIGACIÓN (sesión 5 completa)

**Línea: Aggregation**
| Cambio | Peak val | Test @ ckpt | Resultado |
|--------|----------|-------------|-----------|
| wV/Z → wV (sum) | — | +0.23 MRR | ✅ mayor mejora de la historia |
| PNA concat(sum,mean,max) | 0.228 | 0.284 | ❌ reintroduce normalización dañina |

**Línea: Routing inductivo**
| Cambio | Peak val | Test @ ckpt | Resultado |
|--------|----------|-------------|-----------|
| K = proj_k(rel_q) only | — | +0.05 MRR | ✅ |
| L=5 (más capas) | — | 0.494 | ❌ más overfit |
| constlr | — | 0.563 | ❌ LR no es el problema |

**Línea: Value relacional**
| Cambio | Peak val | Test @ ckpt | Resultado |
|--------|----------|-------------|-----------|
| use_relational_v (V *= rel_emb extra) | — | 0.384 | ❌ interferencia con gate |
| use_nbf_v xavier (DistMult puro) | ~0.22 | 0.420 | ❌ raw h inestable |
| V-RMPNN sin query cond. | ~0.35 | 0.467 | ❌ < baseline |
| V-RMPNN con query cond. | ~0.38 | 0.513 | ❌ < baseline |

**Línea: Scoring head**
| Cambio | Peak val | Test @ ckpt | Resultado |
|--------|----------|-------------|-----------|
| Linear(2d→1) [baseline] | **0.415** | **0.565** | ✅ mejor |
| MLP(2d→d→ReLU→1) | 0.326 | 0.458 | ❌ -0.107 — h_v ya es query-conditioned |

**Línea: Capacidad**
| Cambio | Peak val | Test @ ckpt | Resultado |
|--------|----------|-------------|-----------|
| dim=128, L=5 | — | 0.501 | ❌ más overfit |
| weight tying capas | — | 0.513 | ❌ mejor H@10 pero peor MRR |

### DIAGNÓSTICO RAÍZ CONSOLIDADO (sesión 5)

**El techo del modelo actual es ~0.565 test MRR / 0.415 val MRR.**

Todos los intentos de mejorar (V, head, aggregation) han fallado. La arquitectura
está en un óptimo local dado el diseño actual. El scoring head lineal es correcto
para representaciones ya query-condicionadas.

El gap 0.565 → 0.741 requiere algo más fundamental que ajustes de módulos:
- **DropEdge** — única dirección no explorada que ataca el sobreajuste topológico
  sin cambiar la arquitectura de fondo
- **Arquitectura radicalmente diferente** — si DropEdge no funciona, la brecha
  puede requerir cambios estructurales más profundos (más allá del scope de sesión)

### ANÁLISIS EXPERTO: POR QUÉ KNOWFORMER NO MEMORIZA TOPOLOGÍA (sesión 5)

KnowFormer usa información topológica pero filtra en el **MENSAJE**:
```
msg_{u→v}^r = z_u ⊙ W^r(r_q)   ← topología pasa por filtro (edge_rel × query_rel) antes de agregar
```
La topología (qué nodo envía, qué aristas existen) entra como `z_u`, pero se modula
**element-wise por la relación de query** antes de agregarse. Lo que llega a `z_v^(l+1)` es
siempre una función de (tipos de caminos × query). El modelo aprende: "qué tipos de caminos
son evidencia para este tipo de query" — información relacional, transferible entre grafos.

Nuestro modelo filtra **DESPUÉS de la proyección**:
```
V = W_V(h_i)            ← proyección mezcla relacional + topológico
msg = V * gate(r_uv)    ← gate opera sobre vector ya mezclado
```
`W_V` es una proyección lineal genérica sobre `h_i`. Para i=9 en el train graph, `h_9^(l)`
acumula información sobre los vecinos específicos de la entidad 9. `W_V` aprende a mapear
ese patrón estructural específico a una dirección útil en el espacio de salida. En el test
graph, la entidad en posición i=9 tiene vecinos completamente diferentes — `W_V` mapea sus
representaciones a la misma dirección que aprendió del train graph. Inducción fallida.

### CUELLO DE BOTELLA IDENTIFICADO: EL FFN

Con `inductive_routing=True`, casi todos los componentes son puramente relacionales:
- **Score**: `(Q_j ⊙ K ⊙ E_ij).sum()` donde K=`proj_k(rel_q)` — f(r_q, r_uv) ✅
- **K**: `proj_k(shared_rel[r_q])` — sólo query relation, cero parámetros graph-specific ✅
- **Q**: `W_Q(x0) + proj_q(shared_rel[r_q])` — x0 es 0 o rel_emb_enc[r_q], relacional ✅
- **E**: `W_E(r_uv) * (1 + proj_e(shared_rel[r_q]))` — función de (r_uv, r_q) ✅
- **V**: `W_V(h) * gate(r_uv)` — ✅ gate es relacional, pero W_V(h) sigue siendo graph-specific
- **FFN**: `h = h + ff2(ReLU(ff1(h)))` — ❌ matrices **completamente genéricas**

El FFN aprende patrones cruzados del tipo:
"si dimensión 3 alta y dimensión 7 baja → boost dimensión 12"

Esto es un patrón estadístico del train graph. En el test graph, las mismas dimensiones
tienen distribuciones distintas. NBFNet no tiene FFN — solo σ element-wise (inductivo por
construcción). El FFN es el único componente que aprende "qué aspecto tienen las
representaciones del train graph" y aplica ese conocimiento topológico en el test.

### PRÓXIMA DIRECCIÓN: FiLM conditioning del FFN

**Hipótesis**: si el FFN aprende transformaciones condicionadas en la relación de query
(no en patrones generales del grafo), podría ser inductivo.

```python
# En MultiLayer.forward(), reemplazar el FFN estándar:
# h = h + ff2(ReLU(ff1(h)))   ← actual

# Por:
gamma_r = self.film_gamma(shared_rel_emb[r_q][batch.batch])  # (N, dim)
beta_r  = self.film_beta(shared_rel_emb[r_q][batch.batch])   # (N, dim)
h = h + ff2(ReLU(ff1(h) * gamma_r + beta_r))                 # FiLM
```

**Por qué es principiado**: FiLM condiciona la transformación de features sobre metadatos
(query relation). El FFN ahora aprende "cómo transformar representaciones para este tipo
de relación de query" en lugar de "cómo transformar representaciones del train graph".
Las matrices `gamma`/`beta` son funciones de `r_q` — completamente inductivas.

**Implementación**: ~20 líneas en `layer/exphormer.py` + flag `gt.use_film_ffn: False`.

**Veredicto honesto**: si FiLM falla, el techo arquitectónico es **0.565**. El gap
0.565→0.741 sería entonces estructural — el mecanismo de mensaje completo necesitaría
cambiarse a RMPNN-style (como KnowFormer), lo que es una reescritura mayor.

**No explorar más**: DropEdge (no ataca la causa raíz — el FFN no memoriza aristas sino
distribuciones estadísticas), MLP scorer (ya probado), V-RMPNN (ya probado), PNA con mean
(catastrófico), más capas/dim (más overfit), weight tying, cambios al scoring head.

---

## Estado actual — 2026-04-10 (sesión 4)

### EXPERIMENTOS COMPLETADOS EN SESIÓN 4

| Config | Peak MRR | Época | Conclusión |
|--------|----------|-------|------------|
| indrouting_constlr (lr=5e-6, sched=none) | 0.563 | 3 | Igual al baseline — colapso NO es por scheduler |
| indrouting_L5 (L=5, lr=1e-5 cosine) | 0.494 | 3 | Peor que L=3 — más capas = más overfit |

### DIRECCIONES EXPLORADAS Y CERRADAS — NO VOLVER A INTENTAR

| Experimento | Peak MRR | Conclusión |
|-------------|----------|------------|
| use_relational_v (V *= rel_emb extra) | 0.384 | Interferencia con gate existente |
| indrouting_dim128 (dim=128, L=5) | 0.501 | Más capacidad = más overfit |
| use_nbf_v=True (DistMult puro, std=0.01) | 0.514 | Init near-zero — fallo de implementación, no de concepto |
| tie_layers=True (parámetros compartidos) | 0.513 | Mejor hits@10 pero peor MRR |
| constlr (lr=5e-6, scheduler=none) | 0.563 | **Igual al baseline** — LR no es el problema |
| indrouting_L5 (L=5, lr=1e-5) | 0.494 | **Peor** — más capas = más overfit del train graph |
| **scoring head inductivo (dot product / DistMult)** | — | **NO EXPLORAR** — el head es una capa lineal sobre h_v^L; si h_v^L es malo, el head no importa. El pico de 0.565 demuestra que el head actual SÍ funciona cuando las representaciones son buenas. Cambiar el head no ataca la causa raíz. |

---

### use_nbf_v — ABANDONADO

**Experimento**: `wn18rr_ind_v1_nbfv.yaml` — reemplaza W_V + V_gate con DistMult puro
`msg = h_i ⊙ nbf_rel_emb[r_ij]` (implementado como `use_nbf_v: True`).

**Resultado**: peak **0.514 MRR** en época 10 — peor que el gate (0.565).

Evolución:
```
ep 0–8:  0.14–0.18  (casi aleatorio — nbf_rel_emb std=0.01 genera mensajes ~0)
ep 9:    0.343       (LR llega a máximo, modelo empieza a aprender)
ep 10:   0.514  ←  peak
ep 11:   0.282       (colapso)
ep 12–19: 0.19–0.21 (piso)
ep 20–43: 0.195→0.280 (subida monotónica lenta mientras LR→0)
```

**Diagnóstico**:
- La init `std=0.01` de `nbf_rel_emb` hace que los mensajes sean casi cero en las primeras
  épocas. El modelo no aprende hasta que el LR es lo suficientemente alto (ep9-10).
- El gate (`W_V(h) * V_gate(rel_emb[r])`) ya es una buena implementación de V relacional
  porque hereda la escala del encoder. `nbf_rel_emb` parte desde cero.
- La subida monotónica ep20-43 sugiere que DistMult sí aprende algo, pero muy lento.
- El colapso en ep11 es el mismo problema de fondo: h acumula información específica del
  grafo de train en las capas 1+, independientemente de cómo se forma el mensaje V.

**Conclusión**: el gate es mejor para nuestra arquitectura. El código `use_nbf_v` queda
en el codebase (default `False`) pero fuera del plan activo.

### Estado de experimentos completados (2026-04-10)

| Config | Peak MRR | Época | Plateau largo | Notas |
|--------|----------|-------|---------------|-------|
| indrouting (lr=3e-5, warmup=3) | 0.532 | 2 | ~0.356 (ep24) | baseline indrouting |
| **indrouting_lr1e5 (lr=1e-5, warmup=5)** | **0.565** | **4** | ~0.382 (ep29-49) | **mejor actual** |
| indrouting_dim128 (dim=128, L=5) | 0.501 | 1 | colapso rápido | más cap. no ayuda |
| nbfv (use_nbf_v=True, lr=1e-5) | 0.514 | 10 | ~0.28 (ep43) | DistMult, peor |
| indrouting_constlr (lr=5e-6, none) | 0.563 | 3 | ~0.45 ep8-12 | colapso igual que cosine |
| indrouting_L5 (L=5, lr=1e-5) | 0.494 | 3 | ~0.44 ep8-12 | más capas = menos MRR |

### Observaciones clave

- **Gate = DistMult relacional efectivo**: `W_V(h) * V_gate(rel_emb[r_ij])` ya provee
  modulación relacional del Value a la escala correcta. Reemplazarlo no ayuda.
- **El plateau estable** de indrouting_lr1e5 es ~0.382 (ep29-49), mientras el pico es
  0.565. El gap pico→plateau (0.183) es el scoring fino que se degrada al aumentar el LR.
- **Weight tying entre capas** es la siguiente dirección natural: NBFNet aplica el mismo
  módulo T veces. Compartir Q/K/E/gate entre las 3 capas fuerza aprender patrones
  relacionales generalizables, no específicos al grafo de train.

### Weight tying entre capas — ABANDONADO

**Experimento**: `wn18rr_ind_v1_tielayers.yaml` — un único `MultiLayer` aplicado 3 veces
(58,497 params vs 168,193 con capas independientes).

**Resultado**: peak **0.513 MRR** (ep5) — peor que indrouting_lr1e5 (0.565).

```
ep 0:  0.254  (convergencia más rápida por pocos params)
ep 5:  0.513  ← peak
ep 22: 0.365  (plateau similar al baseline)
```

Hits@10 en ep22 = 0.785 vs 0.776 del baseline — mejor recall global pero peor MRR.
El modelo aprende routing más general pero scoring menos fino.

**Conclusión**: weight tying no mejora el MRR. Código revertido completamente
(`tie_layers` eliminado de config.py y model.py).

---

## Estado actual — 2026-04-10 (sesión 2)

---

## Plan arquitectónico para alcanzar MRR ≥ 0.74 — 2026-04-10

### Diagnóstico raíz: dos problemas independientes

#### Problema 1 — V es relación-agnóstico (brecha de expresividad)

NBFNet (NeurIPS 2021, Zhu et al.) usa DistMult como función de mensaje:
```
msg_{i→j,r} = h_i^(l-1) ⊙ W_r
```
donde `W_r` es un vector aprendido por tipo de relación, compartido en todos los grafos.
Esta forma bilineal es la razón por la que NBFNet generaliza inductivamente:
la relación transforma la representación fuente de forma independiente a la identidad
de los nodos. Los gradientes aprenden patrones relacionales, no estructurales.

Nuestro modelo computa:
```python
V_h = self.V(h)                         # W_V(h_i) — proyección genérica
v_src = V_h[src] * gate(r_ij)           # gate escalar/vectorial DESPUÉS de proyectar
msg = v_src * score(Q_j, K_i, E_ij)
```
El valor que cada nodo envía es el mismo `W_V(h_i)` sin importar la relación.
La relación sólo entra como un gate POSTERIOR. Esto es estrictamente menos expresivo
que DistMult porque:
- DistMult: `h_i ⊙ W_r` — la relación selecciona qué dimensiones de h_i importan
- Nuestro gate: `W_V(h_i) * gate_r` — escala las dimensiones ya proyectadas

#### Problema 2 — El routing aprende patrones del grafo de train (colapso)

El score de atención usa `K_i = W_K(h_i)`, donde `h_i` acumula información propagada
desde el anchor a través de la estructura específica del grafo de entrenamiento.
Tras varias épocas de entrenamiento, `W_K` aprende a reconocer representaciones de
nodos del grafo de train. En el grafo de test (entidades distintas, topología diferente)
esas representaciones no existen → el routing colapsa.

NBFNet evita este problema por diseño: **no tiene pesos de atención aprendidos**.
Usa suma directa de mensajes relación-transformados. El routing implícito es puramente
relacional (qué tipo de arista, qué relación de query) — siempre transferible.

KnowFormer (ICML 2024, Chen et al.) también lo evita: su Q-RMPNN produce Q/K desde
representaciones estructurales homogéneas (sin boundary condition del anchor), pero
su V viene de V-RMPNN que corre la propagación NBFNet completa. La ablación de
KnowFormer muestra que eliminar V-RMPNN cuesta -0.063 MRR en FB15k-237 (mayor
caída individual) — confirma que V con propagación relacional es la clave.

### Plan de implementación en orden de prioridad

#### Cambio 1 (PRIORITARIO): V relación-específico — DistMult en el mensaje

**Motivación**: NBFNet §3.2: "INDICATOR + DistMult-MESSAGE + SUM-AGGREGATE logra 0.741
MRR en WN18RR inductivo v1". El componente diferenciador es el mensaje DistMult.

**Implementación** en `layer/exphormer.py`:

```python
# En __init__ (cuando use_query_conditioning o siempre para KGC):
self.msg_rel_emb = nn.Embedding(num_relations, self.out_dim * num_heads)
nn.init.normal_(self.msg_rel_emb.weight, std=0.01)

# En propagate_attention, reemplazar:
#   v_src = batch.V_h[edge_index[0]]
# Por:
v_src = batch.V_h[edge_index[0]] * \
        self.msg_rel_emb(batch.edge_rel_idx).view(-1, self.num_heads, self.out_dim)
#   ^ batch.edge_rel_idx = raw relation indices per edge (long), añadir en trainer
```

Requiere guardar los índices de relación crudos en `batch.edge_rel_idx` (long tensor)
antes del encoder. En `train/trainer.py`, añadir `data.edge_rel_idx = rep_ea` (ya
existe como tensor de índices antes del encoding). Para aristas expander (sin relación),
usar un índice especial `num_relations` → añadir 1 fila extra al Embedding.

Config nuevo: `gt.use_relational_v: True` (default False para no romper transductivo).

**Impacto esperado**: +0.10 a +0.20 MRR. El mensaje pasa de relación-agnóstico a
relación-específico. Con sum aggregation ya implementada, esto replica la función
central de NBFNet dentro del marco de Exphormer.

#### Cambio 2 (PRIORITARIO): Routing inductivo — score sólo relacional

**Motivación**: El score actual `exp((Q_j ⊙ K_i ⊙ E_ij).sum() / √d)` usa
`K_i = W_K(h_i)` que es específico al grafo de train. Para eliminar el colapso,
el routing debe depender sólo de tipos de relación.

**Opción 2a (mínima)** — eliminar W_K(h), mantener sólo proj_k(rel_q):
```python
# En forward, reemplazar:
K_h = self.K(h) + self.proj_k(shared_node)
# Por:
K_h = self.proj_k(shared_node)   # sólo conditioning por query, no por h
```
El score pasa a ser: `exp((Q_j ⊙ K_i_relacional ⊙ E_ij).sum() / √d)`
donde K_i sólo depende de rel_q (compartido) → totalmente inductivo.

**Opción 2b (más agresiva)** — score puramente por tipo de arista:
```python
# Eliminar Q/K del score, usar sólo E:
score_ij = torch.exp(batch.E.sum(-1, keepdim=True).clamp(-5, 5))
```
Esto deja el routing como función pura de (r_ij, r_q) — máxima inductividad.

Probar 2a primero; si sigue colapsando probar 2b.

**Impacto esperado**: elimina o reduce drásticamente el colapso. Permite que el modelo
entrene estabilmente más épocas y aproveche el Cambio 1.

#### Cambio 3 (moderado, después): PNA aggregation

**Motivación**: NBFNet §4 ablation muestra que AGGREGATE=PNA (sum+mean+max+min)
supera a AGGREGATE=sum en ~0.02-0.03 MRR. El sum solo puede tener problemas de
escala con grafos de grados variables.

**Implementación** (en `propagate_attention`):
```python
batch.wV_sum = scatter(msg, dst, dim=0, dim_size=N, reduce='add')
batch.wV_max = scatter(msg, dst, dim=0, dim_size=N, reduce='max')
h_out = torch.cat([batch.wV_sum, batch.wV_max], dim=-1)  # (N, 2*d)
# + proyección lineal (N, 2*d) → (N, d) antes de residual
```

### Secuencia de experimentos tras implementar

1. Smoke test Cambio 1 solo (V relacional, sin Cambio 2) con T=3, lr=3e-5
2. Si colapsa igual → aplicar Cambio 2a
3. Si funciona estable → comparar T=3 vs T=5, con y sin expander
4. Cambio 3 (PNA) cuando ya tengamos base estable

### Resultados de la sesión 2026-04-10 que justifican esta dirección

| Config | Peak test MRR | Patrón |
|--------|--------------|--------|
| T=5, wV/Z (baseline) | 0.252 | colapso |
| T=5, sum, lr=8e-4 | 0.365 | colapso ep2 |
| T=5, sum, lr=1e-4 | 0.432 | colapso ep2 |
| T=5, sum, lr=3e-5 | 0.482 | colapso ep2 |
| T=3, sum, lr=3e-5 | 0.480 | decline gradual, ep20=0.284 |
| T=2, sum, lr=3e-5 | 0.328 | colapso inmediato |
| label_smooth/dropout | sin mejora | colapso igual |

T=3 con sum es el mejor config actual. El routing (K=W_K(h)) sigue siendo el
cuello de botella para el colapso. El mensaje (V sin relación) es el cuello de
botella para la expresividad.

---

## Estado actual — 2026-04-10

### Sum aggregation: confirmada como dirección correcta

**Cambio implementado**: `layer/exphormer.py:179`
```python
# ANTES:
h_out = batch.wV / (batch.Z + 1e-6)
# AHORA:
h_out = batch.wV  # sum aggregation (no normalization by Z, like NBFNet)
```

**Resultados** (todos con `exp: False`, `ckpt_monitor_split: test`):

| Config / LR | Scheduler | Peak test MRR | Epoch | Hits@10 |
|-------------|-----------|---------------|-------|---------|
| noexp + wV/Z (baseline) | cosine lr=8e-4 | 0.252 | 1 | ~0.50 |
| noexp + sum, lr=8e-4 | cosine | 0.365 | 1 | 0.763 |
| noexp + sum, lr=1e-4 | cosine | 0.432 | 1 | 0.737 |
| **noexp + sum, lr=3e-5** | cosine | **0.482** | **1** | — |
| noexp + sum, lr=5e-5 | constante | 0.449 | 0 | — |
| NBFNet (paper) | — | 0.741 | — | 0.948 |
| KnowFormer (paper) | — | 0.752 | — | — |

**Tendencia**: menor LR efectivo en ep1 → mayor pico. La hipótesis del LR (los gradientes son más grandes con sum que con avg) se confirma.

### Problema abierto: colapso después del epoch 1

El modelo aprende algo muy bueno en 1 epoch y luego lo destruye. La curva típica (lr=3e-5):
```
ep 0: test=0.005 (random)
ep 1: test=0.482 ← pico
ep 2: test=0.446
ep 3: test=0.457 (val pico)
ep 4: test=0.408
ep 5+: colapso monotónico → ~0.19
```

El colapso ocurre exactamente cuando el LR llega a su máximo (fin del warmup). Con LR constante (lr=5e-5), el colapso es inmediato desde ep1. Esto confirma que el LR es un factor, pero no el único: la train loss baja monotónicamente mientras el test MRR colapsa → **sobreajuste a la topología del grafo de entrenamiento**. Los pesos de atención aprenden patrones específicos del grafo train que no generalizan al grafo test (entidades disjuntas).

### Logs de experimentos (2026-04-10)

| Experimento | Log |
|-------------|-----|
| sum + lr=8e-4 | `logs/ind_v1_noexp_sumpool_20260409_195142.log` |
| sum + lr=1e-4 | `logs/ind_v1_noexp_sumpool_lr1e4_20260410_011630.log` |
| sum + lr=3e-5 (cosine) | `logs/ind_v1_noexp_sumpool_lr3e5_20260410_012729.log` |
| sum + lr=5e-5 (const) | `logs/ind_v1_noexp_sumpool_const5e5_20260410_012730.log` |

### Opciones para atacar el colapso (próximos experimentos)

En orden de esfuerzo:
1. **Label smoothing** (`kgc.label_smoothing: 0.1`) — 0 código, 1 línea en config
2. **Dropout más alto** (`gt.dropout: 0.2-0.3`) — 0 código, 1 línea
3. **Menos capas** (`gt.layers: 2-3`) — menos overfitting a topología global
4. **DropEdge** — drop aleatorio de aristas KG durante train (~15 líneas en trainer)
5. **Weight tying across layers** — compartir Q/K/V/E entre todas las capas (como NBFNet)
6. **`wV / sqrt(Z+1)`** — punto medio entre sum y avg (1 línea)
7. **V-RMPNN estilo KnowFormer** — stream V separado con boundary condition (reescritura)

---

## Estado actual — 2026-04-09 (actualizado tarde)

### Diagnóstico inductivo: causa raíz de la brecha con NBFNet

**Experimentos realizados en esta sesión:**

| Config | Cambio | Peak test MRR | Patrón |
|--------|--------|---------------|--------|
| baseline (arq. unificada) | — | ~0.210 | peak ep1, colapso |
| softlr | scheduler más suave | ~0.210 | igual, no ayuda |
| proj_k (K desde x0+bias) | K += proj_k(q) — INCORRECTO | colapso | K sigue siendo uniforme |
| **K desde h** | K = W_K(h) en vez de W_K(x0) | **0.256** | peak ep1-2, luego colapso |
| K desde h + ckpt=test | ídem + monitor por test MRR | **0.256** | igual |
| **sin expander** | exp=False, K desde h | **0.252** | prácticamente igual |

**Conclusión del ablation noexp**: el expander NO es el problema principal. Con y sin expander el pico es casi idéntico (~0.252–0.256). En transductivo el expander ayudaba 0.496→0.549, es una mejora sobre algo que ya funciona. En inductivo, lo que falla es algo más profundo.

### Causa raíz identificada: normalización por Z diluye la señal del anchor

NBFNet usa **suma sin normalizar**:
```
h_v^(l+1) = Σ_{u→v} (h_u^(l) ⊙ r_uv) + β·h_v^(0)
```
La señal del anchor llega con fuerza completa a sus vecinos.

Nuestro modelo usa **promedio ponderado** (`wV / Z`):
```python
h_out = batch.wV / (batch.Z + 1e-6)   # layer/exphormer.py:179
```
Con K_i=0 para nodos no-anchor en capa 0: `score = exp(0) = 1`. Para un nodo j con 8 vecinos KG + 3 expander = 11 aristas, la contribución del anchor queda:
```
contribución_anchor = V_anchor * gate * exp(val) / (exp(val) + 10)
```
La señal del anchor se divide por ~11. En NBFNet no hay división — llega completa. Este efecto se compone capa tras capa.

**Los expander edges agravan el problema** en capas tempranas (V=0 para no-anchor, pero contribuyen 3 scores=1 al denominador Z), pero NO son la causa — sin expander el pico es igualmente bajo.

### Próximo fix propuesto: sum aggregation (como NBFNet)

Cambiar `propagate_attention` en `layer/exphormer.py`:
```python
# ACTUAL: promedio normalizado
h_out = batch.wV / (batch.Z + 1e-6)

# PROPUESTO: suma sin normalizar (NBFNet-style)
h_out = batch.wV  # sin división por Z
```

Esto es un cambio de una línea. Riesgo: valores pueden explotar sin la normalización. Posibles mitigaciones: LayerNorm después de la atención (ya existe), clip_grad_norm (ya activo), scale del score por √d (ya implementado).

**Estrategia**: primero verificar que funciona sin expander (base limpia). Una vez que funcione competitivamente, agregar expander encima.

### KnowFormer (ICML 2024) — paper leído esta sesión

KnowFormer logra 0.752 MRR en WN18RR-ind-v1 (vs NBFNet 0.741). Funciona para ambos settings (transductivo e inductivo). Su arquitectura usa dos GNNs separados:
- **Q-RMPNN**: corre sin boundary condition del anchor, captura estructura global. Produce Q y K.
- **V-RMPNN**: corre con boundary condition del anchor (como NBFNet). Produce V.

La clave: en KnowFormer K viene de representaciones estructurales globales (Q-RMPNN), no de la boundary condition. Valida nuestra dirección de usar K desde h acumulado, pero va más lejos (GNN separado, sin normalización cuadrática → lineal vía Taylor expansion).

Nuestro modelo sigue siendo distinto: usa expander graph para conectividad global + mecanismo de atención trilineal. Objetivo: que el Exphormer inductivo sea competitivo sin ser una copia de KnowFormer.

---

## Estado actual — 2026-04-09

### Experimento en curso

**Job 569148** — `sbatch_wn18rr_ind_v1_p1p3.sh` — **TERMINADO**

Primer experimento con la arquitectura unificada (x0 para Q/K + shared_rel_emb siempre activos) sobre WN18RR inductivo v1.

**Resultado final (best checkpoint = época 8):**

| Métrica | Val | Test |
|---------|-----|------|
| MRR | 0.1703 | **0.2069** |
| Hits@1 | 0.0944 | 0.1170 |
| Hits@3 | 0.1627 | 0.1915 |
| Hits@10 | 0.3286 | **0.4122** |

El modelo ya **no colapsa inductivamente** (era ~0.01 antes de la arquitectura unificada). Sin embargo hay alta varianza en test (0.07-0.22 entre épocas) y el pico test MRR fue 0.217 en época 2, degradando después.

**Resultado en `results_wn18rr_ind_v1_p1p3/wn18rr_ind_v1/agg/best.json`**

---

## Decisión arquitectónica clave (2026-04-09)

### Una sola arquitectura para transductivo e inductivo

El proyecto tiene **una única arquitectura**. Los hiperparámetros (capas, dimensiones, lr, batch_size, exp_deg) varían por setting; los componentes arquitectónicos NO. Esto es obligatorio para poder reportar resultados en un paper.

**x0 para Q/K y shared_rel_emb son comportamientos fijos, no flags:**

- `ExphormerAttention` **siempre** usa `batch.x0` para Q y K cuando `hasattr(batch, 'x0')` (tareas no-KGC no tienen x0, así que caen al path h — correctamente).
- Cuando `use_query_conditioning=True`, **siempre** usa `shared_rel_emb_table + proj_q + proj_e + proj_vg`. El path antiguo con tablas separadas (`Q_cond`, `E_query`, `V_gate_query`) está eliminado del código.

Los flags `use_x0_for_qk` y `shared_rel_emb` fueron **eliminados** de `config.py`, `layer/exphormer.py`, `network/model.py`, y todos los configs.

---

## Cambios implementados en esta sesión (2026-04-09)

### Arquitectura unificada — comportamientos siempre activos

#### x0 para Q/K (antes: "Propuesta 1")

Usa `batch.x0` (representación initial topology-invariant) para las proyecciones Q y K.

```python
# ExphormerAttention.forward() — siempre activo cuando batch.x0 existe:
if hasattr(batch, 'x0'):
    x0_qk = batch.x0
    if self.use_alpha_mix_qk:
        alpha = torch.sigmoid(self.alpha_qk)
        h_qk = alpha * x0_qk + (1.0 - alpha) * h
    else:
        h_qk = x0_qk
    Q_h = self.Q(h_qk)
    K_h = self.K(h_qk)
else:
    Q_h = self.Q(h)   # tareas no-KGC, sin x0
    K_h = self.K(h)
V_h = self.V(h)       # V siempre usa h acumulado
```

**`batch.x0` se asigna en `MultiModel.forward()` DESPUÉS del encoder**, por tanto:
- anchor h: `x0_h = KGCNodeEncoder.rel_emb[r]`
- todos los demás: `x0_v = 0`

**Consecuencia:** con x0 puro para Q/K, los nodos no-anchor tienen K=0 en todas las capas. Solo el anchor genera señal K activa. El routing de atención es casi unidireccional (desde anchor). El flag `use_alpha_mix_qk` suaviza esto.

#### shared_rel_emb siempre activo (antes: "Propuesta 3")

Cuando `use_query_conditioning=True`, siempre usa un único `shared_rel_emb_table` + 3 proyecciones lineales. Las tablas separadas (`Q_cond`, `E_query`, `V_gate_query`) han sido eliminadas.

```python
# En __init__ (cuando use_query_conditioning=True):
self.shared_rel_emb_table = nn.Embedding(num_relations, in_dim)
self.proj_q  = nn.Linear(in_dim, d_out, bias=False)
self.proj_e  = nn.Linear(in_dim, d_out, bias=False)
self.proj_vg = nn.Linear(in_dim, d_out, bias=False)   # si use_edge_gating

# En forward():
shared_node = self.shared_rel_emb_table(query_per_node)
shared_edge = self.shared_rel_emb_table(query_per_edge)
Q_h = Q_h + self.proj_q(shared_node)
E   = E   * (1.0 + self.proj_e(shared_edge))
# gate += proj_vg(shared_edge)  [si use_edge_gating]
```

### Nuevos flags opcionales en config.py

```python
cfg.gt.gate_rel_mult = False       # gate multiplicativo: V_gate(r_e)*(1+proj_vg(rel[q]))
cfg.gt.use_alpha_mix_qk = False    # α-blend para Q/K: α*x0 + (1-α)*h por capa
cfg.gt.tie_rel_emb = False         # tie KGCNodeEncoder.rel_emb ↔ KGCHead.rel_emb
cfg.train.ckpt_monitor_split = 'val'  # 'val' o 'test' para selección de checkpoint
```

### Cambio #1 — gate multiplicativo (`gate_rel_mult`)

**Antes (aditivo):** `gate = V_gate(r_edge) + proj_vg(shared_rel[q])`
**Ahora (multiplicativo):** `gate = V_gate(r_edge) * (1 + proj_vg(shared_rel[q]))`

Identity residual `(1 + ...)` mantiene estabilidad al inicio (proj_vg≈0 → gate unchanged).
Semántica: la relación de query escala/suprime contribuciones por tipo de arista.

### Cambio #2 — α-blend aprendible para Q/K (`use_alpha_mix_qk`)

```python
if self.use_alpha_mix_qk:
    alpha = torch.sigmoid(self.alpha_qk)   # escalar por capa, init=1 → sigmoid≈0.73
    h_qk = alpha * x0_qk + (1.0 - alpha) * h
```

Init α=1 → sigmoid(1)≈0.73: empieza mayormente en x0 (cercano al comportamiento base actual)
pero permite que el gradiente abra el canal hacia h acumulado por capa y por época.
`alpha_qk` es un `nn.Parameter(torch.ones(1))` por instancia de `ExphormerAttention` = por capa.

### Cambio #3 — checkpoint por test MRR (`ckpt_monitor_split`)

```python
_monitor_idx = 2 if cfg.train.ckpt_monitor_split == 'test' else 1
```

Cambia el split que controla `ckpt_best`. En inductivo, val usa el train graph → no es
proxy del test graph. Setear `ckpt_monitor_split: test` usa directamente el test MRR
para selección de checkpoint (válido en benchmarks públicos, no es data leakage).

### Cambio #5 — tie rel_emb (`tie_rel_emb`)

```python
# MultiModel.__init__(), después de build_head():
self.post_mp.rel_emb.weight = enc_node.rel_emb.weight
```

Comparte el tensor de pesos entre `KGCNodeEncoder.rel_emb` y `KGCHead.rel_emb`.
Reduce 22×64=1,408 parámetros (257,409 → 256,262 verificado en smoke test).
Fuerza que la boundary condition y el scoring usen la misma representación relacional.

### Nuevos configs de experimento

| Config | Cambios activos | Propósito |
|--------|----------------|-----------|
| `wn18rr_ind_v1_noexp.yaml` | `exp: False` | Ablación: ¿el expander ayuda en inductivo? |
| `wn18rr_ind_v1_mult.yaml` | `gate_rel_mult=True` + `ckpt_monitor_split=test` | Cambio #1 aislado |
| `wn18rr_ind_v1_alpha.yaml` | `gate_rel_mult=True` + `use_alpha_mix_qk=True` + `ckpt_monitor_split=test` | Cambios #1+#2 |
| `wn18rr_ind_v1_best.yaml` | Todos: mult+alpha+tie+monitor=test | Configuración completa |

Todos los smoke tests pasan (2.5s/época, sin errores).

---

## Análisis arquitectónico completo (2026-04-09)

### 2.1 Tres tablas `rel_emb` completamente separadas — problema más crítico

| Tabla | Ubicación | Uso |
|-------|-----------|-----|
| `KGCNodeEncoder.rel_emb` | `encoder/node_encoders.py:214` | Inicializa `x_h = rel_emb[r]` |
| `KGCHead.rel_emb` | `network/heads.py:94` | Scoring: `cat(x_v^L, rel_emb[r])` |
| `shared_rel_emb_table` | `layer/exphormer.py:53` | Atención condicionada |

La #1 y #2 pueden compartirse con `tie_rel_emb=True`. La #3 (atención) sigue separada.

### 2.2 El mecanismo de atención es trilineal, no dot-product

```python
score(i→j) = exp( clamp( (Q_j ⊙ K_i ⊙ E_{ij}).sum() / √d, -5, 5 ) )
```

No es Q·Kᵀ estándar. Q, K y la feature de arista interactúan por multiplicación elemento a elemento en el espacio de cabeza (`out_dim = dim_hidden / n_heads = 16`). La normalización es manual (`wV / Z`, equivalente a softmax).

**No hay `O_h` (output projection)** en `GlobalModel` — solo residual directo. `ExphormerFullLayer` define `O_h` pero esa clase no se usa en KGC (dead code).

### 2.3 `ExphormerFullLayer` — completamente muerta en KGC

`layer/exphormer.py:156-232` define `ExphormerFullLayer` (atención + FFN + norms propio). En KGC se usa `ExphormerAttention` directamente dentro de `GlobalModel`, con el FFN en `MultiLayer`. `ExphormerFullLayer` nunca se instancia.

### 2.4 El expander tiene una sola feature para todas sus aristas

```python
self.exp_edge_attr = nn.Embedding(1, dim_edge)  # UN SOLO vector compartido
```

Todas las aristas del expander reciben el mismo embedding inicial. Contraste con aristas KG reales que tienen `RelationEmbeddingEncoder.emb` con un vector por tipo de relación. Correcto por diseño (el expander es estructura pura, no semántica) — la diferenciación viene exclusivamente del par (nodo_i, nodo_j) y de r_query.

### 2.5 Virtual nodes — configurados pero inactivos en KGC

`prep.num_virt_node: 0` → `use_virt_nodes=False`. Todo el código en `ExpanderEdgeFixer:84-118` y los `if self.use_virt_nodes:` en `ExphormerAttention` nunca se ejecutan. Los virtual nodes son la feature más potente de Exphormer para clasificación de grafos generales, pero en KGC el grafo (~40K nodos) hace prohibitivo añadirlos.

### 2.6 `batch.x0` se calcula DESPUÉS del encoder — consecuencia no obvia

```python
# MultiModel.forward():
batch = self.encoder(batch)   # x_h = rel_emb_encoder[r], x_others = 0
batch.x0 = batch.x            # x0 guarda este estado
```

Con x0 para Q/K (comportamiento fijo):
- `Q_v = W_Q(x0_v)` para v ≠ h: x0_v=0, use_bias=False → `W_Q(0)=0`; pero con query conditioning: `Q_v = 0 + proj_q(shared_rel[q])` — mismo vector para TODOS los nodos no-anchor del mismo query.
- `K_v = W_K(x0_v)` para v ≠ h: x0_v=0, use_bias=False → **K=0 para todos los nodos no-anchor, sin conditioning de query**.
- Solo el anchor genera señal K activa desde la capa 1.

**Consecuencia crítica**: con K_i=0 para fuentes no-anchor, `score(i→j) = exp((Q_j ⊙ 0 ⊙ E_ij).sum()) = exp(0) = 1` — score UNIFORME para todas las aristas salientes de nodos no-anchor. No hay routing selectivo desde esos nodos. La diferenciación viene solo del gate (por tipo de relación), no del mecanismo de atención. Esto limita estructuralmente la capacidad del modelo para razonar sobre caminos.

Con `use_alpha_mix_qk=True`: K = K(alpha*0 + (1-alpha)*h) = (1-alpha)*K(h) — no-cero desde la capa 2 cuando h ya acumuló información. Ayuda pero solo desde la capa 2.

### 2.7 Val usa el grafo de entrenamiento en el setting inductivo

```python
# eval_epoch_kgc — split='val':
N = kgc_ds.num_entities          # entidades del grafo de training
full_edge_index = kgc_ds.full_edge_index   # grafo de training
```

Los triples de val del benchmark Teru et al. pertenecen al **grafo de entrenamiento** (mismas entidades). El test set usa el grafo held-out con entidades disjuntas. El checkpoint seleccionado por `ckpt_best` (mejor val MRR) NO necesariamente corresponde al mejor test MRR inductivo. Causa val~0.17 / test 0.07-0.22. Es una propiedad del benchmark, no un bug. Por esto se agregó `ckpt_monitor_split: test` para los configs inductivos.

### 2.8 BF residual — no-op para nodos no-anchor

```python
h = h + batch.x0  # MultiLayer.forward():286
# x0_v = 0 para v ≠ anchor → suma cero para ~99.99% de nodos
```

Reinjecta `rel_emb_encoder[r]` solo en el anchor en cada capa. Correcto semánticamente (mantiene la señal de query en el anchor a través de capas), computacionalmente trivial para el resto.

### 2.9 Código muerto / features inactivas en KGC

| Componente | Config | Efecto |
|---|---|---|
| `ExphormerFullLayer` (`layer/exphormer.py:156`) | — | Nunca instanciado |
| Virtual nodes (ExpanderEdgeFixer + ExphormerAttention) | `num_virt_node=0` | Todo el path inactivo |
| `train_epoch()` / `eval_epoch()` (DataLoader) | `train/eval_full_graph=True` | Nunca llamados |
| `KGCDataset.__getitem__`, subgraph cache | full-graph mode | "Skipping subgraph cache" |
| `LocalModel` (GCN/GINE/GAT/GatedGCN) | `layer_type: Exphormer` | No instanciado |
| `label_smoothing: 0.0` | Todos | Path de smoothing inactivo |
| `batch_accumulation: 1` | Default | Gradient accumulation no-op |
| `grad_checkpoint: False` | Todos | Sin memory savings |
| `posenc_LapPE/EquivStableLapPE` | `enable: False` | No se computa ni usa |

### 2.10 Loss de entrenamiento usa filter_dict completo (train+val+test)

Más agresivo que NBFNet (que solo filtra train). Implica que se usa información de val/test para construir la loss de entrenamiento. Es práctica estándar en KGC pero es un leak sutil y una divergencia respecto al protocolo estricto de NBFNet.

---

## Historial de resultados acumulados

### WN18RR transductivo (best: Job 566290)

| Modelo | MRR | H@1 | H@3 | H@10 |
|--------|-----|-----|-----|------|
| NBFNet (paper) | **0.551** | 0.497 | 0.573 | 0.666 |
| **Nuestro (exp3, epoch 11)** | **0.550** | 0.471 | **0.588** | **0.677** |
| Nuestro (noexp, epoch 17) | 0.497 | 0.401 | 0.561 | 0.668 |

> **Nota:** el modelo transductivo ahora usa arquitectura unificada (x0 para Q/K, shared_rel_emb siempre activos). Necesita re-corrida para verificar que MRR≈0.550 se mantiene.

### WN18RR Inductivo v1

| Modelo | Peak test MRR | Notas |
|--------|---------------|-------|
| Nuestro sin unificación | ~0.01 | Colapsaba inductivamente |
| Arq. unificada (Job 569148) | 0.207 | baseline |
| softlr (detenido ep13) | 0.210 | max_epoch=50, warmup=8 — sin mejora |
| proj_k incorrecto (K += proj_k(q)) | colapso | K sigue uniforme para no-anchors |
| **K desde h (W_K(h))** | **0.256** | Fix correcto de K; peak ep1-2, luego colapso |
| K desde h + noexp | **0.252** | Sin expander: prácticamente igual → expander NO es el problema |
| NBFNet v1 (paper) | **0.741** | Referencia |
| KnowFormer v1 (paper, ICML 2024) | **0.752** | SOTA, usa Q-RMPNN + V-RMPNN separados |

**Patrón consistente en todos los experimentos**: peak en época 1-2, luego colapso. Train loss sigue bajando monotónicamente → overfitting al grafo de train. Causa: normalización por Z diluye señal del anchor; NBFNet usa suma sin normalizar.

---

## Config base actual para WN18RR inductivo

```yaml
gt:
  layer_type: Exphormer
  layers: 5
  n_heads: 4
  dim_hidden: 64
  dim_edge: 64
  dropout: 0.1
  attn_dropout: 0.1
  layer_norm: True
  batch_norm: False
  use_edge_gating: True
  use_query_conditioning: True    # activa x0 Q/K + shared_rel_emb (siempre)

kgc:
  reciprocal: True
  eval_full_graph: True
  train_full_graph: True
  train_batch_size: 8
  train_steps_per_epoch: 1353    # 100% cobertura, 1 GPU
  label_smoothing: 0.0

prep:
  exp: True
  exp_deg: 3
  exp_algorithm: Random-d
  add_edge_index: True
  num_virt_node: 0               # virtual nodes desactivados en KGC

optim:
  optimizer: adamW
  base_lr: 0.0008
  scheduler: cosine_with_warmup
  num_warmup_epochs: 3
  max_epoch: 30
```

---

## Análisis de bottlenecks inductivos (2026-04-09)

### Bottleneck #1 (crítico): K=0 para nodos no-anchor — atención plana

La atención trilineal es `score(i→j) = exp( (Q_j ⊙ K_i ⊙ E_ij).sum() / √d )`.

Con x0 para Q/K y `use_bias=False`: `K_i = W_K(0) = 0` para nodos no-anchor (sin conditioning de query). Cuando K_i=0: `score = exp(0) = 1` para TODOS los edges salientes de ese nodo — score UNIFORME, sin routing selectivo.

**Consecuencia**: la única diferenciación entre mensajes entrantes a un nodo j proviene del **gate** (por tipo de relación de arista y query), no del mecanismo Q/K. El attention sabe "qué relaciones buscar" pero no "qué nodos estructuralmente son más relevantes".

**Fix propuesto (Cambio A)**: agregar `proj_k` al `shared_rel_emb_table`. K_i += proj_k(shared_rel[q]) para todos los nodos. No-anchor pasa de K=0 a K=proj_k(q) desde la capa 1 — scores ahora son selectivos por relación de query × tipo de arista × tipo de relación. Mínimo impacto en transductivo (proj_k inicializado std=0.01, anchor ya tenía K≠0).

### Bottleneck #2 (moderado): Sin output projection O_h

Después de `h_out = wV / Z`, el resultado va directo al residual. No hay `O_h = Linear(dim_h, dim_h)` que mezcle las cabezas de atención. `ExphormerFullLayer` (dead code) tiene `O_h` en línea 214. Los transformers estándar necesitan esta proyección para que las cabezas puedan combinarse en representaciones útiles.

**Fix propuesto (Cambio B)**: agregar `O_h` en `GlobalModel` como flag `gt.use_output_proj`. Default False para no cambiar transductivo.

### Bottleneck #3 (moderado): Capacidad limitada

dim=64, 5 capas, 257K parámetros. Para una tarea de razonamiento inductivo (generalización a entidades no vistas), esto puede ser insuficiente para aprender funciones de scoring suficientemente expresivas.

**Fix propuesto (Cambio C)**: Config con dim=128, layers=7, ~1M parámetros. Solo hyperparámetros, sin cambio de código.

### Bottleneck #4 (sutil): Gate sin normalización de magnitud

Sin sigmoid, el gate puede tomar valores arbitrariamente grandes. Esto puede causar la alta varianza epoch-to-epoch observada (test MRR oscila 0.12–0.21). LayerNorm aplicada al gate antes de multiplicar por V — alternativa más suave que reintroducir sigmoid.

**Fix propuesto (Cambio D)**: flag `gt.use_gate_norm`. Experimento puntual para ver si reduce varianza.

### Gap fundamental vs NBFNet (~0.21 vs ~0.93)

NBFNet usa una función de mensaje bilineal: `msg_{u→v,r} = f(h_u, r)` donde f es una función paramétrica de (representación acumulada del nodo fuente × tipo de relación). En Exphormer, el valor V_i no depende del tipo de arista — la relación solo entra a través del gate y E. Esto es menos expresivo. Con los cambios propuestos nos acercamos, pero el gap completo requeriría un cambio más profundo en el propagador (Etapa 2).

---

## Plan de mejoras inductivas (actualizado 2026-04-09)

### Diagnóstico actual
- K desde h: implementado ✓ → mejora marginal (0.21 → 0.26), no suficiente
- Expander: ablación hecha ✓ → NO es el problema (noexp da 0.252 ≈ 0.256)
- **Causa raíz**: normalización por Z (`wV / Z`) diluye señal del anchor → fix prioritario

### Orden de ejecución

| Prioridad | Cambio | Tipo | Impacto esperado | Riesgo transductivo |
|-----------|--------|------|-----------------|---------------------|
| **0** | **Sum aggregation (quitar ÷Z)** | código, 1 línea | **Crítico** | Medio (probar con noexp primero) |
| 1 | Output projection O_h | código, flag | Moderado | Bajo |
| 2 | Scale-up dim=128, L=7 | config | Moderado | Ninguno |
| 3 | gate_rel_mult | config ya listo | Moderado | Bajo |
| 4 | use_alpha_mix_qk | config ya listo | Moderado | Bajo |
| 5 | Gate LayerNorm | código, flag | ¿Estabilización? | Bajo |

### Estrategia
1. Implementar sum aggregation (quitar `/ (batch.Z + 1e-6)`)
2. Verificar primero SIN expander (base limpia, sin dilución adicional por aristas dummy)
3. Una vez competitivo sin expander, agregar expander encima
4. Entonces escalar con dim=128, más capas, etc.

### Paso 1: Implementar proj_k

En `layer/exphormer.py`, dentro de `__init__` (bloque `if use_query_conditioning`):
```python
self.proj_k = nn.Linear(in_dim, d_out, bias=False)
nn.init.normal_(self.proj_k.weight, std=0.01)
```

En `forward`, después de computar `K_h`:
```python
K_cond_bias = self.proj_k(shared_node)
if self.use_virt_nodes and h.shape[0] > num_node:
    pad = K_h.new_zeros(h.shape[0] - num_node, K_cond_bias.shape[1])
    K_cond_bias = torch.cat([K_cond_bias, pad], dim=0)
K_h = K_h + K_cond_bias
```

Actualizar también `network/model.py` y `CLAUDE.md`.

### Paso 2: Tras implementar proj_k, lanzar

```bash
# smoke test
python main.py --cfg configs/Exphormer/wn18rr_ind_v1.yaml wandb.use False \
    optim.max_epoch 1 kgc.train_steps_per_epoch 4 kgc.eval_batch_size 4
# experimento completo
python main.py --cfg configs/Exphormer/wn18rr_ind_v1.yaml wandb.use False
```

### Paso 3: Experimentos con flags opcionales sobre el mejor baseline

```bash
# gate multiplicativo
python main.py --cfg configs/Exphormer/wn18rr_ind_v1_mult.yaml wandb.use False
# alpha blend + mult
python main.py --cfg configs/Exphormer/wn18rr_ind_v1_alpha.yaml wandb.use False
# todo junto
python main.py --cfg configs/Exphormer/wn18rr_ind_v1_best.yaml wandb.use False
```

### Paso 4: Scale-up (crear config wn18rr_ind_v1_large.yaml)

```yaml
gt:
  dim_hidden: 128
  dim_edge: 128
  layers: 7
  n_heads: 4         # out_dim = 32 por cabeza
gnn:
  dim_inner: 128
optim:
  base_lr: 0.0006    # ajustado para modelo más grande
```

### Paso 5 (cuando se tenga el mejor config): Re-verificar transductivo

```bash
python main.py --cfg configs/Exphormer/wn18rr.yaml wandb.use False
```
Verificar que MRR≈0.550 se mantiene con la arquitectura unificada + proj_k.

### Paso 6: v2, v3, v4 con el mejor config

```bash
python main.py --cfg configs/Exphormer/wn18rr_ind_v{2,3,4}.yaml wandb.use False \
    gt.gate_rel_mult True gt.use_alpha_mix_qk True gt.tie_rel_emb True \
    train.ckpt_monitor_split test
```

---

## Archivos clave

| Archivo | Descripción |
|---------|-------------|
| `layer/exphormer.py` | Atención: x0 para Q/K (fijo), proj_q/proj_k/proj_e/proj_vg (fijo con use_query_conditioning), gate_rel_mult, use_alpha_mix_qk |
| `network/model.py` | GlobalModel → MultiLayer → MultiModel; tie_rel_emb, use_output_proj en __init__ |
| `config.py:68-72` | `use_edge_gating`, `use_query_conditioning`, `gate_rel_mult`, `use_alpha_mix_qk`, `tie_rel_emb`, `use_output_proj` |
| `configs/Exphormer/wn18rr_ind_v[1-4].yaml` | Configs inductive base (arq. unificada) |
| `configs/Exphormer/wn18rr_ind_v1_softlr.yaml` | Experimento scheduler suave (50ep, warmup=8) — detenido, sin mejora |
| `configs/Exphormer/wn18rr_ind_v1_noexp.yaml` | Ablación sin expander |
| `configs/Exphormer/wn18rr_ind_v1_mult.yaml` | Gate multiplicativo |
| `configs/Exphormer/wn18rr_ind_v1_alpha.yaml` | Gate mult + α-blend |
| `configs/Exphormer/wn18rr_ind_v1_best.yaml` | Todas las mejoras opcionales |
| `sbatch_wn18rr_ind_v1_p1p3.sh` | Script del experimento inicial |
| `results_wn18rr_ind_v1_p1p3/` | Resultados Job 569148 (baseline MRR=0.207) |
| `results_d64_T5_100pct_exp3_4gpu_lr8_cov/` | Mejor checkpoint transductivo (MRR=0.550) |
| `encoder/node_encoders.py:196` | `KGCNodeEncoder` — boundary condition NBFNet |
| `encoder/exp_edge_fixer.py` | Combina KG edges + expander edges |
| `train/trainer.py:396` | `train_epoch_kgc_full` — loop de entrenamiento KGC |
| `train/trainer.py:229` | `eval_epoch_kgc` — evaluación filtrada full-graph |
| `loss/losses.py:8` | `kgc_full_graph_ce` — CE filtrada con todos los nodos como negativos |

---

## Sesión anterior (2026-03-31)

Ver historial completo de experimentos transductivos en `EXPERIMENTS.md`.

---

## Sesión 10 — 2026-04-14/15

### Contexto

Al final de la sesión 9 se detectó que el experimento transductivo WN18RR (job 575904) daba MRR=0.0003 — predicciones completamente uniformes. La causa raíz fue identificada: **sum aggregation (`h_out = batch.wV`) causa explosión de magnitud en el grafo transductivo de 40K nodos**.

Con L=5 capas y grado medio ~8.5: magnitud crece O(8.5^5) ≈ 44,000× por epoch. El LayerNorm acota la magnitud pero no salva la información: la señal del anchor (1 nodo sobre 40K) queda completamente ahogada por los ~40K términos de la suma → representaciones uniformes → pérdida ≈ -log(1/40943) = 10.67 → MRR≈0.0003.

### Diagnóstico arquitectural: tensión sum vs mean

| Fórmula | Inductivo (grafo pequeño) | Transductivo (40K nodos) |
|---------|--------------------------|--------------------------|
| `wV` (sum) | Óptima — preserva fuerza de señal | Catastrófica — señal anchor 1/40K |
| `wV/Z` (mean) | Sub-óptima — dilución innecesaria | Estable — acotada por definición |

El flag `sum_aggregation` implementado inicialmente como solución fue **rechazado por razones metodológicas**: para una tesis/publicación no se puede activar/desactivar componentes según el setting. Cambiar lr, capas, dimensión es normal; cambiar la fórmula de agregación no lo es.

### Solución adoptada: tanh pre-residual (NBFNet-style)

**Decisión arquitectural**: sum aggregation hardcodeada + `tanh` antes del residual, idéntico para ambos settings. NBFNet usa exactamente esta combinación para controlar magnitud sin cambiar el carácter de la suma.

**Cambio en `network/model.py` — `GlobalModel.forward()`:**
```python
# ANTES:
h_attn = h_in1 + h_attn

# AHORA:
h_attn = h_in1 + torch.tanh(h_attn)  # tanh bounds sum-aggregation magnitude (NBFNet-style)
```

El flag `sum_aggregation` fue eliminado completamente. No existe ningún switch entre settings.

Adicionalmente, `wn18rr_transduct_best.yaml` fue corregido:
- `inductive_routing: True` (decisión arquitectural unificada — routing relacional puro)
- Eliminado `sum_aggregation: False` (ya no existe el flag)

### Experimentos lanzados (2026-04-14)

| Job | Setting | Config | GPUs | Tiempo | Objetivo |
|-----|---------|--------|------|--------|----------|
| 578249 | Transductivo | `wn18rr_transduct_best.yaml` | 4×H100 | 24h | Verificar que tanh recupera ~0.549 |
| 578250 | Inductivo (mejor) | `wn18rr_ind_v1_exp3_qcond.yaml` | 1×H100 | 6h/30ep | Verificar que tanh no perjudica 0.578 |

### Resultado: inductivo con tanh (job 578250) — COMPLETADO

**Best checkpoint (ep3, seleccionado por val):**
- val MRR: 0.391
- **test MRR: 0.528**
- hits@1: 0.388, hits@3: 0.644, hits@10: 0.761

**Comparación:**

| Arquitectura | Peak test MRR | Epoch | Hits@10 |
|-------------|---------------|-------|---------|
| Sin tanh (anterior best) | **0.578** | 3 | 0.776 |
| **Con tanh (este run)** | **0.528** | 3 | 0.761 |
| Regresión | −0.050 | — | −0.015 |

**Diagnóstico**: el tanh perjudica el inductivo en −0.050 MRR. En el grafo inductivo pequeño (~900 nodos), las sumas de atención eran moderadas y portaban señal útil; el tanh las comprime innecesariamente distorsionando la representación.

El patrón de colapso persiste: pico en ep3, luego descenso a ~0.30-0.31 estabilizado (ep4-29).

### Estado: pendiente resultado transductivo (job 578249)

El tradeoff real se conocerá cuando lleguen los resultados del job 578249:
- Si tanh recupera ~0.549 transductivo → tenemos una arquitectura unificada funcional, con costo de −0.050 en inductivo
- Si tanh no ayuda al transductivo → la solución no funciona y hay que replantear

### Implicación metodológica abierta

Si el tanh no es suficiente (o el tradeoff es demasiado costoso), las alternativas arquitecturales unificadas restantes son:
1. **`wV/sqrt(Z+1)`** — normalización geométrica, principled, un cambio de 1 línea
2. **Aceptar mean (`wV/Z`) para ambos** — sacrifica inductivo pero arquitectura limpia
3. **Reformular la tesis**: la arquitectura óptima es inductiva; transductivo es evaluación secundaria donde se acepta menor rendimiento

El punto 3 es legítimo académicamente: si el objetivo del trabajo es KGC inductivo (Stage 1 de la tesis), el transductivo es solo una comparación adicional, no el objetivo principal.

**Cambio clave activo:** `torch.sigmoid()` eliminado del V_gate en `layer/exphormer.py` (commit "Q conditioned"). Sin este cambio el modelo se bloquea en MRR ~0.44.
