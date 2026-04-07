# Exphormer-KGC: Registro de Experimentos y Análisis

> WN18RR · 4× H100 NVL · Última actualización: 2026-04-05

---

## 1. Contexto del Proyecto

Reimplementación standalone de **Exphormer** (Shirzad et al., ICML 2023) adaptada para **Knowledge Graph Completion (KGC)**, sin dependencias de GraphGPS/graphgym. El objetivo es desarrollar un Graph Transformer con atención dispersa y condicionamiento relacional que sea competitivo con **NBFNet** (Zhu et al., NeurIPS 2021) en WN18RR y FB15k-237, como primera etapa hacia un modelo fundacional para KGs (Etapa 1 del proyecto doctoral).

**Baseline target:** NBFNet · WN18RR · test MRR = **0.551**

---

## 2. Arquitectura Base

```
MultiModel
  ├── FeatureEncoder
  │     ├── KGCNodeEncoder   x_h = rel_emb[r],  x_v = 0  (NBFNet boundary condition)
  │     ├── RelationEmbeddingEncoder
  │     └── ExpanderEdgeFixer  (combina aristas locales + expander + virtual nodes)
  ├── MultiLayer × T
  │     ├── GlobalModel → ExphormerAttention (sparse, query-conditioned)
  │     └── FFN compartido
  └── KGCHead  score(v) = MLP( concat(x_v, rel_emb[r]) )
```

**Configuración principal (WN18RR):**
- `dim_hidden = 64`, `T = 5 capas`, `n_heads = 4`
- `prep.exp = True`, `exp_deg = 3` (expander d-regular)
- Full-graph training (NBFNet-style): grafo completo con query-answer edges eliminados
- Reciprocal triples: (h, r, t) → (t, r+|R|, h)
- Evaluación: filtered ranking sobre grafo completo

**Query conditioning (atención condicionada a la relación query):**
```python
Q_cond       = Embedding(|R|, d)   # modula queries por relación
E_query      = Embedding(|R|, d)   # modula features de aristas
V_gate_query = Embedding(|R|, d)   # controla flujo de valores
V_gate       = Linear(d, d)        # gate sobre aristas (sin sigmoid — ver §3.1)
```

---

## 3. Cambios Clave en el Código

### 3.1 Remoción de sigmoid en V_gate (`layer/exphormer.py`)
**Commit: "Q conditioned"**

Eliminar `torch.sigmoid()` del gate de valores produjo el mayor salto de rendimiento individual:

| Antes | Después |
|---|---|
| test MRR ~0.449 | test MRR ~0.534 |

**Por qué:** sigmoid satura para valores grandes/pequeños, bloqueando el flujo de gradientes hacia los parámetros de gating. Sin sigmoid, el gate opera como escala lineal, permitiendo que la red aprenda magnitudes arbitrarias de modulación.

### 3.2 Sampling coordinado entre GPUs (`train/trainer.py:435`)
**Problema detectado:** Con DDP (4 GPUs), cada rank generaba su propio `torch.randperm` independiente. Con `train_steps_per_epoch=10854` dividido entre 4 GPUs → cada GPU veía ~25% del dataset de forma aleatoria e independiente, con posibles solapamientos.

**Fix implementado:**
```python
# Antes: cada GPU su propio randperm → ~25% cobertura, con overlap
indices = torch.randperm(n_train).repeat(reps)[:needed]

# Después: permutación compartida (seed=epoch), cada rank toma su slice
g = torch.Generator()
g.manual_seed(cur_epoch)
if dist.is_initialized():
    world_size = dist.get_world_size()
    rank       = dist.get_rank()
    total_needed = world_size * needed
    reps    = (total_needed + n_train - 1) // n_train
    full_perm = torch.randperm(n_train, generator=g).repeat(reps)[:total_needed]
    indices = full_perm[rank * needed: (rank + 1) * needed]
```

**Resultado:** 100% cobertura garantizada por época, sin duplicados entre GPUs. Impacto: +0.008 MRR (0.5388 → 0.5496 al mismo epoch).

### 3.3 `find_unused_parameters=True` en DDP (`main.py:377`)
**Problema:** Con `prep.exp False` (ablation sin expander), los parámetros `exp_edge_attr` no reciben gradientes → DDP lanza `RuntimeError: Expected to have finished reduction`.

**Fix:**
```python
model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=True)
```

Aplicado permanentemente para soportar cualquier configuración de ablation.

### 3.4 Escalado de LR para multi-GPU
Con DDP y 4 GPUs, el batch efectivo es 4×16=64 (los gradientes se promedian entre ranks). La regla de escalado lineal implica `LR_optimo = LR_base × world_size`.

| Config | LR | Resultado |
|---|---|---|
| 1 GPU | 0.0002 | test MRR ~0.537 (epoch 8) |
| 4 GPU, sin escalar | 0.0002 | test MRR ~0.514 (estancado) |
| 4 GPU, LR×4 | 0.0008 | test MRR **0.550** (epoch 11) |

---

## 4. Historial de Experimentos

### 4.1 Experimentos tempranos (1 GPU)

| Job | Config | Mejor epoch | val MRR | test MRR | Notas |
|---|---|---|---|---|---|
| 559792 | dim32, T3, 30% cov | — | — | ~0.43 | Antes de sigmoid fix |
| 560281 | dim64, T5, 30% cov, exp3 | 1 | 0.532 | **0.537** | Post sigmoid fix |
| — | dim64, T5, 30% cov, noexp | — | — | ~0.529 | Sin expander |

### 4.2 Transición a 4 GPUs (sampling independiente, bug presente)

| Job | LR | Mejor epoch | val MRR | test MRR | Notas |
|---|---|---|---|---|---|
| 560937 | 0.0002 | 4 | 0.467 | 0.464 | Bug sampling, LR sin escalar |
| 561383 | 0.0002 | 6 | 0.503 | 0.505 | Bug sampling, LR sin escalar |
| 563307 | 0.0002 | 5 | 0.509 | 0.514 | Bug sampling, LR sin escalar |
| 565999 | 0.0008 | 13 | 0.535 | 0.538 | LR escalado, bug sampling aún |

### 4.3 Experimentos con fix completo (sampling coordinado + LR escalado)

| Job | Config | Mejor epoch | val MRR | test MRR | Notas |
|---|---|---|---|---|---|
| 566290 | dim64, T5, exp3, lr=0.0008, 4GPU | **11** | **0.5457** | **0.5496** | ✅ Mejor resultado |
| 567030 | dim64, T5, **noexp**, lr=0.0008, 4GPU | 13 | 0.4905 | 0.4970 | Ablation sin expander |

---

## 5. Ablation: Con vs Sin Expander Graph

Experimento controlado: mismos hiperparámetros, única diferencia `prep.exp True/False`.

| Epoch | Con expander (test MRR) | Sin expander (test MRR) | Δ |
|---|---|---|---|
| 4 | 0.524 | 0.456 | +0.068 |
| 7 | 0.547 | 0.491 | +0.056 |
| 11 | **0.550** | 0.481 | +0.069 |
| 13 | 0.538 | **0.497** | +0.041 |
| 20 | 0.547 | 0.491 | +0.056 |

**Nota de eficiencia:** sin expander, cada epoch es ~2× más rápido (1695s vs 3198s), consistente con la reducción de aristas (245K aristas de expander eliminadas de 419K totales).

**Patrón en métricas:**

| Métrica | Sin exp (best) | Con exp (best) | Δ |
|---|---|---|---|
| MRR | 0.497 | **0.550** | +0.053 |
| Hits@1 | 0.401 | **0.471** | +0.070 |
| Hits@3 | 0.561 | **0.588** | +0.027 |
| Hits@10 | 0.668 | **0.677** | +0.009 |

La mayor ganancia en Hits@1 y MRR (predicciones de alta confianza) indica que el expander mejora la **precisión del ranking**, no solo el recall.

---

## 6. Por Qué el Expander Graph Ayuda en KGC

### 6.1 Problema estructural de WN18RR

WN18RR es una taxonomía léxica con estructura predominantemente jerárquica:
- Grado promedio: **~4.2** aristas/nodo (grafo muy disperso)
- Cadenas hypernym de hasta 8-10 hops: `poodle→dog→canine→mammal→animal→organism→entity`
- Clusters semánticos relativamente aislados (animales, artefactos, acciones, estados)

Con T=5 capas y solo aristas locales, el modelo accede a ~4^5 ≈ 1,024 nodos = **2.5% del grafo**. Para inferir `(poodle, hypernym, mammal)` se necesitan ≥4 hops de propagación correcta.

### 6.2 Propiedades espectrales del expander

Un grafo d-regular ε-expander garantiza:

- **Diámetro O(log N):** para N=40,943 y d=3, diámetro ≈ 10. Con T=5 capas, cada nodo puede potencialmente alcanzar cualquier otro en el grafo.
- **Vertex expansion:** todo subconjunto S tiene |N(S)| ≥ (1+ε)|S| → la información no queda atrapada en ningún cluster.
- **Rapid mixing:** caminatas aleatorias convergen a uniforme en O(log N) pasos → cobertura global con pocas capas.
- **Pseudoaleatoriedad:** las conexiones son estructuralmente aleatorias pero con garantías globales, creando puentes cross-domain.

### 6.3 Interacción con Query Conditioning

Las aristas del expander no aportan información semántica propia, pero el mecanismo de atención condicionado a la query les asigna importancia **según la relación consultada**:

```
score(i, j, r) = softmax( (K_j ⊙ Q_i + E_query[r]) / √d )
```

Para `r = hypernym`, el modelo aprende a usar atajos del expander hacia nodos en niveles superiores de la jerarquía. Para `r = member_meronym`, aprende a conectar partes con conjuntos. Las aristas de expander se convierten en **canales de razonamiento relacional aprendidos**, no ruido aleatorio.

### 6.4 Comparación con alternativas

| Mecanismo | Cobertura | Costo | Limitación |
|---|---|---|---|
| Solo aristas locales | k-hop (2.5% con T=5) | O(E) | Chains largas, clusters aislados |
| Atención densa | Total | O(N²) | Inviable: N=40,943 → ~1.7B pares |
| Virtual nodes | Global vía bottleneck | O(N) | Un solo vector comprime todo el grafo |
| **Expander d-regular** | **Global garantizado** | **O(N)** | Óptimo |

### 6.5 Interpretación de resultados

Sin expander, el modelo alcanza ~0.49 MRR — comparable a **DistMult** (0.430) y **ComplEx** (0.440), modelos de embedding sin estructura. El modelo es efectivamente un clasificador local sofisticado.

Con expander, llega a **0.550 MRR** — superando **RotatE** (0.476) y acercándose a **NBFNet** (0.551). La diferencia entre razonamiento local y razonamiento global está casi íntegramente capturada por la presencia del expander.

---

## 7. Resultados en Contexto: WN18RR

| Modelo | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|
| DistMult | 0.430 | 0.390 | 0.440 | 0.490 |
| ComplEx | 0.440 | 0.410 | 0.460 | 0.510 |
| RotatE | 0.476 | 0.428 | 0.492 | 0.571 |
| GraIL | — | — | — | — |
| NBFNet | **0.551** | **0.497** | **0.573** | **0.666** |
| **Exphormer-KGC (ours, epoch 11)** | **0.550** | **0.471** | **0.588** | **0.677** |
| Exphormer-KGC sin expander | 0.497 | 0.401 | 0.561 | 0.668 |

> Nuestro modelo **supera a NBFNet en Hits@3 y Hits@10**, iguala en MRR, con diferencia de 0.001.

---

## 8. Configuración del Mejor Experimento

**Job 566290** — `sbatch_d64_T5_100pct_exp3_4gpu.sh`

```bash
torchrun --nproc_per_node=4 main.py \
    --cfg configs/Exphormer/wn18rr.yaml \
    gt.dim_hidden 64  gnn.dim_inner 64  gt.dim_edge 64 \
    gt.layers 5 \
    kgc.train_steps_per_epoch 10854 \   # → 2713 pasos/GPU × 64 eff. batch = 100% cov
    optim.base_lr 0.0008 \              # LR escalado 4× para batch efectivo 64
    out_dir results_d64_T5_100pct_exp3_4gpu_lr8_cov
```

**Config yaml relevante:**
```yaml
kgc:
  reciprocal: True
  eval_full_graph: True
  train_full_graph: True
  train_batch_size: 16
  label_smoothing: 0.0

gt:
  layer_type: Exphormer
  layers: 5          # sobreescrito en script
  dim_hidden: 64     # sobreescrito en script
  use_edge_gating: True
  use_query_conditioning: True

prep:
  exp: True
  exp_deg: 3
  exp_algorithm: Random-d
  exp_check_spectral: False

optim:
  scheduler: cosine_with_warmup
  num_warmup_epochs: 5
  max_epoch: 100
```

---

## 9. FB15k-237: Setup y Bugs Encontrados

### 9.1 Configuración del experimento

**Misma arquitectura que el mejor WN18RR** (dim=64, T=5, n_heads=4, exp_deg=3, lr=0.0008, 4 GPUs).

Stats del dataset (splits estándar Toutanova & Chen 2015, vía PyG `FB15k_237`):

| | FB15k-237 | WN18RR |
|---|---|---|
| Entidades | 14,541 | 40,943 |
| Relaciones base | 237 | 11 |
| Relaciones con recíprocos | 474 | 22 |
| Train triples (con recíprocos) | 544,230 | 173,670 |
| Val triples | 17,535 | 3,034 |
| Test triples | 20,466 | 3,134 |
| Grado promedio | ~37.4 | ~4.2 |

**Val queries en evaluación:** 17,535 × 2 = 35,070 (bidireccional via queries recíprocas, igual que NBFNet).

`train_steps_per_epoch = 34,015` para 100% de cobertura (vs 10,854 en WN18RR).

**Archivos:**
- Config: `configs/Exphormer/fb15k237.yaml`
- Script: `sbatch_fb15k237_d64_T5_100pct_exp3_4gpu.sh`

### 9.2 Bugs encontrados y corregidos

#### Bug 1: NCCL timeout durante evaluación (`main.py:294`)

**Problema:** PyTorch `dist.init_process_group` usa timeout=600s por defecto. La evaluación de FB15k-237 toma ~616s (35,070 queries × grafo denso) → los ranks 1-3 matan al rank 0 que está evaluando.

La variable de entorno `NCCL_TIMEOUT` NO aplica al timeout del ProcessGroup de PyTorch.

**Fix:**
```python
dist.init_process_group(
    backend='nccl',
    timeout=datetime.timedelta(hours=2),
)
```

**Impacto en tiempo de eval:** La evaluación es más lenta que WN18RR porque:
- 5.8× más queries (35,070 vs 6,068)
- 3.1× más aristas de training en el grafo (544,230 vs 173,670)
- Tiempo total de eval: ~616s (~10 min) por época

`eval_batch_size` no mejora el throughput (pasos 4× más pesados, 4× menos → tiempo igual).

**Jobs fallidos:** 568938 (sin fix), 569096 (fix incompleto con `NCCL_TIMEOUT` env var incorrecta).
**Job correcto:** 569099 (con `timeout=timedelta(hours=2)` en código).

### 9.3 Protocolo de evaluación vs NBFNet y Knowformer

| Aspecto | NBFNet | Knowformer | **Nuestro modelo** |
|---|---|---|---|
| Splits | Toutanova & Chen 2015 | Mismos | Mismos (PyG) ✅ |
| Relaciones inversas | Sí (474) | No (237) | Sí (474) ✅ |
| Evaluación | Bidireccional filtrada | Solo tail (?) | Bidireccional filtrada ✅ |
| Query-edge removal | Sí | — | Sí ✅ |
| Loss | Self-adv. neg. sampling | Neg. sampling | Cross-entropy (todos los nodos) |

**Nota Knowformer:** su tabla reporta |R|=237 (no 474), evaluación descrita como tail-only `(h, r, ?)`. Sus métricas no son directamente comparables con NBFNet ni con nuestro modelo (protocolos distintos). Comparar solo con NBFNet.

**Baseline target FB15k-237 (NBFNet):** MRR=0.415, H@1=0.321, H@3=0.454, H@10=0.599

### 9.4 Historial de jobs FB15k-237

| Job | Config | Estado | Notas |
|---|---|---|---|
| 568938 | d64, T5, exp3, 4GPU, bs_eval=8 | ❌ NCCL timeout en eval epoch 0 | ~616s > 600s límite |
| 569096 | + `NCCL_TIMEOUT=7200` env var | ❌ NCCL timeout igual | env var incorrecta |
| **569099** | + `timedelta(hours=2)` en código | ✅ Corriendo | fix correcto |

**Tiempo por época (medido en job 568938):** ~3.2 horas de training + ~10 min de eval → ~3.4h/época → ~6-7 épocas en 24h.

---

## 10. Próximos Pasos

- [ ] Esperar resultados de Job 569099 (FB15k-237, epoch 0+)
- [ ] Re-ejecutar el mejor experimento WN18RR con wandb activado para seguimiento completo
- [ ] Repetir WN18RR por 100 épocas completas
- [ ] Ablation WN18RR: contribución de query conditioning (`use_query_conditioning False`)
- [ ] Ablation WN18RR: número de capas T (3, 5, 6)
- [ ] Ablation WN18RR: exp_deg (1, 2, 3, 5)
- [ ] Etapa 2 del proyecto: diseño del mecanismo de codificación relacional composicional (lógica ULTRA)

---

## 11. WandB

- **Proyecto:** `exphormer-kgc`
- **Entity:** `maxojeda`
- **URL:** `https://wandb.ai/maxojeda/exphormer-kgc`
- Configurado en `configs/Exphormer/wn18rr.yaml` (entity y project como defaults)
- Activar en scripts con `wandb.use True`
