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

---

## Plan (actualizado 2026-04-23, sesión 16)

### Contexto y baseline a batir

- **Baseline actual val-selected**: `wn18rr_ind_v1_qcdv_lr3e5.yaml` → val=0.386, test=0.486 @ ep5 (arquitectura query-conditioned DistMult V de sesión 12).
- **Baseline histórico test-selected**: test=0.565 @ ep4 (arquitectura pre-refactor con W_V+gate, no comparable directamente).
- **Transductivo**: 0.566 MRR (job 578765, ep7). Job 592207 corriendo con arquitectura restaurada W_V\*gate.
- **Techo objetivo**: NBFNet 0.741, KnowFormer 0.752.
- **Causa raíz identificada (sesión 13)**: colapso post-ep5 del test MRR — hits@1 cae a ~0 mientras hits@10 se mantiene. Val no lo refleja (entidades del train graph).

### Cambio #1 — Inyección de ruido Gaussiano (PENDIENTE)

**Estado**: pendiente. `fc_v_expand` fue eliminado en sesión 16; la motivación original (simetría en nodos no-anchor con `x=0`) sigue siendo válida con la arquitectura W_V\*gate actual.

**Dónde intervenir**: `encoder/node_encoders.py` → `KGCNodeEncoder.forward`: añadir `x_v = noise_std * torch.randn(...)` para nodos no-anchor durante training. No tocar el anchor (destruiría la boundary condition).

**Flag a agregar**: `cfg.gt.noise_std: 0.0`. Sweep: {0.5, 1.0, 2.0, 4.0}.

**Prerrequisito**: primero resolver la regresión de velocidad 2.7× (ver sesión 17) o correr en 1 GPU para validar el efecto.

### Cambio #2 — LR diferenciado (OBSOLETO)

**Estado**: descartado. `fc_v_expand` fue eliminado en sesión 16 y no existe en el código actual. No aplicable.

### Cambio #3 — Unificación de tablas de relaciones ✅ COMPLETADO (sesión 18)

**Nota**: implementado en sesión 15, revertido en sesión 17 (intento fallido de solucionar regresión de velocidad), reimplementado correctamente en sesión 18.

**Estado final**: **2 tablas** en todo el modelo:

| Tabla | Módulo | Shape | Rol |
|-------|--------|-------|-----|
| `query_rel_emb` | `MultiModel` | `(R, d)`, std=0.01 | única tabla r_q — lookup una vez por forward |
| `emb` | `RelationEmbeddingEncoder` | `(R, d)` | r_uv (features de arista del KG) — separada por diseño |

Cada módulo aplica sus propias proyecciones lineales sobre `batch.query_emb`:
- `KGCNodeEncoder`: usa directamente como boundary condition (`x_anchor = batch.query_emb`)
- `ExpanderEdgeFixer`: aplica `proj_exp_edge` (Linear d→d_edge, std=0.01)
- `ExphormerAttention` × L: aplica `proj_q/k/e/vg` por capa
- `KGCHead`: usa directamente para scoring

**KnowFormer** confirmado: usa exactamente 1 tabla (`query_embedding`) + proyecciones por rol/capa. Diseño idéntico al nuestro.

**Resultado final**: 1 `query_rel_emb(R, d)` + proyecciones aprendidas por rol. Más 1 `edge_rel_emb(R, d)` para r_uv. Igual que KnowFormer.

**Estado**: todos los pasos implementados de una vez. Smoke tests pasan. Ahorro de parámetros verificado exacto. El valor es metodológico (tesis) y de consistencia con KnowFormer/NBFNet.

### Lo que NO hay que hacer

- **NO volver a intentar V bypass** (sesión 14 cerrada: optimizer abre el atajo → colapso).
- **NO volver a variantes V ya probadas**: `use_relational_v`, `use_nbf_v`, `use_distmult_v`, V-RMPNN, PNA con mean — todas con diagnóstico documentado en sesión 12.
- **NO escalar dim/L sin arreglar colapso primero**: `dim=128 L=5` ya probado (0.501 < 0.565).
- **NO experimentos transductivos nuevos** hasta que inductivo supere 0.6 con arquitectura unificada — el transductivo actual (0.566 @ ep7) ya es satisfactorio para tesis-stage-1.

### Orden de ejecución sugerido

1. Implementar flag `gt.noise_std` + modificación en `KGCNodeEncoder.forward`.
2. Smoke test (1 epoch, 4 steps) con `noise_std: 1.0` para verificar que no rompe nada.
3. Crear 4 configs (`wn18rr_ind_v1_qcdv_noise{05,10,20,40}.yaml`) + sbatch scripts.
4. Lanzar los 4 en paralelo (1×H100 cada uno). Tiempo estimado: 4-6h cada uno.
5. Analizar resultados, seleccionar mejor `noise_std`.
6. Si mejora → implementar cambio #2 sobre esa config. Si no mejora → documentar en sesión 15 como descarte, pasar a cambio #3 o reevaluar.

### Archivos a tocar (estimación)

| Archivo | Cambio #1 | Cambio #2 |
|---------|-----------|-----------|
| `encoder/node_encoders.py` | ~8 líneas en `KGCNodeEncoder` | — |
| `config.py` | +1 línea (`gt.noise_std`) | +1 línea (`optim.v_lr_mult`) |
| `train/trainer.py` | — | ~10 líneas en optimizer build |
| `configs/Exphormer/wn18rr_ind_v1_qcdv_noise*.yaml` | 4 nuevos | — |
| `sbatch_wn18rr_ind_v1_qcdv_noise*.sh` | 4 nuevos | — |

### Estado al final de sesión 14 (punto de partida)

- Código: post-sesión 13 (qcdv con fc_v_expand, lr sweep completado). V bypass revertido.
- Config mejor (val-selected): `wn18rr_ind_v1_qcdv_lr3e5.yaml`.
- Sin jobs en cola. Listo para sesión 15.

---

## Estado actual — 2026-04-23 (sesión 16)

### Resultados transductivos — CERRADOS

| Job | Config | Val MRR | Test MRR | Épocas | Estado |
|-----|--------|---------|----------|--------|--------|
| 578765 | transduct_best (viejo, W_V\*gate) | 0.5654 | **0.5661** | ep7 | Terminado, plateau definitivo |
| 581959 | transduct_noexp (viejo, sin expander) | 0.5551 | **0.5551** | ep8 | Terminado, plateau definitivo |
| 591749 | transduct_noqc (actual, sin QC) | 0.1146 | 0.1239 | ep10 | Corriendo — **ROTO** |

**Ablación expander confirmada**: +0.011 MRR (0.566 vs 0.555). Coincide exactamente con `metodologia.tex`. ✓

**Por qué noqc está roto** (diagnóstico sesión 16): con `use_query_conditioning=False`, Q está anclado a `x0`. Para nodos no-anchor `x0=0` → `Q=W_Q(0)=0` (sin bias) en **todas las capas**. Score trilineal `Q⊙K⊙E=0` → `exp(0)=1` → atención uniforme sobre todos los vecinos en el 99.99% de nodos. El mecanismo de Q-anclado-a-x0 **requiere** `proj_q(r_q)` para funcionar; no son disociables. El experimento noqc no mide "efecto del QC" sino "QC removido de una arquitectura que lo requiere". Resultado no interpretable como ablación. **Puede matarse** el job 591749.

### Decisión de arquitectura — reversión a W_V\*gate

**Origen de fc_v_expand** identificado: es matemáticamente idéntico a `W_r·q` de NBFNet (no a KnowFormer, cuyo V-RMPNN es una red NBF de 2 capas completa). El comentario "KnowFormer-style" en el código era incorrecto.

**Decisión**: eliminar `fc_v_expand` y volver a la arquitectura que dio 0.566 transductivo. Razones:
1. El 0.566 ya está documentado en `metodologia.tex` y es el resultado de referencia para tesis.
2. fc_v_expand nunca se corrió en transductivo — no hay datos empíricos de que mejore.
3. La arquitectura W_V\*gate es más simple y reproducible.

### Cambios de código — sesión 16

**`layer/exphormer.py`**:
- Eliminado `QKLayer` class completo.
- Eliminados de `ExphormerAttention`: `fc_v_expand`, `expander_v_weight`, `num_relations_v`, `fc_qk_x`, `fc_qk_z`, `qk_layers`, `qk_noise_std`, `num_qk_layers`.
- Restaurados: `self.V = nn.Linear(...)` (siempre), `self.V_gate`, `self.proj_vg` (cuando `use_query_conditioning=True`).
- `propagate_attention`: reemplazado bloque `z_v_per_graph` por `v_src = V_h[src] * E_gate`.
- `forward`: `V_h = self.V(h)` siempre; E_gate computado como `V_gate(edge_attr) + proj_vg(shared_edge)`.
- Init restaurado: `proj_q`, `proj_k`, `proj_e`, `proj_vg` → `std=0.01` (igual que código 0.566).

**`network/model.py`**: eliminados `qk_noise_std`, `num_qk_layers` de `GlobalModel`, `MultiLayer`, `MultiModel`.

**`config.py`**: eliminados `cfg.gt.qk_noise_std`, `cfg.gt.num_qk_layers`.

**Diferencias que quedan** respecto al 0.566 exacto (consecuencia de la tabla compartida de sesión 15):
- `shared_rel_emb_table` (atención): std=0.01 independiente por capa → std=0.02 compartida.
- `KGCNodeEncoder.rel_emb`: std=1.0 (default Embedding) → std=0.02 (compartida).
- `KGCHead.rel_emb`: std=1.0 → std=0.02.
- `exp_edge_query_emb`: std=0.01 → std=0.02.
- Estas no se pueden igualar sin deshacer la unificación. Con 100 épocas y cosine schedule debería converger igual.

### Refactor KnowFormer-style — lookup centralizado

**Motivación**: aunque la sesión 15 unificó las tablas de relación a un único `nn.Embedding` compartido, el `repr` del modelo mostraba 8 entradas `Embedding(22,64)` (una por cada módulo que tenía un atributo apuntando al mismo objeto). Era confuso, aunque funcionalmente correcto. KnowFormer hace **un solo lookup** en su `forward` top-level (`Knowformer.forward` línea 250: `z = self.query_embedding(r_index)`) y pasa el tensor `z` a las capas. Limpiamos para seguir el mismo patrón.

**Cambios en código**:

1. `MultiModel.forward()`: agregado al inicio:
   ```python
   if hasattr(self, 'query_rel_emb'):
       batch.query_emb = self.query_rel_emb(batch.query_relation)   # (B, dim_hidden), 1 lookup
   ```

2. Eliminados de todos los módulos los atributos que apuntaban al embedding compartido:
   - `KGCNodeEncoder.rel_emb` → ahora usa `batch.query_emb` directamente
   - `ExpanderEdgeFixer.exp_edge_query_emb` → indexa `batch.query_emb[graph_idx]`
   - `ExphormerAttention.shared_rel_emb_table` → indexa `batch.query_emb[batch.batch]` (per-node) y `batch.query_emb[edge_graph]` (per-edge)
   - `KGCHead.rel_emb` → usa `batch.query_emb`

3. Eliminados parámetros `shared_query_emb` de la cadena de constructores (`FeatureEncoder`, `GlobalModel`, `MultiLayer`, `build_node_encoder`, `build_head`, `ExphormerAttention`, `ExpanderEdgeFixer`, `KGCNodeEncoder`, `KGCHead`).

4. Eliminado `num_relations` de `ExphormerAttention.__init__` (ya no se necesita; el embedding vive solo en MultiModel).

**Resultado del refactor (smoke test)**:
- params: **271,489** (idéntico — el refactor es organizacional, no cambia params).
- val/test MRR @ ep0: idénticos al pre-refactor (mismas dinámicas de entrenamiento).
- En el `repr` del modelo ahora aparecen solo **2 entradas** `Embedding(22,64)`: `MultiModel.query_rel_emb` (query relations) y `RelationEmbeddingEncoder.emb` (edge relations `r_uv`, separada por diseño). Los demás módulos ya no tienen atributos de embedding.

**Equivalencia matemática**: `embedding(idx)` y `embedding_output[idx]` (donde `embedding_output = embedding(all_indices)`) son operaciones idénticas en términos de gradientes y outputs. PyTorch autotrack la cadena gradiente desde cada uso indirecto hasta el `nn.Parameter` original.

### Job lanzado — 592218

Config: `wn18rr_transduct_best.yaml` (L=5, d=64, n_heads=4, exp=True, use_query_conditioning=True).
Arquitectura: W_V\*gate + tabla compartida + lookup centralizado en `MultiModel.forward`.
Params: **271,489** (idéntico al pre-refactor; vs 281,345 original con 7 tablas independientes).
Log: `logs/wn18rr_transduct_best_592218.out`.
GPUs: 4×H100. ~50 min/época. Resultado esperado en ~6h.

**Job 592207 cancelado** antes del refactor (no completó épocas significativas).

### Lo que NO hay que hacer

- **NO correr `best.yaml actual (fc_v_expand)`** — se decidió no continuar con esa arquitectura.
- **NO reinstaurar fc_v_expand** — fue eliminado en esta sesión.
- **NO interpretar noqc como ablación** — el resultado (~0.12 MRR) es consecuencia de un diseño incompatible, no del efecto del QC.

---

## Estado actual — 2026-04-23 (sesión 15)

### Limpieza de código — COMPLETADA

Eliminados `tie_rel_emb` e `inductive_routing` de todo el codebase: `layer/exphormer.py`, `network/model.py`, `config.py`, todos los YAMLs en `configs/Exphormer/`, `CLAUDE.md`, `SESSION_NOTES.md`. También eliminados los archivos raíz muertos `exphormer.py` y `model.py` (nunca importados por `main.py`). Resultado: 0 ocurrencias de esos flags en todo el proyecto.

### Clarificación arquitecturas transductivo — IMPORTANTE

Análisis de sesión 15 reveló que el MRR 0.566 (job 578765) **no puede compararse directamente** con ningún experimento de la arquitectura actual. El viejo job usaba código pre-sesión-12 donde `use_query_conditioning: True` activaba W_V\*gate (no fc_v_expand). Ese mecanismo gate fue eliminado en sesión 12 y no existe en el código actual.

Las tres arquitecturas comparables en transductivo:

| Experimento | Q/K cond | V mecanismo | FiLM | Params | MRR |
|---|---|---|---|---|---|
| Job 578765 (viejo, código s10) | ✓ proj_q/k | W_V \* gate | ✓ | 281K | **0.566** |
| noqc (job 591749, código actual) | ✗ | W_V estándar | ✗ | 172K | corriendo |
| best.yaml actual (código actual) | ✓ proj_q/k | fc_v_expand | ✓ | 661K | nunca corrido |

**Ablación FiLM (job 591253) — CANCELADA**: el diseño era incorrecto. El "baseline con FiLM" (0.566) era el modelo viejo con W_V\*gate; el "nofilme" usaba la nueva arquitectura con fc_v_expand. No era una ablación limpia de proj_e — eran dos arquitecturas distintas. Se descartó.

### Experimento noqc — LANZADO (job 591749)

Config `wn18rr_transduct_noqc.yaml`: idéntico a `wn18rr_transduct_best.yaml` pero con `use_query_conditioning: False`. Elimina fc_v_expand, proj_q/k, shared_rel_emb_table y usa V estándar (`W_V(h)`). Es el Exphormer puro con KGC encoding (boundary condition + relation edge features), sin ningún condicionamiento de query en la atención.

Params: 172K. Smoke test: exit 0. Log: `logs/wn18rr_transduct_noqc_591749.out`.

**Qué mide**: cuánto aporta cualquier query conditioning en transductivo. Si da cerca de 0.566, el query conditioning en atención no ayuda. Si da bastante menos, el viejo Q/K+gate (281K) sí era útil.

**Pendiente para completar el cuadro**: lanzar `wn18rr_transduct_best.yaml` con código actual (Q/K cond + fc_v_expand + FiLM, 661K) para aislar si fc_v_expand mejora respecto al viejo gate.

### Unificación de tablas de relaciones — IMPLEMENTADO

Análisis completo de las 4+L tablas de embeddings de relación en el modelo vs KnowFormer/NBFNet. Conclusión: tablas 1, 3, 4×L, 5 consumen r_q (misma señal, roles distintos) y deben compartir una única tabla base con proyecciones por rol/capa. Tabla 2 (r_uv, estructural) mantiene la suya.

**Implementación**: `MultiModel` crea `self.query_rel_emb = nn.Embedding(R, d)` y lo pasa a todos los módulos como `shared_query_emb`. Cada módulo lo asigna a su atributo local (`self.rel_emb`, `self.shared_rel_emb_table`, `self.exp_edge_query_emb`) — mismo objeto `nn.Embedding`. Las proyecciones por capa (`proj_q`, `proj_k`, `proj_e`, `fc_v_expand`) quedan per-layer, preservando expresividad.

**Archivos modificados**: `network/model.py` (MultiModel, FeatureEncoder, GlobalModel, MultiLayer), `encoder/node_encoders.py`, `encoder/exp_edge_fixer.py`, `layer/exphormer.py`, `network/heads.py`.

**Ahorro de parámetros verificado por smoke test**:

| Setting | Antes | Después | Ahorro |
|---------|-------|---------|--------|
| Transductivo (L=5, R=22, d=64) | 670,785 | 660,929 | −9,856 = 7×22×64 ✓ |
| Inductivo v1 (L=3, R=18, d=64) | 584,257 | 578,497 | −5,760 = 5×18×64 ✓ |

Ambos smoke tests: exit 0. Backward compat: si `shared_query_emb=None` (non-KGC), cada módulo crea su propia tabla como antes.

---

## Estado actual — 2026-04-20 (sesión 14)

### V bypass inspirado en KnowFormer — DESCARTADO

**Motivación**: análisis de `Knowformer/src/model.py` reveló que KnowFormer usa un término de bypass en su atención linear (`numerator = ... + v*num_node`, líneas 146-185). Esto permite que `V` fluya directamente al output incluso si la atención es débil — útil como "safety net" cuando las queries/keys son ruidosas. Se implementó como intervención unificada (aditiva, no debería regresionar transductivo).

**Cambios aplicados (luego revertidos)**:
- `layer/exphormer.py`: agregado `use_v_bypass` flag, `bypass_scale = nn.Parameter(torch.zeros(1))` init=0, y `h_out = batch.wV + bypass_scale * batch.V_h`.
- `network/model.py`: propagación del flag por `GlobalModel` → `MultiLayer` → `MultiModel`.
- `config.py`: `cfg.gt.use_v_bypass = False` (default).
- Nuevo config `wn18rr_ind_v1_vbypass.yaml` + sbatch correspondiente.

**Intento 1 (sin escalar, `h_out = wV + V_h`)**: ep3 val=0.235 / test=0.295, luego colapso total. Causa: el V bypass duplica la señal con el outer residual de `GlobalModel.forward` (`h_attn = h_in1 + h_attn`) → `h_attn = 2*h_in1 + wV`. Magnitud explota 2^L.

**Intento 2 (con escalar learnable, init=0)**: job 583056 → bloqueó el breakthrough del baseline.

| Métrica | vbypass (ep4 best) | qcdv_lr3e5 baseline (ep5) |
|---------|---------------------|----------------------------|
| val MRR | 0.239 | **0.386** |
| test MRR | 0.294 | **0.486** |

Después de ep4, vbypass colapsa a val~0.03 / test~0.06 (random-level).

**Diagnóstico**: aunque `bypass_scale` init=0 garantiza ep0 idéntico al baseline, el optimizer abre el bypass inmediatamente. `V_h = raw_h` pasa sin filtrar — no tiene el preprocesado relacional de KnowFormer (que tiene un stream V-RMPNN separado con 2 capas NBF internas antes del bypass). En nuestra arquitectura de stream único, el gradiente prefiere el atajo del bypass antes que aprender el mecanismo de atención complejo. Se rompe el refinamiento iterativo de representaciones.

**Lección**: no todo cambio aditivo de KnowFormer transfiere. El bypass es específico de su arquitectura dual-stream (Q-RMPNN + V-RMPNN), no del mecanismo de atención en sí. Intervenciones unificadas deben respetar la topología de flujo de señal de la arquitectura existente, no solo preservar ep0.

**Estado del código**: todas las modificaciones de V bypass revertidas. Config y sbatch eliminados. Código vuelto al estado post-sesión 13 (qcdv con lr sweep).

### Próxima dirección propuesta

Siguiente intervención unificada: **inyección de ruido Gaussiano** para symmetry-breaking al embedding inicial (análogo a `qk_x = torch.zeros(B,N,1).normal_(0,4)` en KnowFormer, línea ~193). Es aditiva y no rompe transductivo. Ataca el colapso post-ep3 sin competir con el mecanismo de atención.

---

## Estado actual — 2026-04-17 (sesión 13)

### LR sweep sobre arquitectura qcdv — COMPLETADO

Config base: `wn18rr_ind_v1_qcdv_lr3e5.yaml` (L=3, dim=64, n_heads=4, no expander, cosine_with_warmup, warmup=5).

| LR | Job | Mejor val ckpt (ep) | val MRR | test MRR | test H@10 | Comportamiento |
|----|-----|---------------------|---------|----------|-----------|----------------|
| 1e-5 | 581966 | ep11 | 0.371 | 0.496 | ~0.73 | estable, no colapsa |
| **3e-5** | 581967 | **ep5** | **0.386** | **0.486** | **0.753** | estable, decline gradual |
| 1e-4 | 581968 | ep2 | 0.374 | 0.497 | 0.753 | test colapsa desde ep6 (hits@1→0) |
| 3e-4 | 581969 | ep1 | 0.358 | 0.483 | 0.758 | colapsa desde ep4 |

**Observaciones clave:**

1. **Colapso aún presente a LR alto**: La nueva arquitectura NO eliminó el colapso — lo ralentizó. A lr≥1e-4 el test MRR (no val) colapsa igualmente. El val no lo refleja porque usa entidades del train graph.

2. **Patrón de colapso del test a lr=1e-4**: hits@1 cae a ~0 desde ep6 mientras H@10 se mantiene (~0.70). El modelo aprende a poner a la respuesta correcta en el top-10 pero no en el top-1 — score casi uniforme.

3. **Mejor LR**: lr=3e-5 da mejor val (0.386) y comportamiento más estable. El test MRR al pico val es 0.486, pero el pico test observado fue 0.486@ep5.

4. **Techo actual (val-selected)**: ~val=0.386, test=0.486 — por debajo del mejor histórico pre-refactor (val=0.415→ test=0.565, test-selected).

5. **T=4 lanzado**: job 581970, config `wn18rr_ind_v1_qcdv_lr3e5_L4.yaml`, log: `logs/wn18rr_ind_v1_qcdv_lr3e5_L4_581970.out`

### HIPÓTESIS PARA PRÓXIMA SESIÓN: inicialización de fc_v_expand

**Problema identificado**: `fc_v_expand` se inicializa con `std=0.01` → arranca casi en cero. Esto significa que durante las primeras épocas, el nuevo V relacional (`h ⊙ z_r`) produce señal casi nula. Durante ese tiempo, el modelo sigue dependiendo de Q/K para aprender, y Q/K son entity-specific → memorizan topología del train graph antes de que el V relacional se active.

Cuando el V relacional finalmente escala (gradientes lo empujan hacia valores mayores), ya es tarde: Q/K han convergido a patrones específicos del train graph que no transfieren al test graph inductivo.

**Direcciones a explorar (en orden de prioridad):**

1. **Inicialización ortogonal / Kaiming de fc_v_expand**: reemplazar `nn.init.normal_(std=0.01)` por `nn.init.kaiming_uniform_` o `nn.init.orthogonal_`. Fuerza a que el V relacional contribuya desde época 0, compitiendo con Q/K desde el inicio. Riesgo bajo, 1 línea.

2. **LR diferenciado para fc_v_expand**: usar un optimizer con param groups — `fc_v_expand` con LR 10× mayor que el resto. Fuerza convergencia rápida del V sin destabilizar Q/K/E. ~5 líneas en `train/trainer.py`.

3. **Warmup específico del V**: congelar Q/K los primeros N epochs (o ponerles LR=0) mientras fc_v_expand aprende. Una vez V activo, descongelar Q/K. Riesgo: Q/K necesitan ver señal de V para aprender routing útil.

4. **Init fc_v_expand ≈ identidad relacional**: para cada relación `r`, inicializar `z_r ≈ ones` (de modo que `h ⊙ z_r ≈ h` al inicio, como el viejo W_V sin transformación). Preserva el flujo de gradiente desde época 0 mientras aprende la dependencia relacional. Implementación: inicializar el output de fc_v_expand en ones → `nn.init.constant_` + normalización.

**Línea base para comparar**: qcdv con lr=3e-5 da val=0.386, test=0.486 @ ep5. Cualquier variante de init debe superar esto.

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

### WN18RR transductivo (job 578765) — COMPLETADO ep40+

| Época | val MRR | test MRR | LR | Nota |
|-------|---------|----------|----|------|
| 7 | **0.565** | **0.566** | ~0.00080 | MEJOR ckpt |
| 39 | 0.554 | 0.555 | cosine decay | |
| 40 | 0.553 | 0.554 | cosine decay | sin mejora desde ep7 |

Best checkpoint definitivo: ep7 → val=0.565, test=**0.566**. Plateau claro desde ep7 → el modelo no mejora con más entrenamiento en la config actual.

### Experimento sin expander transductivo — LANZADO (job 581959)

Config `wn18rr_transduct_noexp.yaml`: idéntica a `wn18rr_transduct_best.yaml` excepto `prep.exp: False`, `exp_deg: 0`, `max_epoch: 15`, `--time=24:00:00`.
Objetivo: comparar MRR con vs sin expander en setting transductivo.

---

### ANÁLISIS RAÍZ: Por qué necesitamos cambiar el mecanismo de V (sesión 12)

**Pregunta**: ¿por qué KnowFormer y NBFNet usan la misma arquitectura para transductivo e inductivo y nosotros no?

**Respuesta**: la diferencia está en el mecanismo de mensaje, no en flags.

#### Nuestro modelo vs KnowFormer — dónde entra la relación

| Modelo | Mensaje V | Propiedad |
|--------|-----------|-----------|
| KnowFormer | `msg = h[u] ⊙ z_r[r_uv](r_q)` | Filtro relacional **antes** de agregar |
| Nuestro | `msg = W_V(h[u]) * gate(r_uv)` | Filtro relacional **después** de proyección genérica |

`z_r[r_uv](r_q) = fc_z(query_emb[r_q])[r_uv]` — función de AMBAS `r_uv` Y `r_q`. Cruce de (arista × query).

`gate(r_uv)` — solo función de `r_uv`. Sin cruce con query.

#### Por qué W_V(h) no transfiere entre grafos

`h[u]` acumula información de los vecinos específicos de `u` en el training graph. `W_V` aprende a proyectar esa distribución de entidades a direcciones útiles en el espacio de salida. En el test graph inductivo, las mismas dimensiones de `h[u]` contienen entidades completamente distintas → `W_V` mapea a las mismas direcciones que aprendió del train graph → inducción fallida.

En KnowFormer, `h[u]` a la capa `l` = resultado de RMPNN partiendo de **ceros** (o ruido aleatorio para QK). Siempre expresa "qué tipos de caminos relacionales llevan a u para query r_q" — misma distribución en train y test → FFN transfiere.

#### Por qué la arquitectura anterior no transfería entre settings

La arquitectura con W_V(h) + gate aprende representaciones específicas del grafo de entrenamiento que no transfieren al grafo de prueba inductivo. KnowFormer no tiene este problema porque su V-RMPNN siempre parte de cero y expresa caminos relacionales, no identidades de entidades.

#### Por qué todos nuestros intentos de V relacional fallaron

| Intento | Problema |
|---------|----------|
| `use_relational_v`: `V *= rel_emb[r_uv]` | Dos señales multiplicativas competitivas (V + gate) |
| `use_nbf_v`: `V = h ⊙ W_r_standalone[r_uv]` | `W_r` solo depende de `r_uv`, no de `r_q` |
| `use_distmult_v`: `V = W_V(h) ⊙ W_r_standalone[r_uv]` | Mismo problema: sin cruce `(r_uv × r_q)` |

Nunca tuvimos `z_r = f(r_uv, r_q)`. Siempre fue `f(r_uv)` solo o `f(r_q)` solo.

#### Solución: Query-conditioned DistMult V (KnowFormer-style)

Reemplazar `V = W_V(h)` por `V = h ⊙ z_r[r_uv](r_q)` donde:

```python
z_full = fc_v_expand(shared_rel_emb[r_q])  # (N, R * d) — función de query
z_full = z_full.view(N, R, num_heads, out_dim)
# per-edge:
z_r_per_edge = z_full[batch_per_edge, edge_rel_idx]   # (E, heads, out_dim)
v_src = h[src] * z_r_per_edge                          # DistMult: h ⊙ f(r_uv, r_q)
```

Sin gate separado (el gate queda incorporado en `z_r`). K entity-specific se preserva. FFN intacto.

**¿Afecta a lo transductivo?**
- K no cambia → routing entidad-específico preservado ✅
- V pierde W_V (mezcla cross-dimensional) pero el FFN compensa ✅
- V gana cruce `(r_uv × r_q)` — más informativo ✅
- KnowFormer usa exactamente este V y logra MRR=0.594 transductivo (vs nuestro 0.566) ✅
- Veredicto: transductivo se mantiene o mejora

**Parámetros nuevos**: `fc_v_expand: d → R*d` — para WN18RR inductivo (R=18, d=64): ~73K params nuevos vs ~5K del W_V + gate actual. Aceptable.

### IMPLEMENTADO (sesión 12, 2026-04-17)

Cambios en código:
- `layer/exphormer.py`: reescrito completamente. Eliminados `use_edge_gating`, `gate_rel_mult`, `use_nbf_v`, `use_distmult_v`, `use_rel_matrix_v`, `use_pna`, `use_alpha_mix_qk`. Nuevo V: `fc_v_expand` (query-conditioned DistMult) siempre activo cuando `use_query_conditioning=True`. Expander edges usan `expander_v_weight` (learnable, init=ones).
- `network/model.py`: eliminados `VRMPNN`, `GlobalModel`/`MultiLayer`/`MultiModel` simplificados. FFN siempre 2-layer. Eliminados `use_film_ffn`, `ffn_type`, todos los flags muertos.
- `config.py`: solo queda `use_query_conditioning`.
- Configs YAML: limpiados de flags muertos.

Params nuevos (fc_v_expand por capa): `in_dim × (R × heads × out_dim)` = 64 × (18×64) = 73K por capa (inductivo v1).
- wn18rr_ind_v1 (3L, R=18): 353K params (vs ~120K anterior)
- wn18rr_transduct (5L, R=22): 670K params (vs ~281K anterior)

Smoke test: ambos settings corren sin errores ✅

**Pendiente**: lanzar experimento real con nueva arquitectura.

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

**Job 575904 (sin tanh, mismo MRR cero)**: el tanh NO era la causa raíz. Causa real: el config transductivo tenía **K configurado como función solo de la query** (sin la componente entity-specific W_K(h)), y el FFN estaba deshabilitado. En transductivo ambas cosas son esenciales: K entity-specific diferencia fuentes por sus representaciones acumuladas, y el FFN aprende transformaciones per-entidad. Sin ellos, los scores colapsan a función solo del tipo de relación → gradientes diluidísimos sobre 40K entidades.

### Fix aplicado

`configs/Exphormer/wn18rr_transduct_best.yaml`: K restaurado a W_K(h) + proj_k(r_q) (entity-specific), FFN estándar restaurado, `eval_period: 1`.

Checkpoint incompatible del run anterior eliminado (`results/wn18rr_transduct_best/0/ckpt.pt`).

**Job 578765** lanzado — log: `logs/wn18rr_transduct_best_578765.out`

Config activo:
```yaml
gt:
  layers: 5, dim_hidden: 64, use_query_conditioning: True
optim:
  base_lr: 0.0008, scheduler: cosine_with_warmup, num_warmup_epochs: 5
```

### Lección clave sesión 10

El config transductivo requiere K = W_K(h) + proj_k(r_q) (routing entity-specific) y FFN activo. Sin la componente W_K(h) en K, el routing colapsa a función solo de la query → MRR ≈ 0.0003 en transductivo.
