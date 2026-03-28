# Session Notes - Exphormer KGC (2026-03-23)

## 1. Estado del experimento en curso

**Job SLURM**: 555602
**Nodo**: compute-gpu-3-1
**Config**: B=16, train_steps_per_epoch=3257 (30% cobertura), dim=32, 6 capas
**Output dir**: `results_b16`
**Log**: `logs/wn18rr_b16_full.txt`

Estimaciones de tiempo:
- Epoch 0: ~32 min
- Ciclo completo (100 epocas): ~53 horas
- Eval cada 2 epocas

Comandos para monitorear:
```bash
# Estado del job
squeue -u mojeda_imfd | grep 555602

# Ver log (binario, usar strings)
strings logs/wn18rr_b16_full.txt | tail -50

# Ver metricas si existen
cat results_b16/0/stats_val.json | tail -5
```

---

## 2. Cambios implementados esta sesion

### A. Expander estatico (kg_dataset.py + trainer.py)

**Antes**: Se generaba un expander random en cada epoca (augmentation dinamica).

**Ahora**: Se genera UNA vez en `_build_from_triples` al cargar el dataset:
- Guardado como `kgc_ds.full_expander_edge_index` (2, E_exp)
- WN18RR: 245,658 aristas de expander (grado 3, ~40K nodos)
- Compartido entre train/val/test splits via `get_shared_state()`

**En trainer.py**: `_tile_expander(base_exp_ei, B, N, device)` replica el expander para B copias del grafo con offsets apropiados.

**Motivacion**: Exphormer original usa expander estatico (generado una vez en preprocesamiento, verificado near-Ramanujan). Esto es diferente de DropEdge (augmentation). El expander es una propiedad estructural fija del grafo.

Codigo relevante (`kg_dataset.py:310-326`):
```python
if getattr(self.cfg.prep, 'exp', False) and num_nodes > 1:
    tmp = Data(num_nodes=num_nodes)
    generate_random_expander(tmp, degree=self.cfg.prep.exp_deg, ...)
    full_expander_edge_index = tmp.expander_edges.t()  # (2, E_exp)
```

### B. ExpanderEdgeFixer (exp_edge_fixer.py)

Se anadio rama para `batch.expander_edge_index` pre-computado (modo full-graph):

- **Modo full-graph**: Si batch tiene `expander_edge_index` (pre-offset global), lo usa directamente
- **Modo subgraph**: Si tiene `expander_edges` (per-graph), usa el path antiguo con `to_data_list()`

El resultado combinado (KG edges + expander) se escribe en `batch.expander_edge_index` para consumo por ExphormerAttention.

### C. Habilitacion de B>1 con per-graph edge masking (trainer.py)

**Antes**: `assert train_bs == 1` - solo 1 query por forward pass.

**Ahora**: B queries por step, cada una con su propio grafo enmascarado.

**Query-edge removal (NBFNet Appendix B)**: Para cada query (h,r,t) se remueven 4 aristas que revelarian la respuesta:
1. `(h->t, r_orig)`: triple original
2. `(t->h, r_orig+nr)`: reverso estructural del original
3. `(t->h, r_orig+bnr)`: triple reciproco
4. `(h->t, r_orig+bnr+nr)`: reverso estructural del reciproco

**Implementacion vectorizada** (sin loop Python, todo en GPU):
```python
# Tilar B copias del full_edge_index con offsets
graph_idx = torch.arange(B, device=device).repeat_interleave(E)  # (B*E,)
src_tiled = full_edge_index[0].repeat(B) + graph_idx * N
dst_tiled = full_edge_index[1].repeat(B) + graph_idx * N

# Mask: check simplificado usando %bnr (los 4 IDs comparten el mismo %bnr)
is_ht = ((src_loc == h_g) & (dst_loc == t_g)) | ((src_loc == t_g) & (dst_loc == h_g))
keep = ~(is_ht & (rel_tiled % bnr == r_g))
```

WN18RR con B=16: 16 x 695K = 11.1M aristas KG procesadas en una operacion de masking.

### D. Fix en kg_dataset.py

Cambio menor: `type('_D', (), ...)()` -> `Data(num_nodes=num_nodes)` para generar el expander temporal correctamente.

---

## 3. Diagnostico de velocidad

### Experimentos corridos y sus tiempos

| Config | steps/epoca | B | Cobertura | min/epoca | Notas |
|--------|-------------|---|-----------|-----------|-------|
| B=1, dim=32, 30% | 52,101 | 1 | 30% | ~69 | Mejor MRR hasta ahora |
| B=1, dim=64, 30% | 52,101 | 1 | 30% | ~111 | 3.6x mas params, similar MRR |
| B=1, dim=32, 100% | 173,670 | 1 | 100% | ~237 | Demasiado lento |
| B=16, dim=32, 100% (loop) | 10,854 | 16 | 100% | ~108 | Loop Python = cuello de botella |
| B=16, dim=32, 100% (vectorized) | 10,854 | 16 | 100% | ~108 | GPU compute = cuello de botella |
| **B=16, dim=32, 30% (actual)** | **3,257** | **16** | **30%** | **~32** | **Job 555602 en curso** |

### Analisis del cuello de botella

Con B=16 y 100% cobertura:
- 16 x 695K = 11.1M aristas KG + 3.93M expander = ~15M aristas por step
- Esto es 4.5x mas computo GPU por epoca que B=1 30%

**Insight clave**: El speedup de B>1 viene de menos iteraciones Python, NO de reducir el computo GPU total. El total de aristas GPU por epoca es constante: `steps x B x E` es aproximadamente igual para misma cobertura.

### Conclusion

B=16 con 30% cobertura es optimo actualmente:
- Mismos datos que B=1 30% (~52K triples por epoca)
- ~2x mas rapido por menos overhead Python
- Mejor utilizacion GPU con tensores mas grandes
- Vectorized masking elimina el loop Python completamente

---

## 4. Mejores resultados historicos (WN18RR)

Del experimento anterior (B=1, dim=32, 30% cobertura, job 548889):
- **Mejor val_mrr = 0.4239** (epoch 18)
- **Mejor test_mrr = 0.4329** (epoch 18)
- Epoch 22: val_mrr = 0.4217 (plateau)

**Baseline NBFNet**: MRR = 0.551 (objetivo a largo plazo)

**Gap actual**: ~0.12 puntos de MRR por debajo de NBFNet.

---

## 5. Arquitectura actual (wn18rr.yaml)

```yaml
# Modelo
gt.layer_type: Exphormer      # Exphormer-only (sin GatedGCN)
gt.layers: 6
gt.dim_hidden: 32
gt.n_heads: 4                 # 8 dims per head
gt.layer_norm: True
gt.dropout: 0.1
gt.use_edge_gating: True      # V gate condicionado en relacion
gt.use_query_conditioning: True  # atencion condicionada en query relation

# Expander
prep.exp: True
prep.exp_deg: 3               # grado del expander
prep.add_edge_index: True     # KG edges + expander en atencion

# Training
kgc.train_batch_size: 16
kgc.train_steps_per_epoch: 3257   # 30% cobertura
kgc.label_smoothing: 0.0

# Optimizacion
optim.base_lr: 0.0002
optim.scheduler: cosine_with_warmup
optim.num_warmup_epochs: 5
optim.max_epoch: 100
optim.clip_grad_norm: True

# Eval
train.eval_period: 2
kgc.eval_batch_size: 4
```

---

## 6. Pendientes / proximos pasos sugeridos

### Inmediato (esta semana)
1. Ver resultados del job 555602 (epoch 0 y primeras epocas)
2. Si MRR comparable al historico (>0.40 en epoca 18): continuar hasta convergencia
3. Si mas lento en convergencia: investigar si B=16 necesita ajuste de LR (gradientes mas estables con B>1 -> podria aumentar lr)

### Corto plazo
4. Implementar splits inductivos Teru et al. (2020) para WN18RR/FB15k-237 (Etapa 1 de tesis)
5. Actualizar `fb15k237.yaml` a arquitectura actual

### Ablaciones pendientes
6. Expander on/off (`prep.exp: True/False`)
7. Numero de virtual nodes (`num_virt_node: 0/1/4`)
8. `add_edge_index` on/off (solo expander vs KG+expander)
9. Numero de capas (3 vs 6 vs 9)

---

## 7. Archivos clave modificados esta sesion

| Archivo | Cambio principal |
|---------|------------------|
| `train/trainer.py` | `_tile_expander()` + vectorized masking B>1 + eliminar `_make_full_graph_expander` |
| `loader/dataset/kg_dataset.py` | Expander estatico en `_build_from_triples` |
| `encoder/exp_edge_fixer.py` | Rama para `expander_edge_index` pre-computado |
| `configs/Exphormer/wn18rr.yaml` | train_batch_size=16, train_steps_per_epoch=3257 |
| `sbatch_wn18rr.sh` | out_dir=results_b16, log=wn18rr_b16_full.txt |

---

## 8. Comandos utiles

```bash
# Monitorear job
squeue -u mojeda_imfd
watch -n 60 'squeue -u mojeda_imfd'

# Ver log del experimento
strings logs/wn18rr_b16_full.txt | tail -100

# Ver metricas de validacion
cat results_b16/0/stats_val.json | python -m json.tool | tail -20

# Cancelar job si necesario
scancel 555602

# Reanudar desde checkpoint (cambiar auto_resume si es necesario)
sbatch sbatch_wn18rr.sh

# Ver uso de GPU en el nodo
ssh compute-gpu-3-1 nvidia-smi
```

---

## 9. Notas de implementacion importantes

### Data leakage fix (sesion anterior)
Durante training de query (h,r,t), la arista directa (h->t,r) estaba presente en `full_edge_index`. El modelo aprendia un shortcut trivial de 1-hop.

**Fix (NBFNet Appendix B)**: per-step, remover 4 aristas que involucran (h,t).

### Filter bug fix (sesion anterior)
Para queries reciprocas `(t_orig, r_inv, h_orig)`, `tail_filter[(t_orig, r_inv)]` solo enmascaraba tails conocidos de training. Val/test-known heads NO estaban enmascarados.

**Fix**: Cuando `r >= base_num_rel`, usar `head_filter[(h, r - base_num_rel)]` en lugar de tail_filter.

### Restricciones de memoria
- `gnn.dim_inner` debe ser igual a `gt.dim_hidden` (assertion en MultiModel)
- `kgc.eval_batch_size: 4` usa ~1.4GB GPU (B=16 causa OOM en eval)
- `grad_checkpoint: True` reduce memoria de ~20GB a ~5GB peak

---

*Documento generado: 2026-03-23*
