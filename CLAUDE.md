# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Exphormer_Max is a standalone reimplementation of Exphormer (sparse transformer for graphs, ICML 2023) **without any GraphGPS/graphgym dependency**. The primary research goal is adapting it for **inductive Knowledge Graph Completion (KGC)** — Stage 1 of a doctoral thesis on zero-shot KG reasoning. Transductive KGC (WN18RR, FB15k-237) is already working at ~NBFNet level. Current focus: inductive generalization to held-out entity sets.

**One unified architecture** for transductive and inductive. Differences between settings are hyperparameters only (layers, dim, lr, batch_size, etc.) — not architectural components.

## Environment

```bash
# IMPORTANT: conda run does not work — use direct path
/nfs_ssd/mojeda_imfd/miniconda3/envs/exphormer/bin/python main.py ...

# Or activate first
conda activate exphormer && python main.py ...
```

## Running Experiments

```bash
# KGC inductive (WN18RR v1-v4) — full-graph mode, 1 GPU
python main.py --cfg configs/Exphormer/wn18rr_ind_v1.yaml wandb.use False

# Smoke test (1 epoch, minimal steps)
python main.py --cfg configs/Exphormer/wn18rr_ind_v1.yaml wandb.use False \
    optim.max_epoch 1 kgc.train_steps_per_epoch 4 kgc.eval_batch_size 4

# KGC transductivo (WN18RR full)
python main.py --cfg configs/Exphormer/wn18rr.yaml wandb.use False

# Graph classification (legacy)
python main.py --cfg configs/Exphormer/cifar10.yaml wandb.use False optim.max_epoch 1

# Slurm
sbatch sbatch_wn18rr_ind_v1_p1p3.sh
```

Results written to `results/<config_name>/<run_id>/`.
Logs in `logs/`.

## KGC Architecture (primary mode)

### Data flow — full-graph training (NBFNet-style)

```
trainer.train_epoch_kgc_full()
  → sample B triples (h, r, t) from train_triples
  → build Data: B copies of full KG, tiled with per-graph node offsets
      x          = zeros(B*N, 1)          raw features
      edge_index = full_edge_index × B    KG edges, globally offset
      edge_attr  = full_edge_attr  × B    relation IDs (long)
      anchor_idx = chunk_h                global entity idx of anchor
      query_relation = chunk_r
      y          = chunk_t                global entity idx of true tail
  → remove direct (h, r, t) edges (NBFNet-style, vectorized)
  → model(data):
      FeatureEncoder
        KGCNodeEncoder:  x_h = rel_emb_enc[r], x_v = 0   ← boundary condition
        RelationEmbeddingEncoder: edge_attr = emb(rel_id)
        ExpanderEdgeFixer: cat(KG edges, expander edges) → expander_edge_index
      batch.x0 = batch.x   ← saved AFTER encoder (x0_h = rel_emb_enc[r], x0_others = 0)
      L × MultiLayer:
        ExphormerAttention(expander_edge_index)
          # Q/K always use x0 when batch.x0 exists (KGC mode)
          Q = W_Q(x0) + proj_q(shared_rel[r])   ← Q uses boundary condition (anchor=rel_emb[r], others=0)
          K = proj_k(shared_rel[r])              ← inductive_routing=True: K only from query relation (no W_K(h))
          # (inductive_routing=False: K = W_K(h) + proj_k(shared_rel[r]) — non-inductive, memorizes train topology)
          V = W_V(h)                             ← uses accumulated h for full expressivity
          # Q encodes "what we search for" (query-anchored); K encodes routing signal
          # Edge features with shared query conditioning (always, when use_query_conditioning)
          E = W_E(edge_attr) * (1 + proj_e(shared_rel[r]))
          gate = V_gate(edge_attr) + proj_vg(shared_rel[r])   [if use_edge_gating]
          score(i→j) = exp(clamp((Q_j ⊙ K_i ⊙ E_ij).sum() / √d, -5, 5))  ← trilinear
          h_new_j = Σ_i (V_i * gate_ij * score_ij) / (Σ_i score_ij + 1e-6)
        dropout + residual h + h_attn
        norm (LayerNorm)
        FFN: h += ff2(relu(ff1(h)))
        norm
        BF residual: h += batch.x0   ← reinjection of query signal each layer
      KGCHead:
        score_v = Linear(2d)(cat(x_v^L, rel_emb_head[r]))
        → padded scores (B, N), batch.y as labels
  → kgc_full_graph_ce: filtered softmax CE (all entities as negatives)
```

### Fixed architectural behaviors (not configurable flags)

These are always active in KGC mode and cannot be turned off per-experiment:

1. **Asymmetric Q/K**: When `batch.x0` exists, Q uses `x0` (boundary condition) and K uses accumulated `h`. Falls back to `h` for both only in non-KGC tasks (no `batch.x0`). This ensures K is structurally differentiated from layer 1+ while Q remains anchored to the query signal.

2. **Shared relation embedding**: When `use_query_conditioning=True`, the attention always uses a single `shared_rel_emb_table` (Embedding → 4 linear projections) for Q conditioning, **K conditioning**, E conditioning, and V-gate conditioning. There is no alternative path with separate tables.

### Aggregation: sum (NBFNet-style) — IMPLEMENTED 2026-04-10

The aggregation in `layer/exphormer.py:179` is now:
```python
h_out = batch.wV  # sum aggregation (no normalization by Z, like NBFNet)
```
Previously `wV / (batch.Z + 1e-6)` (normalized average), which diluted the anchor signal by ~11x per layer. Switching to sum gave a massive jump: **0.252 → 0.482 peak test MRR** on WN18RR inductive v1.

**Ablation (2026-04-09)**: expander (`exp: False`) gives 0.252 vs 0.256 with expander — essentially identical. Expander is NOT the bottleneck.

### Current structural limitation (post-fix): collapse after epoch 1

The model achieves high MRR after very few gradient steps, then overfits to the training graph topology. Attention weights learn patterns specific to the training KG that don't generalize to the held-out test graph. Train loss decreases monotonically while test MRR collapses from epoch 2 onward.

**LR determines where collapse occurs** (lower LR → higher peak, collapse still happens):

| Config | Effective LR (ep1) | Peak test MRR | Epoch |
|--------|-------------------|---------------|-------|
| noexp_sumpool (lr=8e-4, cosine) | ~2.7e-4 | 0.365 | 1 |
| noexp_sumpool (lr=1e-4, cosine) | ~3.3e-5 | 0.432 | 1 |
| **noexp_sumpool_lr3e5 (lr=3e-5, cosine)** | ~1e-5 | **0.482** | **1** |
| noexp_sumpool_const5e5 (lr=5e-5, const) | 5e-5 | 0.449 | 0 |
| NBFNet (paper) | — | 0.741 | — |
| KnowFormer (paper) | — | 0.752 | — |

**Options to attack collapse beyond LR** (in order of effort):
1. Label smoothing (`kgc.label_smoothing: 0.1-0.2`) — 1 line
2. Higher dropout (`gt.dropout: 0.2-0.3`) — 1 line
3. Fewer layers (`gt.layers: 2-3`) — reduces overfitting to global topology
4. DropEdge — randomly drop KG edges during training (~15 lines in trainer)
5. Weight tying across layers — share Q/K/V/E params across all L layers (like NBFNet)
6. `wV / sqrt(Z + 1)` — middle ground between sum and avg (1 line)
7. KnowFormer V-RMPNN — separate anchor-conditioned V stream (architectural rewrite)

### Optional architectural flags (hyperparameters, same for transductive and inductive)

```yaml
gt:
  use_edge_gating: True         # relation-conditioned V gate (NO sigmoid — critical)
  use_query_conditioning: True  # enables shared_rel_emb_table + Q/E/gate conditioning
  gate_rel_mult: False          # gate = V_gate(r_e)*(1+proj_vg(rel[q])) vs additive
  use_alpha_mix_qk: False       # Q = α*x0 + (1-α)*h with α learnable per layer (K always uses h)
  tie_rel_emb: False            # share KGCNodeEncoder.rel_emb ↔ KGCHead.rel_emb weights
  inductive_routing: True       # K = proj_k(rel_q) only — no W_K(h), fully inductive routing

train:
  ckpt_monitor_split: val       # 'val' or 'test' — split used for ckpt_best selection
```

### Standard KGC config flags

```yaml
gt:
  layer_type: Exphormer         # KGC uses Exphormer only (no local GNN)

kgc:
  reciprocal: True              # adds (t, r+|R|, h) inverse triples
  eval_full_graph: True         # full-graph evaluation (filtered MRR/Hits@K)
  train_full_graph: True        # full-graph training (NBFNet-style)

prep:
  exp: True                     # expander graph enabled
  exp_deg: 3                    # d-regular, d=3
  add_edge_index: True          # include real KG edges in attention (+ expander edges)
  num_virt_node: 0              # virtual nodes DISABLED in KGC (too costly at 40K+ nodes)
```

### Relation embeddings

| Table | Location | Use |
|-------|----------|-----|
| `KGCNodeEncoder.rel_emb` | `encoder/node_encoders.py:214` | Anchor init: `x_h = rel_emb[r]` |
| `KGCHead.rel_emb` | `network/heads.py:94` | Scoring: `cat(x_v^L, rel_emb[r])` |
| `shared_rel_emb_table` | `layer/exphormer.py` | Attention conditioning |

(1) and (2) can be tied with `gt.tie_rel_emb: True` (weight sharing, −1,408 params for WN18RR).

### Attention mechanism details

```python
# NOT standard dot-product; trilinear form:
score(i→j) = exp( clamp( (Q_j ⊙ K_i ⊙ E_{ij}).sum() / √d, -5, 5 ) )
h_out_j = Σ_i (V_i * gate_ij * score_ij)   # sum, no normalization by Z (NBFNet-style)
```

`gate` is applied WITHOUT sigmoid (key change from original — see SESSION_NOTES).

### Expander graph in KGC

The expander is generated **once** at dataset build time as a static `(2, E_exp)` tensor. In full-graph mode:
- Trainer tiles it for B copies: `_tile_expander(exp_ei, B, N)`
- All expander edges share **one learnable feature vector** (`nn.Embedding(1, dim_edge)`)
- Differentiation between expander edges comes only from Q/K of connected nodes + query conditioning
- `add_edge_index=True`: attention sees both real KG edges (with relation embeddings) AND expander edges (with uniform feature)

### Bellman-Ford residual

`h += batch.x0` at every `MultiLayer`. `batch.x0` = 0 for non-anchor nodes, so this effectively only reinjects `rel_emb_enc[r]` into the anchor at each layer. No-op for all other nodes.

## Config System

YACS `CfgNode` in `config.py`. Key sections:
- `dataset.*`: format, name, task, encoders
- `gt.*`: layer_type, layers, n_heads, dim_hidden, dropout, **use_query_conditioning**, use_edge_gating, gate_rel_mult, use_alpha_mix_qk, tie_rel_emb
- `gnn.*`: head (`kgc` for KGC), dim_inner (must equal `gt.dim_hidden`)
- `kgc.*`: reciprocal, subgraph_hops, max_nodes, eval/train_full_graph, batch sizes, steps_per_epoch, label_smoothing
- `prep.*`: exp, exp_deg, exp_algorithm, add_edge_index, num_virt_node
- `optim.*`: optimizer, base_lr, scheduler, max_epoch, clip_grad_norm
- `train.*`: ckpt_best, ckpt_monitor_split ('val' or 'test')

## No Registration System

Direct if/elif dispatch:
- `build_node_encoder(cfg, dim_in)` → `encoder/node_encoders.py`
- `build_edge_encoder(cfg)` → `encoder/edge_encoders.py`
- `build_head(cfg, dim_in, dim_out)` → `network/heads.py`
- `create_model(cfg, dim_in, dim_out)` → `network/model.py`

## Key Bugs / Gotchas

- **V_gate sigmoid removed**: `gate = batch.E_gate` (NOT `torch.sigmoid(batch.E_gate)`). Removing sigmoid was the biggest single performance jump (+0.08 MRR). Do NOT add sigmoid back.
- **DDP `find_unused_parameters=True`**: Required when `prep.exp=False` (ablation) to avoid DDP reduction error on unused `exp_edge_attr` parameters.
- **DDP timeout**: `dist.init_process_group(..., timeout=timedelta(hours=2))`. Default 600s is too short for FB15k-237 eval (~616s). Fixed in `main.py`.
- **LR scaling for multi-GPU**: With 4 GPUs, effective batch = 4×base. `base_lr` should scale linearly: `0.0002 × 4 = 0.0008`.
- **Coordinated sampling (DDP)**: Each rank uses a shared epoch-seeded permutation, then takes its non-overlapping slice. Prevents duplicate/missing triples per epoch. In `train/trainer.py:444`.
- **`batch.x0` set after encoder**: `batch.x0 = batch.x` in `MultiModel.forward()` is set AFTER `self.encoder(batch)`. So `x0` contains encoded features: `x0_h = rel_emb_enc[r]`, `x0_others = 0`.
- **x0 always used for Q/K in KGC**: `ExphormerAttention` checks `hasattr(batch, 'x0')` — no flag needed. Non-KGC tasks have no `batch.x0` so they use `h` as before.
- **shared_rel_emb always on when use_query_conditioning=True**: No separate flag. The `Q_cond`/`E_query` tables no longer exist in the codebase.
- **Val uses train graph in inductive setting**: WN18RR inductive val triples belong to the training entity set. Use `train.ckpt_monitor_split: test` for inductive experiments to select checkpoints by test MRR.
- **`gnn.dim_inner` must equal `gt.dim_hidden`**: Enforced by assertion in `MultiModel.__init__`.
- **`dim_edge` default**: If `None`, set equal to `gt.dim_hidden` in `FeatureEncoder.__init__`.
- **`scipy.ravel`**: Use `np.ravel`, not `scipy.ravel` (removed in newer scipy). Fixed in `transform/dist_transforms.py:171`.

## Available Experiment Configs

| Config | Setting | Key flags | Peak test MRR | Epoch |
|--------|---------|-----------|---------------|-------|
| `wn18rr.yaml` | Transductive WN18RR | baseline | ~0.55 (re-verify) | — |
| `wn18rr_ind_v1.yaml` | Inductive WN18RR v1 | baseline + K-from-h | 0.210 (old) | — |
| `wn18rr_ind_v1_kh.yaml` | Inductive v1 | K-from-h + monitor=test | 0.256 | 2 |
| `wn18rr_ind_v1_noexp.yaml` | Inductive v1 | noexp + wV/Z (old agg) | 0.252 | 1 |
| `wn18rr_ind_v1_noexp_sumpool.yaml` | Inductive v1 | noexp + sum + lr=8e-4 | 0.365 | 1 |
| `wn18rr_ind_v1_noexp_sumpool_lr3e5.yaml` | Inductive v1 | noexp + sum + lr=3e-5 | 0.482 | 1 |
| `wn18rr_ind_v1_noexp_sumpool_const5e5.yaml` | Inductive v1 | noexp + sum + lr=5e-5 const | 0.449 | 0 |
| `wn18rr_ind_v1_indrouting.yaml` | Inductive v1 | **inductive_routing=True** + lr=3e-5 | 0.532 | 2 |
| **`wn18rr_ind_v1_indrouting_lr1e5.yaml`** | Inductive v1 | inductive_routing + lr=1e-5 warmup=5 | **0.565** | **4** |
| `wn18rr_ind_v1_indrouting_dim128.yaml` | Inductive v1 | inductive_routing + dim=128 L=5 | 0.501 | 1 |
| `wn18rr_ind_v1_relv.yaml` | Inductive v1 | use_relational_v=True (ABANDONED) | 0.384 | — |
| `wn18rr_ind_v1_indrouting_constlr.yaml` | Inductive v1 | constlr + lr=5e-6 (scheduler=none) | 0.563 | 3 |
| `wn18rr_ind_v1_indrouting_L5.yaml` | Inductive v1 | L=5 + lr=1e-5 cosine | 0.494 | 3 |
| `wn18rr_ind_v1_nbfv2.yaml` | Inductive v1 | use_nbf_v=True (xavier init) | 0.420 | 10 |
| `wn18rr_ind_v1_vrmpnn.yaml` | Inductive v1 | V-RMPNN sin query cond. | 0.467 | 11 |
| `wn18rr_ind_v1_vrmpnn.yaml` (qcond fix) | Inductive v1 | V-RMPNN con query cond. | 0.513 | 12 |
| `wn18rr_ind_v1_pna.yaml` | Inductive v1 | PNA concat(sum,mean,max) | 0.284 | 4 |
| `wn18rr_ind_v1_mlpscorer.yaml` | Inductive v1 | MLP scorer (128→64→ReLU→1) | 0.458 | 4 |
| `wn18rr_ind_v{2,3,4}.yaml` | Inductive v2-v4 | — | not run yet | — |
| NBFNet (paper) | Inductive v1 | — | 0.741 | — |
| KnowFormer (paper) | Inductive v1 | — | 0.752 | — |

**Current best**: `wn18rr_ind_v1_indrouting_lr1e5.yaml` → **0.565 test MRR** (ep4, val ckpt = 0.415).

**REGLA**: todos los experimentos nuevos usan `ckpt_monitor_split: val`.

**Scoring head**: `Linear(2d→1)` sobre `cat(h_v^L, rel_emb[r_q])` es el óptimo para esta arquitectura. MLP scorer fue peor (-0.107 MRR) porque `h_v^L` ya está query-condicionado — el MLP añade complejidad sin expresividad real.

**Next**: FiLM conditioning del FFN (`gt.use_film_ffn: True`) — único componente arquitectónico sin explorar que ataca el cuello de botella (FFN aprende patrones del train graph). Si falla, techo = 0.565.

## Plan para alcanzar MRR ≥ 0.74 (NBFNet level)

### Diagnóstico raíz (2026-04-10, actualizado)

El gap 0.565 → 0.74 tiene una causa principal identificada:

**Causa A — V es relación-agnóstico** (brecha de expresividad, ~0.15 MRR estimado)

NBFNet (Zhu et al., NeurIPS 2021, §3.2) usa mensaje DistMult:
```
msg_{i→j,r} = h_i^(l) ⊙ W_r    # relación transforma directamente la repr. fuente
```
`W_r` es compartido en todos los grafos → inductivo. Los gradientes aprenden
patrones relacionales, no estructurales.

Nuestro modelo: `V = W_V(h_i)`. La relación entra solo como gate post-proyección
(`gate = V_gate(rel_emb[r_ij])`). El gate ya provee dependencia relacional pero
de forma aditiva/escalar, no elemento-a-elemento como DistMult.

**NOTA — use_relational_v CANCELADO (2026-04-10)**: intentar `v_src *= rel_emb[r_ij]`
creó dos señales multiplicativas relacionales competitivas (V_gate + rel_emb directa)
→ 0.384 MRR (interferencia). El flag `use_relational_v` fue retirado del plan.
El gate ya ES el V relacional de primer orden. Si se quiere ir más lejos, la
dirección correcta es V-RMPNN (stream separado al estilo KnowFormer).

**Causa B — Routing aprende patrones del grafo de train** (colapso, ~0.10 MRR)
→ **RESUELTO** con `inductive_routing: True` (Cambio 2, implementado 2026-04-10).

Score score ahora es función de (r_ij, r_q, Q_j) sin K_i específico al grafo:
```python
K_h = self.proj_k(shared_rel[r_q])   # sólo conditioning por query
# (antes: K_h = self.K(h) + self.proj_k(shared_rel[r_q]))
```
Resultado: 0.482 → 0.532 (+0.050 MRR), decline más gradual (ep28=0.374).
Con lr=1e-5 + warmup: **0.565 MRR** (ep4), hits@10=0.776 sostenido.

### Estado actual de los cambios

| Cambio | Estado | Resultado |
|--------|--------|-----------|
| Sum aggregation (NBFNet-style) | ✅ IMPLEMENTADO | 0.252→0.482 (+0.23) |
| Inductive routing (K = f(r_q) only) | ✅ IMPLEMENTADO | 0.482→0.532 (+0.05) |
| LR tuning (lr=1e-5 + warmup=5) | ✅ VALIDADO | 0.532→0.565 (+0.033) |
| use_relational_v (V *= rel_emb) | ❌ CANCELADO | 0.532→0.384 (interferencia) |
| dim=128, L=5 (scale-up) | ❌ NO AYUDA | 0.565→0.501 (overfit más rápido) |
| use_nbf_v (DistMult puro, std=0.01) | ❌ PEOR | 0.514 (init near-zero, gate es mejor) |
| tie_layers (weight tying entre capas) | ❌ PEOR | 0.513 (mejor hits@10 pero peor MRR) |
| constlr (lr=5e-6, scheduler=none) | ❌ IGUAL | 0.563 — colapso NO es por scheduler |
| L=5 capas (lr=1e-5 cosine) | ❌ PEOR | 0.494 — más capas = más overfit train graph |
| use_nbf_v (xavier init) | ❌ PEOR | 0.420 — raw h sin proyección inestable |
| V-RMPNN con query cond. | ❌ PEOR | 0.513 — < baseline, competition con main stream |
| PNA concat(sum,mean,max) | ❌ CATASTRÓFICO | 0.228 — mean reintroduce Z normalization |
| MLP scorer (2d→d→ReLU→1) | ❌ PEOR | 0.458 — h_v^L ya query-conditioned |
| **FiLM FFN conditioning** | **PRÓXIMO** | **única dirección restante sin explorar** |

### Diagnóstico raíz actualizado (2026-04-10, sesión 5)

**El colapso post-ep3 es invariante al LR, número de capas, scheduler, head, y aggregation.**

Tras 20+ experimentos, el cuello de botella identificado es el **FFN**:

Con `inductive_routing=True`, todos los componentes salvo el FFN son puramente relacionales:
- Score, K, Q, E: funciones de `(r_q, r_uv)` — inductivos ✅
- V: `W_V(h) * gate(r_uv)` — gate relacional, pero `W_V(h)` es graph-specific ⚠️
- **FFN**: `h = h + ff2(ReLU(ff1(h)))` — matrices genéricas que aprenden patrones del train graph ❌

NBFNet no tiene FFN (solo σ element-wise). El FFN aprende "qué aspecto tienen las
representaciones del train graph" y aplica eso al test graph donde las distribuciones difieren.

**Por qué KnowFormer no sufre esto**: filtra al nivel del MENSAJE —
`msg = z_u ⊙ W^r(r_q)` — la topología pasa por un filtro relacional antes de agregarse.
Lo que llega a `z_v` siempre es función de (tipos de caminos × query), no de entidades específicas.
Nuestro `W_V(h)` mezcla relacional+topológico antes de cualquier filtrado.

### Próxima dirección: FiLM conditioning del FFN

Único componente arquitectónico no explorado que ataca directamente la causa raíz:

```python
# Reemplazar en MultiLayer.forward():
# h = h + ff2(ReLU(ff1(h)))   ← aprende patrones del train graph

# Por:
gamma_r = self.film_gamma(shared_rel_emb[r_q][batch.batch])  # (N, dim)
beta_r  = self.film_beta(shared_rel_emb[r_q][batch.batch])   # (N, dim)
h = h + ff2(ReLU(ff1(h) * gamma_r + beta_r))                 # inductivo
```

El FFN condicionado aprende "cómo transformar para este tipo de query" en lugar de "cómo
transformar representaciones del train graph". gamma/beta son funciones de r_q → inductivo.

**Veredicto**: si FiLM falla, el techo es **0.565**. El gap 0.565→0.741 sería estructural
(requeriría reescribir el mecanismo de mensaje a RMPNN-style como KnowFormer).

**NO explorar**: DropEdge (no ataca causa raíz — FFN no memoriza aristas sino distribuciones),
MLP scorer (h_v^L ya es query-conditioned — PROBADO: -0.107), V-RMPNN (PROBADO: 0.513),
PNA con mean (CATASTRÓFICO: 0.228), use_nbf_v (PROBADO: 0.420), más capas/dim, weight tying.

### Experimentos cerrados y lecciones (sesiones 4-5)

| Experimento | Resultado | Lección |
|-------------|-----------|---------|
| PNA concat(sum,mean,max) | 0.228 val, FALLO | mean=sum/Z reintroduce normalización dañina |
| V-RMPNN sin query cond. | 0.467 test | stream separado sin query cond. → < baseline |
| V-RMPNN con query cond. | 0.513 test | mejor RMPNN pero < 0.565 — gradient competition |
| MLP scorer (2d→d→1) | 0.458 test | h_v^L ya query-conditioned — MLP añade ruido |
| use_nbf_v xavier | 0.420 test | raw h sin proyección inestable, gate es mejor |

### Config base actual (mejor: indrouting_lr1e5)

```yaml
gt:
  layers: 3
  dim_hidden: 64
  n_heads: 4
  dropout: 0.1
  use_edge_gating: True
  use_query_conditioning: True
  inductive_routing: True

optim:
  base_lr: 0.00001
  scheduler: cosine_with_warmup
  num_warmup_epochs: 5
  max_epoch: 50
```

## Dead Code in KGC Mode

These exist in the codebase but are never exercised when running KGC experiments:
- `ExphormerFullLayer` in `layer/exphormer.py` — never instantiated (KGC uses `ExphormerAttention` directly)
- Virtual node paths in `ExpanderEdgeFixer` and `ExphormerAttention` — `num_virt_node=0`
- `train_epoch()` / `eval_epoch()` (DataLoader-based) — KGC always uses full-graph path
- `KGCDataset.__getitem__`, subgraph cache, `_extract_subgraph` — full-graph mode skips these
- `LocalModel` (GCN/GINE/GAT/GatedGCN) — `layer_type: Exphormer` only
- `grad_checkpoint`, `batch_accumulation`, `max_iter` — all at defaults (off)
- `label_smoothing: 0.0` — smoothing path in `kgc_full_graph_ce` never activates

## Extending the Codebase

**New dataset**: implement in `loader/dataset/`, add dispatch branch in `loader/master_loader.py:load_dataset()`.

**New node/edge encoder**: implement module, add elif in `build_node_encoder()` / `build_edge_encoder()`.

**New layer type**: implement in `layer/`, add elif in `MultiLayer.__init__()` in `network/model.py`.

**New inductive experiment**: copy a `wn18rr_ind_v*.yaml`, update `dataset.name`, `kgc.train_steps_per_epoch`, create a corresponding `sbatch_*.sh` pointing to `logs/`.
