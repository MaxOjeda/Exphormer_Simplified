# CLAUDE.md

Guía para Claude Code en este repositorio.

## Proyecto

Reimplementación standalone de Exphormer (ICML 2023) sin dependencia de GraphGPS/graphgym, adaptada para **Knowledge Graph Completion (KGC) inductivo** — Etapa 1 de tesis doctoral. El grafo expander es el mecanismo central de atención global (contribución de tesis). Una sola arquitectura para transductivo e inductivo; las diferencias son solo hiperparámetros.

## Entorno

```bash
# conda run NO funciona — usar ruta directa
/nfs_ssd/mojeda_imfd/miniconda3/envs/exphormer/bin/python main.py --cfg <config> wandb.use False

# Smoke test (verificar que no rompe nada)
python main.py --cfg configs/Exphormer/wn18rr_ind_v1.yaml wandb.use False \
    optim.max_epoch 1 kgc.train_steps_per_epoch 4 kgc.eval_batch_size 4
```

Resultados en `results/<config_name>/<run_id>/`. Logs en `logs/`.

## Config System

YACS `CfgNode` en `config.py`. Secciones clave:
- `gt.*`: layer_type, layers, n_heads, dim_hidden, dropout, use_query_conditioning, qk_noise_std, num_qk_layers
- `gnn.*`: head (`kgc`), dim_inner (debe ser igual a `gt.dim_hidden`)
- `kgc.*`: reciprocal, eval/train_full_graph, batch sizes, train_steps_per_epoch, label_smoothing
- `prep.*`: exp, exp_deg, add_edge_index, num_virt_node
- `optim.*`: base_lr, scheduler, max_epoch, num_warmup_epochs, clip_grad_norm
- `train.*`: ckpt_best, ckpt_monitor_split (`val` o `test`)

## No Registration System

Dispatch directo por if/elif (no hay registry):
- `build_node_encoder(cfg, dim_in)` → `encoder/node_encoders.py`
- `build_edge_encoder(cfg)` → `encoder/edge_encoders.py`
- `build_head(cfg, dim_in, dim_out)` → `network/heads.py`
- `create_model(cfg, dim_in, dim_out)` → `network/model.py`

## Key Bugs / Gotchas

- **V_gate SIN sigmoid**: `gate = batch.E_gate`, no `torch.sigmoid(...)`. Eliminarlo dio +0.08 MRR. No volver a agregar.
- **`gnn.dim_inner` == `gt.dim_hidden`**: assertion en `MultiModel.__init__`. Si difieren, falla al iniciar.
- **Val usa train graph en inductivo**: los triples de val de WN18RR inductivo pertenecen al conjunto de entidades de train. `ckpt_monitor_split: val` es la convención académica correcta igual.
- **DDP `find_unused_parameters=True`**: necesario cuando `prep.exp=False` para evitar error de reducción DDP sobre `exp_edge_attr`.
- **DDP timeout**: `timeout=timedelta(hours=2)` en `dist.init_process_group`. El default (600s) es insuficiente para eval de FB15k-237 (~616s). Corregido en `main.py`.
- **LR scaling multi-GPU**: con 4 GPUs, multiplicar `base_lr` por 4 (batch efectivo = 4×base).
- **`batch.x0` se setea post-encoder**: en `MultiModel.forward()`, `batch.x0 = batch.x` ocurre DESPUÉS de `self.encoder(batch)`. `x0_h = rel_emb_enc[r]`, `x0_others = 0`.
- **`scipy.ravel` eliminado**: usar `np.ravel`. Corregido en `transform/dist_transforms.py:171`.

## Extending the Codebase

- **Nuevo dataset**: implementar en `loader/dataset/`, agregar rama en `loader/master_loader.py:load_dataset()`.
- **Nuevo encoder**: agregar `elif` en `build_node_encoder()` / `build_edge_encoder()`.
- **Nuevo layer**: implementar en `layer/`, agregar `elif` en `MultiLayer.__init__()` en `network/model.py`.
- **Nuevo experimento inductivo**: copiar un `wn18rr_ind_v*.yaml`, actualizar `dataset.name` y `kgc.train_steps_per_epoch`, crear `sbatch_*.sh` correspondiente en `logs/`.
