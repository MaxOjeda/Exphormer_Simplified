# Session Notes — 2026-03-31

## Estado actual

Jobs 560280 (noexp) y 560281 (exp3) corrieron y terminaron por límite de tiempo.
Resubmitir ambos con `sbatch <script>.sh` (auto_resume=True retoma desde el checkpoint).

---

## Cambio clave (sesión 2026-03-30, activo)

**Se eliminó el `torch.sigmoid()` del V_gate en `layer/exphormer.py:71`.**

```python
# ANTES:
gate = torch.sigmoid(batch.E_gate)   # gate ∈ (0,1) — solo suprime
# AHORA:
gate = batch.E_gate                  # sin restricción — puede amplificar
```

Este cambio, combinado con T=5 y 100% cobertura, produjo el salto 0.449 → 0.534+.

---

## Resultados acumulados — WN18RR, dim=64, T=5, 100%

### Job 560281 — Con expander deg=3 ← MEJOR, PRIORIDAD 1

| Época | val MRR | test MRR | test H@1 | test H@3 | test H@10 |
|-------|---------|----------|----------|----------|-----------|
| 0     | 0.081   | 0.081    | 0.008    | 0.105    | 0.302     |
| 4     | 0.519   | 0.520    | 0.446    | 0.559    | 0.664     |
| 6     | 0.532   | 0.534    | 0.462    | 0.574    | 0.669     |
| **8** | **0.539** | **0.537** | **0.468** | **0.582** | **0.670** |
| 9     | 0.537   | 0.534    | 0.463    | 0.578    | 0.669     |
| 10    | 0.533   | 0.529    | 0.456    | 0.574    | 0.669     |

**Best checkpoint**: época 8 (val_mrr=0.5392) → `results_d64_T5_100pct_exp3/0/ckpt.pt`
Oscilando. Resubmitir.

---

### Job 560280 — Sin expander

| Época | val MRR | test MRR | test H@1 | test H@3 | test H@10 |
|-------|---------|----------|----------|----------|-----------|
| 8     | 0.509   | 0.513    | 0.428    | 0.565    | 0.671     |
| 12    | 0.512   | 0.518    | 0.435    | 0.572    | 0.670     |
| 13    | 0.517   | 0.519    | 0.437    | 0.571    | 0.671     |
| 14    | 0.516   | 0.521    | 0.437    | 0.575    | 0.671     |
| 15    | 0.522   | 0.522    | 0.440    | 0.576    | 0.669     |
| 16    | 0.519   | 0.522    | 0.440    | 0.577    | 0.672     |
| **17**| **0.526** | **0.529** | **0.449** | **0.580** | **0.672** |
| 18    | 0.523   | 0.526    | 0.445    | 0.579    | 0.671     |
| 19    | 0.524   | 0.527    | 0.445    | 0.579    | 0.670     |
| 20    | 0.524   | 0.524    | 0.443    | 0.577    | 0.669     |

**Best checkpoint**: época 17 (val_mrr=0.5263) → `results_d64_T5_100pct_noexp/0/ckpt.pt`
Aún mejorando lentamente (oscilando ~0.524-0.526). Resubmitir.

---

## Comparación con NBFNet (referencia WN18RR)

| Modelo | MRR | H@1 | H@3 | H@10 |
|--------|-----|-----|-----|------|
| NBFNet (paper) | **0.551** | 0.497 | 0.573 | 0.666 |
| **Nuestro exp3 ep8** | **0.537** | 0.468 | **0.582** | **0.670** |
| Nuestro noexp ep17 | 0.529 | 0.449 | **0.580** | **0.672** |
| exp3 ep6 (anterior) | 0.534 | 0.462 | 0.574 | 0.669 |

Gap con NBFNet: **0.014 MRR** (exp3). H@3 y H@10 ya superamos.

---

## Conclusión sobre expander

| Config | MRR (best) | Tiempo/época | Observación |
|--------|-----------|--------------|-------------|
| No expander | 0.529 (ep17) | ~103 min | Aún mejorando ep20+ |
| Expander deg=3 | **0.537** (ep8) | ~193 min | Oscilando ep8-10 |
| Expander deg=5 | ~0.49 (ep4) | ~253 min | NO resubmitir |

Expander deg=3 da +0.008 MRR en épocas tempranas (~8 vs ~17). Ambos siguen mejorando.

---

## Próximos pasos (en orden de prioridad)

### 1. Resubmitir exp3 (PRIORIDAD 1 — sigue sin platear)
```bash
cd /nfs_ssd/mojeda_imfd/Doctorado/Exphormer_Max
sbatch sbatch_d64_T5_100pct_exp3.sh
```

### 2. Resubmitir noexp (comparación limpia)
```bash
sbatch sbatch_d64_T5_100pct_noexp.sh
```

### 3. Experimento: T=7 con exp3
Crear `sbatch_d64_T7_100pct_exp3.sh` con `gt.layers 7`.
WN18RR tiene diámetro ~6; T=5 puede no capturar todos los paths.
Estimado: ~16000s/época (~4.5h) → ~5 épocas/24h.

### 4. Implementar low-rank W_V[r] (mayor impacto teórico)
Modulación multiplicativa por relación en el valor:
`v_src = W_V(h_u) * (1 + delta_V[r_edge])`  (low-rank rank=8, ~+5K params)
Estimado: +0.02-0.04 MRR. Requiere ~20 líneas en `layer/exphormer.py`.

### 5. FB15k-237 cuando WN18RR esté saturado

---

## Config base actual

```yaml
gt.dim_hidden: 64
gt.dim_edge:   64
gnn.dim_inner: 64
gt.layers:     5
gt.n_heads:    4        # 16 dim/head
use_edge_gating: True   # sin sigmoid (cambio de 2026-03-30)
use_query_conditioning: True
prep.exp: True
prep.exp_deg: 3
kgc.train_steps_per_epoch: 10854   # 100% cobertura
optim.max_epoch: 100
optim.base_lr: 0.0002
train.eval_period: 1
train.auto_resume: True
```

## Archivos clave

| Archivo | Descripción |
|---------|-------------|
| `layer/exphormer.py:71` | Sigmoid removido (cambio principal) |
| `sbatch_d64_T5_100pct_exp3.sh` | Resubmitir este primero |
| `sbatch_d64_T5_100pct_noexp.sh` | Resubmitir segundo |
| `results_d64_T5_100pct_exp3/0/ckpt.pt` | Mejor checkpoint (MRR=0.537, ep8) |
| `results_d64_T5_100pct_noexp/0/ckpt.pt` | Mejor noexp checkpoint (MRR=0.529, ep17) |
