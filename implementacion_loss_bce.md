# Implementación: loss BCE + negative sampling + self-adversarial (KnowFormer-style)

**Fecha**: 2026-05-20 (sesión 30)
**Motivación**: aislar el efecto del *loss* en el colapso inductivo. Hasta ahora todo el
entrenamiento KGC usaba CE de grafo completo (softmax sobre las N entidades). KnowFormer
usa BCE con muestreo de negativos + pesado self-adversarial en la mayoría de sus
experimentos inductivos. Este cambio añade ese loss como una opción configurable, sin
tocar la arquitectura.

> Nota de diagnóstico: leyendo el código real de KnowFormer (`Knowformer/lightning.py`)
> se confirmó que **KnowFormer SÍ usa CE de grafo completo** en WN18RR ind v3/v4 y NELL v2
> (ver tabla del README), y entrena con **LR constante 5e-3 sin warmup** (`MultiStepLR`).
> Esto debilita la hipótesis de que el CE-global sea la causa del colapso. Aun así, BCE+neg
> sampling es el loss por defecto de KnowFormer en v1/v2, y vale probarlo como una palanca
> independiente. Las curvas de train y un run a LR constante son los diagnósticos
> complementarios pendientes.

---

## Qué se implementó

### 1. Nueva función de loss — `loss/losses.py`

`kgc_bce_neg_sample(scores, true_tails, filter_dict, chunk_h, chunk_r,
num_negative_sample=7, adversarial_temperature=1.0, head_filter=None, base_num_rel=None)`

Port fiel de `Knowformer/lightning.py:121-149` (`compute_loss` con `loss_fn='bce'` +
`negative_sample`):

1. **Filter mask por query**: se marca como `True` toda respuesta verdadera conocida de
   `(h, r)` (vía `filter_dict`; vía `head_filter[(h, r-base_num_rel)]` para recíprocas).
   Estas nunca se muestrean como negativo. El positivo `t` también queda excluido porque
   vive dentro de `filter_dict[(h,r)]` (igual que en KnowFormer).
2. **Muestreo de negativos**: `K = min(N, 2**num_negative_sample)` negativos uniformes
   sobre las entidades no filtradas, `torch.multinomial(..., replacement=True)` (idéntico
   a KnowFormer).
3. **BCE por candidato**: `logits = gather(scores, [positivo, neg_1..neg_K])`,
   `target = [1, 0, ..., 0]`, `binary_cross_entropy_with_logits(reduction='none')` —
   sigmoide independiente por candidato, **no** softmax.
4. **Pesado self-adversarial (estilo RotatE)**: `weights[:, 1:] = softmax(logits_neg /
   temperature)` calculado bajo `torch.no_grad()` (detached); el positivo conserva peso 1.
   `loss = (bce * weights).sum()`.

**Detalle de escala (importante)**: se usa `.sum()` exactamente como KnowFormer, no
`.mean()`. Por query la contribución es ≈ `BCE_pos·1 + Σ_k BCE_neg_k·w_k` con `Σ w_k = 1`
(o sea ~2 unidades BCE por query), y se suma sobre el batch → **el loss escala con
`train_batch_size`**. Esto difiere de `kgc_full_graph_ce`, que promedia (`.mean()`). Si se
compara la magnitud del loss entre runs CE y BCE, no son directamente comparables, y el LR
efectivo del BCE depende del batch size. Para `train_batch_size=8` el loss inicial sale
~12 (vs ~7.9 del CE).

La función devuelve `(loss, scores)` con `scores` sin modificar, para que el logging de
métricas (que usa `pred_score`) siga funcionando igual que con CE.

### 2. Dispatch en el trainer — `train/trainer.py`

En `train_epoch_kgc` (la rama full-graph), tras `pred, _ = model(data)`:

```python
if getattr(cfg.kgc, 'loss_fn', 'ce') == 'bce':
    loss, pred_score = kgc_bce_neg_sample(
        pred, chunk_t.to(device), filter_dict, chunk_h, chunk_r,
        num_negative_sample=getattr(cfg.kgc, 'num_negative_sample', 7),
        adversarial_temperature=getattr(cfg.kgc, 'adversarial_temperature', 1.0),
        head_filter=kgc_ds.head_filter,
        base_num_rel=kgc_ds.num_base_relations,
    )
else:
    loss, pred_score = kgc_full_graph_ce(...)   # path CE original, sin cambios
```

El default (`loss_fn` ausente o `'ce'`) preserva exactamente el comportamiento previo.

### 3. Parámetros de config — `config.py`

```python
cfg.kgc.loss_fn                 = 'ce'   # 'ce' | 'bce'
cfg.kgc.num_negative_sample     = 7      # K = min(N, 2**num_negative_sample) (bce)
cfg.kgc.adversarial_temperature = 1.0    # temperatura del softmax self-adversarial (bce)
```

### 4. Config y sbatch de experimento

- `configs/Exphormer/wn18rr_ind_v1_bce.yaml` — copia de `wn18rr_ind_v1.yaml` con
  `loss_fn: bce`, `num_negative_sample: 8` (→256 negativos) y
  `adversarial_temperature: 0.5` (los valores de KnowFormer para WN18RR ind v1). Resto de
  la arquitectura sin cambios.
- `sbatch_wn18rr_ind_v1_bce.sh` — 1×H100, 12h.

---

## Qué NO se tocó

- La arquitectura (Q/K/V, gate, BF residual, V-NBF) — intacta.
- El path CE de grafo completo (`kgc_full_graph_ce`) — intacto.
- La evaluación (MRR/Hits filtrado) — sigue siendo grafo completo, idéntica para ambos
  losses. El loss solo cambia el *entrenamiento*.

---

## Verificación

| Test | Resultado |
|------|-----------|
| Smoke `loss_fn=bce` (v1, 1 ep, 4 steps) | exit 0, train loss=12.05, eval corre |
| Smoke `loss_fn=ce` default (regresión) | exit 0, train loss=7.86 (sin cambio) |
| Smoke config `wn18rr_ind_v1_bce.yaml` | exit 0, train loss=12.26 |

---

## Diferencias vs KnowFormer (a tener en cuenta al interpretar)

1. **Arquitectura distinta**: KnowFormer recompone Q/K/V desde NBF streams frescos cada
   capa; nosotros mantenemos `V=h` + BF residual. El loss es ahora idéntico, la
   arquitectura no.
2. **Schedule distinto**: KnowFormer usa Adam + LR constante 5e-3 + `MultiStepLR([10,15])`
   sin warmup. Nuestro config BCE mantiene el cosine-warmup-a-8e-4 de siempre. Si se quiere
   replicar fielmente a KnowFormer habría que cambiar también el schedule (pendiente).
3. **Filtrado en train**: en la rama CE de KnowFormer no se enmascaran las otras colas
   verdaderas; en BCE sí se evita muestrearlas como negativos. Nuestro `kgc_bce_neg_sample`
   las excluye del muestreo (consistente con KnowFormer BCE). Nuestro `kgc_full_graph_ce`
   sí enmascara (más estricto que el CE de KnowFormer, diferencia menor).

---

## Próximos pasos sugeridos

1. Lanzar `wn18rr_ind_v1_bce.yaml` (1×H100) y comparar la trayectoria val/test MRR vs el
   baseline CE (0.58). ¿Desaparece el colapso ep1→ep2?
2. Mirar la curva de **train loss/MRR** a través del colapso (discrimina overfit vs
   divergencia) — diagnóstico ortogonal y barato.
3. Run con schedule de KnowFormer (Adam + LR constante + MultiStepLR) para aislar el efecto
   del schedule del efecto del loss.
4. Si BCE estabiliza: barrer `num_negative_sample` ∈ {6, 8, 10} y `adversarial_temperature`
   ∈ {0.5, 1.0}.
