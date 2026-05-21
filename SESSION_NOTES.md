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

## Estado actual — 2026-05-21 (sesión 32): barrido loss/lr/regularización sobre la arquitectura limpia — techo inductivo ~0.40, colapso confirmado como arquitectónico

### Contexto
Revisión completa de `papers_distilled.md`, `manuscrito_candidatura.pdf` y `metodologia.tex`.
- **Hallazgo del manuscrito**: Etapa 1 solo exige *"competitivo con NBFNet"*; superar a NBFNet es **H2 / Etapa 2-3** (codificación relacional transferible estilo ULTRA). El gap inductivo actual NO es deuda de Etapa 1.
- **`metodologia.tex` quedó desactualizado vs el reset de sesión 31**: describe `inductive_routing` (K solo-query → 0.578), FiLM-E multiplicativo y FFN inner 4d, cuando la arquitectura limpia usa K estándar, E aditivo y **FFN inner 2d** (verificado en el log: `Linear 64→128→64`).

### Restricción metodológica fijada por el usuario
**Una sola arquitectura (forward pass idéntico) para trans e ind. Se permite variar loss e hiperparámetros por setting, NO el core.** Esto descarta reinstalar `inductive_routing` como flag: es una rama en el forward pass, no un escalar → un revisor lo llamaría "dos arquitecturas".

### Experimentos (WN18RR ind v1, arquitectura limpia 279,041 params, 1×H100, best por val)

| Run | Loss | lr / warmup | reg extra | pico | final ep29 | dinámica |
|-----|------|-------------|-----------|------|-----------|----------|
| CE | CE grafo completo | 8e-4 / 3 | — | ep1 **0.232** | 0.015 | crash |
| BCE-128 | BCE self-adv, 128 negs, temp 0.5 | 8e-4 / 3 | — | ep1 **0.393** | 0.017 | crash |
| **BCE-128 lr1e5** | idem | **1e-5 / 8** | — | ep6 **0.400** | 0.319 | **declive suave → meseta ~0.32, SIN crash** |
| BCE-128 lr1e5 reg | idem | 1e-5 / 8 | dropout 0.4 + wd 1e-2 | ep11 0.376 | (cortado ep15) | declive igual, techo más bajo |

### Conclusiones
1. **BCE-128 > CE** (+0.17 MRR) sobre la arquitectura limpia — invierte el hallazgo de sesión 30 (que era sobre V-NBF v5). Negative sampling + self-adversarial sí ayuda aquí.
2. **lr bajo elimina el crash catastrófico** (0.017 → meseta 0.32) pero **NO sube el techo** (0.40, igual que lr alto). El crash es lr-driven; el techo y el overfit leve son lr-independientes.
3. **El colapso NO es divergencia sino overfit que llega a equilibrio**: `train_loss` baja monótona mientras test se estanca en meseta; al decaer el lr en la cola del cosine el test se estabiliza (~0.32), no se rompe. Refina el "punto 4" del diagnóstico previo (el colapso NO coincide causalmente con el peak LR; el LR solo fija la velocidad de descenso a la cuenca de overfit).
4. **Regularización (dropout/wd) NO es palanca**: no aplana el declive y baja el techo. `wd=1e-2` estaba **100× fuera del rango de KnowFormer** (E.4: lr {1e-4..5e-3}, wd {0..1e-4}, negs {2⁶..2¹⁶}, d {16,32,64}).
5. **Techo de la arquitectura unificada (K estándar) ≈ 0.40 val-selected**, robusto a loss/lr/reg.

### Sospechoso #2 (residual BF `h += x0`) — DESCARTADO
El análisis viejo ("dos rutas de inyección del ancla → inestabilidad") es **obsoleto**: la ruta NBF (`VLayerNBF`/`fc_z`) se eliminó en sesión 31; ya no hay doble inyección. Y el residual BF solo toca el **ancla** (`x0[ancla]=rel_emb[r_q]`, `x0[resto]=0`) → es no-op sobre las representaciones de los candidatos que se rankean, y su contenido es **relacional** (transferible). NBFNet tiene el mismo término de frontera y saca 0.741. **No tocar el residual BF.**

### Raíz real del overfit (no atacable bajo la restricción)
`K=W_K(h)`, `V=W_V(h)`, `FFN(h)` leen la `h` acumulada → memorizan la distribución del grafo de train. KnowFormer es inmune porque su Q/K/V salen de streams NBF frescos que nunca leen `h` (por eso entrena a lr 5e-3 sin colapsar). La cura es **arquitectónica = Etapa 2** (codificación relacional), prohibida por la restricción de Etapa 1. Dentro de la restricción, loss/lr/reg ya están agotados.

### Configs nuevos
`configs/Exphormer/wn18rr_ind_v1_bce128_lr1e5.yaml` (mejor inductivo limpio: 0.40), `wn18rr_ind_v1_bce128_lr1e5_reg.yaml` (sobre-regulariza, NO usar tal cual — wd 1e-2 fuera de rango).

### Decisión pendiente (interrumpida)
(a) probar negativos 2¹⁰-2¹² como última palanca de loss en rango KnowFormer, (b) aceptar ~0.40 como número unificado de Etapa 1 (defendible: el manuscrito solo pide "competitivo"), o (c) instrumentar grado↔rank para confirmar el atajo estructural antes de cualquier decisión de Etapa 2.

---

## Estado actual — 2026-05-21 (sesión 31): RESET a la arquitectura limpia del mejor resultado (0466cde)

### Motivación

Se constató que el HEAD había derivado a la arquitectura **V-NBF v5 + C2** — acumulación de
los experimentos fallidos de sesiones 24-30, **nunca validada en transductivo** y que colapsa
en inductivo. La arquitectura del mejor resultado (trans 0.566) vivía enterrada en el commit
`0466cde`. El requisito de tesis es **una sola arquitectura para trans e ind** (solo cambian
hiperparámetros), así que se reconstruyó esa arquitectura **limpia**, sin ningún flag experimental.

### Dos decisiones del usuario en este reset

1. **`inductive_routing` ELIMINADO** → K siempre estándar `K = W_K(h) + proj_k(c_q)` para
   ambos settings. (No afecta al transductivo: el run de 0.566 ya usaba K estándar. SÍ afecta
   al inductivo histórico, que usaba `K = proj_k(q)` solo para llegar a 0.578.)
2. **FiLM en E ELIMINADO** → E aditivo: `E = W_E(φ(r)) + proj_e(c_q)` en vez del
   `E = W_E(φ(r)) ⊙ (1 + proj_e(c_q))` multiplicativo del run de 0.566.

### Arquitectura limpia resultante (`layer/exphormer.py`, `network/model.py`)

```
Q = W_Q(x0) + proj_q(c_q)            # anclado a boundary condition h0
K = W_K(h)  + proj_k(c_q)            # routing estándar entity-específico + query
E = W_E(φ(r)) + proj_e(c_q)          # aditivo (no FiLM)
gate = V_gate(φ(r)) + proj_vg(c_q)   # SIN sigmoid
V = W_V(h)                           # proyección estándar
score = exp(clip((Q⊙K⊙E)/√d, -5, 5)); agregación SUMA (sin /Z); h += x0; FFN 2-capas (inner 2d)
```

- **Tablas relacionales SEPARADAS por componente** (KGCNodeEncoder.rel_emb, cada
  ExphormerAttention.shared_rel_emb_table, ExpanderEdgeFixer.exp_edge_query_emb, KGCHead.rel_emb),
  todas indexadas por `batch.query_relation`. Esto **revierte la unificación de embeddings**
  (`batch.query_emb`) → de paso resuelve la regresión 2.7× de velocidad de sesión 17/20.

### Eliminado del código

V-NBF (`VLayerNBF`/`QKLayerNBF`/`v_nbf`/`v_nbf2`), `VRMPNN`, C1 (K=K(x0)), C2 (gate bilinear
`gate_base`/`fc_zq`), C3 (`norm_V`), C4 (`use_ffn`), `use_nbf_v`, `use_distmult_v`,
`use_rel_matrix_v`, `use_pna`, `gate_rel_mult`, `use_alpha_mix_qk`, `use_film_ffn`,
`noise_std`, `qk_noise_std`, `num_qk_layers`, y el `query_emb` unificado.

### Archivos tocados

- **Reescritos limpios**: `layer/exphormer.py`, `network/model.py`.
- **Restaurados de 0466 (tablas separadas)**: `encoder/node_encoders.py`, `network/heads.py`,
  `encoder/exp_edge_fixer.py` (`git checkout 0466cde -- ...`).
- **`config.py`**: quitados flags muertos de `gt`; añadido `cfg.gt.use_edge_gating`.
- **Configs canónicas** (ind_v1/v2/v3/v4, transduct_best, ind_v1_bce): añadido `use_edge_gating: True`.
  Borradas 3 configs de experimentos: `qkfresh`, `qkrmpnn`, `novw_c1c4`.
- **Intactos**: `loss/losses.py`, `train/trainer.py` (loss BCE configurable preservado), `main.py`.

### Verificación

Smoke tests exit 0 sobre la arquitectura limpia: ind_v1 CE (loss 7.90), ind_v1_bce (loss 12.31).
**279,041 params** (vs 1.4M de la V-NBF v5). Sin referencias colgantes a símbolos eliminados.

### Pendiente / ojo

- **El 0.578 inductivo histórico usó `inductive_routing=True`** (K solo-query), ahora eliminado.
  Con K estándar el inductivo histórico val-selected fue ~0.486. **Re-medir** el inductivo limpio
  antes de asumir 0.58.
- Comentario obsoleto en `train/trainer.py:349` (menciona `use_nbf_v`) y `edge_rel_idx` que queda
  sin uso — inofensivos, no tocados para no arriesgar el código de BCE.

### Experimentos lanzados (cluster SLURM, 1×H100 c/u, 30 ep)

| Job | Loss | Negs | Config |
|-----|------|------|--------|
| 613264 `ind_ce` | CE grafo completo | todos los N | `wn18rr_ind_v1.yaml` |
| 613265 `ind_bce128` | BCE + self-adversarial (temp 0.5) | **128** (`num_negative_sample: 7`) | `wn18rr_ind_v1_bce128.yaml` (nuevo) |

Quedaron PENDING (un `bash` interactivo ocupaba el nodo H100). Objetivo: comparar CE vs BCE-128
sobre la arquitectura limpia y confirmar el inductivo con K estándar.

---

## Estado actual — 2026-05-20 (sesión 30): loss BCE + neg sampling + self-adversarial implementado (configurable)

### Qué se hizo

Se implementó el loss de KnowFormer (`Knowformer/lightning.py:121-149`) como **opción
configurable** vía `cfg.kgc.loss_fn` (`'ce'` | `'bce'`), sin tocar la arquitectura. Esto
ataca el Sospechoso #1 (loss CE de grafo completo) de forma aislada y reversible.

**Archivos modificados:**
- `loss/losses.py` — nueva función `kgc_bce_neg_sample()`: filter mask por query → muestreo
  de `K = min(N, 2**num_negative_sample)` negativos (`multinomial`, replacement) → BCE por
  candidato sobre `[positivo, K negs]` → pesado self-adversarial `softmax(logits_neg/temp)`
  detached, positivo peso 1 → `loss = (bce*weights).sum()` (escala con batch, como KnowFormer).
- `train/trainer.py` — dispatch en `train_epoch_kgc`: `if cfg.kgc.loss_fn=='bce'` usa la
  nueva función, else el `kgc_full_graph_ce` original (default intacto).
- `config.py` — `cfg.kgc.loss_fn='ce'`, `cfg.kgc.num_negative_sample=7`,
  `cfg.kgc.adversarial_temperature=1.0`.
- `configs/Exphormer/wn18rr_ind_v1_bce.yaml` + `sbatch_wn18rr_ind_v1_bce.sh` —
  `loss_fn: bce`, `num_negative_sample: 8` (256 negs), `temp: 0.5` (valores KnowFormer v1).

**Detalle clave**: `.sum()` (no `.mean()`) → el loss BCE escala con `train_batch_size` y no
es directamente comparable en magnitud al CE. Doc completa en `implementacion_loss_bce.md`.

### Hallazgo del análisis del código KnowFormer (recalibra la hipótesis)

Leyendo `Knowformer/README.md` + `lightning.py`:
- **KnowFormer usa CE de grafo completo en WN18RR ind v3/v4 y NELL v2** y obtiene ~0.67.
  → La premisa "ningún paper KGC inductivo usa CE de grafo completo" (Sospechoso #1 del
  `.tex`/sesión 29) es **falsa**. El CE-global no es inherentemente fatal en inductivo.
- **KnowFormer entrena a LR constante 5e-3, sin warmup, `MultiStepLR([10,15], 0.1)`, Adam.**
  Es 6× nuestro peak (8e-4), constante desde el step 1, y no colapsa. → Refuta tanto
  "alcanzar el peak dispara el colapso" como "8e-4 es demasiado alto". La causa del colapso
  apunta de vuelta a la **arquitectura** (`V=h` + BF residual, sesiones 22-25), no al loss
  ni al schedule en aislamiento.

### Verificación
Smoke tests exit 0: bce (loss=12.05), ce default (loss=7.86, sin regresión), config bce
(loss=12.26).

### Resultado del run BCE — `wn18rr_ind_v1_bce.yaml` (1×H100 local, 30 ep, ~45 min)

Log: `logs/wn18rr_ind_v1_bce_local_20260520_193341.out`. Arquitectura **actual (v5)**, no
el baseline simple. num_neg=8 (256 negs), temp=0.5, cosine-warmup-a-8e-4.

| ep | val_mrr | test_mrr | LR | train_loss |
|----|---------|----------|-----|-----------|
| 0 | 0.005 | 0.008 | 0.0 | 12.31 |
| **1** | **0.207** | **0.236** | 2.7e-4 | 9.92 |
| 2 | 0.168 | 0.231 | 5.3e-4 | 9.46 |
| 3 | 0.176 | 0.216 | 8.0e-4 (peak) | 9.36 |
| 4 | 0.201 | 0.126 | 7.97e-4 | 9.24 |
| 5 | 0.171 | 0.062 | 7.89e-4 | 9.11 |
| 10 | 0.129 | 0.037 | 6.7e-4 | 8.65 |
| 20 | 0.120 | 0.023 | 2.4e-4 | 7.29 |
| 29 | 0.114 | 0.023 | 2.7e-6 | 6.47 |

**Best por val: ep1, test MRR = 0.236.** (vs CE en misma arq ~0.335 sesión 29; vs baseline
simple 0.58.)

### Dos conclusiones fuertes

1. **BCE NO arregla el colapso — lo empeora.** Mismo patrón ep1→colapso, y peor que CE
   (0.236 vs 0.335) en la misma arquitectura. El loss no era la cura. Confirma lo que el
   código de KnowFormer ya anticipaba (usa CE-global en v3/v4 y va bien → el loss no es la
   palanca).

2. **La curva de train resuelve overfit-vs-divergencia → es OVERFIT/SHORTCUT, no
   divergencia.** `train_loss` baja monótona toda la corrida (12.31 → 6.47) mientras
   `test_mrr` colapsa (0.236 → 0.023). Si fuera divergencia de optimización, el train loss
   también explotaría — no lo hace. Y el patrón inductivo es nítido: train↓, **val**
   (entidades de train) aguanta ~0.11-0.20, **test** (entidades disjuntas) colapsa duro.
   Clave: al bajar el LR de vuelta (ep20-29, LR 2.4e-4 → 2.7e-6) el test **NO se recupera**
   → no es un pico transitorio de LR alto; una vez sobreajustada, la representación queda
   rota. El modelo se aferra a estructura del grafo de train.

**Implicación**: el colapso inductivo es **arquitectónico** (overfitting a la estructura
del grafo de train, consistente con `V=h` + BF residual de sesiones 22-25), NO del loss ni
del schedule. Atacar el loss fue refutado empíricamente.

### Próximos pasos
1. **(Pendiente para aislar limpio)** correr BCE sobre el baseline SIMPLE de 0.58 (no la
   arq v5) — confirmar que tampoco ayuda ahí. Esperado: igual o peor.
2. Atacar la raíz arquitectónica: el overfitting estructural. Regularización fuerte
   (dropout alto, weight decay), o reducir capacidad del path entity-específico
   (`V=h`/FFN), o early-stopping agresivo (best ya cae en ep1).
3. Run con schedule KnowFormer (Adam + LR constante + MultiStepLR) sigue valiendo como
   diagnóstico de schedule, pero ya con menos prioridad dado que el LR decreciente no
   recupera el test.

---

## Estado anterior — 2026-05-05 (sesión 29): Q/K stream falla — análisis honesto del patrón de 5 sesiones

Tienes razón. Voy a ser honesto en vez de proponer otra "solución".

### Lo que muestran los datos del log 599923

| ep | LR | val MRR | test MRR |
|----|-----|---------|----------|
| 0 | 0 | 0.004 | 0.018 |
| **1** | 1e-5 | **0.237** | **0.335** |
| 2 | 2e-5 | 0.225 | 0.307 |
| 3 | 3e-5 | 0.237 | 0.298 (best por val) |
| 4 | 4e-5 | 0.207 | 0.213 |
| 5 | 5e-5 | 0.191 | 0.166 |
| 6 | 6e-5 | 0.175 | 0.134 |
| 7 | 7e-5 | 0.186 | 0.152 |

**Mejor test 0.335 al ep1, luego colapso. Peor que el 0.58 baseline.** Mi predicción de 0.65+ fue incorrecta.

### La verdad incómoda que llevamos 5 sesiones evitando

Mira el patrón a lo largo de todas tus sesiones:

| Arquitectura | Mejor ep | Mejor test MRR |
|---|---|---|
| K=proj_k(q) only (simple, sesión 5) | ep4 | **0.578** |
| novw (sin W_V, gate aditivo) sesión 23 | ep1 | **0.5802** |
| novw + C2 bilinear | ep1 | 0.4315 |
| V-NBF v1/v2 | varios | 0.13-0.40 |
| V-NBF v4 (smoke test only) | — | — |
| **+ Q/K stream (esta)** | ep1 | **0.335** |

**Cada "mejora" arquitectónica con más capacidad ha sido peor que las versiones simples**. No es un caso aislado, es el patrón consistente de las últimas 5 sesiones.

Y el patrón de colapso ep1→ep2 también es invariante. Está en TODAS las arquitecturas. Eso significa que el colapso no lo causa V, ni Q/K, ni el gate, ni el FFN. Lo causa **algo común a todas las configs** — y mis análisis arquitectónicos han estado mirando el lugar equivocado durante 5 sesiones.

### Lo que NBFNet hace que invalida mi última teoría

Mi argumento del "Paso 1" decía que Q/K constantes eran el problema. Pero **NBFNet no tiene Q/K en absoluto** — es solo `BellmanFord(DistMult, sum)`. Y obtiene 0.741. Más simple que la 0.58 tuya. Si el problema fuera Q/K, NBFNet no llegaría tan alto. Mi diagnóstico era plausible pero estructuralmente no se sostenía.

### Qué propongo ahora — y NO es otra modificación a ciegas

Propongo cambiar el método, no la arquitectura. Tres pasos concretos:

**1. Volver al baseline 0.58 verificado.** Identificar el commit exacto (`0466cde` "trans MRR 0.566" o uno cercano) que da 0.58 inductivo. Lanzar un run para confirmar que sigue dando 0.58 en el código actual o en checkout. Esto fija el suelo. Sin un baseline reproducible no podemos medir nada.

**2. Instrumentar el colapso.** En el run de baseline, registrar epoch a epoch:
- `||W||_2` por capa (¿qué pesos crecen explosivamente?)
- Distribución de scores de atención (¿se saturan al clamp ±5?)
- Magnitud de `h` por nodo en train graph vs test graph (¿hay distribution shift visible?)
- Top-10 nodos predichos para 10 queries fijas a lo largo de épocas (¿qué cambia entre ep1 que va bien y ep3 que ya está cayendo?)

Sin estos datos cualquier teoría es especulación. Llevamos 5 sesiones especulando.

**3. Decidir basándose en los datos.** Solo después de tener instrumentación, cualquier cambio arquitectónico tiene una hipótesis testeable concreta — no "creo que esto funciona porque KnowFormer hace algo parecido".

### Cosas que probablemente NO son el problema (y quitar de la lista)

- Q/K constantes (NBFNet no las tiene y funciona)
- Estructura del V (V-RMPNN, V-NBF, V plain — todas colapsan)
- BF residual presence/absence (sesión 25 confirmó que ambos colapsan)
- Gate aditivo vs bilineal (C2 empeoró, sin C2 también colapsa)

### Cosas que NO hemos investigado seriamente

- **Loss function**: usas full-graph CE sobre 2746 nodos. NBFNet usa negative sampling (típicamente 32-64 negativos). KnowFormer usa neg sampling. **No hay paper KGC inductivo que use full-graph CE**. Esto puede estar generando un gradiente patológico cuando LR sube — el modelo intenta empujar las 2745 puntuaciones no-anchor a -∞ y rompe la representación.
- **LR schedule peak**: cosine sube de 1e-5 (warmup) a 1e-4 (peak). El colapso siempre ocurre cuando LR alcanza peak. ¿Es coincidencia, o es la verdadera causa? Un run con LR constante = 1e-5 confirmaría esto en una corrida.
- **Drop direct edges**: NBFNet documenta que ESTO es lo que evita memorización. ¿Está implementado correctamente en tu trainer? No lo he verificado.

¿Quieres que parta por (a) verificar que el drop direct edges está bien hecho — un grep rápido al trainer — y (b) lanzar un run con LR constante = 1e-5 sobre el baseline 0.58? Son dos diagnósticos que pueden contradecir o validar mi sospecha actual sin tocar arquitectura. Si LR constante no colapsa, el problema es schedule, no estructura. Si drop edges está mal, eso solo puede explicar mucho.

Si prefieres revertir lo del Q/K stream y dejar el código como estaba antes de hoy, dime y lo hago primero.

---

## Estado actual — 2026-05-02 (sesión 28): V-NBF v4 (KnowFormer-aligned) implementado — smoke test OK

### Implementación

Todos los cambios derivados directamente del análisis del código KnowFormer (`analisis_vnbf_vs_knowformer.md`).

**`layer/exphormer.py`** — `ExphormerAttention`:
- Eliminado `batch.v_x_state` (chaining entre outer layers) — V ahora arranca desde ceros frescos cada outer layer
- Anchor: `v_x[anchor] = 1.0` (one-hot estructural) en lugar de `= query_emb`
- Agregado `self.fc_v_x = nn.Linear(in_dim * 2, in_dim, bias=False)` — mezcla h acumulado con v_x
- Agregado `self.v_nbf2 = VLayerNBF(in_dim, num_relation_slots)` — segunda iteración NBF interna
- Forward V-NBF:
  ```python
  v_x = h.new_zeros(num_node, d)            # fresh zeros
  v_x[anchor_global] = 1.0                  # one-hot estructural
  v_x = fc_v_x(cat([h[:num_node], v_x], -1)) # mix con h
  v_x = v_nbf(v_x, edges, rels, q, eg)     # 1ª iteración
  v_x = v_nbf2(v_x, edges, rels, q, eg)    # 2ª iteración
  ```

**`network/model.py`** — `MultiLayer.forward()`:
- Eliminado BF residual `h = h + batch.x0`
- Comentario actualizado explicando por qué: ruta de gradiente multiplicativa causaba colapso ep1→ep2

**Params**: 1,407,553 (vs 977,473 previo; +430K por fc_v_x × 5 + v_nbf2 × 5)

**Smoke test**: exit 0, ep0 test MRR=0.017 (random, esperado — sin BF residual y fc_v_x sin entrenar no hay señal gratis)

**Config y script**:
- `configs/Exphormer/wn18rr_ind_v1_vnbf4_lr1e4.yaml` — idéntico a vnbf3 (L=5, d=64, lr=1e-4, wu=10, 30 ep)
- `sbatch_wn18rr_ind_v1_vnbf4_lr1e4.sh` — 1 H100, 12h

### Por qué estos cambios (vs. V-NBF v1/v2 fallidos)

| Problema en v1/v2 | Solución en v4 |
|---|---|
| V encadenado entre outer layers (acumula entidades) | V desde zeros frescos cada outer layer |
| `v_x[anchor] = query_emb` → triple ruta de gradiente | `v_x[anchor] = 1.0` → single ruta (via fc_z) |
| Sin mezcla con h | `fc_v_x(cat([h, v_x]))` |
| Solo 1 iteración NBF | 2 iteraciones (como KnowFormer num_v_layer=2) |
| BF residual activo | Eliminado |

### Resultados de V-NBF v1/v2 (sesiones 26-27, para no repetir)

| Job | Versión | Mejor test MRR | Motivo del fallo |
|-----|---------|----------------|-----------------|
| 597371 | vnbf (lr=8e-4) | 0.133 | colapso inmediato |
| 597378 | vnbf (lr=1e-4, no chained) | 0.334 | colapso ep2+ |
| 597380 | vnbf2 (lr=1e-4, chained) | 0.396 | colapso ep2+ |
| 597381 | vnbf2_d32 | 0.213 | cancelado |

Causa raíz documentada en `analisis_vnbf_vs_knowformer.md`.

### Próximos pasos (si vnbf4 supera 0.5802)
1. Probar con lr=8e-4 (schedule original que dio 0.5802)
2. Probar en WN18RR ind v2, v3, v4
3. Versión d=32 para comparación de capacidad

---

## Estado actual — 2026-04-30 (sesión 26): V-NBF stream implementado — smoke test OK

### Implementación

**Cambio arquitectónico central**: V ya no es `h^{t-1}` (acumulado). Se reemplazó por un NBF stream fresco cada outer layer + eliminación del BF residual.

**`layer/exphormer.py`** — nueva clase `VLayerNBF` + cambios en `ExphormerAttention`:
- `VLayerNBF(d, num_relation_slots)`: scatter DistMult en 1 iteración sobre KG∪Expander
  - `fc_z`: Linear(d, (R+1)*d, bias=False) — factor de query por relación, std=0.01
  - `forward`: `out[v] = Σ_{(u,r,v)} fc_z(q).view(B,R+1,d)[b,r] ⊙ v_x[u]`
- `ExphormerAttention.__init__`: reemplazado `self.norm_V` (C3 eliminado) por `self.v_nbf = VLayerNBF(in_dim, num_relation_slots)`
- `ExphormerAttention.forward`: V-NBF stream dentro del bloque `use_query_conditioning`:
  ```python
  v_x = zeros(num_node, d)
  v_x[anchor_global] = query_emb   # fresh cada outer layer
  v_x = self.v_nbf(v_x, edge_index, batch.edge_rel_idx, query_emb, edge_graph)
  V_h = v_x
  ```

**`network/model.py`** — `MultiLayer.forward()`:
- Eliminado BF residual: `if hasattr(batch, 'x0'): h = h + batch.x0` → removido
- Comentario actualizado: "BF residual removed — anchor re-injected fresh via VLayerNBF"

**Params**: 977,473 (novw+C2 base 588K + 5 × VLayerNBF 78K = exacto)

**Smoke test**: exit 0, loss=7.92 @ ep0, eval corre sin errores.

**Config y script**:
- `configs/Exphormer/wn18rr_ind_v1_vnbf.yaml` — idéntico a novw_c1 (L=5, d=64, lr=8e-4, wu=3, 30 ep)
- `sbatch_wn18rr_ind_v1_vnbf.sh` — 1 H100, 12h

### Invariantes preservados
- Q, K: anclados a x0 (boundary condition relacional, C1)
- Gate: bilinear (r_uv × r_q), C2
- Atención dispersa O(|V|+|E|) sobre KG∪Expander
- FFN activo (C4 descartado — demostró que FFN da estabilidad)

### Qué cambió respecto a todo lo anterior
| Componente | Antes (C1+C2) | Ahora (V-NBF) |
|---|---|---|
| V | `norm_V(h^{t-1})` | `VLayerNBF(zeros+anchor_fresh, KG∪Exp, q)` |
| Anchor injection | KGCNodeEncoder (una vez) + BF residual cada capa | KGCNodeEncoder (x0 para Q/K) + fresco en V-NBF cada capa |
| BF residual | `h += x0` al final de cada MultiLayer | **eliminado** |

### Próximos pasos (si V-NBF supera 0.5802)
1. Añadir mezcla con x acumulado: `v_x = fc_v_in(cat([x, v_x_zeros]))` antes del scatter
2. Aumentar a 2 iteraciones NBF internas
3. Probar en WN18RR inductivo v2, v3, v4

---

## Estado actual — 2026-04-30 (sesión 25): C1+C2+C3+C4 todos fallan — re-diagnóstico estructural profundo

### Resumen ejecutivo

Se implementaron y testearon los 4 cambios propuestos en `diagnostico_solucion_inductivo.md`. **Ninguno mejoró el baseline novw (0.5802 test MRR)**. El análisis post-fallo identificó que el diagnóstico previo era incorrecto en su premisa: la causa raíz no es `K(h)` ni el FFN ni el gate aditivo — es `V = h^{t-1}` combinado con el BF residual `h += x0`.

### Implementaciones de esta sesión

**C1 (`layer/exphormer.py`):**
```python
# Antes: K_h = self.K(h)
K_h = self.K(h_q)  # C1: anchored to x0, not h — eliminates entity-specific routing
```
`h_q = x0` cuando `use_query_conditioning=True` y `batch.x0` existe. Con esto K nunca lee `h^{t-1}`.

**C2**: ya estaba en el código desde sesión 24 (gate bilinear `gate_base[r] + fc_zq(q)[r]`). C1+C2 se testean juntos.

**C3 (`layer/exphormer.py`, `__init__`):**
```python
self.norm_V = nn.LayerNorm(in_dim, elementwise_affine=False)
# forward:
V_h = self.norm_V(h) if self.use_query_conditioning else self.V(h)
```
Pre-LayerNorm sobre V (sin affine, relational-safe). Objetivo: acotar magnitud del gate bilinear chain.

**C4 (`network/model.py`, `config.py`):**
- `cfg.gt.use_ffn = True` (nuevo flag en config)
- `MultiLayer.__init__` acepta `use_ffn=True`; si False, no crea ni aplica el bloque FFN
- `MultiModel` pasa `use_ffn=getattr(cfg.gt, 'use_ffn', True)` a cada MultiLayer

**Configs y scripts creados:**
- `configs/Exphormer/wn18rr_ind_v1_novw_c1.yaml` (C1+C2)
- `configs/Exphormer/wn18rr_ind_v1_novw_c1c3.yaml` (C1+C2+C3)
- `configs/Exphormer/wn18rr_ind_v1_novw_c1c4.yaml` (C1+C2+C4, `use_ffn: False`)
- `sbatch_wn18rr_ind_v1_novw_c1.sh`, `c1c3.sh`, `c1c4.sh`

### Resultados

| Run | Job | Config | ep1 val | ep2 val (best) | ep2 test | ep3 val | nota |
|-----|-----|--------|---------|----------------|----------|---------|------|
| baseline novw | 596512 | novw | 0.4803 | — | **0.5802** | colapso | sesión 23 |
| C1+C2 | 597127 | novw_c1 | 0.3995 | **0.4245** | **0.5191** | 0.2788 | colapso ep3 |
| C1+C2+C3 | 597128 | novw_c1c3 | ~0.40 | **0.4245** | **0.5191** | colapso | cancelado ep7 |
| C1+C2+C4 | 597132 | novw_c1c4 | — | **0.4339** | — | colapso ep2 | cancelado |

**Ninguna combinación superó 0.5802.** Patrón de colapso idéntico en todos: pico ep1-2 durante warmup, caída catastrófica al alcanzar LR peak (ep3, LR=8e-4).

- C3 (pre-LN sobre V): doble normalización. MultiLayer ya tiene LayerNorm post-attention; agregar otra antes quita escala que el BF residual usa para distinguir distancia del anchor.
- C4 (no FFN): colapso más temprano (ep2 vs ep3). FFN da estabilidad dinámica, no solo memoriza entidades.
- C1 (K relacional pura): no movió nada porque V = h^{t-1} sigue siendo entity-accumulated.

### Re-diagnóstico estructural — causa raíz

**La premisa del diagnóstico anterior estaba equivocada.** `diagnostico_solucion_inductivo.md` identificaba `K(h)`, `FFN(h)`, y el gate aditivo como los violadores. Pero C1 eliminó `K(h)` y no hubo mejora. Razón: **V sigue siendo `h^{t-1}`**.

Análisis de KnowFormer (código real, `Knowformer/src/model.py`):

1. **KnowFormer NO tiene BF residual** (`h += x0`). Nosotros sí, en `MultiLayer.forward()`. Diferencia estructural fundamental no identificada antes.
2. **KnowFormer arranca con `x = 0` global**, sin INDICATOR al inicio. El head se inyecta **fresco cada outer layer** vía `v_x[h_index] = 1` (one-hot), no vía residual.
3. **Q, K, V vienen de NBF streams frescos**, no de `h` acumulado. Cada outer layer lanza dos NBF internos (2 iteraciones) desde ceros, mezclan `x` como contexto solo en el primer paso.

Nuestro flujo post-C1:
```
Q = W_Q(x0) + proj_q(q)   # relacional puro ✓
K = W_K(x0) + proj_k(q)   # relacional puro ✓ (C1)
V = h^{t-1}                # ← CAUSA RAIZ: entity-accumulated, no relacional
h^t = h^t + x0             # ← AGRAVA: BF residual refuerza patrones train-específicos
```

`V = h^{t-1}` transfiere patrones de entidades del grafo de train. El BF residual `h += x0` los refuerza cada capa. C1-C4 trataron síntomas; la raíz es V.

Previo intento V-RMPNN (sesión 14, 0.513) falló porque mantuvo el BF residual: dos mecanismos de inyección del anchor (V-NBF fresco vs BF residual con x0 fijo) se contradicen, el optimizer no los reconcilia.

### Solución propuesta (próximo paso)

**V-NBF stream con fresh anchor injection cada outer layer, eliminar BF residual.**

Pseudocódigo del cambio central:
```python
# Cada outer layer:
v_x = zeros(N, d)
v_x[anchor] = query_emb[batch_idx]       # FRESH cada layer (no via BF residual)
v_x = fc_v_in(cat([x, v_x], -1))         # mezcla con x acumulado
for nbf in v_nbf_layers:                  # 1-2 iteraciones NBF relacionales
    v_x = nbf(v_x, edge_index, edge_attr, query_emb)

Q = W_Q(x0) + proj_q(q)                  # relacional puro (C1)
K = W_K(x0) + proj_k(q)                  # relacional puro (C1)
attn_out = sparse_attn(Q, K, V=v_x, E, gate, edges=KG∪Expander)

x = x + attn_out                          # SIN h += x0
x = LN(x); x = x + FFN(x); x = LN(x)    # KnowFormer también tiene FFN
```

Se preserva la contribución de tesis (expander como topología de atención), el gate bilinear (C2), y la atención dispersa O(|V|+|E|).

Análisis completo escrito en `analisis_arquitectura_inductivo.md`.

### Archivos modificados en esta sesión

- `layer/exphormer.py` — C1 (K→x0) + C3 (norm_V) + docstring actualizado
- `network/model.py` — C4 (use_ffn flag en MultiLayer + MultiModel)
- `config.py` — `cfg.gt.use_ffn = True`
- `analisis_arquitectura_inductivo.md` — análisis arquitectural completo (nuevo archivo)
- configs y sbatch: `novw_c1`, `novw_c1c3`, `novw_c1c4`

---

## Resumen sesiones 17–24 (2026-04-24 a 2026-04-29) — refactor, "novw" y los intentos C1-C4

> Detalle completo en git history. Las sesiones 25-31 (abajo) conservan su detalle porque
> contienen los experimentos V-NBF / C1-C4 / reset que aún se citan activamente.

### Sesiones 17-20 — regresión de velocidad del refactor (2.7×)
La unificación de las tablas de embedding en una sola `query_rel_emb` (sesión 18) introdujo una
regresión de 2.7× (1.03s → 2.81s/iter) bajo gradient checkpoint. **Causa raíz**: un tensor
`requires_grad=True` (`batch.query_emb`) adjunto al objeto Batch queda retenido como activación a
través de los L replays del backward con `use_reentrant=False`, agravado por un matmul
`proj_exp_edge` sobre ~3.9M aristas expander. Bench de 1 GPU confirmó que era el refactor (no DDP
ni entorno: el código de `0466cde` corría a 0.96s). **Resuelto definitivamente en sesión 31**: el
reset restauró las tablas separadas por componente (vuelta a `0466cde`).

### Sesión 21 — ablación FiLM
Clarificado que FiLM (`x*(1+scale)`) solo se aplicaba a **E**; Q/K usan bias aditivo y el V-gate es
multiplicativo sin residual identidad. Ablación `use_film_e` lanzada en transductivo.

### Sesión 22 — por qué KnowFormer generaliza inductivo
Análisis del código de KnowFormer: Q/K/V salen de **streams NBF frescos desde ceros** (con ruido),
nunca leen `h` acumulada → puramente relacional y transferible; nuestro `W_K(h)`/`W_V(h)` memorizan
la distribución de entidades del train. Los experimentos de **ruido gaussiano** (noise_std
0.5/1.0/2.0) FRACASARON (~0.02 MRR): el ruido se acumulaba T veces vía el residual BF y aplastaba la
señal del ancla. FiLM eliminado del codebase.

### Sesión 23 — Cambio 1 (sin W_V, "novw")
Eliminar `W_V` (V = h directo) dio el mejor inductivo histórico: **0.5802 test MRR @ ep1** (lr 8e-4),
pero con colapso al alcanzar el peak LR (mismo patrón pico-en-warmup → crash). Diagnóstico: sin la
amortiguación de `W_V`, el peak LR rompe los patrones relacionales aprendidos en warmup.

### Sesión 24 — C2 (gate bilinear) aislado FALLA
Gate bilinear `gate_base[r] + fc_zq(q)[r]` (estilo KnowFormer) en aislado EMPEORÓ a 0.4315 (vs 0.5802
de novw). **Conclusión metodológica**: C2 sin C1 es contraproducente — dar más capacidad al gate
mientras `W_K(h)` sigue activo le da al optimizer más dimensiones para crear shortcuts
train-específicos. Las propuestas C1-C4 de `diagnostico_solucion_inductivo.md` debían testearse
acumulativamente con C1 (K relacional) como base, no aisladas. (Las sesiones 25-28 probaron C1-C4 y
V-NBF v1-v5 — todas fallaron; ver detalle abajo.)
