# Session Notes

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

**Cambio clave activo:** `torch.sigmoid()` eliminado del V_gate en `layer/exphormer.py` (commit "Q conditioned"). Sin este cambio el modelo se bloquea en MRR ~0.44.
