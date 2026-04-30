# Diagnóstico: dos problemas distintos con una causa común

## El problema NO es uno solo

Has estado tratando "decay" y "ceiling" como un solo problema. Son dos:

**Problema A — decay (colapso post-warmup)**: con novw, el modelo aprende rápido en ep1-2, luego colapsa al llegar al peak LR. Estructural, no de schedule.

**Problema B — ceiling (~0.58 vs 0.75 target)**: aún en el mejor punto, estás 0.17 MRR debajo de KnowFormer. Estructural, no de hiperparámetros.

Tienen **una causa común que explica ambos**, pero requieren **dos correcciones diferentes**. Por eso los parches independientes (warmup, lr, label smoothing) no funcionan: no atacan la causa estructural, y no separan los dos problemas.

## La regla estructural única

> **Una arquitectura inductiva-por-construcción no puede tener `Linear(h_acumulada)` en ningún path que defina el ROUTING o el GATING. Solo puede tener `Linear` sobre cantidades relacionales (x0, query_emb, edge_emb).**

`h` después de t capas codifica "el contexto propagado desde el anchor hasta este nodo, vía estas entidades específicas, en este grafo". `Linear(h)` aprende a proyectar las direcciones de h que son útiles **en el grafo de train**. En el grafo inductivo, las mismas dimensiones de h codifican distribuciones distintas → la proyección aprendida apunta al lado equivocado.

Las únicas operaciones que pueden tomar h sin romper inductivo son:
- Identidad / residual (`h + msg`)
- Normalización con stats per-graph (LayerNorm sin affine, o BatchNorm cuidadoso)
- Suma con cantidades relacionales (`h + x0`)
- Producto Hadamard con factor relacional (`h ⊙ z_q[r]` — DistMult NBF)

KnowFormer **no tiene un solo `Linear(h_t-1)` en su path de Q/K/V**. Verifícalo: en `KnowformerLayer.forward`, los tres streams parten de `x` (la h acumulada) pero **inmediatamente** la mezclan con primitivas relacionales (`fc_qk_x(cat([x, qk_x]))`, `fc_v_x(cat([x, v_x]))`) y luego propagan vía `generalized_rspmm` (relational-aware aggregation). Las MLPs internas operan sobre el resultado de la propagación relacional, no sobre h directa.

Ahora aplicamos la regla a tu arquitectura:

| Componente | Lee `h_t-1`? | Veredicto |
|---|---|---|
| `Q = self.Q(x0) + proj_q(query_emb)` | NO (lee x0) | ✅ inductivo-safe |
| `K = self.K(h) + proj_k(query_emb)` | **SÍ** | ❌ memoriza routing de train |
| `V_h = h` (novw) | SÍ pero solo como contenido | ⚠️ OK si se controla magnitud |
| `gate = W_g(emb_uv) + proj_vg(query_emb)` | NO | ✅ pero **es aditivo, no bilinear** |
| `score = exp(clamp(Q*K*E))` | A través de K | ❌ K(h) contamina |
| `FFN(h)` después de attn | **SÍ** | ❌ aprende transformación de la distribución de h en train |
| `LayerNorm` con elementwise affine | A través de scale/bias | ⚠️ aprende un re-scaling que es per-feature de train |

**Tres componentes violan la regla**: K, FFN, y la rama de score que depende de K. Agrega que el gate, aunque relacional, es solo aditivo y no captura la interacción cruzada `r_uv × q`.

## De aquí salen las dos causas

### Causa de A (decay)

Sin `W_V`, el camino de gradiente desde la loss hasta `query_rel_emb` es:

```
loss → score → KGCHead.scorer(h, query_emb)
              ↑
              h^(T) = ... + h^(T-1)*gate*score + ... + x0
                                ↑                    ↑
                                gate(r,q)            query_emb (anchor)
```

`h^(T)` depende de `query_emb` por **dos rutas multiplicativas**: (1) anchor x0 reinjectado en cada capa via residual BF; (2) propagación a través del gate `proj_vg(query_emb)` en cada capa.

Con W_V, el `Linear(h)` actuaba como **buffer de magnitud** — la proyección lineal absorbe parte del gradiente y bound los activations. Sin W_V, la cadena multiplicativa de gates a través de T=5 capas se vuelve muy sensible al LR. Cuando el optimizer empuja gates fuerte, los productos a 5 capas explotan o colapsan a cero — y eso destruye la propagación relacional aprendida durante el warmup. Es un problema clásico de **gradiente explosivo en cadena multiplicativa profunda sin normalización a nivel de mensaje**.

Esto NO se arregla con warmup más largo (job 596722 lo confirmó: el threshold de colapso es estructural, ~LR=2.4e-4 con T=5).

### Causa de B (ceiling)

Tres componentes que aprenden funciones específicas del grafo de train:

1. **K = W_K(h)**: ya identificado. Cuando lo eliminaste (K solo-query), subiste de 0.486 → 0.578. Te queda `W_K(h)` activo en novw, contribuyendo al ceiling.

2. **gate aditivo `W_g(r) + proj_vg(q)`**: solo tiene rango 1 en el cross `(r, q)`. Dos queries que deberían pesar la misma arista de manera distinta solo pueden diferir por una constante per-query. KnowFormer usa `fc_z(q) → matriz (R, d)` indexada por r — **rango completo en el cross**.

3. **FFN(h)**: aprende `W_2 ReLU(W_1 h)` sobre la distribución de h del grafo de train. En inductivo, h tiene distribución diferente y el FFN proyecta a las direcciones equivocadas. Esto es harder to detect que K pero mismo problema.

# Solución arquitectónica (no parche)

La solución es **una sola idea, expresada en cuatro cambios coherentes que comparten justificación**:

> **Hacer que TODA función diferenciable que toque `h_t-1` lo haga solo como contenido de mensajes, nunca como driver de routing/gating, y reemplazar cualquier `f(r) + g(q)` aditivo por una bilinear `M_q[r]`**.

## Cambio 1 — K relacional pura (no `W_K(h)`)

**Qué**: `K = self.K(x0) + proj_k(query_emb)` en lugar de `K = self.K(h) + proj_k(query_emb)`.

**Por qué**: x0 es la INDICATOR (anchor=query_emb, no-anchor=0). `Linear(x0)` es relational-only por construcción: solo el anchor tiene un K específico, los no-anchor tienen K = `proj_k(q)` constante. Esto reproduce la semántica NBFNet en el routing.

**Justificación**: ya validado parcialmente (K solo-query dio 0.578). Pero K=`Linear(x0)` es estrictamente más expresivo: el anchor obtiene un K diferente del resto, lo que ayuda a la atención a distinguir mensajes que vienen del anchor vs los que vienen de paths intermedios.

**Costo en transductivo**: hay que verificar empíricamente. La hipótesis es que en transductivo, K no necesita memorizar routing porque las entidades son las mismas en train y test. Si pierde algo, puede recuperarse con T mayor o d mayor (cambio de hiperparámetro, no estructural).

## Cambio 2 — Gate bilinear `M_q[r_uv]` en lugar de aditivo

**Qué**: reemplazar
```python
gate = self.V_gate(edge_attr) + self.proj_vg(shared_edge)   # aditivo, rango-1
```
por
```python
M = self.fc_zq(query_emb)                  # (B, R*d)
M = M.view(B, R, d)                        # (B, R, d)
gate = M[edge_graph_idx, edge_rel_idx]     # (E, d) — bilinear lookup
```

donde `edge_rel_idx` ya existe en `ExpEdgeFixer` (incluye sentinel para expander edges).

**Por qué**: el gate aditivo es matemáticamente equivalente a aplicar dos biases independientes — no puede expresar "la relación r1 importa MUCHO para la query q1 pero POCO para q2". El bilinear sí. KnowFormer lo usa exactamente así (`fc_z(z) → (R, d)`).

**Costo computacional**: 1 lookup por arista — barato. Parámetros adicionales: `R × d × d_q` en `fc_zq` (≈ 18 × 64 × 64 = 73K en WN18RR). Manejable.

**Justificación más profunda**: con `gate = M_q[r_uv]`, el mensaje pasa a ser `msg = h_u ⊙ M_q[r_uv]`. Esto es **literalmente DistMult con la matriz de relación condicionada por la query** — la formulación canónica de mensajes NBFNet (papers_distilled.md sección 1: "DistMult > RotatE > TransE"). Has estado a un paso de NBFNet sin haberlo formalizado.

## Cambio 3 — Pre-LayerNorm sobre V para estabilizar magnitud

**Qué**: agregar
```python
V_h = self.norm_V(h)   # nn.LayerNorm(d) sin elementwise_affine, o con affine compartido
```
antes de usar h como mensaje, en modo `use_query_conditioning=True`.

**Por qué**: con V=h y gate multiplicativo a través de T capas, los productos en cadena pueden explotar (causa A — decay). La normalización **per-mensaje** acota la magnitud sin destruir la información direccional. Esto es lo que hace KnowFormer internamente en cada `KnowformerVLayer` (`norm = LayerNorm` antes del shortcut).

**Justificación más profunda**: una LayerNorm sin affine es totalmente relational-safe (no aprende sesgos de distribución de train, solo estandariza). Con affine, hay un riesgo menor pero significativamente menor que el FFN.

## Cambio 4 — Eliminar el FFN o condicionalo por query

**Qué**: dos opciones, en orden de preferencia:

  - **(4a)** Eliminar el FFN dentro de `MultiLayer.forward()`. Mantener solo: `h ← h + attn(h) + h^(0)`, con LayerNorm post-attn.
  - **(4b)** Si 4a sacrifica transductivo, condicionar el FFN por la query: `FFN_q(h) = W_2 ReLU((1 + γ_q) ⊙ W_1 h + β_q)` con `γ_q = fc_γ(query_emb)`, `β_q = fc_β(query_emb)`. Esto es FiLM aplicado al FFN (no a E).

**Por qué eliminar (4a)**: el FFN actual `Linear(2d) → ReLU → Linear(d)` aprende una transformación de h's distribución. KnowFormer **sí tiene FFN** pero lo aplica sobre x DESPUÉS de que x absorbió la información de los streams Q/K/V relacionales — es decir, la entrada del FFN ya es relacional. En tu modelo, la entrada del FFN es h con todo su contenido entity-específico.

Eliminar el FFN parece radical, pero la arquitectura sigue teniendo expresividad: la atención sparse con gate bilinear ya captura las composiciones de relaciones; el FFN agrega capacity que en KGC es probablemente redundante con la composición vía capas.

**Justificación de (4b)**: si decides mantener FFN, condicionarlo por query es la versión "FiLM bien aplicada". El FFN actual es independiente de la query — aprende UNA transformación universal sobre h. Pero h tiene significado distinto para cada query (contiene "el path desde anchor para esta query particular"). Hace más sentido que el FFN dependa de la query.

# Por qué esto NO es un parche

Cada cambio anterior se justifica desde **la misma regla estructural**. No es "probemos esto y veamos". Es:

- C1: `Linear(h)` en routing → `Linear(x0)` en routing → K se vuelve relational-safe
- C2: gate aditivo (rank-1 cross) → gate bilinear (rank completo) → captura interacción r×q
- C3: cadena multiplicativa profunda inestable → LN per-mensaje → magnitud acotada sin perder dirección
- C4: `Linear(h)` global en FFN → eliminado o condicionalizado por query → FFN deja de memorizar distribución de train

Los cuatro están fundamentados en una sola observación: **toda función aprendida sobre h que no sea content-only debe ser relational-conditioned o eliminada**.

El expander queda intacto — sigue siendo el backbone topológico O(N) que es tu contribución de tesis. La atención sparse sobre `(KG ∪ expander)` es la pieza estructural que NO se toca.

# Plan de implementación priorizado

Te propongo este orden, con el criterio de "qué cambio testeo aislado primero":

**Fase 1 — Validar el diagnóstico estructural (3 experimentos cortos, 1 GPU, ind v1):**

1. **C2 solo (gate bilinear)** sobre la arquitectura novw actual. Hipótesis: sube de 0.58 → ~0.65. Si SÍ → confirmamos que el cross-term era el bottleneck principal del ceiling.
2. **C1 + C2** (K relacional + gate bilinear). Hipótesis: sube otro escalón a ~0.68-0.70.
3. **C1 + C2 + C3** (más pre-LN sobre V). Hipótesis: estabiliza el decay y permite usar T=5 sin colapso, MRR ~0.70+.

**Fase 2 — Si Fase 1 confirma:**

4. **C1 + C2 + C3 + C4a** (eliminar FFN). Hipótesis: cierra el resto del gap a ~0.74.

Si C4a regrese transductivo, revertir a **C4b** (FFN condicionado por query).

**Fase 3 — Validación cruzada:**

5. Ejecutar la mejor configuración en **transductivo WN18RR**. Aceptar pérdida pequeña vs 0.566 (digamos 0.55-0.56) si inductivo sube a 0.70+. Si transductivo cae mucho, los cambios se controlan por hiperparámetros (T, d, lr) — no por "deshacer cambios estructurales para transductivo y rehacerlos para inductivo".

# Lo que debes recordar de esto

- **No es buscar un único parche que arregle todo**. Es aplicar una regla estructural consistente a cada componente que la viola. C1, C2, C3, C4 son cuatro aplicaciones de la misma regla.
- **El ceiling y el decay tienen causas relacionadas pero distintas**. C1+C2 atacan el ceiling. C3 ataca el decay. C4 ataca el ceiling residual.
- **Cada cambio tiene un test crítico claro** — si C2 (gate bilinear) NO mueve nada de 0.58, el diagnóstico está mal y hay que revisar. Pero matemáticamente, la diferencia de capacidad expresiva entre rank-1 y rank-completo en el cross-term es enorme y debería ser visible.
