# Diagnóstico real del código actual

## Hallazgo 1: El código NO hace lo que dice

El `exphormer.py` dice `"BF residual ELIMINATED in MultiLayer.forward()"` — pero `model.py` tiene el bloque activo:

```python
if hasattr(batch, 'x0'):
    h = h + batch.x0   # <-- ACTIVO, no eliminado
```

El docstring de V-NBF dice `"fresh anchor injection each outer layer"` — pero el código hace:

```python
if hasattr(batch, 'v_x_state') and batch.v_x_state is not None:
    v_x = batch.v_x_state.clone()   # <-- carga estado contaminado de capa anterior
v_x[anchor_global] = query_emb     # <-- "fresh injection" sobre V ya acumulado
```

El `"fresh injection"` ocurre sobre un V que ya cargó la capa anterior. No es fresco.

## Hallazgo 2: Lo que hace KnowFormer que nosotros NO hacemos

KnowFormer en cada outer layer:

```python
# KnowformerLayer.forward():
v_x = torch.zeros(B, N, d)               # ZEROS REALES cada outer layer
v_x[:, h_index, :] = 1                   # ONE-HOT, sin gradiente de rel_emb aquí
v_x = self.fc_v_x(cat([x, v_x], dim=-1)) # MEZCLA controlada con x acumulado
for layer in self.v_layers:              # 2 iteraciones NBF internas (no 1)
    v_x = layer(v_x, z, r_index, graph)
# ... sin BF residual
```

Tres diferencias estructurales con respecto a lo que implementamos:

| | QC-Exphormer (estado actual) | KnowFormer |
|--|--|--|
| V init cada outer layer | `batch.v_x_state` (acumulado entre capas) | zeros reales |
| Mezcla con x acumulado | Ninguna | `fc_v_x(cat([x, v_x]))` |
| Iteraciones NBF internas | 1 | 2 |
| BF residual | Activo | No existe |

## Hallazgo 3: Por qué ep1 funciona y ep2+ colapsa

`query_rel_emb` recibe gradientes por **tres rutas** simultáneas en el forward actual:
1. `x0 = query_emb` → `h += x0` (BF residual, ×T capas)
2. `v_x[anchor] = query_emb` (V-NBF anchor, ×T capas)
3. `gate = fc_zq(query_emb)` (bilinear gate, ×T capas)

A LR=1e-5 (ep1), actualizaciones pequeñas → el modelo ajusta ligeramente todos estos pesos sin especializarlos. A LR=1e-4 (ep3+), el optimizer encuentra soluciones que maximizan loss en training explotando las tres rutas multiplicativas a la vez. El resultado son representaciones que fit los paths del train graph pero no transfieren.

---

# Solución principiada

No es un parche. Es alinear el V stream con KnowFormer, que **sabemos que funciona** (0.752 MRR), **conservando el Exphormer sparse attention como contribución**.

## Los tres cambios necesarios (todos relacionados, no independientes):

**1. V inicia desde ceros reales cada outer layer + mezcla con x**

```python
# Reemplazar batch.v_x_state con:
v_x = h.new_zeros(num_node, h.shape[-1])          # zeros reales
v_x[anchor_global] = 1.0                           # one-hot: sin gradiente de query_emb aquí
# Mezcla controlada con h (x acumulado):
v_x = self.fc_v_x(torch.cat([h, v_x], dim=-1))    # linear(2d → d)
# 2 iteraciones NBF internas con pesos distintos:
v_x = self.v_nbf(v_x, edge_index, rel_idx, query_emb, edge_graph)
v_x = self.v_nbf2(v_x, edge_index, rel_idx, query_emb, edge_graph)
```

El `fc_v_x` hace lo que `W_V` hacía antes (buffer de gradiente, permite que x acumulado informe V), pero sin convertir V en h directamente.

**2. Sin BF residual**

Quitarlo completamente. Con el mixing `fc_v_x(cat([x, v_x]))`, el x acumulado ya entra en el V stream de forma controlada. El BF residual adicionalmente por separado genera esa triple ruta de gradiente.

**3. Anchor con constante, no con query_emb**

`v_x[anchor] = 1.0` en lugar de `v_x[anchor] = query_emb`. La información de la relación query entra vía `fc_z(query_emb)` (pesos de propagación NBF). Así `query_rel_emb` solo tiene **una** ruta de gradiente (la del NBF scatter), no tres.

---

# Por qué esto no es "otro parche"

Las fallos anteriores (C1, C2, C3, C4, V-RMPNN, noise) eran intervenciones aisladas. Esto es diferente: la solución está **derivada directamente del código KnowFormer** que logra 0.752, aplicando sus mecanismos de V stream mientras preservamos la sparse attention del Exphormer como contribución propia.

Justificación de cada decisión:
- **V desde zeros**: KnowFormer lo hace. Es la forma de evitar que V acumule identidades de entidad de layers previas.
- **fc_v_x(cat([x, v_x]))**: KnowFormer lo hace. Es el mecanismo que permite que el contexto acumulado informe V sin que V sea una copia de h.
- **anchor = 1.0**: KnowFormer lo hace. Separa la identidad del head (structural) del embedding de relación (semantic). El embedding de relación entra solo vía NBF weights.
- **2 iteraciones internas**: KnowFormer usa 2. 1 era insuficiente para coverage de 2-hop en WN18RR.
- **Sin BF residual**: KnowFormer no lo tiene. Con V-NBF, x0 no necesita ser inyectado manualmente porque fc_v_x(cat([x, v_x])) ya incorpora el contexto.

---

# Lo que NO cambiar

- El expander graph (contribución core del Exphormer)
- Q y K anclados a x0 (C1, ya validado que no hace daño)
- El bilinear gate C2 (gate_base + fc_zq) — es inductivo y fue validado
- La arquitectura transductiva/inductiva unificada (solo hiperparámetros cambian)
- La cabeza KGC

---

# Análisis completo del agente (detalle técnico)

## 1. Qué hace exactamente el KGCHead

**Archivo**: `network/heads.py`

El `KGCHead` computa scores de la siguiente manera:

```python
def forward(self, batch):
    h = self.drop(batch.x)              # (N_total, dim_in) - representacion final de nodos
    r_emb = batch.query_emb             # (B, dim_in) - del embedding compartido en MultiModel
    r_per_node = r_emb[batch.batch]     # (N_total, dim_in) - broadcast a cada nodo
    h = torch.cat([h, r_per_node], dim=-1)  # (N_total, 2*dim_in)
    scores = self.scorer(h).squeeze(-1)     # (N_total,) - Linear
```

**Análisis crítico**:

1. **No usa embeddings de entidad directamente** en el head. El `scorer` es un `Linear(2*dim_in, 1)` que toma la concatenación de:
   - `h`: la representación final de cada nodo tras L capas de atención
   - `r_per_node`: el embedding de la relación query (puramente relacional)

2. **El problema está en `h`, no en el head**. La representación `h = batch.x` después de L capas contiene información acumulada de las entidades del grafo. El head simplemente proyecta `[h; r_q]` a un escalar. Si `h` memoriza patrones del grafo de train, el scorer aprenderá a proyectar esos patrones.

3. **Comparación con KnowFormer**:
   ```python
   self.mlp_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
   score = self.mlp_out(x).squeeze(-1)
   ```
   KnowFormer NO concatena `r_q` al final. Solo usa `x` (la representación acumulada). Pero la clave es que `x` de KnowFormer proviene de NBF streams frescos cada capa, no de `h` acumulado.

---

## 2. Qué produce el KGCNode encoder para entidades NO vistas (inductivo)

**Archivo**: `encoder/node_encoders.py`

```python
class KGCNodeEncoder(nn.Module):
    def forward(self, batch):
        N = batch.x.shape[0]
        h = torch.zeros(N, self.dim_emb, device=device)       # TODOS empiezan en cero
        anchor_global = batch.ptr[:-1] + batch.anchor_idx
        h[anchor_global] = batch.query_emb                     # Solo anchor = rel_emb
        batch.x = h
```

**Análisis**:

1. **No hay embeddings de entidad**. El encoder es completamente agnóstico a la identidad de las entidades. Produce:
   - Nodo anchor: `h = query_emb` (embedding de la relación, no de la entidad head)
   - Todos los demás nodos: `h = 0`

2. **Es estructuralmente inductivo** a nivel del encoder. El problema no está aquí.

3. **Comparación con KnowFormer**:
   ```python
   x = torch.zeros((batch_size, graph.num_nodes, self.hidden_dim), device=self.device)
   ```
   KnowFormer también arranca con ceros globales. La diferencia es que KnowFormer **no tiene boundary condition** (`x = 0` para todos, incluso el head). El head se inyecta dentro del V-stream vía `v_x[h_index] = 1` (one-hot, no rel_emb).

---

## 3. Tensores que dependen de identidad de entidad vs. solo tipo de relación

Traza del flujo completo:

**MultiModel.forward()**:
```python
query_emb = self.query_rel_emb(batch.query_relation)  # RELACIONAL: solo depende de r_q
batch.query_emb = query_emb
batch = self.encoder(batch)  # KGCNodeEncoder: x[anchor]=rel_emb, x[others]=0
batch.x0 = batch.x           # x0 es RELACIONAL (solo anchor tiene valor no-cero)
batch = self.layers(batch)   # L capas de MultiLayer
```

**MultiLayer.forward()**:
```python
for model in self.models:
    h_out_list.append(model(batch))  # GlobalModel (Exphormer)
h = sum(h_out_list)
h = h + FFN(h)                       # FFN(h)
if hasattr(batch, 'x0'):
    h = h + batch.x0                 # BF residual - RELACIONAL (x0 es relacional)
batch.x = h
```

**ExphormerAttention.forward()**:
```python
# Q y K (post-C1):
h_q = batch.x0  # RELACIONAL (anchor=rel_emb, others=0)
Q_h = self.Q(h_q) + self.proj_q(shared_node)  # RELACIONAL
K_h = self.K(h_q) + self.proj_k(shared_node)  # RELACIONAL

# Gate (C2 bilinear):
gate = gate_base[edge_rel] + fc_zq(query_emb)[graph, edge_rel]  # RELACIONAL

# V-NBF stream (implementación actual):
if hasattr(batch, 'v_x_state') and batch.v_x_state is not None:
    v_x = batch.v_x_state.clone()     # ACUMULADO de capa anterior
else:
    v_x = h.new_zeros(...)
v_x[anchor_global] = query_emb        # "fresh" sobre V contaminado
v_x = self.v_nbf(...)                 # RELACIONAL en pesos
batch.v_x_state = v_x                 # guarda para siguiente capa

# E (edge features):
E = self.E(edge_attr)  # edge_attr = RelationEmbeddingEncoder(r_uv) - RELACIONAL
```

**Tabla resumen**:

| Tensor | Depende de | Observación |
|--------|------------|-------------|
| `query_emb` | Solo `r_q` | Relacional puro |
| `x0` | Solo `r_q` | Relacional (anchor=rel_emb, others=0) |
| `Q, K` (post-C1) | Solo `x0, query_emb` | Relacional puro |
| `E` | Solo `r_uv` | Relacional puro |
| `gate` (C2) | Solo `r_uv, r_q` | Relacional puro |
| `V_h` (V-NBF chained) | v_x_state acumulado | **Potencialmente problemático** |
| `h^{t-1}` | Todo lo anterior acumulado | **Problemático si se usa como V** |

Con V-NBF implementado correctamente la cadena es teóricamente inductiva, pero hay problemas ocultos detallados a continuación.

---

## 4. Por qué V-NBF debería ser más inductivo que V=h, pero empeoró el resultado

**Por qué V-NBF debería ser inductivo**:

La lógica del diseño es sólida:
- `V = h^{t-1}` contiene información acumulada de entidades específicas del grafo de train
- `V = VLayerNBF(zeros + anchor_fresh, KG, query_emb)` propaga SOLO via pesos relacionales `fc_z(query_emb)[r_uv]`
- Si los pesos de propagación dependen únicamente de `(r_uv, r_q)`, la información en V es transferible a grafos nuevos

**Por qué empeoró en la práctica**:

Problemas identificados en el código actual:

1. **V-NBF está ENCADENADO entre outer layers** (`batch.v_x_state`). La implementación propaga `v_x` de layer l a layer l+1, no la reinicia. Esto significa que V acumula información a través de las L capas externas. Esto NO es lo que hace KnowFormer.

   KnowFormer:
   ```python
   v_x = torch.zeros(batch_size, graph.num_nodes, self.hidden_dim)  # FRESH ZEROS cada outer layer
   v_x[arange(batch_size), h_index] = 1
   v_x = self.fc_v_x(torch.cat([x, v_x], dim=-1))
   for layer in self.v_layers:
       v_x = layer(v_x, z, r_index, graph)
   ```
   V arranca de CEROS en cada outer layer. Solo dentro de la outer layer hay 2 iteraciones NBF internas.

2. **Sin mezcla con x acumulado**. KnowFormer hace `fc_v_x(cat([x, v_x]))` donde `x` es la representación acumulada del outer loop. Esto le da contexto del grafo al V-stream sin que V SE CONVIERTA en x. La implementación actual no tiene esta mezcla.

3. **Solo 1 iteración NBF por outer layer**. KnowFormer usa 2 iteraciones internas. Una sola iteración tiene propagación de 1-hop por outer layer.

4. **El BF residual sigue activo**. Aunque la documentación decía que fue eliminado, el código real lo tiene activo (restaurado en sesión anterior).

5. **Inestabilidad de la cadena de gradiente**. Con V-NBF, el camino de gradiente desde loss hasta `query_rel_emb` pasa por:
   - `batch.x0` (via BF residual, T veces)
   - `v_x[anchor] = query_emb` (via V-NBF anchor, T veces)
   - `gate = fc_zq(query_emb)` (via bilinear, T veces)
   
   Son 3 rutas multiplicativas a través de T=5 capas. Sin W_V como buffer, los gradientes son muy inestables al LR peak.

---

## 5. Qué hace KnowFormer exactamente diferente: forward paso a paso

**Archivo**: `Knowformer/src/model.py`

**Knowformer.forward()**:
```python
def forward(self, batched_data):
    h_index = batched_data['h_index']   # (B,) índice del head
    r_index = batched_data['r_index']   # (B,) índice de la relación query
    graph = batched_data['graph']
    
    z = self.query_embedding(r_index)   # (B, d) - UNA SOLA tabla de relaciones
    
    x = torch.zeros((batch_size, graph.num_nodes, self.hidden_dim))  # CEROS globales
    
    for layer in self.layers:
        x = layer(h_index, r_index, x, z, graph)  # L outer iterations
    
    score = self.mlp_out(x).squeeze(-1)  # (B, N) scores
```

**KnowformerLayer.forward()**:
```python
def forward(self, h_index, r_index, x, z, graph, ...):
    # 1. Q/K stream - FRESCO cada outer layer
    qk_z = self.fc_qk_z(z).view(B, R, d)                          # (B, R, d) pesos por relación
    qk_x = torch.zeros(B, N, 1).normal_(0, 4)                      # RUIDO FRESCO
    qk_x = self.fc_qk_x(torch.cat([x, qk_x], dim=-1))              # Mezcla con x acumulado
    for layer in self.qk_layers:                                    # 2 iteraciones NBF internas
        qk_x = layer(qk_x, qk_z, graph)
    
    # 2. V stream - FRESCO cada outer layer
    v_x = torch.zeros(B, N, d)                                     # CEROS FRESCOS
    v_x[arange(B), h_index] = 1                                    # ONE-HOT del head (no rel_emb!)
    v_x = self.fc_v_x(torch.cat([x, v_x], dim=-1))                 # Mezcla con x acumulado
    for layer in self.v_layers:                                     # 2 iteraciones NBF internas
        v_x = layer(v_x, z, r_index, graph)
    
    # 3. Atención lineal (no softmax)
    q, k = self.fc_to_qk(qk_x).chunk(2, dim=-1)
    v = v_x
    x = x + self.attn(q, k, v)                                     # Residual sobre x acumulado
    x = self.attn_norm(x)
    x = x + self.ffn(x)
    x = self.norm(x)
    return x
```

**KnowformerQKLayer.forward()** - cada iteración NBF interna:
```python
def forward(self, x, z, graph):
    output = generalized_rspmm(edge_index, relation=z, input=x)    # Σ z[r] * x[u]
    x = self.mlp_out(output + self.alpha * x)                      # Residual interno
    x = self.norm(x)
    x = x + x_shortcut                                             # Skip connection
    return x
```

**Tabla comparativa detallada**:

| Aspecto | QC-Exphormer (V-NBF actual) | KnowFormer |
|---------|---------------------|------------|
| Init global | `x0[anchor]=rel_emb, x0[others]=0` | `x = 0` global |
| V init cada outer layer | `v_x = batch.v_x_state` (encadenado) | `v_x = zeros` (fresco) |
| Head injection | `v_x[anchor] = query_emb` | `v_x[h_index] = 1` (one-hot) |
| Mezcla con x | NO | `fc_v_x(cat([x, v_x]))` |
| NBF iterations internas | 1 | 2 |
| BF residual | Activo | No existe |
| Q/K stream | `W_Q/K(x0) + proj_q/k(q)` | NBF separado con ruido |
| Atención | Exp no normalizado (sum) | Lineal |
| Normalización | LayerNorm | LayerNorm |

**Por qué KnowFormer funciona inductivo**:

1. **V y Q/K se recomputan DESDE CEROS cada outer layer**. No hay acumulación de información de entidades en los streams que definen el routing.

2. **La mezcla con x es controlada**: `fc_v_x(cat([x, v_x]))` proyecta la concatenación, permitiendo que el modelo aprenda CUÁNTO de x usar.

3. **Sin BF residual**: No hay refuerzo de patrones train-específicos.

4. **El head se marca con one-hot, no rel_emb**: V propaga "qué tan alcanzable es cada nodo desde el head via relaciones", no "qué embedding tiene el head". La información de relación entra vía pesos NBF `z`.

---

## 6. Mecanismo concreto del colapso ep1→ep2+

**Patrón observado** (consistente entre múltiples experimentos):

| Época | LR | val MRR | test MRR |
|-------|-----|---------|----------|
| 0 | 0 | ~0.21 | ~0.29 (random init) |
| 1 | 1e-5 | ~0.29 | ~0.40 (mejor) |
| 2 | 2e-5 | ~0.17 | ~0.29 (caída) |
| 3+ | escalando | declina | declina |

**Mecanismo**:

1. **Epoch 0 (random init)**: fc_z (Kaiming, std≈0.1) produce proyecciones relacionales aleatorias diversas. Esto es equivalente a random label propagation relacional. Generaliza bien (test=0.29) porque no hay especialización.

2. **Epoch 1 (LR=1e-5)**: Actualizaciones muy pequeñas. El modelo ajusta ligeramente pesos sin especializarlos. Encuentra la dirección correcta (gradiente informativo) pero no va lejos. test=0.40.

3. **Epoch 2+ (LR aumenta)**: El optimizer encuentra soluciones que maximizan el likelihood de training explotando las 3 rutas hacia `query_rel_emb`. Pequeños cambios en `fc_zq.weight` producen grandes cambios en la salida (multiplicación a través de T=5 capas con exp). El modelo memoriza paths del train graph.

**Por qué el modelo ANTES (sin V-NBF) lograba 0.578**:

El modelo anterior usaba `V = W_V(h^{t-1})` que, aunque no es inductivo en teoría, tenía `W_V` como buffer de gradiente. La proyección lineal `W_V` absorbía parte de la inestabilidad, permitiendo LR más altos sin colapso inmediato. El tradeoff era un techo estructural (~0.58 vs ~0.75 de KnowFormer).

---

## 7. Inconsistencias entre documentación y código real

**Inconsistencia 1 — BF residual**:

Docstring `exphormer.py` (línea 26):
> "BF residual (h += x0) is ELIMINATED in MultiLayer.forward()."

Código real `model.py` (líneas 286-288):
```python
if hasattr(batch, 'x0'):
    h = h + batch.x0   # ACTIVO
```
El BF residual **no está eliminado**. Fue "restaurado" en una sesión posterior.

**Inconsistencia 2 — V "fresh" vs. chained**:

Docstring (línea 8):
> "V comes from a fresh DistMult-style NBF propagation each outer layer"

Código real (líneas 229-232):
```python
if hasattr(batch, 'v_x_state') and batch.v_x_state is not None:
    v_x = batch.v_x_state.clone()   # NO es fresco — es el estado de la capa anterior
```

**Inconsistencia 3 — "Fresh anchor injection"**:

Docstring:
> "The anchor signal is injected fresh each outer layer via v_x[anchor] = query_emb."

Código: La inyección ocurre sobre `batch.v_x_state` ya cargado. Es fresh solo en el sentido de que sobreescribe el valor del anchor, pero el resto de V ya tiene información acumulada.

**Inconsistencia 4 — SESSION_NOTES vs. código**:

SESSION_NOTES registra "Eliminado BF residual", pero el código actualmente lo tiene activo. El historial de sesiones introdujo el cambio y luego lo revirtió sin actualizar las notas.

---

## Resumen ejecutivo

**Causa raíz del fallo de V-NBF**:

El modelo V-NBF falla porque combina tres decisiones incompatibles con inductividad:

1. **V encadenado entre outer layers** (`batch.v_x_state`) — acumula información de entidades
2. **BF residual activo** (`h += x0`) — genera triple ruta de gradiente hacia `query_rel_emb`
3. **Sin mezcla fc_v_x(cat([x, v_x]))** — V no tiene buffer para propagar con estabilidad

KnowFormer evita los tres: reinicia V a ceros cada outer layer, no tiene BF residual, y mezcla V con x acumulado via `fc_v_x`.

**Lo que el código actual realmente hace** (diferente de la documentación):
- V viene de V-NBF **encadenado** entre capas (no fresco)
- BF residual **sigue activo** (no eliminado)
- Fresh anchor injection ocurre sobre V contaminado, no sobre ceros
- Solo **1 NBF iteration** interna vs. 2 de KnowFormer
- Sin **mezcla controlada** con x (`fc_v_x(cat([x, v_x]))`)

**La solución** está en implementar los tres cambios de forma conjunta, derivados directamente del código KnowFormer que logra 0.752 MRR inductivo, conservando el Exphormer sparse attention como contribución.
