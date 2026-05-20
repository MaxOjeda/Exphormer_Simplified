He leído la arquitectura completa de ambos lados (`layer/exphormer.py`, `network/model.py`, `encoder/node_encoders.py`, `Knowformer/src/model.py`). Ahora sí puedo darte un análisis estructural fundamentado, no parches.

---

## Diagnóstico: por qué el inductivo decae estructuralmente

El problema no es lr, ni warmup, ni FiLM. Hay **cinco diferencias arquitectónicas concretas** entre tu implementación y KnowFormer que, sumadas, explican por qué decae en inductivo. Las analizo en orden de impacto:

### 1. Tu Q y K son CONSTANTES por nodo (esto es lo crítico que nadie ha visto)

Mira tu `ExphormerAttention.forward` (líneas 198-209):
```python
h_q = batch.x0                                     # = 0 para todo nodo no-anchor
Q_h = self.Q(h_q) + self.proj_q(query_emb[batch.batch])
K_h = self.K(h_q) + self.proj_k(query_emb[batch.batch])
```

Para cualquier nodo no-anchor (>99% del grafo): `x0 = 0`, entonces:
- `Q_v = proj_q(query_emb)` — **idéntico para todo v**
- `K_v = proj_k(query_emb)` — **idéntico para todo v**

El score de atención es:
```
score(w → v) = (Q_v ⊙ K_w ⊙ E_wv).sum() / √d
            ≈ (proj_q(q) ⊙ proj_k(q) ⊙ E_wv).sum()    [para w, v no-anchor]
```

**Esto colapsa la atención a "peso por relación de arista"**. No hay discriminación nodo-a-nodo. Tu mecanismo no es atención, es un re-ponderado de aristas. El anchor sale "gratis" del producto Q*K porque es el único nodo con `x0 ≠ 0`, pero el resto de las decisiones de routing las hace solo `E_wv` y el `gate(r_uv, q)`.

KnowFormer en cambio (líneas 190-202 de `Knowformer/src/model.py`):
```python
qk_x = noise_normal(B, N, 1)                    # ruido de simetría
qk_x = fc_qk_x([x, qk_x])                       # mezcla con x acumulada
for layer in self.qk_layers:                    # 2 NBF iters con z = fc_qk_z(q)
    qk_x = layer(qk_x, qk_z, graph)
q, k = fc_to_qk(qk_x).chunk(2)
```

Q y K son *features nodo-específicas que emergen de propagación relacional*. Cada nodo tiene un Q distinto y un K distinto, derivados de qué caminos relacionales pasan por él. **Esto es lo que hace que la atención discrimine entre nodos sin memorizar identidades.**

Tu arquitectura no tiene esto y **nunca lo ha tenido**. El hack de "K = proj_k(q) solo" (sesión 5, +0.05 MRR inductivo) fue precisamente quitar `W_K(h)` porque memoriza, pero te dejó K como un valor *constante por query*. El siguiente paso lógico ya no es quitar más términos — es dar a Q/K contenido relacional emergente.

### 2. Tu V-NBF es una versión amputada de KnowformerVLayer

Tu `VLayerNBF.forward` (líneas 69-91):
```python
msg = v_x[src] * z_edge
out = v_x.clone()
scatter(msg, dst, dim=0, out=out, reduce='add')
return out
```

KnowFormer's `KnowformerVLayer.forward` (líneas 85-109):
```python
output = generalized_rspmm(...)
x_shortcut = x
x = self.fc_out(output + self.beta * x)   # MLP + mezcla aprendible
x = self.norm(x)                           # LayerNorm
x = x + x_shortcut                         # residual
return x
```

Tu V-NBF no tiene: MLP `fc_out`, parámetro de mezcla `beta`, LayerNorm, ni shortcut residual. Después de 2 scatters, V_v ≈ v_x[v] + Σ DistMult-msgs sin normalizar. La magnitud crece con el grado del nodo. Sin normalización, los nodos de alto grado dominan la atención por puro tamaño, no por contenido.

### 3. BF residual `h += x0` es redundante y desbalancea magnitudes

`MultiLayer.forward:285`:
```python
if hasattr(batch, 'x0'):
    h = h + batch.x0
```

`x0` = anchor: query_emb, otros: 0. Después de L=5 capas:
- Anchor: h_anchor ≈ h_acumulada + 5·query_emb
- No-anchor: h_v ≈ h_acumulada (sin reinyección)

La BatchNorm/LayerNorm posterior intenta normalizar a través del batch (todos los nodos), lo cual **diluye la firma del anchor que acabas de inyectar**. El gradiente de la inyección es contra la normalización: el optimizer nunca llega a un punto estable.

KnowFormer no tiene esto. Reinjecta el anchor *exclusivamente* vía `v_x[h_index] = 1` en el stream V *antes* de las NBF iters, así el anchor se propaga *vía mensajes* en lugar de inyectarse *encima* de h. Esto es geométricamente muy distinto.

### 4. BatchNorm es estructuralmente incompatible con KGC inductivo

Tu config usa `batch_norm: True` (default en `GlobalModel` y `MultiLayer`). BatchNorm acumula running statistics durante train. En inductivo, el grafo de test tiene entidades distintas → distribución de h distinta → las running stats de train no aplican al test.

KnowFormer usa LayerNorm en todas partes. LayerNorm es per-nodo, sin estadísticas globales. Es invariante al cambio de grafo. **Para inductivo, LayerNorm no es opcional.**

Esto solo no explica el gap, pero amplifica todo lo anterior.

### 5. V-NBF iterando sobre `KG ∪ Expander` mezcla dos funciones

Tu `v_x = self.v_nbf(v_x, edge_index, ...)` donde `edge_index` es la unión KG+Expander (línea 178: `edge_index = batch.expander_edge_index`). Las aristas del expander reciben la "relación sentinela" `num_relations`, y con esa relación hacen DistMult sobre v_x.

Esto significa que el optimizer aprende un solo embedding `fc_z(q)[sentinela]` que mezcla sobre el expander. Pero el rol del expander en Exphormer es *topológico* (mixing global aleatorio), no *relacional* (no hay relación real ahí). Hacer DistMult con una "relación expander" interfiere con el aprendizaje de las verdaderas relaciones del KG.

KnowFormer hace V-NBF solo sobre `graph.edge_index` (aristas KG). No tiene expander. Tu arquitectura DEBE separar:
- **V-NBF (y Q/K-NBF) → solo aristas KG** (relacional)
- **Atención dispersa → KG ∪ Expander** (topológico + relacional)

---

## Plan de solución fundamentado (no parche)

Esto NO es "probar otro lr". Es un rediseño de la capa que respeta los principios estructurales que hacen funcionar a KnowFormer, manteniendo el aporte de tesis (atención dispersa via expander).

### Paso 1: Q/K stream con propagación relacional emergente

Reemplazar `Q_h = self.Q(h_q) + proj_q(...)`, `K_h = self.K(h_q) + proj_k(...)` por un stream análogo al V-NBF actual:

```python
class QKLayerNBF(nn.Module):
    """Análogo a KnowformerQKLayer pero sobre nuestras KG edges."""
    def __init__(self, d, num_relation_slots):
        super().__init__()
        self.fc_z = nn.Linear(d, num_relation_slots * d, bias=False)
        self.fc_out = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.alpha = nn.Parameter(torch.empty(1, d)); nn.init.normal_(self.alpha)
        self.norm = nn.LayerNorm(d)
    def forward(self, qk_x, edge_index, edge_rel, query_emb, edge_graph):
        # NBF scatter idéntico al actual
        ...
        x_shortcut = qk_x
        x = self.fc_out(agg + self.alpha * qk_x)
        x = self.norm(x)
        return x + x_shortcut
```

En la capa:
```python
qk_noise = torch.randn(num_node, 1, device=h.device) * self.qk_noise_std \
           if self.training else h.new_zeros(num_node, 1)
qk_x = self.fc_qk_x(torch.cat([h, qk_noise], dim=-1))   # (N, d)
qk_x = self.qk_nbf1(qk_x, kg_edges, kg_rels, query_emb, edge_graph)
qk_x = self.qk_nbf2(qk_x, kg_edges, kg_rels, query_emb, edge_graph)
q_h, k_h = self.fc_to_qk(qk_x).chunk(2, dim=-1)         # ahora Q,K son nodo-específicos
```

El ruido (solo en train) rompe la simetría inicial entre no-anchors igual que en KnowFormer (línea 191). Es estructuralmente correcto: el anchor se distingue por su `x` ≠ 0, los demás por el ruido + propagación.

### Paso 2: V-NBF con la estructura completa de KnowformerVLayer

```python
class VLayerNBF(nn.Module):
    def __init__(self, d, num_relation_slots):
        super().__init__()
        self.fc_z = nn.Linear(d, num_relation_slots * d, bias=False)
        self.fc_out = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.beta = nn.Parameter(torch.empty(1, d)); nn.init.normal_(self.beta)
        self.norm = nn.LayerNorm(d)
    def forward(self, v_x, edge_index, edge_rel, query_emb, edge_graph):
        # ... scatter DistMult idéntico
        x_shortcut = v_x
        x = self.fc_out(agg + self.beta * v_x)
        x = self.norm(x)
        return x + x_shortcut
```

Es exactamente lo que ya tienes pero con `fc_out + beta + LayerNorm + shortcut` que da estabilidad y expresividad. La iteración interna de KnowFormer pasa por esto, no por un scatter desnudo.

### Paso 3: Separar aristas KG vs aristas para atención

`v_nbf` y `qk_nbf` iteran sobre KG edges *solamente*. La atención dispersa (Q*K*E score + scatter de mensajes pesados) opera sobre KG ∪ Expander. Esto requiere mantener dos `edge_index`:
- `batch.kg_edge_index`, `batch.kg_edge_rel` para los streams NBF
- `batch.expander_edge_index`, `batch.edge_rel_idx` (con sentinela) para la atención

Es un cambio en el dataloader/encoder pero no afecta la pipeline general.

### Paso 4: Eliminar BF residual `h += x0`

Línea 285 de `network/model.py`: borrar. La reinyección del anchor sucede vía `v_x[anchor] = 1.0` cada outer layer (es lo que ya tienes en V-NBF). Eliminar el `+= x0` es lo que hace KnowFormer y lo que tu propio análisis en sesión 25 identificó como necesario.

### Paso 5: BatchNorm → LayerNorm en todo el modelo

En `MultiLayer.__init__` y `GlobalModel.__init__` cambiar el default a `layer_norm=True, batch_norm=False`. LayerNorm:
- Es per-nodo, no requiere running stats
- Funciona bien con DDP sin sincronización
- Es lo que usa KnowFormer y NBFNet

Para transductivo no es regresión (LayerNorm también funciona bien ahí). Para inductivo es necesario.

---

## Por qué esto sí funciona para AMBOS settings

La arquitectura propuesta es ÚNICA. Lo que cambia entre transductivo e inductivo son **hiperparámetros**:

| Hiperparámetro | Transductivo (WN18RR) | Inductivo (WN18RR v1) | Razón |
|---|---|---|---|
| `gt.layers` | 5 | 3-5 | trans necesita más profundidad para entidades vistas |
| `gt.dim_hidden` | 64-96 | 32-64 | trans tolera más capacidad |
| `qk_noise_std` | 1.0-2.0 | 4.0 (como KnowFormer) | ind necesita más ruido |
| `dropout` | 0.0-0.1 | 0.1-0.2 | ind necesita regularización |
| `num_qk_layer` / `num_v_layer` | 2 / 2 | 2 / 2 | igual |

El argumento estructural: si Q, K, V son funciones de **propagación relacional** (no de embeddings entidad-específicos), la arquitectura es invariante a renombrar entidades. Es lo que NBFNet y KnowFormer formalizan. Tu mecanismo se vuelve genuinamente inductivo *por construcción*, no por hiperparámetros.

Para transductivo, los streams Q/K y V siguen funcionando — `x` acumula información relacional útil que se mezcla en `fc_qk_x([x, noise])`, y el modelo la aprovecha. No hay regresión esperada.

---

## Predicciones concretas

Si implementas Paso 1-5:

- **Transductivo WN18RR**: 0.566 → ~0.57-0.58 MRR (similar o mejor; el modelo es más expresivo en Q/K)
- **Inductivo WN18RR v1**: 0.578 → ~0.70-0.74 MRR (cierre de gap fundamental con NBFNet/KnowFormer)
- **Estabilidad de entrenamiento**: el colapso ep1→ep2 que ves consistentemente desaparece, porque los gradientes ya no convergen multiplicativamente sobre `query_rel_emb` por tres rutas (anchor x0, V gate, V-NBF). Quedan dos rutas (Q/K-NBF y V-NBF), ambas con MLP+LayerNorm+residual que dan estabilidad dinámica.

---

## El expander sigue siendo tu contribución

Esto es importante y no se pierde: el expander entra en la atención dispersa (Paso 3). La novedad de tu tesis sigue intacta:

- KnowFormer hace **atención lineal densa** O(|V|·d²)
- Tu propuesta hace **atención dispersa via expander** O((|V|+|E|+|E_exp|)·d) con Q/K/V emergentes de NBF

Esa combinación NO la tiene ningún paper previo. La pregunta empírica de tu tesis (¿el expander interactúa bien con propagación relacional?) ahora se puede responder limpiamente, porque dejas que la propagación relacional funcione como debe.

---

## Próximos pasos concretos (orden recomendado)

1. **Bloque mínimo**: implementar Paso 1 (Q/K stream con NBF) + Paso 5 (LayerNorm). Mantener V-NBF como está y BF residual. Run inductivo v1. Esperado: salto a 0.65+.
2. **Si confirma el salto**: agregar Paso 2 (V-NBF estructurado) + Paso 4 (eliminar BF residual). Run de nuevo. Esperado: 0.70+.
3. **Si llega a 0.70+**: Paso 3 (separar KG vs expander en NBF). Run final. Esperado: 0.72-0.74.
4. **Validar transductivo no regresa**: 1 run de 100 ep en WN18RR full. Esperado: ≥0.566.

Cada paso es testeable independientemente y tiene una predicción concreta. Si el Paso 1 no mueve nada, tienes información: el problema no es Q/K, tendrías que mirar otra cosa. Pero hay alta confianza estructural de que sí mueve.

¿Quieres que empiece por implementar el Paso 1 (Q/K stream)?
