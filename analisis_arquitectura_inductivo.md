Voy a hacer este análisis lentamente. El problema es real y necesita una respuesta fundamentada. Lo organizo en 4 secciones: (A) qué hace exactamente KnowFormer, (B) por qué nuestra arquitectura tiene un techo estructural, (C) por qué el plan del diagnóstico falló, (D) la solución real.

## A) Qué hace exactamente KnowFormer

Releyendo el código completo de `KnowformerLayer.forward()` (línea 187):

```python
qk_x = zeros(B, N, 1).normal_(0, 4)              # ← FRESH cada outer layer
qk_x = fc_qk_x(cat([x, qk_x]))                    # mezcla con x (acumulado)
for layer in self.qk_layers:                      # 2 iteraciones internas NBF
    qk_x = layer(qk_x, qk_z, graph)               # qk_z = fc_qk_z(query_emb)

v_x = zeros(B, N, d)
v_x[arange(B), h_index] = 1                       # ← FRESH onehot del head cada outer layer
v_x = fc_v_x(cat([x, v_x]))                       # mezcla con x
for layer in self.v_layers:                       # 2 iteraciones internas NBF
    v_x = layer(v_x, z, r_index, graph)           # z = query_emb directo

q, k = fc_to_qk(qk_x).chunk(2)
v = v_x

x = x + attn(q, k, v)                             # residual
x = attn_norm(x)
x = x + ffn(x)
x = norm(x)
```

Y el outer loop (`Knowformer.forward()`):
```python
x = zeros((B, N, d))                              # ← ARRANCA EN CERO. NO HAY BOUNDARY CONDITION INICIAL.
for layer in self.layers:                         # L outer iterations
    x = layer(h_index, r_index, x, z, ...)
```

**Tres observaciones cruciales que cambian todo el análisis:**

1. **KnowFormer NO tiene `h += x0` (BF residual).** Nuestra arquitectura sí lo tiene (`MultiLayer.forward()` línea 284: `h = h + batch.x0`). Esto es **diferencia estructural fundamental** que nadie identificó hasta ahora.

2. **KnowFormer arranca con `x = 0` global.** No hay anchor con `rel_emb` al inicio. La señal del head **se inyecta fresca cada outer layer** vía `v_x[h_index] = 1` (one-hot), no vía residual del initial state.

3. **Q, K, V no son funciones de `x` acumulado.** Son funciones de **propagaciones NBF frescas** que arrancan de cero cada outer layer y solo MEZCLAN `x` como contexto en su primer paso (`fc_qk_x(cat([x, qk_x]))`). Después, las 2 iteraciones internas operan puramente sobre el stream NBF.

## B) Por qué nuestra arquitectura tiene un techo estructural

Nuestro modelo tiene este flujo (post-C1):

```python
# En cada outer layer:
Q = W_Q(x0) + proj_q(q)              # x0 fijo (boundary condition global)
K = W_K(x0) + proj_k(q)              # idem
V = h^{t-1}                          # ← AQUI ESTA EL PROBLEMA
gate = bilinear(r_uv, r_q)           # relacional puro
msg[v] = Σ_w V_w * gate * score(Q,K,E)
h^t = LN(h^{t-1} + msg)
h^t = LN(h^t + FFN(h^t))
h^t = h^t + x0                       # BF residual
```

`V = h^{t-1}` es la única vía por la que la información estructural se propaga. Pero `h^{t-1}` ha acumulado información de todas las entidades que el grafo de train tiene. En inductivo, esas mismas dimensiones contienen información de entidades distintas.

**El BF residual `h += x0` empeora esto**: refuerza el anchor pero también refuerza los patrones acumulados en `h` cada capa, fijando los sesgos del grafo de train.

C1, C2, C3, C4 no atacaron esto porque V seguía siendo `h^{t-1}`. Por eso ninguno mejoró.

## C) Por qué falló el plan del diagnóstico

El diagnóstico decía: "K(h), FFN(h), gate aditivo violan la regla". Identificaba síntomas. Pero la regla estructural correcta no es "no Linear sobre h" — es **"V no debe ser h acumulado, V debe venir de una propagación relacional fresca"**.

Lo previo (V-RMPNN, sesión 14) fracasó porque agregó NBF para V **manteniendo el BF residual**. Eso crea dos sistemas de propagación de la señal del anchor que se contradicen: V-NBF inyecta head fresco, BF residual inyecta x0 fijo. El optimizador no puede reconciliarlos.

## D) La solución real

**Reestructurar la capa para que V (y posiblemente Q/K) vengan de un NBF interno fresco, eliminar el BF residual, e inyectar el anchor fresco cada outer layer dentro del NBF interno.**

Versión mínima viable (V-only, mantiene Q/K desde x0):

```python
class ExphormerKGCLayer:
    def __init__(self, d, num_relations, num_v_iter=2):
        self.fc_v_in = nn.Linear(d * 2, d)              # mezcla x con anchor onehot
        self.v_nbf_layers = ModuleList([
            VLayerNBF(d, num_relations) for _ in range(num_v_iter)
        ])
        self.Q = nn.Linear(d, d)                        # opera sobre x0 (relational)
        self.K = nn.Linear(d, d)                        # idem
        self.proj_q = nn.Linear(d, d)
        self.proj_k = nn.Linear(d, d)
        self.E = nn.Linear(d_edge, d)
        # No W_V — V viene del NBF interno
        # Bilinear gate sobre la salida del NBF
        self.fc_zq = nn.Linear(d, (R+1)*d)
        self.gate_base = nn.Parameter(torch.empty(R+1, d))
        self.attn_norm = nn.LayerNorm(d)
        self.ffn = FFN(d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, x0_anchor, query_emb, edge_index, edge_attr, expander_edge_index):
        # 1. V stream: NBF fresco con head fresh-injected each layer
        v_x = torch.zeros_like(x)
        v_x[anchor_global] = query_emb[batch_idx]       # FRESH cada layer (NO via BF residual)
        v_x = self.fc_v_in(torch.cat([x, v_x], -1))     # mezcla con x acumulado
        for nbf in self.v_nbf_layers:
            v_x = nbf(v_x, edge_index, edge_attr, query_emb)  # 2 internal NBF steps
        
        # 2. Q/K desde representación relacional (x0 anchor + query)
        Q = self.Q(x0_anchor) + self.proj_q(query_emb_per_node)
        K = self.K(x0_anchor) + self.proj_k(query_emb_per_node)
        E = self.E(edge_attr)
        gate = self.gate_base[r_uv] + self.fc_zq(query_emb).view(B, R+1, d)[graph_idx, r_uv]
        
        # 3. Sparse attention con V = v_x (no h acumulado)
        attn_out = sparse_attn(Q, K, V=v_x, E, gate, edges=KG ∪ Expander)
        
        # 4. Residual + FFN (SIN h += x0)
        x = x + attn_out
        x = self.attn_norm(x)
        x = x + self.ffn(x)
        x = self.norm(x)
        return x
```

**Cambios clave vs lo actual:**

| Componente | Actual | Propuesto |
|---|---|---|
| V | `V = h^{t-1}` | `V = NBF(zeros+anchor_fresh, KG, query)` |
| Anchor injection | BF residual `h += x0` cada layer | `v_x[anchor]=query_emb` cada NBF, **sin** BF residual |
| Q, K | `W_Q/K(x0) + proj_q/k(q)` | igual (ya es relacional puro post-C1) |
| Outer x init | `x_anchor = query_emb, x_others = 0` | `x = 0` global (KnowFormer-style) |
| FFN | activo | activo (KnowFormer también lo tiene) |

**Lo que se preserva (contribución de tesis):**
- Expander como topología de atención (`KG ∪ Expander`)
- Atención dispersa O(|V|+|E|)
- Gate bilinear sobre `(r_uv, r_q)` (de C2)
- Q/K relacional puro (de C1)

**Por qué esto sí funcionará para inductivo:**
- V no contiene información de entidades específicas — viene de propagación relacional con peso `z(r,r_q)` que es transferible
- Sin BF residual, no hay refuerzo de patrones train-específicos en h
- La identidad del head se reinyecta fresca cada layer (queda viva sin necesidad del residual)

**Por qué seguirá funcionando para transductivo:**
- Las entidades del train tienen las mismas relaciones que en test — el NBF V-stream funciona igual
- La atención sparse + expander mantiene la estructura global
- Es estrictamente más expresivo que el actual (más capas internas de NBF + atención sparse)

**Riesgos honestos:**
- Costo computacional: 2 NBFs internos × L outer layers = 2L scatter operations adicionales
- Implementación no trivial: necesita un kernel scatter NBF eficiente (KnowFormer usa `generalized_rspmm` de torchdrug — podemos usar `scatter_add` con weighted edges, o portar el kernel)
- Hay que verificar que la mezcla `fc_v_in(cat([x, v_x]))` no contamine v_x con info de entidades vía x

**Recomendación**: lo implemento progresivamente. Primero un prototipo mínimo (1 NBF iteration, sin mezcla con x, fresh anchor sin BF residual). Si eso ya bate 0.578, expandimos a 2 NBF + mezcla + outer loop completo.

¿Procedo con el prototipo? Te avisaré cada paso para que valides antes de seguir.
