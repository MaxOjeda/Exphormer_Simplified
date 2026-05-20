Después de leer el código completo, tengo un diagnóstico estructural. **El problema no es V**. Voy a explicar qué está pasando realmente y proponer una solución fundamentada.

---

## A. Lo que realmente está pasando

Nuestra arquitectura combina **dos paradigmas incompatibles**: (1) Transformer global con *boundary condition* fija + (2) intento de propagación relacional. NBFNet (0.741) y KnowFormer (0.752) eligen UNO solo:

- **NBFNet** = puramente GNN relacional. h^{(l)}_v = INDICATOR(v=anchor)·z_q + Σ AGG(h^{(l-1)}_u ⊙ z_{r_{uv}}). Cada paso es **DistMult con peso z_r dependiente del edge**. T=6 layers.
- **KnowFormer** = Transformer donde **cada layer reconstruye QKV via NBF fresco** desde zeros. Los QKV no son funciones de h_acumulado; son funciones del NBF interno con z_r. Los outer layers solo agregan refinamiento.

Nuestro modelo hace **ninguna de las dos**: Q,K = f(x0) (boundary fija), V = f(h^{t-1}) (acumulado), score = Q⊙K⊙E sin z_r dentro del score. Es un Transformer estándar con boundary, sin propagación relacional real.

## B. Tres defectos estructurales concretos del código actual

### 1. C1 destruye la expresividad de la atención

`layer/exphormer.py:189-199`:
```python
h_q = batch.x0       # x0[anchor]=query_emb, x0[others]=0
Q_h = self.Q(h_q) + self.proj_q(query_emb[batch])   # post-C1
K_h = self.K(h_q) + self.proj_k(query_emb[batch])
```

Para todos los nodos no-anchor: `W_Q(0) + proj_q(q) = proj_q(q)` — **idéntico**. Solo el anchor es distinto. Esto significa que para cualquier par (u,v) con u,v ≠ anchor, **el score Q_u·K_v depende solo de E (edge feature)**. La atención degenera: pasa de O(N²) interacciones a "anchor vs todos + filtro por edge type".

C1 fue introducido pensando que `h^{t-1}` violaba inductividad. Pero NBFNet también usa h_u en su mensaje (que está acumulado). El verdadero problema no era h en Q,K — era que sin z_r en el score, la "atención" no es relacional en absoluto.

### 2. El score de atención no usa z_r (la relación del edge)

`layer/exphormer.py:160-163`:
```python
score = mul(K_h[src], Q_h[dst])
score = score * E   # E = W_E(rel_emb[r_uv])
score = exp(score.sum(-1, ...).clamp(-5,5))
```

`E` solo entra **multiplicativamente** en el score. Esto es incorrecto: en NBFNet el peso del mensaje es `h_u ⊙ z_r` (Hadamard, no scalar). En KnowFormer la atención bi-afín es `q^T W_r k + bias_r`. Nuestro score reduce todo el efecto relacional a un scalar `Q⊙K·E`, por lo que el modelo solo distingue ~22 tipos de edge globalmente, no patrones relacionales finos por par.

### 3. La sum-aggregation sin Z favorece nodos high-degree

`layer/exphormer.py:170-171`:
```python
batch.wV = torch.zeros_like(batch.V_h)
scatter(msg, dst_idx, dim=0, out=batch.wV, reduce='add')   # sin /Z
```

Sin la normalización Z, un nodo con degree 30 acumula ~10× más mensaje que uno con degree 3. WN18RR es muy heterogénea en grados (anchor en train tiene paths densos hacia patterns memorizados vs. nodos unseen tienen vecindarios sparse). NBFNet usa PNA precisamente para manejar esto. La eliminación de Z fue +0.23 MRR para datasets donde degree es uniforme; aquí es probablemente parte del ceiling.

## C. Por qué V-NBF v1–v5 fallaron (todas)

Todos los V-NBF fallaron porque solo cambiaban el stream de V mientras los puntos 1, 2 y 3 quedaban intactos. Si los Q,K son near-degenerate y el score no usa z_r, **mejor V no compensa**: la atención coloca peso ~uniforme sobre todos los vecinos, así que un V perfectamente inductivo es promediado y diluido.

Es como intentar mejorar un router GPS dándole mejores mapas, cuando el GPS está rotos y siempre apunta al norte.

## D. Solución fundamentada — Camino de 3 niveles

Recomiendo **abandonar la línea V-NBF** y reformular la atención misma. Tres opciones, ordenadas por riesgo y ambición. Cualquiera funciona simétricamente para transductivo/inductivo (solo cambian hiperparámetros).

### **Opción A — Conservadora: revertir C1, atención DistMult-style** (riesgo bajo, upside +0.05–0.10 MRR)

**Cambios**:
1. **Revertir C1**: Q = W_Q(h), K = W_K(h), eliminar x0 anchoring. Recupera la expresividad O(N²) de atención.
2. **Score relacional bi-afín**: agregar un término `q · u_r + v_r · k` al score, donde u_r, v_r ∈ R^{d} son aprendidos por edge type (sentinel para expander).
3. **Eliminar BF residual** `h += x0`: con x0 inyectado solo en h^{(0)} via encoder, no en cada layer.
4. **V = W_V(h) con gate** (volver al baseline transductivo con +W_V): ya validado a 0.566 trans / 0.578 ind.
5. **Restaurar Z normalization** o evaluar PNA aggregation.

**Justificación**:
- El defecto principal (atención degenerate) se corrige.
- z_r entra al score directamente — el modelo aprende patrones relacionales por edge, no por gate aditivo.
- Inductivo: h^{(0)} = query_emb·INDICATOR(anchor); las entity-specific patterns en h vienen de propagación relacional (z_r), no de embeddings de entidad. Sigue siendo inductivo.
- Es el primer ablation que ataca *la causa* identificada: pérdida de expresividad de atención.

**Riesgo**: si el techo de 0.58 viene del agregator y no de la atención, esto da poca ganancia.

### **Opción B — Híbrida: NBFNet-base + Exphormer-augmentation** (riesgo medio, upside +0.10–0.15 MRR)

Reformular la capa de manera que la **propagación relacional sea el componente principal** y **el expander sea augmentation auxiliar**, no al revés.

```python
class NBFExphormerLayer:
    forward(h, x0, edges_KG, edges_Exp, query_emb):
        # 1) NBFNet propagation step (KG only, relational)
        # h_v^new = AGG_{(u,r,v) in KG} (h_u ⊙ z_r) where z_r = fc_z(query_emb)[r]
        nbf_msg = scatter_sum(h[src] * z[r_uv], dst)
        nbf_out = mlp(h + nbf_msg)
        
        # 2) Expander attention (sparse, global, edge-type-agnostic)
        # Provee contexto global más allá del KG
        exp_attn = sparse_attn(Q=W_Q(nbf_out), K=W_K(nbf_out), 
                                V=W_V(nbf_out), edges=edges_Exp)
        
        # 3) Mezcla aprendida (gate)
        h_new = LN(h + nbf_out + λ * exp_attn)
        h_new = LN(h_new + FFN(h_new))
        return h_new
```

**Justificación**:
- La base (paso 1) replica NBFNet exactamente (T iteraciones × DistMult con z_r) → garantiza al menos 0.74 si T=6.
- El expander (paso 2) actúa como "global skip connection" — atención sparse no relacional sobre el grafo expander → la **contribución de tesis se preserva** como augmentation que ataca el problema de "long-range dependencies" de NBFNet.
- Si λ→0, recuperamos NBFNet puro (0.741).
- Si λ>0 mejora, hemos demostrado que el expander añade valor más allá de NBFNet.
- **Inductivo y transductivo automáticos**: NBFNet ya es ambos.

**Riesgo**: implementación más invasiva; requiere reescribir ExphormerAttention. ~3-5 días de trabajo + debug.

### **Opción C — Radical: KnowFormer-style con expander en QKV streams** (riesgo alto, upside +0.15–0.20 MRR)

Adoptar el patrón KnowFormer completo (Q/K/V streams reconstruidos via NBF fresco cada outer layer), pero **agregar el expander dentro de los NBF internos** o como segundo stream paralelo.

```python
# Cada outer layer:
qk_x = noise_init(N, d)   # ruido fresco, KnowFormer-style
qk_x = fc_qk_x(cat([h, qk_x]))
for _ in range(2):
    qk_x = NBF_step(qk_x, KG ∪ Expander, z_qk)  # ← expander dentro

v_x = zeros(N, d); v_x[anchor] = 1.0
v_x = fc_v_x(cat([h, v_x]))
for _ in range(2):
    v_x = NBF_step(v_x, KG ∪ Expander, z_v)

q, k = chunk(fc_to_qk(qk_x))
attn_out = linear_attn(q, k, v_x)
h = LN(h + attn_out); h = LN(h + FFN(h))
```

**Justificación**:
- Replica KnowFormer (0.752) con la modificación de incluir expander en los NBF internos → contribución específica de tesis: "expander accelerates NBF reachability in KnowFormer".
- Hipótesis testeable: ¿el expander reduce el T necesario o mejora long-range KGC?

**Riesgo**: la mayor reescritura. Si el expander dentro del NBF degrada porque mezcla edges no-relacionales con relacionales, no funciona y queda KnowFormer puro (que entonces no es contribución original).

## E. Lo que DEFINITIVAMENTE no se debe seguir haciendo

- Más variantes de V-NBF (v6, v7, ...). Hemos demostrado empíricamente que el problema no es V.
- Tunear LR/wu/dropout sobre la arquitectura actual sin cambiarla. Es polish sobre estructura defectuosa.
- "Patch C5/C6/C7". El espacio ya está agotado.

## F. Mi recomendación concreta

**Empezar por Opción A** (1-2 días, bajo riesgo): revertir C1 + score bi-afín + restaurar Z normalization. Esto valida si la causa raíz es la atención degenerate. Si pasa de 0.578 → 0.65+, confirma el diagnóstico y procedemos a B. Si no, vamos directo a B.

**Después Opción B** (1 semana): NBFNet-base + Expander augmentation. Esto es la línea académicamente más defendible: "extendemos NBFNet con un mecanismo de atención sparse global vía grafo expander". Es una contribución natural y verificable.

Opción C la guardaría como fallback si B no llega a 0.70.

¿Por cuál arrancamos?
