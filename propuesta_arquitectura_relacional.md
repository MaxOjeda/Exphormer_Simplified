# Propuesta de arquitectura relacional — Ruta 2

**Síntesis de la tesis: atención dispersa sobre expander (Exphormer) + mensajes desde streams relacionales frescos (estilo KnowFormer), sin canales de memorización de entidad.**

Fecha: 2026-05-21 (sesión 32). Autor del análisis: trabajo conjunto Mauricio Ojeda + asistente.

---

## 0. Resumen ejecutivo

La arquitectura QC-Exphormer actual es un **óptimo local transductivo**: su 0.566 MRR en WN18RR transductivo proviene precisamente de los componentes que **memorizan la topología del grafo de entrenamiento** (`K = W_K(h)`, `V` derivado de `h` acumulada, `FFN(h)`). Esos mismos componentes **topan el inductivo en ~0.40 MRR** y son inamovibles: 7 palancas unificadas (loss CE/BCE, lr, warmup, regularización, `novw`, gate multiplicativo, scorer bilineal) aterrizaron todas en ~0.40. El cuello no es de tuning; es estructural.

La solución no es otro parche sobre esta arquitectura, sino **reemplazar el mecanismo de mensaje por uno puramente relacional**: las representaciones Q/K/V se computan desde *streams de propagación relacional frescos* en cada capa, cuyo **ruteo depende solo de la relación de la query** (operador tipo DistMult `⊙ z(r_q)`), nunca de un mapa lineal aprendido sobre la representación acumulada de entidades. Sobre esos streams se aplica la **atención dispersa de Exphormer sobre el grafo de interacción `H = KG ∪ expander`** — la contribución central de la tesis.

Esta es la única ruta con **evidencia empírica de alcanzar ≥0.70** (KnowFormer obtiene 0.752 inductivo / 0.579 transductivo con un mecanismo relacional puro), es **genuinamente unificada** (un solo forward para ambos settings), y **es la dirección real de la tesis** (Etapas 2-3), no un atajo.

---

## 1. El problema: por qué la arquitectura actual no puede llegar

### 1.1 El diagnóstico, confirmado experimentalmente

El mensaje actual en `layer/exphormer.py` es:

```
m_{w→v} = score_{w→v} · (V_w ⊙ gate_{wv})
  score_{w→v} = exp( ⟨ Q_v ⊙ K_w ⊙ E_{wv} ⟩ / √d )
  Q_v = W_Q(x0_v) + proj_q(c_q)          # x0 = boundary (relacional) ✓
  K_w = W_K(h_w) + proj_k(c_q)           # ← LEE h ACUMULADA
  V_w = W_V(h_w)   (o h_w con novw)       # ← LEE h ACUMULADA
  gate = V_gate(φ(r)) + proj_vg(c_q)
```

Hay tres **canales de memorización de entidad** — rutas por las que la representación acumulada `h` (que codifica *qué entidades* son vecinas en el grafo de train) influye en el ruteo o el contenido del mensaje:

1. **El ruteo de atención** vía `K = W_K(h)`: `W_K` aprende a proyectar las representaciones que emergen en el grafo de train hacia direcciones de "buen source de mensaje" *para ese grafo*. En el grafo inductivo de test, las mismas dimensiones de `h` contienen distribuciones de entidades distintas → el ruteo aprendido no transfiere.
2. **El valor** `V = W_V(h)` (o `V = h` con `novw`): transfiere la distribución de representaciones del grafo de train. Nota crítica: `novw` (quitar `W_V`) **no** resuelve esto — `h` sigue siendo entity-accumulated; solo elimina una proyección.
3. **El `FFN(h)`**: aprende transformaciones de la distribución de `h` del train.

### 1.2 Por qué ninguna palanca unificada lo mueve (datos de sesión 32)

| Palanca probada | ¿toca los canales 1-3? | Resultado inductivo (best-by-val) |
|---|---|---|
| loss CE | no | 0.232 (pico, luego crash) |
| loss BCE-128 | no | 0.393 (pico) / 0.40 (estable lr bajo) |
| lr / warmup | no | 0.40 (techo, lr-independiente) |
| dropout 0.4 + wd 1e-2 | no | 0.376 (más bajo) |
| `novw` (V=h) | parcial (quita W_V, no la dependencia de h) | 0.367 |
| gate multiplicativo (FiLM) | no | 0.356 |
| scorer bilineal por-query | no | 0.353 |

**Invariante a las 7 palancas: ~0.40.** Lo único en todo el historial que sube el inductivo es **K solo-query** (`K = proj_k(c_q)`, eliminando `W_K(h)` → 0.48→0.53→0.58) — que ataca el canal #1 directamente, pero **rompe el transductivo a 0.0003** porque ahí el ruteo entity-específico es esencial.

### 1.3 La tensión es fundamental

Un único conjunto de pesos **no puede** tener pico en transductivo *e* inductivo *mientras posea canales de entidad*:
- **Transductivo**: cuando las entidades son conocidas, su identidad es señal legítima. `W_K(h)` ayuda.
- **Inductivo**: las entidades de test son otras; `W_K(h)` proyecta a direcciones aprendidas para entidades inexistentes → estorba.

La única forma de ser bueno en ambos con una arquitectura unificada es **no tener canales de entidad en absoluto**. Es exactamente lo que hacen NBFNet (0.551 trans / 0.741 ind) y KnowFormer (0.579 trans / 0.752 ind): sus mensajes son **composición relacional pura**, y consiguen el rendimiento transductivo *por la vía relacional*, no memorizando entidades.

> **Reencuadre clave**: no es "sacrificar transductivo por inductivo". KnowFormer **gana en ambos** y es unificado. Adoptar un mecanismo relacional puro no es un trade-off — es estrictamente superior *si se implementa bien*. El riesgo es de implementación, no de concepto.

---

## 2. Por qué esto ES una solución real de la tesis (no un parche)

El manuscrito de candidatura define una tesis en 3 etapas hacia un **modelo fundacional para grafos con generalización zero-shot**. La arquitectura actual está atascada en un óptimo transductivo que, por construcción, no avanza hacia ese objetivo. La Ruta 2 es la materialización de las Etapas 2-3.

### 2.1 Conexión directa con las hipótesis y preguntas del manuscrito

- **H1** (atención dispersa expander + representaciones relacionales transferibles → dependencias globales en O(N) sin sacrificar expresividad, superando MPNN): la Ruta 2 es exactamente esta arquitectura. El expander aporta la atención global O(|V|+|E|); los streams relacionales aportan la expresividad que las MPNN locales no tienen.
- **H2** (GT disperso con codificación relacional composicional → zero-shot, superando NBFNet): la Ruta 2 es el sustrato sobre el cual la codificación relacional transferible (Etapa 2) se inserta. Sin un mecanismo relacional puro, H2 es inalcanzable.
- **P1** (¿cómo modificar la atención dispersa expander para inyectar codificación relacional transferible?): la Ruta 2 *es* la respuesta arquitectónica a P1 — define **dónde** y **cómo** entra la señal relacional (en los streams, vía `z(r_q)`).
- **P2** (¿cómo afecta la dispersión al poder expresivo vs GT densos / GNN de caminos?): la Ruta 2 permite el experimento limpio. KnowFormer usa atención **densa** (kernel lineal sobre todos los pares); nosotros usamos atención **dispersa** sobre el expander. Comparar ambos *con el mismo mecanismo relacional* aísla el efecto de la dispersión — una contribución teórico-empírica que ningún paper previo tiene.
- **P4** (¿qué componentes son mínimos para el balance O(N) ↔ expresividad?): la Ruta 2 es ablacionable componente a componente (stream Q/K, stream V, expander, número de iteraciones NBF internas).

### 2.2 La arquitectura actual es un callejón sin salida para el objetivo de la tesis

La contribución de la tesis es el **expander como mecanismo de atención global para razonamiento relacional**. Pero el expander, sobre la arquitectura actual, es **irrelevante en inductivo** (ablación: Δ ≈ 0.004 MRR con/sin expander). ¿Por qué? Porque el cuello de botella inductivo es la memorización de entidades, no la topología — así que mejorar la topología (expander) no mueve nada. **Solo cuando el mensaje sea relacional puro, el expander tendrá un efecto medible** (mixing global de señales relacionales), y la contribución de la tesis cobrará sentido empírico. La Ruta 2 es la condición necesaria para que la tesis tenga algo que demostrar.

---

## 3. La arquitectura propuesta (detalle técnico)

### 3.1 Principio invariante

> **Toda función que defina el ruteo o el contenido del mensaje debe depender únicamente de cantidades relacionales (relación de la query `r_q`, relación de arista `r_uv`, condición de frontera del ancla), nunca de un mapa lineal aprendido sobre la representación acumulada de entidades.**

La representación acumulada `x` puede usarse como *contexto inicial* de los streams (como en KnowFormer), pero el **operador de propagación** que la transforma es relacional (DistMult `⊙ z(r_q)`), no `W(x)`. Como `x` se construye desde cero (boundary relacional) vía propagación relacional, nunca entra información de embeddings de entidad.

### 3.2 La capa relacional-dispersa (por capa externa `l`)

Notación: `N` nodos, `d` dim oculta, `r_q` relación de la query, `KG` aristas del grafo (bidireccionales con relaciones inversas), `H = KG ∪ expander` grafo de interacción.

```
ENTRADA: x ∈ R^{N×d}   (x^{(0)} = 0, como KnowFormer; o boundary relacional)

# ---- (a) Stream Q/K: contexto relacional, simetría rota con ruido ----
qk ← fc_qk_in( concat[ x, ε ] ),   ε ~ N(0, σ²)        # ruido rompe simetría
for i in 1..L_qk:                                       # L_qk ≈ 2 iteraciones NBF
    qk ← Φ_i( α·qk + RSPMM_KG( qk, z_qk(r_q) ) )       # propagación RELACIONAL
q, k ← split( W_qk(qk) )                                # (N×d), (N×d)

# ---- (b) Stream V: pairwise/head-aware ----
v ← fc_v_in( concat[ x, onehot(ancla) ] )               # head labeling, fresco
for i in 1..L_v:                                        # L_v ≈ 2 iteraciones NBF
    v ← Ψ_i( β·v + RSPMM_KG( v, z_v(r_q) ) )            # propagación RELACIONAL
V ← v                                                   # (N×d)

# ---- (c) Atención DISPERSA sobre H = KG ∪ expander (contribución Exphormer) ----
para cada arista (w→v) ∈ H:
    e_{wv} ← embedding relacional de la arista (KG: φ(r_wv); expander: e_exp(r_q))
    s_{w→v} ← exp( ⟨ q_v ⊙ k_w ⊙ e_{wv} ⟩ / √d )       # score relacional
    m_v    ← Σ_{w ∈ N_H(v)} s_{w→v} · V_w               # SUMA (NBFNet-style, sin /Z)

# ---- (d) Actualización (residual de capa, NO BF anchor residual) ----
x ← LayerNorm( x + W_O · m )
x ← LayerNorm( x + FFN(x) )
SALIDA: x
```

donde `RSPMM_KG(h, z)` es la iteración NBF generalizada (DistMult disperso):
```
out[v] = Σ_{(u, r, v) ∈ KG}  z[r] ⊙ h[u]
```
con `z_qk(r_q)`, `z_v(r_q)` factores relacionales **indexados por la relación de la query** (vienen de tablas `R[r_q]·W + b` por relación).

### 3.3 Scoring y pérdida

- **Scoring**: `s(v | h, r) = readout( x_v^{(L)}, r_q )`. Mantener el head actual o el bilineal (`⟨x_v, proj(r_q)⟩`). Se reutiliza `network/heads.py`.
- **Pérdida**: BCE-128 con self-adversarial (ya implementado, `loss/losses.py`) — confirmado superior a CE.
- **Entrenamiento**: full-graph NBFNet-style con drop de aristas directas y recíprocos (ya implementado y verificado correcto en `train/trainer.py`).

### 3.4 Qué NO tiene esta arquitectura (y por qué importa)

- **Sin `W_K(h)`, `W_V(h)`** — no hay ruteo ni valor que lea `h` acumulada. (Canales #1, #2 eliminados.)
- **Sin BF residual `h += x0`** — el ancla se reinyecta fresco cada capa vía el `onehot(ancla)` del stream V y el ruido del stream Q/K, no vía un residual fijo que se acumula. (Esto evita el conflicto de "doble inyección" que mató a V-NBF v1-v3; ver §6.)
- **`x^{(0)} = 0`** — sin embeddings de entidad, como KnowFormer. La identidad emerge solo de propagación relacional.

### 3.5 Qué se reutiliza del código actual (correcto y verificado)

- `train/trainer.py`: batching full-graph, drop de aristas directas, eval inductivo con swap al grafo de test, head+tail vía recíprocos, filtered ranking. **Todo verificado correcto en el code-read de sesión 32.**
- `loss/losses.py`: BCE-128 self-adversarial + CE filtrado.
- `encoder/exp_edge_fixer.py`: generación y tiling del expander, merge KG ∪ expander.
- `network/model.py`: estructura `FeatureEncoder → N×capa → head`. Se reemplaza solo `ExphormerAttention`/`MultiLayer` por la capa relacional.
- Dataset, splits, métricas: sin cambios.

El cambio se concentra en **una clase nueva de capa**; el andamiaje (datos, eval, loss, expander) queda intacto.

---

## 4. Cómo se diferencia de KnowFormer (la novedad / contribución)

KnowFormer (Liu et al., ICML 2024) introduce los streams relacionales Q/K y V. La tesis los **reutiliza como mecanismo de mensaje**, pero difiere en el eje que es la contribución central:

| Eje | KnowFormer | **Ruta 2 (esta tesis)** |
|---|---|---|
| **Topología de atención** | **Densa**: kernel lineal (Taylor 1er orden) que aproxima atención sobre **todos los pares** vía "query prototypes". O(N) por el truco del kernel `K^T V`. | **Dispersa explícita**: atención sobre `H = KG ∪ expander` (Exphormer). O(\|V\|+\|E\|) por construcción, sin truco de kernel. |
| **Mecanismo global** | Término de bypass del kernel (`+1^T V + v·\|V\|`) | **Grafo expander** d-regular near-Ramanujan → mixing en O(log n) capas (garantía espectral de Exphormer). |
| **Fundamento teórico de la eficiencia** | Aproximación de Taylor del kernel exponencial | Propiedades espectrales del expander (aproxima al grafo completo: `(1−ε)L_K ⪯ L_G ⪯ (1+ε)L_K`) |
| **Codificación relacional** | Tabla `R` aprendida **por dataset**, init aleatorio → **NO transferible zero-shot** (inductivo en entidades, no en relaciones) | **Etapa 2**: `z(r_q)` proviene de un encoder relacional **transferible** (grafo de interacción de relaciones + expander de relaciones, estilo ULTRA) → zero-shot a relaciones no vistas |
| **Setting alcanzable** | Inductivo en entidades (mismo conjunto de relaciones) | Inductivo en entidades (Etapa 1) **→ zero-shot en relaciones (Etapas 2-3)** |

### 4.1 Las dos diferencias que constituyen la contribución

**(A) Dispersión expander vs densidad kernel — contribución de Etapa 1.**
KnowFormer logra O(N) mediante un truco algebraico (kernel lineal) sobre atención conceptualmente densa. Nosotros logramos O(|V|+|E|) mediante una **estructura de grafo explícita** (el expander). Son dos filosofías distintas de escalabilidad:
- El kernel de KnowFormer pierde la noción de *localidad estructural* (todos los pares interactúan por igual, modulados solo por el kernel).
- El expander **preserva la estructura del grafo** (aristas KG = vecindario relacional) y añade *atajos globales controlados* (aristas expander) con garantías espectrales.

La pregunta **P2 del manuscrito** ("¿cómo afecta la dispersión al poder expresivo vs densos?") solo se puede responder con este diseño: mismo mecanismo relacional, una versión densa (kernel KnowFormer) vs dispersa (expander). Es un experimento controlado que **ningún paper previo ha hecho**.

**(B) Codificación relacional transferible — contribución de Etapas 2-3 (donde se supera a KnowFormer).**
KnowFormer **no es zero-shot**: su tabla de relaciones `R` se aprende por dataset y no transfiere a relaciones nuevas. La tesis inserta, en el lugar de `z(r_q)`, un **encoder relacional composicional** (Etapa 2): las relaciones se representan como funciones de su posición en un *grafo de interacción de relaciones* (cómo se componen, invierten, co-ocurren), siguiendo la lógica de ULTRA (Galkin et al., 2024). Sobre ese grafo de relaciones se aplica **otro expander** (escalabilidad + mixing), cerrando el círculo de la contribución Exphormer en dos niveles (entidades y relaciones).

La arquitectura de la Ruta 2 está diseñada para que `z(r_q)` sea un **punto de inserción limpio**: en Etapa 1 es una tabla aprendida (como KnowFormer); en Etapa 2 se reemplaza por el encoder transferible **sin tocar el resto de la capa**. Esa previsión arquitectónica es lo que hace que Etapa 1 (Ruta 2) sea el puente correcto y no un desvío.

### 4.2 Resumen de la novedad en una frase

> *Un Graph Transformer para razonamiento en KGs que reemplaza la atención densa de KnowFormer por atención dispersa sobre grafos expander (Exphormer), preservando garantías espectrales de cobertura global en O(|V|+|E|), y cuya codificación relacional es transferible (ULTRA) para generalización zero-shot — algo que ni Exphormer (no relacional), ni NBFNet (sin atención global), ni KnowFormer (no zero-shot) ofrecen.*

---

## 5. Cómo se diferencia de Exphormer y NBFNet

| | NBFNet | Exphormer (original) | Ruta 2 |
|---|---|---|---|
| Atención global | no (solo vecindario, BF iterativo) | sí (expander), pero **no relacional** | sí (expander) **y relacional** |
| Mensaje | DistMult `h ⊙ w_r` | `W_V(h)` estándar + edge features | streams relacionales frescos `z(r_q)` + atención dispersa |
| Memoriza entidades | no | **sí** (W_V, W_K sobre h) | **no** |
| Zero-shot relaciones | no | no | **sí (Etapa 2)** |

- **vs NBFNet**: añadimos atención global (expander) que NBFNet no tiene — permite que señales relacionales de nodos lejanos en el KG se mezclen en O(log n) capas en vez de requerir T = diámetro. Esta es la mejora de expresividad sobre el modelo de caminos.
- **vs Exphormer**: hacemos el mensaje *relacional* (no `W_V(h)`), que es lo que Exphormer no fue diseñado para hacer. El Exphormer original memoriza entidades igual que nuestra arquitectura actual — por eso el Exphormer "tal cual" no sirve para inductivo.

---

## 6. Por qué esta vez es distinto (lecciones de los fracasos de V-NBF)

Sesiones 26-28 implementaron variantes "V-NBF" que fallaron (0.13-0.40). El re-análisis (sesión 25, 31) identificó **por qué**, y la Ruta 2 corrige cada causa:

| Causa del fracaso de V-NBF v1-v3 | Corrección en Ruta 2 |
|---|---|
| **Doble inyección del ancla**: BF residual (`h += x0`) + NBF fresco peleándose; el optimizer no los reconcilia (mismo fallo que V-RMPNN sesión 14) | **Sin BF residual.** El ancla entra solo vía `onehot(ancla)` en el stream V y ruido en Q/K, fresco cada capa. Una sola vía de inyección. |
| **V encadenado entre capas externas** (acumulaba entidades) | Streams **frescos desde cero** cada capa externa. |
| **`K = W_K(h)` seguía activo** (canal #1 sin eliminar) | Q/K vienen del stream relacional; **no existe `W_K(h)`**. |
| **Solo 1 iteración NBF interna** | `L_qk = L_v = 2` (como KnowFormer). |
| **Entrenamiento que colapsaba** (CE + lr 8e-4) ocultaba el potencial | **BCE-128 + lr 1e-5** (descubierto en sesión 32): elimina el crash, da meseta estable → lectura limpia del techo real. |

Además: la versión más fiel a KnowFormer (**V-NBF v4**, sesión 28: zeros frescos, one-hot, sin BF residual, 2 iteraciones) **solo llegó a smoke test y nunca se evaluó completa**. La Ruta 2 es esencialmente v4 hecho bien, completo, con entrenamiento estable y con la atención dispersa expander en lugar de la fusión a medias. **La ruta correcta está sub-explorada, no refutada.**

---

## 7. Plan de implementación

### 7.1 Cambios de código (concentrados en una capa nueva)

1. **`layer/relational_layer.py`** (nuevo): clase `RelationalSparseLayer` con:
   - `RSPMM` (scatter DistMult disperso sobre KG) — reutilizable para ambos streams.
   - Stream Q/K (`fc_qk_in`, `L_qk` iteraciones, `W_qk`, ruido).
   - Stream V (`fc_v_in`, `L_v` iteraciones, head one-hot).
   - Atención dispersa sobre `batch.expander_edge_index` (KG ∪ expander) con score relacional + suma.
   - Tablas relacionales `z_qk`, `z_v` indexadas por `batch.query_relation`.
2. **`network/model.py`**: rama `elif layer_type == 'RelationalSparse'` en `MultiLayer`/`MultiModel`; **eliminar el BF residual** cuando esta capa esté activa (`x0` no se re-suma). Threading de `L_qk`, `L_v`, `noise_std`.
3. **`config.py`**: `cfg.gt.layer_type = 'RelationalSparse'`, `cfg.gt.num_qk_layers = 2`, `cfg.gt.num_v_layers = 2`, `cfg.gt.qk_noise_std = ...`.
4. **`encoder/node_encoders.py`**: el ancla ya no inyecta `rel_emb[q]` como boundary persistente; `x^{(0)} = 0` y el `anchor_idx` se pasa a la capa para el one-hot del stream V. (O mantener boundary como contexto inicial — decisión a ablacionar.)
5. **Config nuevo**: `wn18rr_ind_v1_relational.yaml` (L externas, `L_qk`/`L_v`, BCE-128, lr 1e-5).

### 7.2 Decisiones de diseño a resolver empíricamente (ablaciones)

- ¿`x^{(0)} = 0` (KnowFormer puro) o boundary relacional `rel_emb[q]` en el ancla (NBFNet)?
- ¿Los streams propagan solo sobre KG, o también sobre el expander? (El expander en los streams daría reach global a la propagación relacional; en la atención da mixing de features. Probar ambos.)
- `L_qk`, `L_v` ∈ {1, 2, 3}; `L` externas ∈ {3, 5, 6}.
- Ruido `σ` en el stream Q/K (symmetry breaking).

### 7.3 Plan de validación (de-risking incremental)

1. **Smoke test** (exit 0, params razonables, ep0 ≈ random).
2. **Inductivo v1 primero** (donde tenemos el recipe estable y el ciclo es ~40 min): objetivo intermedio **superar 0.58** (el mejor histórico), objetivo final **≥0.70**.
   - Criterio de éxito temprano: la **meseta estable** (no el pico transitorio) debe superar 0.40. Si la meseta sube → el mecanismo relacional funciona.
   - Criterio de fracaso: si mesetea en ~0.40 igual que todo lo demás → el problema no está donde creemos; volver a diagnóstico.
3. **Ablación del expander** sobre esta arquitectura: ahora *debería* importar (a diferencia de la actual). Es el experimento que valida la contribución de tesis.
4. **Transductivo** (cuando haya 4 GPUs): verificar que la arquitectura relacional pura mantiene ≥0.55 (KnowFormer logra 0.579, así que es alcanzable). Confirma la unificación real.
5. **WN18RR ind v2/v3/v4 y FB15k-237 ind** para robustez.

### 7.4 Esfuerzo y riesgo

- **Esfuerzo**: varios días (capa nueva + ablaciones). El andamiaje (datos/eval/loss/expander) ya está.
- **Riesgo principal**: que los streams relacionales, con la atención dispersa encima, no alcancen 0.70 (KnowFormer usa atención densa; la dispersión podría costar expresividad — justamente P2). Mitigación: ablación densa-vs-dispersa; si la dispersión cuesta demasiado, es en sí un resultado publicable sobre el límite expresivo del expander en razonamiento relacional.
- **Riesgo secundario**: inestabilidad de entrenamiento. Mitigado por el recipe BCE-128 + lr bajo.

---

## 8. Camino a Etapas 2-3 (zero-shot) — por qué este es el puente

1. **Etapa 1 (esta propuesta)**: `z(r_q)` = tabla relacional aprendida. Inductivo en entidades. Objetivo: ≥0.70 en WN18RR/FB15k-237 ind, competitivo con NBFNet/KnowFormer. Entregable ICLR/LOG.
2. **Etapa 2**: reemplazar `z(r_q)` por un **encoder relacional transferible**: construir el grafo de interacción de relaciones (head-head, head-tail, tail-tail, inversas), aplicar un **expander de relaciones**, y derivar `z(r)` de la posición estructural de `r` — no de su identidad. La capa de la Ruta 2 no cambia; solo cambia de dónde viene `z`. → generalización a relaciones no vistas.
3. **Etapa 3**: entrenamiento multi-grafo (FB15k-237 + WN18RR + NELL-995), evaluación zero-shot en KGs no vistos. Estudio del grado del expander en contexto multi-grafo (contribución teórica esperada del manuscrito).

La Ruta 2 es la **única arquitectura de Etapa 1 desde la cual las Etapas 2-3 son un cambio localizado** (`z`), en vez de una reescritura. La arquitectura actual, con sus canales de entidad, no admite la inserción de un encoder relacional transferible sin antes resolver la memorización — es decir, **sin antes hacer la Ruta 2**.

---

## 9. Criterios de decisión (resumen)

**Adoptar la Ruta 2 si**: se acepta que (a) el objetivo de la tesis es zero-shot/foundational, no transductivo; (b) 7 palancas unificadas confirman que la arquitectura actual topa en ~0.40 por construcción; (c) el único mecanismo con evidencia de ≥0.70 unificado es el relacional puro (KnowFormer); y (d) la dispersión expander + codificación transferible son la novedad real frente a KnowFormer.

**Fallback (Ruta 1)**: si se necesita un número intermedio rápido, K-solo-query como flag por-setting documenta ~0.58 (no 0.70, "casi dos arquitecturas") mientras se construye la Ruta 2. No es la solución, es un puente temporal.

---

## Apéndice: estado al cierre de sesión 32

- Flags ya implementados que quedan disponibles (default no cambian nada): `gt.use_w_v`, `gt.gate_film`, `kgc.bilinear_scorer`.
- Mejor inductivo limpio actual: **0.40** (`wn18rr_ind_v1_bce128_lr1e5.yaml`, best-by-val).
- Recipe de entrenamiento estable validado: **BCE-128 + lr 1e-5 + warmup 8** (sin crash, meseta).
- Pipeline (datos/eval/loss/expander) verificado correcto en code-read — reutilizable tal cual.
