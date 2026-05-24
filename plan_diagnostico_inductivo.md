# Plan de diagnóstico — por qué el techo inductivo es 0.40

Fecha: 2026-05-24 (sesión 33+). Autor: trabajo conjunto Mauricio Ojeda + asistente.

## Contexto

Tras 30+ sesiones de cambios arquitectónicos (sum aggregation, novw, V-NBF v1-v5, C1-C4, FiLM, scorer bilineal, K-relacional, RelationalSparseLayer, etc.) el techo inductivo en WN18RR ind v1 se mantiene robusto en **~0.40 test MRR** con la arquitectura unificada. NBFNet llega a 0.741 y KnowFormer a 0.752 en el mismo split. El gap es **2× MRR**, no es de tuning fino.

El patrón experimental hasta ahora ha sido: hipótesis arquitectónica → cambio → run → falla → siguiente hipótesis. Llevamos meses sin **mirar al modelo por dentro**. Antes de seguir tocando, este plan instrumenta el sistema actual para entender qué hace realmente y por qué no aprende inductivo.

El objetivo explícito del usuario: **ser competitivos con NBFNet y KnowFormer en inductivo**. La instrumentación debe servir a esa meta, no a curiosidad académica.

---

## 0. Recalibración antes de empezar

Una observación honesta sobre el "test > val" reportado en sesión 33 (RelationalSparseLayer) y reproducido en BCE-128 + lr 1e-5 sobre arch clean (sesión 33+):

- WN18RR ind v1 val tiene **2746 nodos candidatos**.
- WN18RR ind v1 test tiene **922 nodos candidatos**.
- Es ~3× más fácil rankear top-k en test por puro tamaño del candidate set.

H@10 val=0.50 vs H@10 test=0.69 muestra que el efecto no es solo tamaño, pero la magnitud absoluta del "ratio invertido" está inflada. **Reglas para todo este plan**:

- Reportar siempre **H@k** lado a lado con MRR.
- Cuando se compare val vs test, normalizar por `log(N_candidates)` o usar métricas escala-invariantes.
- No volver a leer "test > val" como firma cualitativa de razonamiento relacional sin esta corrección.

---

## A — Estratificación de queries por estructura (¿qué falla?)

Por cada query (h, r, ?) en val/test, agregar metadatos estructurales y graficar rank predicho vs cada uno.

### A1. Rank vs grado del nodo tail en el grafo de eval
Hipótesis: el modelo predice mejor sobre nodos de alto grado (más evidencia local). Si el patrón es brutal, es la firma de un modelo que falla en cold-start de entidad.

### A2. Rank vs longitud del camino más corto entre h y tail
NBFNet brilla porque modela paths explícitamente. Si nuestro modelo solo acierta en pares con path-length ≤ 1 (vecinos directos), el cuello es la propagación multi-hop — explicaría por qué L=5 no ayuda más que L=3.

### A3. Rank vs frecuencia de la relación r en train
Si las relaciones raras tiran el MRR, el cuello es el aprendizaje de embeddings relacionales (con 10K triples y 11 relaciones, algunas tienen <100 ejemplos). Implicancia directa para Etapa 2: ULTRA composicional resolvería esto.

### A4. Diff de rank por query entre ep6 (peak) y ep20 (plateau)
¿Qué queries se degradan al avanzar el entrenamiento? ¿Son las mismas que ganaron al inicio? Detecta "olvido catastrófico" de cierta clase de query. Pista directa de qué clase de patrón la arquitectura sobreescribe en la cola del entrenamiento.

**Costo**: ~1 día. Script de eval con bookkeeping; reutiliza pipeline actual + loop que guarda (query, rank, features estructurales) en CSV.

---

## B — Estado interno (¿qué pasa dentro del modelo?)

Hooks PyTorch para capturar activaciones intermedias durante eval.

### B1. Distribución de `||h_v||` por capa, train graph vs test graph
Histograma + KL divergence entre train y test, por capa. Si la norma diverge en capa 3+ → el modelo aprende features train-específicos que escalan fuera de distribución en test. Equivalente al activation drift en clasificación de imágenes.

### B2. Mapa de atención: fracción de masa sobre aristas KG vs expander vs anchor
Si la atención sobre el expander es <5% del total → el expander no contribuye (refuerza P2 del manuscrito en sentido negativo: dispersión sin uso). Si está saturando el clamp ±5 → entropía cero, el modelo se compromete con un solo vecino y pierde robustez.

### B3. Entropía de atención por capa
Si la entropía cae monotónicamente con la capa → el modelo se vuelve más "decidido" en capas profundas. Comparar la entropía promedio en queries que acierta vs falla. Posible firma: en queries que falla, la atención está demasiado concentrada en aristas espurias.

### B4. `||W_K(h) − W_K(h')||` para `h, h'` de train vs test
Probe directo: ¿`W_K` proyecta a direcciones distintas en distribución train vs test? Si sí, confirma cuantitativamente el diagnóstico de canales de entidad (sesión 32).

**Costo**: ~1 día (hooks + agregación + plots).

---

## C — Cirugía de pesos (¿qué causa el declive post-ep6?)

El experimento más informativo causalmente. Requiere guardar checkpoints cada época (modificar `ckpt_period: 1` en una corrida nueva).

### C1. Eval con weights ep20 sustituyendo SOLO `W_K` por su versión ep6
Si test_mrr recupera al cambiar `W_K` → `W_K` es la causa del declive. Repetir uno a uno con `W_V`, `FFN`, `rel_emb`, edge embeddings, proj_q/k/e. Ablación causal directa.

### C2. Inverso: pesos ep6 con `W_K` actualizado a ep20
Si esto rompe el ep6 → confirma simétricamente que `W_K` (o el componente que sea) es necesario y suficiente para el declive.

### C3. Plot de evolución de `||W||_F` por matriz de peso a través de epochs
¿Cuáles divergen más entre ep6 y ep20? Marca cuáles weights sospechar primero en C1, ahorra ablaciones.

**Costo**: ~1 día. Script que carga dos checkpoints y hace swap por nombre de parámetro.

**Valor**: máximo de todo el plan. Da respuesta causal sin necesidad de teorizar.

---

## D — Comparación head-to-head con NBFNet/KnowFormer

El usuario explícitamente dice "competitivos con NBFNet y KnowFormer". Sin medirlos lado a lado sobre las mismas queries, no sabemos qué los hace ganar.

### D1. Correr NBFNet sobre WN18RR ind v1 → per-query MRR
Repo público: `https://github.com/DeepGraphLearning/NBFNet`. Guardar rank predicho por cada query de test.

### D2. Idem con KnowFormer
Código local en `Knowformer/` (sesión 30). Setup ya familiar.

### D3. Confusión cruzada per query
- Queries donde **ambos baselines ganan** → estructuralmente fáciles, no nos dicen nada nuevo.
- Queries donde **NBFNet gana, nosotros no** → **el target**. Estratificar estas por features A1-A3 (grado, path, relación). Hay un patrón.
- Queries donde **nosotros ganamos, ellos no** → baja probabilidad pero documenta dónde el expander/atención global sí aporta valor diferenciador.

**Costo**: ~2 días (setup envs + correr + script de comparación).

**Valor**: estratégico máximo. Sin esto, "competitivo con NBFNet" es aspiracional, no medible.

---

## E — Probes arquitectónicos rápidos

Eval con modificaciones triviales del modelo entrenado, sin retraining.

### E1. Eval con `L` (capas) variable al inferir
Entrenar con L=5, evaluar con L=1, 2, 3, 4, 5. ¿A qué profundidad pica test? Si pica antes que train → overpropagation / oversmoothing relacional confirmados. Trivial: una eval pass por L.

### E2. Eval con `prep.exp=False` reutilizando weights entrenados con expander
Si test no se mueve → el modelo no usa el expander, lo aprendió a ignorar. Más informativo que entrenar de cero sin expander (controla por cambios en optimización).

### E3. Ablación de query conditioning en test time
Sustituir `query_emb[r_q]` por un embedding promedio o aleatorio en test eval. Si test mantiene MRR → el modelo no usa la query, se aferra a estructura. Si colapsa → la query sí se usa.

**Costo total E**: medio día.

---

## F — Augmentación informativa de los datos (si todo lo anterior dice "estructural")

### F1. Drop edges agresivo en train graph
Pasar de drop direct edges a drop 50%/75% de aristas adyacentes al anchor. Si MRR sube en test → la abundancia de información local en train inducía dependencia que no transfiere.

### F2. Train multi-graph (FB15k-237 + WN18RR + NELL-995)
Entrenar simultáneamente en ind v1, v2, v3, v4 (o cross-dataset). Si el modelo mejora en v1-test después → la diversidad estructural rompe el shortcut estructural. **Adelanta parcialmente Etapa 3** del manuscrito.

**Costo**: 2-3 días (F2 requiere loader nuevo).

---

## Priorización

Por valor/costo y por **lo que nunca hemos hecho**:

| # | Intervención | Costo | Valor | ¿Decisivo? |
|---|-------------|-------|-------|------------|
| **1** | **C1-C3 cirugía de pesos** | 1 día | máximo (causal directo) | sí — dice qué weights causan el declive |
| **2** | **A1-A4 estratificación queries** | 1 día | alto (descriptivo) | sí — dice qué clase de queries fallamos |
| **3** | **D1-D3 NBFNet/KnowFormer head-to-head** | 2 días | estratégico máximo | sí — define qué significa "competitivo" cuantitativamente |
| 4 | E1-E3 probes arquitectónicos | 0.5 días | medio | aporta señal rápida, complemento de 1-3 |
| 5 | B1-B4 estado interno | 1 día | medio (interpretativo) | sí, pero después de 1-3 |
| 6 | F1-F2 augmentación datos | 2-3 días | alto (potencial cura) | solo si 1-3 indican que es problema estructural |

---

## Primer entregable concreto

Un único script de instrumentación que combine los más baratos:

```
scripts/diagnose_inductive.py
  --ckpt-pattern "results/wn18rr_ind_v1_bce128_lr1e5/0/ckpt/epoch_*.pt"
  --config       configs/Exphormer/wn18rr_ind_v1_bce128_lr1e5.yaml
  --output       analysis/diag_bce128_lr1e5/
```

Lo que hace en una sola corrida:
1. Carga cada checkpoint ep0..ep30 (re-train previo con `ckpt_period: 1`).
2. Eval sobre val y test, guardando **per-query rank** + features estructurales (degree del tail, longitud de path, freq de relación).
3. Hooks PyTorch para capturar **distribución de scores de atención** por tipo de arista, por capa.
4. Estadísticas `||h||` train vs test, por capa.
5. Norms `||W||_F` de cada matriz de peso a través de epochs.
6. Outputs: CSV por query + plots agregados (matplotlib) en `analysis/diag_*/`.

Total: ~2 días de wall-clock antes de tener respuestas reales (1 día script + 1 día re-train con todos los checkpoints + análisis).

---

## Lo que esto NO va a darnos

Honestidad sobre los límites:

- La instrumentación es **descriptiva y causal local** (qué weights, qué queries). NO nos dice si otra arquitectura llegaría a 0.70.
- Para responder "¿debemos cambiar de paradigma?" la única evidencia es **D1-D3 NBFNet head-to-head**. Por eso es prioridad #3 a pesar del costo mayor.
- Los hallazgos pueden converger a un diagnóstico que **no tenga solución dentro de la restricción de "una sola arquitectura unificada"** (ej. "el modelo necesita memorizar entidades en train para transductivo; sin eso, inductivo no aprende lo suficiente"). En ese caso la decisión es del usuario: relajar la restricción, reframear el objetivo (Etapa 1 = "competitivo", no "superar") o cambiar de paradigma (NBFNet-only para inductivo, Exphormer-only para transductivo).

---

## Puntos de decisión inmediatos

1. ¿Arrancar con (1)+(2) combinados en un script único, o priorizar (3) NBFNet head-to-head primero para fijar el target cuantitativo?
2. ¿La corrida base para instrumentar es BCE-128 + lr 1e-5 (techo ~0.40 estable) o CE (con crash patológico ep1→ep3, otro fenómeno distinto)?
3. ¿Se aceptan los 2 días de implementación o se busca atajo (ej. solo C1-C3 con dos checkpoints, sin todos los epochs)?
