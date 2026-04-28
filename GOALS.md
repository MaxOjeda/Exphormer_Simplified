# GOALS.md

Objetivos de investigación de la tesis doctoral (Maximiliano Ojeda, PUC Chile).

## Objetivo general

Diseñar y validar un **Graph Transformer con atención dispersa basada en grafos expander** que integre codificación relacional composicional y transferible, capaz de realizar inferencia de enlaces con generalización zero-shot en grafos de conocimiento no vistos.

## Hipótesis

**H1**: La integración de representaciones relacionales transferibles dentro de atención dispersa guiada por grafos expander capturará dependencias globales en O(N), superando a GNNs de message-passing en link prediction.

**H2**: Un GT disperso con codificación relacional composicional generalizará zero-shot a relaciones y entidades no vistas, superando a NBFNet en benchmarks de KGs no vistos.

---

## Etapa 1 — Adaptación de Exphormer para KGC

**Qué**: modificar la atención de Exphormer para operar condicionada a una query relacional (u, q). Las representaciones de entidades surgen dinámicamente durante la propagación — sin embeddings de entidad. Solo embeddings de relaciones como parámetros aprendibles (compartidos y transferibles).

**Validación**:
- Transductivo: WN18RR y FB15k-237 vs NBFNet, RotatE, DistMult
- Inductivo: splits v1–v4 de WN18RR y FB15k-237 vs GraIL y NBFNet
- Ablaciones: contribución de cada componente del interaction graph (vecindario local / expander / nodos virtuales)

**Resultado esperado**: modelo competitivo con NBFNet, complejidad O(|V|+|E|), setting inductivo sin embeddings de entidad.

**Venue objetivo**: ICLR / LoG

### Estado actual (2026-04-21)

| Setting | Nuestro MRR | Referencia | Gap |
|---------|-------------|------------|-----|
| Transductivo WN18RR | 0.566 | NBFNet 0.551 | ✅ superado |
| Inductivo v1 (val-selected) | 0.486 | NBFNet 0.741 | −0.255 |
| Inductivo v1 (test-selected, hist.) | 0.565 | KnowFormer 0.752 | −0.187 |
| Inductivo v2 | 0.514 | ~0.68 (est.) | −0.17 |
| Inductivo v3 | 0.229 | ~0.67 (est.) | −0.44 |
| Inductivo v4 | 0.474 | ~0.73 (est.) | −0.26 |

**Restricción arquitectónica de la tesis**: el grafo expander es el mecanismo central de atención global — no un ablation a descartar. Las modificaciones deben mantenerse dentro de la arquitectura Exphormer (atención dispersa sobre interaction graph = local + expander + virtual nodes).

---

## Etapa 2 — Codificación Relacional Composicional y Transferible

**Qué**: representar relaciones como funciones de su posición estructural en el grafo de relaciones (inspirado en ULTRA), no como vectores por identidad. Una relación no vista se representa desde su estructura local sin haber sido entrenada.

**Componentes clave**: grafo de interacción entre relaciones + expander de relaciones + compatibilidad con la atención dispersa de Etapa 1.

**Validación**: splits inductivos sobre relaciones de FB15k-237 y WN18RR.

---

## Etapa 3 — Integración y Zero-Shot

**Qué**: modelo unificado entrenado simultáneamente en FB15k-237 + WN18RR + NELL-995, evaluado zero-shot en KGs no vistos (sin fine-tuning).

**Comparación**: ULTRA, NBFNet, KnowFormer.

**Contribución teórica adicional**: análisis del comportamiento del expander en entrenamiento multigrafo (grado fijo vs adaptativo según tamaño del grafo).

**Venue objetivo**: NeurIPS / ICML / ICLR
