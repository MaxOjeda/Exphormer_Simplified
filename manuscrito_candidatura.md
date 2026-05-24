# Pontificia Universidad Católica de Chile
# Escuela de Ingeniería
# Dirección de Postgrado

## Manuscrito del Proyecto de Tesis

[cite_start]**Graph Transformers con Atención Dispersa y Codificación Relacional Transferible para Razonamiento Zero-Shot en Grafos de Conocimiento** [cite: 3]

[cite_start]La investigación propuesta constituye el Proyecto de Tesis doctoral de **Maximiliano Eduardo Ojeda Aguila**, estudiante regular del Programa de Doctorado en Ciencias de la Ingeniería, área Ciencias de la Computación ingresado el primer semestre del año 2024[cite: 4].

---

## [cite_start]I.- RESUMEN DEL PROYECTO DE TESIS [cite: 11]

[cite_start]El desarrollo de arquitecturas basadas en Graph Transformers (GT) ha marcado un hito en el aprendizaje profundo sobre grafos al capturar dependencias de largo alcance mediante mecanismos de atención (Ying et al., 2021), superando limitaciones teóricas de las redes neuronales de grafos (GNN) tradicionales como *oversmoothing* (Chen et al., 2020) y *oversquashing* (Alon & Yahav, 2020)[cite: 12]. [cite_start]Sin embargo, a pesar de resultados sobresalientes en áreas como predicción de propiedades moleculares (Rong et al., 2020), simulaciones dinámicas (Liao et al., 2022) y generación de grafos (Vignac et al., 2022), su uso práctico es limitado debido a que estas arquitecturas suelen presentar una complejidad computacional cuadrática que impide su aplicación en grafos de gran escala y carecen de la capacidad de generalizar hacia estructuras o entidades que no fueron observadas durante el entrenamiento (Zhu et al., 2021)[cite: 13]. 

[cite_start]Si bien existen propuestas para reducir la complejidad de atención a un tiempo lineal, ya sea integrando Transformers lineales con GNNs (Rampášek et al., 2022; Deng et al., 2024) o bien utilizar Graph Transformers con atención dispersa (Shirzad et al., 2023; Shirzad et al., 2024; Müller et al. 2023), aún persiste la incertidumbre sobre qué componentes arquitectónicos son estrictamente necesarios para preservar el poder expresivo y la capacidad de generalización en escenarios diversos[cite: 14].

[cite_start]En este contexto, la presente propuesta de investigación tiene como objetivo principal el estudio teórico y experimental de Graph Transformers para aprendizaje *zero-shot* y gran escala[cite: 15]. [cite_start]A partir del análisis de modelos de atención dispersa como Exphormer (Shirzad et al., 2023), se busca diseñar una arquitectura capaz de transferir conocimiento hacia relaciones no vistas, funcionando como un modelo fundacional para grafos (Galkin et al., 2024; Mao et al., 2024)[cite: 16]. [cite_start]Se plantea que la integración de representaciones relacionales transferibles, basadas en la lógica de ULTRA (Galkin et al., 2023), dentro de arquitecturas de atención dispersa, permitirá capturar dependencias globales sin el costo de la atención densa ni en las limitaciones de paso de mensajes de las GNN inductivas (Teru et al., 2020)[cite: 17].

[cite_start]La investigación se abordará mediante un diseño teórico y validación experimental en distintas fases[cite: 18]. [cite_start]En primer lugar, se realizará un estudio comparativo de estrategias de atención dispersa, tomando como base el modelo Exphormer, para determinar cómo la dispersión basada en grafos *expander* afecta la captura de dependencias globales en tareas de *link prediction* en grafos de conocimiento[cite: 19]. [cite_start]Después, la etapa de diseño de la arquitectura inductiva se centrará en integrar representación relaciones siguiendo la lógica de ULTRA (Galkin et al., 2023)[cite: 20]. [cite_start]Finalmente, se desarrollará un prototipo de Graph Transformer que combine la eficiencia de una atención dispersa con la capacidad *zero-shot* de generalización, el cuál será entrenado en varios grafos de conocimiento y evaluando en datasets no vistos[cite: 21].

[cite_start]Esta investigación permitirá avanzar en el estado del arte de los modelos fundacionales para grafos, proporcionando una arquitectura que equilibre la eficiencia computacional con un alto poder expresivo[cite: 22]. [cite_start]Se espera desarrollar modelos Graph Transformers con capacidad de generalización *zero-shot* que capturen señales estructurales globales[cite: 23]. [cite_start]Por último, este trabajo contribuirá a la comprensión teórica de cómo la atención dispersa y el aprendizaje relacional facilitan la captura de patrones estructurales complejos[cite: 24].

---

## [cite_start]II.- INVESTIGACIÓN PROPUESTA [cite: 29]

### [cite_start]II.1 Descripción General [cite: 30]

#### [cite_start]1. Estado del arte [cite: 31]
[cite_start]La investigación en el aprendizaje profundo sobre grafos se ha sustentado principalmente en las Redes Neuronales de Grafos (GNN) basadas en el paradigma de paso de mensajes (*message passing*)[cite: 32]. [cite_start]Estas arquitecturas, representadas por modelos como GCN (Kipf et al., 2016), GAT (Veličković, 2017) y GraphSAGE (Hamilton et al., 2017), operan mediante la agregación iterativa de vecindades locales[cite: 33]. [cite_start]No obstante, la literatura reciente ha formalizado tres limitaciones críticas que restringen su aplicabilidad en tareas complejas[cite: 34]. 

* [cite_start]**Poder expresivo limitado:** Está limitado por el test de isomorfismo de grafos de Weissfeiler-Lehman de primer orden (1-WL), lo que impide a las MPNN distinguir ciertas estructuras topológicas no isomorfas (Xu et al., 2019)[cite: 35].
* [cite_start]**Oversmoothing:** La repetición de capas de agregación provoca que las representaciones de los nodos converjan a un valor común, perdiendo su especificidad (Chen et al., 2020)[cite: 36].
* [cite_start]**Oversquashing:** Alon & Yahav (2020) caracterizaron este cuello de botella computacional donde la información de nodos distantes no logra propagarse eficazmente a través del grafo, limitando la captura de dependencias de largo alcance (Topping et al., 2021)[cite: 37].

[cite_start]Para mitigar estas problemáticas, se propuso adaptar la arquitectura Transformer (Vaswani et al., 2017), originalmente diseñada para secuencias, al dominio de los grafos y desarrollar la arquitectura Graph Transformer (GT)[cite: 38]. [cite_start]El trabajo de Dwivedi & Bresson (2020) fue crucial al sistematizar el uso de mecanismos de atención global en grafos, reemplazando la agregación local (Yun et al., 2019)[cite: 39]. [cite_start]En este estudio, los autores introdujeron el uso de *Positional Encodings* (PE) basados en los autovectores de la matriz Laplaciana del grafo, permitiendo que el modelo recupere la información estructural que se pierde al tratar al grafo como un conjunto de nodos en una capa de atención (Dwivedi et al., 2022)[cite: 40]. [cite_start]A partir de este hito, la investigación se centrará en cómo integrar la topología del grafo en el mecanismo de atención[cite: 41].

En esta línea, Kreuzer et al. (2021) [cite_start]plantearon un cambio necesario al diseño de los GT, sugiriendo que la atención global por sí sola no es suficiente para superar a las GNN en tareas de clasificación de nodos si no se integra adecuadamente el sesgo inductivo de la conectividad local[cite: 42]. [cite_start]Este trabajo impulsó la creación de arquitecturas híbridas que combinan capas de paso de mensajes con capas de atención global, permitiendo al modelo atender tanto a la vecindad como a la estructura global del grafo[cite: 43]. [cite_start]Un avance significativo en este sentido fue Graphormer (Ying et al., 2021), que demostró que el uso de *Spatial Encodings* (basados en las distancias de camino más corto entre nodos) y *Centrality Encodings* permite a los Transformers superar por amplio margen a las GNN tradicionales en benchmarks de gran escala como el PCQM4Mv2 (Hu et al., 2021)[cite: 44].

[cite_start]A pesar de los avances impulsados por modelos como Graphormer, la adopción práctica de los Graph Transformers densos se enfrenta a la complejidad computacional cuadrática de su mecanismo de atención (Vaswani et al., 2017)[cite: 49]. [cite_start]Dado un grafo con $N$ nodos, la atención tradicional requiere $O(N^{2})$ tanto en tiempo de cómputo como en uso de memoria (Shirzad et al., 2023)[cite: 50]. [cite_start]Esta complejidad ocurre por la necesidad de calcular las interacciones de atención entre cada par de nodos del grafo, lo que resulta prohibitivo para los grafos de gran escala que pueden estar en los sistemas modernos, tales como grafos de conocimiento o redes sociales[cite: 51]. [cite_start]El costo cuadrático no solo limita el tamaño de los grafos que se pueden procesar, sino que también restringe la profundidad de la red y el tamaño del *batch* (Rampášek et al., 2022), impactando directamente la capacidad de entrenamiento de modelos fundacionales[cite: 52].

[cite_start]La investigación reciente se ha dividido en varias direcciones para mitigar el costo cuadrático de la atención (Rampášek et al., 2022), buscando una transición hacia una complejidad lineal $O(N)$ o casi-lineal (Zaheer et al., 2020; Choromanski et al., 2021), sin sacrificar el poder expresivo global que distingue a los GTs de las GNNs (Kreuzer et al., 2021; Xu et al., 2019)[cite: 53].

[cite_start]Una línea de trabajo se centra en integrar la eficiencia lineal de las GNNs con el poder de captura de largo alcance de los Transformers[cite: 54]. Rampášek et al. (2022) [cite_start]propusieron una "receta" para un GT escalable, que combina la agregación local de GNNs con Transformers lineales, como BigBird (Zaheer et al., 2020)[cite: 55]. [cite_start]Esta aproximación busca equilibrar el sesgo inductivo local (Battaglia et al., 2018) con una forma de atención global de costo reducido, aunque el diseño de un mecanismo de atención que sea a la vez eficiente y estructuralmente sensible sigue siendo un reto[cite: 56].

[cite_start]La estrategia más directamente relacionada con la presente propuesta es el uso de la **atención dispersa**[cite: 57]. [cite_start]En lugar de calcular todas las interacciones $N^2$, se introduce una matriz de atención con estructura predefinida que solo calcula las interacciones más críticas[cite: 58]. [cite_start]El modelo Exphormer (Shirzad et al., 2023) es un hito fundamental en esta dirección[cite: 59]. [cite_start]Exphormer utiliza la teoría de los grafos *expander* para definir un patrón de atención dispersa[cite: 60]. [cite_start]Los grafos *expander* son estructuras altamente conectadas, con propiedades de expansión óptimas (Hoory et al., 2006)[cite: 61]. [cite_start]Al forzar la matriz de atención a seguir el patrón de un grafo *expander*, se garantiza teóricamente que, incluso con una conectividad lineal, la información de cualquier nodo puede propagarse rápidamente a través de la red (Shirzad et al., 2023), permitiendo la captura de dependencias globales sin incurrir en el costo cuadrático de la atención densa[cite: 62]. [cite_start]El enfoque de Exphormer demuestra una solución elegante para la escalabilidad[cite: 63].

[cite_start]Más allá de la escalabilidad, el segundo gran desafío de los Graph Transformers es la capacidad de **generalización zero-shot**, especialmente crucial en el dominio de los Grafos de Conocimiento (KGs)[cite: 64]. [cite_start]La mayoría de los modelos de representación de grafos están diseñados para tareas de naturaleza transductiva (Yang et al., 2015; Sun et al., 2019) o inductiva (Hamilton et al., 2017; Teru et al., 2020), donde las entidades o relaciones evaluadas ya fueron vistas durante el entrenamiento[cite: 65]. [cite_start]Sin embargo, para que un GT actúe como un verdadero Modelo Fundacional para Grafos (Mao et al., 2024), debe transferir conocimiento de forma efectiva y realizar predicciones sobre estructuras o relaciones no vistas[cite: 66, 72].

En el razonamiento sobre KGs, la comunidad científica ha explorado principalmente dos caminos: 
* [cite_start]**Modelos Basados en Caminos y Lógica:** El modelo NBFNet (Zhu et al., 2021) y sus derivados (Zhang & Yao, 2022; Zhu et al., 2023), se basan en la idea de utilizar redes neuronales para modelar la composición de relaciones a través de caminos de longitud variable[cite: 74]. [cite_start]Específicamente, NBFNet emula el algoritmo de Bellman-Ford (Bellman, 1958), propagando información a través de la estructura del grafo para realizar la inferencia de enlaces[cite: 75]. [cite_start]Este enfoque, si bien es efectivo para el razonamiento lógico en grafos, opera de manera localizada y no aprovecha la atención global inherente a la arquitectura de los Transformers[cite: 76].
* [cite_start]**Transformers en KGs:** Una línea de investigación más reciente ha comenzado a integrar el mecanismo de atención en la inferencia relacional[cite: 77]. [cite_start]Modelos como KnowFormer (Liu et al., 2024) y Relphormer (Bi et al., 2024) proponen arquitecturas que adaptan el Transformer para modelar la interacción entre las entidades y las relaciones de un KG, a menudo empleando una matriz de atención relacional para inyectar información sobre la estructura del grafo en el mecanismo de atención[cite: 78]. [cite_start]Por su parte, SimKGC (Wang et al., 2022) aborda el problema desde la perspectiva de la representación contrastiva, tratando las tripletas como secuencias de texto para ser codificadas por un Transformer y alineadas en el espacio latente, lo que facilita la generalización a entidades no vistas[cite: 79].

[cite_start]En este contexto, el trabajo de ULTRA (Galkin et al., 2023) ha proporcionado un marco conceptual para la generalización relacional al proponer representaciones que son inherentemente transferibles[cite: 80]. [cite_start]La lógica de ULTRA se basa en la idea de codificar las relaciones de una manera que las haga composicionales y reutilizables, permitiendo que el modelo infiera sobre relaciones no vistas a partir de la composición de relaciones que sí fueron vistas[cite: 81]. [cite_start]La integración de esta lógica de representación relacional dentro de la arquitectura de un Graph Transformer escalable representa la próxima frontera[cite: 82].

#### [cite_start]2. Brecha de Investigación [cite: 83]
Considerando el estado del arte, se pueden destacar los siguientes puntos:
1. [cite_start]**Potencia y Capacidad Global:** Los Graph Transformers densos han demostrado un poder expresivo superior para capturar dependencias globales, pero su complejidad cuadrática los hace inviables para grafos de gran escala[cite: 85].
2. [cite_start]**Eficiencia:** Los modelos basados en atención dispersa, como Exphormer, han resuelto el problema de la escalabilidad al reducir el costo a $O(N)$ mediante el uso de grafos *expander*, preservando la capacidad de captura global[cite: 86].
3. [cite_start]**Generalización Relacional:** A pesar de los avances en modelos como NBFNet, KnowFormer y SimKGC, la mayoría de las arquitecturas GT, incluidas las dispersas (Exphormer), aún carecen de un mecanismo robusto para la generalización *zero-shot* en relaciones no vistas[cite: 91]. [cite_start]Este es un requisito clave para un modelo fundacional universal de grafos[cite: 92].
4. [cite_start]**Capacidad de Inferencia Inductiva:** A pesar de las mejoras en tareas transductivas, los Graph Transformers limitan la inferencia inductiva, crucial para su aplicación en Grafos de Conocimiento dinámicos[cite: 93].

#### [cite_start]3. Preguntas de investigación [cite: 94]
[cite_start]La brecha de investigación identificada se centra en la necesidad de desarrollar una arquitectura Graph Transformer escalable con capacidad de generalización *zero-shot* para Grafos de Conocimiento[cite: 95]. Se plantean las siguientes preguntas:
* [cite_start]**P1:** ¿Cómo debe modificarse la arquitectura de atención dispersa basada en grafos *expander* para inyectar un mecanismo de codificación relacional transferible? [cite: 97]
* [cite_start]**P2:** ¿De qué manera la estructura de atención dispersa afecta el poder expresivo para capturar las señales estructurales globales cruciales para la inferencia relacional en comparación con los Graph Transformers densos o las GNNs basadas en caminos? [cite: 98]
* [cite_start]**P3:** ¿Cuál es la estrategia óptima para integrar la lógica de representación relacional composicional dentro de un Graph Transformer disperso, y cómo impacta esta integración en la capacidad del modelo para inferir sobre relaciones no vistas? [cite: 99]
* [cite_start]**P4:** ¿Qué componentes arquitectónicos son estrictamente necesarios para preservar el equilibrio entre la eficiencia computacional $O(N)$ y el alto poder expresivo requerido para la formación de un Modelo Fundacional para Grafos? [cite: 100]

#### [cite_start]4. Hipótesis [cite: 101]
* [cite_start]**H1:** La integración de representaciones relacionales transferibles dentro de una arquitectura de atención dispersa guiada por grafos *expander*, permitirá capturar dependencias estructurales globales en Grafos de Conocimiento con una complejidad computacional lineal, sin sacrificar el poder expresivo necesario para superar a las GNNs basadas en paso de mensajes en tareas de inferencia de enlaces[cite: 103].
* [cite_start]**H2:** Un Graph Transformer disperso que incorpore un mecanismo de codificación relacional composicional y transferible logrará generalizar de forma *zero-shot* hacia relaciones y entidades no observadas durante el entrenamiento, superando en desempeño a modelos inductivos como NBFNet en benchmarks de Grafos de Conocimiento no vistos, gracias a que la estructura de atención *expander* facilita la propagación de señales estructurales globales necesarias para la inferencia[cite: 104].

#### [cite_start]5. Objetivos [cite: 107]

##### [cite_start]Objetivo General [cite: 108]
[cite_start]Diseñar y validar experimentalmente una arquitectura Graph Transformer con atención dispersa basada en grafos *expander*, que integre un mecanismo de codificación relacional composicional y transferible, capaz de realizar inferencia de enlaces con generalización *zero-shot* en Grafos de Conocimiento no vistos[cite: 110].

##### [cite_start]Objetivos Específicos [cite: 111]
* [cite_start]**OE1:** Comparar estrategias de atención dispersa, tomando como base Exphormer, para explicar cómo el patrón de conectividad *expander* afecta la captura de dependencias globales y el poder expresivo en tareas de inferencia de enlaces en Grafos de Conocimiento[cite: 112].
* [cite_start]**OE2:** Diseñar un mecanismo de codificación relacional composicional y transferible, inspirado en la lógica de ULTRA, que sea compatible con arquitecturas de atención dispersa y permita representar relaciones no vistas como composición de relaciones aprendidas[cite: 113].
* [cite_start]**OE3:** Integrar el mecanismo de codificación relacional propuesto dentro de la arquitectura de atención dispersa de Exphormer, desarrollando un prototipo de Graph Transformer inductivo entrenado sobre múltiples Grafos de Conocimiento simultáneamente[cite: 114].
* [cite_start]**OE4:** Evaluar experimentalmente la capacidad de generalización *zero-shot* del modelo propuesto en datasets de Grafos de Conocimiento no vistos durante el entrenamiento, comparando su desempeño contra modelos del estado del arte como NBFNet, KnowFormer y ULTRA[cite: 115].
* [cite_start]**OE5:** Analizar teórica y empíricamente qué componentes de la arquitectura son estrictamente necesarios para el equilibrio entre eficiencia computacional y poder expresivo, identificando los elementos mínimos requeridos para un Modelo Fundacional para Grafos[cite: 116].

#### [cite_start]6. Metodología y Plan de Trabajo [cite: 117]
La investigación se estructura en tres etapas secuenciales. [cite_start]Cada etapa se sostiene sobre los resultados de la anterior, avanzando desde la adaptación de una arquitectura base hacia un modelo fundacional con capacidad de generalización *zero-shot*[cite: 118]. [cite_start]Este enfoque progresivo permite identificar los desafíos técnicos de forma incremental, reduciendo el riesgo durante la investigación y permitiendo contribuciones parciales publicables en cada etapa[cite: 120].

##### [cite_start]Etapa 1: Adaptación de Exphormer para Link Prediction en Grafos de Conocimiento [cite: 121]
[cite_start]El punto de partida es el modelo Exphormer (Shirzad et al., 2023), cuyo mecanismo de atención dispersa basado en grafos *expander* (Hoory et al., 2006) ha demostrado eficiencia y capacidad de captura de dependencias globales en tareas de clasificación de nodos y grafos[cite: 121, 126]. [cite_start]Sin embargo, Exphormer no fue diseñado para inferencia en KGs ni para operar en un setting inductivo[cite: 127]. 

[cite_start]Esta etapa consiste en modificar la arquitectura de atención de Exphormer para que opere condicionada a una *query* relacional $(u,q)$, de manera que las representaciones de las entidades surjan dinámicamente durante la propagación en función de la consulta, en lugar de ser *embeddings* fijos (Zhang & Yao, 2022)[cite: 128]. [cite_start]Esto es esencial para el setting inductivo donde una entidad no vista durante el entrenamiento no necesita un vector propio, pues su representación se construye a partir de su vecindario y de los *embeddings* de las relaciones que la conectan[cite: 129]. [cite_start]Los únicos parámetros aprendibles asociados a la estructura del grafo son los *embeddings* de relaciones, que son compartidos y transferibles (Zhu et al., 2021)[cite: 130].

[cite_start]La validación se realizará primero en el setting transductivo sobre los benchmarks estándar WN18RR (Dettmers et al., 2018) y FB15k-237 (Toutanova & Chen, 2015), comparando contra NBFNet (Zhu et al., 2021) y modelos de *embedding* como RotatE (Sun et al., 2019) y DistMult (Yang et al., 2015)[cite: 131]. [cite_start]Posteriormente se evaluará en las particiones inductivas de los mismos datasets (Teru et al., 2020), comparando con GraIL (Teru et al., 2020) y NBFNet[cite: 132]. [cite_start]Se realizarán ablaciones para determinar la contribución de cada componente del grafo de interacción: vecindario local, *expander graph* y nodos virtuales[cite: 133].

[cite_start]El resultado esperado de esta etapa es un modelo Exphormer condicionado a la *query* que sea competitivo con NBFNet en *link prediction*, con complejidad lineal en el tamaño del grafo, y que funcione en el setting inductivo sin requerir *embeddings* de entidad[cite: 134]. [cite_start]Los resultados de esta etapa se consolidarán en una publicación que será enviada a una conferencia de primer nivel, como ICLR o Learning on Graphs (LoG)[cite: 135].

##### [cite_start]Etapa 2: Diseño de un Mecanismo de Codificación Relacional Composicional y Transferible [cite: 136]
[cite_start]La Etapa 1 produce un modelo inductivo respecto a entidades nuevas, pero los *embeddings* de relaciones siguen siendo específicos de cada KG[cite: 137]. [cite_start]Esto impide la transferencia *zero-shot* a grafos de conocimiento completamente nuevos, donde el conjunto de relaciones también es distinto[cite: 138]. [cite_start]Esta etapa aborda ese problema diseñando un mecanismo de codificación relacional que sea intrínsecamente transferible entre KGs[cite: 139]. 

[cite_start]Inspirado en la lógica de ULTRA (Galkin et al., 2024), la idea central es que las relaciones no se representen como vectores independientes aprendidos por su identidad (Bordes et al., 2013; Sun et al., 2019), sino como funciones de su posición estructural en el grafo de relaciones (Zhu et al., 2021): cómo se componen, se invierten y se relacionan con otras relaciones dentro del KG[cite: 140]. [cite_start]De esta forma, una relación no vista en un KG nuevo puede representarse a partir de su estructura local en ese grafo, sin necesitar haber sido observada durante el entrenamiento[cite: 141].

[cite_start]El diseño de este mecanismo implicará definir un grafo de interacción entre relaciones, determinar cómo construir representaciones relacionales a partir de ese grafo de manera compatible con el mecanismo de atención dispersa de la Etapa 1, y validar que dichas representaciones capturen patrones composicionales relevantes para la inferencia de enlaces[cite: 142, 146]. [cite_start]La validación se realizará en benchmarks de inferencia inductiva sobre relaciones, evaluando la capacidad del modelo para razonar sobre relaciones no vistas a partir de sus patrones de composición, utilizando los splits inductivos de FB15k-237 y WN18RR (Teru et al., 2020)[cite: 147].

[cite_start]Un aspecto central de esta etapa es el análisis de compatibilidad entre el grafo de relaciones y el grafo de interacción *expander*[cite: 148]. [cite_start]Se investigará si la estructura *sparse* del *expander* preserva suficiente información composicional cuando se aplica sobre el espacio de relaciones, y se explorarán estrategias para construir el *expander* de relaciones de forma que maximice la propagación de señales composicionales[cite: 149].

##### [cite_start]Etapa 3: Integración, Entrenamiento y Evaluación Zero-Shot [cite: 150]
[cite_start]La etapa final integra los dos componentes anteriores: el mecanismo de atención dispersa condicionada a la *query* y el módulo de codificación relacional transferible, en una arquitectura unificada que constituye el prototipo de modelo fundacional para KGs propuesto en esta tesis[cite: 151].

[cite_start]El modelo integrado se entrenará simultáneamente sobre múltiples grafos de conocimiento, siguiendo el paradigma de entrenamiento multigrafo establecido por ULTRA[cite: 152]. [cite_start]El conjunto de entrenamiento incluirá KGs de distinto dominio y tamaño, como FB15k-237 (Toutanova & Chen, 2015), WN18RR (Dettmers et al., 2018) y NELL-995 (Xiong et al., 2017), con el objetivo de que el modelo aprenda patrones estructurales y relacionales generalizables[cite: 153]. [cite_start]La evaluación *zero-shot* se realizará sobre KGs completamente no vistos durante el entrenamiento, sin ningún *fine-tuning* adicional, comparando el desempeño contra ULTRA, NBFNet y KnowFormer[cite: 154].

[cite_start]Un componente importante de esta etapa es el estudio del impacto del entrenamiento multigrafo sobre la estructura del *expander*[cite: 155]. [cite_start]Dado que distintos KGs tienen tamaños y densidades muy diferentes, se investigará si es necesario adaptar el grado del *expander* o su proceso de generación en función del grafo, o si un *expander* de grado fijo es suficientemente robusto para generalizar entre grafos de distinta naturaleza[cite: 156]. [cite_start]Esto tiene implicancias directas tanto para la eficiencia del modelo como para su capacidad de transferencia, y constituye una contribución teórica sobre las propiedades de los grafos *expander* en contextos multigrafo[cite: 157].

[cite_start]El resultado de esta etapa final es una arquitectura Graph Transformer con atención dispersa y capacidad de generalización *zero-shot*, junto con un análisis que fundamente sus propiedades teóricas y empíricas, y que represente una contribución concreta al estado del arte en modelos fundacionales para grafos de conocimiento[cite: 158]. [cite_start]Los resultados de esta etapa se consolidarán en un artículo científico que será sometido a una conferencia de primer nivel en aprendizaje automático y aprendizaje sobre grafos, como NeurIPS, ICML o ICLR[cite: 159].

[cite_start]En el Anexo de este documento se entrega la carta Gantt de las actividades de investigación[cite: 160].

---

### [cite_start]II.2 INFRAESTRUCTURA DISPONIBLE [cite: 165]
Señale medios y recursos con que cuenta para realizar el proyecto. (Extensión máxima de media página) [cite_start][cite: 165, 166].

[cite_start]El estudiante cuenta con financiamiento del **Instituto Milenio Fundamentos de los Datos**, el cual proporciona recursos para el desarrollo de la investigación[cite: 167]. [cite_start]Adicionalmente, cuenta con la **Beca VRI: Ayudante e Instructor Becario de la Pontificia Universidad Católica de Chile**, otorgada por la Escuela de Graduados, que asegura su mantención durante la estadía formal en el programa doctoral[cite: 168]. 

[cite_start]Respecto al espacio físico, cuenta con acceso a las dependencias del Departamento de Ciencias de la Computación de la Escuela de Ingeniería UC[cite: 169]. [cite_start]En cuanto a equipamiento tecnológico, dispone de un equipo de trabajo personal con las siguientes características[cite: 170]:
* [cite_start]**Procesador:** AMD Ryzen 5 5600 [cite: 170]
* [cite_start]**Memoria RAM:** 32 GB [cite: 170]
* [cite_start]**Tarjeta Gráfica:** NVIDIA GeForce RTX 3060 de 12 GB [cite: 170]

[cite_start]Para experimentos de mayor escala, tiene acceso al **Clúster HPC de la Facultad de Ingeniería y Ciencias de la Universidad Adolfo Ibáñez**, equipado con **GPUs NVIDIA H100**, lo que permitirá entrenar y evaluar los modelos propuestos sobre grafos de conocimiento de gran escala[cite: 171].

[cite_start]**Detalle de ítems y montos globales que aporta el Director de Tesis:** [cite: 172]
> [cite_start]Cuento con el apoyo complementario del Instituto Milenio Fundamentos de los Datos, para los costos de manutención durante mi doctorado, pero el monto total depende de la disponibilidad de otros fondos (DCC, VRI, etc) que consiga año a año[cite: 173]. [cite_start]Asimismo, tengo la posibilidad de ser financiado para viajes a conferencias y similares por el Instituto Milenio, y por fondos de investigación entregados por el Departamento de Ciencia de la Computación al Director de Tesis[cite: 174].

---

### [cite_start]II.3 Otros aspectos relevantes para la evaluación [cite: 175]
(Extensión máxima de media página) [cite_start][cite: 176].

[cite_start]Durante la estadía en el doctorado ha sido profesor de la asignatura **Minería de Datos** en la Pontificia Universidad Católica de Chile, y de la asignatura **Algoritmos y Complejidad** en la Universidad Técnica Federico Santa María, esta última impartida durante dos períodos académicos[cite: 177]. [cite_start]Adicionalmente, se ha desempeñado como ayudante de la asignatura **Procesamiento Masivo de Datos** en el Magíster en Analítica para los Negocios de la UC, programa dictado en modalidad en línea a través de la plataforma Coursera[cite: 178].

---

## [cite_start]III.- PRESUPUESTO [cite: 183]

### [cite_start]III.1 RECURSOS DISPONIBLES: Indique cuáles son los recursos que concurren a financiar su tesis[cite: 184].

[cite_start]El financiamiento para esta tesis proviene principalmente del **Instituto Milenio Fundamentos de los Datos**, el cual provee un monto mensual de manutención por 4 años, junto con el apoyo a asistencia a congresos y gastos operacionales[cite: 185]. [cite_start]Además, la manutención mensual se complementa con los fondos concursables cada año que ofrece el Departamento de Ciencia de la Computación o la Vicerrectoría de Investigación y Postgrado[cite: 186].

---

## [cite_start]IV.- ANTECEDENTES CURRICULARES DEL ESTUDIANTE [cite: 187]

### [cite_start]IV.1 ANTECEDENTES ACADÉMICOS Y PROFESIONALES [cite: 188]

| TÍTULOS/GRADOS/CURSO (*) | UNIVERSIDAD | PAÍS | AÑO OBTENCIÓN |
| :--- | :--- | :--- | :--- |
| Magíster en Ciencias de la Ingeniería | Pontificia Universidad Católica | Chile | 2024 |
| Ingeniero Civil Informático | Universidad Técnica Federico Santa María | Chile | 2023 |
| Licenciado en Ciencias de la Ingeniería Informática | Universidad Técnica Federico Santa María | Chile | 2021 |

[cite_start][cite: 189]

### [cite_start]IV.2 TRAYECTORIA EN EL PROGRAMA DE DOCTORADO [cite: 190]
[cite_start]*(Se adjunta el certificado de notas de los cursos realizados en el Programa de Doctorado, indicando semestre y año en que fue cursado y números de créditos)*[cite: 191, 192, 193].

### [cite_start]IV.3 PARTICIPACIÓN DEL ESTUDIANTE EN OTROS PROYECTOS (En ejecución o finalizados) [cite: 199]

| Inicio | Término | TÍTULO | FINANCIAMIENTO (origen y monto) | FUNCIONES DESEMPEÑADAS |
| :--- | :--- | :--- | :--- | :--- |
| 2024 | 2027 | FONDECYT REGULAR 1241462, *Detection of propaganda in social media and characterization of its effects on social networks*, Role: PI: Marcelo Mendoza, 2024-2027. | Aporte en gastos operacionales y viajes. | Ayudante de investigación |
| 2018 | 2027 | INICIATIVA CIENTÍFICA MILENIO, Instituto Milenio de Investigación sobre los Fundamentos de los Datos (IMFD) | Aporte en gastos operacionales y viajes. | Estudiante investigador |

[cite_start][cite: 200]

### [cite_start]IV.4 Publicaciones in extenso [cite: 201]

* **Lermanda, V., Ojeda, M., Reutter, J.** *Evaluating Knowledge Graph Construction from Text Without Supervision*. [cite_start]European Semantic Web Conference (ESWC), 2026[cite: 202].
* **Ojeda, M., & Reutter, J. (2025).** *Using publicly available data for predicting socioeconomic values in urban context*. Computational Urban Science, 2025, vol. 5, no 1, p. [cite_start]32[cite: 202].
* **Carvallo, A., Mendoza, M., Fernández, M., Ojeda, M., Guevara, L., Varela, D., Bórquez, M., Buzeta, N., Ayala, F.** *Hate Explained: Evaluating NER-Enriched Text in Human and Machine Moderation of Hate Speech*. [cite_start]Workshop on Online Abuse and Harms (WOAH) at ACL, 2025[cite: 202].
* **Fernández, M., Ojeda, M., Guevara, L., Varela, D., Mendoza, M., & Barrón-Cedeño, A.** *VICTOR VECTORS @DIPROMATS 2024: Propaganda Detection with LLM Paraphrasing and Machine Translation*. [cite_start]CEUR Workshop Proceedings, 3756, 2024[cite: 205, 207].

### [cite_start]IV.5 Presentaciones a Congresos [cite: 208]

| TÍTULO | CONGRESO | LUGAR/FECHA |
| :--- | :--- | :--- |
| *Evaluating Knowledge Graph Construction from Text Without Supervision* | 23rd European Semantic Web Conference (ESWC) | Dubrovnik, Croatia / Mayo 2026 |

[cite_start][cite: 209]

---

## [cite_start]ANEXO 1: REFERENCIAS [cite: 215, 216]

* Alon, U., & Yahav, E. (2020). On the bottleneck of graph neural networks and its practical implications. [cite_start]*arXiv preprint arXiv:2006.05205*[cite: 217].
* Battaglia, P. W., Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A., Zambaldi, V., Malinowski, M., ... & Pascanu, R. (2018). Relational inductive biases, deep learning, and graph networks. [cite_start]*arXiv preprint arXiv:1806.01261*, 10[cite: 218, 219].
* Bellman, R. (1958). On a routing problem. [cite_start]*Quarterly of applied mathematics*, 16(1), 87-90[cite: 220].
* Bi, Z., Cheng, S., Chen, J., Liang, X., Xiong, F., & Zhang, N. (2024). Relphormer: Relational graph transformer for knowledge graph representations. [cite_start]*Neurocomputing*, 566, 127044[cite: 221, 222].
* Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. [cite_start]*Advances in neural information processing systems*, 26[cite: 223, 224].
* Chen, D., Lin, Y., Li, W., Li, P., Zhou, J., & Sun, X. (2020, April). Measuring and relieving the over-smoothing problem for graph neural networks from the topological view. [cite_start]In *Proceedings of the AAAI conference on artificial intelligence* (Vol. 34, No. 04, pp. 3438-3445)[cite: 225, 226, 227].
* Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., ... & Weller, A. (2020). Rethinking attention with performers. [cite_start]*arXiv preprint arXiv:2009.14794*[cite: 228, 229].
* Dettmers, T., Minervini, P., Stenetorp, P., & Riedel, S. (2018, April). Convolutional 2d knowledge graph embeddings. [cite_start]In *Proceedings of the AAAI conference on artificial intelligence* (Vol. 32, No. 1)[cite: 230, 231].
* Deng, C., Yue, Z., & Zhang, Z. (2024). Polynormer: Polynomial-expressive graph transformer in linear time. [cite_start]*arXiv preprint arXiv:2403.01232*[cite: 232].
* Dwivedi, V. P., & Bresson, X. (2020). A generalization of transformer networks to graphs. [cite_start]*arXiv preprint arXiv:2012.09699*[cite: 233].
* Dwivedi, V. P., Luu, A. T., Laurent, T., Bengio, Y., & Bresson, X. (2021). Graph neural networks with learnable structural and positional representations. [cite_start]*arXiv preprint arXiv:2110.07875*[cite: 234, 235].
* Galkin, M., Yuan, X., Mostafa, H., Tang, J., & Zhu, Z. (2023). Towards foundation models for knowledge graph reasoning. [cite_start]*arXiv preprint arXiv:2310.04562*[cite: 236, 237].
* Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. [cite_start]*Advances in neural information processing systems*, 30[cite: 238, 239].
* Hoory, S., Linial, N., & Wigderson, A. (2006). Expander graphs and their applications. [cite_start]*Bulletin of the American Mathematical Society*, 43(4), 439-561[cite: 245, 246].
* Hu, W., Fey, M., Ren, H., Nakata, M., Dong, Y., & Leskovec, J. (2021). Ogb-lsc: A large-scale challenge for machine learning on graphs. [cite_start]*arXiv preprint arXiv:2103.09430*[cite: 247, 248].
* Kipf, T. N. (2016). Semi-supervised classification with graph convolutional networks. [cite_start]*arXiv preprint arXiv:1609.02907*[cite: 249].
* Kreuzer, D., Beaini, D., Hamilton, W., Létourneau, V., & Tossou, P. (2021). Rethinking graph transformers with spectral attention. [cite_start]*Advances in Neural Information Processing Systems*, 34, 21618-21629[cite: 250, 251].
* Liao, Y. L., & Smidt, T. (2022). Equiformer: Equivariant graph attention transformer for 3d atomistic graphs. [cite_start]*arXiv preprint arXiv:2206.11990*[cite: 252].
* Liu, J., Mao, Q., Jiang, W., & Li, J. (2024). Knowformer: Revisiting transformers for knowledge graph reasoning. [cite_start]*arXiv preprint arXiv:2409.12865*[cite: 253].
* Mao, H., Chen, Z., Tang, W., Zhao, J., Ma, Y., Zhao, T., ... & Tang, J. (2024, July). Position: Graph foundation models are already here. [cite_start]In *Forty-first International Conference on Machine Learning*[cite: 254, 255].
* Müller, L., Galkin, M., Morris, C., & Rampášek, L. (2023). Attending to graph transformers. [cite_start]*arXiv preprint arXiv:2302.04181*[cite: 256].
* Rampášek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D. (2022). Recipe for a general, powerful, scalable graph transformer. [cite_start]*Advances in Neural Information Processing Systems*, 35, 14501-14515[cite: 257, 258].
* Rong, Y., Bian, Y., Xu, T., Xie, W., Wei, Y., Huang, W., & Huang, J. (2020). Self-supervised graph transformer on large-scale molecular data. [cite_start]*Advances in neural information processing systems*, 33, 12559-12571[cite: 259, 260].
* Rong, Y., Huang, W., Xu, T., & Huang, J. (2019). Dropedge: Towards deep graph convolutional networks on node classification. [cite_start]*arXiv preprint arXiv:1907.10903*[cite: 261, 262].
* Shirzad, H., Velingker, A., Venkatachalam, B., Sutherland, D. J., & Sinop, A. K. (2023, July). Exphormer: Sparse transformers for graphs. In *International Conference on Machine Learning* (pp. 31613-31632). [cite_start]PMLR[cite: 263, 264].
* Shirzad, H., Lin, H., Venkatachalam, B., Velingker, A., Woodruff, D. P., & Sutherland, D. J. (2024). Even sparser graph transformers. [cite_start]*Advances in Neural Information Processing Systems*, 37, 71277-71305[cite: 265, 266].
* Sun, Z., Deng, Z. H., Nie, J. Y., & Tang, J. (2019). Rotate: Knowledge graph embedding by relational rotation in complex space. [cite_start]*arXiv preprint arXiv:1902.10197*[cite: 267, 268].
* Teru, K., Denis, E., & Hamilton, W. (2020, November). Inductive relation prediction by subgraph reasoning. In *International conference on machine learning* (pp. 9448-9457). [cite_start]PMLR[cite: 273, 274].
* Topping, J., Di Giovanni, F., Chamberlain, B. P., Dong, X., & Bronstein, M. M. (2021). Understanding over-squashing and bottlenecks on graphs via curvature. [cite_start]*arXiv preprint arXiv:2111.14522*[cite: 275, 276].
* Toutanova, K., Chen, D., Pantel, P., Poon, H., Choudhury, P., & Gamon, M. (2015, September). Representing text for joint embedding of text and knowledge bases. [cite_start]In *Proceedings of the 2015 conference on empirical methods in natural language processing* (pp. 1499-1509)[cite: 277, 278, 279].
* Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. [cite_start]*Advances in neural information processing systems*, 30[cite: 280, 281].
* Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. [cite_start]*arXiv preprint arXiv:1710.10903*[cite: 282].
* Vignac, C., Krawczuk, I., Siraudin, A., Wang, B., Cevher, V., & Frossard, P. (2022). Digress: Discrete denoising diffusion for graph generation. [cite_start]*arXiv preprint arXiv:2209.14734*[cite: 283, 284].
* Wang, L., Zhao, W., Wei, Z., & Liu, J. (2022, May). Simkgc: Simple contrastive knowledge graph completion with pre-trained language models. [cite_start]In *Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: long papers)* (pp. 4281-4294)[cite: 285, 286, 287].
* Xiong, W., Hoang, T., & Wang, W. Y. (2017, September). Deeppath: A reinforcement learning method for knowledge graph reasoning. [cite_start]In *Proceedings of the 2017 conference on empirical methods in natural language processing* (pp. 564-573)[cite: 288, 289].
* Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2018). How powerful are graph neural networks?. [cite_start]*arXiv preprint arXiv:1810.00826*[cite: 290].
* Yang, B., Yih, W. T., He, X., Gao, J., & Deng, L. (2014). Embedding entities and relations for learning and inference in knowledge bases. [cite_start]*arXiv preprint arXiv:1412.6575*[cite: 291, 292].
* Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., ... & Liu, T. Y. (2021). Do transformers really perform badly for graph representation?. [cite_start]*Advances in neural information processing systems*, 34, 28877-28888[cite: 293, 294].
* Yun, S., Jeong, M., Kim, R., Kang, J., & Kim, H. J. (2019). Graph transformer networks. [cite_start]*Advances in neural information processing systems*, 32[cite: 295, 296].
* Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., ... & Ahmed, A. (2020). Big bird: Transformers for longer sequences. [cite_start]*Advances in neural information processing systems*, 33, 17283-17297[cite: 302, 303].
* Zhang, Y., & Yao, Q. (2022, April). Knowledge graph reasoning with relational digraph. [cite_start]In *Proceedings of the ACM web conference 2022* (pp. 912-924)[cite: 304, 305].
* Zhu, Z., Yuan, X., Galkin, M., Xhonneux, L. P., Zhang, M., Gazeau, M., & Tang, J. (2023). $A^{*}$ net: A scalable path-based reasoning approach for knowledge graphs. *Advances in neural information processing systems*, 36, 59323-59336[cite: 306, 307].
* Zhu, Z., Zhang, Z., Xhonneux, L. P., & Tang, J. (2021). Neural bellman-ford networks: A general graph neural network framework for link prediction. [cite_start]*Advances in neural information processing systems*, 34, 29476-29490[cite: 308, 309].

---

## [cite_start]ANEXO: CARTA GANTT [cite: 313, 315]

[cite_start]En el plan de trabajo de esta propuesta se consideran 4 trimestres por año[cite: 316]. [cite_start]En la Figura 1, se presenta una carta Gantt con la distribución de actividades durante la investigación[cite: 317]:

| Actividades | 2024 (T1-T4) | 2025 (T1-T4) | 2026 (T1-T4) | 2027 (T1-T4) | 2028 (T1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Aprobación de asignaturas | | | | | |
| Redacción manuscrito de candidatura | | | | | |
| Defensa de candidatura | | | | | |
| **Etapa 1: Exphormer para Link Prediction en KGs** | | | | | |
| 1.1 Revisión bibliográfica y reproducción baselines | | | | | |
| 1.2 Mecanismo de atención condicionado a la query | | | | | |
| 1.3 Evaluación transductiva WN18RR y FB15k237 | | | | | |
| 1.5 Ablaciones componentes del grafo de interacción <br> 1.6 Síntesis y redacción de resultados Etapa 1 | | | | | |
| **Etapa 2: Codificación Relacional Composicional** | | | | | |
| 2.1 Diseño del grafo de interacción entre relaciones | | | | | |
| 2.2 Implementar módulo de codificación relacional | | | | | |
| 2.3 Análisis de compatibilidad con el expander | | | | | |
| 2.4 Evaluación sobre relaciones no vistas | | | | | |
| 2.5 Síntesis y redacción de resultados Etapa 2 | | | | | |
| **Etapa 3: Integración y Evaluación Zero-Shot** | | | | | |
| 3.1 Integración de componentes de Etapas 1 y 2 | | | | | |
| 3.2 Entrenamiento multigrafo (FB15k237, WN18RR) | | | | | |
| 3.3 Evaluación zero-shot en KGs no vistos | | | | | |
| 3.4 Análisis teórico y ablaciones finales | | | | | |
| 3.5 Síntesis y redacción de resultados Etapa 3 | | | | | |
| Pasantía | | | | | |
| Publicaciones | | | | | |
| Escritura de tesis | | | | | |
| Defensa de tesis | | | | | |

[cite_start]*Figura 1: Carta Gantt plan de trabajo durante la investigación* [cite: 319]