# Actividad 6 | Reporte técnico de evaluación y validación de modelos de clasificación bancaria

**Asignatura:** Gestión de Proyectos de Inteligencia Artificial  
**Problema:** clasificación de 77 intenciones bancarias  
**Evidencias:** `evaluacion_actividad6_ejecutada.ipynb`, `actividad6_complemento_roc_ab.ipynb`  
**Tablero visual:** [Trello — Actividad 6](https://trello.com/b/6adeb877c9f91d34a5819b19)

## Resumen ejecutivo

Se compararon dos clasificadores de intenciones bancarias, **Regresión Logística (RL)** y **Multinomial Naive Bayes (NB)**, mediante métricas en prueba, matrices de confusión, validación cruzada, comparación con una regla de clase mayoritaria, curvas ROC/AUC multiclase, umbrales de aceptación y análisis estadísticos. En un conjunto de **3,079 observaciones y 77 clases**, RL alcanzó **F1 macro = 0.900783**, frente a **0.855759** de NB, y cometió **306 errores**, frente a **441**. RL cumplió la meta SMART de **F1 macro ≥ 0.90 en prueba**. La validación cruzada de cinco particiones favoreció también a RL, aunque su F1 macro medio (**0.882391**) fue inferior a 0.90. Se propone **RL como candidata a piloto supervisado**, no como sistema autorizado para producción.

## 1. Definición del problema, contexto y objetivo SMART

La identificación de la intención expresada en una consulta bancaria permite orientar la solicitud hacia el flujo de atención adecuado. La tarea es **multiclase**: cada observación se asigna a una de **77 intenciones**. Confundir solicitudes relacionadas, por ejemplo, con transferencias, tarjetas o verificación de identidad puede producir canalizaciones incorrectas y retrabajo. El proyecto evalúa capacidad predictiva, errores y viabilidad de derivar a revisión humana los casos inciertos.

**Objetivo SMART del experimento:** evaluar RL y NB sobre el conjunto de prueba disponible, utilizando F1 macro como métrica principal, y seleccionar un modelo que alcance **F1 macro ≥ 0.90**, con evidencia adicional de validación cruzada, errores, discriminación ROC/AUC y comportamiento ante umbrales. El alcance temporal de esta meta es la evaluación de la Actividad 6; **no** constituye una promesa de desempeño futuro en operación.

Se prioriza **F1 macro** porque promedia el F1 de las clases dando el mismo peso a cada intención, aun si su frecuencia difiere. La exactitud global puede ocultar un desempeño deficiente en intenciones poco frecuentes. Para un despliegue bancario se necesitarían, además, indicadores por clase y severidad operativa del error.

## 2. Metodología y fuentes de evidencia

Los notebooks ejecutados cargan los artefactos de resultados de la Actividad 5 (`Actividad5_ejecutada.zip`) y analizan predicciones, puntuaciones y métricas registradas. **No deben interpretarse como un nuevo entrenamiento completo dentro de la Actividad 6**: la evidencia de esta fase corresponde principalmente a evaluación y validación de resultados previos. El conjunto evaluado contiene **3,079 muestras** de **77 categorías**.

Los modelos son: **Regresión Logística**, un clasificador discriminativo, y **Multinomial Naive Bayes**, un clasificador probabilístico. Además, el notebook principal recupera un **baseline trivial de clase mayoritaria del entrenamiento**. Esta distinción es importante: **NB es el modelo alternativo**, mientras que la **regla de clase mayoritaria es el baseline de referencia**.

La evaluación combina: (i) métricas sobre el conjunto de prueba; (ii) matrices de confusión y principales pares de error; (iii) validación cruzada de cinco particiones registrada en los artefactos; (iv) curvas ROC/AUC One-vs-Rest (OvR); (v) umbrales de aceptación por confianza máxima; (vi) prueba pareada de McNemar; y (vii) simulación A/B retrospectiva con grupos históricos separados aleatoriamente.

**Trazabilidad:** las cifras de las secciones siguientes proceden de las salidas ejecutadas de los dos notebooks. Para reproducirlas se necesita también el ZIP de resultados de la Actividad 5, que los notebooks solicitan cargar. El reporte no afirma haber reentrenado modelos ni validado su desempeño en producción.

## 3. Métricas y comparación con baseline

La **exactitud** es la fracción de predicciones correctas; **precision** mide qué proporción de predicciones de una clase es correcta; **recall** mide qué proporción de casos reales de esa clase se recupera; y **F1** es la media armónica de precision y recall. El **F1 macro** es el promedio no ponderado de los F1 por intención.

| Métrica de prueba | Regresión Logística | Naive Bayes |
|---|---:|---:|
| **F1 macro** | **0.900783** | 0.855759 |
| **Exactitud** | **90.0617 %** | 85.6772 % |
| **Errores de clasificación** | **306** | 441 |
| **Meta F1 macro ≥ 0.90** | **Cumple** | No cumple |

La diferencia de F1 macro es **0.045024** a favor de RL. En las mismas 3,079 observaciones, RL acertó **2,773** y NB **2,638**: **135 errores menos** para RL. Estos conteos son coherentes con las exactitudes anteriores.

**Baseline de clase mayoritaria (registrado en el notebook):** exactitud **0.012991** (aprox. **1.30 %**) y F1 macro **0.000333**. Este baseline predice siempre la clase más frecuente del entrenamiento; sirve como mínimo de referencia, no como competidor de calidad comparable. Ambos modelos aprendidos lo superan ampliamente. Comparar únicamente RL con NB sin mencionar este baseline habría dejado incompleto uno de los criterios de la rúbrica.

**Interpretación:** RL cumple el objetivo fijado en el conjunto de prueba, pero el F1 macro global no revela por sí solo qué intenciones concentran los errores. Por ello se complementa con las matrices y el análisis por pares de categorías.

## 4. Matrices de confusión e interpretación

Una matriz de confusión de **77 × 77** organiza las clases reales por filas y las predicciones por columnas; su diagonal representa aciertos y las celdas fuera de la diagonal, confusiones. Los notebooks generan las visualizaciones de ambos modelos y un resumen de las confusiones más frecuentes.

| Modelo | Errores totales | Ejemplos de confusiones frecuentes en el notebook |
|---|---:|---|
| Regresión Logística | **306** | `balance_not_updated_after_bank_transfer` → `transfer_not_received_by_recipient` (5); `virtual_card_not_working` → `get_disposable_virtual_card` (5); `fiat_currency_support` → `exchange_via_app` (4) |
| Naive Bayes | **441** | `card_swallowed` → `declined_cash_withdrawal` (8); `contactless_not_working` → `card_not_working` (8); `why_verify_identity` → `verify_my_identity` (7) |

Estas confusiones muestran que algunas intenciones tienen vocabulario o contextos próximos, aunque las etiquetas representen necesidades distintas. La frecuencia de un error no equivale automáticamente a su **gravedad**: para una evaluación operativa habría que distinguir confusiones informativas de aquellas capaces de afectar una operación o la experiencia del cliente.

**Evidencia visual:** las matrices completas están en la sección **«Matriz de confusión y errores frecuentes»** de `evaluacion_actividad6_ejecutada.ipynb` y también se organizaron en el tablero de Trello. Su lectura debe acompañarse de las etiquetas de clase y no solo del total de errores.

## 5. Validación cruzada y consistencia

La validación cruzada registrada utiliza **cinco particiones** y resume el F1 macro medio y su desviación estándar:

| Modelo | F1 macro medio (5 folds) | Desviación estándar | F1 macro en prueba |
|---|---:|---:|---:|
| **Regresión Logística** | **0.882391** | **0.005681** | **0.900783** |
| Naive Bayes | 0.839078 | 0.012154 | 0.855759 |

RL presenta una ventaja media de **0.043313** puntos de F1 macro sobre NB y menor variabilidad entre folds. Su resultado de prueba supera en **0.018392** su media de validación cruzada. Esta brecha exige prudencia: **el objetivo se cumplió en prueba, pero no en el promedio de validación cruzada**. La desviación entre folds describe variabilidad observada y **no constituye por sí misma un intervalo de confianza independiente**.

Los notebooks consumen los resúmenes de validación cruzada almacenados en los resultados de la Actividad 5; no incluyen el detalle individual de los cinco folds en estas salidas. Para una auditoría exhaustiva del esquema de partición y del proceso de entrenamiento debe consultarse el código de origen de esa actividad.

## 6. Curvas ROC y AUC multiclase

Se utilizó el enfoque **One-vs-Rest (OvR)**: para cada una de las 77 intenciones se considera temporalmente esa intención como positiva y las otras 76 como negativas. La curva ROC relaciona la tasa de verdaderos positivos con la tasa de falsos positivos al variar el umbral de puntuación. El AUC resume la capacidad discriminativa a través de dichos umbrales.

| Indicador | Regresión Logística | Naive Bayes |
|---|---:|---:|
| **AUC micro** | **0.998064** | 0.996836 |
| **AUC macro OvR** | **0.998050** | 0.997335 |

Ambos modelos muestran alta discriminación según estos indicadores; RL tiene una ventaja pequeña pero consistente. **AUC no es equivalente a F1 ni a exactitud**: un modelo puede ordenar adecuadamente las probabilidades y aun así equivocarse al escoger una única etiqueta. Por eso se integra con los resultados de prueba y la matriz de confusión.

**Evidencia visual:** las dos curvas están en `actividad6_complemento_roc_ab.ipynb`, sección **«Curvas ROC multiclase (One-vs-Rest)»**, y sus capturas están vinculadas en Trello.

## 7. Ajuste de umbrales de confianza y revisión humana

El notebook principal estudia una regla de **aceptación selectiva**: se acepta automáticamente la etiqueta de mayor puntuación solo si su `score_maximo` alcanza el umbral elegido; de lo contrario, se deriva el caso a revisión. **Este umbral multiclase de aceptación no es el mismo que el umbral binario usado para construir las curvas ROC OvR.**

**Regresión Logística — resultados observados:**

| Umbral | Cobertura automática | Exactitud entre aceptados | Casos a revisión | Errores automáticos |
|---:|---:|---:|---:|---:|
| 0.0 | 100.00 % | 90.06 % | 0 | 306 |
| 0.3 | 94.28 % | 92.87 % | 176 | 207 |
| 0.5 | 83.89 % | 96.28 % | 496 | 96 |
| **0.7** | **69.34 %** | **98.17 %** | **944** | **39** |
| 0.8 | 60.18 % | 98.97 % | 1,226 | 19 |
| 0.9 | 43.55 % | 99.33 % | 1,738 | 9 |

El umbral **0.70** representa un compromiso ilustrativo: automatiza **69.34 %** de los casos, con **98.17 % de exactitud entre los aceptados**, y deriva **944 de 3,079** a revisión. Frente al umbral 0, reduce los errores automáticos de **306 a 39**, pero aumenta la carga humana. Un umbral de 0.9 mejora aún más la exactitud entre aceptados, a costa de revisar **1,738** casos.

**No se ha demostrado que 0.70 sea óptimo**: la elección definitiva depende del costo de cada error, la capacidad de revisión y la calibración de las puntuaciones. La gráfica de **cobertura vs. exactitud** del notebook permite observar este intercambio. La exactitud de los casos aceptados **no debe presentarse como exactitud sobre todas las solicitudes**.

## 8. Comparación estadística y simulación A/B retrospectiva

### 8.1 Comparación pareada de los modelos

El notebook principal compara aciertos y errores de RL y NB sobre **las mismas observaciones**. La tabla pareada registra: ambos fallan **240** casos; solo NB acierta **66**; solo RL acierta **201**; ambos aciertan **2,572**. La prueba de **McNemar** reportada en la ejecución arroja **p ≈ 2.39 × 10⁻¹⁶**, lo que respalda una diferencia estadística en tasas de acierto. Un artefacto histórico adicional registra un valor exacto aproximado de **4.95 × 10⁻¹⁷**; la conclusión de significancia es consistente, aunque el método específico produce valores distintos.

Esta comparación **no es una prueba A/B en producción**. El artefacto histórico también registra una diferencia de exactitud pareada de **0.043845** (4.38 puntos porcentuales), con intervalo bootstrap estratificado de 95 % de aproximadamente **3.31 a 5.42 puntos**.

### 8.2 Simulación A/B retrospectiva

El notebook complementario realiza otra comparación, **distinta de la anterior**: asigna aleatoriamente observaciones históricas a dos grupos y evalúa un modelo por grupo.

| Grupo histórico simulado | Casos | Aciertos | Exactitud |
|---|---:|---:|---:|
| **A: Naive Bayes** | 1,539 | 1,303 | **84.67 %** |
| **B: Regresión Logística** | 1,540 | 1,395 | **90.58 %** |

La diferencia observada es **5.92 puntos porcentuales** a favor del grupo B, con **IC del 95 % aproximado de 3.60 a 8.24 puntos** y **p bilateral ≈ 6.13 × 10⁻⁷** según la prueba ejecutada. Esta diferencia no coincide exactamente con la comparación pareada anterior porque **son diseños y subconjuntos de evaluación diferentes**; no se deben mezclar sus porcentajes.

**Límite fundamental:** la simulación se hizo con datos históricos y una asignación artificial. No mide impacto causal en tiempos de atención, satisfacción, costos ni riesgos en usuarios reales. Para un A/B operativo sería necesario definir población, asignación aleatoria real, métrica primaria, duración, salvaguardas y criterios de suspensión.

## 9. Decisión técnica e impacto potencial en el negocio

**Modelo recomendado: Regresión Logística, para un piloto controlado con supervisión humana.** La decisión integra: (a) cumplimiento de la meta F1 macro en prueba; (b) **135 errores menos** que NB; (c) mejor media y menor dispersión en validación cruzada; (d) AUC OvR ligeramente superior; (e) reducción de errores automáticos al elevar el umbral; y (f) diferencias estadísticamente significativas en los análisis históricos.

En banca, una clasificación más confiable **podría** mejorar el direccionamiento de consultas, reducir retrabajo y reservar la atención humana para casos inciertos. **No se midieron ahorros, tiempos de respuesta ni satisfacción**, de modo que esos beneficios son hipótesis a contrastar y no resultados demostrados.

Se recomienda: **(1)** evaluar precisión, recall y errores por intención, con especial atención a errores críticos; **(2)** probar y calibrar umbrales con capacidad real de revisión; **(3)** mantener una vía de atención humana y trazabilidad; **(4)** monitorear deriva de datos y desempeño posterior; **(5)** proteger los datos utilizados; y **(6)** ejecutar un piloto A/B genuino antes de automatizar decisiones operativas a escala.

## 10. Conclusiones y limitaciones

La evidencia favorece a **Regresión Logística** frente a **Multinomial Naive Bayes** para el problema de 77 intenciones bancarias. RL alcanzó **F1 macro 0.900783** en prueba, superó a NB en validación cruzada y cometió menos errores. El análisis ROC/AUC mostró elevada capacidad discriminativa; el ajuste de umbrales cuantificó la relación entre automatización y revisión humana; y las comparaciones estadísticas respaldaron diferencias de desempeño sobre datos históricos.

Las limitaciones principales son: el **F1 macro medio en validación cruzada de RL (0.882391) no alcanza 0.90**; los resultados provienen de datos históricos; el umbral 0.70 no está optimizado contra costos operativos; y la simulación A/B no sustituye un experimento real. Además, los notebooks de la Actividad 6 dependen de artefactos de la Actividad 5 para reproducir los resultados. En consecuencia, la decisión es **continuar hacia una validación operativa controlada**, no desplegar automáticamente el clasificador.

## 11. Archivos, evidencias y referencias internas

- **`evaluacion_actividad6_ejecutada.ipynb`**: baseline de clase mayoritaria, métricas, matrices de confusión, umbrales, validación cruzada y McNemar.
- **`actividad6_complemento_roc_ab.ipynb`**: curvas ROC/AUC OvR y simulación A/B retrospectiva.
- **`Actividad5_ejecutada.zip`**: insumo de resultados requerido para volver a ejecutar los notebooks, según las celdas de carga.
- **[Tablero de Trello](https://trello.com/b/6adeb877c9f91d34a5819b19)**: visualización, comparación, organización de evidencias y decisión final.

**Nota de entrega:** este reporte está redactado en Markdown para GitHub. La indicación académica habla de una **extensión sugerida** de 6–8 páginas, no de una paginación inherente a Markdown; si se requiere acreditar una longitud física, exportar a PDF y revisar su paginación. Las gráficas se consultan en los notebooks y en Trello, por lo que el repositorio debe conservar esos vínculos y archivos accesibles.
