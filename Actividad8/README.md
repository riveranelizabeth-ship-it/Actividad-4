# Actividad 8 — Monitoreo de SentimentAI Banking

Este informe presenta la línea base de SentimentAI Banking en Colab, la respuesta a dos incidentes simulados y la prueba de detección de *data drift*.

## Estructura del README

| Criterio de la rúbrica | Qué encontrarás |
| --- | --- |
| [1. Diseño de monitorización y observabilidad](#1-monitorización-y-observabilidad) · 25 % | Indicadores, línea base en Colab, logs y trazas por solicitud. |
| [2. Diseño y aplicación de alertas inteligentes](#2-alertas-inteligentes) · 25 % | SLO, presupuesto de error, umbrales, prioridades y reglas para evitar alertas repetidas. |
| [3. Simulación de incidentes y uso de runbooks](#3-simulación-de-incidentes-y-runbooks) · 25 % | Procedimientos, tablero, registro de alertas y resultados antes, durante y después de las dos fallas simuladas. |
| [4. Documentación de la respuesta automatizada](#4-respuesta-automatizada) · 25 % | Cambio automático al perfil estable, verificación del cierre y registro en MLflow. |

**Cierre del informe.** [Análisis de resultados](#5-análisis-de-resultados) y [conclusiones](#6-conclusiones).

**Código y evidencias.** [Notebook de línea base](01_linea_base.ipynb), [notebook de incidentes](02_incidentes.ipynb), [script de simulación y alertas](simular_incidentes.py), [prueba de detección de *data drift*](#prueba-de-detección-de-data-drift) y [carpeta de evidencias](evidencias/).

## 1. Monitorización y observabilidad

El modelo clasifica comentarios bancarios como positivos, neutrales o negativos para facilitar la revisión humana. Se toma como versión estable XLM-RoBERTa, el modelo seleccionado en la [Actividad 4](../Readme.md). El [conjunto de referencia](../data/SentimentAI_Banking_Dataset_300.csv) contiene 300 comentarios sintéticos etiquetados, con 100 por clase. Se reutilizó para verificar el comportamiento durante el ejercicio.

| Indicador | Referencia histórica | Uso operativo |
| --- | ---: | --- |
| Recall de comentarios negativos | 95 de 100 | Medir cuántos comentarios negativos se detectan en la muestra etiquetada. |
| F1 macro | 0.9767 | Revisar el desempeño en las tres clases con el mismo peso. |
| Latencia individual p95 | 45.08 ms en GPU Tesla T4 | Comparar tiempos de inferencia. La línea base en CPU se presenta abajo. |

[Resultados originales](../results/metricas_comparativas.csv) · [Revisión del modelo](../model_revisions.json)

### Línea base medida en Colab

| Indicador | Resultado |
| --- | ---: |
| Solicitudes evaluadas | 300 |
| Errores de ejecución | 0 |
| Latencia p95 | 174.45 ms |
| Solicitudes correctas en 500 ms o menos | 300 de 300 (100 %) |
| Presupuesto de error consumido | 0 de 15 incumplimientos permitidos (0 %) |
| Recall de comentarios negativos | 95 de 100 (95 %) |
| F1 macro | 0.976738 |

La prueba usó XLM-RoBERTa en CPU y midió inferencia individual dentro del cuaderno. El umbral de 500 ms se conserva para simular incidentes en este entorno. Los resultados de calidad corresponden al mismo conjunto sintético de referencia de la Actividad 4.

MLflow registró la corrida `linea-base` en el experimento `Actividad 8 - SentimentAI Banking`. La ejecución generó `solicitudes.csv` y `metricas.json`, guardados como artefactos de la corrida y exportados en `/content/evidencias_linea_base`.

- [Notebook de línea base](01_linea_base.ipynb)
- [Resumen de métricas exportado](evidencias/linea_base/metricas.json)
- [Registro de las 300 solicitudes](evidencias/linea_base/solicitudes.csv)

El registro por solicitud incluye fecha y hora UTC, `trace_id`, revisión del modelo, ID del ejemplo, etiqueta, predicción, duración total, estado y tipo de error. Con el mismo `trace_id` se consultan los tiempos de preprocesamiento e inferencia en el CSV. El registro excluye el texto del comentario.

El tablero operativo se diseñó para actualizarse al cerrar cada ventana y mostrar latencia p95, cumplimiento del SLO, consumo de presupuesto, recall negativo y estado de las alertas. En Colab, el gráfico muestra latencia y recall con las marcas de alerta y cierre. MLflow conserva las métricas por ventana y el archivo de eventos. El tablero de evidencia se genera al terminar la reproducción. La actualización en vivo es parte del diseño operativo.

La prueba de *data drift* comparó la longitud de los comentarios y la proporción de palabras fuera del vocabulario de referencia. Los resultados se presentan en la sección 4.

## 2. Alertas inteligentes

SLO aplicado en el ejercicio: al menos el 95 % de las solicitudes válidas debe completarse correctamente en 500 ms o menos dentro de la ventana de evaluación. El presupuesto de error es el 5 % de esas solicitudes. Con 200 solicitudes, se admiten hasta 10 incumplimientos. El consumo del presupuesto se calcula como incumplimientos observados / incumplimientos permitidos.

Cada ventana de la simulación reproduce 300 solicitudes, con 15 incumplimientos permitidos. La latencia corresponde a la duración de cada solicitud dentro del cuaderno.

| Señal | Condición de apertura | Prioridad y acción |
| --- | --- | --- |
| Latencia y errores | Más de 15 incumplimientos en cada una de dos ventanas consecutivas | P1. Retirar la demora inyectada. |
| Recall negativo | Menos de 85 % en cada una de dos ventanas consecutivas, con 100 negativos etiquetados por ventana | P1. Restaurar el perfil estable. |
| Cambio en los textos | Se supera al menos un umbral del detector de drift | Candidato a P2. Revisar la muestra y evaluar el efecto con etiquetas. |

Los dos runbooks de incidentes agrupan avisos por incidente activo y verifican el cierre tras dos ventanas consecutivas recuperadas. La revisión de drift se ejecuta como prueba independiente.

## 3. Simulación de incidentes y runbooks

| Incidente simulado | Detección | Intervención | Verificación |
| --- | --- | --- | --- |
| Incremento de latencia | Incumplimiento sostenido del SLO durante la inyección de demora | Retirar la condición que introduce la demora en la prueba | Medir de nuevo latencia y solicitudes fuera del SLO |
| Degradación en detección de comentarios negativos | Caída sostenida del recall en ejemplos etiquetados tras activar una variante de prueba | Restaurar la versión estable con el runbook de rollback | Confirmar la versión activa y volver a medir recall en la misma muestra |

**Runbook de latencia.** Confirmar dos ventanas con presupuesto excedido. Identificar el perfil de demora en el registro. Desactivar la demora y comprobar durante dos ventanas que se recupera el SLO. Registrar el cierre.

**Runbook de degradación.** Confirmar dos ventanas con recall bajo y verificar que el perfil degradado esté activo. Restaurar el perfil estable y medir el recall en dos ventanas consecutivas con ejemplos etiquetados. Registrar el cierre después de comprobar la recuperación.

### Resultados de la simulación

![Dashboard de Colab con latencia y recall por ventana](evidencias/incidentes/dashboard_incidentes.png)

El p95 resume la latencia en cada ventana. La alerta de latencia se evalúa con la proporción de solicitudes que incumplen el SLO.

Se reprodujeron las 300 solicitudes originales en cada ventana. Se sumaron 700 ms en el perfil de demora. La variante degradada cambió a neutral las predicciones negativas en posiciones pares del registro (índice desde cero). Las acciones cambian el perfil de la simulación.

| Indicador | Ventana 1 · Línea base | Ventana 5 · Alerta | Ventana 7 · Recuperación |
| --- | ---: | ---: | ---: |
| Latencia p95 · incidente de latencia | 174.45 ms | 874.45 ms | 174.45 ms |
| Cumplimiento del SLO · incidente de latencia | 100 % | 0 % | 100 % |
| Consumo del presupuesto · incidente de latencia | 0 % | 2000 % (300 de 15) | 0 % |
| Recall negativo · incidente de calidad | 95 % | 50 % | 95 % |

El pico aislado de la ventana 2 quedó sin alerta. Cada incidente generó un aviso en la ventana 5, tras dos ventanas consecutivas de incumplimiento. Se retiró la demora y se ejecutó el rollback de la variante degradada, respectivamente. El cierre se verificó tras dos ventanas recuperadas.

- [Script y configuración de alertas](simular_incidentes.py)
- [Evidencia de ventanas, alertas y acciones](evidencias/incidentes/resultados.json)
- [Notebook para visualizar y registrar los incidentes en MLflow](02_incidentes.ipynb)

## 4. Respuesta automatizada

El script ejecuta automáticamente el cambio al perfil estable cuando se confirma una alerta. Registra la ventana de activación, la acción y los perfiles anterior y posterior. El cierre exige dos ventanas recuperadas. El procedimiento operativo asigna al responsable del servicio la atención de los incidentes que permanecen abiertos. La señal de drift indica un cambio que amerita revisión.

La ejecución de Colab mostró el tablero y confirmó la corrida `simulacion-incidentes` en MLflow. El notebook exporta `dashboard_incidentes.png` y `resultados_incidentes.json`. El tablero resume ventanas de reproducción y muestra la alerta y el cierre.

### Prueba de detección de data drift

Se construyó un vocabulario con 150 comentarios, se calibraron las métricas con otros 50 y se evaluaron 100 comentarios de control antes y después de añadirles seis palabras de un tema nuevo. Los umbrales experimentales abren revisión si la mediana de longitud cambia al menos 30 % o si la proporción de palabras fuera del vocabulario aumenta al menos 15 puntos porcentuales frente a calibración.

| Ventana | Mediana de palabras | Palabras fuera del vocabulario | Alerta |
| --- | ---: | ---: | --- |
| Calibración | 9 | 20.71 % | — |
| Control | 9 | 20.34 % | No |
| Cambio sintético | 15 | 51.19 % | Sí |

La señal abre una revisión del cambio en los textos. Esta prueba evaluó la detección de cambios en los textos de entrada.

- [Código del detector](detectar_drift.py)
- [Resultados exportados de drift](evidencias/drift/resultados.json)

## 5. Análisis de resultados

- **Línea base.** Las 300 solicitudes se completaron correctamente dentro del límite de 500 ms. El p95 de 174.45 ms respalda el uso del límite en el ejercicio con CPU. El F1 macro de 0.9767 y el recall negativo del 95 % corresponden al conjunto sintético de la Actividad 4.
- **Incidente de latencia.** El p95 aumentó de 174.45 a 874.45 ms. En la ventana de alerta hubo 300 incumplimientos frente a los 15 permitidos, equivalentes a 20 veces el presupuesto de esa ventana. Al retirar la demora, el p95 volvió a 174.45 ms y el cumplimiento del SLO se recuperó al 100 %.
- **Incidente de calidad.** La variante degradada redujo el recall negativo de 95 % a 50 %, una caída de 45 puntos porcentuales. El rollback automático del perfil simulado recuperó la detección de 95 de los 100 casos negativos.
- **Comportamiento de las alertas.** El pico aislado de la ventana 2 quedó sin alerta. Cada falla sostenida generó un aviso en la ventana 5 y el cierre se verificó en la ventana 7, tras dos ventanas consecutivas de recuperación.
- **Detección de drift.** La muestra de control permaneció cerca de la calibración. Al modificar los textos, la longitud mediana aumentó de 9 a 15 palabras y la proporción de palabras fuera del vocabulario pasó de 20.34 % a 51.19 % respecto al control. El detector activó la revisión de esa ventana.

## 6. Conclusiones

La simulación de SentimentAI Banking demostró que las reglas por ventana detectaron fallas sostenidas y que los runbooks recuperaron el cumplimiento del SLO y la detección de comentarios negativos. Las métricas de MLflow, las trazas por solicitud y el registro de eventos permiten seguir el resultado desde la línea base hasta el cierre de cada incidente.

La prueba independiente de drift confirmó la detección de un cambio sintético en los textos de entrada. El alcance de los resultados se circunscribe a datos sintéticos en Colab, a un rollback entre perfiles de la simulación y a un tablero generado al finalizar la reproducción.
