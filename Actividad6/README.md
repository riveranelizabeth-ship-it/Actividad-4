# Actividad 6 | Evaluación y validación de modelos de IA

### Clasificación de 77 intenciones bancarias

## Objetivo

Evaluar y comparar modelos de Regresión Logística y Multinomial Naive Bayes para identificar la alternativa más adecuada para clasificar consultas bancarias.

**Meta SMART:** Alcanzar un F1 macro ≥ 0.90 en el conjunto de prueba.

## Metodología

Se realizaron experimentos de evaluación y validación mediante:

- Métricas de desempeño y matrices de confusión.
- Validación cruzada de cinco particiones.
- Curvas ROC/AUC multiclase (One-vs-Rest).
- Análisis de umbrales de confianza.
- Simulación A/B retrospectiva.

## Resultados principales

| Indicador | Regresión Logística | Naive Bayes |
|---|---:|---:|
| F1 macro (prueba) | **0.9008** | 0.8558 |
| F1 macro (validación cruzada) | **0.8824** | 0.8391 |
| Errores de clasificación | **306** | 441 |
| AUC macro | **0.9980** | 0.9973 |

## Decisión técnica

**Modelo seleccionado: Regresión Logística.**

Cumplió el objetivo SMART en el conjunto de prueba, registró 135 errores menos que Naive Bayes y mostró mayor estabilidad en validación cruzada.

La simulación A/B retrospectiva también favoreció su selección, aunque sus resultados requieren validación en condiciones operativas reales.

## Impacto y recomendaciones

El modelo podría mejorar la clasificación y el direccionamiento de consultas bancarias, facilitando la automatización de solicitudes frecuentes.

Para su implementación se recomienda evaluar umbrales de confianza, incorporar supervisión humana y monitorear errores y desempeño por intención.

## Evidencias del proyecto

**Notebooks:**

- `evaluacion_actividad6_ejecutada.ipynb` — Evaluación, métricas y validación.
- `actividad6_complemento_roc_ab.ipynb` — Curvas ROC/AUC y simulación A/B.

**Trello:** Actividad 6 | Evaluación y validación de modelos de IA.

[Agregar enlace al tablero de Trello]

## Conclusión

Regresión Logística presentó el mejor desempeño global y constituye la alternativa más sólida entre los modelos evaluados. Su adopción deberá acompañarse de validación operativa, monitoreo y mecanismos de control.
