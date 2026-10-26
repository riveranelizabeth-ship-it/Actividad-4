1. Objetivo del proyecto
Evaluar y validar modelos de aprendizaje automático para clasificar 77 intenciones bancarias, con el propósito de identificar la alternativa que ofrezca el mejor desempeño y mayor estabilidad.
Objetivo SMART: Alcanzar un F1 macro igual o superior a 0.90 en el conjunto de prueba.
2. Modelos evaluados
Se compararon dos algoritmos de clasificación:
•	Regresión Logística: modelo discriminativo utilizado para predecir la intención bancaria correspondiente.
•	Multinomial Naive Bayes: modelo probabilístico utilizado como alternativa de comparación.
3. Resultados de evaluación
Indicador	Regresión Logística	Naive Bayes
F1 macro en prueba	0.9008	0.8558
F1 macro en validación cruzada	0.8824	0.8391
Desviación estándar en validación cruzada	0.0057	0.0122
Errores de clasificación	306	441
Interpretación: Regresión Logística obtuvo un mejor desempeño en el conjunto de prueba y en la validación cruzada. Además, presentó menor variabilidad entre las cinco particiones evaluadas.
El modelo alcanzó el objetivo SMART establecido y registró 135 errores de clasificación menos que Naive Bayes.
Estos resultados respaldan la selección de Regresión Logística como modelo principal, aunque su implementación requiere validaciones adicionales y supervisión humana.

