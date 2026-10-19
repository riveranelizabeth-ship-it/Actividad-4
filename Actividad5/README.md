# SentimentAI Banking - versión 2 (Actividad 5)

**Autora:** Elizabeth Rivera Noguez. Gestión de proyectos de inteligencia artificial.

## Evolución del proyecto
La actividad 4 evalúa sentimiento con modelos preentrenados sobre 300 comentarios sintéticos en español. Esta versión entrena dos algoritmos para **clasificar intenciones bancarias en inglés** con BANKING77 público. No es una mejora directamente comparable de las métricas de sentimiento. Conservar los archivos de la actividad 4; agregar esta carpeta `Actividad5/` al mismo repositorio.

## Fuente y alcance
PolyAI: https://github.com/PolyAI-LDN/task-specific-datasets
Casanueva, I., Temcinas, T., Gerz, D., Henderson, M. y Vulić, I. (2020). Efficient Intent Detection with Dual Sentence Encoders. https://aclanthology.org/2020.nlp4convai-1.5/
Licencia CC BY 4.0, incluida en datos/datos_ini/LICENSE_BANKING77.txt. Atribución a PolyAI y autores; modificaciones: normalización y exclusiones documentadas. No son registros de BBVA ni datos representativos de clientes mexicanos. Fuente y revisión fijadas en datos/manifest_origen.json.

## Estructura
- datos/datos_ini/: CSV oficiales originales y licencia.
- datos/datos_limp/: CSV normalizados, calidad.json y exclusiones.csv.
- datos/manifest_origen.json: procedencia, revisión y SHA256.
- fuentes/entrena.ipynb: notebook de Colab con salidas de la ejecución.
- fuentes/datos_prep.py: descarga verificada y limpieza.
- fuentes/train.py: GridSearchCV, entrenamiento y registro MLflow.
- fuentes/restaurar_mlflow.py: recuperación del seguimiento exportado.
- resultados/: métricas, CV, predicciones, probabilidades y gráficos reales.
- evidencias/seguimiento_mlflow.zip: base de seguimiento y artefactos reales exportados.
- evidencias/LEEME.md: índice de las seis capturas reales del panel.
- Reporte_tecnico.pdf, README.md, CHANGELOG.md y requirements.txt.

## Ejecución en Colab
1. Abrir `fuentes/entrena.ipynb` en Colab. El notebook conserva las salidas del entrenamiento.
2. Para repetirlo, tener el ZIP de esta entrega, `SentimentAI_Banking_v2_Actividad5.zip`; ejecutar la primera celda y seleccionarlo cuando se solicite. No requiere GPU ni cuenta de MLflow.
3. Ejecutar instalación, preparación y entrenamiento en orden. Si pip solicita reiniciar, volver a ejecutar la primera celda antes de continuar.
4. El panel de MLflow se muestra dentro de Colab mediante `serve_kernel_port_as_iframe`. Las capturas en `evidencias/` permiten revisarlo sin una sesión activa.
5. Para consultar la ejecución conservada sin entrenar otra vez, extraer la entrega y seguir las instrucciones de restauración local.

## Ejecución local
Python 3.12 recomendado. Desde Actividad5/: `python -m pip install -r requirements.txt`, después `python fuentes/train.py`. Para abrir MLflow: `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`. Visitar http://localhost:5000.
El SQLite y los artefactos usan rutas locales. Para consultar el seguimiento entregado, ejecutar antes `python fuentes/restaurar_mlflow.py`; para reproducir científicamente, ejecutar train.py y generar runs nuevos. No se requiere servidor externo ni publicación automática.

## Diccionario y reglas
CSV original: `text` (consulta en inglés) y `category` (una de 77 intenciones). CSV limpio agrega `id` (split e índice original) y `text_original`; `text` contiene Unicode NFKC, espacios unificados y minúsculas. No se eliminan negaciones, puntuación ni palabras vacías.
No se detectaron vacíos ni etiquetas conflictivas. Se retiraron 7 coincidencias de train con test y 5 duplicados dentro de las particiones. Quedan 9,992 textos de train y 3,079 de test. Todo cambio tiene registro en exclusiones.csv y SHA256 de salida. Estas reglas reducen duplicados exactos normalizados; no garantizan ausencia de paráfrasis similares.
TF-IDF se ajusta dentro de cada fold, usa normalización L2 y no requiere escalado numérico adicional. Las categorías son etiquetas; no se les asigna un orden artificial como variable predictora.

## Diseño del experimento
Semilla 42; cinco folds estratificados comunes a ambos algoritmos; test oficial limpio independiente. Pipeline TF-IDF + modelo, min_df=2, sublinear_tf=True, max_features=30000.
Regresión logística: C=[0.5,2,8], solver=lbfgs, max_iter=1000. Naive Bayes multinomial: alpha=[0.1,0.5,1]. Ambos contrastan ngram_range=(1,1) y (1,2): seis candidatos por modelo y 60 ajustes CV más dos refits. C controla regularización y alpha suaviza términos raros. Se selecciona por F1 macro CV (peso igual a cada categoría), nunca por el test.
Accuracy, precisión, recall y F1 macro acompañan ROC-AUC OVR macro. ROC-AUC alto en 77 clases no implica buen reconocimiento top-1: la separación uno-contra-resto puede ser alta aunque existan confusiones entre intenciones cercanas. No se afirman scores calibrados.

## Hiperparámetros seleccionados
Regresión logística: C=8 y unigramas. Naive Bayes: alpha=0.1 y unigramas. Son las configuraciones con mejor F1 macro de CV entre las evaluadas; no se escogieron por el test.

## Resultados medidos en Colab
```csv
modelo,cv_f1_macro,cv_f1_std,busqueda_segundos,accuracy,precision_macro,recall_macro,f1_macro,roc_auc_ovr_macro
Regresion_logistica,0.8823911252753882,0.005681131483118226,120.97686101800002,0.9006170834686587,0.9049316329887487,0.9006327006327006,0.9007832938048358,0.998049938492605
Multinomial_NB,0.8390779394347989,0.01215352869216011,14.565118680000069,0.8567716791165964,0.8654339250914908,0.8567932067932067,0.8557590980865664,0.9973345955969871
```

La selección por CV favorece la regresión logística: F1 macro CV 0.8824 frente a 0.8391. En test alcanza 90.06% de accuracy y F1 macro 0.9008, frente a 85.68% y 0.8558 de Naive Bayes. La diferencia observada de accuracy es 4.38 puntos porcentuales; bootstrap pareado estratificado (1,000 réplicas, semilla 42) produce IC95% de 3.31 a 5.42 puntos. McNemar exacto: 201 aciertos exclusivos de logística y 66 de NB, p=4.95e-17. Esto evidencia una diferencia en este test; no demuestra utilidad en producción ni generalización a español.
Las desviaciones CV describen variación entre particiones solapadas; no son intervalos independientes ni una prueba inferencial de superioridad. El bootstrap supone que las consultas del test son intercambiables dentro de cada clase.

## Interpretación y deuda técnica
La distribución de clases y longitud se visualizan con datos limpios; los gráficos de confusiones muestran las diez rutas de error más frecuentes. Revisar predicciones completas y ejemplos de error del reporte. El baseline de clase mayoritaria queda en resultados/baseline.json.
Antes de uso real: validar en español con etiquetas independientes; controlar cambio de dominio; evaluar consultas fuera de categoría y abstención; calibrar scores en validación separada si se usarán umbrales; monitorizar categorías y errores; automatizar pruebas de esquema, hash, separación de conjuntos y recuperación del modelo. El prototipo apoya clasificación de consultas, no determina derechos, fraude, otorgamiento de crédito ni procedencia de reclamaciones.

## Trazabilidad y evidencias
MLflow registra dos runs padre, seis hijos por algoritmo y un run de comparación; parámetros, métricas, datasets, modelos recuperables y artefactos. Las versiones de código y datos se identifican por SHA256. Cuando se ejecuta dentro de Git, se registra además el commit; esta ejecución local se identifica como sin_commit_local y conserva hash del código. El registro contiene 15 ejecuciones finalizadas de Colab. Se incluyen el notebook con salidas, las seis capturas y una copia original sin editar en evidencias/entrena_original_colab.ipynb. La versión de trabajo corrige la apertura del panel y añade la conclusión; no se alteraron los resultados guardados. Python de la ejecución: 3.13.16; scikit-learn: 1.8.0; MLflow: 2.22.2. Los tiempos de búsqueda fueron 120.98 s y 14.57 s; dependen del entorno. Los modelos se guardaron como artefactos MLflow recuperables; no se publicaron en un registro central de modelos.

## Uso de IA
Se utilizó asistencia de IA para código, documentación y revisión de consistencia. Los resultados se obtuvieron ejecutando los algoritmos; no fueron inventados. La autora debe revisar ejemplos, conclusiones y evidencias antes de entregar.

## Referencias técnicas
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- https://mlflow.org/docs/2.22.2/tracking.html
- https://github.com/PolyAI-LDN/task-specific-datasets

## Evidencias del panel
Ver [índice de capturas](evidencias/LEEME.md). La captura comparativa muestra Naive Bayes a la izquierda y regresión logística a la derecha; los nombres de las columnas quedan fuera del recorte y se identifican con sus parámetros alpha=0.1 y C=8. Las capturas individuales documentan el nombre de cada modelo.
