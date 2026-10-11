# SentimentAI Banking — Actividad 4

## Caso de uso
Clasificación exploratoria de comentarios bancarios en español en positivo, neutral y negativo, para identificar experiencias y orientar revisión humana. Rama: procesamiento de lenguaje natural. No se automatizan decisiones sobre derechos, fraude ni procedencia de reclamaciones.

## Ecosistema Hugging Face y criterios
Models proporciona checkpoints y fichas; Datasets permite gestionar datos; Spaces sirve para desplegar demostraciones. Aquí se descargan tres modelos desde Models y se utiliza un CSV local sintético, sin publicar un Space. Se exige español, tres clases comparables, facilidad de carga y documentación. Se contrasta F1 macro, recall negativo, latencia, parámetros, memoria y condiciones de uso.

|             | repo                                          | revision                                 | licencia_metadatos   |
|:------------|:----------------------------------------------|:-----------------------------------------|:---------------------|
| BETO        | finiteautomata/beto-sentiment-analysis        | 9384c7f339a6d62c9e2e1d686f225f92e3a3353b | no declarada         |
| RoBERTuito  | pysentimiento/robertuito-sentiment-analysis   | a2cc0f67ebd705c55191e25a05ba23d885fcc09b | no declarada         |
| XLM-RoBERTa | cardiffnlp/twitter-xlm-roberta-base-sentiment | f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8 | no declarada         |

BETO es una referencia monolingüe BERT; RoBERTuito usa RoBERTa para texto social español; XLM-RoBERTa aporta una alternativa multilingüe. No se atribuyen diferencias únicamente a la arquitectura. BETO anuncia retiro futuro: se conserva la revisión exacta y no se reemplaza silenciosamente.

## Datos y anotación
300 textos sintéticos asistidos por IA, con 100 por clase y 30 por cada uno de diez temas. No son registros de clientes ni datos internos de BBVA. Las etiquetas se propusieron durante la generación y no constituyen una anotación independiente. Positivo: valoración favorable; neutral: ausencia de valoración clara; negativo: insatisfacción. En casos mixtos prevalece el juicio global explícito. Revisar ironía y negación antes de interpretar errores.

Campos: `id` identificador único; `comentario` entrada; `sentimiento` referencia; `tema` contexto (no usado como entrada); `origen` procedencia sintética. Se diseñaron 10 textos por tema y clase y se mezclaron con semilla 42. `src/reconstruir_dataset.py` reconstruye exactamente los bytes archivados; no genera ejemplos nuevos. SHA-256: `c60ef41538a6538838b24cc06be77ec1a1b78c3b5db0401319e9eca4314ccc58`.

## Entorno y reproducción
GPU: **Tesla T4**; PyTorch: **2.11.0+cu130**; CUDA: **13.0**. Ejecución UTC: 2026-10-11T14:28:08.385433+00:00. `results/entorno.json` conserva condiciones, `model_revisions.json` revisiones y `results/dependencias_ejecutadas.txt` versiones efectivas.

1. Abre `SentimentAI_Banking_Actividad4.ipynb` en Colab con el botón **Open in Colab** y activa GPU. El cuaderno se guardó en la raíz del repositorio; si se organiza posteriormente en `notebooks/`, utiliza esa ruta.
2. Ejecuta instalación, configuración y autenticación con un token de lectura en Secretos `HF_TOKEN` o mediante el campo de entrada oculta que ofrece el cuaderno. No pegues el token en el código.
3. Ejecuta todas las celdas; los datos están incluidos en el notebook.
4. Para repetir revisiones anteriores, conserva `model_revisions.json` en la raíz `/content/SentimentAI_Banking` antes de ejecutar la celda de revisiones.
5. Descarga resultados y guarda el notebook con sus salidas. Conserva su ubicación en la raíz o actualiza los enlaces si lo mueves a `notebooks/`.
6. Sube la estructura completa a GitHub, verifica acceso del profesor y entrega el enlace.

## Metodología
Pesos congelados y mismos 300 textos para todos. FP32, lote 16, máximo 128 tokens, semilla 42; conserva acentos, negaciones y puntuación. El tokenizador aplica su normalización propia. No se utilizan todos los preprocesadores de las librerías originales. Mapeo explícito de `id2label`. No hay entrenamiento ni optimización con estos datos: es una evaluación exploratoria; futuros ajustes requieren desarrollo y prueba independientes.

## Calidad medida
| modelo      |   accuracy |   precision_macro |   recall_macro |   f1_macro |   recall_negativo |
|:------------|-----------:|------------------:|---------------:|-----------:|------------------:|
| BETO        |     0.89   |            0.9132 |         0.89   |     0.8908 |              0.91 |
| RoBERTuito  |     0.9533 |            0.9557 |         0.9533 |     0.9532 |              0.99 |
| XLM-RoBERTa |     0.9767 |            0.9776 |         0.9767 |     0.9767 |              0.95 |

Baseline constante neutral: accuracy 0.3333, F1 macro 0.1667. Precision, recall y F1 globales usan promedio macro y cero para divisiones indefinidas. Métricas por clase, matrices y predicciones completas están en `results/`.

## Eficiencia y recursos
| modelo      |   parametros_millones |   parametros_fp32_MB |   latencia_mediana_ms |   latencia_p95_ms |   gpu_pico_asignado_MB |   comentarios_por_segundo_lote |   textos_truncados | licencia_metadatos   |
|:------------|----------------------:|---------------------:|----------------------:|------------------:|-----------------------:|-------------------------------:|-------------------:|:---------------------|
| BETO        |               109.856 |              419.068 |                17.214 |            49.217 |                439.566 |                        617.013 |                  0 | no declarada         |
| RoBERTuito  |               108.791 |              415.005 |                 8.227 |            14.22  |                435.726 |                        779.337 |                  0 | no declarada         |
| XLM-RoBERTa |               278.046 |             1060.66  |                26.582 |            45.076 |               1086.84  |                        659.614 |                  0 | no declarada         |

Latencia individual: mismos 30 comentarios, 3 repeticiones, 5 solicitudes de calentamiento y sincronización GPU. Incluye tokenización, transferencias y salida CPU; excluye descarga/carga. La mediana y p95 resumen 90 solicitudes por modelo. Rendimiento por lotes se mide separadamente. El tamaño FP32 corresponde a parámetros, no al archivo descargado. Aunque el encabezado usa MB, el cálculo divide entre 1024²: los tamaños de parámetros y memoria están expresados en MiB. La memoria es el pico asignado por PyTorch, incluye pesos y tensores y no toda la reserva CUDA. `carga_y_descarga_s` incluye el efecto de red y caché. Las diferencias de tiempo entre sesiones o GPU no deben compararse directamente. No se estima costo monetario sin una tarifa concreta; parámetros, memoria y tiempo son indicadores de recursos.

## Balance y selección exploratoria
Se toman modelos a no más de 0.01 del F1 macro máximo y se prioriza recall negativo; después, menor mediana de latencia. La tolerancia no implica significancia estadística. Esta regla seleccionó **XLM-RoBERTa**, con F1 macro **0.9767**, recall negativo **0.9500** y mediana **26.58 ms**. El más rápido fue **RoBERTuito**. XLM-RoBERTa acertó 293 de 300 comentarios frente a 286 de RoBERTuito: siete aciertos adicionales, con una latencia individual aproximadamente 3.23 veces mayor y cerca de 2.5 veces el pico de memoria asignada. RoBERTuito detectó 99 de los 100 negativos; XLM-RoBERTa detectó 95. Por ello, la selección depende del criterio: XLM-RoBERTa es la opción académica conforme a la regla previamente definida; RoBERTuito es una alternativa relevante si se priorizan detección de insatisfacción y rapidez. Cambiar esa prioridad requeriría declarar un nuevo criterio y validarlo con datos independientes.

## Ajuste mínimo, umbral y revisión humana
No se hace fine-tuning en esta fase; primero se mide la capacidad transferida de los checkpoints. Se fija score máximo 0.70 para sugerir revisión humana, sin alterar la clasificación base ni optimizar el umbral con la evaluación.

| modelo      |   cobertura_umbral_070 |   accuracy_retenidos |
|:------------|-----------------------:|---------------------:|
| BETO        |                 0.98   |               0.898  |
| RoBERTuito  |                 0.9467 |               0.9859 |
| XLM-RoBERTa |                 0.91   |               0.989  |

Con el umbral 0.70 se sugieren para revisión 6 comentarios con BETO, 16 con RoBERTuito y 27 con XLM-RoBERTa. Entre los comentarios retenidos todavía quedan 30, 4 y 3 errores, respectivamente. Esto demuestra que revisar solo scores bajos no elimina los errores de score alto.

La cobertura indica qué fracción alcanza o supera el umbral; la accuracy retenida se calcula solo en ese subconjunto y no sustituye la métrica global. Scores sin calibración no garantizan probabilidades reales. Para calibrar o ajustar este umbral, usa nuevos conjuntos de desarrollo y prueba.

## Análisis de errores concretos
Los errores se comparan con las etiquetas de referencia propuestas al generar los textos; estas aún requieren revisión humana independiente.

| Modelo | Comentario | Referencia | Predicción | Score máximo |
|---|---|---|---|---|
| BETO | Qué práctica está la función para guardar mis cuentas frecuentes. | positivo | neutral | 0.9984 |
| RoBERTuito | Pagué la luz en segundos, qué servicio tan práctico. | positivo | negativo | 0.7231 |
| XLM-RoBERTa | Qué maravilla, otra actualización que deja la app inutilizable. | negativo | positivo | 0.8762 |

**BETO: valoración favorable clasificada como neutral.** La expresión «qué práctica» valora positivamente la función. Aun así, BETO predijo neutral con score 0.9984. Este caso es consistente con su matriz: 23 de los 100 positivos fueron clasificados como neutrales. El score expresa la salida del modelo y no garantiza que la interpretación sea correcta.

**RoBERTuito: satisfacción explícita clasificada como negativa.** El comentario sobre pagar la luz describe rapidez y utilidad, pero el modelo lo clasificó como negativo. Su score 0.7231 supera el umbral y no activaría la revisión por score bajo. El ejemplo demuestra un error, pero por sí solo no permite atribuirlo a una palabra o causa interna del modelo.

**XLM-RoBERTa: ironía clasificada como positiva.** «Qué maravilla» introduce una queja: la app quedó inutilizable. La interpretación global es negativa, aunque la predicción fue positiva con score 0.8762. El modelo clasificó cinco negativos como positivos; varios contienen elogios irónicos. Esta asociación observada sugiere probar más ejemplos de ironía, sin demostrar por sí sola el mecanismo del error.

En el conjunto completo, BETO cometió 33 errores, RoBERTuito 14 y XLM-RoBERTa 7. Los cinco falsos negativos de XLM-RoBERTa son relevantes para el caso de uso porque podrían ocultar insatisfacción. Tres de ellos superan 0.70 y permanecerían fuera de la revisión sugerida por score bajo. Conviene complementar el umbral con auditoría de predicciones de score alto y evaluación específica de ironía, negación y opiniones mixtas. `results/errores.csv` conserva todos los casos.

## Licencias, riesgos y límites
Las licencias indicadas en la tabla son metadatos del Hub; si no se declaran, no se presume autorización comercial. La ficha de BETO menciona uso no comercial/investigación y condiciones de datasets de terceros; para RoBERTuito y XLM-RoBERTa deben revisarse fichas y condiciones de datos antes de plantear uso empresarial. Se conserva copia de cada ficha en `results/ficha_*.md`.

El dataset equilibrado y artificial no representa frecuencia, diversidad lingüística ni complejidad de clientes reales. Algunas frases tienen pistas muy explícitas; puede sobreestimar calidad. Riesgos: ironía, opiniones mixtas, etiquetas dudosas, cambio de dominio desde Twitter, falsos negativos, scores no calibrados y futura indisponibilidad de BETO. Próxima validación: textos independientes, anotación humana, revisión de desacuerdos, privacidad y evaluación por producto/canal. No se recomienda producción con esta evidencia.

## Conclusión
XLM-RoBERTa obtuvo el mejor desempeño global sobre los 300 comentarios sintéticos: exactitud de 97.67 % y F1 macro de 0.9767. Se recomienda para continuar la evaluación académica conforme al criterio definido antes de ejecutar. Esta ventaja exige más recursos y una latencia individual de 26.58 ms, frente a 8.23 ms de RoBERTuito.

RoBERTuito ofrece un balance favorable cuando la prioridad es detectar insatisfacción con rapidez: identificó el 99 % de los negativos y obtuvo exactitud de 95.33 %. BETO alcanzó 89 % de exactitud y confundió especialmente comentarios positivos con neutrales. La recomendación global y la alternativa orientada a negativos responden a prioridades distintas que deben explicitarse.

Los errores de ironía y los scores altos asociados a predicciones incorrectas muestran que el umbral 0.70 debe complementarse con revisión humana. La siguiente fase requiere datos representativos e independientes, revisión de etiquetas y condiciones de uso de los modelos. El experimento no acredita impacto operativo ni justifica una implementación en producción: proporciona evidencia reproducible para decidir qué alternativas seguir evaluando.

## Trazabilidad con la rúbrica
| Criterio | Evidencia |
|---|---|
| Selección y análisis (25%) | Caso de uso, ecosistema, modelos, criterios y fichas. |
| Colab con GPU (25%) | Notebook guardado con salidas, entorno, autenticación y revisiones. |
| Evaluación (25%) | Métricas, latencias crudas, matrices, gráfica y ejemplos de errores. |
| Documentación (25%) | README, dataset, script, dependencias y repositorio accesible. |

## Fuentes
- [BETO — ficha](https://huggingface.co/finiteautomata/beto-sentiment-analysis).
- [RoBERTuito — ficha](https://huggingface.co/pysentimiento/robertuito-sentiment-analysis).
- [XLM-RoBERTa — ficha](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment).
- Pérez et al. (2022). [RoBERTuito](https://aclanthology.org/2022.lrec-1.785/).
- Barbieri et al. (2022). [XLM-T](https://aclanthology.org/2022.lrec-1.27/).
- [Autenticación Hugging Face](https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication).

**Uso de IA:** datos sintéticos y código preparados con asistencia de ChatGPT; métricas calculadas al ejecutar, y análisis de errores preparado con asistencia de IA para revisión y validación de la autora. Las etiquetas siguen pendientes de revisión independiente. No se reutilizan las métricas publicadas por los desarrolladores como resultados propios.
