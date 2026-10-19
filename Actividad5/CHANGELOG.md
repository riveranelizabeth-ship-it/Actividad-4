# CHANGELOG

## 2.0.0 - 2026-10-18
- Se conserva la actividad 4 como antecedente; nueva tarea de intención bancaria en inglés.
- Se incorpora BANKING77 de PolyAI, revisión 57ec275d8078af65b7731c2a98be812d844a6d6b; CC BY 4.0.
- Normalización NFKC, espacios y minúsculas; 7 solapamientos retirados de train y 5 duplicados adicionales. Sin nulos vacíos ni conflictos detectados.
- Versiones separadas de datos originales/limpios, manifiesto SHA256 y exclusiones.
- Dos pipelines, 12 candidatos en total y validación de cinco folds. Selección por F1 macro CV.
- Ejecución real, seguimiento MLflow, modelos guardados y prueba de recuperación.
- Comparación pareada mediante McNemar y bootstrap de diferencia de accuracy.
- Integrada la ejecución de Colab: Python 3.13.16, 15 runs finalizados y seis capturas reales.
- Notebook con resultados conservados; apertura del panel corregida para iframe y conclusión incorporada. Copia original recibida conservada como evidencia.
- Reporte actualizado con resultados de Colab y anexo de capturas.
- La publicación en GitHub se realiza después de esta preparación; el entrenamiento conserva la etiqueta sin_commit_local y el hash de código original.
