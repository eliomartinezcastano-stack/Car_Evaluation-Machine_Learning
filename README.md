# Car_Evaluation - Machine_Learning

# Objetivo del proyecto 
Este proyecto tiene como objetivo construir un modelo de Machine Learning capaz de predecir la aceptabilidad de un coche (unacc, acc, good, vgood) a partir de sus características técnicas y de coste (precio de compra, mantenimiento, seguridad, etc.). Además, se implementa una app en Streamlit para evaluar configuraciones de coches de forma interactiva.

# Contexto del negocio 
Este modelo se podría utilizar como:
    - Filtro inicial en un **portal de compra de coches**
    - Herramienta de apoyo para comparar configuraciones antes de tomar una decisión
    - Sistema de recomendación para distintos perfiles de usuario (p. ej., familias, estudiantes, etc.)
# Dataset
Se utiliza el Car Evaluation Dataset, procedente del UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/19/car+evaluation

- Variables (features):

    - buying: precio de compra (vhigh, high, med, low)

    - maint: coste de mantenimiento (vhigh, high, med, low)

    - doors: número de puertas (2, 3, 4, 5more)

    - persons: capacidad de personas (2, 4, more)

    - lug_boot: tamaño del maletero (small, med, big)

    - safety: nivel de seguridad (low, med, high)


- Target:
    - class: aceptabilidad del coche (unacc, acc, good, vgood)

# Notas sobre calidad del dato
- El dataset no contiene valores nulos.

- Todas las variables son categóricas y requieren encoding para usarse en modelos de ML.

- La variable objetivo está desbalanceada (predomina unacc). Este desbalance proviene del modelo experto original (reglas estrictas), por lo que se decidió no aplicar técnicas de balanceo en este proyecto.

# Preguntas clave 
- ¿Podemos predecir la aceptabilidad de un coche usando solo 6 características?

- ¿Qué modelo clasifica mejor las 4 clases sin quedarse únicamente con la accuracy?

- ¿Mejora el rendimiento al aplicar hyperparameter tuning?

- ¿Cómo convertir el modelo en una herramienta usable (Streamlit) para evaluar coches?


# Proceso de análisis 
1. EDA básica

    - Revisión de dimensiones, tipos de variables y distribución del target.

2. Preprocesamiento

    - Separación X/y

    - Ordinal Encoding para convertir categorías a valores numéricos.

    - Train/Test split con stratify=y para mantener proporción de clases.

3. Modelado

    - Entrenamiento y comparación de:

        - Decision Tree

        - Random Forest

        - Gradient Boosting

4. Evaluación

    - Métricas principales:

        - Accuracy

        - F1 macro (prioritaria por el desbalance y por ser multiclase)

    - Uso de classification_report y confusion_matrix para interpretar resultados.

5. Optimización

    - Ajuste de hiperparámetros con:

        - GridSearchCV

        - RandomizedSearchCV

    - Selección del mejor modelo según F1 macro.

6. Implementación

    - Exportación del modelo (y encoder/pipeline) y creación de una app Streamlit con:

        - selección de características

        - predicción

        - feedback visual

        - sugerencias cuando el coche es unacc

# Resultados / Insights 

- Los modelos de árboles obtuvieron resultados muy altos; el Random Forest fue el modelo más robusto.

- Usar F1 macro fue clave para no “engañarnos” con la accuracy en un dataset desbalanceado.

- El tuning con RandomizedSearchCV permitió mejorar la fiabilidad del modelo (mejor generalización), aunque no siempre suba la accuracy respecto a una única partición train/test.

- La clase unacc es la más fácil de detectar; las clases minoritarias (good, vgood) son más difíciles por su menor representación.

# Recomendaciones de negocio 
- Usar el modelo como filtro inicial: descartar rápidamente configuraciones claramente “inaceptables”.

- Incorporar la app como comparador: permitir que un usuario pruebe configuraciones y reciba sugerencias (p.ej., aumentar capacidad o mejorar seguridad).

- Si el objetivo fuese una recomendación real en la actualidad, sería recomendable reentrenar con datos reales (preferencias de usuarios y ventas) en lugar de reglas expertas.

# Limitaciones
- El dataset proviene de un modelo experto basado en reglas, no de preferencias reales de mercado actuales.

- El encoding ordinal impone un orden numérico que no siempre representa distancias reales entre categorías (aunque funciona bien para este dataset).

- El desbalance es inherente al dataset; al no aplicar balanceo, el modelo puede seguir favoreciendo clases mayoritarias.

- Con solo 6 variables, el modelo no captura factores reales como marca, consumo, fiabilidad, precio en mercado, etc.

# Próximos pasos 
- Redefinir o actualizar los criterios de aceptabilidad con reglas más actuales, o aprenderlos a partir de datos reales de usuarios.

- Mejorar la app: guardar historial de evaluaciones, modo “comparar dos coches”, y recomendaciones más guiadas.