import streamlit as st
import pandas as pd
import joblib
import time 

# DICCIONARIOS DE TRADUCCIÓN
CLASS_ES = {
    "unacc": "No aceptable",
    "acc": "Aceptable",
    "good": "Bueno",
    "vgood": "Muy bueno",
}

buying_map = {
    "Bajo": "low",
    "Medio": "med",
    "Alto": "high",
    "Muy alto": "vhigh"
}
maint_map = {
    "Bajo": "low",
    "Medio": "med",
    "Alto": "high",
    "Muy alto": "vhigh"
}
doors_map = {
    "2 puertas": "2",
    "3 puertas": "3",
    "4 puertas": "4",
    "5 o más": "5more"
}
persons_map = {
    "2 personas": "2",
    "4 personas": "4",
    "Más de 4": "more"
}
lug_boot_map = {
    "Pequeño": "small",
    "Medio": "med",
    "Grande": "big"
}
safety_map = {
    "Baja": "low",
    "Media": "med",
    "Alta": "high"
}

# Función de sugerencias
def improvement_suggestions(buying, maint, persons, safety):
    suggestions = []

    if persons == "2":
        suggestions.append("Aumentar la capacidad a 4 personas")
    
    if safety == "low":
        suggestions.append("Mejorar el nivel de seguridad")
    
    if buying in ["high", "vhigh"]:
        suggestions.append("Reducir el precio de compra")

    if maint in ["high", "vhigh"]:
        suggestions.append("Reducir el coste de manetenimiento")
    
    if not suggestions:
        suggestions.append("No se detectan mejoras claras")
    
    return suggestions

# Para hacer la PRESENTACIÓN
st.set_page_config(page_title="Car Evaluation", layout="centered")

st.sidebar.title("Presentación")
page = st.sidebar.radio(
    "Ir a sección:",
    [
        "0) Portada",
        "1) Introducción",
        "2) Dataset",
        "3) Modelos",
        "4) Demo",
        "5) Conclusiones"
    ]
)


# Cargamos pipeline (encoder + modelo)
model = joblib.load("models/car_model_pipeline.pkl")

# Creamos las secciones para la presentación
if page == "0) Portada":
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center;'> Car Evaluation",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align: center; color: grey;'>Preddición de la aceptabilidad de un coche</h3>",
        unsafe_allow_html=True
    )
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 20px;'>Elio Martínez Castaño</p",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: grey;'>Proyecto de Machine Learning</p>",
        unsafe_allow_html=True
    )
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.info("Usa el menú lateral para navegar por la presentación")

if page == "1) Introducción":
    st.title("Car Evaluation - Machine Learning")
    st.markdown("""
    ### Objetivo del proyecto 
    Desarrollar un modelo de **Machine Learning** capaz de predecir la **aceptabilidad de un coche**
    a partir de sus cracterísticas principales, simulando un sistema de ayuda a la decisión.
    """)
    st.markdown("""
    ### Conexto del negocio
    Este modelo se podría utilizar como:
    - Filtro inicial en un **portal de compra de coches**
    - Herramienta de apoyo para comparar configuraciones
    - Sistema de recomendación para distintos perfiles de usuario
    """)
    st.markdown("""
    ### Enfoque del proyecto
    - Entrenamiento y comparación de varios modelos de clasificación
    - Selección del modelo final según **F1 macro**
    - Integración en una aplicación interactiva
    """)

elif page == "2) Dataset":
    st.header("Dataset")
    df = pd.read_csv("data/car.data.csv", header=None)
    df.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

    st.write(f"**Filas:** {df.shape[0]} | **Columnas:** {df.shape[1]}")
    st.write("**Features:** buying, maint, doors, persons, lug_boot, safety")
    st.write("**Target:** class (unacc / acc / good / vgood)")

    # Distribución del target
    st.subheader("Distribución del target")
    order = ['unacc', 'acc', 'good', 'vgood']

    df_plot = df.copy()
    df_plot['class'] = pd.Categorical(df_plot['class'], categories=order, ordered=True)

    class_counts = df_plot['class'].value_counts(sort=False)
    st.bar_chart(class_counts)
    st.caption(
        "La clase 'unacc' es mayoritaria, ya que refleja las reglas del dataset original."
    )

    with st.expander("Ver primeras filas"):
        st.dataframe(df.head(10))

elif page == "3) Modelos":
    st.header("Modelos")
    st.write("Se probaron distintos modelos de clasificación para predecir la aceptabilidad del coche.")
    st.write("La métrica principal utilizada fue **F1 macro**, adecuada para problemas multiclase con desbalance.")

    results = pd.DataFrame({
        "Modelo": [
            "KNN (baseline)",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "Random Forest (Random Search)"
        ],
        "Accuracy": [
            0.90,
            0.97,
            0.97,
            0.97,
            0.96
        ],
        "F1 macro": [
            0.78,
            0.94,
            0.90,
            0.92,
            0.94
        ]
    })

    st.subheader("Comparación de modelos")
    st.dataframe(results)

    st.markdown("""
    **Modelo final elegido:**  
    👉 **Random Forest optimizado con RandomizedSearchCV**

    **Motivos:**
    - Buen equilibrio entre accuracy y F1 macro  
    - Modelo robusto y estable  
    - Mejor generalización que el modelo sin tuning
    """)

elif page == "4) Demo":
    st.header("Demo: evaluar un coche")
    st.title("Car Evaluation Model")
    st.write("Predicción de la aceptabilidad de un coche según sus características.")

    # Selectores de entrada
    buying_ui = st.selectbox("Precio de compra", list(buying_map.keys()))
    maint_ui = st.selectbox("Coste de mantenimiento", list(maint_map.keys()))
    doors_ui = st.selectbox("Número de puertas", list(doors_map.keys()))
    persons_ui = st.selectbox("Capacidad de personas", list(persons_map.keys()))
    lug_boot_ui = st.selectbox("Tamaño del maletero", list(lug_boot_map.keys()))
    safety_ui = st.selectbox("Nivel de seguridad", list(safety_map.keys()))

    # Convertimos a valores que entiende el modelo
    buying = buying_map[buying_ui]
    maint = maint_map[maint_ui]
    doors = doors_map[doors_ui]
    persons = persons_map[persons_ui]
    lug_boot = lug_boot_map[lug_boot_ui]
    safety = safety_map[safety_ui]

    # Botón de predicción
    if st.button("Evaluar coche"):
        new_car = pd.DataFrame([{
            "buying": buying,
            "maint": maint,
            "doors": doors,
            "persons": persons,
            "lug_boot": lug_boot,
            "safety": safety
        }])
        #Animación
        progress = st.progress(0)
        placeholder = st.empty()

        placeholder.image(
            "https://i.pinimg.com/originals/64/db/32/64db32b962775f18573ede759f5ec6f2.gif",
            width=250
        )

        for i in range(100):
            time.sleep(0.015)
            progress.progress(i + 1)

        placeholder.empty()
        progress.empty()
        # Predicción
        pred = model.predict(new_car)[0]
        pred_es = CLASS_ES.get(pred, pred)

        # Añadimos colores a los resultados 
        
        
        if pred == "vgood":
            st.success(f"**{pred_es}**")
        elif pred == "good":
            st.info(f"**{pred_es}**")
        elif pred == "acc":
            st.warning(f"**{pred_es}**")
        else:
            st.error(f"**{pred_es}**")
        
        # Sugerencias de mejora si el resultado es "No aceptable (unacc)"
        if pred == "unacc":
            st.markdown("### Sugerencias para mejorar la aceptabilidad")

            suggestions = improvement_suggestions(
                buying=buying,
                maint=maint,
                persons=persons,
                safety=safety
            )

            for s in suggestions:
                st.write(s)
        
        # Ver datos enviados al modelo
        with st.expander("Ver datos enviados al modelo"):
            st.dataframe(new_car)

elif page == "5) Conclusiones":
    st.header("Conclusiones")
    st.write("""
    - Se ha desarrollado un modelo de **clasificación multiclase** para predecir la aceptabilidad de un coche.
    - Se probaron varios modelos (KNN, Decision Tree, Random Forest y Gradient Boosting).
    - La métrica principal fue **F1 macro**, adecuada para el desbalance entre clases.
    - El **modelo final** es un **Random Forest optimizado con RandomizedSearchCV**, por su buen equilibrio entre rendimiento y robustez.
    - No se aplicaron técnicas de balanceo porque el desbalance refleja reglas reales del dataset original.
    """)

    st.subheader("Próximos pasos")
    st.write("""
    - Añadir más datos o variables reales.
    - Reentrenar el modelo con criterios más actuales.
    - Mejorar la experiencia de usuario en la app.
    """)

