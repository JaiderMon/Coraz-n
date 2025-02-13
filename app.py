import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo y el escalador guardados
modelo_knn = joblib.load('modelo_knn.bin')
escalador = joblib.load('escalador.bin')

# Título y descripción de la aplicación
st.title('Asistente IA para cardiólogos')
st.write("""
    **Introducción:**
    Esta aplicación tiene como objetivo predecir si un paciente sufre o no de problemas cardíacos, utilizando un modelo de Inteligencia Artificial entrenado con el algoritmo KNN.
    El modelo toma en cuenta factores como la edad y el colesterol del paciente para realizar la predicción. 

    La predicción es presentada en un formato fácil de entender, mostrando si el paciente tiene o no un problema cardíaco.
""")

# Crear tabs
tab_data, tab_pred = st.tabs(["Capturar Datos", "Predicción"])

# Tab para capturar los datos
with tab_data:
    st.subheader("Ingrese los datos del paciente")
    edad = st.slider("Edad", 18, 80, 30)
    colesterol = st.slider("Colesterol", 50, 600, 200)
    st.write("Ingrese los datos para realizar la predicción.")

    # Guardar los datos para la predicción posterior
    if st.button('Guardar datos'):
        st.session_state.edad = edad
        st.session_state.colesterol = colesterol
        st.success("Datos guardados correctamente. Ahora puede ir a la pestaña de 'Predicción'.")

# Tab para realizar la predicción
with tab_pred:
    if hasattr(st.session_state, 'edad') and hasattr(st.session_state, 'colesterol'):
        st.subheader("Predicción")
        # Recuperar los datos guardados
        edad = st.session_state.edad
        colesterol = st.session_state.colesterol
        
        # Mostrar los datos
        st.write(f"**Edad:** {edad} años")
        st.write(f"**Colesterol:** {colesterol} mg/dL")
        
        if st.button('Predecir'):
            # Crear un dataframe con los datos de entrada
            datos_entrada = pd.DataFrame({
                'edad': [edad],
                'colesterol': [colesterol]
            })
            
            # Normalizar los datos con el escalador
            datos_normalizados = escalador.transform(datos_entrada)
            
            # Realizar la predicción usando el modelo KNN
            prediccion = modelo_knn.predict(datos_normalizados)
            
            # Mostrar la predicción
            if prediccion == 1:
                st.write("**Predicción:** El paciente tiene problema cardíaco.")
                st.image("https://www.clinicadeloccidente.com/wp-content/uploads/sintomas-cardio-linkedin.jpg")
            else:
                st.write("**Predicción:** El paciente no tiene problema cardíaco.")
                st.image("https://colombianadetrasplantes.com/web/wp-content/uploads/2023/05/01-PORTADA.-01-scaled.jpg")
    else:
        st.write("Por favor, ingrese los datos en la pestaña de 'Capturar Datos' antes de predecir.")

# Pie de página
st.write("""
    ---  
    **Realizado por Jaider Monsalve**
""")
