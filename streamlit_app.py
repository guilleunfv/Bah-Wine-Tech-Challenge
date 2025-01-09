import os
import openai
import pandas as pd
import plotly.express as px
import streamlit as st

# Configura tu clave de OpenAI desde la variable de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("La clave de la API de OpenAI no está configurada. Por favor, configúrala como una variable de entorno llamada 'OPENAI_API_KEY'.")
    st.stop()

# ----------------------------------------------------------
# PARTE 1: Análisis de Datos de Exportación de Vinhos
# ----------------------------------------------------------

@st.cache_data
def cargar_datos():
    # URL del dataset
    url = "https://drive.google.com/uc?id=1-mrtTLjOPh_XVk1mkDH00SUJxWkuOu5o"
    datos = pd.read_csv(url, delimiter=';', encoding='utf-8', quotechar='"')
    datos.columns = datos.columns.str.strip()
    return datos

# Cargar datos
df = cargar_datos()

# Título del Proyecto
st.title("🍷 Análisis de Datos de Exportación de Vinhos + Chatbot GPT-3.5")
st.markdown("Este proyecto combina **análisis de datos** de exportación de vinhos brasileños con un **chatbot** GPT-3.5 para consultas avanzadas.")

# Mostrar datos generales
st.subheader("Vista General de los Datos")
st.dataframe(df)

# Filtros interactivos
st.sidebar.header("Filtros Interactivos")
anos = st.sidebar.multiselect("Selecciona Años", df['Año'].unique(), default=df['Año'].unique())
paises = st.sidebar.multiselect("Selecciona Países", df['País'].unique(), default=df['País'].unique()[:10])

# Aplicar filtros
df_filtrado = df[df['Año'].isin(anos) & df['País'].isin(paises)]

st.subheader("Datos Filtrados")
st.dataframe(df_filtrado)

# Gráfico 1: Tendencia de Exportación
st.subheader("Tendencia de Exportación (US$ FOB por Año)")
if not df_filtrado.empty:
    fig = px.line(
        df_filtrado.groupby('Año')['Valor US$ FOB'].sum().reset_index(),
        x='Año',
        y='Valor US$ FOB',
        title='Tendencia de Exportación de Vinhos',
        markers=True
    )
    st.plotly_chart(fig)
else:
    st.warning("No hay datos para mostrar en este gráfico.")

# Gráfico 2: Exportación por País
st.subheader("Exportación por País")
if not df_filtrado.empty:
    fig = px.bar(
        df_filtrado.groupby('País')['Valor US$ FOB'].sum().reset_index().sort_values(by='Valor US$ FOB', ascending=False).head(10),
        x='País',
        y='Valor US$ FOB',
        title='Top 10 Países de Destino'
    )
    st.plotly_chart(fig)
else:
    st.warning("No hay datos para mostrar en este gráfico.")

# ----------------------------------------------------------
# PARTE 2: Chatbot
# ----------------------------------------------------------

st.subheader("🤖 Chatbot GPT-3.5")

# Almacenar mensajes en session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Campo de entrada para usuario
prompt = st.chat_input("Escribe tu pregunta o solicitud...")
if prompt:
    # Añadir el mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
     response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages
    )
    # Extraer la respuesta del chatbot
    answer = response["choices"][0]["message"]["content"]

    # Mostrar la respuesta del asistente
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Guardar la respuesta en el historial
    st.session_state.messages.append({"role": "assistant", "content": answer})

except Exception as e:
    st.error(f"Error al conectar con OpenAI: {e}")

# ----------------------------------------------------------
# FIM
# ----------------------------------------------------------
