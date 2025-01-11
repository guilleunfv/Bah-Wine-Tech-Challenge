import os
import openai
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Configuração da API Key
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# ----------------------------------------------------------
# Título do App
st.title("🍷 Bah Chat: Dados e Decisões Inteligentes")

st.write("""
Bem-vindo ao Bah Chat!  
Este aplicativo combina **análise de dados** de exportação de vinhos brasileiros e um **chatbot** baseado no modelo **GPT-3.5-turbo**.  
Explore dados interativos e obtenha respostas para tomar decisões estratégicas de exportação!
""")

# ----------------------------------------------------------
# Função para carregar os dados
@st.cache_data
def carregar_dados():
    url = "https://drive.google.com/uc?id=1-mrtTLjOPh_XVk1mkDH00SUJxWkuOu5o"
    dados = pd.read_csv(url, delimiter=';', encoding='utf-8', quotechar='"')
    dados.columns = dados.columns.str.strip()
    return dados

# Carga inicial dos dados
df = carregar_dados()

# ----------------------------------------------------------
# PARTE 1: Análise de Dados
# ----------------------------------------------------------

st.subheader("📊 Análise de Dados de Exportação de Vinhos")

# Gráfico 1: Exportação total por ano
st.write("### 📈 Tendência de Exportação por Ano")
anos_grafico1 = st.multiselect(
    "Selecione os Anos para o Gráfico de Tendência", 
    df['Año'].unique(), 
    default=df['Año'].unique()
)
df_filtrado1 = df[df['Año'].isin(anos_grafico1)]
fig1, ax1 = plt.subplots(figsize=(10, 4))
df_group_ano = df_filtrado1.groupby('Año')['Valor US$ FOB'].sum().sort_index()
ax1.plot(df_group_ano.index, df_group_ano.values, color='#8B0000', marker='o')
ax1.set_title("Exportação Total por Ano", fontsize=16)
ax1.set_xlabel("Ano", fontsize=14)
ax1.set_ylabel("Valor Total (US$ FOB)", fontsize=14)
st.pyplot(fig1)

# Gráfico 2: Exportação por país
st.write("### 🌎 Exportação por País")
paises_grafico2 = st.multiselect(
    "Selecione os Países para o Gráfico de Exportação por País", 
    df['País'].unique(), 
    default=df['País'].unique()[:10]
)
df_filtrado2 = df[df['País'].isin(paises_grafico2)]
fig2, ax2 = plt.subplots(figsize=(10, 4))
df_group_pais = df_filtrado2.groupby('País')['Valor US$ FOB'].sum().sort_values(ascending=False).head(10)
df_group_pais.plot(kind='bar', color='#A52A2A', ax=ax2)
ax2.set_title("Top Países por Valor de Exportação", fontsize=16)
ax2.set_xlabel("País", fontsize=14)
ax2.set_ylabel("Valor Total (US$ FOB)", fontsize=14)
st.pyplot(fig2)

# Gráfico 3: Média de exportação por ano
st.write("### 📊 Média de Exportação Anual")
fig3, ax3 = plt.subplots(figsize=(10, 4))
df_group_media = df.groupby('Año')['Valor US$ FOB'].mean()
ax3.bar(df_group_media.index, df_group_media.values, color='#4682B4')
ax3.set_title("Média de Exportação Anual (US$ FOB)", fontsize=16)
ax3.set_xlabel("Ano", fontsize=14)
ax3.set_ylabel("Média (US$ FOB)", fontsize=14)
st.pyplot(fig3)

# ----------------------------------------------------------
# PARTE 2: Chatbot
# ----------------------------------------------------------

st.subheader("🤖 Bah Chat: Inteligência com GPT-3.5")

# Histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibindo mensagens anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada do usuário
prompt = st.chat_input("Pergunte sobre os dados de exportação...")
if prompt:
    # Adiciona a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Filtrar os dados com base na consulta
    try:
        # Analisar consulta e responder com base nos dados
        resposta = ""
        if "total exportado" in prompt.lower():
            total_exportado = df['Valor US$ FOB'].sum()
            resposta = f"O valor total exportado foi de **US$ {total_exportado:,.2f}**."
        elif "ano" in prompt.lower():
            df_por_ano = df.groupby('Año')['Valor US$ FOB'].sum()
            resposta = "Aqui está o total exportado por ano:\n" + df_por_ano.to_string()
        else:
            # Chamando a API para outras respostas
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.messages,
                temperature=0.7,
                max_tokens=500
            )
            resposta = response["choices"][0]["message"]["content"]

        # Exibindo a resposta
        with st.chat_message("assistant"):
            st.markdown(resposta)

        # Adiciona a resposta ao histórico
        st.session_state.messages.append({"role": "assistant", "content": resposta})

    except Exception as e:
        st.error(f"Erro ao conectar com o Bah Chat: {e}")

# ----------------------------------------------------------
# FIM
# ----------------------------------------------------------
