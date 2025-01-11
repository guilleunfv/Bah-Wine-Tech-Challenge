import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------
# PARTE 1: Carregar os Dados
# ----------------------------------------------------------

@st.cache_data
def carregar_dados():
    # URL dos dados
    url = "https://drive.google.com/uc?id=1-mrtTLjOPh_XVk1mkDH00SUJxWkuOu5o"
    dados = pd.read_csv(url, delimiter=';', encoding='utf-8', quotechar='"')
    dados.columns = dados.columns.str.strip()  # Limpa espaços extras nas colunas
    return dados

# Carregar os dados
df = carregar_dados()

# ----------------------------------------------------------
# PARTE 2: Gráficos Interativos
# ----------------------------------------------------------

def criar_graficos(df):
    st.subheader("📊 Análises Gráficas")
    
    # Gráfico 1: Exportação Total por Ano
    st.write("### Exportação Total por Ano")
    df_por_ano = df.groupby('Año')['Valor US$ FOB'].sum().sort_index()
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df_por_ano.index, df_por_ano.values, marker='o', color='#8B0000')
    ax1.set_title("Exportação Total por Ano", fontsize=16)
    ax1.set_xlabel("Ano", fontsize=14)
    ax1.set_ylabel("Valor Total (US$ FOB)", fontsize=14)
    st.pyplot(fig1)

    # Gráfico 2: Exportação por País
    st.write("### Exportação por País")
    df_por_pais = df.groupby('País')['Valor US$ FOB'].sum().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    df_por_pais.plot(kind='bar', color='#4682B4', ax=ax2)
    ax2.set_title("Exportação por País", fontsize=16)
    ax2.set_xlabel("País", fontsize=14)
    ax2.set_ylabel("Valor Total (US$ FOB)", fontsize=14)
    st.pyplot(fig2)

    # Gráfico 3: Preço Médio por Ano
    st.write("### Preço Médio por Ano")
    df['Preço Médio'] = df['Valor US$ FOB'] / df['Volume']
    df_preco_medio = df.groupby('Año')['Preço Médio'].mean()
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.bar(df_preco_medio.index, df_preco_medio.values, color='#FFD700')
    ax3.set_title("Preço Médio por Ano", fontsize=16)
    ax3.set_xlabel("Ano", fontsize=14)
    ax3.set_ylabel("Preço Médio (US$ por Litro)", fontsize=14)
    st.pyplot(fig3)

# ----------------------------------------------------------
# PARTE 3: Chatbot para Perguntas Baseadas no DataFrame
# ----------------------------------------------------------

def responder_pergunta(pergunta, df):
    """
    Responde à pergunta do usuário com base no DataFrame fornecido.
    """
    pergunta = pergunta.lower()
    if "quem vende mais" in pergunta:
        pais_top = df.groupby("País")["Valor US$ FOB"].sum().idxmax()
        valor_top = df.groupby("País")["Valor US$ FOB"].sum().max()
        return f"O país que mais compra vinhos do Brasil é **{pais_top}**, com um total de **US$ {valor_top:,.2f}**."
    elif "ano com mais exportações" in pergunta:
        ano_top = df.groupby("Año")["Valor US$ FOB"].sum().idxmax()
        valor_top = df.groupby("Año")["Valor US$ FOB"].sum().max()
        return f"O ano com mais exportações foi **{ano_top}**, com um total de **US$ {valor_top:,.2f}**."
    elif "preço médio" in pergunta:
        preco_medio = df["Valor US$ FOB"].sum() / df["Volume"].sum()
        return f"O preço médio por litro exportado é **US$ {preco_medio:.2f}**."
    elif "exportações totais" in pergunta:
        total_exportado = df["Valor US$ FOB"].sum()
        return f"As exportações totais de vinhos somam **US$ {total_exportado:,.2f}**."
    elif "quais países" in pergunta:
        paises = df["País"].unique()
        return f"Os vinhos brasileiros foram exportados para os seguintes países: {', '.join(paises)}."
    else:
        return "Desculpe, não entendi sua pergunta. Tente reformular ou pergunte algo relacionado aos dados de exportação."

# ----------------------------------------------------------
# PARTE 4: Machine Learning (Predição de Exportações)
# ----------------------------------------------------------

def previsao_exportacao(df):
    st.subheader("🤖 Previsão de Exportações")
    st.write("Este modelo usa regressão linear para prever o valor exportado com base no volume.")

    # Prepara os dados
    df = df.dropna(subset=["Valor US$ FOB", "Volume"])
    X = df[["Volume"]]
    y = df["Valor US$ FOB"]

    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treina o modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Previsão
    y_pred = modelo.predict(X_test)
    erro = mean_squared_error(y_test, y_pred)

    # Entrada do usuário para prever
    volume_input = st.number_input("Digite o volume para prever o valor exportado (em litros):", min_value=0.0)
    if volume_input:
        previsao = modelo.predict([[volume_input]])
        st.write(f"Com um volume de **{volume_input} litros**, o valor exportado previsto é **US$ {previsao[0]:,.2f}**.")
    st.write(f"Erro do modelo (MSE): {erro:.2f}")

# ----------------------------------------------------------
# PARTE 5: Interface do Usuário
# ----------------------------------------------------------

st.title("🍷 Análise de Exportação de Vinhos")
st.write("Explore os dados, visualize gráficos e faça perguntas ao chatbot sobre exportações de vinhos brasileiros.")

# Exibe os dados
with st.expander("📄 Ver Dados Brutos"):
    st.dataframe(df)

# Gráficos
criar_graficos(df)

# Chatbot
st.subheader("🤖 Chatbot")
pergunta = st.text_input("Digite sua pergunta:")
if pergunta:
    resposta = responder_pergunta(pergunta, df)
    st.write(f"**Resposta:** {resposta}")

# Previsão
previsao_exportacao(df)


