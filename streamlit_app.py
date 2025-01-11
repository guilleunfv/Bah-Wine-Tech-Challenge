import os
import openai
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Configuração da API Key
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("A chave API da OpenAI não foi encontrada. Configure-a no `secrets` ou como variável de ambiente.")
    st.stop()

# ----------------------------------------------------------
# Título do App
st.title("🍷 Bah Chat: Análise de Dados e Chatbot de Vinhos")

st.write("""
Bem-vindo ao Bah Chat!  
Este aplicativo combina **análise de dados** de exportação de vinhos brasileiros e um **chatbot** baseado no modelo **GPT-3.5-turbo**.  
Use os gráficos e dados interativos para explorar informações e faça perguntas no chatbot!
""")

# ----------------------------------------------------------
# PARTE 1: Análise de Dados de Exportação de Vinhos
# ----------------------------------------------------------

@st.cache_data
def carregar_dados():
    try:
        url = "https://drive.google.com/uc?id=1-mrtTLjOPh_XVk1mkDH00SUJxWkuOu5o"
        dados = pd.read_csv(url, delimiter=';', encoding='utf-8', quotechar='"')
        dados.columns = dados.columns.str.strip()
        return dados
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()

df = carregar_dados()

if not df.empty:
    st.subheader("📊 Dados Gerais de Exportação de Vinhos")
    st.dataframe(df)

    # Filtros Interativos
    st.sidebar.header("Filtros")
    anos = st.sidebar.multiselect(
        "Selecione os Anos", 
        df['Año'].unique(), 
        default=df['Año'].unique()
    )
    paises = st.sidebar.multiselect(
        "Selecione os Países", 
        df['País'].unique(), 
        default=df['País'].unique()[:10]
    )

    df_filtrado = df[df['Año'].isin(anos) & df['País'].isin(paises)]

    st.subheader("📈 Dados Filtrados")
    st.dataframe(df_filtrado)

    # Gráfico 1: Tendência de Exportação
    st.subheader("📉 Tendência de Exportação (US$ FOB por Ano)")
    if not df_filtrado.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        df_group_ano = df_filtrado.groupby('Año')['Valor US$ FOB'].sum().sort_index()
        ax.plot(df_group_ano.index, df_group_ano.values, color='#8B0000', marker='o')
        ax.set_title("Tendência de Exportação de Vinhos", fontsize=16)
        ax.set_xlabel("Ano", fontsize=14)
        ax.set_ylabel("Valor Total (US$ FOB)", fontsize=14)
        st.pyplot(fig)
    else:
        st.warning("Nenhum dado disponível para os filtros selecionados.")

    # Gráfico 2: Exportação por País
    st.subheader("🌎 Exportação por País")
    if not df_filtrado.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        df_group_pais = df_filtrado.groupby('País')['Valor US$ FOB'].sum().sort_values(ascending=False).head(10)
        df_group_pais.plot(kind='bar', color='#A52A2A', ax=ax)
        ax.set_title("Top 10 Países de Destino", fontsize=16)
        ax.set_xlabel("País", fontsize=14)
        ax.set_ylabel("Valor Total (US$ FOB)", fontsize=14)
        st.pyplot(fig)
    else:
        st.warning("Nenhum dado disponível para os filtros selecionados.")

# ----------------------------------------------------------
# PARTE 2: Chatbot GPT-3.5 Conectado aos Dados
# ----------------------------------------------------------

st.subheader("🤖 Bah Chat: Chatbot GPT-3.5")

# Função para gerar resposta com base no DataFrame filtrado
def gerar_resposta(messages, df_filtrado):
    # Tenta gerar resposta com base nos dados
    ultima_pergunta = messages[-1]["content"]
    try:
        if "quem vende mais" in ultima_pergunta.lower():
            pais_top = df_filtrado.groupby("País")["Valor US$ FOB"].sum().idxmax()
            valor_top = df_filtrado.groupby("País")["Valor US$ FOB"].sum().max()
            return f"O país que mais compra vinhos do Brasil é **{pais_top}**, com um total de **US$ {valor_top:,.2f}**."
        elif "ano com mais exportações" in ultima_pergunta.lower():
            ano_top = df_filtrado.groupby("Año")["Valor US$ FOB"].sum().idxmax()
            valor_top = df_filtrado.groupby("Año")["Valor US$ FOB"].sum().max()
            return f"O ano com mais exportações foi **{ano_top}**, com um total de **US$ {valor_top:,.2f}**."
        else:
            # Se não encontra resposta, usa o modelo GPT
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Erro ao processar a pergunta: {e}"

# Histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "Você é um assistente útil para análise de exportação de vinhos."}]

# Exibindo mensagens anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada do usuário
prompt = st.chat_input("Faça uma pergunta sobre os dados ou exportações de vinhos...")
if prompt:
    # Adiciona a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gera a resposta
    resposta = gerar_resposta(st.session_state.messages, df_filtrado)

    # Adiciona a resposta ao histórico e exibe
    st.session_state.messages.append({"role": "assistant", "content": resposta})
    with st.chat_message("assistant"):
        st.markdown(resposta)

