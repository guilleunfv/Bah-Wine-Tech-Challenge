import openai
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Configura a chave da OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ----------------------------------------------------------
# PARTE 1: Análise de Dados de Exportação de Vinhos
# ----------------------------------------------------------

@st.cache_data
def carregar_dados():
    url = "https://drive.google.com/uc?id=1-mrtTLjOPh_XVk1mkDH00SUJxWkuOu5o"
    dados = pd.read_csv(url, delimiter=';', encoding='utf-8', quotechar='"')
    dados.columns = dados.columns.str.strip()
    return dados

# Carregar os dados
df = carregar_dados()

# Layout do app
st.title("🍷 Análise de Exportação de Vinhos + Bah Chat")
st.markdown(
    """
    Bem-vindo ao **Bah Chat**, o chatbot e ferramenta de análise de dados de exportação de vinhos brasileiros.
    Use os gráficos interativos e interaja com o chatbot para explorar os dados de exportação de vinhos.
    """
)

# Dados gerais
st.subheader("📊 Dados Gerais de Exportação")
st.dataframe(df)

# Filtros interativos
st.sidebar.header("🔍 Filtros Interativos")
anos = st.sidebar.multiselect(
    "Selecione os Anos", 
    df['Año'].unique(), 
    default=df['Año'].unique()
)
df_filtrado = df[df['Año'].isin(anos)]

paises = st.sidebar.multiselect(
    "Selecione os Países", 
    df['País'].unique(), 
    default=df['País'].unique()[:10]
)
df_filtrado = df_filtrado[df_filtrado['País'].isin(paises)]

st.subheader("🔎 Dados Filtrados")
st.dataframe(df_filtrado)

# Gráfico 1: Tendência de Exportação
st.subheader("📈 Tendência de Exportação (US$ FOB por Ano)")
fig, ax = plt.subplots(figsize=(10, 4))
df_group_ano = df_filtrado.groupby('Año')['Valor US$ FOB'].sum().sort_index()
ax.plot(df_group_ano.index, df_group_ano.values, color='#8B0000', marker='o')
ax.set_title("Tendência de Exportação de Vinhos", fontsize=16, color='#8B0000')
ax.set_xlabel("Ano", fontsize=14)
ax.set_ylabel("Valor Total (US$ FOB)", fontsize=14)
st.pyplot(fig)

# Gráfico 2: Exportação por País
st.subheader("🌍 Exportação por País")
fig, ax = plt.subplots(figsize=(10, 4))
df_group_pais = df_filtrado.groupby('País')['Valor US$ FOB'].sum().sort_values(ascending=False).head(10)
df_group_pais.plot(kind='bar', color='#A52A2A', ax=ax)
ax.set_title("Top 10 Países de Destino", fontsize=16, color='#8B0000')
ax.set_xlabel("País", fontsize=14)
ax.set_ylabel("Valor Total (US$ FOB)", fontsize=14)
st.pyplot(fig)

# ----------------------------------------------------------
# PARTE 2: Bah Chat
# ----------------------------------------------------------

st.subheader("🤖 Bah Chat")

# Inicializa o histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada de texto do usuário
prompt = st.chat_input("Digite sua pergunta ou solicitação...")
if prompt:
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Faz a chamada para a API do ChatGPT
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages
        )
        # Obtém a resposta
        answer = response["choices"][0]["message"]["content"]

        # Exibe a resposta no chat
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Adiciona a resposta ao histórico
        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Erro ao conectar com o Bah Chat: {e}")

# Fim do app
st.markdown("---")
st.markdown("💡 **Dica:** Use o Bah Chat para obter insights rápidos sobre os dados.")
