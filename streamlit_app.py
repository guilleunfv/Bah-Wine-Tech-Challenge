import os
import openai
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da API Key
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# ----------------------------------------------------------
# Função para carregar e limpar os dados
@st.cache_data
def carregar_dados():
    url1 = "https://drive.google.com/uc?id=1-mrtTLjOPh_XVk1mkDH00SUJxWkuOu5o"
    url2 = "https://docs.google.com/spreadsheets/d/1MOajBCCgeG2D48OWjh8DO-mJGjYw4L1O/export?format=csv"
    dados1 = pd.read_csv(url1, delimiter=';', encoding='utf-8', quotechar='"')
    dados2 = pd.read_csv(url2, delimiter=',', encoding='utf-8')

    dados1.columns = dados1.columns.str.strip()
    dados1['País'] = dados1['País'].str.strip()
    dados1['Año'] = pd.to_numeric(dados1['Año'], errors='coerce')
    dados1 = dados1[(dados1['Año'] >= 2009) & (dados1['Año'] <= 2024)]  # Filtrar últimos 15 anos

    dados2.columns = dados2.columns.str.strip()

    return dados1, dados2

# Carregar dados
df1, df2 = carregar_dados()

# ----------------------------------------------------------
# Adicionar link para download dos dados filtrados
def gerar_download_link(dataframe, filename):
    csv = dataframe.to_csv(index=False, sep=';', encoding='utf-8')
    st.sidebar.download_button(
        label="📥 Baixar Dados Filtrados",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )

# ----------------------------------------------------------
# Título do App
st.title("🍷 Tech Challenge - Análise Profunda das Exportações de Vinhos")

st.markdown("""
Bem-vindo ao **Tech Challenge**!  
Este painel apresenta uma análise abrangente e detalhada sobre as exportações de vinhos brasileiros, ideal para decisões estratégicas.
""")

# ----------------------------------------------------------
# Filtros Globais
st.sidebar.header("Filtros Globais")
anos_selecionados = st.sidebar.multiselect(
    "Selecione os Anos",
    df1['Año'].unique(),
    default=df1['Año'].unique()[-5:]
)
paises_selecionados = st.sidebar.multiselect(
    "Selecione os Países",
    df1['País'].unique(),
    default=df1['País'].unique()[:10]
)

# Filtrar os dados
df_filtrado = df1[df1['Año'].isin(anos_selecionados) & df1['País'].isin(paises_selecionados)]

# Adicionar link de download para os dados filtrados
gerar_download_link(df_filtrado, "dados_filtrados_vinhos.csv")

# ----------------------------------------------------------
# Gráfico 1: Países com Maior Volume Exportado
st.subheader("🌍 Países com Maior Volume Exportado (Litros)")
df_volume_pais = df_filtrado.groupby('País')['Peso Neto'].sum().reset_index().sort_values(by='Peso Neto', ascending=False)
fig, ax = plt.subplots()
sns.barplot(data=df_volume_pais.head(10), x='Peso Neto', y='País', ax=ax, palette='viridis', hue=None, dodge=False)
ax.set_title("Top 10 Países por Volume Exportado (de Brasil)")
ax.set_xlabel("Volume Exportado (KG)")
ax.set_ylabel("País de Destino")
st.pyplot(fig)

# ----------------------------------------------------------
# Gráfico 2: Países com Maior Receita de Exportação
st.subheader("💰 Países com Maior Receita de Exportação (US$)")
df_receita_pais = df_filtrado.groupby('País')['Valor US$ FOB'].sum().reset_index().sort_values(by='Valor US$ FOB', ascending=False)
fig, ax = plt.subplots()
sns.barplot(data=df_receita_pais.head(10), x='Valor US$ FOB', y='País', ax=ax, palette='coolwarm', hue=None, dodge=False)
ax.set_title("Top 10 Países por Receita de Exportação (de Brasil)")
ax.set_xlabel("Receita Total (US$)")
ax.set_ylabel("País de Destino")
st.pyplot(fig)

# ----------------------------------------------------------
# Gráfico 3: Tendência Anual de Exportações
st.subheader("📈 Tendência Anual de Exportações (Volume e Receita)")
df_tendencia = df_filtrado.groupby('Año')[['Peso Neto', 'Valor US$ FOB']].sum().reset_index()
fig, ax = plt.subplots()
ax.plot(df_tendencia['Año'], df_tendencia['Peso Neto'], marker='o', label='Volume (KG)', color='#2E8B57')
ax.set_ylabel("Volume Exportado (KG)")
ax2 = ax.twinx()
ax2.plot(df_tendencia['Año'], df_tendencia['Valor US$ FOB'], marker='s', label='Receita (US$)', color='#8B0000')
ax2.set_ylabel("Receita Total (US$)")
ax.set_title("Tendência Anual de Exportações de Vinhos Brasileiros")
ax.set_xlabel("Ano")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
st.pyplot(fig)

# ----------------------------------------------------------
# Gráfico 4: Categorias de Vinhos mais Exportadas
st.subheader("🍇 Categorias de Vinhos mais Exportadas")
if 'Descripción NCM' in df_filtrado.columns:
    df_categoria = df_filtrado.groupby('Descripción NCM')['Valor US$ FOB'].sum().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=df_categoria.sort_values(by='Valor US$ FOB', ascending=False), x='Valor US$ FOB', y='Descripción NCM', ax=ax, palette='magma', hue=None, dodge=False)
    ax.set_title("Receita por Categoria de Vinho (Exportação Brasileira)")
    ax.set_xlabel("Receita Total (US$)")
    ax.set_ylabel("Categoria de Vinho")
    st.pyplot(fig)
else:
    st.warning("Coluna 'Descripción NCM' não encontrada no dataset.")

# ----------------------------------------------------------
# Gráfico 5: Distribuição de Exportações por Ano e País
st.subheader("📊 Distribuição de Exportações por Ano e País")
df_ano_pais = df_filtrado.groupby(['Año', 'País'])[['Peso Neto', 'Valor US$ FOB']].sum().reset_index()
if not df_ano_pais.empty:
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap_data = df_ano_pais.pivot(index="País", columns="Año", values="Peso Neto")
    sns.heatmap(
        heatmap_data, 
        cmap="YlGnBu", annot=True, fmt=".0f", linewidths=0.5, ax=ax
    )
    ax.set_title("Distribuição de Volume Exportado por Ano e País (de Brasil)")
    st.pyplot(fig)
else:
    st.warning("Nenhum dado disponível para exibir a distribuição por Ano e País.")

# ----------------------------------------------------------
# Tabela Detalhada
st.subheader("📋 Tabela Detalhada")
st.dataframe(df_filtrado[['Año', 'País', 'Descripción NCM', 'Valor US$ FOB', 'Peso Neto']])

# ----------------------------------------------------------
# Chatbot com Base nos Dados
st.subheader("🤖 Chatbot Inteligente")
if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input("Faça uma pergunta sobre as exportações de vinho...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um especialista em exportação de vinhos brasileiros. Responda com base nos dados fornecidos."},
                *st.session_state.messages
            ],
            max_tokens=500,
        )
        resposta = response["choices"][0]["message"]["content"]
        st.chat_message("assistant").markdown(resposta)
        st.session_state.messages.append({"role": "assistant", "content": resposta})
    except Exception as e:
        st.error(f"Erro ao conectar com o ChatGPT: {e}")

# ----------------------------------------------------------
# Conclusão
st.markdown("""
### Conclusão
Este painel apresenta uma análise aprofundada das exportações de vinhos brasileiros, com gráficos detalhados e interativos para apoiar decisões estratégicas.
""")