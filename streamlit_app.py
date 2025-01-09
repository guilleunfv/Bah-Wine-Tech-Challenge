from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import openai

# Configura tu clave de OpenAI
openai.api_key = "TU_API_KEY"

# Cargar el dataset
url = "https://drive.google.com/uc?id=1-mrtTLjOPh_XVk1mkDH00SUJxWkuOu5o"
df = pd.read_csv(url, delimiter=';', encoding='utf-8')

# Inicializar Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Análisis de Datos de Exportación de Vinhos + Chatbot", style={"textAlign": "center"}),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': year, 'value': year} for year in df['Año'].unique()],
        multi=True,
        placeholder="Seleccione años"
    ),
    dcc.Graph(id='line-chart'),
    html.H2("Chatbot GPT-3.5"),
    dcc.Textarea(id='user-input', placeholder='Escribe tu pregunta...', style={'width': '100%', 'height': 100}),
    html.Button("Enviar", id="submit-btn"),
    html.Div(id='chat-output')
])

@app.callback(
    Output('line-chart', 'figure'),
    Input('year-dropdown', 'value')
)
def update_graph(selected_years):
    if not selected_years:
        filtered_df = df
    else:
        filtered_df = df[df['Año'].isin(selected_years)]
    grouped = filtered_df.groupby('Año')['Valor US$ FOB'].sum().reset_index()
    fig = px.line(grouped, x='Año', y='Valor US$ FOB', title="Exportaciones por Año")
    return fig

@app.callback(
    Output('chat-output', 'children'),
    Input('submit-btn', 'n_clicks'),
    Input('user-input', 'value'),
    prevent_initial_call=True
)
def get_chat_response(n_clicks, user_input):
    if not user_input:
        return "Por favor, escribe una pregunta."
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=user_input,
        max_tokens=100
    )
    return response['choices'][0]['text']

if __name__ == '__main__':
    app.run_server(debug=True)
