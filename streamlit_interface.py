from ast import Pass
from ctypes import alignment
import streamlit as st
import pandas as pd
import requests
import base64
import time
import uuid
import re

# API base URL
API_URL = "http://localhost:8888/"

# REMOVING STREAMLIT STYLE
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stApp [data-testid="stToolbar"]{display:none;}
            </style>
            """

# Display logo at the top of the page
st.image("LogoSTI.png", use_container_width=False)


st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Função para verificar o status da API
def check_api_status():
    try:
        response = requests.get(f"{API_URL}/health/liveness")
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        return False

# Verificar se a API está disponível
if not check_api_status():
    st.error("API fora do ar. Tentando reconectar em 30 segundos...")
    time.sleep(30)  # Espera de 1 minuto antes de tentar novamente
    st.rerun()  # Recarrega o aplicativo após a espera

username = "generic"

# Initialize session state with a unique session ID if not already set
if "selected_session_id" not in st.session_state:
    st.session_state.selected_session_id = str(uuid.uuid4())
    requests.post(f"{API_URL}/chat/{st.session_state.selected_session_id}/start_session?name={username}")



########### MAIN SCREEN START -----------------------------------------------------------------------

# st.header(f"{st.session_state.selected_session_id}", anchor="top",divider=True)

# Retrieve and display messages from the selected session
messages_response = requests.get(f"{API_URL}/chat/{st.session_state.selected_session_id}/messages?name={username}")
if messages_response.status_code == 200:
    messages = messages_response.json()['messages']
    for message in messages:
        if message['role'] == "user":
            st.write("\n")
            role = "🧑‍💻 Humano"
        else:
            role = "🤖 IA"

        st.write(f"**{role}**: {message['content']}")
else:
    pass


# Form to send a message
with st.form("send_message"):
    col1, col2 = st.columns([9,1])
    with col1:
        message_content = st.text_input("", label_visibility="collapsed", placeholder="🧑‍💻 Digite sua mensagem")
    with col2:
        send_message_button = st.form_submit_button("➤", help="Enviar mensagem")

# Lógica de envio da mensagem
if send_message_button:
    if message_content.strip():
        # Definir o endpoint baseado na escolha do radio
        endpoint = f"{API_URL}/chat/{st.session_state.selected_session_id}/send_message_vector_db"

        # Prepare the data
        message_data = {
            "message": message_content,  # Pass the message content
            "name": username             # Pass the username
        }

        # Set the headers to indicate that we are sending JSON
        headers = {
            "Content-Type": "application/json"
        }

        # Tentativas de envio (com retry)
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(endpoint, params=message_data)
            
            if response.status_code == 200:
                st.success("Mensagem enviada com sucesso!")
                st.rerun()  # Reiniciar a aplicação para atualizar o histórico do chat
                break  # Sair do loop após o envio bem-sucedido
            elif response.status_code == 400:
                st.error("Mensagem inválida ou vazia.")
                break  # Parar as tentativas em caso de erro no lado do cliente
            else:
                if attempt < max_retries - 1:
                    st.warning(f"Erro ao enviar mensagem! Tentando novamente... ({attempt + 1}/{max_retries})")
                else:
                    st.error("Erro ao enviar a mensagem! Falha após várias tentativas.")
    else:
        st.error("Por favor, insira uma mensagem válida.")



# Display modal if the state is set to True
if st.session_state.get("show_modal_modelos", False):
    with st.container():
        # Form to access AI models
        with st.form("peticao_session"):
            st.header("Consultar petição de processo")
            cnj_name = st.text_input("CNJ")

            # Validate CNJ input
            if cnj_name:
                # Define the regex pattern for CNJ
                cnj_regex = r"^\s*\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\s*$"
                if re.match(cnj_regex, cnj_name):
                    # Load the DataFrame
                    df = pd.read_csv("cnj_peticoes.csv")
                    # Filter the DataFrame based on the CNJ
                    filtered_df = df[df['cnj'] == cnj_name]

                    # Check if the filtered DataFrame is empty
                    if not filtered_df.empty:
                        st.success("CNJ encontrado!")

                        # Extract attributes and display them
                        for index, row in filtered_df.iterrows():
                            st.subheader(f"CNJ:{row['cnj']}")
                            st.write(f'ID Petição:\n> {row["peticao_idDocumentoField"]}')
                            st.write(f'Macrotema - Probabilidade:\n> {row["macrotema_predito"]} - {str(row["macrotema_probabilidade"]*100)[:4]}%' )
                            st.write(f'Assunto - Probabilidade:\n> {row["assunto_predito"]} - {str(row["assunto_probabilidade"]*100)[:4]}%' )
                            st.write(f'Texto:\n>{row["peticao_texto"]}')
                            selected_text = row["peticao_texto"]

                    else:
                        st.error("CNJ não encontrado.")
                else:
                    st.error("Formato inválido do CNJ. Por favor, use o formato: 0002192-73.2007.8.05.0250")
            
            c1, c2, _ = st.columns([2,2, 6])
            with c1:
                add_cnj_button = st.form_submit_button("Buscar Petição Inicial")
            with c2:
                add_peticao_button = st.form_submit_button("Adicionar Documento ao Chat")

                if add_peticao_button and selected_text:
                    # Encode the selected text to Base64
                    base64_text = base64.b64encode(selected_text.encode('utf-8')).decode('utf-8')

                    # Prepare the document data for the API request
                    document_data = {
                        "base64_file": base64_text
                    }

                    session_id = st.session_state.get("selected_session_id")  # Retrieve selected session ID from the session state

                    if session_id:  # Ensure the session ID is available
                        # API call to add the document
                        try:
                            response = requests.post(f"{API_URL}/chat/{session_id}/document?name={username}", json=document_data)

                            if response.status_code == 200:
                                st.success("Documento adicionado com sucesso ao chat!")
                            else:
                                st.error(f"Erro ao adicionar documento: {response.text}")
                        except Exception as e:
                            st.error(f"Erro ao conectar à API: {e}")
                    else:
                        st.error("Nenhum ID de sessão selecionado.")
            