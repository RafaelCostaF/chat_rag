from ast import Pass
from ctypes import alignment
import streamlit as st
import pandas as pd
import requests
import base64
import time
import re

# API base URL
API_URL = "http://localhost:8888/chat"

# Obtendo os parâmetros da URL
params = st.query_params
username = params.get('name', [''])

# st.set_page_config(layout="wide")

# Add the logo to the sidebar
st.sidebar.image("LogoSTI.png" , use_column_width=True)

# REMOVING STREAMLIT STYLE
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stApp [data-testid="stToolbar"]{display:none;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



# Track if the modal is open
if 'show_modal_anexo' not in st.session_state:
    st.session_state.show_modal_anexo = False
    
# Track if the modal is open
if 'show_modal_modelos' not in st.session_state:
    st.session_state.show_modal_modelos = False

# Função para verificar o status da API
def check_api_status():
    try:
        response = requests.get(f"{API_URL}/sessions?name={username}")
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        return False

# Function to toggle modal visibility
def toggle_modal_anexo():
    st.session_state.show_modal_anexo = not st.session_state.show_modal_anexo

# Function to toggle modal visibility
def toggle_modal_modelos():
    st.session_state.show_modal_modelos = not st.session_state.show_modal_modelos

# Verificar se a API está disponível
if not check_api_status():
    st.error("API fora do ar. Tentando reconectar em 30 segundos...")
    time.sleep(30)  # Espera de 1 minuto antes de tentar novamente
    st.rerun()  # Recarrega o aplicativo após a espera

# Initialize session state to hold the selected session and message history
if "selected_session_id" not in st.session_state:
    st.session_state.selected_session_id = None







########### SIDEBAR START -----------------------------------------------------------------------

# Sidebar: Start a new chat session

if username == '':
    st.sidebar.header("Olá !")
else:
    formatted_name = ' '.join(part.capitalize() for part in username.split('.'))
    st.sidebar.header(f'Olá, {formatted_name}! \n Selecione ou inicie um novo chat')

# st.sidebar.subheader("Iniciar ou Selecionar Chat")

# API call to retrieve all sessions
sessions_response = requests.get(f"{API_URL}/sessions?name={username}", headers={'accept': 'application/json'})

if sessions_response.status_code == 200:
    sessions = sessions_response.json()

    session_options = {session['id']: session['id'] for session in sessions['sessions']}

    # Add a blank option ('') as the first option in the dropdown
    session_options_with_blank = [''] + list(session_options.keys())

    # Set the dropdown without automatically selecting any session
    selected_session_id = st.sidebar.selectbox(
        "Escolha o Chat",
        options=session_options_with_blank,
        index=session_options_with_blank.index(st.session_state.selected_session_id)
        if st.session_state.get('selected_session_id') in session_options else 0
    )
    
    # Store the selected session ID in session state
    if 'selected_session_id' not in st.session_state or st.session_state.selected_session_id != selected_session_id:
        st.session_state.selected_session_id = selected_session_id
        st.rerun()
    

else:
    st.sidebar.error("Erro ao buscar sessões!")

# Form to start a new session
with st.sidebar.form("start_session"):
    new_session_name = st.text_input("Nome do novo Chat")
    start_session_button = st.form_submit_button("Iniciar Chat")

# If "Start Session" button is clicked
if start_session_button:
    if new_session_name:
        # API call to start a session
        response = requests.post(f"{API_URL}/{new_session_name}/start_session?name={username}")
        if response.status_code == 200:
            st.session_state.selected_session_id = new_session_name
            st.rerun()
            st.sidebar.success(f"Chat '{new_session_name}' iniciada com sucesso!")
            
        elif response.status_code == 400:
            st.sidebar.error("Chat com este nome já existe!")
        else:
            st.sidebar.error("Erro ao iniciar o chat!")
    else:
        st.sidebar.error("Por favor, insira um nome para o chat.")


########### SIDEBAR END -----------------------------------------------------------------------






########### MAIN SCREEN START -----------------------------------------------------------------------

# Main Page Content: Accessing the selected session
if st.session_state.selected_session_id:
    st.header(f"{st.session_state.selected_session_id}", anchor="top",divider=True)

    # Retrieve and display messages from the selected session
    messages_response = requests.get(f"{API_URL}/{st.session_state.selected_session_id}/messages?name={username}")
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
        st.error("Erro ao buscar mensagens!")



    # Form to send a message
    with st.form("send_message"):
        col1, col2 = st.columns([9,1])
        with col1:
            message_content = st.text_input("", label_visibility="collapsed", placeholder="🧑‍💻 Digite sua mensagem")
        with col2:
            send_message_button = st.form_submit_button("➤", help="Enviar mensagem")
        radio_button_db = st.radio(
                "Onde realizar a consulta?",
                ["Arquivos desta Seção", "Dados PGE"]
            )

    # Lógica de envio da mensagem
    if send_message_button:
        if message_content.strip():
            # Definir o endpoint baseado na escolha do radio
            if radio_button_db == "Dados PGE":
                endpoint = f"{API_URL}/{st.session_state.selected_session_id}/send_message_vector_db?name={username}"
            else:
                endpoint = f"{API_URL}/{st.session_state.selected_session_id}/send_message?name={username}"

            # Preparar os dados da mensagem
            message_data = {"content": message_content}

            # Tentativas de envio (com retry)
            max_retries = 3
            for attempt in range(max_retries):
                response = requests.post(endpoint, json=message_data)
                
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


    # Create columns for the buttons
    col_a, col_b, col_c = st.columns([1, 1, 1])  # Two equal-width columns

    with col_a:
        anexo_button = st.button("Anexar Documentos 📃", on_click=toggle_modal_anexo)

    # with col_b:
    #     ai_models_button = st.button("Modelos de Classificação💡", on_click=toggle_modal_modelos)


    
    # Display modal if the state is set to True
    if st.session_state.show_modal_anexo:
        with st.container():
            # Form to upload and add a document
            with st.form("add_document", clear_on_submit=True):
                st.header("Anexar documentos ao chat")
                # Make the GET request
                
                response = requests.get(f"{API_URL}/{st.session_state.selected_session_id}/documents/summary?name={username}", headers={'accept': 'application/json'})
                if response.status_code == 200:
                    # st.subheader("Documentos já anexados:")
                    documents = response.json()
                    st.write(documents) 

                else:
                    if response.status_code == 405:
                        st.text("Não existem documentos anexados")
                    else:
                        st.error("Erro ao carregar documentos!")

                uploaded_files = st.file_uploader("Escolha um arquivo", type=["txt", "pdf", "docx"], accept_multiple_files=True)
                add_document_button = st.form_submit_button("Carregar Documentos")
                if add_document_button:
                    if uploaded_files:
                        # Show a loading spinner while processing
                        with st.spinner("Enviando documentos..."):
                            for uploaded_file in uploaded_files:
                                # Read the file and encode it to base64
                                file_bytes = uploaded_file.read()
                                base64_encoded_file = base64.b64encode(file_bytes).decode('utf-8')
                                document_data = {"base64_file": base64_encoded_file}

                                # API call to add the document
                                response = requests.post(f"{API_URL}/{st.session_state.selected_session_id}/document?name={username}", json=document_data)

                                if response.status_code == 200:
                                    extracted_text = response.json().get('extracted_text', '')
                                    st.success(f"Documento '{uploaded_file.name}' carregado com sucesso! \n")
                                else:
                                    if response.status_code == 400:
                                        st.error(f"Documento '{uploaded_file.name}' já existe nesse chat.")
                                    else:
                                        st.error(f"Erro ao enviar documento '{uploaded_file.name}'!")
                            time.sleep(0.5)
                            st.rerun()
                    else:
                        st.error("Por favor, carregue pelo menos um arquivo válido.")



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
                                response = requests.post(f"{API_URL}/{session_id}/document?name={username}", json=document_data)

                                if response.status_code == 200:
                                    st.success("Documento adicionado com sucesso ao chat!")
                                else:
                                    st.error(f"Erro ao adicionar documento: {response.text}")
                            except Exception as e:
                                st.error(f"Erro ao conectar à API: {e}")
                        else:
                            st.error("Nenhum ID de sessão selecionado.")
                


else:
    st.write("Por favor, selecione uma sessão na barra lateral.")