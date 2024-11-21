import numpy as np
import base64
from datetime import datetime
import time
from fastapi import FastAPI, HTTPException, Request
import logging
from pydantic import BaseModel
from tinydb import TinyDB, Query
from collections import defaultdict
from typing import List
import string
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json
import requests
import re


 # FUNÇÕES API ---------------- 

def MySplit( l:list[Document], caracters:list[str]):
    NewList = []
    CharToSplit = 0
    Temp = ""

    for doc in l:

        TextToSplit = Temp + doc.page_content
        Temp = ""


        while CharToSplit < len(caracters):

            if CharToSplit - 1 < 0:
                extra = ""
            else:
                extra = caracters[CharToSplit - 1]


            if caracters[CharToSplit] in TextToSplit:

                s = TextToSplit.split(caracters[CharToSplit],1)
                NewList.append( Document( page_content = extra + s[0], 
                                          metadata = {'source': doc.metadata['source'], 'page': doc.metadata['page']}) )
                TextToSplit = s[1]
                CharToSplit += 1

            else:
                Temp = TextToSplit
                break

        if CharToSplit == len(caracters):
            Temp = TextToSplit


    NewList.append( Document( page_content = caracters[CharToSplit - 1] + TextToSplit, 
                    metadata = {'source': doc.metadata['source'], 'page': doc.metadata['page']})
                    )
    return NewList

def MySplitRegex( l:list[Document], regex:str ) -> list[Document]:
    NewList = [Document(page_content="")]

    for doc in l:

        TextToSplit = doc.page_content
        TextToSplit = re.split(regex, TextToSplit)

        startChar = 0

        if re.match(regex, TextToSplit[0]) == None:
            NewList[-1].page_content = NewList[-1].page_content + TextToSplit[0]
            startChar = 1
            

        for i in range(startChar, len(TextToSplit), 2):
            NewList.append(Document(metadata = {'source': doc.metadata['source'], 'page': doc.metadata['page']},
                                    page_content = TextToSplit[i] + TextToSplit[i+1]))


    return NewList

def extract_text_from_json(response_text):
    """
    Extrai o texto da resposta JSON que contém 
    múltiplos objetos JSON separados por novas linhas.
    """
    lines = response_text.splitlines()
    full_text = []

    for line in lines:
        try:
            # Tenta carregar cada linha como um JSON
            json_obj = json.loads(line)
            # Adiciona o texto da resposta ao resultado
            if 'response' in json_obj:
                full_text.append(json_obj['response'])
        except json.JSONDecodeError as e:
            print(f"Erro ao processar parte do JSON: {e} na linha: {line}")

    return ''.join(full_text)
    
class OllamaLLM:
    def __init__(self, model, api_url="http://ollama:11434/api/generate"):
        self.model = model
        self.api_url = api_url
        

    def generate(self, prompt, model):
        response = requests.post(self.api_url, json={"prompt": prompt, "model": self.model})
        if response.status_code == 200:
            print(f"Conexao feita com sucesso!")
            #print(f"Response: {response.json()}")
            return response#["response"]
        else:
            return f"Error: {response.status_code}"



def generate_response(messages):

    # Filter messages to get only those from the user
    user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
    
    # Check if there are any user messages
    if user_messages:
        last_user_message = user_messages[-1]
    else:
        last_user_message = None
        
    # Step 1: Obter entrada do usuário
    user_input = last_user_message

    context = retriever.invoke(question)

    
    prompt = f"""
    
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Você um chatBot prestativo que ajuda estudantes a entender normas da faculdade.
    Responda as perguntas baseadas no contexto:{context}. Diga o documento e a pagina que usou para responder a pergunta.
    Se não souber responder, diga "Eu não sei".<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    
    Minha pergunta é: {user_input}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    
    """
        
    
    predict = ollama_llm.generate(prompt)

    text = extract_text_from_json(predict.text)


    return text

def split_document(document, chunk_size=1000):
    # Split by period, or any other method you prefer
    sentences = document.split('. ')
    chunks = []
    chunk = []

    for sentence in sentences:
        chunk.append(sentence)
        if len(' '.join(chunk)) > chunk_size:
            chunks.append(' '.join(chunk))
            chunk = []

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks

def remove_stopwords(text):
    # Load Portuguese stopwords
    stop_words = set(stopwords.words('portuguese'))
    
    # Tokenize text by splitting on whitespace
    words = text.split()
    
    # Filter out stopwords and punctuation
    filtered_text = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
    
    # Join the filtered words back into a single string
    return ' '.join(filtered_text)








# CONFIG INICIAL -------------------------------

app = FastAPI()

# LOGGING 
# Set up logging configuration to write logs to a file in the current directory
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler("./api_log.log"),  # Log file path in the current directory
        logging.StreamHandler()  # Optional: Also output to console
    ]
)
# Create a logger instance
logger = logging.getLogger("api_logger")
# Example usage of the logger
logger.info("Logging setup complete and writing to ./api_log.log")

# Store request count for each endpoint
request_count = defaultdict(int)

# Initialize TinyDB (using a file-based database)
db = TinyDB('chat_db.json')
sessions_table = db.table('sessions')




modelo_llm = "llama3.1:70b"

ollama_llm = OllamaLLM(model=modelo_llm)

embedding=OllamaEmbeddings(model=modelo_llm, base_url="http://ollama:11434/",show_progress=True)

loader = PyPDFLoader("documents/FAQ.pdf")
FAQregex = "(?:\n \n|\n \n )([0-9]+\\.)"

loader1 = PyPDFLoader("documents/a.pdf")
aRegex = "(?:\n|\n )(Art\\. [0-9]+)"


pages = loader.load()
for page in pages:
    page.page_content = page.page_content.replace("\u200b", "").replace("  ", " ")
    
pages = MySplitRegex(pages, FAQregex)

pages1 = loader1.load()
pages1 = MySplitRegex(pages1, aRegex)

pages += pages1


vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding)

retriever = vectorstore.as_retriever()







# BASE MODELS -------------------------------

class Session(BaseModel):
    name: str

class Message(BaseModel):
    content: str
    





# ENDPOINTS API -------------------------------


# Middleware to log request details
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Increment request count for each endpoint
    request_count[request.url.path] += 1
    
    # Get current date and time
    request_time = datetime.now()
    
    # Log the request details
    logger.info(f"Request for {request.url.path} at {request_time}")
    
    # Process the request
    response = await call_next(request)
    
    return response
    
@app.post("/chat/start_session")
def start_session(session: Session):
    # Check if session with the same name (used as id) already exists
    existing_session = next((s for s in sessions_table if s['id'] == session.name), None)
    
    if existing_session:
        raise HTTPException(status_code=400, detail="Session with this name already exists.")
    
    # Insert the new session with 'id' being the session name
    sessions_table.insert({'id': session.name, 'messages': []})
    return {"session_id": session.name}  # Return session name as session ID

@app.get("/chat/sessions")
def get_sessions():
    sessions = [{'id': session['id']} for session in sessions_table]
    return sessions

@app.post("/chat/{session_id}/send_message")
def send_message(session_id: str, message: Message):
    Session = Query()
    session = sessions_table.get(Session.id == session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sessão de chat não encontrada")
    if message.content.strip() == "":
        raise HTTPException(status_code=400, detail="Mensagem de usuário vazia ou inválida")
    
    # Add user's message to the session with role set to "user"
    session['messages'].append({"role": "user", "content": message.content})
    
    response = generate_response(session['messages'])
    
    # Add assistant's response to the session
    session['messages'].append({"role": "system", "content": response})
    
    # Update the session in the database
    sessions_table.update(session, Session.id == session_id)
    
    return {"response": response}

@app.get("/chat/{session_id}/messages")
def get_messages(session_id: str):
    Session = Query()
    session = sessions_table.get(Session.id == session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Sessão de chat não encontrada")    
    
    return {"messages": session['messages']}







# Execução da API ------------------------------------ 

if __name__ == "__main__":
    import uvicorn
    import sys

    if sys.platform == "win32" and sys.version_info >= (3, 8):
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(app, host="0.0.0.0", port=8888)
