from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import base64
import os
from io import BytesIO
# from text_extractor.extractor import extrair_texto
import transformers
import torch
from datetime import datetime
import time
from fastapi import FastAPI, HTTPException, Request, Query
import logging
from pydantic import BaseModel
from tinydb import TinyDB, where, Query as TinyQuery
from uuid import uuid4
import copy
from collections import defaultdict
from typing import List
import nltk
from nltk.corpus import stopwords
import string
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)

nltk.download('stopwords')

app = FastAPI()

embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("db_faiss", embeddings, allow_dangerous_deserialization=True)


# Carregar o modelo de geração de texto
# Get the model path from an environment variable
model_id = os.getenv("MODEL_PATH")
if not model_id:
    raise ValueError("Environment variable MODEL_PATH is required but not set.")

text_pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Carregar o modelo de embeddings para recuperação
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


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


# Função para extrair texto de um arquivo base64
def extrair(base64_file):
    file_data = base64.b64decode(base64_file)
    file = BytesIO(file_data)

    texto = extrair_texto(file)

    texto_base = texto['text']
    
    return texto_base

class Session(BaseModel):
    name: str

class Message(BaseModel):
    content: str
    
class Document(BaseModel):
    base64_file: str
    
    
    
# Initialize TinyDB (using a file-based database)
db = TinyDB('chat_db.json')
sessions_table = db.table('sessions')
users_table = db.table('users')
    

# Add a new user if it doesn't exist
def add_user_if_not_exists(name: str):
    User = Query()
    user = users_table.get(User.name == name)
    if not user:
        users_table.insert({'name': name})

# Retrieve the user
def get_user(name: str):
    User = Query()
    return users_table.get(User.name == name)

# Retrieve a session for a specific user
def get_user_session(user, session_id):
    Session = Query()
    return sessions_table.get((Session.id == session_id) & (Session.user_id == user.doc_id))


# Add a new user if it doesn't exist
def add_user_if_not_exists(name: str):
    User = TinyQuery()
    user = users_table.get(User.name == name)
    if not user:
        users_table.insert({'name': name})

# Retrieve the user
def get_user(name: str):
    User = TinyQuery()
    return users_table.get(User.name == name)

# Retrieve a session for a specific user
def get_user_session(user, session_id):
    Session = TinyQuery()
    return sessions_table.get((Session.id == session_id) & (Session.user_id == user.doc_id))


@app.post("/chat/{session_id}/start_session")
def start_session(session_id: str, name: str = Query(...)):
    # Add the user to the database if not exists
    add_user_if_not_exists(name)
    
    # Retrieve the user
    user = get_user(name)
    
    # Check if session with the same ID already exists for this user
    existing_session = get_user_session(user, session_id)
    if existing_session:
        raise HTTPException(status_code=400, detail="Session with this ID already exists for this user.")
    
    # Insert the new session with 'id' being the session_id and associate it with the user
    sessions_table.insert({'id': session_id, 'user_id': user.doc_id, 'messages': []})
    
    return {"session_id": session_id, "username": name}


@app.get("/chat/sessions")
def get_sessions(name: str = Query(...)):
    # Retrieve the user
    user = get_user(name)
    if not user:
        # Create the user if not found
        add_user_if_not_exists(name)
        user = get_user(name)  # Retrieve the newly created user again

    # Ensure that the user is valid
    if user is None:
        raise HTTPException(status_code=404, detail="User could not be created.")

    # Get sessions for this user
    sessions = [{'id': session['id']} for session in sessions_table if session.get('user_id') == user.doc_id]

    return {"sessions": sessions}  # Return sessions in a dictionary for consistency


@app.post("/chat/{session_id}/send_message")
def send_message(session_id: str, message: Message, name: str = Query(...)):
    # Retrieve the user
    user = get_user(name)
    if not user:
        # Create the user if not found
        add_user_if_not_exists(name)
        user = get_user(name)  # Retrieve the newly created user
    
    # Retrieve the session for this user
    session = get_user_session(user, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found for this user")
    
    # Check for an empty or invalid message
    if message.content.strip() == "":
        raise HTTPException(status_code=400, detail="Empty or invalid user message")
    
    # Add the user's message to the session's messages
    session['messages'].append({"role": "user", "content": message.content})
    
    # Extract content from each document in session['documents'] if it exists
    document_contents = [doc['content'] for doc in session.get('documents', [])]
    
    # Generate a response based on the session messages and document contents
    response = generate_response(session['messages'], document_contents)
    
    # Add assistant's response to the session
    session['messages'].append({"role": "system", "content": response})
    
    # Update the session in the database using the correct session ID
    sessions_table.update(session, where('id') == session_id)
    
    return {"response": response}



@app.post("/chat/{session_id}/send_message_vector_db")
def send_message_vector_db(session_id: str, message: Message, name: str = Query(...)):
     # Retrieve the user
    user = get_user(name)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Retrieve the session for this user
    session = get_user_session(user, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found for this user")
    
    if message.content.strip() == "":
        raise HTTPException(status_code=400, detail="Empty or invalid user message")
    
    # Add user's message to the session
    session['messages'].append({"role": "user", "content": message.content})
    
    response = generate_response_full_vector_db(session['messages'])
    
    # Add assistant's response to the session
    session['messages'].append({"role": "system", "content": response})
    
    # Update the session in the database using the correct session ID
    sessions_table.update(session, where('id') == session_id)
    
    return {"response": response}


@app.get("/chat/{session_id}/messages")
def get_messages(session_id: str, name: str = Query(...)):
    # Retrieve the user
    user = get_user(name)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Retrieve the session for this user
    session = get_user_session(user, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found for this user")
    
    return {"messages": session['messages']}

@app.post("/chat/{session_id}/document")
def add_document(session_id: str, document: Document, name: str = Query(...)):
    # Retrieve the user
    user = get_user(name)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Retrieve the session for this user
    session = get_user_session(user, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found for this user")
    
    extracted_text = extrair(document.base64_file)
    
    if 'documents' not in session:
        session['documents'] = []
    
    if any(doc['content'] == extracted_text for doc in session['documents']):
        raise HTTPException(status_code=400, detail="Document already exists in this session.")
    
    session['documents'].append({"content": remove_stopwords(extracted_text)})
    
    # Update the session in the database
    Session = TinyQuery()
    sessions_table.update(session, Session.id == session_id)
    
    return {"detail": "Document added successfully", "extracted_text": extracted_text}

@app.get("/chat/{session_id}/documents")
def get_documents(session_id: str, name: str = Query(...)):
    # Retrieve the user
    user = get_user(name)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Retrieve the session for this user
    session = get_user_session(user, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found for this user")
    
    # Get documents associated with the session
    documents = session.get('documents', [])
    
    return {"documents": documents}

@app.get("/chat/{session_id}/documents/summary", response_model=List[str])
def get_documents_summary(session_id: str, name: str = Query(...)):
    # Retrieve the user
    user = get_user(name)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Retrieve the session for this user
    session = get_user_session(user, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found for this user")
    
    # Get documents associated with the session
    documents = session.get('documents', [])
    # print(documents)
    # Process each document to create the summary
    summary = []
    for document in documents:
        # Clean the document: remove unwanted characters
        cleaned_document = ' '.join(document['content'].split())  # This will remove extra spaces and tabs
        if len(cleaned_document) <= 50:
            summary.append(cleaned_document)
        else:
            summary.append(cleaned_document[:25] + " ... " + cleaned_document[-25:])
    
    return summary

def generate_response(messages, documents=[]):

    # Filter messages to get only those from the user
    user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
    
    # Check if there are any user messages
    if user_messages:
        last_user_message = user_messages[-1]
    else:
        last_user_message = None
        
    # Step 1: Obter entrada do usuário
    user_input = last_user_message

    retrieved_docs = ""
    
    if documents != []:
        # Carregar os documentos e gerar embeddings
        document_embeddings = embedding_model.encode(documents, convert_to_tensor=True)

        # Indexação usando FAISS
        index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        faiss.normalize_L2(document_embeddings.cpu().numpy())
        index.add(document_embeddings.cpu().numpy())

        # Step 2: Recuperar documentos relevantes
        user_input_embedding = embedding_model.encode([user_input], convert_to_tensor=True)

        # K=1 pois, com documentos largos, não faz sentidos misturá-los na busca do texto para 1 prompt
        # Isso acontece pois a qualidade da busca não é perfeita, então pode ser que tenhamos um texto de um documento x
        # e de outro totalmente diferente (na segmentação do documento)
        D, I = index.search(user_input_embedding.cpu().numpy(), k=3)

        retrieved_docs = "\n".join([documents[i] for i in I[0]])
            
        # Step 3: Combinar documentos recuperados com entrada do usuário
        context = f"{retrieved_docs}\n{user_input}"

    else:
        context = user_input

    terminators = [
        text_pipeline.tokenizer.eos_token_id,
        text_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # Fazendo cópia da lista de mensagens, para não modificar o original
    copy_msg = copy.deepcopy(messages)
    
    # Colocando última mensagem como contexto
    copy_msg[-1]["content"] = context
    
    copy_msg = copy_msg[-12:]
        
    
    try:
        # Step 4: Gerar resposta
        outputs = text_pipeline(
            copy_msg,
            max_new_tokens=4096,
            eos_token_id=terminators,
            repetition_penalty=1.03,
        )
    
    except torch.cuda.OutOfMemoryError as e:
        print("CUDA out of memory error encountered.")
        # Reduce the size of max_new_tokens and try again
        try:
            # Step 1: Segment the retrieved document into smaller chunks
            segmented_documents = split_document(retrieved_docs)  # Use the single document

            # Step 2: Generate embeddings for each segment
            document_embeddings = embedding_model.encode(segmented_documents, convert_to_tensor=True)

            # Step 3: Index using FAISS
            index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
            faiss.normalize_L2(document_embeddings.cpu().numpy())
            index.add(document_embeddings.cpu().numpy())

            # Step 4: Retrieve relevant segments based on user input
            user_input_embedding = embedding_model.encode([user_input], convert_to_tensor=True)
            D, I = index.search(user_input_embedding.cpu().numpy(), k=6)

            # Combine retrieved segments into a single string
            retrieved_docs = "\n".join([segmented_documents[i] for i in I[0]])
            
            with open("sub_retrieved.txt","w") as file:
                file.writelines(retrieved_docs)
            
            # Step 5: Combine retrieved segments with user input
            context = f"{retrieved_docs}\n{user_input}"

            copy_msg[-1]["content"] = context

            # Step 6: Generate response
            outputs = text_pipeline(
                copy_msg,
                max_new_tokens=4096,
                eos_token_id=terminators,
                repetition_penalty=1.03,
            )
            
        except torch.cuda.OutOfMemoryError:
            try:
                # Step 1: Segment the retrieved document into smaller chunks
                segmented_documents = split_document(retrieved_docs)  # Use the single document

                # Step 2: Generate embeddings for each segment
                document_embeddings = embedding_model.encode(segmented_documents, convert_to_tensor=True)

                # Step 3: Index using FAISS
                index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
                faiss.normalize_L2(document_embeddings.cpu().numpy())
                index.add(document_embeddings.cpu().numpy())

                # Step 4: Retrieve relevant segments based on user input
                user_input_embedding = embedding_model.encode([user_input], convert_to_tensor=True)
                D, I = index.search(user_input_embedding.cpu().numpy(), k=3)

                # Combine retrieved segments into a single string
                retrieved_docs = "\n".join([segmented_documents[i] for i in I[0]])

                with open("sub_retrieved.txt","w") as file:
                    file.writelines(retrieved_docs)

                # Step 5: Combine retrieved segments with user input
                context = f"{retrieved_docs}\n{user_input}"

                copy_msg[-1]["content"] = context

                # Step 6: Generate response
                outputs = text_pipeline(
                    copy_msg,
                    max_new_tokens=4096,
                    eos_token_id=terminators,
                    repetition_penalty=1.03,
                )

            except torch.cuda.OutOfMemoryError:
                raise HTTPException(
                    status_code=507,  # 507 Insufficient Storage is a fitting status code
                    detail="Erro de memória: A memória máxima de GPU foi alocada e a tentativa de recuperação falhou. Tente liberar memória de GPU ou reduzir o tamanho do modelo.",
                )

    
    # Extract only the content of the assistant
    assistant_content = [msg['content'] for msg in outputs[0]['generated_text'] if msg['role'] == 'assistant']

    # If you expect only one response and want it as a string:
    assistant_content_text = assistant_content[0] if assistant_content else None

    return assistant_content_text


def generate_response_full_vector_db(messages):
    
    # Filtra mensagens para obter apenas aquelas do usuário
    user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
    
    # Verifica se há mensagens do usuário
    if user_messages:
        last_user_message = user_messages[-1]
    else:
        last_user_message = None

    # Obter a entrada do usuário
    user_input = last_user_message

    # Extrai embeddings da entrada do usuário
    user_input_embedding = embedding_model.encode([user_input], convert_to_tensor=True)
    
    #Realiza a busca por similaridade
    docs = vector_store.similarity_search(user_input, k=5)


    # Acessa os conteúdos dos documentos recuperados
    retrieved_docs = "\n".join([doc.page_content for doc in docs])

    # Cria o contexto combinando os documentos recuperados e a entrada do usuário
    context = f"DOCUMENTOS BASE:\n{retrieved_docs}\nINPUT USUÁRIO:\n{user_input}"

    
    # Termina a função com a lógica de geração de resposta
    terminators = [
        text_pipeline.tokenizer.eos_token_id,
        text_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Fazendo uma cópia da lista de mensagens, para não modificar o original
    copy_msg = copy.deepcopy(messages)

    # Coloca a última mensagem como contexto
    copy_msg[-1]["content"] = context
    copy_msg = copy_msg[-12:]  # Limita as mensagens para evitar exceder o limite de tokens

    try:
        # Gera a resposta
        outputs = text_pipeline(
            copy_msg,
            max_new_tokens=4096,
            eos_token_id=terminators,
            repetition_penalty=1.03,
        )
    
    except torch.cuda.OutOfMemoryError:
        print("Erro de memória da CUDA encontrado.")



    # Extract only the content of the assistant
    assistant_content = [msg['content'] for msg in outputs[0]['generated_text'] if msg['role'] == 'assistant']

    # If you expect only one response and want it as a string:
    assistant_content_text = assistant_content[0] if assistant_content else None

    return assistant_content_text

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


# Execução da API

if __name__ == "__main__":
    import uvicorn
    import sys

    if sys.platform == "win32" and sys.version_info >= (3, 8):
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(app, host="0.0.0.0", port=8888)
