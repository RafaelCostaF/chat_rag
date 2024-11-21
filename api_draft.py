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
