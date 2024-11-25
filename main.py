import numpy as np
import base64
from datetime import datetime
import time
from fastapi import FastAPI, HTTPException, Request
import logging
from pydantic import BaseModel
from tinydb import TinyDB, Query, where
from collections import defaultdict
from typing import List
import string
import json
import requests
import re

from Functions.messages import generate_response
from Functions.vector_store import recreate_vector_store, load_vector_store
from Functions.ollama import query_ollama


# START initial config -------------------------------

vector_store = load_vector_store() 
query_engine = vector_store.as_query_engine()

app = FastAPI()

# START LOGGING --------------------
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

# END LOGGING --------------------


# Store request count for each endpoint
request_count = defaultdict(int)


# START database settings ---------------------

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

# END database settings ---------------------



# END initial config ----------------------------------





# START endpoints

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

@app.post("/recreate_vector_db")
def recreate_vector_db():
    try:
        recreate_vector_store()
        return {"message": "Vector database successfully created."}
    except Exception as e:
        logging.error(f"Failed to create vector database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create vector database. Exception: {str(e)}")



@app.post("/chat/{session_id}/start_session")
def start_session(session_id: str, name):
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
def get_sessions(name: str):
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


@app.post("/chat/{session_id}/send_message_vector_db")
def send_message_vector_db(session_id: str, message: str, name: str):
     # Retrieve the user
    user = get_user(name)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Retrieve the session for this user
    session = get_user_session(user, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found for this user")
    
    if message.strip() == "":
        raise HTTPException(status_code=400, detail="Empty or invalid user message")
    
    # Add user's message to the session
    session['messages'].append({"role": "user", "content": message})
    
    response = generate_response(session['messages'],query_engine.query,query_ollama)
    
    # Add assistant's response to the session
    session['messages'].append({"role": "system", "content": response})
    
    # Update the session in the database using the correct session ID
    sessions_table.update(session, where('id') == session_id)
    
    return {"response": response}

@app.get("/chat/{session_id}/messages")
def get_messages(session_id: str, name: str):
    # Retrieve the user
    user = get_user(name)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Retrieve the session for this user
    session = get_user_session(user, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found for this user")
    
    return {"messages": session['messages']}


# Can be changed later to add more documents to the Vector Storage
# @app.post("/chat/{session_id}/document")
# def add_document(session_id: str, document: Document, name: str = Query(...)):
#     # Retrieve the user
#     user = get_user(name)
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")
    
#     # Retrieve the session for this user
#     session = get_user_session(user, session_id)
#     if not session:
#         raise HTTPException(status_code=404, detail="Chat session not found for this user")
    
#     extracted_text = extrair(document.base64_file)
    
#     if 'documents' not in session:
#         session['documents'] = []
    
#     if any(doc['content'] == extracted_text for doc in session['documents']):
#         raise HTTPException(status_code=400, detail="Document already exists in this session.")
    
#     session['documents'].append({"content": remove_stopwords(extracted_text)})
    
#     # Update the session in the database
#     Session = TinyQuery()
#     sessions_table.update(session, Session.id == session_id)
    
#     return {"detail": "Document added successfully", "extracted_text": extracted_text}

# @app.get("/chat/{session_id}/documents")
# def get_documents(session_id: str, name: str = Query(...)):
#     # Retrieve the user
#     user = get_user(name)
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")
    
#     # Retrieve the session for this user
#     session = get_user_session(user, session_id)
#     if not session:
#         raise HTTPException(status_code=404, detail="Chat session not found for this user")
    
#     # Get documents associated with the session
#     documents = session.get('documents', [])
    
#     return {"documents": documents}

# @app.get("/chat/{session_id}/documents/summary", response_model=List[str])
# def get_documents_summary(session_id: str, name: str = Query(...)):
#     # Retrieve the user
#     user = get_user(name)
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")
    
#     # Retrieve the session for this user
#     session = get_user_session(user, session_id)
#     if not session:
#         raise HTTPException(status_code=404, detail="Chat session not found for this user")
    
#     # Get documents associated with the session
#     documents = session.get('documents', [])
#     # print(documents)
#     # Process each document to create the summary
#     summary = []
#     for document in documents:
#         # Clean the document: remove unwanted characters
#         cleaned_document = ' '.join(document['content'].split())  # This will remove extra spaces and tabs
#         if len(cleaned_document) <= 50:
#             summary.append(cleaned_document)
#         else:
#             summary.append(cleaned_document[:25] + " ... " + cleaned_document[-25:])
    
#     return summary






# Execução da API ------------------------------------ 

if __name__ == "__main__":
    import uvicorn
    import sys

    if sys.platform == "win32" and sys.version_info >= (3, 8):
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(app, host="0.0.0.0", port=8888)