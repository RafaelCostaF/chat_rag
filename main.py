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
import json
import requests
import re

from Functions.messages import generate_response
from Functions.vector_store import recreate_vector_store, load_vector_store


# START initial config -------------------------------

vector_store = load_vector_store() 

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

# Initialize TinyDB (using a file-based database)
db = TinyDB('chat_db.json')
sessions_table = db.table('sessions')

# END initial config ----------------------------------





# START endpoints

@app.post("/recreate_vector_db")
def recreate_vector_db():
    try:
        recreate_vector_store()
        return {"message": "Vector database successfully created."}
    except Exception as e:
        logging.error(f"Failed to create vector database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create vector database. Exception: {str(e)}")
















# Execução da API ------------------------------------ 

if __name__ == "__main__":
    import uvicorn
    import sys

    if sys.platform == "win32" and sys.version_info >= (3, 8):
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(app, host="0.0.0.0", port=8888)