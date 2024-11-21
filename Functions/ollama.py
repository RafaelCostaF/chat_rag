import os
import requests
from dotenv import load_dotenv

def query_ollama(prompt: str):
    """
    Queries the specified model running on the Ollama server.

    The server URL and model name are specified in the `.env` file as
    `OLLAMA_URL` and `OLLAMA_MODEL`.

    Args:
        prompt (str): The text prompt to send to the model.

    Returns:
        dict: The JSON response from the Ollama API.
    """
    # Load environment variables from .env
    load_dotenv()
    ollama_url = os.getenv("OLLAMA_URL")
    ollama_model = os.getenv("OLLAMA_MODEL")
    
    if not ollama_url:
        raise ValueError("OLLAMA_URL not found in .env file")
    if not ollama_model:
        raise ValueError("OLLAMA_MODEL not found in .env file")
    
    endpoint = f"{ollama_url}/api/v1/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": ollama_model,
        "prompt": prompt
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}