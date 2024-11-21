import os
import json
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