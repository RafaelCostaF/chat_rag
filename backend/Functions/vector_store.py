from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.settings import Settings
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import os
from dotenv import load_dotenv

load_dotenv()
VECTOR_STORAGE_PATH = os.getenv("VECTOR_STORAGE_PATH")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


# Configure LLM
Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=360.0)

# Configure embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

def recreate_vector_store():
    # Load documents from the specified directory
    documents = SimpleDirectoryReader(VECTOR_STORAGE_PATH).load_data()

    # Create index from documents
    storage_context = StorageContext.from_defaults(persist_dir="index_storage")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # Persist the index to the specified directory
    index.storage_context.persist()

    return True

def load_vector_store():
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORAGE_PATH)

    # load index
    index = load_index_from_storage(storage_context)

    return index