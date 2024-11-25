# Use Debian Bullseye as the base image
FROM debian:bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Update the system and install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    software-properties-common \
    python3.9 \
    python3.9-distutils \
    python3-pip \
    && apt-get clean

# Set Python 3.9 as the default Python version
RUN ln -sf /usr/bin/python3.9 /usr/bin/python

# Install required Python packages
RUN pip install --no-cache-dir \
    numpy \
    fastapi \
    tinydb \
    requests \
    llama_index \
    llama_index-embeddings-huggingface \
    llama_index-llms-ollama \
    python-dotenv \
    uvicorn

# Clone the repository
RUN git clone https://github.com/RafaelCostaF/chat_rag.git /app

# Set the working directory
WORKDIR /app

# Expose port 8888 for the FastAPI app
EXPOSE 8888

# Set the entrypoint to run the application
CMD ["python", "main.py"]
