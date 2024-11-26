FROM  python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the application files
COPY backend /app
COPY requirements.txt /app

RUN pip install -r requirements.txt

# Expose ports for FastAPI
EXPOSE 8888

# Set the entrypoint to run the supervisor
CMD ["python", "main.py"]
