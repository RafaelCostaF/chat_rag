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
    supervisor \
    && apt-get clean

# Set Python 3.9 as the default Python version
RUN ln -sf /usr/bin/python3.9 /usr/bin/python

# Set the working directory
WORKDIR /app

# Copy the application files
COPY backend /app
COPY frontend /app

# Copy the supervisord configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN pip install -r requirements.txt

# Expose ports for FastAPI and Streamlit
EXPOSE 8888 8501

# Set the entrypoint to run the supervisor
CMD ["supervisord", "-n"]
