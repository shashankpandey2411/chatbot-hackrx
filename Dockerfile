# Use a base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy your application files into the container
COPY . .

# Pull the model (this will run while the container is built)
RUN ollama pull cniongolo/biomistral

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for the FastAPI app
EXPOSE 8000

# Start the Ollama app and then run the FastAPI app
CMD ["sh", "-c", "ollama run & uvicorn fast1:app --host 0.0.0.0 --port 8000"]
