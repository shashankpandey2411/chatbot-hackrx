# Use a base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy your application files
COPY . .

# Pull the model separately
RUN sleep 5 && ollama pull cniongolo/biomistral

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for the FastAPI app
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "fast1:app", "--host", "0.0.0.0", "--port", "8000"]

