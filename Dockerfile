FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI (adjust according to the actual download method)
# Assuming you can download the binary for Ollama CLI from a URL, replace with actual URL if needed
RUN curl -fsSL https://ollama.ai/install | bash

# Pull the model
RUN ollama pull cniongolo/biomistral

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "fast1:app", "--host", "0.0.0.0", "--port", "8000"]

