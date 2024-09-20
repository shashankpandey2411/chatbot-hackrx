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

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create entrypoint script
RUN echo '#!/bin/sh\nollama run &\nsleep 5\nollama pull cniongolo/biomistral\nuvicorn fast1:app --host 0.0.0.0 --port 8000' > /start.sh && chmod +x /start.sh

# Expose the port for the FastAPI app
EXPOSE 8000

# Use the entrypoint script
ENTRYPOINT ["/start.sh"]
