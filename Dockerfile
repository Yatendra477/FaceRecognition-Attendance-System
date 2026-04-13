FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories that will need write permissions at runtime
RUN mkdir -p /app/models /app/data /app/dataset /app/embeddings \
    && chmod -R 777 /app/models /app/data /app/dataset /app/embeddings

# Expose port 7860 (Default for Hugging Face Spaces)
EXPOSE 7860

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
