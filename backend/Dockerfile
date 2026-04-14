FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by TensorFlow and Pillow
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — only reinstalls if requirements change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all backend source files
COPY . .

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Start FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]