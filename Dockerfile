# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy model files and directories
COPY models/ ./models/
COPY preprocessors/ ./preprocessors/

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port (Cloud Run will override this)
EXPOSE 8080

# Use gunicorn for production
# --workers: number of worker processes (adjust based on your needs)
# --threads: number of threads per worker
# --timeout: worker timeout in seconds
# --bind: bind to all interfaces on port 8080
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 60 app:app
