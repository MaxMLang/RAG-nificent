FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/pdf_data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose the port that fly.io expects
EXPOSE 8080

# Create a non-root user to run the application
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Command to run the application - explicitly binding to 0.0.0.0:8080
CMD ["chainlit", "run", "src/app.py", "--host", "0.0.0.0", "--port", "8080"] 