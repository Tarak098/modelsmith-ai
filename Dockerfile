# Use official slim Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WORKSPACE_DIR=/app

# Set working directory inside the container
WORKDIR /app

# Copy dependency requirements
COPY auto_ai/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application codebase and assets
COPY . .

# Expose port (Cloud Run sets PORT environment variable dynamically)
EXPOSE 8080

# Run FastAPI app using uvicorn
CMD exec uvicorn auto_ai.app.main:app --host 0.0.0.0 --port ${PORT:-8080}
