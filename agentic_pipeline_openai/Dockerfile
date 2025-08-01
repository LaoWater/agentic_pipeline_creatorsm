# Use Python 3.12 base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose port 8000 (to match local setup)
EXPOSE 8000

# Run the FastAPI app (match your local command)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]