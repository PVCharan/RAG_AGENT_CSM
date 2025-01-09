# Use a lightweight Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install essential system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the port Flask runs on
EXPOSE 8080

# Run the Flask application
CMD ["python", "app.py"]
