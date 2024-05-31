FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy only the necessary files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port $PORT"]
