FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port $PORT"]
