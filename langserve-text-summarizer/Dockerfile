FROM python:3.12.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY chain.py app.py ./
COPY tests/ ./tests/

# Make port 8000 available to the world outside this container
EXPOSE 8080

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]