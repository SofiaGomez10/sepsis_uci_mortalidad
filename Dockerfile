# Imagen para desplegar la aplicación dentro de la red de una institución (sin salida a internet).
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY src ./src
COPY kb ./kb
COPY models ./models
COPY reports ./reports
COPY .streamlit/config.toml ./.streamlit/config.toml
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
