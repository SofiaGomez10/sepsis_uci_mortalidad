# Imagen de la aplicación de riesgo de mortalidad en sepsis.
# Construir y ejecutar: docker compose up --build  

# Imagen base con Python 3.13, la misma versión con la que se guardó el modelo.
FROM python:3.13-slim
WORKDIR /app

# Se instalan primero las dependencias, para que Docker las reutilice si solo cambia el código.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Se copian solo los archivos que necesita la aplicación. No se incluyen los datos ni las pruebas.
COPY app.py ./
COPY src ./src
COPY models ./models
COPY reports ./reports
COPY .streamlit/config.toml ./.streamlit/config.toml

# Puerto en el que escucha Streamlit.
EXPOSE 8501

# Comprobación periódica de que la aplicación responde.
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Arranque de la aplicación, accesible desde fuera del contenedor.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]