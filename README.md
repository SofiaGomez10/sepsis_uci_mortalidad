# Estratificación de riesgo de mortalidad en UCI por sepsis

Tercer entregable del proyecto de Inteligencia Artificial (Universidad Autónoma de Occidente, 2026-2).

**Integrantes:** Sofía Gómez Rodríguez, Emmanuel Escobar Santos y Víctor Andrés Martínez Preciado.

Aplicación web que, a partir de los datos del ingreso a UCI, estima la probabilidad de que un adulto con sepsis fallezca durante su estancia, lo ubica en un estrato de riesgo y muestra qué variables explican la estimación.

> Prototipo académico de apoyo a la decisión clínica. No es un dispositivo médico ni sustituye el juicio del profesional de salud.

## Contenido del repositorio

| Ruta | Contenido |
|---|---|
| `app.py` | Aplicación Streamlit con tres pestañas: evaluación del paciente, desempeño del modelo y ficha técnica. |
| `src/modelo.py` | Validación de los datos ingresados, predicción calibrada y explicación por paciente. |
| `src/estilo.py` | Estilos y componentes visuales de la aplicación. |
| `notebooks/Entrega3_modelo_final.ipynb` | Cuaderno comentado que entrena, evalúa y exporta el modelo. Puede abrirse en Colab. |
| `models/modelo_sepsis.joblib` | Modelo entrenado: pipeline ajustado, calibración, puntos de corte, rangos válidos y métricas. |
| `models/metricas.json` | Métricas del modelo en formato legible. |
| `reports/figuras/` | Figuras de resultados generadas por el cuaderno. |
| `tests/test_app.py` | Pruebas automáticas del modelo y de la predicción. |
| `data/` | Ubicación del conjunto de datos, que no se incluye en el repositorio (ver `data/README.md`). |
| `Dockerfile` | Instrucciones para construir la imagen de la aplicación. |
| `docker-compose.yml` | Configuración para ejecutar la aplicación en un contenedor con un solo comando. |

## Arquitectura

El proyecto se divide en dos etapas independientes, conectadas por un único archivo: `models/modelo_sepsis.joblib`.

**1. Entrenamiento (cuaderno).**
1. Se cargan los datos y se dividen en entrenamiento (75 %) y prueba (25 %).
2. Se ajusta un pipeline que integra el escalado, la imputación KNN, la codificación del foco de infección y la regresión logística con regularización L1.
3. Se calibran las probabilidades con el método de Platt y se definen los puntos de corte de los estratos, usando validación cruzada sobre el entrenamiento.
4. Se evalúa el modelo en el conjunto de prueba.
5. Se exportan el modelo y sus métricas a la carpeta `models/`.

**2. Servicio (aplicación).**
1. El médico ingresa los datos del paciente en el formulario.
2. Se validan los datos: campos obligatorios, rangos y coherencia clínica.
3. El modelo calcula el riesgo calibrado, el estrato y la contribución de cada variable.

La imputación KNN se utiliza únicamente durante el entrenamiento, porque el conjunto de datos contiene valores faltantes. En la aplicación todos los campos son obligatorios; la única excepción es la presión arterial media, que se calcula a partir de la sistólica y la diastólica si se deja vacía. La versión anterior, que sí imputaba, se conserva comentada en `src/modelo.py` y `app.py`.

La aplicación no entrena ni recalcula nada: solo carga el modelo exportado.

## Ejecución en local

Requiere Python 3.13. Las versiones de las librerías están fijadas en `requirements.txt`, porque el modelo debe abrirse con la misma versión de scikit-learn con la que se guardó.

**Opción 1. Con Docker**

```bash
docker compose up -d
```

**Opción 2. Con Python**

```bash
python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

En ambos casos, la aplicación queda disponible en `http://localhost:8501`.

**Pruebas automáticas**

```bash
pip install -r requirements-dev.txt
pytest -q
```

**Reentrenamiento.** Para volver a entrenar el modelo, copie el conjunto de datos en `data/CriticalData_Sepsis.csv` y ejecute el cuaderno. Esto regenera los archivos de `models/` y `reports/figuras/`.

## Despliegue en Streamlit Community Cloud

1. Suba el repositorio a GitHub.
2. Ingrese a <https://share.streamlit.io>, inicie sesión con GitHub y seleccione "Create app".
3. Elija el repositorio, la rama `main` y el archivo principal `app.py`.
4. En "Advanced settings", seleccione Python 3.13.
5. Pulse "Deploy". En pocos minutos la aplicación queda disponible en una dirección del tipo `https://<nombre>.streamlit.app`.

Cada `git push` a la rama `main` actualiza automáticamente la aplicación desplegada.

## Resultados del modelo

Evaluación en el conjunto de prueba (2.333 pacientes que el modelo no vio durante el entrenamiento).

| Métrica | Valor | Interpretación |
|---|---|---|
| AUC-ROC | 0,594 (IC 95 %: 0,571 a 0,617) | Probabilidad de que el modelo asigne mayor riesgo a un paciente que fallece que a uno que sobrevive. Un valor de 0,5 equivale al azar y 1,0 a una discriminación perfecta. |
| AUC-ROC en validación cruzada | 0,582 | Mismo indicador, promediado en distintas particiones del entrenamiento. Su similitud con el valor en prueba indica que el resultado es estable. |
| AUC-ROC del SOFA al ingreso | 0,564 | Desempeño de la escala SOFA en los mismos pacientes. Es la referencia clínica con la que se compara el modelo. |
| AUPRC | 0,566 (azar: 0,465) | Capacidad de identificar a los pacientes que fallecen sin generar demasiadas falsas alarmas. El valor de referencia es la proporción de fallecidos. |
| Puntaje de Brier | 0,240 (referencia: 0,249) | Error de las probabilidades predichas; cuanto menor, mejor. La referencia corresponde a predecir siempre la mortalidad promedio. |
| Mortalidad observada por estrato | 40,4 % · 43,8 % · 63,6 % | Proporción de pacientes que fallecieron en los estratos de menor, intermedio y mayor riesgo. Que aumente de un estrato a otro indica que la estratificación es coherente. |