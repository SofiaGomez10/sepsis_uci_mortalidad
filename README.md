# Estratificación de riesgo de mortalidad en UCI por sepsis

Tercer entregable del proyecto de Inteligencia Artificial (UAO, 2026-2).
Sofia Gomez Rodriguez · Emmanuel Escobar Santos · Victor Andres Martinez Preciado

Aplicación web que estima, con datos del ingreso a UCI, la probabilidad de que un adulto con sepsis fallezca
durante su estancia, la ubica en un estrato de riesgo y muestra qué variables sustentan la estimación.

> Prototipo académico de apoyo a la decisión. No es un dispositivo médico ni sustituye el juicio clínico.

## Qué contiene

| Ruta | Contenido |
|---|---|
| `app.py` | Aplicación Streamlit (tres pestañas: evaluación, desempeño, ficha técnica) |
| `src/modelo.py` | Inferencia: validación de entradas (todas obligatorias), predicción calibrada y explicación por paciente |
| `src/estilo.py` | Hoja de estilos y componentes visuales |
| `notebooks/Entrega3_modelo_final.ipynb` | Cuaderno comentado: entrena, evalúa y exporta el modelo (se abre en Colab) |
| `notebooks/entrega3_modelo_final.py` | El mismo cuaderno en texto plano (jupytext), para revisar cambios en Git |
| `models/modelo_sepsis.joblib` | Artefacto: pipeline ajustado, calibración, cortes, rangos y métricas |
| `models/metricas.json` | Métricas del modelo en formato legible |
| `reports/figuras/` | Figuras de resultados generadas por el cuaderno |
| `tests/test_app.py` | Pruebas automáticas del artefacto y la inferencia |
| `Dockerfile` | Imagen para desplegar dentro de la red de una institución |

## Arquitectura

```
ENTRENAMIENTO (fuera de línea, cuaderno)               SERVICIO (aplicación)
CSV  ->  partición 75/25  ->  Pipeline                  formulario -> validación -> modelo -> calibración
         [escalado + imputación KNN + one-hot           -> probabilidad + estrato + contribuciones
          + regresión logística L1]
     ->  calibración de Platt y cortes (validación
         cruzada)
     ->  evaluación en prueba
     ->  modelo_sepsis.joblib + metricas.json
```

La imputación KNN es un paso de entrenamiento (el conjunto de datos trae faltantes). En el servicio está
desactivada: todos los campos son obligatorios, salvo la presión arterial media, que se calcula si se deja vacía.
El código anterior queda comentado en `src/modelo.py` y `app.py` bajo el rótulo "Imputación en producción: DESACTIVADA".

La aplicación no entrena ni recalcula nada: carga un único artefacto.

## Ejecutar en local

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Se abre en `http://localhost:8501`. Requiere Python 3.13 (las versiones están fijadas en `requirements.txt` porque
el artefacto debe abrirse con la misma versión de scikit-learn con la que se guardó).

## Desplegar en Streamlit Community Cloud

1. Suba esta carpeta a un repositorio de GitHub (el CSV no se sube: está en `.gitignore`).
2. Entre a <https://share.streamlit.io>, inicie sesión con GitHub y elija "Create app".
3. Seleccione el repositorio, la rama `main` y el archivo principal `app.py`.
4. En "Advanced settings" elija **Python 3.13**. La versión de Python no se puede cambiar después sin borrar y
   volver a crear la app.
5. "Deploy". En pocos minutos queda disponible en una dirección `https://<nombre>.streamlit.app`.

Cada `git push` a la rama vuelve a desplegar la aplicación automáticamente.

## Reentrenar

```bash
pip install -r requirements-dev.txt
# copie CriticalData_Sepsis.csv en data/
jupyter execute notebooks/Entrega3_modelo_final.ipynb      # o ábralo en Colab y ejecute todo
pytest -q
```

El cuaderno regenera `models/` y `reports/figuras/`. Si cambian las variables o sus nombres, actualice las etiquetas
en `src/modelo.py`.

## Resultados del modelo (conjunto de prueba, 2.333 pacientes)

| Métrica | Valor |
|---|---|
| AUC-ROC | 0,594 (IC 95 %: 0,571 a 0,617) |
| AUC-ROC en validación cruzada | 0,582 |
| AUC-ROC del SOFA al ingreso | 0,564 |
| AUPRC | 0,566 (prevalencia 0,465) |
| Brier | 0,240 (referencia 0,249) |
| Mortalidad observada por estrato | 40,4 % · 43,8 % · 63,6 % |

## Límites

Discriminación baja; un único conjunto de datos de procedencia no documentada (posiblemente sintético); sin
validación externa; solo usa información del ingreso. Ver la pestaña "Ficha técnica" de la aplicación.
