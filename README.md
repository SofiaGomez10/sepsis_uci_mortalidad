# Estratificación de riesgo de mortalidad en UCI por sepsis

Tercer entregable del proyecto de Inteligencia Artificial (UAO, 2026-2).
Sofia Gomez Rodriguez · Emmanuel Escobar Santos · Victor Andres Martinez Preciado

Aplicación web que estima, con datos del ingreso a UCI, la probabilidad de que un adulto con sepsis fallezca
durante su estancia, la ubica en un estrato de riesgo, explica qué variables sustentan la estimación e
incluye un asistente que responde preguntas del personal médico sobre el modelo y explica, con los valores del paciente evaluado, cómo influyó cada dato en su resultado.

> Prototipo académico de apoyo a la decisión. No es un dispositivo médico ni sustituye el juicio clínico.

## Qué contiene

| Ruta | Contenido |
|---|---|
| `app.py` | Aplicación Streamlit (cuatro pestañas: evaluación, asistente, desempeño, ficha técnica) |
| `src/modelo.py` | Inferencia: validación de entradas (todas obligatorias), predicción calibrada y explicación por paciente |
| `src/asistente.py` | Asistente tipo RAG: base de conocimiento, recuperación TF-IDF y generación opcional con LLM |
| `src/estilo.py` | Hoja de estilos y componentes visuales |
| `kb/conceptos.md` | Glosario para personal clínico (coeficiente, odds ratio, AUC, calibración, etc.) |
| `notebooks/Entrega3_modelo_final.ipynb` | Cuaderno comentado: entrena, evalúa y exporta el modelo (se abre en Colab) |
| `notebooks/entrega3_modelo_final.py` | El mismo cuaderno en texto plano (jupytext), para revisar cambios en Git |
| `models/modelo_sepsis.joblib` | Artefacto: pipeline ajustado, calibración, cortes, rangos y métricas |
| `models/metricas.json` | Métricas del modelo en formato legible |
| `reports/figuras/` | Figuras de resultados generadas por el cuaderno |
| `tests/test_app.py` | Pruebas automáticas del artefacto, la inferencia y el asistente |
| `Dockerfile` | Imagen para desplegar dentro de la red de una institución |

## Arquitectura

```
ENTRENAMIENTO (fuera de línea, cuaderno)               SERVICIO (aplicación)
CSV  ->  partición 75/25  ->  Pipeline                  formulario  ->  campos obligatorios, rangos y coherencia
         [escalado + imputación KNN + one-hot           ->  escalado + one-hot + logística  ->  calibración 
          + regresión logística L1]                     ->  probabilidad + estrato + contribuciones
     ->  calibración de Platt y cortes (validación
         cruzada)                                       pregunta  ->  recuperación TF-IDF en la base de
     ->  evaluación en prueba                                         conocimiento (glosario + ficha generada
     ->  modelo_sepsis.joblib  + metricas.json  ------>               desde el artefacto)  ->  respuesta
```

La imputación KNN es un paso de entrenamiento (el conjunto de datos trae faltantes). En el servicio está
desactivada: todos los campos son obligatorios, salvo la presión arterial media, que se calcula si se deja vacía.
El código anterior queda comentado en `src/modelo.py` y `app.py` bajo el rótulo "Imputación en producción: DESACTIVADA".

La aplicación no entrena ni recalcula nada: carga un único artefacto. Las cifras que muestra la interfaz y las que
cita el asistente salen de ese mismo archivo, por lo que no pueden contradecirse.

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
5. (Opcional) En "Secrets" pegue el contenido de `.streamlit/secrets.toml.example` con una clave real para
   activar el modo generativo del asistente.
6. "Deploy". En pocos minutos queda disponible en una dirección `https://<nombre>.streamlit.app`.

Cada `git push` a la rama vuelve a desplegar la aplicación automáticamente.

## Asistente de interpretación

* **Modo local (por defecto):** recupera el fragmento más pertinente de la base de conocimiento. No usa servicios
  externos, no tiene costo y no puede inventar cifras.
* **Modo generativo (opcional):** si se configuran `LLM_API_KEY`, `LLM_BASE_URL` y `LLM_MODEL`, un modelo de lenguaje
  redacta la respuesta usando solo los fragmentos recuperados. Funciona con cualquier servicio compatible con la API
  de OpenAI (`/chat/completions`). Los datos del paciente evaluado solo se envían si el usuario marca la casilla.

* **Preguntas sobre el paciente evaluado:** si ya se calculó un riesgo, el asistente explica con los valores de ese
  paciente cómo influyó cada dato. Ejemplo: "¿Por qué el lactato dio así en este paciente?" devuelve el valor digitado,
  el promedio de la cohorte, los puntos porcentuales que aporta, su lugar entre los 11 factores y la explicación general
  del coeficiente. Sin variable nombrada ("¿por qué le sale ese riesgo?") da el resumen de todos los factores. Esta parte
  se calcula siempre en local (`explicar_variable` y `resumen_paciente` en `src/asistente.py`), también en modo generativo.

Para ampliar lo que sabe el asistente, agregue secciones `## Título` con una línea `claves: ...` en `kb/conceptos.md`.

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
