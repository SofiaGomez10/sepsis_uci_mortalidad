"""Predicción del riesgo para un paciente a partir del modelo ya entrenado.

Este módulo no entrena el modelo: carga el archivo generado por el cuaderno
`notebooks/Entrega3_modelo_final.ipynb`, valida los datos ingresados y calcula
el riesgo, el estrato y la contribución de cada variable.
"""

from __future__ import annotations

import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

RUTA_ARTEFACTO = Path(__file__).resolve().parent.parent / "models" / "modelo_sepsis.joblib"

# Nombre clínico, unidad, ayuda y formato de cada variable, tal como se muestran en la aplicación.
VARIABLES = {
    "Age": {"corta": "Edad (años)", "etiqueta": "Edad", "unidad": "años", "ayuda": "Edad del paciente al ingreso a UCI.", "paso": 1.0, "decimales": 0},
    "FrailtyScore": {"corta": "Fragilidad (1-9)", "etiqueta": "Fragilidad clínica", "unidad": "de 9", "ayuda": "Escala de Fragilidad Clínica previa al episodio: 1 = muy en forma, 9 = enfermedad terminal. Los datos de entrenamiento solo incluyen valores de 1 a 7.", "paso": 1.0, "decimales": 0},
    "SBP_0": {"corta": "PA sistólica (mmHg)", "etiqueta": "Presión arterial sistólica", "unidad": "mmHg", "ayuda": "Primera medición al ingreso.", "paso": 1.0, "decimales": 0},
    "DBP_0": {"corta": "PA diastólica (mmHg)", "etiqueta": "Presión arterial diastólica", "unidad": "mmHg", "ayuda": "Primera medición al ingreso.", "paso": 1.0, "decimales": 0},
    "MAP_0": {"corta": "PA media (mmHg)", "etiqueta": "Presión arterial media", "unidad": "mmHg", "ayuda": "Si se deja vacía se calcula como (PAS + 2 x PAD) / 3.", "paso": 1.0, "decimales": 0},
    "Lactate_0h": {"corta": "Lactato (mmol/L)", "etiqueta": "Lactato", "unidad": "mmol/L", "ayuda": "Lactato sérico al ingreso.", "paso": 0.1, "decimales": 1},
    "INR_0h": {"corta": "INR", "etiqueta": "INR", "unidad": "", "ayuda": "Razón internacional normalizada al ingreso.", "paso": 0.1, "decimales": 2},
    "SOFA_0h": {"corta": "SOFA (puntos)", "etiqueta": "SOFA", "unidad": "puntos", "ayuda": "Puntaje SOFA al ingreso.", "paso": 1.0, "decimales": 0},
    "SepticShock": {"etiqueta": "Choque séptico", "unidad": "", "ayuda": "Hipotensión que requiere vasopresor para mantener PAM de 65 mmHg o más, con lactato mayor de 2 mmol/L pese a reanimación (Sepsis-3)."},
    "VasopressorUse_0h": {"etiqueta": "Vasopresor al ingreso", "unidad": "", "ayuda": "Recibe vasopresor en la hora 0."},
    "InfectionSource": {"etiqueta": "Foco de infección", "unidad": "", "ayuda": "Foco identificado al ingreso. Use 'desconocido' si aún no se ha determinado."},
}
# Nombre en pantalla de cada categoría del foco de infección.
ETIQUETA_FOCO = {"abdominal": "Abdominal", "desconocido": "Desconocido", "piel": "Piel y tejidos blandos", "pulmonar": "Pulmonar", "urinaria": "Urinario"}


def sigmoide(z: float) -> float:
    """Convierte un puntaje (log-odds) en una probabilidad entre 0 y 1."""
    return 1.0 / (1.0 + math.exp(-z))


def cargar(ruta: Path | str = RUTA_ARTEFACTO) -> dict:
    """Carga el artefacto exportado por el cuaderno de entrenamiento."""
    return joblib.load(ruta)

def etiqueta(variable: str) -> str:
    """Devuelve el nombre clínico de una variable (por ejemplo, "Lactate_0h" pasa a "Lactato")."""
    return VARIABLES.get(variable, {}).get("etiqueta", variable)


def minuscula(texto: str) -> str:
    """Pasa a minúscula la primera letra, salvo en siglas como INR o SOFA."""
    return texto if texto.isupper() else texto[0].lower() + texto[1:]


def completar_pam(valores: dict) -> tuple[dict, bool]:
    """Si falta la presión arterial media, la calcula como (sistólica + 2 × diastólica) / 3.
    Devuelve los valores y si fue necesario calcularla."""
    v = dict(valores)
    if v.get("MAP_0") is None and v.get("SBP_0") is not None and v.get("DBP_0") is not None:
        v["MAP_0"] = round((v["SBP_0"] + 2 * v["DBP_0"]) / 3, 1)
        return v, True
    return v, False


def validar(art: dict, valores: dict) -> list[dict]:
    """Revisa los datos ingresados antes de calcular el riesgo.

    Devuelve una lista de mensajes: los de nivel 'error' impiden calcular,
    y los de nivel 'aviso' permiten calcular pero advierten al médico."""
    avisos = []

    # 1 y 2.Valores negativos o fuera del rango observado en los datos de entrenamiento.
    for var, r in art["rangos"].items():
        x = valores.get(var)
        if x is None or var in art["cols_bin"]:
            continue
        if x < 0:
            avisos.append({"nivel": "error", "texto": f"{etiqueta(var)}: no puede ser negativo."})
        elif x < r["min"] or x > r["max"]:
            avisos.append({"nivel": "aviso", "texto": (
                f"{etiqueta(var)} = {x:g} está fuera del rango con el que se entrenó el modelo "
                f"({r['min']:g} a {r['max']:g}). El modelo extrapola y la estimación es menos confiable.")})

    # 3 y 4.Coherencia entre las presiones arteriales.
    pas, pad, pam = valores.get("SBP_0"), valores.get("DBP_0"), valores.get("MAP_0")
    if pas is not None and pad is not None and pad >= pas:
        avisos.append({"nivel": "error", "texto": "La presión diastólica debe ser menor que la sistólica."})
    if pas is not None and pad is not None and pam is not None and not (pad - 2 <= pam <= pas + 2):
        avisos.append({"nivel": "aviso", "texto": "La presión arterial media no está entre la diastólica y la sistólica. Verifique los valores."})

    # 5.Coherencia clínica: el choque séptico requiere vasopresor (Sepsis-3).
    if valores.get("SepticShock") == 1 and valores.get("VasopressorUse_0h") == 0:
        avisos.append({"nivel": "aviso", "texto": (
            "Se registró choque séptico sin vasopresor. Por definición (Sepsis-3) el choque séptico requiere "
            "vasopresor; en los datos de entrenamiento todos los pacientes con choque lo recibían.")})

    # 6.Todos los campos son obligatorios: el modelo no estima valores ausentes.
    # La presión arterial media es la única excepción, porque completar_pam() la calcula antes.
    faltan = [etiqueta(v) for v in art["variables"] if valores.get(v) is None]
    if faltan:
        avisos.append({"nivel": "error", "texto": "Faltan datos obligatorios: " + ", ".join(minuscula(x) for x in faltan)
                       + ". El modelo no estima valores ausentes; registre el dato para calcular."})

    # Versión anterior, cambiada permitía calcular con campos vacíos (imputándolos)
    # y solo advertía si faltaban más de 4 variables.
    # n_faltan = sum(valores.get(v) is None for v in art["variables"])
    # if n_faltan > 4:
    #     avisos.append({"nivel": "aviso", "texto": f"Faltan {n_faltan} de 11 variables. La estimación se apoya sobre todo en valores imputados."})

    return avisos


def predecir(art: dict, valores: dict) -> dict:
    """Calcula el riesgo calibrado, el estrato y la contribución de cada variable para un paciente.

    "valores" es un diccionario con el valor de cada variable y debe estar completo.
    La imputación se usó solo para entrenar; en la aplicación no se estiman valores ausentes."""

    # 1.Se rechaza el cálculo si falta algún dato.
    faltan = [v for v in art["variables"] if valores.get(v) is None]
    if faltan:
        raise ValueError(f"Faltan datos obligatorios: {faltan}.")

    # 2.Se organizan los datos del paciente en una tabla con el formato que espera el modelo.
    pipe = art["pipeline"]
    fila = pd.DataFrame([{v: valores[v] for v in art["variables"]}])[art["variables"]]
    for c in art["cols_num"]:
        fila[c] = fila[c].astype(float)
    fila[art["cols_cat"]] = fila[art["cols_cat"]].astype(object)

    # 3.Se aplican el escalado, la codificación del foco y la regresión logística ya ajustados,
    #omitiendo el imputador. El resultado es el puntaje del modelo (log-odds).
    pre, logit = pipe.named_steps["pre"], pipe.named_steps["modelo"]
    nombres = art["nombres_transformados"]
    escalador = pre.named_transformers_["num"].named_steps["escalar"]
    onehot = pre.named_transformers_["cat"].named_steps["onehot"]
    xt = pd.Series(np.concatenate([escalador.transform(fila[art["cols_num"]])[0],
                                   onehot.transform(fila[art["cols_cat"]].values)[0]]), index=nombres)
    z = float(logit.decision_function(xt.to_frame().T.values)[0])

    # Versión anterior, cambiada: usaba el pipeline completo, que incluye el imputador.
    # z = float(pipe.decision_function(fila)[0])
    # xt = pd.Series(pre.transform(fila)[0], index=nombres)

    # 4.Se convierte el puntaje en una probabilidad calibrada con el método de Platt.
    a, b = art["calibracion"]["a"], art["calibracion"]["b"]
    riesgo = sigmoide(a * z + b)

    # 5.Contribución de cada variable: su coeficiente multiplicado por la diferencia
    # entre el valor del paciente y el promedio de entrenamiento.
    fondo = pd.Series(art["fondo"])[nombres]
    contrib = pd.Series(logit.coef_[0], index=nombres) * (xt - fondo)
    usado = {c: float(valores[c]) for c in art["cols_num"]}

    filas = []
    for c in art["cols_num"]:
        filas.append({"variable": c, "etiqueta": etiqueta(c), "contribucion": float(contrib[c]),
                      "valor": valores.get(c), "valor_usado": usado[c]})

    # El foco de infección se codificó en varias columnas; sus contribuciones se suman en una sola.
    foco = valores.get("InfectionSource")
    c_foco = float(sum(contrib[n] for n in nombres if n.startswith("InfectionSource_")))
    filas.append({"variable": "InfectionSource", "etiqueta": etiqueta("InfectionSource"), "contribucion": c_foco,
                  "valor": foco, "valor_usado": foco})

    # 6.Efecto de cada variable en puntos porcentuales: diferencia entre el riesgo del paciente
    #   y el riesgo que tendría si esa variable estuviera en el promedio.
    for f in filas:
        f["efecto_pp"] = (riesgo - sigmoide(a * (z - f["contribucion"]) + b)) * 100
    filas.sort(key=lambda f: abs(f["contribucion"]), reverse=True)

    # 7.Se asigna el estrato según los puntos de corte.
    cortes = art["cortes"]
    estrato = art["estratos"][2] if riesgo >= cortes["alto"] else art["estratos"][1] if riesgo >= cortes["bajo"] else art["estratos"][0]

    # suma_ok verifica que el valor base más las contribuciones reproduzca exactamente el puntaje.
    return {"riesgo": riesgo, "log_odds": z, "estrato": estrato, "contribuciones": filas,
            "riesgo_base": sigmoide(a * art["valor_base"] + b), "valores": dict(valores),
            "suma_ok": bool(np.isclose(art["valor_base"] + float(contrib.sum()), z))}


def formato_valor(variable: str, valor) -> str:
    """Convierte el valor de una variable en texto legible para la pantalla.
    Por ejemplo: 1 en "sí", "urinaria" en "urinario" y 4.8 en "4,8 mmol/L"."""
    if valor is None:
        return "sin dato"

    # Variables binarias: se muestran como sí o no.
    if variable in ("SepticShock", "VasopressorUse_0h"):
        return "sí" if valor == 1 else "no"

    # Foco de infección: se muestra con su nombre clínico.
    if variable == "InfectionSource":
        return ETIQUETA_FOCO.get(valor, str(valor)).lower()

    # Variables numéricas: con sus decimales, su unidad y coma decimal.
    meta = VARIABLES[variable]
    return f"{valor:.{meta['decimales']}f} {meta['unidad']}".strip().replace(".", ",")
