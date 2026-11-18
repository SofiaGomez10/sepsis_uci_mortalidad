"""Inferencia: carga del artefacto, validación de entradas, predicción y explicación por paciente.

Este módulo no entrena nada. Todo lo que usa (pipeline, calibración, cortes, rangos, métricas)
viene del artefacto que exporta `notebooks/Entrega3_modelo_final.ipynb`.
"""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RUTA_ARTEFACTO = Path(__file__).resolve().parent.parent / "models" / "modelo_sepsis.joblib"

# Nombre clínico, unidad y ayuda de cada variable (lo que ve el médico en pantalla).
VARIABLES = {
    "Age": {"corta": "Edad (años)", "etiqueta": "Edad", "unidad": "años", "ayuda": "Edad del paciente al ingreso a UCI.", "paso": 1.0, "decimales": 0},
    "FrailtyScore": {"corta": "Fragilidad (1-7)", "etiqueta": "Fragilidad clínica", "unidad": "de 7", "ayuda": "Escala de fragilidad clínica previa al episodio: 1 = robusto, 7 = fragilidad severa.", "paso": 1.0, "decimales": 0},
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
ETIQUETA_FOCO = {"abdominal": "Abdominal", "desconocido": "Desconocido", "piel": "Piel y tejidos blandos", "pulmonar": "Pulmonar", "urinaria": "Urinario"}


def sigmoide(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def cargar(ruta: Path | str = RUTA_ARTEFACTO) -> dict:
    """Carga el artefacto exportado por el cuaderno de entrenamiento."""
    return joblib.load(ruta)


def etiqueta(variable: str) -> str:
    return VARIABLES.get(variable, {}).get("etiqueta", variable)


def minuscula(texto: str) -> str:
    """Pasa a minúscula la etiqueta, salvo las siglas (INR, SOFA)."""
    return texto if texto.isupper() else texto[0].lower() + texto[1:]


def completar_pam(valores: dict) -> tuple[dict, bool]:
    """Si falta la PAM y hay sistólica y diastólica, la calcula con la fórmula clínica habitual."""
    v = dict(valores)
    if v.get("MAP_0") is None and v.get("SBP_0") is not None and v.get("DBP_0") is not None:
        v["MAP_0"] = round((v["SBP_0"] + 2 * v["DBP_0"]) / 3, 1)
        return v, True
    return v, False


def validar(art: dict, valores: dict) -> list[dict]:
    """Revisa las entradas contra los rangos de entrenamiento y contra reglas de coherencia clínica.

    Devuelve avisos con nivel 'error' (impide calcular) o 'aviso' (se calcula, pero se advierte)."""
    avisos = []
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
    pas, pad, pam = valores.get("SBP_0"), valores.get("DBP_0"), valores.get("MAP_0")
    if pas is not None and pad is not None and pad >= pas:
        avisos.append({"nivel": "error", "texto": "La presión diastólica debe ser menor que la sistólica."})
    if pas is not None and pad is not None and pam is not None and not (pad - 2 <= pam <= pas + 2):
        avisos.append({"nivel": "aviso", "texto": "La presión arterial media no está entre la diastólica y la sistólica. Verifique los valores."})
    if valores.get("SepticShock") == 1 and valores.get("VasopressorUse_0h") == 0:
        avisos.append({"nivel": "aviso", "texto": (
            "Se registró choque séptico sin vasopresor. Por definición (Sepsis-3) el choque séptico requiere "
            "vasopresor; en los datos de entrenamiento todos los pacientes con choque lo recibían.")})
    # En producción todos los campos son obligatorios: la imputación se usó solo para entrenar.
    # (La presión arterial media es la única excepción lógica: si se deja vacía, completar_pam() la calcula antes.)
    faltan = [etiqueta(v) for v in art["variables"] if valores.get(v) is None]
    if faltan:
        avisos.append({"nivel": "error", "texto": "Faltan datos obligatorios: " + ", ".join(minuscula(x) for x in faltan)
                       + ". El modelo no estima valores ausentes; registre el dato para calcular."})
    # --- Imputación en producción: DESACTIVADA -------------------------------------------------------
    # Antes se permitía calcular con campos vacíos y solo se advertía:
    # n_faltan = sum(valores.get(v) is None for v in art["variables"])
    # if n_faltan > 4:
    #     avisos.append({"nivel": "aviso", "texto": f"Faltan {n_faltan} de 11 variables. La estimación se apoya sobre todo en valores imputados."})
    return avisos


def predecir(art: dict, valores: dict) -> dict:
    """Calcula el riesgo calibrado, el estrato y la contribución de cada variable para un paciente.

    `valores` es un diccionario variable -> valor y debe venir COMPLETO. La imputación KNN del pipeline
    se usó para entrenar (el conjunto de datos traía faltantes); en producción no se imputa nada."""
    faltan = [v for v in art["variables"] if valores.get(v) is None]
    if faltan:
        raise ValueError(f"Faltan datos obligatorios: {faltan}. En producción no se imputan valores.")
    pipe = art["pipeline"]
    fila = pd.DataFrame([{v: valores[v] for v in art["variables"]}])[art["variables"]]
    for c in art["cols_num"]:
        fila[c] = fila[c].astype(float)
    fila[art["cols_cat"]] = fila[art["cols_cat"]].astype(object)

    # Se aplican los pasos ya ajustados del pipeline SALTANDO el imputador: escalado + one-hot + logística.
    pre, logit = pipe.named_steps["pre"], pipe.named_steps["modelo"]
    nombres = art["nombres_transformados"]
    escalador = pre.named_transformers_["num"].named_steps["escalar"]
    onehot = pre.named_transformers_["cat"].named_steps["onehot"]
    xt = pd.Series(np.concatenate([escalador.transform(fila[art["cols_num"]])[0],
                                   onehot.transform(fila[art["cols_cat"]].values)[0]]), index=nombres)
    z = float(logit.decision_function(xt.to_frame().T.values)[0])    # log-odds del modelo
    # --- Imputación en producción: DESACTIVADA -------------------------------------------------------
    # El flujo anterior pasaba por el pipeline completo, que incluye el KNNImputer y rellenaba los vacíos:
    # z = float(pipe.decision_function(fila)[0])
    # xt = pd.Series(pre.transform(fila)[0], index=nombres)
    a, b = art["calibracion"]["a"], art["calibracion"]["b"]
    riesgo = sigmoide(a * z + b)                                      # probabilidad calibrada (Platt)

    # Explicación exacta de un modelo lineal: coef x (valor estandarizado - promedio de entrenamiento).
    fondo = pd.Series(art["fondo"])[nombres]
    contrib = pd.Series(logit.coef_[0], index=nombres) * (xt - fondo)

    usado = {c: float(valores[c]) for c in art["cols_num"]}         # sin imputación, el valor usado es el digitado

    filas = []
    for c in art["cols_num"]:
        i = art["cols_num"].index(c)
        filas.append({"variable": c, "etiqueta": etiqueta(c), "contribucion": float(contrib[c]),
                      "valor": valores.get(c), "valor_usado": usado[c],
                      # Datos para que el asistente explique cada variable con los valores del paciente:
                      "promedio": float(escalador.mean_[i]),          # promedio de la cohorte (unidades originales)
                      "z": float(xt[c]),                              # cuántas desviaciones estándar se aleja del promedio
                      "coef": float(logit.coef_[0][nombres.index(c)])})  # coeficiente por desviación estándar
    foco = valores.get("InfectionSource")
    c_foco = float(sum(contrib[n] for n in nombres if n.startswith("InfectionSource_")))
    filas.append({"variable": "InfectionSource", "etiqueta": etiqueta("InfectionSource"), "contribucion": c_foco,
                  "valor": foco, "valor_usado": foco, "promedio": None, "z": None, "coef": None})
    for f in filas:
        # Efecto en puntos porcentuales: riesgo actual menos el riesgo si esa variable estuviera en el promedio.
        f["efecto_pp"] = (riesgo - sigmoide(a * (z - f["contribucion"]) + b)) * 100
    filas.sort(key=lambda f: abs(f["contribucion"]), reverse=True)
    for i, f in enumerate(filas, 1):
        f["puesto"] = i                                              # lugar por magnitud del aporte en este paciente

    cortes = art["cortes"]
    estrato = art["estratos"][2] if riesgo >= cortes["alto"] else art["estratos"][1] if riesgo >= cortes["bajo"] else art["estratos"][0]
    return {"riesgo": riesgo, "log_odds": z, "estrato": estrato, "contribuciones": filas,
            "riesgo_base": sigmoide(a * art["valor_base"] + b), "valores": dict(valores),
            # "imputadas": [f["etiqueta"] for f in filas if f["imputado"]],     # imputación desactivada en producción
            "suma_ok": bool(np.isclose(art["valor_base"] + float(contrib.sum()), z))}


def formato_valor(variable: str, valor) -> str:
    """Texto legible del valor de una variable."""
    if valor is None:
        return "sin dato"
    if variable in ("SepticShock", "VasopressorUse_0h"):
        if valor in (0, 1):
            return "sí" if valor == 1 else "no"
        return f"probabilidad estimada {valor:.0%}"
    if variable == "InfectionSource":
        return ETIQUETA_FOCO.get(valor, str(valor)).lower()
    meta = VARIABLES[variable]
    return f"{valor:.{meta['decimales']}f} {meta['unidad']}".strip().replace(".", ",")
