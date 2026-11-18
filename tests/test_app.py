"""Pruebas del artefacto, la inferencia y el asistente.  Ejecutar:  pytest -q"""
import json
from pathlib import Path

import pytest

from src import asistente as A
from src import modelo as M

RAIZ = Path(__file__).resolve().parent.parent
BASE = {"InfectionSource": "abdominal", "SepticShock": 0, "VasopressorUse_0h": 0, "MAP_0": 75, "SOFA_0h": 2, "DBP_0": 60,
        "SBP_0": 105, "Lactate_0h": 2.0, "INR_0h": 1.2, "Age": 60, "FrailtyScore": 4}


@pytest.fixture(scope="module")
def art():
    return M.cargar()


@pytest.fixture(scope="module")
def kb(art):
    return A.BaseConocimiento(art)


def test_artefacto_completo(art):
    for clave in ["pipeline", "calibracion", "cortes", "rangos", "metricas", "coeficientes", "fondo", "valor_base"]:
        assert clave in art
    assert art["cortes"]["bajo"] < art["cortes"]["alto"]
    assert json.loads((RAIZ / "models" / "metricas.json").read_text(encoding="utf-8"))["auc_test"] == pytest.approx(art["metricas"]["auc_test"])


def test_prediccion_es_probabilidad_y_la_explicacion_cuadra(art):
    r = M.predecir(art, BASE)
    assert 0 < r["riesgo"] < 1 and r["suma_ok"]
    a, b = art["calibracion"]["a"], art["calibracion"]["b"]
    z = art["valor_base"] + sum(f["contribucion"] for f in r["contribuciones"])
    assert M.sigmoide(a * z + b) == pytest.approx(r["riesgo"])


def test_choque_septico_sube_el_riesgo_y_cambia_el_estrato(art):
    sin = M.predecir(art, BASE)
    con = M.predecir(art, {**BASE, "SepticShock": 1, "VasopressorUse_0h": 1})
    assert con["riesgo"] > sin["riesgo"] + 0.10
    assert con["estrato"] == "Mayor riesgo"


def test_en_produccion_no_se_imputa(art):
    incompleto = {**BASE, "Lactate_0h": None, "INR_0h": None}
    with pytest.raises(ValueError):
        M.predecir(art, incompleto)                                   # el modelo se niega a predecir con vacíos
    errores = [a["texto"] for a in M.validar(art, incompleto) if a["nivel"] == "error"]
    assert len(errores) == 1 and "lactato" in errores[0] and "INR" in errores[0]


def test_saltar_el_imputador_no_cambia_la_prediccion(art):
    # Con datos completos, el flujo de servicio (sin imputador) debe coincidir con el pipeline completo.
    import pandas as pd
    for extra in [{}, {"SepticShock": 1, "VasopressorUse_0h": 1, "InfectionSource": "desconocido", "Age": 83}]:
        v = {**BASE, **extra}
        z_pipeline = float(art["pipeline"].decision_function(pd.DataFrame([v])[art["variables"]])[0])
        assert M.predecir(art, v)["log_odds"] == pytest.approx(z_pipeline, abs=1e-12)


def test_validaciones(art):
    assert any(a["nivel"] == "error" for a in M.validar(art, {**BASE, "SBP_0": 60, "DBP_0": 80}))
    assert any("fuera del rango" in a["texto"] for a in M.validar(art, {**BASE, "Lactate_0h": 25}))
    assert any("Sepsis-3" in a["texto"] for a in M.validar(art, {**BASE, "SepticShock": 1, "VasopressorUse_0h": 0}))
    assert M.validar(art, BASE) == []
    completo, calculada = M.completar_pam({**BASE, "MAP_0": None})
    assert calculada and completo["MAP_0"] == pytest.approx((105 + 2 * 60) / 3, abs=0.1)


@pytest.mark.parametrize("pregunta, esperado", [
    ("¿Qué significa un odds ratio de 2 para el choque séptico?", "choque séptico"),
    ("¿Por qué el lactato casi no pesa?", "lactato"),
    ("como interpreto el coeficiente del INR", "INR"),
    ("¿El modelo es mejor que el SOFA?", "Comparación del modelo con SOFA"),
    ("que pasa si no tengo el lactato", "falta un dato"),
    ("de donde salieron los datos", "Origen de los datos"),
    ("que es la regularizacion L1", "L1"),
    ("probaron naive bayes", "no alcanza un mejor desempeño"),
    ("¿la aplicación guarda los datos del paciente?", "Privacidad"),
])
def test_recuperacion(kb, pregunta, esperado):
    assert esperado in A.responder(kb, pregunta)["fuentes"][0]["titulo"]


def test_asistente_respeta_sus_limites(kb, art):
    assert A.responder(kb, "¿Qué antibiótico le doy?")["texto"] == A.AVISO_FUERA
    assert A.responder(kb, "¿Cuál es la capital de Francia?")["texto"] == A.SIN_RESULTADO
    assert "Aún no hay un paciente" in A.responder(kb, "¿Por qué le sale ese riesgo a este paciente?")["texto"]
    r = A.responder(kb, "¿Por qué le sale ese riesgo a este paciente?", M.predecir(art, {**BASE, "SepticShock": 1, "VasopressorUse_0h": 1}))
    assert "choque séptico" in r["texto"] and "puntos" in r["texto"]


def test_las_cifras_del_asistente_salen_del_artefacto(kb, art):
    texto = next(f["texto"] for f in kb.fragmentos if f["titulo"].startswith("Qué tan bueno"))
    assert f"{art['metricas']['auc_test']:.3f}".replace(".", ",") in texto


def test_asistente_explica_una_variable_con_los_valores_del_paciente(kb, art):
    res = M.predecir(art, {**BASE, "SepticShock": 1, "VasopressorUse_0h": 1, "Lactate_0h": 9.5})
    r = A.responder(kb, "¿Por qué el lactato dio así en este paciente?", res)
    assert "lactato = 9,5 mmol/L" in r["texto"] and "promedio de la cohorte" in r["texto"]
    assert r["fuentes"][0]["titulo"] == A.TITULO_VARIABLE["Lactate_0h"]
    # La cifra del paciente es la misma que calcula la aplicación.
    pp = next(f["efecto_pp"] for f in res["contribuciones"] if f["variable"] == "Lactate_0h")
    assert A._n(abs(pp), 1) in r["texto"]


def test_asistente_variable_con_coeficiente_cero_y_varias_variables(kb, art):
    res = M.predecir(art, {**BASE, "SepticShock": 1, "VasopressorUse_0h": 1})
    assert "coeficiente es cero" in A.responder(kb, "¿Y la presión arterial media?", res)["texto"]
    r = A.responder(kb, "¿Cómo influyen la edad y el choque séptico en este paciente?", res)
    assert "edad =" in r["texto"] and "choque séptico =" in r["texto"]


def test_asistente_no_usa_al_paciente_en_preguntas_generales_ni_sin_paciente(kb, art):
    res = M.predecir(art, BASE)
    assert "En este paciente" not in A.responder(kb, "¿Qué significa un odds ratio de 2 para el choque séptico?", res)["texto"]
    assert "En este paciente" not in A.responder(kb, "¿Por qué el lactato casi no pesa?")["texto"]       # sin paciente evaluado
    assert "Aún no hay un paciente" in A.responder(kb, "¿Por qué el lactato dio así en este paciente?")["texto"]
