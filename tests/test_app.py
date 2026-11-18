"""Pruebas del artefacto y la inferencia.  Ejecutar:  pytest -q"""
import json
from pathlib import Path

import pytest

from src import modelo as M

RAIZ = Path(__file__).resolve().parent.parent
BASE = {"InfectionSource": "abdominal", "SepticShock": 0, "VasopressorUse_0h": 0, "MAP_0": 75, "SOFA_0h": 2, "DBP_0": 60,
        "SBP_0": 105, "Lactate_0h": 2.0, "INR_0h": 1.2, "Age": 60, "FrailtyScore": 4}


@pytest.fixture(scope="module")
def art():
    return M.cargar()



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


