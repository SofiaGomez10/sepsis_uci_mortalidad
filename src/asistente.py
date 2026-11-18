"""Asistente de interpretación para personal clínico (recuperación aumentada, tipo RAG).

Flujo:
  1. Base de conocimiento = fragmentos conceptuales (kb/*.md) + fragmentos generados desde el
     artefacto del modelo (coeficientes, métricas, cortes). Las cifras nunca se escriben a mano:
     salen del mismo archivo que usa la predicción, así el asistente no puede contradecir a la app.
  2. Recuperación = TF-IDF + similitud coseno sobre los fragmentos (local, sin servicios externos).
  3. Si hay un paciente evaluado y la pregunta lo menciona o nombra una variable, se antepone una explicación
     determinista con los valores de ese paciente (explicar_variable / resumen_paciente), siempre en local.
  4. Respuesta = (a) local: se devuelve el fragmento más pertinente; o (b) generativa: si hay un
     LLM configurado en los secretos, redacta la respuesta usando SOLO los fragmentos recuperados.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import modelo as M

DIR_KB = Path(__file__).resolve().parent.parent / "kb"

VACIAS = set("""a al algo ante asi aun bien cada como con cual cuales cuando de del desde donde dos el ella ellos en entre
era es esa ese eso esta estan estas este esto estos fue ha hace hacia han hasta hay la las le les lo los mas me mi mis
muy ni no nos o os otra otro para pero por porque que quien se sea ser si sin sobre son su sus tal tambien te tiene
tienen todo todos tu un una uno unos y ya yo usted ustedes dice decir quiere puedo puede seria favor explicame explica
explicar dime saber quisiera significa""".split())

# Términos que el médico puede usar y que en la base aparecen de otra forma.
SINONIMOS = {
    "shock": "choque", "septico": "choque septico", "tension": "presion arterial", "pam": "presion arterial media map",
    "pas": "presion arterial sistolica sbp", "pad": "presion arterial diastolica dbp", "tam": "presion arterial media map",
    "noradrenalina": "vasopresor", "norepinefrina": "vasopresor", "vasoactivo": "vasopresor", "vasopresores": "vasopresor",
    "momios": "odds ratio", "or": "odds ratio", "peso": "coeficiente", "pesos": "coeficiente", "beta": "coeficiente",
    "coagulacion": "inr", "anticoagulado": "inr", "fragil": "fragilidad", "anciano": "edad", "viejo": "edad", "adulto": "edad",
    "neumonia": "foco pulmonar", "respiratorio": "foco pulmonar", "urosepsis": "foco urinario", "itu": "foco urinario",
    "exactitud": "auc discriminacion desempeno", "precision": "auc discriminacion desempeno vpp", "fiable": "calibracion limites confiable",
    "confiar": "limites calibracion confiable", "confiable": "limites calibracion", "sinteticos": "origen datos", "amci": "origen datos",
    "sofa": "sofa escala", "apache": "apache sofa escala comparacion", "faltante": "falta dato obligatorio imputacion", "falta": "falta dato obligatorio imputacion",
    "umbral": "corte estrato", "umbrales": "corte estrato", "estratos": "estrato corte", "tratamiento": "decision clinica uso previsto",
    "privacidad": "datos personales privacidad", "naive": "naive bayes otros modelos", "bayes": "naive bayes otros modelos",
}


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9%]+", " ", texto).strip()


def _n(x: float, d: int = 2) -> str:
    """Número con coma decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def _pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}".replace(".", ",") + " %"


def fragmentos_del_modelo(art: dict) -> list[dict]:
    """Fragmentos con cifras, generados desde el artefacto."""
    m, coef, cortes = art["metricas"], art["coeficientes"], art["cortes"]
    est = m["estratos_test"]
    frs = []

    def add(titulo, claves, texto):
        frs.append({"titulo": titulo, "claves": claves, "texto": texto, "origen": "Ficha del modelo"})

    # --- una ficha por variable -------------------------------------------------------------
    lectura = {
        "SepticShock": "Es el factor con más peso del modelo. Tener choque séptico multiplica los odds de fallecer por {oru}: en un paciente típico sin choque ni vasopresor el riesgo estimado es cercano a 41 % y con choque séptico (que siempre se acompaña de vasopresor) sube a cerca de 61 %. Coincide con la clínica: Sepsis-3 describe el choque séptico como el subgrupo con mortalidad hospitalaria superior al 40 %.",
        "VasopressorUse_0h": "Recibir vasopresor al ingreso multiplica los odds por {oru}. El efecto propio es pequeño porque casi toda su información ya está en el choque séptico: en los datos, todos los pacientes con choque recibían vasopresor, de modo que este coeficiente recoge sobre todo a quienes lo reciben sin cumplir criterios de choque.",
        "Age": "Cada año adicional multiplica los odds por {oru}; por cada {de} años (una desviación estándar) los multiplica por {orde}. El efecto es real pero pequeño en estos datos: entre un paciente de 40 y uno de 80 años, con todo lo demás igual, el riesgo estimado cambia unos 3 puntos porcentuales.",
        "FrailtyScore": "Cada punto adicional en la escala de fragilidad multiplica los odds por {oru}. Entre el mínimo (1) y el máximo (7) de la escala el riesgo estimado cambia unos 3 a 4 puntos porcentuales.",
        "INR_0h": "Cada unidad adicional de INR multiplica los odds por {oru}. Refleja coagulopatía asociada a disfunción hepática o coagulación intravascular. El modelo no distingue si el INR elevado se debe a anticoagulación previa; eso debe valorarlo el clínico.",
        "Lactate_0h": "Cada mmol/L adicional de lactato multiplica los odds por {oru}, un efecto casi nulo. Esto sorprende clínicamente, porque el lactato es un marcador pronóstico reconocido. La explicación es doble: en este conjunto de datos el lactato casi no se relaciona con la mortalidad (correlación de 0,05), y lo poco que aporta ya está contenido en el choque séptico, cuya definición incluye lactato mayor de 2 mmol/L. No debe leerse como que el lactato no importa.",
        "SOFA_0h": "Cada punto de SOFA multiplica los odds por {oru}: prácticamente no aporta una vez se conocen el choque séptico y el vasopresor, que son el componente cardiovascular del propio SOFA. En esta cohorte, además, el SOFA discrimina poco por sí solo (AUC de " + _n(m["sofa"]["auc"], 3) + "), y sus valores son bajos (entre 0 y 6).",
        "DBP_0": "Cada mmHg adicional de presión diastólica multiplica los odds por {oru}. El signo positivo (más presión, más riesgo) es contraintuitivo y no debe interpretarse clínicamente: el efecto es mínimo (unos 2 puntos porcentuales entre los extremos observados) y es un artefacto de la colinealidad con el choque séptico, el vasopresor y las otras presiones. Al conocer que hay choque, el modelo ya sabe que hay hipotensión.",
        "MAP_0": "El coeficiente de la presión arterial media es cero: el modelo no la usa directamente. No significa que la PAM no importe. La hipotensión ya entra al modelo a través del choque séptico y el vasopresor, y la regularización L1 descarta las variables que repiten información. Sí se usa para completar otros datos cuando faltan.",
        "SBP_0": "El coeficiente de la presión sistólica es cero: su información ya está representada por el choque séptico, el vasopresor y la presión diastólica, y la regularización L1 la descartó. Se sigue pidiendo porque ayuda a estimar otros datos cuando faltan y a calcular la presión arterial media.",
    }
    for var, plantilla in lectura.items():
        c = coef[var]
        texto = plantilla.format(oru=_n(c["OR_por_unidad"], 3 if abs(c["OR_por_unidad"] - 1) < 0.05 else 2),
                                 orde=_n(c["OR_por_DE"], 2), de=_n(c["DE_de_la_variable"] or 0, 0))
        texto += f" (Coeficiente por desviación estándar: {_n(c['coef_por_DE'], 3)}; odds ratio por desviación estándar: {_n(c['OR_por_DE'], 3)}.)"
        add(f"Cómo influye en el modelo: {M.minuscula(M.etiqueta(var))}", f"{var} {M.etiqueta(var)} coeficiente odds ratio efecto influye importa peso", texto)

    f_desc, f_pul = coef["InfectionSource_desconocido"], coef["InfectionSource_pulmonar"]
    add("Cómo influye en el modelo: foco de infección",
        "InfectionSource foco infeccion origen abdominal pulmonar urinario piel desconocido coeficiente odds ratio referencia protector",
        f"El foco abdominal es la categoría de referencia: los demás se comparan contra él. Un foco desconocido multiplica los odds por {_n(f_desc['OR_por_unidad'])}, lo que es plausible porque un foco no identificado retrasa su control. El foco pulmonar los multiplica por {_n(f_pul['OR_por_unidad'])}, una reducción mínima que describe estos datos y no debe leerse como efecto protector. Los focos urinario y de piel tienen coeficiente cero: el modelo no los distingue del abdominal.")

    orden = sorted(((k, v["coef_por_DE"]) for k, v in coef.items() if abs(v["coef_por_DE"]) > 1e-6), key=lambda kv: -abs(kv[1]))
    nombres = {"InfectionSource_desconocido": "foco desconocido", "InfectionSource_pulmonar": "foco pulmonar"}
    listado = "; ".join(f"{nombres.get(k, M.minuscula(M.etiqueta(k)))} ({_n(v, 3)})" for k, v in orden)
    add("Qué variables pesan más en el modelo", "ranking importancia variables mas importantes pesan mas orden principales factores cuales usa",
        f"Ordenadas por la magnitud de su coeficiente estandarizado: {listado}. El choque séptico pesa más del doble que cualquier otra variable. Quedaron en cero la presión arterial media, la sistólica y los focos urinario y de piel. Todos los coeficientes son pequeños: ninguna variable por sí sola mueve mucho el riesgo.")

    # --- desempeño y límites ----------------------------------------------------------------
    add("Qué tan bueno es el modelo: discriminación", "AUC desempeno rendimiento que tan bueno exactitud acierta discriminacion resultado confiable",
        f"En los {m['n_test']} pacientes de prueba el AUC fue {_n(m['auc_test'], 3)} (IC 95 %: {_n(m['auc_test_ic95'][0], 3)} a {_n(m['auc_test_ic95'][1], 3)}) y en validación cruzada {_n(m['auc_cv'], 3)}. Es una discriminación baja: mejor que el azar (0,5), pero lejos de 0,8. En la práctica, el modelo separa bien a los pacientes con choque séptico del resto y distingue poco dentro de cada grupo. Debe usarse como un dato adicional, nunca como criterio único.")
    s = m["sofa"]
    add("Comparación del modelo con SOFA", "SOFA comparacion mejor que escala APACHE estandar referencia supera",
        f"En los {s['n']} pacientes de prueba con SOFA registrado, el SOFA al ingreso tuvo un AUC de {_n(s['auc'], 3)} y el modelo de {_n(s['auc_modelo_misma_muestra'], 3)}. La diferencia es de {_n(s['diferencia'], 3)} (IC 95 %: {_n(s['diferencia_ic95'][0], 3)} a {_n(s['diferencia_ic95'][1], 3)}): pequeña, pero el intervalo no incluye el cero. El modelo complementa al SOFA, no lo reemplaza. No fue posible comparar contra APACHE II ni SAPS II porque esas columnas vienen vacías en el conjunto de datos y faltan variables para recalcularlas.")
    add("Calibración del modelo: se puede creer en el porcentaje", "calibracion Brier porcentaje confiable probabilidad real pendiente Platt",
        f"Sí, dentro de su rango. El riesgo medio predicho en prueba coincide con la mortalidad observada ({_pct(m['mortalidad'])}), el puntaje de Brier es {_n(m['brier_test'], 4)} (frente a {_n(m['brier_referencia'], 4)} de predecir siempre el promedio) y la pendiente de calibración es {_n(m['calibracion_pendiente'], 2)} (ideal: 1). La limitación es otra: el modelo solo entrega riesgos entre {_pct(m['riesgo_min_test'], 0)} y {_pct(m['riesgo_max_test'], 0)}. Nunca dirá que un paciente tiene 5 % o 95 % de riesgo.")
    add("Qué significan los estratos de riesgo y los puntos de corte", "estrato corte umbral menor intermedio mayor riesgo categoria como se definieron alerta",
        f"Menor riesgo: por debajo de {_pct(cortes['bajo'])}; en prueba falleció el {_pct(est['Menor riesgo']['mortalidad_observada'])} de {int(est['Menor riesgo']['n'])} pacientes. Riesgo intermedio: entre {_pct(cortes['bajo'])} y {_pct(cortes['alto'])}; falleció el {_pct(est['Riesgo intermedio']['mortalidad_observada'])} de {int(est['Riesgo intermedio']['n'])}. Mayor riesgo: {_pct(cortes['alto'])} o más; falleció el {_pct(est['Mayor riesgo']['mortalidad_observada'])} de {int(est['Mayor riesgo']['n'])}. El corte inferior se fijó para detectar al 60 % de los fallecidos y el superior para mantener una especificidad de 85 %, ambos con validación cruzada. Los nombres son relativos a esta cohorte: 'menor riesgo' no significa riesgo bajo, porque la mortalidad de partida ya es de {_pct(m['mortalidad'])}.")
    b, a = m["en_corte_bajo"], m["en_corte_alto"]
    add("Sensibilidad y especificidad del modelo en cada corte", "sensibilidad especificidad VPP VPN falsos positivos negativos matriz confusion cuantos detecta alertas",
        f"Con el corte inferior ({_pct(b['corte'])}): sensibilidad {_pct(b['sensibilidad'])}, especificidad {_pct(b['especificidad'])}, VPP {_pct(b['VPP'])}, VPN {_pct(b['VPN'])}; detecta {int(b['VP'])} de {int(b['VP'] + b['FN'])} fallecidos con {int(b['FP'])} falsas alarmas. Con el corte superior ({_pct(a['corte'])}): sensibilidad {_pct(a['sensibilidad'])}, especificidad {_pct(a['especificidad'])}, VPP {_pct(a['VPP'])}, VPN {_pct(a['VPN'])}. Ningún corte logra a la vez alta sensibilidad y alta especificidad; es la consecuencia directa de un AUC cercano a 0,6.")
    add("Límites del modelo y uso previsto", "limites limitaciones uso previsto decision clinica confiar puedo usarlo tratamiento reemplaza juicio advertencia para que sirve no sirve",
        "Es un prototipo académico de apoyo a la decisión. Sirve para tener una estimación inicial y transparente del riesgo al ingreso a UCI y para ver qué factores la explican. No debe usarse para decidir limitación del esfuerzo terapéutico, negar un ingreso a UCI ni reemplazar el juicio del intensivista. Sus límites son: discriminación baja; entrenado con un único conjunto de datos de procedencia no documentada; sin validación externa en una institución real; solo usa datos del ingreso (no ve la evolución); y válido únicamente para adultos con sepsis en UCI. Antes de un uso asistencial requeriría reentrenamiento y validación con datos locales, y aprobación del comité de ética.")
    add("Origen de los datos de entrenamiento", "datos origen procedencia AMCI sinteticos anonimizados de donde salen pacientes reales cohorte poblacion",
        f"El modelo se entrenó con un conjunto de datos de {m['n_total']} registros de adultos con sepsis en UCI, entregado por la AMCI en una iniciativa con la Universidad Autónoma de Occidente; {m['n_utilizable']} tenían el desenlace registrado y se usaron {m['n_train']} para entrenar y {m['n_test']} para probar. No se cuenta con documentación sobre las instituciones de origen, el periodo ni el proceso de anonimización. El equipo sospecha, sin confirmación, que los datos son sintéticos o fueron alterados: la mortalidad es de {_pct(m['mortalidad'])} incluso con SOFA de 0 a 1, las comorbilidades casi no cambian la mortalidad y ningún algoritmo supera un AUC de 0,60. Por eso los resultados describen este conjunto de datos y no deben extrapolarse a una UCI real.")
    add("Por qué el modelo no alcanza un mejor desempeño", "por que tan bajo AUC mejorar otros modelos naive bayes random forest boosting mas variables techo",
        f"Se probaron regresión logística, Naive Bayes, random forest y gradient boosting, y también usar todas las variables disponibles en lugar de once. Todos quedaron entre 0,57 y 0,58 de AUC en validación cruzada, y usar las variables de las primeras 24 horas tampoco ayudó ({_n(m['auc_cv_todas_24h'], 3)}). Cuando modelos muy distintos llegan al mismo techo, la limitación está en la información que contienen los datos y no en el algoritmo. Se conservó la regresión logística porque iguala a los demás y es la única que se puede explicar variable por variable.")
    add("Qué es este modelo y cómo calcula el riesgo", "que es el modelo como funciona como calcula regresion logistica formula algoritmo que hace",
        "Es una regresión logística con regularización L1. Toma once datos del ingreso a UCI (edad, fragilidad, presiones sistólica, diastólica y media, lactato, INR, SOFA, choque séptico, vasopresor y foco de infección), los estandariza, multiplica cada uno por su coeficiente y suma. Esa suma se convierte en una probabilidad con la función sigmoide y se ajusta con una calibración para que el porcentaje corresponda a la mortalidad observada. Funciona como una escala de puntos cuyos pesos se aprendieron de los datos.")
    add("Privacidad y manejo de los datos del paciente", "privacidad datos personales guarda almacena confidencial ley 1581 historia clinica identificacion seguro",
        "La aplicación no pide ni almacena identificadores del paciente y no guarda los valores digitados: se usan solo para el cálculo de la sesión. El asistente, en su modo local, no envía nada a servicios externos. Si se activa el modo generativo, la pregunta y los fragmentos recuperados se envían al proveedor del modelo de lenguaje; los datos del paciente evaluado solo se incluyen si el usuario lo autoriza con la casilla correspondiente. En Colombia los datos de salud son datos sensibles (Ley 1581 de 2012), por lo que un uso con pacientes reales exigiría un despliegue dentro de la red de la institución.")
    return frs


def fragmentos_conceptuales() -> list[dict]:
    """Lee kb/*.md: cada encabezado '## ' abre un fragmento; la línea 'claves:' aporta términos de búsqueda."""
    frs = []
    for ruta in sorted(DIR_KB.glob("*.md")):
        for bloque in re.split(r"^## ", ruta.read_text(encoding="utf-8"), flags=re.M)[1:]:
            lineas = bloque.strip().split("\n")
            claves = lineas[1][len("claves:"):].strip() if len(lineas) > 1 and lineas[1].startswith("claves:") else ""
            cuerpo = " ".join(l.strip() for l in lineas[(2 if claves else 1):] if l.strip())
            frs.append({"titulo": lineas[0].strip(), "claves": claves, "texto": cuerpo, "origen": "Glosario"})
    return frs


class BaseConocimiento:
    """Índice TF-IDF sobre los fragmentos. El título y las claves pesan más que el cuerpo."""

    def __init__(self, art: dict):
        self.fragmentos = fragmentos_del_modelo(art) + fragmentos_conceptuales()
        docs = [normalizar(" ".join([f["titulo"]] * 3 + [f["claves"]] * 3 + [f["texto"]])) for f in self.fragmentos]
        self.vectorizador = TfidfVectorizer(ngram_range=(1, 2), stop_words=sorted(VACIAS), sublinear_tf=True)
        self.matriz = self.vectorizador.fit_transform(docs)

    def _expandir(self, pregunta: str) -> str:
        palabras = normalizar(pregunta).split()
        return " ".join(palabras + [SINONIMOS[p] for p in palabras if p in SINONIMOS])

    def buscar(self, pregunta: str, k: int = 3, minimo: float = 0.06) -> list[dict]:
        consulta = self.vectorizador.transform([self._expandir(pregunta)])
        sim = (self.matriz @ consulta.T).toarray().ravel()            # coseno: TF-IDF ya viene normalizado
        orden = np.argsort(-sim)[:k]
        return [dict(self.fragmentos[i], puntaje=float(sim[i])) for i in orden if sim[i] >= minimo]


def resumen_paciente(res: dict) -> str:
    """Explicación determinista del último paciente evaluado, en lenguaje clínico."""
    arriba = [f for f in res["contribuciones"] if f["efecto_pp"] >= 0.3]
    abajo = [f for f in res["contribuciones"] if f["efecto_pp"] <= -0.3]
    def lista(fs):
        return "; ".join(f"{M.minuscula(f['etiqueta'])} ({M.formato_valor(f['variable'], f['valor_usado'])}"
                         f"): {f['efecto_pp']:+.1f} puntos".replace(".", ",") for f in fs)
    t = (f"Para el paciente evaluado el riesgo estimado es {_pct(res['riesgo'])} ({res['estrato'].lower()}), "
         f"frente a {_pct(res['riesgo_base'])} de un paciente promedio de la cohorte. ")
    t += f"Lo elevan: {lista(arriba)}. " if arriba else "Ningún factor lo eleva de forma apreciable. "
    t += f"Lo reducen: {lista(abajo)}. " if abajo else ""
    t += "Cada cifra indica cuánto cambiaría el riesgo si ese dato estuviera en el promedio de la cohorte; no son efectos causales."
    return t


# Expresiones que apuntan claramente al paciente evaluado. Si no hay paciente, se le avisa al usuario.
PATRON_PACIENTE = re.compile(r"\b(este|esta|mi) paciente\b|\bpaciente evaluado\b|\b(su|mi|este|ese) riesgo\b|"
                             r"\bpor que (le |me )?(sale|salio|dio|da|tiene)\b|\bme (sale|salio|dio)\b|\beste (caso|resultado|valor|dato)\b|\bel resultado\b")
# Expresiones más ambiguas: solo cuentan si ya hay un paciente evaluado (si no, se responde en general).
PATRON_PACIENTE_DEBIL = re.compile(r"\b(el|del|al|ese|tal) paciente\b|\ben este caso\b|\baqui\b|\baca\b|\bel caso\b")

# Preguntas de "por qué / cuánto pesa" sobre una variable: con un paciente evaluado se contestan también con sus valores.
PATRON_PORQUE = re.compile(r"\bpor que\b|\bpesa\b|\bpeso\b|\baporta\b|\binfluye\b|\bimporta\b|\bcuanto\b|\bcuanta\b|\bsube\b|\bbaja\b")

# Variable del modelo a la que se refiere la pregunta (texto ya normalizado: sin tildes ni mayúsculas).
PATRON_VARIABLE = {
    "SepticShock": re.compile(r"\bchoque\b|\bshock\b|\bseptico\b"),
    "VasopressorUse_0h": re.compile(r"vasopresor|vasoactiv|noradrenalina|norepinefrina"),
    "Lactate_0h": re.compile(r"\blactato\b|\blactate\b"),
    "INR_0h": re.compile(r"\binr\b|coagulacion"),
    "SOFA_0h": re.compile(r"\bsofa\b"),
    "Age": re.compile(r"\bedad\b|\bage\b"),
    "FrailtyScore": re.compile(r"fragilidad|\bfragil\b"),
    "SBP_0": re.compile(r"sistolica|\bpas\b|\bsbp\b"),
    "DBP_0": re.compile(r"diastolica|\bpad\b|\bdbp\b"),
    "MAP_0": re.compile(r"presion arterial media|\bpam\b|\btam\b|\bmap\b"),
    "InfectionSource": re.compile(r"\bfoco\b|infeccion|urosepsis|neumonia"),
}
TITULO_VARIABLE = {v: ("Cómo influye en el modelo: foco de infección" if v == "InfectionSource"
                       else "Cómo influye en el modelo: " + M.minuscula(M.etiqueta(v))) for v in PATRON_VARIABLE}


def variables_mencionadas(q: str) -> list[str]:
    """Variables del modelo nombradas en la pregunta (q ya normalizada), en orden de aparición."""
    hallazgos = [(m.start(), v) for v, pat in PATRON_VARIABLE.items() if (m := pat.search(q))]
    return [v for _, v in sorted(hallazgos)][:3]


def explicar_variable(res: dict, variable: str) -> str:
    """Explica con los valores del paciente evaluado cómo influyó UNA variable en su resultado (determinista)."""
    f = next(x for x in res["contribuciones"] if x["variable"] == variable)
    nombre, pp, total = M.minuscula(f["etiqueta"]), f["efecto_pp"], len(res["contribuciones"])
    dato = M.formato_valor(variable, f["valor_usado"])
    if variable == "InfectionSource":
        t = f"En este paciente el foco de infección es {dato}. "
    elif variable in ("SepticShock", "VasopressorUse_0h"):
        t = f"En este paciente: {nombre} = {dato}; en la cohorte lo presenta el {_pct(f['promedio'], 0)}. "
    else:
        t = f"En este paciente: {nombre} = {dato}; el promedio de la cohorte es {M.formato_valor(variable, f['promedio'])}. "

    if f["coef"] is not None and abs(f["coef"]) < 1e-6:
        return t + ("El modelo no usa esta variable (su coeficiente es cero), por lo que no influyó en el riesgo de este paciente. "
                    "No significa que clínicamente no importe: su información ya está en otras variables.")
    if abs(pp) < 0.3:
        t += f"Prácticamente no modifica su riesgo ({_n(pp, 1)} puntos porcentuales frente a un paciente promedio)"
        if f["z"] is not None and abs(f["z"]) >= 1:
            t += ", aunque el dato se aleja del promedio: el coeficiente de esta variable es pequeño y por eso aporta poco"
        t += ". "
    else:
        sentido = "lo eleva" if pp > 0 else "lo reduce"
        t += f"Este dato {sentido} en {_n(abs(pp), 1)} puntos porcentuales respecto a un paciente con el valor promedio. "
        if f["z"] is not None and abs(f["z"]) >= 1 and abs(pp) < 1:
            t += "Aunque el dato se aleja bastante del promedio, el coeficiente de esta variable es pequeño y por eso su aporte es poco. "
    t += (f"Es el factor n.º {f['puesto']} de {total} por tamaño de aporte en este caso. Es una contribución descriptiva del modelo, no un efecto causal.")
    return t


PATRON_FUERA = re.compile(r"\b(dosis|antibiotico|que (le )?(doy|administro|formulo)|tratamiento recomend|debo (intubar|iniciar|suspender)|pronostico de vida)\b")

AVISO_FUERA = ("Esa pregunta es de manejo clínico y este asistente solo explica cómo funciona el modelo y cómo leer su resultado. "
               "Para decisiones terapéuticas rigen las guías vigentes (por ejemplo, Surviving Sepsis Campaign) y el criterio del equipo tratante.")
SIN_RESULTADO = ("No encontré información sobre eso en la documentación del modelo. Puedo explicar qué significa cada coeficiente u odds ratio, "
                 "cómo leer el resultado de un paciente, qué tan confiable es el modelo y cuáles son sus límites.")

INSTRUCCIONES_LLM = """Eres el asistente de interpretación de una herramienta académica que estima el riesgo de mortalidad en UCI por sepsis con una regresión logística.
Tu lector es personal médico sin formación en ingeniería ni estadística avanzada.
Reglas:
- Responde SOLO con la información de los FRAGMENTOS (y de los DATOS DEL PACIENTE si aparecen). No agregues cifras ni afirmaciones que no estén allí.
- Si los fragmentos no contienen la respuesta, dilo y sugiere qué sí puedes explicar.
- No des recomendaciones de tratamiento, dosis ni conductas. Si te las piden, indica que eso corresponde a las guías clínicas y al equipo tratante.
- Recuerda cuando sea pertinente que el modelo es de apoyo y que su discriminación es baja.
- Español claro, sin jerga técnica innecesaria, sin emojis, máximo 170 palabras. Usa coma decimal."""


def llamar_llm(pregunta: str, fragmentos: list[dict], contexto_paciente: str | None, config: dict, historial: list[dict]) -> str:
    """Genera la respuesta con un LLM a través de una API compatible con OpenAI (chat/completions)."""
    import requests

    contexto = "\n\n".join(f"[{i + 1}] {f['titulo']}\n{f['texto']}" for i, f in enumerate(fragmentos))
    usuario = f"FRAGMENTOS:\n{contexto}\n\n"
    if contexto_paciente:
        usuario += f"DATOS DEL PACIENTE EVALUADO:\n{contexto_paciente}\n\n"
    usuario += f"PREGUNTA: {pregunta}"
    mensajes = [{"role": "system", "content": INSTRUCCIONES_LLM}] + historial[-4:] + [{"role": "user", "content": usuario}]
    r = requests.post(config["base_url"].rstrip("/") + "/chat/completions",
                      headers={"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"},
                      json={"model": config["model"], "messages": mensajes, "temperature": 0.1, "max_tokens": 500}, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def responder(kb: BaseConocimiento, pregunta: str, resultado: dict | None = None, llm: dict | None = None,
              compartir_paciente: bool = False, historial: list[dict] | None = None) -> dict:
    """Devuelve {'texto', 'fuentes', 'modo'} para una pregunta del usuario."""
    q = normalizar(pregunta)
    if PATRON_FUERA.search(q):
        return {"texto": AVISO_FUERA, "fuentes": [], "modo": "local"}

    variables = variables_mencionadas(q)
    fuerte = bool(PATRON_PACIENTE.search(q))
    pregunta_sobre_variable = bool(variables) and (bool(PATRON_PORQUE.search(q)) or len(q.split()) <= 6)
    sobre_paciente = fuerte or (resultado is not None and (bool(PATRON_PACIENTE_DEBIL.search(q)) or pregunta_sobre_variable))
    fragmentos = kb.buscar(pregunta)
    if variables and sobre_paciente and resultado:   # al explicar una variable del paciente, su ficha va primero
        titulos = {TITULO_VARIABLE[v] for v in variables}
        fichas = [dict(f, puntaje=1.0) for f in kb.fragmentos if f["titulo"] in titulos]
        fragmentos = (fichas + [f for f in fragmentos if f["titulo"] not in titulos])[:3]

    if fuerte and not resultado:
        return {"texto": "Aún no hay un paciente evaluado en esta sesión. Calcule el riesgo en la pestaña 'Evaluación del paciente' y vuelva a preguntar.",
                "fuentes": [], "modo": "local"}
    if sobre_paciente and resultado:
        # Con variable nombrada se explica esa variable con los valores del paciente; si no, el resumen completo.
        texto_paciente = "\n\n".join(explicar_variable(resultado, v) for v in variables) if variables else resumen_paciente(resultado)
    else:
        texto_paciente = None
    if not fragmentos and not texto_paciente:
        return {"texto": SIN_RESULTADO, "fuentes": [], "modo": "local"}

    if llm and llm.get("api_key") and llm.get("model"):
        try:
            texto = llamar_llm(pregunta, fragmentos, texto_paciente if compartir_paciente else None, llm, historial or [])
            if texto_paciente and not compartir_paciente:
                texto = texto_paciente + "\n\n" + texto          # la parte del paciente se resuelve localmente
            return {"texto": texto, "fuentes": fragmentos, "modo": "generativo"}
        except Exception as e:                                   # si el servicio falla, se responde en modo local
            aviso = f"(El servicio de lenguaje no respondió: {type(e).__name__}. Respuesta en modo local.)\n\n"
        else:
            aviso = ""
    else:
        aviso = ""

    partes = [texto_paciente] if texto_paciente else []
    if fragmentos and not (texto_paciente and fragmentos[0]["puntaje"] < 0.15):
        partes.append(fragmentos[0]["texto"])
    return {"texto": aviso + "\n\n".join(partes), "fuentes": fragmentos, "modo": "local"}


PREGUNTAS_SUGERIDAS = [
    "¿Qué significa un odds ratio de 2 para el choque séptico?",
    "¿Por qué el lactato casi no pesa en el modelo?",
    "¿Qué tan confiable es el porcentaje que entrega?",
    "¿Por qué la presión arterial media tiene coeficiente cero?",
    "¿El modelo es mejor que el SOFA?",
    "¿Qué pasa si me falta un dato?",
]
# Solo tienen sentido cuando ya se evaluó un paciente.
PREGUNTAS_PACIENTE = [
    "¿Por qué le sale ese riesgo a este paciente?",
    "¿Cuánto aporta el choque séptico en este paciente?",
    "¿Por qué el lactato dio así en este paciente?",
]
