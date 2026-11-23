"""Aplicación web para estimar el riesgo de mortalidad en UCI de adultos con sepsis.

Para ejecutarla en local: streamlit run app.py
La aplicación solo carga el modelo ya entrenado (models/modelo_sepsis.joblib): no entrena ni guarda datos.
"""
from html import escape
from pathlib import Path

import streamlit as st

from src import estilo as E
from src import modelo as M

RAIZ = Path(__file__).resolve().parent
FIGURAS = RAIZ / "reports" / "figuras"

st.set_page_config(page_title="Riesgo en sepsis · UCI", layout="wide", initial_sidebar_state="collapsed")
st.markdown(E.CSS, unsafe_allow_html=True)


# Carga del modelo. Se guarda en caché para que se lea una sola vez y no en cada interacción.
@st.cache_resource(show_spinner="Cargando modelo...")
def recursos():
    return M.cargar()


ART = recursos()
MET, CORTES = ART["metricas"], ART["cortes"]
n = E.n

st.markdown(E.barra(MET["fecha_entrenamiento"], ART["versiones"]["scikit-learn"]), unsafe_allow_html=True)


# Configuración del formulario: variables numéricas, límites permitidos y casos de ejemplo.
NUMERICAS = ["Age", "FrailtyScore", "SBP_0", "DBP_0", "MAP_0", "Lactate_0h", "INR_0h", "SOFA_0h"]

# Límites que acepta cada campo del formulario (impiden digitar valores imposibles).
LIMITES = {"Age": (18.0, 110.0), "FrailtyScore": (1.0, 9.0), "SBP_0": (30.0, 260.0), "DBP_0": (10.0, 160.0),
           "MAP_0": (20.0, 200.0), "Lactate_0h": (0.0, 30.0), "INR_0h": (0.5, 15.0), "SOFA_0h": (0.0, 24.0)}

# Opciones de las variables binarias. Antes existía "Sin dato", que se imputaba; ahora todos los campos son obligatorios.
SI_NO = ["No", "Sí"]

# Pacientes de ejemplo que se cargan con un botón, útiles para demostrar la aplicación.
CASOS = {
    "Choque séptico": {"Age": 79.0, "FrailtyScore": 6.0, "SBP_0": 78.0, "DBP_0": 44.0, "MAP_0": None, "Lactate_0h": 5.4,
                       "INR_0h": 1.9, "SOFA_0h": 5.0, "SepticShock": "Sí", "VasopressorUse_0h": "Sí", "InfectionSource": "desconocido"},
    "Sin choque": {"Age": 46.0, "FrailtyScore": 2.0, "SBP_0": 112.0, "DBP_0": 68.0, "MAP_0": 83.0, "Lactate_0h": 1.6,
                   "INR_0h": 1.1, "SOFA_0h": 2.0, "SepticShock": "No", "VasopressorUse_0h": "No", "InfectionSource": "pulmonar"},
    "Solo vasopresor": {"Age": 68.0, "FrailtyScore": 4.0, "SBP_0": 92.0, "DBP_0": 55.0, "MAP_0": None, "Lactate_0h": 1.8,
                        "INR_0h": 1.3, "SOFA_0h": 3.0, "SepticShock": "No", "VasopressorUse_0h": "Sí", "InfectionSource": "urinaria"},
}

# Formulario vacío, usado al iniciar y al pulsar "Limpiar formulario".
VACIO = {**{v: None for v in NUMERICAS}, "SepticShock": "No", "VasopressorUse_0h": "No", "InfectionSource": "desconocido"}

# Estado de la sesión: guarda lo digitado y el último resultado mientras el usuario interactúa con la página.
for clave, valor in VACIO.items():
    st.session_state.setdefault(f"in_{clave}", valor)
st.session_state.setdefault("resultado", None)
st.session_state.setdefault("avisos", [])


def cargar_caso(valores: dict):
    """Llena el formulario con los valores dados y borra el resultado anterior."""
    for clave, valor in valores.items():
        st.session_state[f"in_{clave}"] = valor
    st.session_state["resultado"], st.session_state["avisos"] = None, []


def numero(var: str):
    """Crea el campo numérico de una variable con su rótulo, límites, paso y texto de ayuda."""
    meta, (lo, hi) = M.VARIABLES[var], LIMITES[var]
    return st.number_input(meta["corta"], min_value=lo, max_value=hi, step=meta["paso"], key=f"in_{var}",
                           format=f"%.{meta['decimales']}f", placeholder="Se calcula" if var == "MAP_0" else "Requerido", help=meta["ayuda"])


t_eval, t_des, t_ficha = st.tabs(["Evaluación del paciente", "Desempeño del modelo", "Ficha técnica"])


# Pestaña 1. Evaluación del paciente: formulario a la izquierda y resultado a la derecha.
with t_eval:
    izq, der = st.columns([0.42, 0.58], gap="large")

    # Formulario con los 11 datos del ingreso, agrupados como en la práctica clínica.
    with izq:
        st.markdown('<div class="rotulo">Datos al ingreso a UCI (hora 0)</div>', unsafe_allow_html=True)
        with st.form("paciente", border=False):
            st.markdown('<div class="seccion">Paciente</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: numero("Age")
            with c2: numero("FrailtyScore")
            st.selectbox("Foco de infección", ART["categorias_foco"], key="in_InfectionSource",
                         format_func=lambda x: M.ETIQUETA_FOCO.get(x, x), help=M.VARIABLES["InfectionSource"]["ayuda"])

            st.markdown('<div class="seccion">Hemodinamia</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: numero("SBP_0")
            with c2: numero("DBP_0")
            with c3: numero("MAP_0")
            c1, c2 = st.columns(2)
            with c1: st.radio("Choque séptico", SI_NO, key="in_SepticShock", horizontal=True, help=M.VARIABLES["SepticShock"]["ayuda"])
            with c2: st.radio("Vasopresor al ingreso", SI_NO, key="in_VasopressorUse_0h", horizontal=True, help=M.VARIABLES["VasopressorUse_0h"]["ayuda"])

            st.markdown('<div class="seccion">Laboratorio y gravedad</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: numero("Lactate_0h")
            with c2: numero("INR_0h")
            with c3: numero("SOFA_0h")
            st.caption("Todos los campos son obligatorios. La presión arterial media puede dejarse vacía: se calcula a partir de la sistólica y la diastólica.")
            calcular = st.form_submit_button("Calcular riesgo", type="primary", width='stretch')

        # Botones para cargar los casos de ejemplo o limpiar el formulario.
        st.markdown('<div class="rotulo" style="margin-top:14px">Casos de ejemplo</div>', unsafe_allow_html=True)
        cols = st.columns(len(CASOS))
        for col, (nombre, valores) in zip(cols, CASOS.items()):
            col.button(nombre, on_click=cargar_caso, args=(valores,), width='stretch')
        st.button("Limpiar formulario", on_click=cargar_caso, args=(VACIO,))

    # Al pulsar "Calcular riesgo": se reúnen los datos, se completa la PAM si falta, se validan y si no hay errores, se calcula el riesgo.
    if calcular:
        tri = {"No": 0, "Sí": 1}
        valores = {v: st.session_state[f"in_{v}"] for v in NUMERICAS}
        valores.update({"SepticShock": tri[st.session_state["in_SepticShock"]], "VasopressorUse_0h": tri[st.session_state["in_VasopressorUse_0h"]],
                        "InfectionSource": st.session_state["in_InfectionSource"]})
        valores, pam_calculada = M.completar_pam(valores)
        avisos = M.validar(ART, valores)
        if pam_calculada:
            avisos.append({"nivel": "info", "texto": f"Presión arterial media calculada a partir de sistólica y diastólica: {valores['MAP_0']:.0f} mmHg."})
        st.session_state["avisos"] = avisos
        st.session_state["resultado"] = None if any(a["nivel"] == "error" for a in avisos) else M.predecir(ART, valores)

    # Resultado: errores que impiden calcular, o bien el riesgo, el estrato y su explicación.
    with der:
        res, avisos = st.session_state["resultado"], st.session_state["avisos"]
        st.markdown('<div class="rotulo">Resultado</div>', unsafe_allow_html=True)
        for a in [a for a in avisos if a["nivel"] == "error"]:
            st.markdown(f'<div class="nota error"><b>No se puede calcular.</b> {escape(a["texto"])}</div>', unsafe_allow_html=True)

        if res is None:
            if not any(a["nivel"] == "error" for a in avisos):
                st.markdown('<div class="tarjeta vacio"><b>Sin evaluación en curso</b>Complete los datos del ingreso y seleccione "Calcular riesgo",<br>o cargue uno de los casos de ejemplo.</div>', unsafe_allow_html=True)
        else:
            # Tarjeta principal: riesgo, estrato, comparación con la cohorte y mortalidad observada en el estrato.
            color = E.COLOR_ESTRATO[res["estrato"]]
            e = MET["estratos_test"][res["estrato"]]
            dif = (res["riesgo"] - MET["mortalidad"]) * 100
            comp = "similar al" if abs(dif) < 1 else f"{n(abs(dif))} puntos {'por encima' if dif > 0 else 'por debajo'} del"
            st.markdown(f"""<div class="tarjeta">
              <div class="resultado">
                <div><div class="cifra">{n(res['riesgo'] * 100)}<small>%</small></div></div>
                <div><span class="pastilla" style="background:{color}">{res['estrato']}</span></div>
                <div class="lectura">Probabilidad estimada de fallecer durante la estancia en UCI, {comp} promedio de la cohorte ({n(MET['mortalidad'] * 100)} %).</div>
              </div>
              {E.escala(res['riesgo'], CORTES)}
              <div class="nota">En el grupo de prueba, de <b>{int(e['n'])}</b> pacientes clasificados en <b>{res['estrato'].lower()}</b> falleció el
              <b>{n(e['mortalidad_observada'] * 100)} %</b>. Los estratos son relativos a esta cohorte, cuya mortalidad de partida es alta;
              "menor riesgo" no equivale a riesgo bajo.</div>
            </div>""", unsafe_allow_html=True)

            # Explicación, cuánto aporta cada variable al riesgo de este paciente.
            st.markdown('<div class="tarjeta"><div class="rotulo">Qué explica esta estimación</div>'
                        + E.filas_contribucion(res["contribuciones"], M.formato_valor) + "</div>", unsafe_allow_html=True)

            # Avisos que no impiden calcular (valores fuera de rango, PAM calculada, etc.).
            notas = [a for a in avisos if a["nivel"] != "error"]
            # Versión anterior, cambiada: informaba qué valores se habían estimado por falta de dato.
            # if res["imputadas"]:
            #     notas.append({"nivel": "aviso", "texto": "Valores estimados por falta de dato: " + ", ".join(M.minuscula(x) for x in res["imputadas"])
            #                   + ". La estimación es menos específica para este paciente."})
            if notas:
                st.markdown('<div class="rotulo">Observaciones sobre los datos</div>', unsafe_allow_html=True)
                for a in notas:
                    st.markdown(f'<div class="nota {"alerta" if a["nivel"] == "aviso" else ""}">{escape(a["texto"])}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="nota" style="margin-top:14px"><b>Cómo usar este resultado.</b> El modelo ordena a los pacientes mejor que el azar y que el SOFA al ingreso, '
                        f'pero su capacidad de discriminación es baja (AUC {n(MET["auc_test"], 2)}). Úselo como un dato más junto a la valoración clínica.</div>', unsafe_allow_html=True)


# Pestaña 2. Desempeño del modelo: indicadores principales y figuras de resultados.
with t_des:
    s, b, a = MET["sofa"], MET["en_corte_bajo"], MET["en_corte_alto"]
    st.markdown('<div class="kpis">'
                + E.kpi("Discriminación (AUC)", n(MET["auc_test"], 3), f"IC 95 %: {n(MET['auc_test_ic95'][0], 3)} a {n(MET['auc_test_ic95'][1], 3)}. Validación cruzada: {n(MET['auc_cv'], 3)}")
                + E.kpi("SOFA al ingreso (AUC)", n(s["auc"], 3), f"El modelo lo supera en {n(s['diferencia'], 3)} (IC 95 %: {n(s['diferencia_ic95'][0], 3)} a {n(s['diferencia_ic95'][1], 3)})")
                + E.kpi("Calibración (Brier)", n(MET["brier_test"], 3), f"Referencia sin modelo: {n(MET['brier_referencia'], 3)}. Pendiente: {n(MET['calibracion_pendiente'], 2)}")
                + E.kpi("Sensibilidad en corte inferior", n(b["sensibilidad"] * 100, 1) + " %", f"Especificidad {n(b['especificidad'] * 100, 1)} %. VPP {n(b['VPP'] * 100, 1)} %")
                + E.kpi("Especificidad en corte superior", n(a["especificidad"] * 100, 1) + " %", f"Sensibilidad {n(a['sensibilidad'] * 100, 1)} %. VPP {n(a['VPP'] * 100, 1)} %")
                + "</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="nota" style="margin-bottom:14px">Resultados sobre <b>{MET["n_test"]}</b> pacientes de prueba que el modelo no vio durante el entrenamiento '
                f'({MET["n_train"]} pacientes). Mortalidad en UCI de la cohorte: {n(MET["mortalidad"] * 100)} %.</div>', unsafe_allow_html=True)

    # Figuras generadas por el cuaderno, mostradas de dos en dos con su pie explicativo.
    figuras = [("03_roc_modelo_vs_sofa", "Curva ROC. El modelo queda por encima del SOFA y ambos, cerca de la diagonal del azar: la discriminación es modesta."),
               ("05_calibracion", "Calibración. Tras el ajuste, los deciles de riesgo quedan alrededor de la diagonal: el porcentaje predicho corresponde a la mortalidad observada."),
               ("06_estratos_riesgo", "Estratos. La mortalidad observada aumenta de un estrato al siguiente."),
               ("09_curva_decision", "Curva de decisión. Usar el modelo aporta más beneficio neto que intensificar a todos o a ninguno en el rango central de umbrales."),
               ("07_distribucion_riesgos", "Distribución de riesgos. El modelo solo asigna valores en un rango estrecho; el grupo de la derecha son, sobre todo, pacientes con choque séptico."),
               ("02_odds_ratios", "Odds ratios. El choque séptico es el factor con mayor peso; las demás variables aportan poco individualmente.")]
    for i in range(0, len(figuras), 2):
        for col, (archivo, pie) in zip(st.columns(2, gap="large"), figuras[i:i + 2]):
            with col:
                st.image(str(FIGURAS / f"{archivo}.png"), width='stretch')
                st.caption(pie)


# Pestaña 3. Ficha técnica: coeficientes, descripción del modelo, límites de uso y privacidad.
with t_ficha:
    izq, der = st.columns(2, gap="large")

    # Tabla de coeficientes, ordenada de mayor a menor peso.
    with izq:
        filas = ""
        for nombre, c in sorted(ART["coeficientes"].items(), key=lambda kv: -abs(kv[1]["coef_por_DE"])):
            etiqueta = {"InfectionSource_desconocido": "Foco desconocido", "InfectionSource_pulmonar": "Foco pulmonar",
                        "InfectionSource_piel": "Foco piel", "InfectionSource_urinaria": "Foco urinario"}.get(nombre, M.etiqueta(nombre))
            filas += (f'<tr><td>{escape(etiqueta)}</td><td class="num">{n(c["coef_por_DE"], 3)}</td>'
                      f'<td class="num">{n(c["OR_por_DE"], 3)}</td><td class="num">{n(c["OR_por_unidad"], 3)}</td></tr>')
        st.markdown('<div class="tarjeta"><div class="rotulo">Coeficientes del modelo</div><table class="ficha"><tr><th>Variable</th>'
                    '<th style="text-align:right">Coeficiente por DE</th><th style="text-align:right">OR por DE</th><th style="text-align:right">OR por unidad</th></tr>'
                    + filas + f'</table><div class="nota">Intercepto: {n(ART["intercepto"], 4)}. Los focos de infección se comparan contra el foco abdominal. '
                    'DE: desviación estándar. OR: odds ratio.</div></div>', unsafe_allow_html=True)

    # Descripción del modelo, límites de uso y privacidad.
    with der:
        v = ART["versiones"]
        st.markdown(f"""<div class="tarjeta"><div class="rotulo">Ficha del modelo</div><table class="ficha">
        <tr><td><b>Uso previsto</b></td><td>Apoyo a la estratificación de riesgo al ingreso a UCI en adultos con sepsis.</td></tr>
        <tr><td><b>Desenlace</b></td><td>Mortalidad durante la estancia en UCI.</td></tr>
        <tr><td><b>Algoritmo</b></td><td>Regresión logística con regularización L1 (C = 0,1), clases balanceadas y calibración de Platt.</td></tr>
        <tr><td><b>Entradas</b></td><td>11 variables de la hora 0, todas obligatorias. La imputación KNN (k = 20) se usó solo en el entrenamiento.</td></tr>
        <tr><td><b>Datos</b></td><td>{MET['n_utilizable']} registros con desenlace de un conjunto de {MET['n_total']} suministrado por la AMCI; procedencia y anonimización no documentadas.</td></tr>
        <tr><td><b>Validación</b></td><td>Partición estratificada 75/25, validación cruzada de 5 particiones. Sin validación externa.</td></tr>
        <tr><td><b>Entrenado</b></td><td>{MET['fecha_entrenamiento']} · scikit-learn {v['scikit-learn']} · Python {v['python']}</td></tr>
        </table></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="tarjeta"><div class="rotulo">Límites de uso</div>
        <div style="font-size:.88rem; line-height:1.6; color:#3D4B59">
        Discriminación baja: no separa de forma confiable a pacientes individuales.<br>
        Entrenado con un solo conjunto de datos, posiblemente sintético; no se ha validado en una UCI real.<br>
        Solo usa información del ingreso; no incorpora la evolución del paciente.<br>
        No aplica a población pediátrica ni a pacientes fuera de UCI.<br>
        Requiere reentrenamiento, validación local y aval ético antes de cualquier uso asistencial.</div></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="tarjeta"><div class="rotulo">Privacidad</div><div style="font-size:.88rem; line-height:1.6; color:#3D4B59">
        La aplicación no solicita identificadores ni almacena los valores digitados.</div></div>""", unsafe_allow_html=True)