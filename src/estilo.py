"""Hoja de estilos y componentes HTML de la interfaz (aspecto de software clínico: sobrio, legible, sin adornos)."""
from html import escape

COLOR = {"tinta": "#16212E", "tenue": "#5B6B7A", "borde": "#DDE3E9", "fondo": "#F4F6F8", "marino": "#12324A",
         "acento": "#0F6E77", "sube": "#A63D32", "baja": "#2F7D7A"}
COLOR_ESTRATO = {"Menor riesgo": "#2F7D7A", "Riesgo intermedio": "#B7791F", "Mayor riesgo": "#A63D32"}

CSS = """
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {visibility: hidden; height: 0;}
header[data-testid="stHeader"] {background: transparent; height: 0;}
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1240px;}
html, body, [class*="css"] {font-feature-settings: "tnum" 1, "lnum" 1;}

.barra {background: #12324A; color: #fff; border-radius: 10px; padding: 18px 26px; display: flex; justify-content: space-between; align-items: center; gap: 24px; margin-bottom: 14px;}
.barra .marca {display: flex; align-items: center; gap: 14px;}
.barra .sello {width: 38px; height: 38px; border-radius: 8px; background: #0F6E77; display: flex; align-items: center; justify-content: center;}
.barra .sello span {display: block; width: 18px; height: 18px; background: linear-gradient(#fff,#fff) center/100% 4px no-repeat, linear-gradient(#fff,#fff) center/4px 100% no-repeat;}
.barra h1 {font-size: 1.22rem; font-weight: 650; margin: 0; padding: 0; color: #fff; letter-spacing: .1px;}
.barra .sub {font-size: .83rem; color: #B9CAD8; margin-top: 2px;}
.barra .meta {text-align: right; font-size: .76rem; color: #B9CAD8; line-height: 1.5; white-space: nowrap;}
.barra .meta b {color: #fff; font-weight: 600;}
.aviso-uso {background: #fff; border: 1px solid #DDE3E9; border-left: 4px solid #B7791F; border-radius: 8px; padding: 10px 16px; font-size: .85rem; color: #3D4B59; margin-bottom: 10px;}

.stTabs [data-baseweb="tab-list"] {gap: 4px; border-bottom: 1px solid #DDE3E9;}
.stTabs [data-baseweb="tab"] {font-weight: 600; font-size: .93rem; color: #5B6B7A; padding: 10px 18px;}
.stTabs [aria-selected="true"] {color: #12324A;}

.tarjeta {background: #fff; border: 1px solid #DDE3E9; border-radius: 10px; padding: 20px 22px; margin-bottom: 14px;}
.rotulo {font-size: .72rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #5B6B7A; margin-bottom: 10px;}
.seccion {font-size: .74rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #0F6E77; border-bottom: 1px solid #DDE3E9; padding-bottom: 6px; margin: 6px 0 10px 0;}

.resultado {display: flex; align-items: flex-end; gap: 22px; flex-wrap: wrap;}
.resultado .cifra {font-size: 3.5rem; font-weight: 700; line-height: 1; color: #16212E; letter-spacing: -1px;}
.resultado .cifra small {font-size: 1.5rem; font-weight: 600; color: #5B6B7A; margin-left: 2px;}
.pastilla {display: inline-block; color: #fff; font-weight: 650; font-size: .86rem; padding: 5px 13px; border-radius: 999px; letter-spacing: .02em;}
.resultado .lectura {font-size: .9rem; color: #3D4B59; line-height: 1.5; flex: 1; min-width: 240px;}

.escala {position: relative; height: 62px; margin: 22px 4px 4px 4px;}
.escala .pista {position: absolute; top: 22px; left: 0; right: 0; height: 12px; border-radius: 6px; overflow: hidden; display: flex;}
.escala .marca {position: absolute; top: 12px; width: 4px; height: 32px; background: #16212E; border-radius: 2px; transform: translateX(-2px); box-shadow: 0 0 0 2px #fff;}
.escala .tope {position: absolute; top: 40px; font-size: .72rem; color: #5B6B7A; transform: translateX(-50%); white-space: nowrap;}
.escala .zona {position: absolute; top: 0; font-size: .72rem; font-weight: 600; transform: translateX(-50%); white-space: nowrap;}

.fila-c {display: grid; grid-template-columns: minmax(150px, 34%) 1fr 92px; align-items: center; gap: 12px; padding: 7px 0; border-bottom: 1px solid #EEF1F4;}
.fila-c:last-child {border-bottom: none;}
.fila-c .n {font-size: .88rem; color: #16212E; font-weight: 600; line-height: 1.25;}
.fila-c .n span {display: block; font-weight: 400; font-size: .78rem; color: #5B6B7A;}
.fila-c .b {position: relative; height: 14px; background: linear-gradient(#DDE3E9,#DDE3E9) center/1px 100% no-repeat;}
.fila-c .b i {position: absolute; top: 2px; height: 10px; border-radius: 3px;}
.fila-c .v {font-size: .86rem; font-weight: 650; text-align: right; color: #16212E;}
.leyenda {font-size: .78rem; color: #5B6B7A; margin-top: 10px; display: flex; gap: 18px; flex-wrap: wrap;}
.leyenda i {display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px;}

.nota {font-size: .84rem; color: #3D4B59; background: #F4F6F8; border: 1px solid #DDE3E9; border-radius: 8px; padding: 10px 14px; margin-top: 8px; line-height: 1.45;}
.nota.alerta {border-left: 4px solid #B7791F; background: #FFFBF2;}
.nota.error {border-left: 4px solid #A63D32; background: #FDF5F4;}
.vacio {text-align: center; color: #5B6B7A; padding: 70px 30px; font-size: .95rem; line-height: 1.6;}
.vacio b {display: block; color: #16212E; font-size: 1.05rem; margin-bottom: 6px;}

.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 14px;}
.kpi {background: #fff; border: 1px solid #DDE3E9; border-radius: 10px; padding: 14px 16px;}
.kpi .t {font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #5B6B7A;}
.kpi .x {font-size: 1.7rem; font-weight: 700; color: #16212E; margin: 4px 0 2px 0;}
.kpi .d {font-size: .78rem; color: #5B6B7A; line-height: 1.35;}

.msg {max-width: 88%; padding: 12px 16px; border-radius: 10px; font-size: .92rem; line-height: 1.55; margin-bottom: 10px; border: 1px solid #DDE3E9;}
.msg.u {margin-left: auto; background: #12324A; color: #fff; border-color: #12324A;}
.msg.a {background: #fff; color: #16212E;}
.msg .quien {font-size: .68rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; opacity: .7; margin-bottom: 4px;}
.msg .fuentes {font-size: .76rem; color: #5B6B7A; margin-top: 10px; padding-top: 8px; border-top: 1px solid #EEF1F4;}

table.ficha {width: 100%; border-collapse: collapse; font-size: .88rem;}
table.ficha th {text-align: left; font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; color: #5B6B7A; border-bottom: 1px solid #DDE3E9; padding: 8px 10px;}
table.ficha td {padding: 8px 10px; border-bottom: 1px solid #EEF1F4; vertical-align: top; color: #16212E;}
table.ficha td.num {text-align: right; font-variant-numeric: tabular-nums;}
div[data-testid="stForm"] {background: #fff; border: 1px solid #DDE3E9; border-radius: 10px; padding: 18px 20px;}
.stButton > button, .stFormSubmitButton > button {border-radius: 8px; font-weight: 600;}
</style>
"""


def n(x: float, d: int = 1) -> str:
    """Número con coma decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def barra(fecha: str, version_sklearn: str) -> str:
    return f"""<div class="barra"><div class="marca"><div class="sello"><span></span></div><div>
    <h1>Estratificación de riesgo en sepsis · UCI adultos</h1>
    <div class="sub">Probabilidad estimada de mortalidad en UCI con datos del ingreso</div></div></div>
    <div class="meta"><b>Prototipo académico</b><br>Modelo: regresión logística L1 · {escape(fecha)}<br>Universidad Autónoma de Occidente</div></div>
    <div class="aviso-uso"><b>Herramienta de apoyo, no de decisión.</b> No sustituye el juicio clínico ni debe usarse para limitar o negar atención. No ingrese datos que identifiquen al paciente.</div>"""


def escala(riesgo: float, cortes: dict, lo: float = 0.30, hi: float = 0.75) -> str:
    """Regla horizontal con los tres estratos y la posición del paciente."""
    pos = lambda p: max(0.0, min(100.0, (p - lo) / (hi - lo) * 100))
    a, b = pos(cortes["bajo"]), pos(cortes["alto"])
    c = COLOR_ESTRATO
    return f"""<div class="escala">
      <div class="zona" style="left:{a / 2:.1f}%; color:{c['Menor riesgo']}">Menor</div>
      <div class="zona" style="left:{(a + b) / 2:.1f}%; color:{c['Riesgo intermedio']}">Intermedio</div>
      <div class="zona" style="left:{(b + 100) / 2:.1f}%; color:{c['Mayor riesgo']}">Mayor</div>
      <div class="pista"><div style="width:{a:.1f}%; background:{c['Menor riesgo']}; opacity:.85"></div>
        <div style="width:{b - a:.1f}%; background:{c['Riesgo intermedio']}; opacity:.85; border-left:2px solid #fff; border-right:2px solid #fff"></div>
        <div style="flex:1; background:{c['Mayor riesgo']}; opacity:.85"></div></div>
      <div class="marca" style="left:{pos(riesgo):.1f}%"></div>
      <div class="tope" style="left:0; transform:none">{lo * 100:.0f} %</div>
      <div class="tope" style="left:{a:.1f}%">{n(cortes['bajo'] * 100)} %</div>
      <div class="tope" style="left:{b:.1f}%">{n(cortes['alto'] * 100)} %</div>
      <div class="tope" style="left:100%; transform:translateX(-100%)">{hi * 100:.0f} %</div></div>"""


def filas_contribucion(filas: list[dict], formato) -> str:
    """Barras divergentes: a la derecha lo que sube el riesgo, a la izquierda lo que lo baja."""
    tope = max(max(abs(f["efecto_pp"]) for f in filas), 1.0)
    html = ""
    for f in filas:
        ancho = abs(f["efecto_pp"]) / tope * 50
        sube = f["efecto_pp"] >= 0
        estilo = (f"left:50%; width:{ancho:.1f}%; background:{COLOR['sube']}" if sube
                  else f"left:{50 - ancho:.1f}%; width:{ancho:.1f}%; background:{COLOR['baja']}")
        detalle = formato(f["variable"], f["valor_usado"])        # (antes se añadía "· estimado" a los valores imputados)
        valor = "sin efecto" if abs(f["efecto_pp"]) < 0.05 else f"{'+' if sube else '−'}{n(abs(f['efecto_pp']))} pts"
        html += (f'<div class="fila-c"><div class="n">{escape(f["etiqueta"])}<span>{escape(detalle)}</span></div>'
                 f'<div class="b"><i style="{estilo}"></i></div><div class="v">{valor}</div></div>')
    html += (f'<div class="leyenda"><span><i style="background:{COLOR["sube"]}"></i>Aumenta el riesgo estimado</span>'
             f'<span><i style="background:{COLOR["baja"]}"></i>Lo reduce</span>'
             '<span>pts = puntos porcentuales frente al promedio de la cohorte</span></div>')
    return html


def kpi(titulo: str, cifra: str, detalle: str) -> str:
    return f'<div class="kpi"><div class="t">{escape(titulo)}</div><div class="x">{cifra}</div><div class="d">{detalle}</div></div>'


def mensaje(rol: str, texto: str, fuentes: list | None = None, modo: str | None = None) -> str:
    cuerpo = escape(texto).replace("\n\n", "<br><br>").replace("\n", "<br>")
    pie = ""
    if fuentes:
        pie = '<div class="fuentes">Fuentes consultadas: ' + " · ".join(escape(f["titulo"]) for f in fuentes) + \
              (f' &nbsp;|&nbsp; Modo {escape(modo)}' if modo else "") + "</div>"
    quien = "Usted" if rol == "u" else "Asistente de interpretación"
    return f'<div class="msg {rol}"><div class="quien">{quien}</div>{cuerpo}{pie}</div>'
