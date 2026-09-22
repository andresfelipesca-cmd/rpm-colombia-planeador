# -*- coding: utf-8 -*-
"""
app.py - Planeador de Producción RPM Colombia

Ejecución:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from rpm_core import (
    CAP_MINUTOS_DIA,
    MACHINES,
    analisis_cuello_de_botella,
    analisis_sensibilidad,
    analisis_sensibilidad_imputacion,
    calibrar_demanda_extendida,
    cargar_datos,
    cargar_datos_extendido,
    clusterizar_maquinas,
    clusterizar_productos,
    comparar_matriz_publicada_reconstruida,
    concordancia_kmeans_ward_maquinas,
    construir_cij_con_imputacion,
    demanda_entera,
    elbow_silhouette,
    metricas_concentracion_demanda,
    monte_carlo_demanda,
    politica_empirica,
    prueba_estadistica_clusters,
    resolver_asignacion,
    resolver_multidia,
    soporte_observado_solucion,
    validacion_imputacion_repetida,
)

st.set_page_config(
    page_title="RPM Colombia | Planeador de producción",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Identidad visual: técnica, sobria y con contraste accesible
# ---------------------------------------------------------------------------
BG = "#0B1220"
CARD = "#111827"
CARD_2 = "#172033"
GRID = "#2A364B"
TEXT = "#E5E7EB"
MUTED = "#94A3B8"
BLUE = "#5F7D9D"
TEAL = "#4D8B82"
AMBER = "#B49A5A"
RED = "#A65F64"
SLATE = "#718096"
LIGHT = "#CBD5E1"
GREEN = "#688F78"

# Referencia de revalidación multiperiodo documentada en la tesis.
# Estos valores pertenecen a una corrida separada y nunca sustituyen la
# ejecución en vivo de la aplicación.
MULTI_TESIS_UB_MIN = 34234.87
MULTI_TESIS_LB_MIN = 34071.20
MULTI_TESIS_GAP_PCT = 0.478

st.markdown(
    f"""
    <style>
      .stApp {{ background: {BG}; }}
      .block-container {{ padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1500px; }}
      h1, h2, h3 {{ letter-spacing: -0.015em; }}
      h1 {{ font-weight: 650 !important; }}
      div[data-testid="stMetric"] {{
          background: linear-gradient(180deg, {CARD_2} 0%, {CARD} 100%);
          border: 1px solid {GRID}; border-radius: 12px;
          padding: 14px 16px 12px 16px;
      }}
      div[data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}
      div[data-testid="stMetricValue"] {{ color: {TEXT} !important; }}
      div[data-baseweb="tab-list"] {{ gap: 8px; }}
      button[data-baseweb="tab"] {{
          background: {CARD}; border: 1px solid {GRID}; border-radius: 8px;
          padding-left: 14px; padding-right: 14px;
      }}
      .technical-note {{
          background: {CARD}; border-left: 3px solid {BLUE};
          padding: 12px 14px; border-radius: 4px; color: {LIGHT};
          margin: 8px 0 14px 0;
      }}
      .badge {{ display:inline-block; padding:4px 10px; border-radius:999px;
                font-size:0.82rem; font-weight:600; margin-right:6px; }}
      .ok {{ background:rgba(91,154,117,.17); color:#9ED0AD; border:1px solid rgba(91,154,117,.4); }}
      .warn {{ background:rgba(212,167,44,.15); color:#E7C96B; border:1px solid rgba(212,167,44,.4); }}
      .bad {{ background:rgba(201,93,99,.15); color:#E8A3A7; border:1px solid rgba(201,93,99,.4); }}
      .muted {{ color:{MUTED}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_num(x, dec=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/D"
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def badge(texto: str, tipo: str = "ok"):
    st.markdown(f'<span class="badge {tipo}">{texto}</span>', unsafe_allow_html=True)


def fig_layout(fig, title=None, height=390, legend=True, **kwargs):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT, size=12),
        height=height,
        margin=kwargs.pop("margin", dict(l=45, r=25, t=58 if title else 25, b=45)),
        showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=LIGHT)),
        hoverlabel=dict(bgcolor="#0F172A", font_color=TEXT),
        **kwargs,
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left", font=dict(size=16, color=TEXT)))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


@st.cache_data(show_spinner=False)
def cargar_referencia_monte_carlo():
    """Carga el resultado N=1000 documentado en la tesis sin recalcularlo."""
    path = Path(__file__).resolve().parent / "results" / "monte_carlo_reference.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def ejecutar_monte_carlo_cache(cij, demanda, setup, n_escenarios, seed, low, high, cap_minutos, tiempo_limite_seg):
    """Evita repetir una corrida pesada con exactamente los mismos parametros."""
    return monte_carlo_demanda(
        cij, demanda, setup, n_escenarios=int(n_escenarios), seed=int(seed),
        low=float(low), high=float(high), cap_minutos=cap_minutos,
        tiempo_limite_seg=int(tiempo_limite_seg),
    )


def grafico_pareto_demanda(demanda: pd.Series):
    d = demanda.sort_values(ascending=False)
    cum = d.cumsum() / d.sum() * 100
    x = np.arange(1, len(d) + 1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=x, y=d.values, marker_color=BLUE, name="Demanda"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=cum.values, mode="lines", line=dict(color=TEAL, width=2.6),
                             name="Acumulado"), secondary_y=True)
    fig.add_hline(y=80, line_dash="dot", line_color=AMBER, secondary_y=True)
    fig.update_xaxes(title="Referencias ordenadas por demanda")
    fig.update_yaxes(title="Unidades/día", secondary_y=False)
    fig.update_yaxes(title="Demanda acumulada (%)", range=[0, 105], secondary_y=True)
    return fig_layout(fig, "Concentración de la demanda: análisis de Pareto", height=390)


def grafico_cobertura_maquina(mask_observado: pd.DataFrame):
    pct = mask_observado.mean(axis=0) * 100
    overall = mask_observado.to_numpy().mean() * 100
    fig = go.Figure()
    fig.add_bar(x=pct.index, y=pct.values, marker_color=BLUE,
                text=[f"{v:.1f}%" for v in pct.values], textposition="outside")
    fig.add_hline(y=overall, line_dash="dash", line_color=TEAL,
                  annotation_text=f"Cobertura global {overall:.1f}%", annotation_font_color=TEAL)
    fig.update_yaxes(title="Celdas observadas (%)", range=[0, 110])
    fig.update_xaxes(title="Máquina")
    return fig_layout(fig, "Cobertura empírica de Cij por máquina", height=360, legend=False)


def grafico_utilizacion(util: pd.Series, caps: pd.Series):
    pct = util / caps * 100
    orden = pct.sort_values().index
    colors = [RED if pct[m] >= 98 else (AMBER if pct[m] >= 90 else BLUE) for m in orden]
    fig = go.Figure(go.Bar(
        y=orden, x=pct.loc[orden], orientation="h", marker_color=colors,
        text=[f"{pct[m]:.1f}%" for m in orden], textposition="outside",
    ))
    fig.add_vline(x=100, line_dash="dash", line_color=RED)
    fig.update_xaxes(title="Utilización de capacidad (%)", range=[0, 108])
    fig.update_yaxes(title="")
    return fig_layout(fig, "Utilización por máquina", height=430, legend=False)


def grafico_descomposicion(resultado: dict):
    prod = resultado["tiempo_produccion_min"]
    setup_t = resultado["tiempo_setup_min"]
    slack = resultado["holgura_min"].clip(lower=0)
    fig = go.Figure()
    fig.add_bar(x=MACHINES, y=prod[MACHINES], name="Producción", marker_color=BLUE)
    fig.add_bar(x=MACHINES, y=setup_t[MACHINES], name="Alistamientos", marker_color=AMBER)
    fig.add_bar(x=MACHINES, y=slack[MACHINES], name="Holgura", marker_color=SLATE)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Minutos por día")
    fig.update_xaxes(title="Máquina")
    return fig_layout(fig, "Composición de la capacidad: producción, setup y holgura", height=430)


def heatmap_utilizacion(tabla_pct: pd.DataFrame, titulo: str):
    colorscale = [[0.0, "#223047"], [0.60, BLUE], [0.90, AMBER], [1.0, RED]]
    fig = go.Figure(go.Heatmap(
        z=tabla_pct.values,
        x=tabla_pct.columns.tolist(),
        y=[f"Día {d}" for d in tabla_pct.index],
        zmin=0, zmax=100,
        colorscale=colorscale,
        text=tabla_pct.round(1).values,
        texttemplate="%{text}%",
        colorbar=dict(title="%"),
    ))
    fig.update_yaxes(autorange="reversed")
    return fig_layout(fig, titulo, height=max(280, 120 + 46 * len(tabla_pct)), legend=False)


def long_plan(asignacion: pd.DataFrame):
    plan = asignacion.stack().reset_index()
    plan.columns = ["producto", "maquina", "unidades"]
    plan = plan[plan["unidades"] > 0].copy()
    return plan.sort_values(["maquina", "unidades"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.title("Planeador de producción | RPM Colombia")
st.caption(
    "Herramienta de apoyo a la decisión basada en programación lineal entera mixta. "
    "La aplicación separa datos, diagnóstico, optimización y análisis de estabilidad para "
    "que cada resultado pueda ser trazado y defendido técnicamente."
)

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
st.sidebar.header("Datos")
archivo = st.sidebar.file_uploader("Excel del proyecto", type=["xlsx"])
usar_ejemplo = st.sidebar.checkbox("Usar AnexodeDatosxlsx.xlsx", value=archivo is None)

st.sidebar.header("Optimización")
cap_min = st.sidebar.number_input("Capacidad diaria por máquina (min)", 60, 2880, CAP_MINUTOS_DIA, 60)
tiempo_limite = st.sidebar.slider("Tiempo máximo por MILP (s)", 10, 300, 60)
n_clusters = st.sidebar.slider("Clusters de productos", 2, 5, 3)

st.sidebar.markdown("---")
correr = st.sidebar.button("Calcular plan diario", type="primary", use_container_width=True)

if archivo is not None:
    excel_bytes = archivo.getvalue()
    def source():
        return io.BytesIO(excel_bytes)
elif usar_ejemplo:
    def source():
        return "AnexodeDatosxlsx.xlsx"
else:
    st.info("Sube el Excel o activa el archivo de ejemplo.")
    st.stop()

try:
    datos = cargar_datos(source())
    cij_publicada, demanda, setup = datos["cij"], datos["demanda"], datos["setup"]
    audit = construir_cij_con_imputacion(source(), productos=cij_publicada.index)
    comp_matriz = comparar_matriz_publicada_reconstruida(cij_publicada, audit)
except Exception as exc:
    st.error(f"No fue posible cargar el archivo con la estructura esperada: {exc}")
    st.stop()

modo_cij = st.sidebar.radio(
    "Fuente de tiempos Cij",
    ["Matriz publicada (reproduce tesis)", "Reconstruida desde registros crudos"],
    index=0,
    help=(
        "La matriz publicada reproduce Z*=11.547,41. La reconstruida calcula promedios observados y "
        "completa faltantes con la media por máquina; se ofrece como ruta de auditoría, no como sustitución silenciosa."
    ),
)
if modo_cij.startswith("Matriz publicada"):
    cij = cij_publicada.copy()
    st.sidebar.caption("Modo reproducibilidad: usa la matriz de MODELO PL asociada al resultado principal de la tesis.")
else:
    cij = audit["cij"].reindex(index=cij_publicada.index, columns=MACHINES).copy()
    st.sidebar.caption("Modo auditoría: reconstruye Cij desde las hojas crudas e imputa faltantes por media de máquina.")

conc = metricas_concentracion_demanda(demanda)

# KPIs de entrada
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Referencias", len(cij))
k2.metric("Máquinas", len(MACHINES))
k3.metric("Demanda diaria", f"{fmt_num(demanda.sum(), 0)} unid.")
k4.metric("Cobertura observada de Cij", f"{audit['R_C']*100:.1f}%", f"{audit['n_imputadas']} celdas imputadas")
k5.metric("Fuente Cij", "Publicada" if modo_cij.startswith("Matriz publicada") else "Reconstruida")

with st.expander("Calidad y trazabilidad de los datos", expanded=False):
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Celdas observadas", audit["n_observadas"])
    q2.metric("Celdas imputadas", audit["n_imputadas"])
    q3.metric("Referencias que acumulan 80%", conc["n80"])
    q4.metric("Gini de demanda", f"{conc['gini']:.3f}", help="0 indica distribución uniforme; valores mayores reflejan más concentración.")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(grafico_cobertura_maquina(audit["mask_observado"]), use_container_width=True)
    with c2:
        st.plotly_chart(grafico_pareto_demanda(demanda), use_container_width=True)

    st.markdown(
        '<div class="technical-note"><b>Control de trazabilidad.</b> La matriz publicada en la hoja MODELO PL '
        'reproduce el resultado principal de la tesis. La reconstrucción desde las hojas crudas permite auditar '
        'qué celdas fueron observadas y cuáles requieren imputación. Las dos rutas se mantienen separadas para no '
        'confundir reproducción del resultado con reconstrucción del dato.</div>',
        unsafe_allow_html=True,
    )
    st.write(
        f"Diferencia absoluta media entre la matriz publicada y la reconstrucción: "
        f"**{comp_matriz['mae_total']:.4f} min/unidad**. En celdas observadas: "
        f"**{comp_matriz['mae_observadas']:.4f}**; en celdas imputadas: "
        f"**{comp_matriz['mae_imputadas']:.4f}**. Se detectan "
        f"**{comp_matriz['n_celdas_distintas_1e6']} celdas** con diferencia superior a 10⁻⁶."
    )
    st.caption(
        "La existencia de diferencias no invalida el resultado publicado: indica que reproducir la matriz de MODELO PL "
        "y reconstruir Cij desde los registros crudos son dos ejercicios distintos. La app permite seleccionar explícitamente cuál se optimiza."
    )

    if st.button("Revalidar métodos de imputación (100 particiones)"):
        with st.spinner("Ocultando repetidamente 20% de celdas observadas y comparando cuatro métodos..."):
            val_imp = validacion_imputacion_repetida(source(), cij.index, n_repeticiones=100, fraccion_oculta=0.20, seed=42)
        st.session_state["validacion_imputacion"] = val_imp
    val_imp = st.session_state.get("validacion_imputacion")
    if val_imp is not None:
        resumen_imp = val_imp["resumen"].copy()
        best = resumen_imp.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Mejor método repetido", best["metodo"])
        m2.metric("MAE medio", f"{best['mae_media']:.4f} min/u")
        m3.metric("Veces con menor MAE", f"{int(best['veces_mejor_mae'])}/100")
        fig = go.Figure()
        fig.add_bar(
            x=resumen_imp["metodo"], y=resumen_imp["mae_media"], marker_color=[TEAL, BLUE, SLATE, AMBER],
            error_y=dict(type="data", symmetric=False,
                         array=resumen_imp["mae_ci95_high"]-resumen_imp["mae_media"],
                         arrayminus=resumen_imp["mae_media"]-resumen_imp["mae_ci95_low"]),
            text=[f"{v:.4f}" for v in resumen_imp["mae_media"]], textposition="outside",
        )
        fig.update_yaxes(title="MAE (min/unidad)")
        st.plotly_chart(fig_layout(fig, "Revalidación repetida de la imputación", height=390, legend=False), use_container_width=True)
        st.caption(
            "Esta revalidación repetida es un análisis adicional. No reemplaza la tabla histórica de la tesis; "
            "sirve para comprobar si la elección del método depende excesivamente de una única partición aleatoria."
        )

st.divider()

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
st.header("Caracterización no supervisada")
tab_prod, tab_maq = st.tabs(["Productos", "Máquinas"])

with tab_prod:
    cv = cij.std(axis=1) / cij.mean(axis=1)
    feats_prod = pd.DataFrame({"demanda": demanda, "cv_tiempo": cv})
    tabla_k_prod = elbow_silhouette(feats_prod, k_range=range(2, 7))
    clusters = clusterizar_productos(cij, demanda, n_clusters=n_clusters)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=tabla_k_prod["k"], y=tabla_k_prod["inercia"], mode="lines+markers",
                                   line=dict(color=BLUE, width=2.5), marker=dict(size=8)))
        fig.update_xaxes(title="Número de clusters k")
        fig.update_yaxes(title="Inercia")
        st.plotly_chart(fig_layout(fig, "Método del codo", height=330, legend=False), use_container_width=True)
    with c2:
        best_k = int(tabla_k_prod.loc[tabla_k_prod["silhouette"].idxmax(), "k"])
        fig = go.Figure(go.Scatter(x=tabla_k_prod["k"], y=tabla_k_prod["silhouette"], mode="lines+markers",
                                   line=dict(color=TEAL, width=2.5), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=[best_k], y=[tabla_k_prod["silhouette"].max()], mode="markers",
                                 marker=dict(size=14, color=AMBER), name=f"Máximo: k={best_k}"))
        fig.update_xaxes(title="Número de clusters k")
        fig.update_yaxes(title="Silhouette")
        st.plotly_chart(fig_layout(fig, "Separación de clusters", height=330), use_container_width=True)

    fig = go.Figure()
    role_colors = {
        "Producción continua (alto volumen)": TEAL,
        "Rotación media": BLUE,
        "Comodín / baja escala": AMBER,
    }
    for rol, sub in clusters.groupby("rol_sugerido"):
        fig.add_trace(go.Scatter(
            x=sub["demanda"], y=sub["cv_tiempo"], mode="markers", name=rol,
            marker=dict(size=9, color=role_colors.get(rol, SLATE), opacity=0.82,
                        line=dict(color=BG, width=0.5)),
            text=sub.index,
            hovertemplate="%{text}<br>Demanda=%{x:.0f}<br>CV=%{y:.3f}<extra></extra>",
        ))
    fig.update_xaxes(title="Demanda diaria (unidades)")
    fig.update_yaxes(title="Coeficiente de variación del tiempo")
    st.plotly_chart(fig_layout(fig, "Mapa de productos: volumen y sensibilidad a la máquina", height=430), use_container_width=True)

    test_dem = prueba_estadistica_clusters(clusters["demanda"], clusters["cluster"])
    test_cv = prueba_estadistica_clusters(clusters["cv_tiempo"], clusters["cluster"])
    s1, s2 = st.columns(2)
    s1.metric("Kruskal-Wallis demanda", f"H={test_dem['estadistico_H']:.2f}", f"ε²={test_dem['epsilon2']:.3f}")
    s2.metric("Kruskal-Wallis variabilidad", f"H={test_cv['estadistico_H']:.2f}", f"ε²={test_cv['epsilon2']:.3f}")
    st.caption(
        "Estas pruebas describen separación sobre variables usadas para construir los clusters; no deben interpretarse "
        "como validación externa del agrupamiento. El tamaño de efecto ε² complementa el valor p."
    )

with tab_maq:
    feats_maq = pd.DataFrame({
        "setup_promedio": setup,
        "tiempo_operacion_promedio": cij.mean(axis=0),
        "varianza_tiempo": cij.var(axis=0),
    })
    tabla_k_maq = elbow_silhouette(feats_maq, k_range=range(2, 6))
    n_clusters_maq = st.slider("Clusters de máquinas", 2, 5, 3, key="clusters_maq")
    clusters_maq = clusterizar_maquinas(cij, setup, n_clusters=n_clusters_maq)
    concord = concordancia_kmeans_ward_maquinas(cij, setup, n_clusters=n_clusters_maq)

    m1, m2, m3 = st.columns(3)
    m1.metric("ARI K-Means vs Ward", f"{concord['ari']:.3f}")
    m2.metric("Silhouette K-Means", f"{concord['silhouette_kmeans']:.3f}")
    m3.metric("Silhouette Ward", f"{concord['silhouette_ward']:.3f}")
    st.dataframe(
        clusters_maq[["setup_promedio", "tiempo_operacion_promedio", "varianza_tiempo", "grupo_sugerido"]],
        use_container_width=True,
    )
    st.caption("ARI=1 significa concordancia exacta entre las dos particiones comparadas; no demuestra que exista una única partición verdadera.")

st.divider()

# ---------------------------------------------------------------------------
# Resolver plan base
# ---------------------------------------------------------------------------
if correr:
    with st.spinner("Resolviendo el MILP diario..."):
        try:
            resultado = resolver_asignacion(cij, demanda_entera(demanda), setup, cap_min, tiempo_limite)
            st.session_state["resultado_diario"] = resultado
        except Exception as exc:
            st.error(f"El modelo no pudo resolverse: {exc}")

resultado = st.session_state.get("resultado_diario")
if resultado is None:
    st.info("Selecciona **Calcular plan diario** para habilitar los análisis de decisión.")
    st.stop()

caps = resultado["capacidad_min"]
util_pct = resultado["utilizacion_min"] / caps * 100
z = resultado["valor_objetivo"]
z_prod = float(resultado["tiempo_produccion_min"].sum())
z_setup = float(resultado["tiempo_setup_min"].sum())
setup_share = 100 * z_setup / z if z else np.nan
slack_total = float(resultado["holgura_min"].sum())
balance_sd = float(util_pct.std(ddof=0))
soporte = soporte_observado_solucion(resultado["asignacion"], audit["mask_observado"])

st.header("Plan diario optimizado")
ka, kb, kc, kd, ke, kf = st.columns(6)
ka.metric("Objetivo", f"{fmt_num(z, 2)} min")
kb.metric("Producción", f"{fmt_num(z_prod, 1)} min")
kc.metric("Setups", f"{fmt_num(z_setup, 1)} min", f"{setup_share:.1f}% del objetivo")
kd.metric("Holgura agregada", f"{fmt_num(slack_total, 1)} min")
ke.metric("Utilización máxima", f"{util_pct.max():.1f}%", util_pct.idxmax())
kf.metric("Balance de carga", f"σ={balance_sd:.1f} pts")

if resultado["gap_certificado"]:
    badge("Óptimo certificado · GAP 0%", "ok")
elif resultado["gap_pct"] is not None:
    badge(f"Solución sin cierre total · GAP {resultado['gap_pct']:.2f}%", "warn")
else:
    badge(f"Estado solver: {resultado['estado']}", "warn")
if resultado.get("solver_termination", {}).get("stopped_on_time"):
    st.caption("CBC se detuvo por límite de tiempo. La solución mostrada es incumbente factible si los residuos son nulos; no debe presentarse como óptimo certificado.")

r = resultado["residuos"]
if max(r.values()) <= 1e-6:
    badge("Residuos de factibilidad ≤ 10⁻⁶", "ok")
else:
    badge("Revisar residuos de factibilidad", "bad")

st.markdown("")

plan_tab, comp_tab, sens_tab, cuello_tab, multi_tab, mc_tab = st.tabs([
    "Plan y capacidad", "Vs. política", "Sensibilidad", "Capacidad crítica", "Multi-día", "Monte Carlo"
])

# ---------------------------------------------------------------------------
# Tab 1: plan
# ---------------------------------------------------------------------------
with plan_tab:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(grafico_utilizacion(resultado["utilizacion_min"], caps), use_container_width=True)
    with c2:
        st.plotly_chart(grafico_descomposicion(resultado), use_container_width=True)

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Setups activos", int(resultado["n_setups"].sum()))
    s2.metric("Pares producto-máquina activos", resultado["pares_activos"])
    s3.metric("Referencias divididas", resultado["referencias_divididas"])
    s4.metric("Pares con Cij observado", f"{soporte['share_pares_observados_pct']:.1f}%",
              f"{soporte['pares_imputados']} pares imputados")
    s5.metric("Unidades con Cij observado", f"{soporte['share_unidades_observadas_pct']:.1f}%",
              f"{fmt_num(soporte['unidades_en_celdas_imputadas'], 0)} unid. sobre Cij imputado")

    if soporte["unidades_en_celdas_imputadas"] > 0:
        st.caption(
            "La dependencia de datos imputados debe interpretarse por pares activos y también por volumen: "
            "un único par puede concentrar muchas unidades. Por eso se reportan ambas métricas."
        )

    st.markdown(
        '<div class="technical-note"><b>Lectura operativa.</b> La utilización alta indica poco margen, pero no basta para '
        'identificar el cuello de botella. El análisis marginal de la pestaña <i>Capacidad crítica</i> vuelve a resolver '
        'el MILP para medir cuánto vale realmente un minuto adicional en cada máquina.</div>',
        unsafe_allow_html=True,
    )

    plan = long_plan(resultado["asignacion"])
    plan["tiempo_unitario_min"] = [cij.loc[p, m] for p, m in zip(plan["producto"], plan["maquina"])]
    plan["origen_Cij"] = ["Observado" if audit["mask_observado"].loc[p, m] else "Imputado"
                           for p, m in zip(plan["producto"], plan["maquina"])]
    st.subheader("Asignaciones recomendadas")
    st.dataframe(plan, use_container_width=True, hide_index=True)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        plan.to_excel(writer, sheet_name="Plan", index=False)
        pd.DataFrame({
            "utilizacion_min": resultado["utilizacion_min"],
            "produccion_min": resultado["tiempo_produccion_min"],
            "setup_min": resultado["tiempo_setup_min"],
            "holgura_min": resultado["holgura_min"],
            "capacidad_min": caps,
            "utilizacion_pct": util_pct,
        }).to_excel(writer, sheet_name="Capacidad")
        pd.DataFrame([{
            "objetivo_min": z,
            "produccion_min": z_prod,
            "setup_min": z_setup,
            "setup_share_pct": setup_share,
            "holgura_total_min": slack_total,
            "balance_sd_pct": balance_sd,
            "gap_pct": resultado["gap_pct"],
            **{f"residuo_{k}": v for k, v in r.items()},
        }]).to_excel(writer, sheet_name="Metricas", index=False)
    st.download_button("Descargar plan y métricas en Excel", buffer.getvalue(), "plan_diario_rpm.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------------------------------------------------------
# Tab 2: comparación
# ---------------------------------------------------------------------------
with comp_tab:
    emp = politica_empirica(cij, demanda_entera(demanda), setup, cap_min)
    incumplidas = int(emp["demanda_incumplida"].sum()) if len(emp["demanda_incumplida"]) else 0
    if incumplidas > 0:
        st.warning(
            f"La heurística greedy no logra cubrir {incumplidas} unidades con la capacidad disponible. "
            "En ese caso no es metodológicamente correcto calcular un porcentaje de mejora sobre su objetivo."
        )
    else:
        mejora = (emp["valor_objetivo"] - z) / emp["valor_objetivo"] * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("MILP", f"{fmt_num(z, 1)} min")
        c2.metric("Política greedy integral", f"{fmt_num(emp['valor_objetivo'], 1)} min")
        c3.metric("Reducción relativa", f"{mejora:.2f}%")

        fig = go.Figure()
        fig.add_bar(x=MACHINES, y=resultado["utilizacion_min"][MACHINES], name="MILP", marker_color=TEAL)
        fig.add_bar(x=MACHINES, y=emp["utilizacion_min"][MACHINES], name="Greedy", marker_color=SLATE)
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="Minutos utilizados")
        fig.update_xaxes(title="Máquina")
        st.plotly_chart(fig_layout(fig, "Distribución de carga: MILP vs política de referencia", height=420), use_container_width=True)

    st.caption(
        "La política de referencia es una heurística reproducible, no un registro histórico de decisiones de supervisores. "
        "En esta versión también asigna unidades enteras, de modo que la comparación usa el mismo dominio físico que el MILP."
    )

# ---------------------------------------------------------------------------
# Tab 3: sensibilidad
# ---------------------------------------------------------------------------
with sens_tab:
    st.subheader("Sensibilidad a la demanda")
    st.write(
        "La demanda escalada se redondea a unidades enteras antes de resolver. Esto corrige la incompatibilidad anterior "
        "entre una demanda fraccionaria y variables de producción enteras."
    )
    if st.button("Ejecutar sensibilidad de demanda", key="run_sens"):
        with st.spinner("Resolviendo escenarios 80%, 90%, 100%, 110% y 120%..."):
            sens = analisis_sensibilidad(cij, demanda, setup, cap_min, tiempo_limite_seg=tiempo_limite)
        st.session_state["sensibilidad_demanda"] = sens
    sens = st.session_state.get("sensibilidad_demanda")
    if sens is not None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=sens["factor_demanda"] * 100, y=sens["valor_objetivo_min"], mode="lines+markers",
            line=dict(color=BLUE, width=2.6), marker=dict(size=8), name="Objetivo"
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=sens["factor_demanda"] * 100, y=sens["utilizacion_maxima_pct"], mode="lines+markers",
            line=dict(color=AMBER, width=2.2, dash="dot"), marker=dict(size=7), name="Utilización máxima"
        ), secondary_y=True)
        fig.update_xaxes(title="Demanda relativa al escenario base (%)")
        fig.update_yaxes(title="Tiempo agregado óptimo (min)", secondary_y=False)
        fig.update_yaxes(title="Utilización máxima (%)", secondary_y=True)
        st.plotly_chart(fig_layout(fig, "Respuesta del sistema ante cambios de demanda", height=430), use_container_width=True)
        st.dataframe(sens, use_container_width=True, hide_index=True)

    st.subheader("Sensibilidad a las celdas imputadas")
    if st.button("Perturbar celdas imputadas ±20%", key="run_imp_sens"):
        with st.spinner("Reoptimizando la matriz con valores imputados -20%, base y +20%..."):
            imp_sens = analisis_sensibilidad_imputacion(
                cij, audit["mask_observado"], demanda_entera(demanda), setup,
                factores=(0.8, 1.0, 1.2), cap_minutos=cap_min, tiempo_limite_seg=tiempo_limite,
            )
        st.session_state["sensibilidad_imputacion"] = imp_sens
    imp_sens = st.session_state.get("sensibilidad_imputacion")
    if imp_sens is not None:
        fig = go.Figure(go.Bar(
            x=[f"{v:+.0f}%" for v in imp_sens["variacion_imputadas_pct"]],
            y=imp_sens["valor_objetivo_min"],
            marker_color=[TEAL, BLUE, AMBER],
            text=[fmt_num(v, 1) for v in imp_sens["valor_objetivo_min"]], textposition="outside",
        ))
        fig.update_xaxes(title="Perturbación aplicada solo a celdas imputadas")
        fig.update_yaxes(title="Objetivo óptimo (min)")
        st.plotly_chart(fig_layout(fig, "Dependencia del óptimo respecto a la imputación", height=380, legend=False), use_container_width=True)
        st.dataframe(imp_sens, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 4: capacidad crítica
# ---------------------------------------------------------------------------
with cuello_tab:
    st.write(
        "Se agregan 30 minutos a una máquina por vez y se vuelve a resolver exactamente el mismo MILP. "
        "La razón ηj mide la reducción del objetivo por minuto adicional de capacidad."
    )
    delta_cap = st.number_input("Incremento de capacidad por escenario (min)", 5, 120, 30, 5)
    if st.button("Calcular valor marginal de capacidad", key="run_bottleneck"):
        with st.spinner("Reoptimizando capacidad máquina por máquina..."):
            cuello = analisis_cuello_de_botella(cij, demanda_entera(demanda), setup, cap_min, delta_cap, tiempo_limite)
        st.session_state["cuello"] = cuello
    cuello = st.session_state.get("cuello")
    if cuello is not None:
        valid = cuello.dropna(subset=["eta_min_por_min"]).sort_values("eta_min_por_min")
        fig = go.Figure(go.Bar(
            y=valid["maquina"], x=valid["eta_min_por_min"], orientation="h", marker_color=BLUE,
            text=[f"{v:.3f}" for v in valid["eta_min_por_min"]], textposition="outside",
        ))
        fig.update_xaxes(title="ηj = reducción del objetivo / minuto adicional")
        st.plotly_chart(fig_layout(fig, "Valor marginal finito de capacidad", height=410, legend=False), use_container_width=True)

        fig2 = go.Figure(go.Scatter(
            x=cuello["utilizacion_actual_pct"], y=cuello["eta_min_por_min"], mode="markers+text",
            text=cuello["maquina"], textposition="top center",
            marker=dict(size=11, color=TEAL, line=dict(color=LIGHT, width=0.5)),
        ))
        fig2.update_xaxes(title="Utilización actual (%)")
        fig2.update_yaxes(title="ηj (min/min)")
        st.plotly_chart(fig_layout(fig2, "Utilización no equivale a valor marginal", height=390, legend=False), use_container_width=True)
        st.dataframe(cuello, use_container_width=True, hide_index=True)
        if len(cuello):
            top = cuello.iloc[0]
            st.success(
                f"Mayor respuesta marginal: {top['maquina']}. Un aumento de {top['delta_capacidad_min']:.0f} min "
                f"reduce el objetivo en {top['mejora_objetivo_min']:.2f} min, η={top['eta_min_por_min']:.4f}."
            )

# ---------------------------------------------------------------------------
# Tab 5: multi-día
# ---------------------------------------------------------------------------
with multi_tab:
    c1, c2, c3, c4 = st.columns(4)
    dias_habiles_input = c1.number_input("Días hábiles para estimar demanda", 60, 300, 180, 10)
    dias_horizonte = c2.slider("Horizonte de planeación (días)", 2, 7, 3)
    tiempo_multi = c3.slider("Tiempo máximo CBC multi-día (s)", 30, 600, 600)
    gap_obj_multi_pct = c4.select_slider("GAP objetivo multi-día", options=[0.1, 0.25, 0.5, 1.0, 2.0], value=0.5, format_func=lambda x: f"{x:.2f}%")

    if st.button("Calcular plan multi-día", key="run_multi"):
        with st.spinner("Preparando catálogo completo y demanda calibrada..."):
            datos_200 = cargar_datos_extendido(source(), dias_habiles=dias_habiles_input)
            cal = calibrar_demanda_extendida(demanda, datos_200["demanda"])
            demanda_teorica = cal["demanda_calibrada"] * dias_horizonte
            demanda_horizonte = demanda_entera(demanda_teorica, 1.0, "round")
            ajuste = float(demanda_horizonte.sum() - demanda_teorica.sum())
        with st.spinner(f"Resolviendo {len(datos_200['cij'])} referencias en {dias_horizonte} días..."):
            res_multi = resolver_multidia(
                datos_200["cij"], demanda_horizonte, datos_200["setup"], dias_horizonte,
                cap_minutos=cap_min, tiempo_limite_seg=tiempo_multi,
                gap_rel=float(gap_obj_multi_pct) / 100.0,
            )
        st.session_state["resultado_multi"] = {
            "res": res_multi, "cal": cal, "ajuste": ajuste,
            "demanda_teorica": float(demanda_teorica.sum()), "demanda_entera": int(demanda_horizonte.sum()),
            "demanda_horizonte": demanda_horizonte.copy(),
            "gap_objetivo_pct": float(gap_obj_multi_pct),
            "tiempo_limite_seg": int(tiempo_multi),
        }

    multi_state = st.session_state.get("resultado_multi")
    if multi_state is not None:
        res_multi = multi_state["res"]
        gap_obj_usado = float(multi_state.get("gap_objetivo_pct", gap_obj_multi_pct))
        tiempo_usado = int(multi_state.get("tiempo_limite_seg", tiempo_multi))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Estado", res_multi.get("estado_presentacion", res_multi["estado"]))
        m2.metric("Objetivo", f"{fmt_num(res_multi['valor_objetivo'], 1)} min" if res_multi["valor_objetivo"] is not None else "N/D")
        m3.metric("Demanda entera del horizonte", fmt_num(multi_state["demanda_entera"], 0),
                  f"ajuste {multi_state['ajuste']:+.1f} unid.")
        gap_text = "0.00%" if res_multi["gap_certificado"] else (f"{res_multi['gap_pct']:.2f}%" if res_multi["gap_pct"] is not None else "N/D")
        m4.metric("GAP", gap_text)
        st.caption(
            f"Factor de calibración de demanda extendida: {multi_state['cal']['factor_sesgo']:.3f}x. "
            f"El redondeo del horizonte se reporta explícitamente porque la variable de producción es entera. "
            f"Esta corrida usó CBC con límite de {tiempo_usado} s y GAP objetivo de {gap_obj_usado:.2f}%."
        )
        lb_multi = res_multi.get("lower_bound")
        ub_multi = res_multi.get("upper_bound")
        b1, b2, b3 = st.columns(3)
        b1.metric("LB solver", f"{fmt_num(lb_multi, 1)} min" if lb_multi is not None else "N/D")
        b2.metric("UB / incumbente", f"{fmt_num(ub_multi, 1)} min" if ub_multi is not None else "N/D")
        b3.metric("Criterio GAP", f"≤ {gap_obj_usado:.2f}%")

        term = res_multi.get("solver_termination", {})
        if res_multi.get("gap_certificado"):
            st.success("Optimalidad exacta certificada por CBC para esta corrida.")
        elif term.get("stopped_on_time") and res_multi.get("factible_incumbente"):
            st.warning(
                "CBC alcanzó el límite de tiempo. Se conserva el mejor plan factible encontrado, "
                "pero no se presenta como óptimo. Interprete conjuntamente UB, LB y GAP."
            )
        elif term.get("stopped_on_gap") or res_multi.get("gap_objetivo_alcanzado"):
            st.info(
                "CBC alcanzó el criterio de GAP configurado. El plan es factible y acotado, "
                "pero no se etiqueta como óptimo exacto mientras la brecha sea distinta de cero."
            )
        elif res_multi.get("factible_incumbente"):
            st.info("Se encontró un incumbente factible, sin certificado de optimalidad exacta.")

        # Comparación transparente con la corrida de revalidación documentada.
        # Se presenta como referencia histórica y no como resultado de esta ejecución.
        st.markdown("#### Referencia de revalidación documentada en la tesis")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("UB documentado", f"{fmt_num(MULTI_TESIS_UB_MIN, 2)} min")
        r2.metric("LB documentada (raíz)", f"{fmt_num(MULTI_TESIS_LB_MIN, 2)} min")
        r3.metric("GAP conservador documentado", f"≤ {MULTI_TESIS_GAP_PCT:.3f}%")
        if ub_multi is not None:
            delta_ref = float(ub_multi) - MULTI_TESIS_UB_MIN
            r4.metric(
                "Δ UB ejecución vs. referencia",
                f"{fmt_num(delta_ref, 2)} min",
                help="Diferencia entre el incumbente de la ejecución actual y el incumbente documentado en la tesis. No es una brecha de optimalidad.",
            )
        else:
            r4.metric("Δ UB ejecución vs. referencia", "N/D")
        st.caption(
            "Referencia externa a la corrida actual: UB = 34.234,87 min, LB de raíz = 34.071,20 min "
            "y GAP conservador ≤ 0,478%. Se muestra únicamente para trazabilidad con la tesis; "
            "el plan visible y el Excel descargable siempre corresponden a la ejecución en vivo."
        )

        if len(res_multi["plan"]):
            util_pct_multi = res_multi["utilizacion_dia_maquina"].div(res_multi["capacidad_min"], axis=1) * 100
            st.plotly_chart(heatmap_utilizacion(util_pct_multi, "Utilización por día y máquina"), use_container_width=True)
            mm1, mm2, mm3 = st.columns(3)
            mm1.metric("Unidades producidas", fmt_num(res_multi["plan"]["unidades"].sum(), 0))
            mm2.metric("Lote promedio", fmt_num(res_multi["plan"]["unidades"].mean(), 1))
            mm3.metric("σ utilización día-máquina", f"{util_pct_multi.stack().std(ddof=0):.1f} pts")
            plan_multi_ordenado = res_multi["plan"].sort_values(
                ["dia", "maquina", "unidades"], ascending=[True, True, False]
            )
            st.dataframe(plan_multi_ordenado, use_container_width=True, hide_index=True)

            # Exportación completa del plan multiperiodo, como en la versión anterior de la app.
            buffer_multi = io.BytesIO()
            resumen_multi = pd.DataFrame([
                {
                    "estado": res_multi.get("estado_presentacion", res_multi.get("estado")),
                    "estado_solver_raw": res_multi.get("estado"),
                    "gap_certificado": bool(res_multi.get("gap_certificado")),
                    "gap_objetivo_alcanzado": bool(res_multi.get("gap_objetivo_alcanzado")),
                    "limite_tiempo_alcanzado": bool(res_multi.get("solver_termination", {}).get("stopped_on_time")),
                    "objetivo_min": res_multi.get("valor_objetivo"),
                    "upper_bound_min": ub_multi,
                    "lower_bound_min": lb_multi,
                    "gap_pct": res_multi.get("gap_pct"),
                    "gap_objetivo_pct": gap_obj_usado,
                    "tiempo_limite_seg": tiempo_usado,
                    "dias_horizonte": res_multi.get("dias"),
                    "demanda_entera_unid": multi_state.get("demanda_entera"),
                    "ajuste_redondeo_unid": multi_state.get("ajuste"),
                    "factor_calibracion": multi_state["cal"].get("factor_sesgo"),
                    "residuo_demanda_max": res_multi.get("residuos", {}).get("demanda_max"),
                    "residuo_capacidad_max": res_multi.get("residuos", {}).get("capacidad_max"),
                }
            ])
            demanda_export = multi_state.get("demanda_horizonte")
            if isinstance(demanda_export, pd.Series):
                demanda_export = demanda_export.rename("demanda_unidades").rename_axis("producto").reset_index()

            with pd.ExcelWriter(buffer_multi, engine="openpyxl") as writer:
                resumen_multi.to_excel(writer, sheet_name="Resumen", index=False)
                plan_multi_ordenado.to_excel(writer, sheet_name="Plan_MultiDia", index=False)
                if isinstance(demanda_export, pd.DataFrame):
                    demanda_export.to_excel(writer, sheet_name="Demanda_Horizonte", index=False)
                res_multi["utilizacion_dia_maquina"].to_excel(writer, sheet_name="Utilizacion_Min")
                util_pct_multi.to_excel(writer, sheet_name="Utilizacion_Pct")
                res_multi["setups_dia_maquina"].to_excel(writer, sheet_name="Setups_Dia_Maquina")
                res_multi["tiempo_produccion_dia_maquina"].to_excel(writer, sheet_name="Produccion_Min")
                res_multi["tiempo_setup_dia_maquina"].to_excel(writer, sheet_name="Setup_Min")
                res_multi["holgura_dia_maquina"].to_excel(writer, sheet_name="Holgura_Min")

            st.download_button(
                "Descargar plan multi-día en Excel",
                data=buffer_multi.getvalue(),
                file_name="plan_multidia_rpm.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Tab 6: Monte Carlo
# ---------------------------------------------------------------------------
with mc_tab:
    st.write(
        "La simulacion Monte Carlo es deliberadamente opcional. La app muestra primero la corrida de referencia "
        "documentada en la tesis y solo resuelve escenarios nuevos cuando el usuario lo solicita."
    )

    ref_mc = cargar_referencia_monte_carlo()
    if ref_mc is not None:
        st.markdown("#### Referencia documentada en la tesis")
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Escenarios", f"{ref_mc['n']}")
        r2.metric("Mejora media", f"{ref_mc['media_pct']:.2f}%")
        r3.metric("Mediana", f"{ref_mc['mediana_pct']:.2f}%")
        r4.metric("Desv. estandar", f"{ref_mc['sd_pct']:.2f} pp")
        r5.metric("IC 95%", f"[{ref_mc['ci95_low_pct']:.2f}, {ref_mc['ci95_high_pct']:.2f}]%")
        rr1, rr2, rr3, rr4 = st.columns(4)
        rr1.metric("Error estandar", f"{ref_mc['se_pct']:.3f} pp")
        rr2.metric("Rango", f"[{ref_mc['min_pct']:.2f}, {ref_mc['max_pct']:.2f}]%")
        rr3.metric("Mejora positiva", f"{100*ref_mc['positive_count']/ref_mc['n']:.1f}%")
        rr4.metric("GAP solver", f"{ref_mc['solver_gap_pct']:.2f}%")
        st.caption(ref_mc["methodology_note"])

    st.markdown("#### Ejecutar una simulacion nueva")
    c1, c2, c3 = st.columns(3)
    preset = c1.radio(
        "Tamano de corrida",
        ["Rapida (30)", "Intermedia (100)", "Validacion (1000)"],
        horizontal=False,
        index=0,
    )
    n_map = {"Rapida (30)": 30, "Intermedia (100)": 100, "Validacion (1000)": 1000}
    n_mc = n_map[preset]
    seed_mc = c2.number_input("Semilla", min_value=0, max_value=999999, value=42, step=1)
    tiempo_mc = c3.slider("Limite CBC por escenario (s)", 5, 120, min(30, tiempo_limite), 5)

    st.caption(
        "Cada escenario perturba cada demanda de referencia de forma independiente con U(0,85; 1,15), "
        "redondea a unidades enteras y compara el MILP con la politica greedy integral. "
        "Los resultados se almacenan en cache durante 24 horas para no repetir una corrida identica."
    )
    if n_mc >= 1000:
        st.warning(
            "La corrida N=1000 es computacionalmente costosa y no se ejecuta automaticamente. "
            "En Streamlit Cloud conviene usarla solo cuando se desea reproducir la validacion completa."
        )

    if st.button("Ejecutar nueva simulacion Monte Carlo", key="run_mc"):
        with st.spinner(f"Resolviendo {n_mc} escenarios MILP y {n_mc} politicas de referencia..."):
            mc = ejecutar_monte_carlo_cache(
                cij, demanda, setup, n_mc, int(seed_mc), 0.85, 1.15,
                cap_min, int(tiempo_mc),
            )
        st.session_state["mc"] = mc

    mc = st.session_state.get("mc")
    if mc is not None:
        s_mc = mc["resumen"]
        st.markdown("#### Resultado de la nueva corrida")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Optimos certificados", f"{s_mc['n_optimos_certificados']}/{s_mc['n_solicitados']}")
        m2.metric("Mejora media", f"{s_mc['media_pct']:.2f}%")
        m3.metric("Mediana", f"{s_mc['mediana_pct']:.2f}%")
        m4.metric("Desv. estandar", f"{s_mc['sd_pct']:.2f} pp")
        m5.metric("IC 95% media", f"[{s_mc['ci95_low_pct']:.2f}, {s_mc['ci95_high_pct']:.2f}]%")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Error estandar", f"{s_mc['se_pct']:.3f} pp")
        q2.metric("Intervalo P5-P95", f"[{s_mc['p5_pct']:.2f}, {s_mc['p95_pct']:.2f}]%")
        q3.metric("Mejora positiva", f"{100*s_mc['proporcion_mejora_positiva']:.1f}%")
        q4.metric("Rango observado", f"[{s_mc['min_pct']:.2f}, {s_mc['max_pct']:.2f}]%")

        e = mc["escenarios"].dropna(subset=["mejora_certificada_pct"]).copy()
        if s_mc["n_no_certificados"] > 0:
            st.warning(
                f"{s_mc['n_no_certificados']} escenarios no cerraron con optimalidad certificada dentro del limite de tiempo y "
                "no se incluyen en los estadisticos inferenciales."
            )
        if len(e):
            e["media_acumulada"] = e["mejora_certificada_pct"].expanding().mean()
            e_ord = e.sort_values("mejora_certificada_pct").reset_index(drop=True)
            e_ord["orden"] = np.arange(1, len(e_ord) + 1)

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Histogram(
                    x=e["mejora_certificada_pct"], nbinsx=18, marker_color=BLUE, opacity=0.88
                ))
                fig.add_vline(x=s_mc["media_pct"], line_color=TEAL, line_width=2,
                              annotation_text=f"media {s_mc['media_pct']:.2f}%", annotation_font_color=TEAL)
                fig.add_vline(x=s_mc["p5_pct"], line_color=SLATE, line_dash="dot", line_width=1.4)
                fig.add_vline(x=s_mc["p95_pct"], line_color=SLATE, line_dash="dot", line_width=1.4)
                fig.update_xaxes(title="Mejora relativa certificada (%)")
                fig.update_yaxes(title="Frecuencia")
                st.plotly_chart(fig_layout(fig, "Distribucion de la mejora", height=380, legend=False), use_container_width=True)
            with c2:
                fig = go.Figure(go.Box(
                    y=e["mejora_certificada_pct"], boxmean=True, marker_color=TEAL,
                    line=dict(color=LIGHT), fillcolor="rgba(42,157,143,0.25)"
                ))
                fig.update_yaxes(title="Mejora relativa certificada (%)")
                fig.update_xaxes(showticklabels=False)
                st.plotly_chart(fig_layout(fig, "Dispersion y valores extremos", height=380, legend=False), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                fig = go.Figure(go.Scatter(
                    x=e_ord["orden"], y=e_ord["mejora_certificada_pct"], mode="lines",
                    line=dict(color=BLUE, width=2.1)
                ))
                fig.add_hline(y=s_mc["media_pct"], line_color=TEAL, line_dash="dash", line_width=1.5)
                fig.update_xaxes(title="Escenarios ordenados")
                fig.update_yaxes(title="Mejora relativa certificada (%)")
                st.plotly_chart(fig_layout(fig, "Escenarios ordenados por mejora", height=380, legend=False), use_container_width=True)
            with c4:
                fig = go.Figure(go.Scatter(
                    x=np.arange(1, len(e) + 1), y=e["media_acumulada"], mode="lines",
                    line=dict(color=TEAL, width=2.4)
                ))
                fig.add_hline(y=s_mc["media_pct"], line_color=SLATE, line_dash="dot", line_width=1.2)
                fig.update_xaxes(title="Numero de escenarios certificados")
                fig.update_yaxes(title="Media acumulada de mejora (%)")
                st.plotly_chart(fig_layout(fig, "Convergencia de la media Monte Carlo", height=380, legend=False), use_container_width=True)

            csv_mc = mc["escenarios"].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Descargar escenarios Monte Carlo (CSV)", csv_mc,
                file_name=f"monte_carlo_N{n_mc}_seed{int(seed_mc)}.csv", mime="text/csv"
            )

        st.caption(
            "Los estadisticos se calculan solo con escenarios cuyo MILP cerro con optimalidad certificada. "
            "La frecuencia de mejora positiva es empirica y queda condicionada al mecanismo U(0,85;1,15); "
            "no constituye una garantia para cualquier demanda futura."
        )

