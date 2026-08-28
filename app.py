# -*- coding: utf-8 -*-
"""
app.py — Planeador de Producción RPM Colombia (dashboard profesional)

Cómo correrlo:
    pip install streamlit pandas openpyxl scikit-learn pulp plotly
    streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rpm_core import (
    CAP_MINUTOS_DIA,
    MACHINES,
    analisis_cuello_de_botella,
    analisis_sensibilidad,
    calibrar_demanda_extendida,
    cargar_datos,
    cargar_datos_extendido,
    clusterizar_maquinas,
    clusterizar_productos,
    elbow_silhouette,
    politica_empirica,
    prueba_estadistica_clusters,
    resolver_asignacion,
    resolver_multidia,
)

st.set_page_config(page_title="RPM Colombia — Planeador de Producción", layout="wide",
                    page_icon="🏭", initial_sidebar_state="expanded")

# ==========================================================================
# PALETA Y ESTILO
# ==========================================================================
BG = "#0E1117"
CARD_BG = "#161B26"
CARD_BORDER = "#232938"
TEXT = "#E6E8EC"
MUTED = "#8A94A6"
ACCENT = "#3DA5D9"
TEAL = "#33C2A3"
SUCCESS = "#2ECC71"
WARNING = "#F5A623"
DANGER = "#E9576B"
NAVY = "#223B5E"

PLOTLY_TEMPLATE = "plotly_dark"
FONT = dict(family="Segoe UI, -apple-system, sans-serif", color=TEXT)

st.markdown(f"""
<style>
    .block-container {{ padding-top: 1.4rem; }}
    div[data-testid="stMetric"] {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED} !important; font-size: 0.85rem; }}
    .kpi-title {{
        font-size: 0.95rem; color: {MUTED}; text-transform: uppercase;
        letter-spacing: 0.04em; margin-bottom: 2px;
    }}
    .section-card {{
        background: {CARD_BG}; border: 1px solid {CARD_BORDER};
        border-radius: 12px; padding: 18px 20px; margin-bottom: 14px;
    }}
    .badge-ok {{ background: rgba(46,204,113,0.15); color: {SUCCESS};
                 padding: 3px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }}
    .badge-warn {{ background: rgba(245,166,35,0.15); color: {WARNING};
                   padding: 3px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }}
    .badge-bad {{ background: rgba(233,87,107,0.15); color: {DANGER};
                  padding: 3px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }}
</style>
""", unsafe_allow_html=True)

st.title("🏭 Planeador de Producción — RPM Colombia")
st.caption(
    "Prototipo de apoyo a la decisión: a partir de los tiempos históricos por máquina y "
    "la demanda del día, recomienda cuánto producir en cada máquina y agrupa los "
    "productos por su rol óptimo en planta."
)


# ==========================================================================
# Helpers de gráficos Plotly
# ==========================================================================
def fig_layout(fig, title=None, height=380, **kw):
    margen = kw.pop("margin", dict(l=40, r=20, t=50 if title else 20, b=40))
    layout_kwargs = dict(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=FONT, height=height, margin=margen,
    )
    if title:
        # Solo se incluye la clave "title" cuando hay texto real. Pasar
        # title=None explícitamente a update_layout() activa un bug de la
        # versión de Plotly.js empaquetada con Streamlit que muestra el
        # texto literal "undefined" en los gráficos tipo Indicator.
        layout_kwargs["title"] = dict(text=title, font=dict(size=15, color=TEXT))
    layout_kwargs.update(kw)
    fig.update_layout(**layout_kwargs)
    return fig


def grafico_utilizacion(util, cap, cuello_top=None):
    colores = [DANGER if v >= cap * 0.98 else (WARNING if v >= cap * 0.85 else ACCENT) for v in util.values]
    if cuello_top:
        colores = [WARNING if m == cuello_top else c for m, c in zip(util.index, colores)]
    fig = go.Figure()
    fig.add_bar(x=util.index, y=util.values, marker_color=colores,
                text=[f"{v:,.0f}" for v in util.values], textposition="outside",
                textfont=dict(color=TEXT, size=11))
    fig.add_hline(y=cap, line_dash="dash", line_color=DANGER,
                   annotation_text=f"Capacidad ({cap:,.0f} min)", annotation_font_color=DANGER)
    fig.update_yaxes(title="Minutos utilizados", gridcolor=CARD_BORDER, range=[0, cap * 1.18])
    fig.update_xaxes(title="Máquina")
    return fig_layout(fig, "Utilización de capacidad por máquina")


def gauge(valor, titulo, sufijo="%", rango=(0, 100), umbral_bueno=90, umbral_malo=70, invertido=False):
    if invertido:
        color = SUCCESS if valor <= (100 - umbral_bueno) else (WARNING if valor <= (100 - umbral_malo) else DANGER)
    else:
        color = SUCCESS if valor >= umbral_bueno else (WARNING if valor >= umbral_malo else DANGER)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor,
        number={"suffix": sufijo, "font": {"size": 30, "color": TEXT}},
        gauge={
            "axis": {"range": rango, "tickcolor": MUTED, "tickfont": {"color": MUTED, "size": 9}},
            "bar": {"color": color, "thickness": 0.75},
            "bgcolor": CARD_BORDER,
            "borderwidth": 0,
        },
    ))
    # Nota: el título NO se pasa dentro del Indicator porque una versión de
    # Plotly.js empaquetada con Streamlit tiene un bug conocido que muestra
    # el texto literal "undefined" junto al título cuando se usa la
    # propiedad "title" de un Indicator tipo gauge. En su lugar, el título
    # se dibuja como texto de Streamlit justo encima del gráfico (ver
    # función gauge_con_titulo).
    return fig_layout(fig, height=170, margin=dict(l=20, r=20, t=10, b=10))


def gauge_con_titulo(valor, titulo, **kwargs):
    st.markdown(
        f'<p style="color:{MUTED};font-size:13px;margin:6px 0 -6px 4px;'
        f'text-transform:uppercase;letter-spacing:0.03em">{titulo}</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(gauge(valor, titulo, **kwargs), use_container_width=True)


def donut(labels, values, colores, titulo=None):
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.58, marker=dict(colors=colores, line=dict(color=CARD_BG, width=2)),
        textinfo="percent", textfont=dict(color=TEXT, size=12),
    ))
    fig.update_layout(showlegend=True, legend=dict(font=dict(color=TEXT, size=11), orientation="v"))
    return fig_layout(fig, titulo, height=320)


def heatmap_utilizacion(tabla_pct, titulo):
    fig = go.Figure(go.Heatmap(
        z=tabla_pct.values, x=tabla_pct.columns.tolist(), y=[f"Día {d}" for d in tabla_pct.index],
        colorscale=[[0, "#1F6E5C"], [0.7, "#F5A623"], [1, "#E9576B"]],
        zmin=0, zmax=100, text=tabla_pct.values.round(1), texttemplate="%{text}%",
        textfont=dict(size=10, color=TEXT), colorbar=dict(title="%", tickfont=dict(color=TEXT)),
    ))
    fig.update_xaxes(title="Máquina", side="top")
    fig.update_yaxes(title="", autorange="reversed")
    return fig_layout(fig, titulo, height=120 + 45 * len(tabla_pct))


def badge(texto, tipo="ok"):
    clase = {"ok": "badge-ok", "warn": "badge-warn", "bad": "badge-bad"}[tipo]
    st.markdown(f'<span class="{clase}">{texto}</span>', unsafe_allow_html=True)


# ==========================================================================
# Barra lateral — datos y parámetros
# ==========================================================================
st.sidebar.header("1. Datos de entrada")

archivo = st.sidebar.file_uploader(
    "Sube el Excel de datos (mismo formato del Anexo A)", type=["xlsx"]
)
usar_ejemplo = st.sidebar.checkbox("Usar el Excel de ejemplo del proyecto", value=(archivo is None))

st.sidebar.header("2. Parámetros del modelo")
cap_min = st.sidebar.number_input(
    "Capacidad diaria por máquina (min)", min_value=60, max_value=2880, value=CAP_MINUTOS_DIA, step=60
)
n_clusters = st.sidebar.slider("Número de grupos (clustering de productos)", 2, 5, 3)
tiempo_limite = st.sidebar.slider("Tiempo máximo de cómputo del solver (seg)", 10, 180, 60)

correr = st.sidebar.button("🚀 Calcular plan de producción", type="primary", use_container_width=True)


@st.cache_data(show_spinner=False)
def _cargar(path_o_buffer):
    return cargar_datos(path_o_buffer)


# --------------------------------------------------------------------
# Cargar datos
# --------------------------------------------------------------------
datos = None
if archivo is not None:
    datos = _cargar(archivo)
elif usar_ejemplo:
    datos = _cargar("AnexodeDatosxlsx.xlsx")

if datos is None:
    st.info("Sube un archivo Excel o marca la casilla para usar el ejemplo del proyecto.")
    st.stop()

cij, demanda, setup = datos["cij"], datos["demanda"], datos["setup"]

col1, col2, col3 = st.columns(3)
col1.metric("Referencias cargadas", len(cij))
col2.metric("Máquinas", len(MACHINES))
col3.metric("Demanda total del día (unid.)", f"{int(demanda.sum()):,}".replace(",", "."))

st.divider()

# --------------------------------------------------------------------
# Sección: clustering de productos y máquinas (validado)
# --------------------------------------------------------------------
st.header("📊 Clustering validado (productos y máquinas)")

tab_prod, tab_maq = st.tabs(["Productos", "Máquinas"])

with tab_prod:
    st.write(
        "Cada producto se clasifica según **su volumen de demanda** y "
        "**qué tanto varía su tiempo de producción entre máquinas**. Esto "
        "formaliza con datos la propuesta de la tesis de separar máquinas "
        "'de producción continua' de máquinas 'comodín' (sección 7.4.3)."
    )

    cv = cij.std(axis=1) / cij.mean(axis=1)
    feats_prod = pd.DataFrame({"demanda": demanda, "cv_tiempo": cv})
    tabla_k_prod = elbow_silhouette(feats_prod, k_range=range(2, 7))

    c1, c2 = st.columns(2)
    with c1:
        fig_e = go.Figure()
        fig_e.add_scatter(x=tabla_k_prod["k"], y=tabla_k_prod["inercia"], mode="lines+markers",
                           line=dict(color=ACCENT, width=2.5), marker=dict(size=8))
        fig_e.update_xaxes(title="k (número de grupos)", gridcolor=CARD_BORDER)
        fig_e.update_yaxes(title="Inercia", gridcolor=CARD_BORDER)
        st.plotly_chart(fig_layout(fig_e, "Método del codo", height=300), use_container_width=True)
    with c2:
        mejor_k_i = int(tabla_k_prod.loc[tabla_k_prod["silhouette"].idxmax(), "k"])
        fig_s = go.Figure()
        fig_s.add_scatter(x=tabla_k_prod["k"], y=tabla_k_prod["silhouette"], mode="lines+markers",
                           line=dict(color=TEAL, width=2.5), marker=dict(size=8))
        fig_s.add_scatter(x=[mejor_k_i], y=[tabla_k_prod["silhouette"].max()], mode="markers",
                           marker=dict(size=14, color=DANGER), name=f"Mejor k={mejor_k_i}")
        fig_s.update_xaxes(title="k (número de grupos)", gridcolor=CARD_BORDER)
        fig_s.update_yaxes(title="Silhouette Score", gridcolor=CARD_BORDER)
        fig_s.update_layout(showlegend=False)
        st.plotly_chart(fig_layout(fig_s, "Validación por Silhouette", height=300), use_container_width=True)

    mejor_k = int(tabla_k_prod.loc[tabla_k_prod["silhouette"].idxmax(), "k"])
    st.caption(f"El mejor k según Silhouette es **{mejor_k}**. Se muestra el resultado con k={n_clusters} "
               "(ajustable en la barra lateral) para mantener la interpretación de negocio en 3 roles.")

    clusters = clusterizar_productos(cij, demanda, n_clusters=n_clusters)

    ROLES_COLOR = {
        "Producción continua (alto volumen)": TEAL,
        "Rotación media": ACCENT,
        "Comodín / baja escala": WARNING,
    }

    c3, c4 = st.columns([1, 2])
    with c3:
        conteo = clusters["rol_sugerido"].value_counts()
        colores_d = [ROLES_COLOR.get(r, MUTED) for r in conteo.index]
        st.plotly_chart(donut(conteo.index, conteo.values, colores_d, "Distribución por rol"),
                         use_container_width=True)
    with c4:
        fig2 = go.Figure()
        for rol, sub in clusters.groupby("rol_sugerido"):
            fig2.add_scatter(x=sub["demanda"], y=sub["cv_tiempo"], mode="markers", name=rol,
                              marker=dict(size=9, color=ROLES_COLOR.get(rol, MUTED),
                                          line=dict(width=0.5, color=CARD_BG)))
        fig2.update_xaxes(title="Demanda diaria (unidades)", gridcolor=CARD_BORDER)
        fig2.update_yaxes(title="Variabilidad del tiempo (CV)", gridcolor=CARD_BORDER)
        fig2.update_layout(legend=dict(font=dict(color=TEXT, size=10)))
        st.plotly_chart(fig_layout(fig2, "Mapa de productos: volumen vs. variabilidad", height=320),
                         use_container_width=True)

    test_dem = prueba_estadistica_clusters(clusters["demanda"], clusters["cluster"])
    test_cv = prueba_estadistica_clusters(clusters["cv_tiempo"], clusters["cluster"])
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown(f"**Kruskal-Wallis — Demanda:** H={test_dem['estadistico_H']:.2f}, p={test_dem['valor_p']:.2e}")
        badge("Significativo" if test_dem["significativo_al_5pct"] else "No significativo",
              "ok" if test_dem["significativo_al_5pct"] else "bad")
    with cc2:
        st.markdown(f"**Kruskal-Wallis — Variabilidad:** H={test_cv['estadistico_H']:.2f}, p={test_cv['valor_p']:.2e}")
        badge("Significativo" if test_cv["significativo_al_5pct"] else "No significativo",
              "ok" if test_cv["significativo_al_5pct"] else "bad")

    with st.expander("Ver tabla completa de clusters de productos"):
        st.dataframe(clusters.sort_values("demanda", ascending=False), use_container_width=True)

with tab_maq:
    st.write(
        "Agrupa las 10 máquinas según su **comportamiento medido** (SetUp "
        "promedio, tiempo de operación promedio y su varianza), en vez de "
        "por antigüedad percibida como hace la Tabla 3 de la tesis."
    )

    tiempo_prom = cij.mean(axis=0)
    tiempo_var = cij.var(axis=0)
    feats_maq = pd.DataFrame({
        "setup_promedio": setup,
        "tiempo_operacion_promedio": tiempo_prom,
        "varianza_tiempo": tiempo_var,
    })
    tabla_k_maq = elbow_silhouette(feats_maq, k_range=range(2, 6))

    c5, c6 = st.columns(2)
    with c5:
        fig_em = go.Figure()
        fig_em.add_scatter(x=tabla_k_maq["k"], y=tabla_k_maq["inercia"], mode="lines+markers",
                            line=dict(color=ACCENT, width=2.5), marker=dict(size=8))
        fig_em.update_xaxes(title="k", gridcolor=CARD_BORDER)
        fig_em.update_yaxes(title="Inercia", gridcolor=CARD_BORDER)
        st.plotly_chart(fig_layout(fig_em, "Método del codo — máquinas", height=300), use_container_width=True)
    with c6:
        mejor_k_m = int(tabla_k_maq.loc[tabla_k_maq["silhouette"].idxmax(), "k"])
        fig_sm = go.Figure()
        fig_sm.add_scatter(x=tabla_k_maq["k"], y=tabla_k_maq["silhouette"], mode="lines+markers",
                            line=dict(color=TEAL, width=2.5), marker=dict(size=8))
        fig_sm.add_scatter(x=[mejor_k_m], y=[tabla_k_maq["silhouette"].max()], mode="markers",
                            marker=dict(size=14, color=DANGER))
        fig_sm.update_xaxes(title="k", gridcolor=CARD_BORDER)
        fig_sm.update_yaxes(title="Silhouette Score", gridcolor=CARD_BORDER)
        fig_sm.update_layout(showlegend=False)
        st.plotly_chart(fig_layout(fig_sm, "Validación por Silhouette — máquinas", height=300), use_container_width=True)

    n_clusters_maq = st.slider("Número de grupos de máquinas", 2, 5, 3)
    clusters_maq = clusterizar_maquinas(cij, setup, n_clusters=n_clusters_maq)
    st.dataframe(
        clusters_maq[["setup_promedio", "tiempo_operacion_promedio", "varianza_tiempo", "grupo_sugerido"]]
        .sort_values("setup_promedio", ascending=False),
        use_container_width=True,
    )

    test_setup = prueba_estadistica_clusters(clusters_maq["setup_promedio"], clusters_maq["cluster"])
    st.markdown(f"**Kruskal-Wallis (SetUp):** H={test_setup['estadistico_H']:.2f}, p={test_setup['valor_p']:.3f}")
    badge("Significativo" if test_setup["significativo_al_5pct"] else "No significativo",
          "ok" if test_setup["significativo_al_5pct"] else "bad")
    if not test_setup["significativo_al_5pct"]:
        st.caption(
            "Con solo 10 máquinas, la prueba tiene poca potencia estadística — "
            "no alcanzar significancia no invalida la agrupación, pero sí debe "
            "declararse como limitación en la tesis (sección de Limitaciones, Fase 7)."
        )

st.divider()

# --------------------------------------------------------------------
# Sección: optimización
# --------------------------------------------------------------------
st.header("⚙️ Plan de asignación óptimo")

if correr:
    with st.spinner("Resolviendo el modelo de programación lineal..."):
        resultado = resolver_asignacion(
            cij, demanda, setup, cap_minutos=cap_min, tiempo_limite_seg=tiempo_limite
        )
    st.session_state["resultado"] = resultado

resultado = st.session_state.get("resultado")

if resultado is None:
    st.warning("Da clic en **Calcular plan de producción** en la barra lateral para resolver el modelo.")
    st.stop()

asignacion = resultado["asignacion"]
util = resultado["utilizacion_min"]
n_setups = resultado["n_setups"]
util_pct_max = float((util / resultado["cap_minutos"] * 100).max())

# ---- Fila de tarjetas KPI ----
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Estado del solver", resultado["estado"],
              "GAP 0%" if resultado["gap_certificado"] else "sin certificar")
with kpi2:
    st.metric("Tiempo total", f"{resultado['valor_objetivo']:,.0f} min".replace(",", "."))
with kpi3:
    st.metric("Utilización máxima", f"{util_pct_max:.1f}%")
with kpi4:
    maquina_top = util.idxmax()
    st.metric("Máquina más cargada", maquina_top, f"{util_pct_max:.1f}%")
with kpi5:
    st.metric("Total de alistamientos", int(n_setups.sum()))

st.markdown("")

tab_plan, tab_comp, tab_sens, tab_cuello, tab_multi = st.tabs([
    "📋 Plan y utilización",
    "⚖️ Vs. política empírica",
    "📈 Sensibilidad a la demanda",
    "🚧 Cuello de botella",
    "🗓️ Plan multi-día (200 refs)",
])

# ---------------- TAB 1: Plan y utilización ----------------
with tab_plan:
    col_chart, col_gauge = st.columns([2.3, 1])
    with col_chart:
        st.plotly_chart(grafico_utilizacion(util, resultado["cap_minutos"]), use_container_width=True)
    with col_gauge:
        gauge_con_titulo(util_pct_max, "Utilización máxima", umbral_bueno=95, umbral_malo=80)
        gauge_con_titulo(100 if resultado["gap_certificado"] else 0, "GAP certificado", umbral_bueno=99)

    resumen = pd.DataFrame({
        "Utilización (min)": util.round(1),
        "% de capacidad": (util / resultado["cap_minutos"] * 100).round(1),
        "N° de Set-ups": n_setups,
    })
    st.dataframe(resumen, use_container_width=True)

    st.subheader("¿Cuánto producir de cada referencia y en qué máquina?")
    tabla = asignacion[asignacion.sum(axis=1) > 0].copy()
    tabla = tabla.loc[demanda.reindex(tabla.index).sort_values(ascending=False).index]
    tabla_mostrar = tabla.round(0).replace(0, "")
    st.dataframe(tabla_mostrar, use_container_width=True)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        tabla.round(1).to_excel(writer, sheet_name="Asignacion")
        resumen.to_excel(writer, sheet_name="Utilizacion")
        clusters.to_excel(writer, sheet_name="Clusters")
    st.download_button(
        "⬇️ Descargar plan en Excel",
        data=buffer.getvalue(),
        file_name="plan_produccion_rpm.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ---------------- TAB 2: Comparación vs. política empírica ----------------
with tab_comp:
    st.write(
        "Compara el modelo óptimo contra una **política empírica de referencia** "
        "(heurística greedy: asigna cada producto, de mayor a menor demanda, a la "
        "máquina más rápida con capacidad disponible — sin anticipar el efecto "
        "sobre los demás productos, como haría un supervisor sin apoyo de un "
        "modelo). *No es un dato histórico real de RPM Colombia: es una línea "
        "base estándar de la literatura para poder cuantificar la mejora.*"
    )
    if st.button("Calcular política empírica y comparar"):
        with st.spinner("Calculando política empírica..."):
            emp = politica_empirica(cij, demanda, setup, cap_minutos=resultado["cap_minutos"])
        mejora = (emp["valor_objetivo"] - resultado["valor_objetivo"]) / emp["valor_objetivo"] * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Tiempo total — modelo óptimo", f"{resultado['valor_objetivo']:.0f} min")
        c2.metric("Tiempo total — política empírica", f"{emp['valor_objetivo']:.0f} min")
        c3.metric("Mejora del modelo óptimo", f"{mejora:.1f} %")

        fig_comp = go.Figure()
        fig_comp.add_bar(x=["Modelo óptimo", "Política empírica"],
                          y=[resultado["valor_objetivo"], emp["valor_objetivo"]],
                          marker_color=[ACCENT, MUTED],
                          text=[f"{resultado['valor_objetivo']:,.0f} min", f"{emp['valor_objetivo']:,.0f} min"],
                          textposition="outside", textfont=dict(color=TEXT))
        fig_comp.update_yaxes(title="Tiempo total (min)", gridcolor=CARD_BORDER)
        st.plotly_chart(fig_layout(fig_comp, f"Mejora del modelo óptimo: {mejora:.1f}%", height=380),
                         use_container_width=True)

        if len(emp["demanda_incumplida"]) > 0:
            st.warning("La política empírica NO alcanza a cumplir toda la demanda en estas referencias:")
            st.dataframe(emp["demanda_incumplida"])
        else:
            st.info("La política empírica sí alcanza a cumplir el 100 % de la demanda, pero usando más tiempo total.")

# ---------------- TAB 3: Sensibilidad a la demanda ----------------
with tab_sens:
    st.write(
        "Resuelve el modelo variando la demanda para ver qué tan estable es la "
        "solución — la pregunta típica de sustentación: *¿qué pasa si la demanda "
        "sube o baja?*"
    )
    if st.button("Ejecutar análisis de sensibilidad (5 corridas)"):
        with st.spinner("Resolviendo 5 escenarios de demanda (±20 %, ±10 %, base)..."):
            sens = analisis_sensibilidad(cij, demanda, setup, cap_minutos=resultado["cap_minutos"],
                                          tiempo_limite_seg=tiempo_limite)
        st.dataframe(sens, use_container_width=True)

        fig4 = go.Figure()
        fig4.add_scatter(x=sens["factor_demanda"], y=sens["valor_objetivo_min"], mode="lines+markers",
                          name="Tiempo total", line=dict(color=ACCENT, width=2.5), marker=dict(size=8),
                          yaxis="y1")
        fig4.add_scatter(x=sens["factor_demanda"], y=sens["utilizacion_maxima_pct"], mode="lines+markers",
                          name="Utilización máxima (%)", line=dict(color=DANGER, width=2, dash="dash"),
                          marker=dict(size=8, symbol="square"), yaxis="y2")
        fig4.update_layout(
            xaxis=dict(title="Factor de demanda (1.0 = demanda real)", gridcolor=CARD_BORDER),
            yaxis=dict(title="Tiempo total (min)", gridcolor=CARD_BORDER, titlefont=dict(color=ACCENT)),
            yaxis2=dict(title="Utilización máxima (%)", overlaying="y", side="right", showgrid=False,
                        titlefont=dict(color=DANGER)),
            legend=dict(font=dict(color=TEXT, size=11), orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_layout(fig4, "Sensibilidad del sistema a la demanda", height=440,
                                    margin=dict(l=50, r=50, t=95, b=40)),
                         use_container_width=True)

# ---------------- TAB 4: Cuello de botella ----------------
with tab_cuello:
    st.write(
        "Identifica qué máquina es el verdadero **cuello de botella** del "
        "sistema: se le da a cada máquina, una por una, 30 minutos adicionales "
        "de capacidad y se mide cuánto mejora el tiempo total. La que más "
        "mejora genera es el cuello de botella real — el 'tambor' del sistema "
        "en términos de la Teoría de Restricciones (DBR)."
    )
    st.caption(
        "Nota técnica: se usa re-optimización directa en vez del precio sombra "
        "clásico, porque en este modelo (con variables binarias Yij) el dual "
        "resulta degenerado — ver justificación en rpm_core.py."
    )
    if st.button("Calcular cuello de botella (11 resoluciones, ~1-2 min)"):
        with st.spinner("Resolviendo 11 escenarios de capacidad..."):
            cuello = analisis_cuello_de_botella(cij, demanda, setup,
                                                 cap_minutos=resultado["cap_minutos"],
                                                 tiempo_limite_seg=tiempo_limite)
        cuello_ordenado = cuello.dropna(subset=["mejora_min_por_+30min_capacidad"]).sort_values(
            "mejora_min_por_+30min_capacidad")
        excluidas = cuello[cuello["mejora_min_por_+30min_capacidad"].isna()]
        if len(excluidas) > 0:
            st.warning(
                f"⚠️ {len(excluidas)} máquina(s) no alcanzaron GAP=0% certificado dentro del "
                f"tiempo límite del solver ({tiempo_limite}s) y se excluyeron del gráfico para no "
                f"mostrar un valor engañoso: {', '.join(excluidas['maquina'].tolist())}. "
                f"Sube el tiempo máximo de cómputo en la barra lateral para incluirlas."
            )
        fig_c = go.Figure()
        colores_c = [DANGER if v == cuello_ordenado["mejora_min_por_+30min_capacidad"].max() else ACCENT
                     for v in cuello_ordenado["mejora_min_por_+30min_capacidad"]]
        fig_c.add_bar(y=cuello_ordenado["maquina"], x=cuello_ordenado["mejora_min_por_+30min_capacidad"],
                      orientation="h", marker_color=colores_c,
                      text=[f"{v:.2f} min" for v in cuello_ordenado["mejora_min_por_+30min_capacidad"]],
                      textposition="outside", textfont=dict(color=TEXT))
        fig_c.update_xaxes(title="Mejora en el tiempo total (min) por +30 min de capacidad", gridcolor=CARD_BORDER)
        st.plotly_chart(fig_layout(fig_c, "¿Cuál máquina es el cuello de botella real?", height=420),
                         use_container_width=True)

        st.dataframe(cuello, use_container_width=True)
        top = cuello.iloc[0]
        st.success(f"**{top['maquina']}** es el cuello de botella real del sistema: "
                   f"cada 30 minutos adicionales de su capacidad reducen el tiempo "
                   f"total en {top['mejora_min_por_+30min_capacidad']:.2f} min.")

# ---------------- TAB 5: Plan multi-día (200 referencias) ----------------
with tab_multi:
    st.write(
        "Extiende el modelo a las **200 referencias del catálogo completo**, "
        "repartiendo la producción en varios días en vez de intentar meterlas "
        "todas en uno solo. Corrige, además, un sesgo detectado en la manera "
        "de estimar la demanda de las referencias que no tienen medición "
        "directa (ventas del semestre ÷ días hábiles), que sobreestimaba la "
        "demanda real."
    )

    cma, cmb, cmc = st.columns(3)
    with cma:
        dias_habiles_input = st.number_input(
            "Días hábiles usados para estimar la demanda", min_value=60, max_value=300, value=180, step=10,
        )
    with cmb:
        dias_horizonte = st.slider("Días del horizonte de planeación", 2, 7, 3)
    with cmc:
        tiempo_limite_multi = st.slider(
            "Tiempo máx. del solver multi-día (seg)", 30, 300, 120,
            help="El modelo multi-día es más grande (200 refs x 10 máquinas x N días); puede tardar más.",
        )

    if st.button("🗓️ Calcular plan multi-día", type="primary"):
        with st.spinner("Cargando el catálogo completo (200 referencias)..."):
            datos_200 = cargar_datos_extendido(
                archivo if archivo is not None else "AnexodeDatosxlsx.xlsx",
                dias_habiles=dias_habiles_input,
            )
        cij_200, demanda_200, setup_200 = datos_200["cij"], datos_200["demanda"], datos_200["setup"]

        cal = calibrar_demanda_extendida(demanda, demanda_200)
        st.info(
            f"Se detectó un factor de sobreestimación de **{cal['factor_sesgo']:.2f}x** "
            f"al comparar las {cal['n_referencias_comunes']} referencias que existen en "
            f"ambas fuentes de datos. La demanda de las 200 referencias se corrigió "
            f"dividiendo por ese factor antes de resolver el modelo."
        )
        demanda_horizonte = cal["demanda_calibrada"] * dias_horizonte

        with st.spinner(f"Resolviendo el modelo multi-día ({dias_horizonte} días, "
                         f"{len(cij_200)} referencias)... puede tardar 1-3 minutos."):
            res_multi = resolver_multidia(
                cij_200, demanda_horizonte, setup_200,
                dias=dias_horizonte, cap_minutos=cap_min, tiempo_limite_seg=tiempo_limite_multi,
            )
        st.session_state["resultado_multi"] = res_multi

    res_multi = st.session_state.get("resultado_multi")

    if res_multi is None:
        st.warning("Da clic en **Calcular plan multi-día** para resolver el modelo.")
    else:
        estado_txt = res_multi["estado"]

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Estado del solver", estado_txt)
        with m2:
            st.metric("Tiempo total (horizonte)", f"{res_multi['valor_objetivo']:,.0f} min".replace(",", "."))
        with m3:
            st.metric("Horizonte", f"{res_multi['dias']} días")
        with m4:
            refs_cubiertas = res_multi["plan"]["producto"].nunique() if len(res_multi["plan"]) else 0
            st.metric("Referencias cubiertas", refs_cubiertas)

        if estado_txt == "Optimal":
            st.markdown("")
            badge("GAP = 0 % — óptimo certificado, catálogo completo cubierto", "ok")
        elif estado_txt == "Infeasible":
            badge(f"Infactible con {res_multi['dias']} días — prueba a aumentar el horizonte", "bad")
        else:
            badge("Sin GAP certificado — sube el tiempo límite del solver", "warn")

        if len(res_multi["plan"]) > 0:
            st.markdown("")
            util_pct = (res_multi["utilizacion_dia_maquina"] / res_multi["cap_minutos"] * 100).round(1)
            st.plotly_chart(heatmap_utilizacion(util_pct, "Utilización por día y máquina (%)"),
                             use_container_width=True)

            st.subheader("Alistamientos (setups) por día y máquina")
            st.dataframe(res_multi["setups_dia_maquina"], use_container_width=True)

            st.subheader("Plan de producción — qué, dónde y cuándo")
            dia_ver = st.selectbox("Ver el plan del día:", sorted(res_multi["plan"]["dia"].unique()))
            plan_dia = res_multi["plan"][res_multi["plan"]["dia"] == dia_ver].copy()
            plan_dia = plan_dia.sort_values("unidades", ascending=False)
            plan_dia["unidades"] = plan_dia["unidades"].round(0)
            st.dataframe(plan_dia[["maquina", "producto", "unidades"]], use_container_width=True, hide_index=True)

            buffer_multi = io.BytesIO()
            with pd.ExcelWriter(buffer_multi, engine="openpyxl") as writer:
                res_multi["plan"].round(1).to_excel(writer, sheet_name="Plan_MultiDia", index=False)
                util_pct.to_excel(writer, sheet_name="Utilizacion_Dia_Maquina")
                res_multi["setups_dia_maquina"].to_excel(writer, sheet_name="Setups_Dia_Maquina")
            st.download_button(
                "⬇️ Descargar plan multi-día en Excel",
                data=buffer_multi.getvalue(),
                file_name="plan_multidia_rpm.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
