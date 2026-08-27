# -*- coding: utf-8 -*-
"""
app.py — Planeador de Producción RPM Colombia (prototipo)

Cómo correrlo:
    pip install streamlit pandas openpyxl scikit-learn pulp matplotlib
    streamlit run app.py

Sube el mismo Excel anexo de la tesis (o edítalo con la demanda del día)
y la app calcula, en segundos, qué producir en cada máquina y muestra
la utilización de capacidad y la segmentación de productos.
"""

import io

import matplotlib.pyplot as plt
import pandas as pd
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

st.set_page_config(page_title="RPM Colombia — Planeador de Producción", layout="wide")

st.title("🏭 Planeador de Producción — RPM Colombia")
st.caption(
    "Prototipo de apoyo a la decisión: a partir de los tiempos históricos "
    "por máquina y la demanda del día, recomienda cuánto producir en cada "
    "máquina y agrupa los productos por su rol óptimo en planta."
)

# --------------------------------------------------------------------
# Barra lateral — datos y parámetros
# --------------------------------------------------------------------
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

correr = st.sidebar.button("🚀 Calcular plan de producción", type="primary")


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
col3.metric("Demanda total del día (unid.)", f"{int(demanda.sum()):,}")

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
        fig_e, ax_e = plt.subplots(figsize=(5, 3.5))
        ax_e.plot(tabla_k_prod["k"], tabla_k_prod["inercia"], marker="o")
        ax_e.set_xlabel("k (número de grupos)")
        ax_e.set_ylabel("Inercia")
        ax_e.set_title("Método del codo")
        st.pyplot(fig_e)
    with c2:
        fig_s, ax_s = plt.subplots(figsize=(5, 3.5))
        ax_s.plot(tabla_k_prod["k"], tabla_k_prod["silhouette"], marker="o", color="#2E7D32")
        ax_s.set_xlabel("k (número de grupos)")
        ax_s.set_ylabel("Silhouette Score")
        ax_s.set_title("Validación por Silhouette")
        st.pyplot(fig_s)

    mejor_k = int(tabla_k_prod.loc[tabla_k_prod["silhouette"].idxmax(), "k"])
    st.caption(f"El mejor k según Silhouette es **{mejor_k}**. Se muestra el resultado con k={n_clusters} "
               "(ajustable en la barra lateral) para mantener la interpretación de negocio en 3 roles.")

    clusters = clusterizar_productos(cij, demanda, n_clusters=n_clusters)

    c3, c4 = st.columns([1, 2])
    with c3:
        conteo = clusters["rol_sugerido"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(conteo.values, labels=conteo.index, autopct="%1.0f%%", startangle=90)
        ax.set_title("Distribución de productos por rol sugerido")
        st.pyplot(fig)
    with c4:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for rol, sub in clusters.groupby("rol_sugerido"):
            ax2.scatter(sub["demanda"], sub["cv_tiempo"], label=rol, alpha=0.7)
        ax2.set_xlabel("Demanda diaria (unidades)")
        ax2.set_ylabel("Variabilidad del tiempo entre máquinas (CV)")
        ax2.set_title("Mapa de productos: volumen vs. variabilidad")
        ax2.legend(fontsize=8)
        st.pyplot(fig2)

    test_dem = prueba_estadistica_clusters(clusters["demanda"], clusters["cluster"])
    test_cv = prueba_estadistica_clusters(clusters["cv_tiempo"], clusters["cluster"])
    st.markdown(
        f"**Prueba de Kruskal-Wallis** — ¿los grupos son estadísticamente distintos?  \n"
        f"Demanda: H={test_dem['estadistico_H']:.2f}, p={test_dem['valor_p']:.2e} "
        f"({'✅ significativo' if test_dem['significativo_al_5pct'] else '❌ no significativo'})  \n"
        f"Variabilidad: H={test_cv['estadistico_H']:.2f}, p={test_cv['valor_p']:.2e} "
        f"({'✅ significativo' if test_cv['significativo_al_5pct'] else '❌ no significativo'})"
    )

    with st.expander("Ver tabla completa de clusters de productos"):
        st.dataframe(clusters.sort_values("demanda", ascending=False))

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
        fig_em, ax_em = plt.subplots(figsize=(5, 3.5))
        ax_em.plot(tabla_k_maq["k"], tabla_k_maq["inercia"], marker="o")
        ax_em.set_xlabel("k (número de grupos)")
        ax_em.set_ylabel("Inercia")
        ax_em.set_title("Método del codo — máquinas")
        st.pyplot(fig_em)
    with c6:
        fig_sm, ax_sm = plt.subplots(figsize=(5, 3.5))
        ax_sm.plot(tabla_k_maq["k"], tabla_k_maq["silhouette"], marker="o", color="#2E7D32")
        ax_sm.set_xlabel("k (número de grupos)")
        ax_sm.set_ylabel("Silhouette Score")
        ax_sm.set_title("Validación por Silhouette — máquinas")
        st.pyplot(fig_sm)

    n_clusters_maq = st.slider("Número de grupos de máquinas", 2, 5, 3)
    clusters_maq = clusterizar_maquinas(cij, setup, n_clusters=n_clusters_maq)
    st.dataframe(
        clusters_maq[["setup_promedio", "tiempo_operacion_promedio", "varianza_tiempo", "grupo_sugerido"]]
        .sort_values("setup_promedio", ascending=False),
        use_container_width=True,
    )

    test_setup = prueba_estadistica_clusters(clusters_maq["setup_promedio"], clusters_maq["cluster"])
    st.markdown(
        f"**Prueba de Kruskal-Wallis (SetUp)**: H={test_setup['estadistico_H']:.2f}, "
        f"p={test_setup['valor_p']:.3f} "
        f"({'✅ significativo' if test_setup['significativo_al_5pct'] else '❌ no significativo'})"
    )
    if not test_setup["significativo_al_5pct"]:
        st.caption(
            "Con solo 10 máquinas, la prueba tiene poca potencia estadística — "
            "no alcanzar significancia no invalida la agrupación, pero sí debe "
            "declararse como limitación en la tesis (sección de Limitaciones, Fase 7)."
        )

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

st.success(
    f"Estado del solver: **{resultado['estado']}** "
    f"({'GAP = 0 %, óptimo certificado' if resultado['gap_certificado'] else 'sin GAP certificado, sube el tiempo límite'})"
    f"  |  Tiempo total (producción + setup): **{resultado['valor_objetivo']:.0f} min**"
)

asignacion = resultado["asignacion"]
util = resultado["utilizacion_min"]
n_setups = resultado["n_setups"]

tab_plan, tab_comp, tab_sens, tab_cuello, tab_multi = st.tabs([
    "📋 Plan y utilización",
    "⚖️ Vs. política empírica",
    "📈 Sensibilidad a la demanda",
    "🚧 Cuello de botella",
    "🗓️ Plan multi-día (200 refs)",
])

# ---------------- TAB 1: Plan y utilización ----------------
with tab_plan:
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(util.index, util.values, color="#4C72B0", label="Usado")
    ax3.axhline(resultado["cap_minutos"], color="red", linestyle="--", label="Capacidad (1440 min)")
    ax3.set_ylabel("Minutos usados en el día")
    ax3.set_title("Utilización de capacidad por máquina")
    ax3.legend()
    st.pyplot(fig3)

    resumen = pd.DataFrame({
        "Utilización (min)": util.round(1),
        "% de capacidad": (util / resultado["cap_minutos"] * 100).round(1),
        "N° de Set-ups": n_setups,
    })
    st.dataframe(resumen)

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

        fig4, ax4 = plt.subplots(figsize=(7, 4))
        ax4.plot(sens["factor_demanda"], sens["valor_objetivo_min"], marker="o", color="#1B3A6B")
        ax4.set_xlabel("Factor de demanda (1.0 = demanda real)")
        ax4.set_ylabel("Tiempo total del sistema (min)")
        ax4.set_title("Sensibilidad del tiempo total a la demanda")
        st.pyplot(fig4)

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

    dias_habiles_input = st.number_input(
        "Días hábiles usados para estimar la demanda (a partir de ventas del semestre)",
        min_value=60, max_value=300, value=180, step=10,
    )
    dias_horizonte = st.slider("Días del horizonte de planeación", 2, 7, 3)
    tiempo_limite_multi = st.slider(
        "Tiempo máximo de cómputo del solver multi-día (seg)", 30, 300, 120,
        help="El modelo multi-día es más grande (200 referencias x 10 máquinas x N días); "
             "puede tardar más que el modelo diario.",
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
        if estado_txt == "Optimal":
            st.success(
                f"Estado del solver: **{estado_txt}** (GAP = 0 %, óptimo certificado) — "
                f"las 200 referencias caben en el horizonte de {res_multi['dias']} días. "
                f"Tiempo total: **{res_multi['valor_objetivo']:.0f} min**."
            )
        elif estado_txt == "Infeasible":
            st.error(
                f"Estado del solver: **{estado_txt}** — con {res_multi['dias']} días el "
                f"catálogo completo no alcanza a caber. Prueba a aumentar el horizonte "
                f"(por ejemplo, a 4 o 5 días)."
            )
        else:
            st.warning(f"Estado del solver: **{estado_txt}** — sube el tiempo límite del solver.")

        if len(res_multi["plan"]) > 0:
            st.subheader("Utilización por día y máquina (%)")
            util_pct = (res_multi["utilizacion_dia_maquina"] / res_multi["cap_minutos"] * 100).round(1)
            st.dataframe(
                util_pct.style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=100).format("{:.1f}%"),
                use_container_width=True,
            )

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
