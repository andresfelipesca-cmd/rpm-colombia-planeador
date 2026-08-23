# -*- coding: utf-8 -*-
"""
rpm_core.py
Motor de datos, clustering y optimización para la asignación de
máquinas por producto en RPM Colombia.

Este módulo se separó del front-end (Streamlit) a propósito, para que
cualquier persona lo pueda usar también desde un notebook, un script
de consola o incluso un macro de Excel que llame a Python (xlwings).

Autor: prototipo generado para el proyecto de maestría
        "Optimización del sistema de producción en RPM Colombia".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MACHINES = [f"M{i}" for i in range(1, 11)]
CAP_MINUTOS_DIA = 1440  # capacidad diaria por máquina (24h x 60 min)


# --------------------------------------------------------------------------
# 1. CARGA DE DATOS
# --------------------------------------------------------------------------
def cargar_datos(path_excel: str) -> dict:
    """
    Lee el Excel anexo de la tesis y devuelve:
      - cij: DataFrame (producto x máquina) con el tiempo de producción
             de una unidad, en MINUTOS (ya imputado, sin ceros).
      - demanda: Serie (producto) con la demanda diaria a cumplir.
      - setup: Serie (máquina) con el tiempo de alistamiento promedio,
               en minutos.
    Usa el bloque de 99 referencias que ya trae la hoja "MODELO PL"
    (el mismo subconjunto que se optimizó en la tesis), pero la función
    está escrita para aceptar cualquier número de filas: si en el
    futuro se agregan más referencias a esa hoja, el resto del código
    no necesita cambiar.
    """
    xls = pd.ExcelFile(path_excel)

    # --- Tiempos de Set-up promedio por máquina (Tabla 1 de la tesis) ---
    setup_raw = pd.read_excel(xls, "Data de Máquinas", header=0, nrows=10,
                               usecols=[0, 1, 2])
    setup_raw.columns = ["Maquina_desc", "Indicador", "SetUp"]
    setup = setup_raw.set_index("Indicador")["SetUp"]
    setup.index = setup.index.str.strip()

    # --- Bloque Cij + demanda dentro de "MODELO PL" ---
    modelo = pd.read_excel(xls, "MODELO PL", header=None)

    # Bloque 1: tiempos Cij (empieza en la fila 3 de Excel -> índice 2)
    cij_block = modelo.iloc[2:107, 0:11].copy()
    cij_block.columns = ["Producto"] + MACHINES
    cij_block["Producto"] = cij_block["Producto"].astype(str).str.strip()
    cij_block = cij_block.dropna(subset=["Producto"])
    cij_block = cij_block[cij_block["Producto"] != "nan"]
    # Filas de resumen que no son productos (setup por máquina, F.O., etc.)
    cij_block[MACHINES] = cij_block[MACHINES].apply(pd.to_numeric, errors="coerce")
    cij_block = cij_block.dropna(subset=MACHINES)
    cij = cij_block.set_index("Producto")[MACHINES].astype(float)

    # Bloque 2: demanda (empieza en la fila 109 -> índice 108).
    # Columna A = producto, columna N (índice 13) = demanda diaria.
    dem_block = modelo.iloc[108:207, [0, 13]].copy()
    dem_block.columns = ["Producto", "Demanda"]
    dem_block["Producto"] = dem_block["Producto"].astype(str).str.strip()
    dem_block = dem_block.dropna(subset=["Demanda"])
    dem_block = dem_block[dem_block["Producto"] != "PRODUCTO"]
    demanda = dem_block.set_index("Producto")["Demanda"].astype(float)

    # Nos quedamos solo con productos que tengan AMBOS datos
    productos_validos = cij.index.intersection(demanda.index)
    cij = cij.loc[productos_validos]
    demanda = demanda.loc[productos_validos]

    return {"cij": cij, "demanda": demanda, "setup": setup}


def cargar_datos_extendido(path_excel: str, dias_habiles: int = 180) -> dict:
    """
    Versión EXTENDIDA de cargar_datos(): en vez de las ~99 referencias que
    ya trae resuelta la hoja "MODELO PL", usa las 200 referencias
    completas del histórico (hoja "DATOS SIMULADOS", tiempos ya
    imputados) y estima la demanda diaria de cada una a partir de sus
    ventas del semestre (hoja "Unidades Vendidas (SEMESTRE)"), dividiendo
    entre `dias_habiles`.

    Esto demuestra que, al no depender de Excel Solver, el modelo puede
    correr con el catálogo completo, no solo con el top ~99. La demanda
    diaria aquí es una ESTIMACIÓN (unidades del semestre / días hábiles),
    no un dato medido día a día como sí lo es para las 99 referencias de
    cargar_datos() — esto debe declararse así en la tesis si se usa esta
    versión extendida.
    """
    xls = pd.ExcelFile(path_excel)

    setup_raw = pd.read_excel(xls, "Data de Máquinas", header=0, nrows=10,
                               usecols=[0, 1, 2])
    setup_raw.columns = ["Maquina_desc", "Indicador", "SetUp"]
    setup = setup_raw.set_index("Indicador")["SetUp"]
    setup.index = setup.index.str.strip()

    # Tiempos Cij imputados (segundos -> minutos), las 200 referencias
    sim = pd.read_excel(xls, "DATOS SIMULADOS", header=1)
    sim = sim.rename(columns={sim.columns[0]: "Producto"})
    sim["Producto"] = sim["Producto"].astype(str).str.strip()
    sim = sim.dropna(subset=["Producto"])
    sim = sim[sim["Producto"] != "nan"]
    cij = sim.set_index("Producto")[MACHINES].astype(float) / 60.0

    # Demanda diaria estimada a partir de ventas del semestre
    ventas = pd.read_excel(xls, "Unidades Vendidas (SEMESTRE)", header=0)
    ventas.columns = [str(c).strip() for c in ventas.columns]
    ventas["PRODUCTO"] = ventas["PRODUCTO"].astype(str).str.strip()
    ventas = ventas.dropna(subset=["PRODUCTO"])
    demanda = (ventas.set_index("PRODUCTO")["UNIDADES"].astype(float) / dias_habiles)

    productos_validos = cij.index.intersection(demanda.index)
    cij = cij.loc[productos_validos]
    demanda = demanda.loc[productos_validos]

    return {"cij": cij, "demanda": demanda, "setup": setup,
            "dias_habiles": dias_habiles}


# --------------------------------------------------------------------------
# 2. CLUSTERING DE PRODUCTOS (demanda + variabilidad entre máquinas)
# --------------------------------------------------------------------------
def clusterizar_productos(cij: pd.DataFrame, demanda: pd.Series,
                           n_clusters: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Agrupa los productos según dos variables clave para decidir en qué
    tipo de máquina conviene producirlos:

      - volumen: demanda diaria (a mayor volumen, más conviene una
        máquina "de producción continua" con poco setup).
      - variabilidad: qué tan disperso es el tiempo de producción entre
        las 10 máquinas (coeficiente de variación de Cij). Un producto
        con alta variabilidad "le importa mucho" en qué máquina caiga;
        uno con baja variabilidad da casi igual.

    Devuelve un DataFrame con el cluster y una etiqueta de rol sugerido
    ("Producción continua", "Rotación media", "Comodín / baja escala"),
    que es justamente la segmentación que la tesis propone de forma
    cualitativa en el punto 7.4.3 — aquí queda soportada con datos.
    """
    cv = cij.std(axis=1) / cij.mean(axis=1)
    feats = pd.DataFrame({
        "demanda": demanda,
        "cv_tiempo": cv,
    })

    X = StandardScaler().fit_transform(feats.values)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    feats["cluster"] = km.fit_predict(X)

    # Ordenamos los clusters de mayor a menor demanda promedio para
    # asignar etiquetas de negocio consistentes
    orden = feats.groupby("cluster")["demanda"].mean().sort_values(ascending=False).index
    etiquetas = {orden[0]: "Producción continua (alto volumen)"}
    if n_clusters >= 2:
        etiquetas[orden[-1]] = "Comodín / baja escala"
    for c in orden[1:-1] if n_clusters > 2 else []:
        etiquetas[c] = "Rotación media"

    feats["rol_sugerido"] = feats["cluster"].map(etiquetas)
    return feats


# --------------------------------------------------------------------------
# 2b. VALIDACIÓN DEL NÚMERO DE GRUPOS (Elbow + Silhouette)
# --------------------------------------------------------------------------
def elbow_silhouette(features: pd.DataFrame, k_range=range(2, 7), seed: int = 42) -> pd.DataFrame:
    """
    Corre K-Means con distintos valores de k y reporta, para cada uno:
      - inercia: qué tan compactos quedan los grupos (método del codo;
        se busca el "codo" de la curva, no necesariamente el mínimo).
      - silhouette: qué tan bien separado está cada punto de los demás
        grupos (-1 a 1; más alto es mejor). Aquí sí se busca el máximo.

    Esto reemplaza la elección manual de "3 grupos" (como en la Tabla 3
    de la tesis) por un número justificado con evidencia.
    """
    from sklearn.metrics import silhouette_score

    X = StandardScaler().fit_transform(features.values)
    filas = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        filas.append({
            "k": k,
            "inercia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
        })
    return pd.DataFrame(filas)


def prueba_estadistica_clusters(valores: pd.Series, grupos: pd.Series) -> dict:
    """
    Prueba de Kruskal-Wallis: confirma (o no) que los grupos encontrados
    por K-Means son estadísticamente distintos entre sí en la variable
    dada, en vez de solo "verse distintos" en una gráfica. Se usa
    Kruskal-Wallis (no ANOVA clásico) porque no exige que los datos sean
    normales dentro de cada grupo — más seguro con muestras pequeñas
    como las de este proyecto.
    """
    from scipy import stats

    muestras = [valores[grupos == g].values for g in sorted(grupos.unique())]
    muestras = [m for m in muestras if len(m) > 0]
    h, p = stats.kruskal(*muestras)
    return {
        "estadistico_H": h,
        "valor_p": p,
        "significativo_al_5pct": p < 0.05,
    }


# --------------------------------------------------------------------------
# 2c. CLUSTERING DE MÁQUINAS (valida/corrige la Tabla 3 de la tesis)
# --------------------------------------------------------------------------
def clusterizar_maquinas(cij: pd.DataFrame, setup: pd.Series,
                          n_clusters: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Agrupa las 10 máquinas según su comportamiento MEDIDO, no según su
    antigüedad percibida (que es el criterio manual de la Tabla 3 de la
    tesis, y que tiene una inconsistencia detectada: M10, clasificada
    como "intermedia", tiene un SetUp más alto que M6, clasificada como
    "antigua"). Variables usadas:
      - SetUp promedio de la máquina.
      - Tiempo de operación promedio (Cij promedio, sobre todos los
        productos que puede fabricar).
      - Varianza del tiempo de operación entre productos.
    """
    tiempo_prom = cij.mean(axis=0)
    tiempo_var = cij.var(axis=0)
    feats = pd.DataFrame({
        "setup_promedio": setup,
        "tiempo_operacion_promedio": tiempo_prom,
        "varianza_tiempo": tiempo_var,
    })

    X = StandardScaler().fit_transform(feats.values)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    feats["cluster"] = km.fit_predict(X)

    orden = feats.groupby("cluster")["setup_promedio"].mean().sort_values(ascending=False).index
    etiquetas = {orden[0]: "Grupo 1 (mayor SetUp)"}
    if n_clusters >= 2:
        etiquetas[orden[-1]] = f"Grupo {n_clusters} (menor SetUp)"
    for pos, c in enumerate(orden[1:-1] if n_clusters > 2 else [], start=2):
        etiquetas[c] = f"Grupo {pos} (SetUp intermedio)"

    feats["grupo_sugerido"] = feats["cluster"].map(etiquetas)
    return feats


# --------------------------------------------------------------------------
# 3. MODELO DE OPTIMIZACIÓN (MILP corregido)
# --------------------------------------------------------------------------
def resolver_asignacion(cij: pd.DataFrame, demanda: pd.Series,
                         setup: pd.Series,
                         cap_minutos: float = CAP_MINUTOS_DIA,
                         tiempo_limite_seg: int = 60) -> dict:
    """
    Formulación corregida respecto a la de la tesis (sección 7.3):

      - Xij >= 0          cantidad de producto i asignada a la máquina j
      - Yij binaria       1 si la máquina j produce ALGO del producto i
      - FO: min  sum(Cij * Xij) + sum(Sj * Yij)
      - sum_j Xij >= Di                          (cumplir demanda)
      - sum_i (Cij*Xij) + sum_i (Sj*Yij) <= 1440  (por máquina, NO 21 ni "24h")
      - Xij <= Di * Yij                          (Big-M AJUSTADO a Di, no
                                                    un número arbitrario:
                                                    esto acelera mucho la
                                                    resolución del solver)

    A diferencia del modelo de la tesis, aquí NO existe límite de 200/990
    variables: esto puede correr con 99, 200 o 700+ referencias sin
    cambiar una línea de código, porque no depende de Excel Solver.
    """
    productos = list(cij.index)
    prob = pulp.LpProblem("Asignacion_RPM", pulp.LpMinimize)

    X = pulp.LpVariable.dicts("X", (productos, MACHINES), lowBound=0)
    Y = pulp.LpVariable.dicts("Y", (productos, MACHINES), cat="Binary")

    # Función objetivo
    prob += (
        pulp.lpSum(cij.loc[i, j] * X[i][j] for i in productos for j in MACHINES)
        + pulp.lpSum(setup[j] * Y[i][j] for i in productos for j in MACHINES)
    )

    # Cumplimiento de demanda
    for i in productos:
        prob += pulp.lpSum(X[i][j] for j in MACHINES) >= demanda[i], f"Demanda_{i}"

    # Capacidad diaria POR MÁQUINA (1440 min, corrige el error de la tesis)
    for j in MACHINES:
        prob += (
            pulp.lpSum(cij.loc[i, j] * X[i][j] for i in productos)
            + pulp.lpSum(setup[j] * Y[i][j] for i in productos)
            <= cap_minutos
        ), f"Capacidad_{j}"

    # Enlace Xij <= Di * Yij (Big-M ajustado, no arbitrario)
    for i in productos:
        for j in MACHINES:
            prob += X[i][j] <= demanda[i] * Y[i][j], f"Link_{i}_{j}"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=tiempo_limite_seg)
    prob.solve(solver)

    estado = pulp.LpStatus[prob.status]
    # Con CBC, estado == "Optimal" implica que se demostró optimalidad
    # (rama y acotamiento agotado, GAP = 0 %), no solo una solución
    # factible como con el algoritmo evolutivo original de la tesis.
    gap_certificado = (estado == "Optimal")

    asignacion = pd.DataFrame(0.0, index=productos, columns=MACHINES)
    setups = pd.DataFrame(0, index=productos, columns=MACHINES)
    for i in productos:
        for j in MACHINES:
            asignacion.loc[i, j] = X[i][j].value() or 0.0
            setups.loc[i, j] = int(round(Y[i][j].value() or 0))

    utilizacion = pd.Series(0.0, index=MACHINES)
    n_setups = pd.Series(0, index=MACHINES)
    for j in MACHINES:
        tiempo_prod = sum(cij.loc[i, j] * asignacion.loc[i, j] for i in productos)
        tiempo_setup = sum(setup[j] * setups.loc[i, j] for i in productos)
        utilizacion[j] = tiempo_prod + tiempo_setup
        n_setups[j] = setups[j].sum()

    return {
        "estado": estado,
        "gap_certificado": gap_certificado,
        "valor_objetivo": pulp.value(prob.objective),
        "asignacion": asignacion,
        "setups": setups,
        "utilizacion_min": utilizacion,
        "n_setups": n_setups,
        "cap_minutos": cap_minutos,
    }


# --------------------------------------------------------------------------
# 4. PRECIOS SOMBRA (variables duales) — cuello de botella real
# --------------------------------------------------------------------------
def calcular_precios_sombra(cij: pd.DataFrame, demanda: pd.Series,
                             setup: pd.Series, setups_fijos: pd.DataFrame,
                             cap_minutos: float = CAP_MINUTOS_DIA) -> pd.Series:
    """
    Los precios sombra de un MILP no están bien definidos mientras las
    variables Yij sigan siendo binarias (el dual solo existe para
    programas LINEALES). La técnica estándar es: fijar el patrón de
    setups (Yij) en el valor óptimo ya encontrado por resolver_asignacion,
    y resolver el LP resultante (ahora solo con Xij, continuas). Los
    precios sombra de ESE LP sí son válidos, y responden la pregunta:
    "manteniendo el mismo patrón de cambios de molde, ¿cuánto bajaría el
    tiempo total del sistema si esta máquina tuviera 1 minuto más de
    capacidad?" — un valor alto identifica el cuello de botella real.
    """
    productos = list(cij.index)
    prob = pulp.LpProblem("Precios_Sombra", pulp.LpMinimize)
    X = pulp.LpVariable.dicts("X", (productos, MACHINES), lowBound=0)

    prob += pulp.lpSum(cij.loc[i, j] * X[i][j] for i in productos for j in MACHINES)

    for i in productos:
        prob += pulp.lpSum(X[i][j] for j in MACHINES) >= demanda[i], f"Demanda_{i}"

    for j in MACHINES:
        tiempo_setup_fijo = sum(setup[j] * setups_fijos.loc[i, j] for i in productos)
        prob += (
            pulp.lpSum(cij.loc[i, j] * X[i][j] for i in productos)
            <= cap_minutos - tiempo_setup_fijo
        ), f"Capacidad_{j}"

    for i in productos:
        for j in MACHINES:
            if setups_fijos.loc[i, j] == 0:
                prob += X[i][j] == 0, f"Fijo_{i}_{j}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    precios = {}
    for j in MACHINES:
        restr = prob.constraints[f"Capacidad_{j}"]
        precios[j] = restr.pi if restr.pi is not None else 0.0
    return pd.Series(precios, name="precio_sombra_min")


# --------------------------------------------------------------------------
# 5. POLÍTICA EMPÍRICA DE REFERENCIA (línea base sin optimización)
# --------------------------------------------------------------------------
def politica_empirica(cij: pd.DataFrame, demanda: pd.Series, setup: pd.Series,
                       cap_minutos: float = CAP_MINUTOS_DIA) -> dict:
    """
    Heurística "miope" (greedy) que representa cómo asignaría un
    supervisor experimentado SIN apoyo de un modelo de optimización:
    procesa los productos de mayor a menor demanda y, para cada uno,
    lo asigna a la máquina más rápida (menor Cij) que todavía tenga
    capacidad disponible, sin anticipar el efecto sobre los productos
    que faltan por asignar.

    Nota metodológica (importante para la tesis): no existe en los datos
    un registro histórico verificable de qué máquina usó cada supervisor
    para cada referencia. Esta política es una heurística estándar de la
    literatura de programación de la producción (asignación greedy por
    menor tiempo unitario), usada como punto de comparación razonable
    frente al óptimo — debe describirse así, no como un dato histórico
    real de RPM Colombia.
    """
    productos_orden = demanda.sort_values(ascending=False).index.tolist()
    capacidad_restante = pd.Series(float(cap_minutos), index=MACHINES)
    setup_pagado = pd.DataFrame(False, index=cij.index, columns=MACHINES)
    asignacion = pd.DataFrame(0.0, index=cij.index, columns=MACHINES)
    pendientes = demanda.astype(float).copy()

    for i in productos_orden:
        maquinas_orden = cij.loc[i].sort_values().index.tolist()
        for j in maquinas_orden:
            if pendientes[i] <= 1e-6:
                break
            setup_necesario = 0.0 if setup_pagado.loc[i, j] else float(setup[j])
            cap_disp = capacidad_restante[j] - setup_necesario
            if cap_disp <= 0:
                continue
            unidades_posibles = cap_disp / cij.loc[i, j]
            unidades_asignar = min(unidades_posibles, pendientes[i])
            if unidades_asignar <= 0:
                continue
            tiempo_usado = unidades_asignar * cij.loc[i, j] + setup_necesario
            capacidad_restante[j] -= tiempo_usado
            asignacion.loc[i, j] += unidades_asignar
            pendientes[i] -= unidades_asignar
            setup_pagado.loc[i, j] = True

    demanda_incumplida = pendientes[pendientes > 1e-6]
    utilizacion = cap_minutos - capacidad_restante
    valor_objetivo = float(utilizacion.sum())

    return {
        "asignacion": asignacion,
        "utilizacion_min": utilizacion,
        "valor_objetivo": valor_objetivo,
        "demanda_incumplida": demanda_incumplida,
        "cap_minutos": cap_minutos,
    }


# --------------------------------------------------------------------------
# 5b. CUELLO DE BOTELLA POR RE-OPTIMIZACIÓN (alternativa robusta al dual)
# --------------------------------------------------------------------------
def analisis_cuello_de_botella(cij: pd.DataFrame, demanda: pd.Series,
                                setup: pd.Series, cap_minutos: float = CAP_MINUTOS_DIA,
                                delta: float = 30.0, tiempo_limite_seg: int = 60) -> pd.DataFrame:
    """
    Alternativa práctica y más robusta que el precio sombra clásico
    (que en este problema resulta degenerado: ver nota en
    calcular_precios_sombra). En vez de leer un dual, se vuelve a resolver
    el MILP completo dándole `delta` minutos ADICIONALES de capacidad a
    una sola máquina a la vez, y se mide cuánto baja el tiempo total del
    sistema. La máquina cuya capacidad adicional genera la mayor mejora
    es, por definición, el cuello de botella real del sistema — sin
    depender de ninguna sutileza de dualidad.
    """
    base = resolver_asignacion(cij, demanda, setup, cap_minutos, tiempo_limite_seg)
    filas = []
    for j in MACHINES:
        cap_extra = {m: cap_minutos for m in MACHINES}
        # resolver_asignacion usa una capacidad única; para variar solo
        # una máquina se resuelve un modelo ad-hoc con capacidad por máquina
        productos = list(cij.index)
        prob = pulp.LpProblem("Sensibilidad_Capacidad", pulp.LpMinimize)
        X = pulp.LpVariable.dicts("X", (productos, MACHINES), lowBound=0)
        Y = pulp.LpVariable.dicts("Y", (productos, MACHINES), cat="Binary")
        prob += (
            pulp.lpSum(cij.loc[i, m] * X[i][m] for i in productos for m in MACHINES)
            + pulp.lpSum(setup[m] * Y[i][m] for i in productos for m in MACHINES)
        )
        for i in productos:
            prob += pulp.lpSum(X[i][m] for m in MACHINES) >= demanda[i]
        for m in MACHINES:
            cap_m = cap_minutos + delta if m == j else cap_minutos
            prob += (
                pulp.lpSum(cij.loc[i, m] * X[i][m] for i in productos)
                + pulp.lpSum(setup[m] * Y[i][m] for i in productos)
                <= cap_m
            )
        for i in productos:
            for m in MACHINES:
                prob += X[i][m] <= demanda[i] * Y[i][m]
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=tiempo_limite_seg))

        estado_m = pulp.LpStatus[prob.status]
        obj = pulp.value(prob.objective)
        mejora = base["valor_objetivo"] - obj if obj is not None else None
        filas.append({
            "maquina": j,
            "estado": estado_m,
            "utilizacion_actual_pct": round(base["utilizacion_min"][j] / cap_minutos * 100, 1),
            "mejora_min_por_+30min_capacidad": round(mejora, 2) if mejora is not None else None,
        })

    df = pd.DataFrame(filas)
    # Cualquier mejora negativa indica que el solver no alcanzó el óptimo
    # certificado dentro del tiempo límite para esa corrida (con más
    # capacidad disponible el óptimo nunca puede ser peor) — se recorta a
    # 0 y se marca, en vez de reportar un número engañoso.
    no_optimo = df["mejora_min_por_+30min_capacidad"] < -1e-6
    if no_optimo.any():
        df.loc[no_optimo, "estado"] = df.loc[no_optimo, "estado"] + " (sin GAP=0, aumentar tiempo_limite_seg)"
        df.loc[no_optimo, "mejora_min_por_+30min_capacidad"] = np.nan
    df = df.sort_values("mejora_min_por_+30min_capacidad", ascending=False)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 6. ANÁLISIS DE SENSIBILIDAD
# --------------------------------------------------------------------------
def analisis_sensibilidad(cij: pd.DataFrame, demanda: pd.Series, setup: pd.Series,
                           cap_minutos: float = CAP_MINUTOS_DIA,
                           factores_demanda=(0.8, 0.9, 1.0, 1.1, 1.2),
                           tiempo_limite_seg: int = 60) -> pd.DataFrame:
    """
    Resuelve el modelo variando la demanda en cada factor (0.8 = 20 %
    menos, 1.2 = 20 % más) y reporta cómo cambian: el tiempo total, la
    utilización promedio y la máquina más cargada. Sirve para responder,
    con evidencia, la pregunta típica de sustentación: "¿qué tan estable
    es esta solución frente a cambios en la demanda?"
    """
    filas = []
    for f in factores_demanda:
        dem_f = demanda * f
        res = resolver_asignacion(cij, dem_f, setup, cap_minutos, tiempo_limite_seg)
        util_pct = (res["utilizacion_min"] / cap_minutos * 100)
        filas.append({
            "factor_demanda": f,
            "estado": res["estado"],
            "valor_objetivo_min": res["valor_objetivo"],
            "utilizacion_promedio_pct": util_pct.mean(),
            "utilizacion_maxima_pct": util_pct.max(),
            "maquina_mas_cargada": util_pct.idxmax(),
            "total_setups": int(res["n_setups"].sum()),
        })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# 7. COMPARACIÓN MULTI-ESCENARIO (evidencia robusta para la Hipótesis)
# --------------------------------------------------------------------------
def comparar_politicas_multiescenario(cij: pd.DataFrame, demanda: pd.Series, setup: pd.Series,
                                       cap_minutos: float = CAP_MINUTOS_DIA,
                                       factores_demanda=(0.8, 0.9, 1.0, 1.1, 1.2),
                                       tiempo_limite_seg: int = 60) -> pd.DataFrame:
    """
    Compara el modelo óptimo contra la política empírica en varios
    escenarios de demanda (no un solo punto), para dar evidencia robusta
    de que la mejora del modelo no es un resultado aislado. Es la base de
    la prueba de hipótesis formal (Fase 7): si el modelo es
    consistentemente mejor en todos los escenarios, la evidencia es mucho
    más sólida que comparar un único caso.
    """
    filas = []
    for f in factores_demanda:
        dem_f = demanda * f
        res_lp = resolver_asignacion(cij, dem_f, setup, cap_minutos, tiempo_limite_seg)
        res_emp = politica_empirica(cij, dem_f, setup, cap_minutos)
        mejora = (res_emp["valor_objetivo"] - res_lp["valor_objetivo"]) / res_emp["valor_objetivo"] * 100
        filas.append({
            "factor_demanda": f,
            "estado_lp": res_lp["estado"],
            "tiempo_lp_min": res_lp["valor_objetivo"],
            "tiempo_empirico_min": res_emp["valor_objetivo"],
            "mejora_pct": mejora,
            "demanda_incumplida_empirica": len(res_emp["demanda_incumplida"]) > 0,
        })
    return pd.DataFrame(filas)
