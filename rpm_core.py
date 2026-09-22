# -*- coding: utf-8 -*-
"""
rpm_core.py
Motor de datos, estadística y optimización para RPM Colombia.

La lógica de negocio está separada de Streamlit para que los cálculos se
puedan reutilizar desde un notebook, un script o pruebas automatizadas.

Principios de esta versión:
  1. Las unidades de producción son enteras.
  2. La demanda se satisface con igualdad.
  3. El mismo MILP se reutiliza en sensibilidad y cuello de botella.
  4. Las capacidades pueden variar por máquina.
  5. Los resultados distinguen factibilidad, optimalidad y estabilidad.
  6. La procedencia observada/imputada de C_ij se conserva para auditoría.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
import pulp
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

MACHINES = [f"M{i}" for i in range(1, 11)]
CAP_MINUTOS_DIA = 1440
TOL = 1e-6


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------
def _serie_capacidades(cap_minutos) -> pd.Series:
    """Normaliza una capacidad escalar, diccionario o Serie a una Serie M1..M10."""
    if np.isscalar(cap_minutos):
        caps = pd.Series(float(cap_minutos), index=MACHINES, dtype=float)
    elif isinstance(cap_minutos, pd.Series):
        caps = cap_minutos.reindex(MACHINES).astype(float)
    else:
        caps = pd.Series(cap_minutos, dtype=float).reindex(MACHINES)
    if caps.isna().any():
        faltan = caps[caps.isna()].index.tolist()
        raise ValueError(f"Faltan capacidades para: {faltan}")
    if (caps <= 0).any():
        raise ValueError("Todas las capacidades deben ser positivas.")
    return caps


def demanda_entera(demanda: pd.Series, factor: float = 1.0,
                    metodo: str = "round") -> pd.Series:
    """
    Escala una demanda y la lleva a unidades enteras.

    `round` usa redondeo convencional half-up: floor(x + 0.5). Para análisis
    de sensibilidad es preferible a dejar valores fraccionarios, porque el
    modelo declara X_ij en Z_+.
    """
    base = pd.Series(demanda, dtype=float)
    if (base < 0).any():
        raise ValueError("La demanda no puede ser negativa.")
    x = base.to_numpy() * float(factor)
    if metodo == "round":
        y = np.floor(x + 0.5)
    elif metodo == "ceil":
        y = np.ceil(x)
    elif metodo == "floor":
        y = np.floor(x)
    else:
        raise ValueError("metodo debe ser 'round', 'ceil' o 'floor'.")
    y = y.astype(int)
    # Si una referencia tenía demanda positiva, no desaparece por redondeo.
    y[(x > 0) & (y < 1)] = 1
    return pd.Series(y, index=base.index, dtype=int, name=demanda.name)


def _validar_demanda_entera(demanda: pd.Series, nombre: str = "demanda") -> pd.Series:
    s = pd.Series(demanda, dtype=float)
    if s.isna().any() or (~np.isfinite(s.to_numpy())).any():
        raise ValueError(f"{nombre} contiene valores no finitos.")
    if (s < 0).any():
        raise ValueError(f"{nombre} contiene valores negativos.")
    residuo = np.abs(s.to_numpy() - np.rint(s.to_numpy()))
    if (residuo > 1e-8).any():
        ejemplos = s.iloc[np.where(residuo > 1e-8)[0][:5]].to_dict()
        raise ValueError(
            f"{nombre} debe estar expresada en unidades enteras porque X es entera. "
            f"Ejemplos fraccionarios: {ejemplos}. Usa demanda_entera()."
        )
    return s.round().astype(int)


def metricas_concentracion_demanda(demanda: pd.Series) -> dict:
    """Pareto y Gini de la demanda, como indicadores descriptivos de concentración."""
    d = pd.Series(demanda, dtype=float).clip(lower=0).sort_values(ascending=False)
    total = float(d.sum())
    if total <= 0:
        return {"n80": 0, "share_top10_pct": 0.0, "gini": 0.0}
    acumulada = d.cumsum() / total
    n80 = int((acumulada < 0.80).sum() + 1)
    share_top10 = float(d.head(10).sum() / total * 100)
    x = np.sort(d.to_numpy())
    n = len(x)
    if n == 0 or x.sum() == 0:
        gini = 0.0
    else:
        gini = float((2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n)
    return {"n80": n80, "share_top10_pct": share_top10, "gini": gini}


# ---------------------------------------------------------------------------
# 1. Carga y calidad de datos
# ---------------------------------------------------------------------------
def _cargar_setup(xls) -> pd.Series:
    setup_raw = pd.read_excel(xls, "Data de Máquinas", header=0, nrows=10,
                              usecols=[0, 1, 2])
    setup_raw.columns = ["Maquina_desc", "Indicador", "SetUp"]
    setup = setup_raw.set_index("Indicador")["SetUp"].astype(float)
    setup.index = setup.index.astype(str).str.strip()
    return setup.reindex(MACHINES)


def cargar_datos(path_excel: str) -> dict:
    """
    Carga la instancia de 99 referencias publicada en la hoja MODELO PL.

    Esta ruta reproduce la matriz C_ij utilizada para obtener Z*=11.547,41.
    La procedencia observada/imputada se puede auditar por separado con
    `construir_cij_con_imputacion`.
    """
    xls = pd.ExcelFile(path_excel)
    setup = _cargar_setup(xls)
    modelo = pd.read_excel(xls, "MODELO PL", header=None)

    cij_block = modelo.iloc[2:107, 0:11].copy()
    cij_block.columns = ["Producto"] + MACHINES
    cij_block["Producto"] = cij_block["Producto"].astype(str).str.strip()
    cij_block = cij_block.dropna(subset=["Producto"])
    cij_block = cij_block[cij_block["Producto"] != "nan"]
    cij_block[MACHINES] = cij_block[MACHINES].apply(pd.to_numeric, errors="coerce")
    cij_block = cij_block.dropna(subset=MACHINES)
    cij = cij_block.set_index("Producto")[MACHINES].astype(float)

    dem_block = modelo.iloc[108:207, [0, 13]].copy()
    dem_block.columns = ["Producto", "Demanda"]
    dem_block["Producto"] = dem_block["Producto"].astype(str).str.strip()
    dem_block = dem_block.dropna(subset=["Demanda"])
    dem_block = dem_block[dem_block["Producto"] != "PRODUCTO"]
    demanda = dem_block.set_index("Producto")["Demanda"].astype(float)

    productos_validos = cij.index.intersection(demanda.index)
    cij = cij.loc[productos_validos]
    demanda = demanda.loc[productos_validos]

    return {"cij": cij, "demanda": demanda, "setup": setup}


def construir_cij_con_imputacion(path_excel: str,
                                  productos: Iterable[str] | None = None) -> dict:
    """
    Reconstruye C_ij desde las hojas crudas Datos M1..Datos M10.

    Para cada combinación observada se promedian registros positivos y se
    convierten de segundos a minutos. Las celdas sin registro se imputan con
    la media de la máquina. Se devuelve la máscara observada para auditoría.
    """
    registros_por_maquina = {}
    for j in MACHINES:
        try:
            d = pd.read_excel(path_excel, sheet_name=f"Datos {j}", header=1)
        except Exception:
            continue
        d = d[d.iloc[:, 0].notna()]
        productos_j = d.iloc[:, 0].astype(str).str.strip()
        registros = d.iloc[:, 1:31].apply(pd.to_numeric, errors="coerce")
        reg_validos = registros.where(registros > 0)
        promedio_min = reg_validos.mean(axis=1) / 60.0
        registros_por_maquina[j] = pd.Series(promedio_min.values, index=productos_j.values)

    C_obs = pd.DataFrame(registros_por_maquina).reindex(columns=MACHINES)
    if productos is not None:
        productos = [p for p in productos if p in C_obs.index]
        C_obs = C_obs.loc[productos]

    mask_observado = C_obs.notna()
    total_celdas = int(C_obs.size)
    n_observadas = int(mask_observado.sum().sum())
    media_maquina = C_obs.mean(axis=0, skipna=True)
    C_imputada = C_obs.fillna(media_maquina)

    return {
        "cij": C_imputada,
        "cij_observada": C_obs,
        "R_C": n_observadas / total_celdas if total_celdas else float("nan"),
        "n_observadas": n_observadas,
        "n_imputadas": total_celdas - n_observadas,
        "total_celdas": total_celdas,
        "mask_observado": mask_observado,
        "media_maquina": media_maquina,
    }


def comparar_matriz_publicada_reconstruida(cij_publicada: pd.DataFrame,
                                            audit: dict) -> dict:
    """Cuantifica diferencias entre MODELO PL y la reconstrucción desde registros crudos."""
    common = cij_publicada.index.intersection(audit["cij"].index)
    a = cij_publicada.loc[common, MACHINES].astype(float)
    b = audit["cij"].loc[common, MACHINES].astype(float)
    mask = audit["mask_observado"].loc[common, MACHINES]
    diff = (a - b).abs()
    return {
        "n_referencias": len(common),
        "mae_total": float(diff.stack().mean()),
        "mae_observadas": float(diff.where(mask).stack().mean()),
        "mae_imputadas": float(diff.where(~mask).stack().mean()),
        "max_diff": float(diff.stack().max()),
        "n_celdas_distintas_1e6": int((diff > 1e-6).sum().sum()),
    }


def validacion_imputacion_repetida(path_excel: str,
                                    productos: Iterable[str],
                                    n_repeticiones: int = 100,
                                    fraccion_oculta: float = 0.20,
                                    seed: int = 42) -> dict:
    """
    Revalidación adicional de imputación por ocultamiento repetido.

    No reemplaza la tabla histórica de la tesis. Añade una comprobación más
    estable: se repite el ocultamiento aleatorio del 20% de celdas observadas
    y se comparan cuatro predictores. Reporta MAE/RMSE medios y frecuencia con
    la que cada método obtiene el menor MAE.
    """
    audit = construir_cij_con_imputacion(path_excel, productos=productos)
    C = audit["cij_observada"].copy()
    obs = np.argwhere(C.notna().to_numpy())
    n_hide = max(1, int(round(fraccion_oculta * len(obs))))
    registros = []

    for rep in range(int(n_repeticiones)):
        rng = np.random.default_rng(seed + rep)
        sel = obs[rng.choice(len(obs), n_hide, replace=False)]
        train = C.copy()
        y = []
        for r, c in sel:
            y.append(C.iat[r, c])
            train.iat[r, c] = np.nan

        global_mean = float(train.stack().mean())
        row_mean = train.mean(axis=1)
        col_mean = train.mean(axis=0)
        yy = np.asarray(y, dtype=float)

        metodos = {
            "Media global": lambda r, c: global_mean,
            "Media por referencia": lambda r, c: row_mean.iloc[r],
            "Media por máquina": lambda r, c: col_mean.iloc[c],
            "Aditivo referencia × máquina": lambda r, c: row_mean.iloc[r] + col_mean.iloc[c] - global_mean,
        }
        for nombre, predictor in metodos.items():
            pred = np.asarray([predictor(r, c) for r, c in sel], dtype=float)
            ok = np.isfinite(pred) & np.isfinite(yy)
            err = yy[ok] - pred[ok]
            registros.append({
                "repeticion": rep + 1,
                "metodo": nombre,
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "n_validacion": int(ok.sum()),
            })

    detalle = pd.DataFrame(registros)
    resumen = detalle.groupby("metodo").agg(
        mae_media=("mae", "mean"),
        mae_sd=("mae", "std"),
        rmse_media=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
    ).reset_index()
    pivot = detalle.pivot(index="repeticion", columns="metodo", values="mae")
    ganadores = pivot.idxmin(axis=1).value_counts()
    resumen["veces_mejor_mae"] = resumen["metodo"].map(ganadores).fillna(0).astype(int)
    resumen["frecuencia_mejor_pct"] = resumen["veces_mejor_mae"] / n_repeticiones * 100

    # IC 95% aproximado para la media del MAE entre repeticiones.
    resumen["mae_ci95_low"] = np.nan
    resumen["mae_ci95_high"] = np.nan
    for idx, row in resumen.iterrows():
        g = detalle.loc[detalle["metodo"] == row["metodo"], "mae"]
        if len(g) > 1:
            se = g.std(ddof=1) / math.sqrt(len(g))
            q = stats.t.ppf(0.975, df=len(g) - 1)
            resumen.loc[idx, "mae_ci95_low"] = g.mean() - q * se
            resumen.loc[idx, "mae_ci95_high"] = g.mean() + q * se

    return {"resumen": resumen.sort_values("mae_media"), "detalle": detalle, "audit": audit}


def cargar_datos_extendido(path_excel: str, dias_habiles: int = 180) -> dict:
    """Carga las 200 referencias y estima demanda diaria desde ventas semestrales."""
    xls = pd.ExcelFile(path_excel)
    setup = _cargar_setup(xls)

    sim = pd.read_excel(xls, "DATOS SIMULADOS", header=1)
    sim = sim.rename(columns={sim.columns[0]: "Producto"})
    sim["Producto"] = sim["Producto"].astype(str).str.strip()
    sim = sim.dropna(subset=["Producto"])
    sim = sim[sim["Producto"] != "nan"]
    cij = sim.set_index("Producto")[MACHINES].astype(float) / 60.0

    ventas = pd.read_excel(xls, "Unidades Vendidas (SEMESTRE)", header=0)
    ventas.columns = [str(c).strip() for c in ventas.columns]
    ventas["PRODUCTO"] = ventas["PRODUCTO"].astype(str).str.strip()
    ventas = ventas.dropna(subset=["PRODUCTO"])
    demanda = ventas.set_index("PRODUCTO")["UNIDADES"].astype(float) / float(dias_habiles)

    productos_validos = cij.index.intersection(demanda.index)
    return {
        "cij": cij.loc[productos_validos],
        "demanda": demanda.loc[productos_validos],
        "setup": setup,
        "dias_habiles": dias_habiles,
    }


# ---------------------------------------------------------------------------
# 2. Clustering y validación
# ---------------------------------------------------------------------------
def clusterizar_productos(cij: pd.DataFrame, demanda: pd.Series,
                          n_clusters: int = 3, seed: int = 42) -> pd.DataFrame:
    cv = cij.std(axis=1) / cij.mean(axis=1)
    feats = pd.DataFrame({"demanda": demanda, "cv_tiempo": cv})
    X = StandardScaler().fit_transform(feats.values)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    feats["cluster"] = km.fit_predict(X)

    orden = feats.groupby("cluster")["demanda"].mean().sort_values(ascending=False).index
    etiquetas = {orden[0]: "Producción continua (alto volumen)"}
    if n_clusters >= 2:
        etiquetas[orden[-1]] = "Comodín / baja escala"
    for c in orden[1:-1] if n_clusters > 2 else []:
        etiquetas[c] = "Rotación media"
    feats["rol_sugerido"] = feats["cluster"].map(etiquetas)
    return feats


def elbow_silhouette(features: pd.DataFrame, k_range=range(2, 7), seed: int = 42) -> pd.DataFrame:
    X = StandardScaler().fit_transform(features.values)
    filas = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        filas.append({"k": k, "inercia": km.inertia_, "silhouette": silhouette_score(X, labels)})
    return pd.DataFrame(filas)


def prueba_estadistica_clusters(valores: pd.Series, grupos: pd.Series) -> dict:
    """Kruskal-Wallis con epsilon-cuadrado como tamaño de efecto descriptivo."""
    muestras = [valores[grupos == g].dropna().values for g in sorted(grupos.unique())]
    muestras = [m for m in muestras if len(m) > 0]
    h, p = stats.kruskal(*muestras)
    n = sum(len(m) for m in muestras)
    k = len(muestras)
    eps2 = max(0.0, (h - k + 1) / (n - k)) if n > k else float("nan")
    return {
        "estadistico_H": float(h),
        "valor_p": float(p),
        "epsilon2": float(eps2),
        "significativo_al_5pct": bool(p < 0.05),
    }


def _features_maquinas(cij: pd.DataFrame, setup: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "setup_promedio": setup.reindex(MACHINES),
        "tiempo_operacion_promedio": cij[MACHINES].mean(axis=0),
        "varianza_tiempo": cij[MACHINES].var(axis=0),
    })


def clusterizar_maquinas(cij: pd.DataFrame, setup: pd.Series,
                         n_clusters: int = 3, seed: int = 42) -> pd.DataFrame:
    feats = _features_maquinas(cij, setup)
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


def concordancia_kmeans_ward_maquinas(cij: pd.DataFrame, setup: pd.Series,
                                       n_clusters: int = 3, seed: int = 42) -> dict:
    """Compara K-Means con Ward sobre exactamente las mismas variables estandarizadas."""
    feats = _features_maquinas(cij, setup)
    X = StandardScaler().fit_transform(feats.values)
    labels_km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(X)
    labels_ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X)
    ari = adjusted_rand_score(labels_km, labels_ward)
    sil_km = silhouette_score(X, labels_km)
    sil_ward = silhouette_score(X, labels_ward)
    out = feats.copy()
    out["cluster_kmeans"] = labels_km
    out["cluster_ward"] = labels_ward
    return {
        "ari": float(ari),
        "silhouette_kmeans": float(sil_km),
        "silhouette_ward": float(sil_ward),
        "tabla": out,
    }


# ---------------------------------------------------------------------------
# 3. Solver y MILP diario
# ---------------------------------------------------------------------------
def _parsear_log_cbc(path: str) -> dict:
    """Extrae cotas, GAP y motivo de terminación del log de CBC."""
    out = {
        "lower_bound": None, "upper_bound": None, "gap_pct": None,
        "stopped_on_time": False, "proven_optimal": False,
        "termination_known": False,
    }
    if not path or not os.path.exists(path):
        return out
    try:
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return out

    low = text.lower()
    out["stopped_on_time"] = (
        "stopped on time limit" in low
        or "result - stopped on time limit" in low
    )
    out["proven_optimal"] = (
        "result - optimal solution found" in low
        or "optimal solution found" in low
        or "search completed - best objective" in low
    ) and not out["stopped_on_time"]
    out["termination_known"] = out["stopped_on_time"] or out["proven_optimal"]

    pats = {
        "upper_bound": [r"Objective value:\s*([-+0-9.eE]+)", r"Upper bound:\s*([-+0-9.eE]+)"],
        "lower_bound": [r"Lower bound:\s*([-+0-9.eE]+)"],
        "gap_raw": [r"Gap:\s*([-+0-9.eE]+)"],
    }
    vals = {}
    for key, patterns in pats.items():
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                try:
                    vals[key] = float(m.group(1))
                    break
                except Exception:
                    pass
    out.update({k: vals.get(k) for k in ["lower_bound", "upper_bound"]})
    if out["lower_bound"] is not None and out["upper_bound"] is not None and abs(out["upper_bound"]) > TOL:
        out["gap_pct"] = abs(out["upper_bound"] - out["lower_bound"]) / abs(out["upper_bound"]) * 100
    elif "gap_raw" in vals:
        out["gap_pct"] = abs(vals["gap_raw"]) * 100
    return out

def _resolver_prob_cbc(prob: pulp.LpProblem, tiempo_limite_seg: int, gap_rel: float | None = None):
    log_path = tempfile.NamedTemporaryFile(prefix="rpm_cbc_", suffix=".log", delete=False).name
    try:
        try:
            solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=tiempo_limite_seg, gapRel=gap_rel, logPath=log_path)
        except TypeError:
            solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=tiempo_limite_seg, gapRel=gap_rel)
            log_path = None
        prob.solve(solver)
        info = _parsear_log_cbc(log_path) if log_path else {
            "lower_bound": None, "upper_bound": None, "gap_pct": None,
            "stopped_on_time": False, "proven_optimal": False, "termination_known": False,
        }
    finally:
        if log_path and os.path.exists(log_path):
            try:
                os.remove(log_path)
            except OSError:
                pass
    return info


def resolver_asignacion(cij: pd.DataFrame, demanda: pd.Series,
                         setup: pd.Series,
                         cap_minutos=CAP_MINUTOS_DIA,
                         tiempo_limite_seg: int = 60) -> dict:
    """MILP diario con X_ij entera, Y_ij binaria y capacidad específica por máquina."""
    productos = list(cij.index)
    demanda_i = _validar_demanda_entera(demanda.reindex(productos))
    setup = setup.reindex(MACHINES).astype(float)
    caps = _serie_capacidades(cap_minutos)
    C = cij.reindex(index=productos, columns=MACHINES).astype(float)
    if C.isna().any().any() or (C <= 0).any().any():
        raise ValueError("C_ij debe estar completa y contener tiempos positivos.")

    prob = pulp.LpProblem("Asignacion_RPM", pulp.LpMinimize)
    X = pulp.LpVariable.dicts("X", (productos, MACHINES), lowBound=0, cat="Integer")
    Y = pulp.LpVariable.dicts("Y", (productos, MACHINES), cat="Binary")

    prob += (
        pulp.lpSum(C.loc[i, j] * X[i][j] for i in productos for j in MACHINES)
        + pulp.lpSum(setup[j] * Y[i][j] for i in productos for j in MACHINES)
    )

    U = pd.DataFrame(0, index=productos, columns=MACHINES, dtype=int)
    for i in productos:
        prob += pulp.lpSum(X[i][j] for j in MACHINES) == int(demanda_i[i]), f"Demanda_{i}"
    for j in MACHINES:
        prob += (
            pulp.lpSum(C.loc[i, j] * X[i][j] for i in productos)
            + pulp.lpSum(setup[j] * Y[i][j] for i in productos)
            <= float(caps[j])
        ), f"Capacidad_{j}"
    for i in productos:
        for j in MACHINES:
            cij_val = float(C.loc[i, j])
            max_por_cap = math.floor((float(caps[j]) - float(setup[j])) / cij_val) if cij_val > TOL else int(demanda_i[i])
            u_ij = max(0, min(int(demanda_i[i]), max_por_cap))
            U.loc[i, j] = u_ij
            prob += X[i][j] <= u_ij * Y[i][j], f"Link_{i}_{j}"

    log_info = _resolver_prob_cbc(prob, tiempo_limite_seg)
    estado = pulp.LpStatus[prob.status]
    objetivo = pulp.value(prob.objective)
    gap_certificado = (
        bool(log_info.get("proven_optimal"))
        if log_info.get("termination_known")
        else estado == "Optimal"
    )

    asignacion = pd.DataFrame(0, index=productos, columns=MACHINES, dtype=int)
    setups = pd.DataFrame(0, index=productos, columns=MACHINES, dtype=int)
    for i in productos:
        for j in MACHINES:
            xv = X[i][j].value()
            yv = Y[i][j].value()
            asignacion.loc[i, j] = int(round(xv or 0.0))
            setups.loc[i, j] = int(round(yv or 0.0))

    tiempo_prod = (C * asignacion).sum(axis=0)
    tiempo_setup = pd.Series({j: float(setup[j] * setups[j].sum()) for j in MACHINES})
    utilizacion = tiempo_prod + tiempo_setup
    holgura = caps - utilizacion
    n_setups = setups.sum(axis=0).astype(int)

    # Residuos de factibilidad, útiles para auditoría y sustentación.
    r_dem = (asignacion.sum(axis=1) - demanda_i).abs()
    r_cap = (utilizacion - caps).clip(lower=0)
    r_link = (asignacion - U * setups).clip(lower=0)
    r_int = (asignacion.astype(float) - np.rint(asignacion.astype(float))).abs()
    residuos = {
        "demanda_max": float(r_dem.max()) if len(r_dem) else 0.0,
        "capacidad_max": float(r_cap.max()) if len(r_cap) else 0.0,
        "enlace_max": float(r_link.to_numpy().max()) if r_link.size else 0.0,
        "integralidad_max": float(r_int.to_numpy().max()) if r_int.size else 0.0,
    }

    if gap_certificado:
        gap_pct = 0.0
        lower_bound = float(objetivo) if objetivo is not None else None
        upper_bound = float(objetivo) if objetivo is not None else None
    else:
        gap_pct = log_info.get("gap_pct")
        lower_bound = log_info.get("lower_bound")
        upper_bound = log_info.get("upper_bound") or (float(objetivo) if objetivo is not None else None)

    refs_divididas = int((asignacion.gt(0).sum(axis=1) > 1).sum())
    pares_activos = int(asignacion.gt(0).sum().sum())

    return {
        "estado": estado,
        "gap_certificado": gap_certificado,
        "gap_pct": gap_pct,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "solver_termination": {
            "stopped_on_time": bool(log_info.get("stopped_on_time", False)),
            "proven_optimal": bool(log_info.get("proven_optimal", False)),
        },
        "valor_objetivo": float(objetivo) if objetivo is not None else None,
        "asignacion": asignacion,
        "setups": setups,
        "U": U,
        "utilizacion_min": utilizacion,
        "tiempo_produccion_min": tiempo_prod,
        "tiempo_setup_min": tiempo_setup,
        "holgura_min": holgura,
        "n_setups": n_setups,
        "capacidad_min": caps,
        "cap_minutos": float(caps.iloc[0]) if caps.nunique() == 1 else caps,
        "residuos": residuos,
        "pares_activos": pares_activos,
        "referencias_divididas": refs_divididas,
    }


def soporte_observado_solucion(asignacion: pd.DataFrame, mask_observado: pd.DataFrame) -> dict:
    common_i = asignacion.index.intersection(mask_observado.index)
    common_j = asignacion.columns.intersection(mask_observado.columns)
    A = asignacion.loc[common_i, common_j]
    M = mask_observado.loc[common_i, common_j]
    activos = A > 0
    activos_total = int(activos.sum().sum())
    activos_imp = int((activos & ~M).sum().sum())
    unidades_total = float(A.sum().sum())
    unidades_imp = float(A.where(~M, 0).sum().sum())
    return {
        "pares_activos": activos_total,
        "pares_imputados": activos_imp,
        "pares_observados": activos_total - activos_imp,
        "share_pares_observados_pct": 100 * (activos_total - activos_imp) / activos_total if activos_total else np.nan,
        "unidades_totales": unidades_total,
        "unidades_en_celdas_imputadas": unidades_imp,
        "share_unidades_observadas_pct": 100 * (unidades_total - unidades_imp) / unidades_total if unidades_total else np.nan,
    }


# ---------------------------------------------------------------------------
# 4. Dual condicional y política de referencia
# ---------------------------------------------------------------------------
def calcular_precios_sombra(cij: pd.DataFrame, demanda: pd.Series,
                             setup: pd.Series, setups_fijos: pd.DataFrame,
                             cap_minutos=CAP_MINUTOS_DIA) -> pd.Series:
    productos = list(cij.index)
    demanda_i = _validar_demanda_entera(demanda.reindex(productos))
    caps = _serie_capacidades(cap_minutos)
    prob = pulp.LpProblem("Precios_Sombra", pulp.LpMinimize)
    X = pulp.LpVariable.dicts("X", (productos, MACHINES), lowBound=0)
    prob += pulp.lpSum(cij.loc[i, j] * X[i][j] for i in productos for j in MACHINES)
    for i in productos:
        prob += pulp.lpSum(X[i][j] for j in MACHINES) == int(demanda_i[i]), f"Demanda_{i}"
    for j in MACHINES:
        tiempo_setup_fijo = sum(setup[j] * setups_fijos.loc[i, j] for i in productos)
        prob += pulp.lpSum(cij.loc[i, j] * X[i][j] for i in productos) <= caps[j] - tiempo_setup_fijo, f"Capacidad_{j}"
    for i in productos:
        for j in MACHINES:
            if setups_fijos.loc[i, j] == 0:
                prob += X[i][j] == 0, f"Fijo_{i}_{j}"
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return pd.Series({j: prob.constraints[f"Capacidad_{j}"].pi or 0.0 for j in MACHINES}, name="precio_sombra_min")


def politica_empirica(cij: pd.DataFrame, demanda: pd.Series, setup: pd.Series,
                       cap_minutos=CAP_MINUTOS_DIA) -> dict:
    """Heurística greedy íntegra: asigna unidades completas a máquinas rápidas con capacidad."""
    demanda_i = _validar_demanda_entera(demanda.reindex(cij.index))
    caps = _serie_capacidades(cap_minutos)
    productos_orden = demanda_i.sort_values(ascending=False).index.tolist()
    capacidad_restante = caps.copy()
    setup_pagado = pd.DataFrame(False, index=cij.index, columns=MACHINES)
    asignacion = pd.DataFrame(0, index=cij.index, columns=MACHINES, dtype=int)
    pendientes = demanda_i.copy()

    for i in productos_orden:
        for j in cij.loc[i, MACHINES].sort_values().index.tolist():
            if pendientes[i] <= 0:
                break
            setup_necesario = 0.0 if setup_pagado.loc[i, j] else float(setup[j])
            cap_disp = float(capacidad_restante[j]) - setup_necesario
            if cap_disp <= 0:
                continue
            unidades_posibles = int(math.floor(cap_disp / float(cij.loc[i, j]) + 1e-10))
            unidades_asignar = min(unidades_posibles, int(pendientes[i]))
            if unidades_asignar <= 0:
                continue
            capacidad_restante[j] -= unidades_asignar * float(cij.loc[i, j]) + setup_necesario
            asignacion.loc[i, j] += unidades_asignar
            pendientes[i] -= unidades_asignar
            setup_pagado.loc[i, j] = True

    demanda_incumplida = pendientes[pendientes > 0]
    utilizacion = caps - capacidad_restante
    n_setups = setup_pagado.sum(axis=0).astype(int)
    return {
        "asignacion": asignacion,
        "utilizacion_min": utilizacion,
        "valor_objetivo": float(utilizacion.sum()),
        "demanda_incumplida": demanda_incumplida,
        "n_setups": n_setups,
        "capacidad_min": caps,
        "cap_minutos": float(caps.iloc[0]) if caps.nunique() == 1 else caps,
    }


# ---------------------------------------------------------------------------
# 5. Sensibilidad, cuello de botella y simulación
# ---------------------------------------------------------------------------
def analisis_cuello_de_botella(cij: pd.DataFrame, demanda: pd.Series,
                                setup: pd.Series, cap_minutos=CAP_MINUTOS_DIA,
                                delta: float = 30.0, tiempo_limite_seg: int = 60) -> pd.DataFrame:
    """Reoptimiza exactamente el mismo MILP con +delta minutos en una máquina."""
    caps = _serie_capacidades(cap_minutos)
    base = resolver_asignacion(cij, demanda, setup, caps, tiempo_limite_seg)
    filas = []
    for j in MACHINES:
        caps_j = caps.copy()
        caps_j[j] += float(delta)
        res = resolver_asignacion(cij, demanda, setup, caps_j, tiempo_limite_seg)
        mejora = None
        eta = None
        if base["valor_objetivo"] is not None and res["valor_objetivo"] is not None:
            mejora = base["valor_objetivo"] - res["valor_objetivo"]
            if abs(mejora) < 1e-8:
                mejora = 0.0
            eta = mejora / float(delta)
        filas.append({
            "maquina": j,
            "estado": res["estado"],
            "gap_pct": res["gap_pct"],
            "utilizacion_actual_pct": 100 * base["utilizacion_min"][j] / caps[j],
            "holgura_actual_min": base["holgura_min"][j],
            "delta_capacidad_min": float(delta),
            "mejora_objetivo_min": mejora,
            "eta_min_por_min": eta,
        })
    return pd.DataFrame(filas).sort_values(["mejora_objetivo_min", "utilizacion_actual_pct"], ascending=[False, False]).reset_index(drop=True)


def analisis_sensibilidad(cij: pd.DataFrame, demanda: pd.Series, setup: pd.Series,
                           cap_minutos=CAP_MINUTOS_DIA,
                           factores_demanda=(0.8, 0.9, 1.0, 1.1, 1.2),
                           tiempo_limite_seg: int = 60,
                           metodo_redondeo: str = "round") -> pd.DataFrame:
    filas = []
    for f in factores_demanda:
        teorica = float((pd.Series(demanda, dtype=float) * f).sum())
        dem_f = demanda_entera(demanda, f, metodo_redondeo)
        res = resolver_asignacion(cij, dem_f, setup, cap_minutos, tiempo_limite_seg)
        caps = res["capacidad_min"]
        util_pct = res["utilizacion_min"] / caps * 100
        filas.append({
            "factor_demanda": float(f),
            "demanda_teorica_total": teorica,
            "demanda_entera_total": int(dem_f.sum()),
            "ajuste_redondeo_unidades": float(dem_f.sum() - teorica),
            "estado": res["estado"],
            "gap_pct": res["gap_pct"],
            "valor_objetivo_min": res["valor_objetivo"],
            "utilizacion_promedio_pct": float(util_pct.mean()),
            "utilizacion_maxima_pct": float(util_pct.max()),
            "maquina_mas_cargada": util_pct.idxmax(),
            "total_setups": int(res["n_setups"].sum()),
        })
    return pd.DataFrame(filas)


def analisis_sensibilidad_imputacion(cij: pd.DataFrame, mask_observado: pd.DataFrame,
                                      demanda: pd.Series, setup: pd.Series,
                                      factores=(0.8, 1.0, 1.2),
                                      cap_minutos=CAP_MINUTOS_DIA,
                                      tiempo_limite_seg: int = 60) -> pd.DataFrame:
    """Perturba solo las celdas identificadas como imputadas y reoptimiza."""
    mask = mask_observado.reindex(index=cij.index, columns=cij.columns).fillna(False)
    filas = []
    for f in factores:
        C = cij.copy().astype(float)
        C = C.where(mask, C * float(f))
        res = resolver_asignacion(C, demanda, setup, cap_minutos, tiempo_limite_seg)
        filas.append({
            "factor_imputadas": float(f),
            "variacion_imputadas_pct": (float(f) - 1.0) * 100,
            "estado": res["estado"],
            "gap_pct": res["gap_pct"],
            "valor_objetivo_min": res["valor_objetivo"],
        })
    df = pd.DataFrame(filas)
    base = df.loc[np.isclose(df["factor_imputadas"], 1.0), "valor_objetivo_min"]
    if len(base):
        z0 = float(base.iloc[0])
        df["variacion_objetivo_pct"] = (df["valor_objetivo_min"] / z0 - 1.0) * 100
    return df


def comparar_politicas_multiescenario(cij: pd.DataFrame, demanda: pd.Series, setup: pd.Series,
                                       cap_minutos=CAP_MINUTOS_DIA,
                                       factores_demanda=(0.8, 0.9, 1.0, 1.1, 1.2),
                                       tiempo_limite_seg: int = 60,
                                       metodo_redondeo: str = "round") -> pd.DataFrame:
    filas = []
    for f in factores_demanda:
        dem_f = demanda_entera(demanda, f, metodo_redondeo)
        res_opt = resolver_asignacion(cij, dem_f, setup, cap_minutos, tiempo_limite_seg)
        res_emp = politica_empirica(cij, dem_f, setup, cap_minutos)
        mejora = np.nan
        if res_opt["valor_objetivo"] is not None and res_emp["valor_objetivo"] > 0 and len(res_emp["demanda_incumplida"]) == 0:
            mejora = (res_emp["valor_objetivo"] - res_opt["valor_objetivo"]) / res_emp["valor_objetivo"] * 100
        filas.append({
            "factor_demanda": float(f),
            "demanda_entera_total": int(dem_f.sum()),
            "estado_optimo": res_opt["estado"],
            "gap_pct": res_opt["gap_pct"],
            "tiempo_optimo_min": res_opt["valor_objetivo"],
            "tiempo_empirico_min": res_emp["valor_objetivo"],
            "mejora_pct": mejora,
            "unidades_incumplidas_empirica": int(res_emp["demanda_incumplida"].sum()) if len(res_emp["demanda_incumplida"]) else 0,
        })
    return pd.DataFrame(filas)


def monte_carlo_demanda(cij: pd.DataFrame, demanda: pd.Series, setup: pd.Series,
                         n_escenarios: int = 30, seed: int = 42,
                         low: float = 0.85, high: float = 1.15,
                         cap_minutos=CAP_MINUTOS_DIA,
                         tiempo_limite_seg: int = 60) -> dict:
    """
    Perturba cada referencia independientemente con U(low, high), redondea a
    unidades enteras y compara MILP contra la política greedy integral.
    """
    rng = np.random.default_rng(seed)
    base = pd.Series(demanda, dtype=float)
    filas = []
    for s in range(1, int(n_escenarios) + 1):
        factores = rng.uniform(low, high, size=len(base))
        valores = np.floor(base.to_numpy() * factores + 0.5).astype(int)
        valores[(base.to_numpy() > 0) & (valores < 1)] = 1
        dem_s = pd.Series(valores, index=base.index, dtype=int)
        opt = resolver_asignacion(cij, dem_s, setup, cap_minutos, tiempo_limite_seg)
        emp = politica_empirica(cij, dem_s, setup, cap_minutos)
        mejora = np.nan
        mejora_certificada = np.nan
        empirica_factible = len(emp["demanda_incumplida"]) == 0
        if opt["valor_objetivo"] is not None and emp["valor_objetivo"] > 0 and empirica_factible:
            mejora = (emp["valor_objetivo"] - opt["valor_objetivo"]) / emp["valor_objetivo"] * 100
            if opt["gap_certificado"]:
                mejora_certificada = mejora
        filas.append({
            "escenario": s,
            "demanda_total": int(dem_s.sum()),
            "estado": opt["estado"],
            "gap_pct": opt["gap_pct"],
            "optimo_certificado": bool(opt["gap_certificado"]),
            "tiempo_optimo_min": opt["valor_objetivo"],
            "tiempo_empirico_min": emp["valor_objetivo"],
            "mejora_pct": mejora,
            "mejora_certificada_pct": mejora_certificada,
            "empirica_factible": empirica_factible,
        })
    escenarios = pd.DataFrame(filas)
    # El resumen inferencial usa únicamente escenarios cuyo MILP cerró con
    # optimalidad certificada. Los incumbentes no certificados se conservan
    # en la tabla para diagnóstico, pero no se mezclan con los óptimos.
    r = escenarios["mejora_certificada_pct"].dropna()
    if len(r) > 1:
        sd = float(r.std(ddof=1))
        se = sd / math.sqrt(len(r))
        q = stats.t.ppf(0.975, df=len(r) - 1)
        ci = (float(r.mean() - q * se), float(r.mean() + q * se))
    elif len(r) == 1:
        sd, se, ci = 0.0, 0.0, (float(r.iloc[0]), float(r.iloc[0]))
    else:
        sd, se, ci = np.nan, np.nan, (np.nan, np.nan)
    resumen = {
        "n_solicitados": int(n_escenarios),
        "n_validos": int(len(r)),
        "n_optimos_certificados": int(escenarios["optimo_certificado"].sum()),
        "n_no_certificados": int((~escenarios["optimo_certificado"]).sum()),
        "n_empirica_infactible": int((~escenarios["empirica_factible"]).sum()),
        "seed": int(seed),
        "uniform_low": float(low),
        "uniform_high": float(high),
        "media_pct": float(r.mean()) if len(r) else np.nan,
        "mediana_pct": float(r.median()) if len(r) else np.nan,
        "sd_pct": sd,
        "se_pct": se,
        "p5_pct": float(r.quantile(0.05)) if len(r) else np.nan,
        "p95_pct": float(r.quantile(0.95)) if len(r) else np.nan,
        "min_pct": float(r.min()) if len(r) else np.nan,
        "max_pct": float(r.max()) if len(r) else np.nan,
        "ci95_low_pct": ci[0],
        "ci95_high_pct": ci[1],
        "proporcion_mejora_positiva": float((r > 0).mean()) if len(r) else np.nan,
    }
    return {"escenarios": escenarios, "resumen": resumen}


# ---------------------------------------------------------------------------
# 6. Modelo multiperiodo
# ---------------------------------------------------------------------------
def resolver_multidia(cij: pd.DataFrame, demanda_horizonte: pd.Series, setup: pd.Series,
                       dias: int, cap_minutos=CAP_MINUTOS_DIA,
                       tiempo_limite_seg: int = 120,
                       gap_rel: float | None = 0.005) -> dict:
    """Planificación multiperiodo determinista con unidades enteras."""
    productos = list(cij.index)
    demanda_i = _validar_demanda_entera(demanda_horizonte.reindex(productos), "demanda_horizonte")
    caps = _serie_capacidades(cap_minutos)
    dias_lista = list(range(1, int(dias) + 1))
    C = cij.reindex(index=productos, columns=MACHINES).astype(float)
    setup = setup.reindex(MACHINES).astype(float)

    prob = pulp.LpProblem("Programacion_MultiPeriodo", pulp.LpMinimize)
    X = pulp.LpVariable.dicts("X", (productos, MACHINES, dias_lista), lowBound=0, cat="Integer")
    Y = pulp.LpVariable.dicts("Y", (productos, MACHINES, dias_lista), cat="Binary")
    prob += (
        pulp.lpSum(C.loc[i, j] * X[i][j][t] for i in productos for j in MACHINES for t in dias_lista)
        + pulp.lpSum(setup[j] * Y[i][j][t] for i in productos for j in MACHINES for t in dias_lista)
    )
    for i in productos:
        prob += pulp.lpSum(X[i][j][t] for j in MACHINES for t in dias_lista) == int(demanda_i[i]), f"Demanda_{i}"
    for j in MACHINES:
        for t in dias_lista:
            prob += (
                pulp.lpSum(C.loc[i, j] * X[i][j][t] for i in productos)
                + pulp.lpSum(setup[j] * Y[i][j][t] for i in productos)
                <= caps[j]
            ), f"Capacidad_{j}_{t}"
    for i in productos:
        for j in MACHINES:
            u_ij = max(0, min(int(demanda_i[i]), math.floor((caps[j] - setup[j]) / C.loc[i, j])))
            for t in dias_lista:
                prob += X[i][j][t] <= u_ij * Y[i][j][t], f"Link_{i}_{j}_{t}"

    log_info = _resolver_prob_cbc(prob, tiempo_limite_seg, gap_rel=gap_rel)
    estado = pulp.LpStatus[prob.status]
    objetivo = pulp.value(prob.objective)
    gap_certificado = (
        bool(log_info.get("proven_optimal"))
        if log_info.get("termination_known")
        else estado == "Optimal" and (gap_rel is None or gap_rel == 0)
    )
    gap_pct = 0.0 if gap_certificado else log_info.get("gap_pct")

    filas = []
    util = pd.DataFrame(0.0, index=dias_lista, columns=MACHINES)
    prod_t = pd.DataFrame(0.0, index=dias_lista, columns=MACHINES)
    setup_t = pd.DataFrame(0.0, index=dias_lista, columns=MACHINES)
    setups_dm = pd.DataFrame(0, index=dias_lista, columns=MACHINES)
    for t in dias_lista:
        for j in MACHINES:
            n_set = 0
            tiempo_prod = 0.0
            for i in productos:
                cantidad = int(round(X[i][j][t].value() or 0.0))
                yv = int(round(Y[i][j][t].value() or 0.0))
                n_set += yv
                tiempo_prod += C.loc[i, j] * cantidad
                if cantidad > 0:
                    filas.append({"dia": t, "maquina": j, "producto": i, "unidades": cantidad})
            prod_t.loc[t, j] = tiempo_prod
            setup_t.loc[t, j] = n_set * setup[j]
            util.loc[t, j] = prod_t.loc[t, j] + setup_t.loc[t, j]
            setups_dm.loc[t, j] = n_set
    plan = pd.DataFrame(filas)
    holgura = pd.DataFrame(np.tile(caps.to_numpy(), (len(dias_lista), 1)), index=dias_lista, columns=MACHINES) - util

    # Residuos acumulados de demanda.
    producido = plan.groupby("producto")["unidades"].sum().reindex(productos, fill_value=0) if len(plan) else pd.Series(0, index=productos)
    r_dem = (producido - demanda_i).abs()
    r_cap = (-holgura).clip(lower=0)

    if gap_certificado:
        lower_bound = upper_bound = float(objetivo) if objetivo is not None else None
    else:
        lower_bound = log_info.get("lower_bound")
        upper_bound = log_info.get("upper_bound") or (float(objetivo) if objetivo is not None else None)

    return {
        "estado": estado,
        "gap_certificado": gap_certificado,
        "gap_pct": gap_pct,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "solver_termination": {
            "stopped_on_time": bool(log_info.get("stopped_on_time", False)),
            "proven_optimal": bool(log_info.get("proven_optimal", False)),
        },
        "valor_objetivo": float(objetivo) if objetivo is not None else None,
        "plan": plan,
        "utilizacion_dia_maquina": util,
        "tiempo_produccion_dia_maquina": prod_t,
        "tiempo_setup_dia_maquina": setup_t,
        "holgura_dia_maquina": holgura,
        "setups_dia_maquina": setups_dm,
        "dias": int(dias),
        "capacidad_min": caps,
        "cap_minutos": float(caps.iloc[0]) if caps.nunique() == 1 else caps,
        "residuos": {
            "demanda_max": float(r_dem.max()) if len(r_dem) else 0.0,
            "capacidad_max": float(r_cap.to_numpy().max()) if r_cap.size else 0.0,
        },
    }


def calibrar_demanda_extendida(demanda_medida: pd.Series, demanda_estimada: pd.Series) -> dict:
    comunes = demanda_medida.index.intersection(demanda_estimada.index)
    if len(comunes) == 0:
        raise ValueError("No hay referencias comunes para calibrar la demanda extendida.")
    den = float(demanda_medida.loc[comunes].sum())
    if den <= 0:
        raise ValueError("La demanda medida común debe sumar un valor positivo.")
    factor_sesgo = float(demanda_estimada.loc[comunes].sum() / den)
    demanda_calibrada = demanda_estimada / factor_sesgo
    return {
        "factor_sesgo": factor_sesgo,
        "demanda_calibrada": demanda_calibrada,
        "n_referencias_comunes": len(comunes),
    }
