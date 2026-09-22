"""Chequeo de regresión para la instancia publicada de la tesis.

Ejecutar después de `pip install -r requirements.txt`:
    python validar_instancia.py

Los valores esperados son controles de reproducibilidad, no parámetros del
modelo ni resultados hardcodeados de la aplicación.
"""
from rpm_core import (
    cargar_datos,
    construir_cij_con_imputacion,
    concordancia_kmeans_ward_maquinas,
    demanda_entera,
    resolver_asignacion,
)

ARCHIVO = "AnexodeDatosxlsx.xlsx"
EXPECTED_Z = 11547.412555555555
TOL_Z = 0.05


def main():
    d = cargar_datos(ARCHIVO)
    C, D, S = d["cij"], demanda_entera(d["demanda"]), d["setup"]
    audit = construir_cij_con_imputacion(ARCHIVO, productos=C.index)
    res = resolver_asignacion(C, D, S, cap_minutos=1440, tiempo_limite_seg=120)
    concord = concordancia_kmeans_ward_maquinas(C, S, n_clusters=3)

    assert len(C) == 99, f"Se esperaban 99 referencias y se obtuvieron {len(C)}"
    assert audit["n_observadas"] == 819
    assert audit["n_imputadas"] == 171
    assert res["valor_objetivo"] is not None
    assert abs(res["valor_objetivo"] - EXPECTED_Z) <= TOL_Z, res["valor_objetivo"]
    assert max(res["residuos"].values()) <= 1e-6, res["residuos"]
    assert abs(concord["ari"] - 1.0) <= 1e-9, concord["ari"]

    print("Chequeo de regresión superado")
    print(f"Z* = {res['valor_objetivo']:.6f} min")
    print(f"GAP = {res['gap_pct']}")
    print(f"Cobertura Cij = {audit['R_C']*100:.2f}%")
    print(f"ARI K-Means/Ward = {concord['ari']:.3f}")


if __name__ == "__main__":
    main()
