# Planeador de Producción RPM Colombia

Aplicación Streamlit asociada al trabajo de optimización de asignación de referencias a máquinas mediante programación lineal entera mixta (MILP/PLEM).

## Qué contiene esta versión

- MILP diario con `X_ij` entera y `Y_ij` binaria.
- Demanda exacta con igualdad.
- Restricción de enlace con `U_ij = min(D_i, floor((Cap_j-S_j)/C_ij))`.
- Capacidad específica por máquina, utilizada también en reoptimización de cuellos de botella.
- Política greedy de referencia con unidades enteras.
- Sensibilidad de demanda con discretización explícita a unidades enteras.
- Sensibilidad de las celdas imputadas en `C_ij`.
- Clustering de productos y máquinas, Silhouette, Ward y ARI.
- Revalidación repetida de imputación con ocultamiento de celdas observadas.
- Modelo multiperiodo determinista, con demanda del horizonte convertida a unidades enteras.
- Simulación Monte Carlo configurable con semilla reproducible.
- Métricas de factibilidad, utilización, holgura, setups, balance de carga y soporte observado/imputado.

## Correcciones metodológicas respecto al prototipo anterior

1. La sensibilidad ya no envía demandas fraccionarias a un modelo con `X` entera.
2. El cuello de botella reutiliza exactamente el mismo MILP base; solo cambia la capacidad de una máquina.
3. La política de referencia asigna unidades enteras.
4. El modelo multi-día exige una demanda entera y la app reporta el ajuste de redondeo.
5. Cuando CBC no cierra el MILP, la app intenta recuperar de su log la mejor cota y el GAP; si no están disponibles, no inventa un valor.
6. La matriz publicada (`MODELO PL`) y la reconstrucción desde registros crudos se presentan como rutas distintas para evitar confundir reproducción con auditoría del dato.

## Ejecución

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Archivo de ejemplo

`AnexodeDatosxlsx.xlsx` contiene las hojas utilizadas por la aplicación. La instancia diaria principal se toma de `MODELO PL`; las hojas `Datos M1` a `Datos M10` se usan para auditar la cobertura observada y reconstruir la imputación.

## Reproducibilidad

- K-Means: `random_state=42`.
- Revalidación de imputación: semilla base 42 y 100 particiones por defecto.
- Monte Carlo: semilla configurable; valor por defecto 42.
- CBC: el tiempo máximo de cómputo se controla desde la barra lateral.

## Interpretación importante

La app es una herramienta de apoyo a la decisión. La matriz de compatibilidad física referencia-máquina no está codificada porque el archivo suministrado no contiene una matriz validada de incompatibilidades. Una combinación no observada se trata como dato faltante, no como incompatibilidad física.

## Selección explícita de la matriz Cij

La barra lateral ofrece dos modos:

- **Matriz publicada (reproduce tesis)**: usa `MODELO PL` y conserva el resultado canónico `Z*=11.547,41 min`.
- **Reconstruida desde registros crudos**: calcula los promedios de las hojas `Datos M1` ... `Datos M10` e imputa celdas faltantes con la media por máquina.

No se presentan como matrices equivalentes. El selector existe precisamente para separar reproducción del resultado y auditoría de datos.

## Informe de auditoría

`AUDITORIA_TECNICA.md` resume las correcciones metodológicas, las métricas añadidas y los resultados de una validación independiente del caso base.

## Monte Carlo opcional y referencia precalculada

La pestaña **Monte Carlo** no ejecuta simulaciones al cargar la app. Primero muestra la corrida de referencia N=1000 documentada en la tesis (`results/monte_carlo_reference.json`) y ofrece tres tamaños de ejecución bajo demanda: 30, 100 y 1000 escenarios.

Las corridas nuevas se almacenan en caché durante 24 horas cuando se repiten con exactamente los mismos parámetros. La referencia histórica no se reemplaza automáticamente.

Para recalcular N=1000 desde línea de comandos:

```bash
python validation/recalcular_monte_carlo.py --n 1000 --seed 42 --time-limit 30
```

## Tabla maestra de resultados

La carpeta `results/` contiene la tabla maestra final de trazabilidad en Excel y CSV. Esta tabla separa:

- resultados confirmados por validación independiente;
- métricas nuevas añadidas a la app;
- valores históricos conservados como referencia;
- cifras que requieren una corrida CBC definitiva antes de actualizar la tesis.

En particular, la sensibilidad de demanda al 120% y el modelo multiperiodo no deben presentarse como óptimos si el solver no cierra el GAP.

## Criterio de parada del modelo multi-día

El módulo multi-día permite fijar explícitamente el GAP objetivo de CBC. El valor recomendado por defecto es **0,5%**, porque el modelo de 200 referencias crece rápidamente en variables binarias y no es necesario consumir cómputo hasta GAP 0 para una herramienta operativa. La interfaz muestra LB, UB/incumbente y GAP cuando CBC los reporta, y no presenta una solución con GAP positivo como óptimo certificado.

## Validación numérica

Los resultados de cierre, incluidos los cambios de sensibilidad y del modelo multi-día, están documentados en `RESULTADOS_CIERRE_NUMERICO.md` y en `results/TABLA_MAESTRA_RESULTADOS_FINAL.xlsx`. El Monte Carlo N=1000 permanece como referencia histórica de la tesis y su nueva ejecución es opcional.
