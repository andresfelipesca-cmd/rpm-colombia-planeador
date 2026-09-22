# Cierre numérico y trazabilidad de la aplicación RPM

## Alcance

Esta nota consolida las validaciones realizadas sobre la versión corregida del motor de optimización. El objetivo es separar resultados confirmados, resultados aproximados con GAP explícito y resultados históricos que todavía no deben sobrescribirse.

## 1. MILP diario canónico

La formulación con unidades enteras, demanda por igualdad, capacidad por máquina y enlace ajustado reproduce:

- Objetivo: **11.547,4126 min**.
- GAP: **0%** en la validación independiente.
- Política empírica integral: **13.684,4886 min**.
- Mejora relativa: **15,6168%**, compatible con el 15,6% reportado en la tesis.

## 2. Capacidad crítica

Con +30 min de capacidad y reoptimización del mismo MILP:

| Máquina | Reducción del objetivo | eta_j (min/min) |
|---|---:|---:|
| M8 | 5,4161 min | 0,18054 |
| M7 | 3,2289 min | 0,10763 |
| M9 | 0,3550 min | 0,01183 |

La jerarquía M8 > M7 > M9 queda confirmada.

## 3. Sensibilidad de demanda con unidades enteras

La demanda escalada se discretiza mediante redondeo half-up antes de resolver el MILP.

| Factor | Demanda entera | Objetivo / incumbente | GAP observado |
|---:|---:|---:|---:|
| 0,8 | 11.353 | 9.508,938 | ~0,00085% |
| 0,9 | 12.773 | 10.517,487 | ~0,00092% |
| 1,0 | 14.188 | 11.547,413 | 0% |
| 1,1 | 15.614 | 12.585,379 | ~0,00070% |
| 1,2 | 17.023 | 13.675,718 | ~0,631% |

El escenario 120% debe presentarse como solución factible con cota dentro del tiempo de validación, no como GAP 0.

### Comparación con la política empírica integral

La política greedy corregida satisface la demanda en 80%, 90% y 100%. Las mejoras del MILP son **18,9603%**, **17,2171%** y **15,6168%**, respectivamente. En 110% la heurística deja **69 unidades** sin atender y en 120% deja **487 unidades** sin atender; en esos dos escenarios no se reporta una mejora porcentual de tiempo porque el nivel de servicio no es comparable.

## 4. Sensibilidad de celdas imputadas

Se perturbaron únicamente las 171 celdas identificadas como imputadas dentro de la matriz usada por el modelo.

| Perturbación | Incumbente | Cota inferior | GAP aproximado |
|---:|---:|---:|---:|
| -20% | 11.187,628 | 11.159,608 | 0,251% |
| 0% | 11.547,413 | 11.547,413 | 0% |
| +20% | 11.560,151 | 11.546,074 | 0,122% |

Estas cifras son una revalidación independiente y no deben presentarse como óptimos exactos en los dos extremos mientras el GAP sea positivo.

## 5. Multiperiodo de tres días

La formulación corregida usa demanda entera para 200 referencias. Con 180 días hábiles para estimar la demanda, el factor de calibración es **2,3743175**. Para tres días:

- Demanda teórica del horizonte: **45.991,483** unidades.
- Demanda discretizada: **45.987** unidades.
- Incumbente CBC: **34.234,8717 min**.
- Cota inferior conservadora observada en raíz: **34.071,197 min**.
- GAP conservador usando esa cota: **<= 0,478%**.
- CBC se detuvo por el criterio de GAP de 0,5%.

La cifra histórica de 34.506 min con GAP 1,16% no corresponde a esta formulación entera corregida y no debe mezclarse con ella. Los archivos `results/multiday_plan_final.csv`, `multiday_setups_final.csv` y `multiday_utilizacion_final.csv` contienen el plan y los agregados utilizados para actualizar la tesis.

## 6. Monte Carlo

La app conserva como referencia histórica documentada:

- N = 1000.
- seed = 42.
- U(0,85; 1,15) independiente por referencia.
- Media = 15,14%.
- IC95% = [15,09%; 15,19%].

La nueva versión no ejecuta N=1000 automáticamente. Una corrida independiente de control con N=10 confirmó el funcionamiento del módulo y produjo una mejora media aproximada de 15,17%, pero algunos escenarios conservaron pequeños GAPs. Por tanto, N=10 es solo prueba funcional y **no reemplaza** el resultado N=1000 de la tesis.

## 7. Regla de reporte

- `CONFIRMADO`: puede mantenerse en la tesis.
- `FACTIBLE CON GAP`: debe reportarse junto con la cota y no llamarse óptimo.
- `REFERENCIA HISTÓRICA`: se conserva hasta una nueva corrida completa comparable.
- `CAMBIA EN FORMULACIÓN CORREGIDA`: requiere actualizar la tesis si se adopta la nueva formulación como definitiva.

La tabla maestra `results/TABLA_MAESTRA_RESULTADOS_FINAL.xlsx` es la fuente de trazabilidad recomendada.
