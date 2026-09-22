# Auditoría técnica y mejoras de la aplicación RPM Colombia

## Propósito de esta versión

Esta versión separa tres objetivos que no deben confundirse: reproducir el resultado principal de la tesis, auditar la calidad de los datos y explorar la estabilidad de las decisiones. El núcleo sigue siendo el MILP diario con unidades enteras, demanda exacta, capacidad por máquina y activación binaria por referencia-máquina.

## Correcciones realizadas

1. **Integralidad de la demanda en sensibilidad.** Los escenarios de 80%, 90%, 110% y 120% ya no envían demandas fraccionarias a un modelo con `X_ij` entera. La app aplica un redondeo half-up explícito y reporta el ajuste total de unidades.
2. **Cuello de botella consistente.** El análisis de capacidad vuelve a resolver exactamente el mismo MILP base; la única modificación es `Cap_j + delta` para una máquina a la vez.
3. **Política de referencia integral.** La heurística greedy asigna unidades completas y por tanto comparte el mismo dominio físico del MILP.
4. **Multi-día con demanda entera.** La demanda calibrada del horizonte se discretiza antes de construir las restricciones de igualdad. El ajuste de redondeo se muestra al usuario.
5. **GAP y terminación del solver.** Se lee el log de CBC para distinguir óptimo probado, parada por tiempo y solución incumbente. Cuando no hay cierre, no se fuerza un GAP de 0%.
6. **Trazabilidad de Cij.** La app distingue entre la matriz publicada en `MODELO PL` y la matriz reconstruida desde hojas crudas con imputación por media de máquina.

## Validación independiente del caso base

Se verificó el MILP diario con un segundo solver, independiente del código PuLP/CBC de la app.

- Referencias: 99
- Máquinas: 10
- Demanda total: 14.188 unidades
- Objetivo: **11.547,4126 min**
- GAP: **0,00%**
- Tiempo de producción: **9.147,2106 min**
- Tiempo de setup: **2.400,2020 min**
- Participación de setup: **20,786%** del objetivo
- Holgura agregada: **2.852,5874 min**
- Utilización promedio: **80,190%**
- Desviación estándar de utilización: **25,087 puntos porcentuales**
- Pares producto-máquina activos: **99**
- Referencias divididas entre varias máquinas: **0**
- Política greedy integral: **13.684,4886 min**
- Reducción relativa del MILP: **15,6168%**

## Dependencia de datos observados e imputados

La matriz cruda contiene 819 pares observados y 171 faltantes, para una cobertura empírica de 82,7%.

En la solución base:

- 98 de 99 pares activos se apoyan en un `C_ij` observado.
- 1 par activo utiliza un `C_ij` clasificado como imputado.
- Ese único par concentra 1.337 unidades, equivalentes a aproximadamente 9,4% de las unidades del plan.

Por esta razón la app reporta tanto la proporción de pares observados como la proporción de unidades asociadas a tiempos observados. Contar solo pares puede ocultar una dependencia volumétrica importante.

## Revalidación estadística de imputación

Se añadió un análisis de 100 particiones aleatorias. En cada repetición se oculta 20% de las celdas observadas y se comparan cuatro métodos sobre las mismas celdas ocultas.

| Método | MAE medio (min/u) | IC 95% del MAE medio | RMSE medio | Veces con menor MAE |
|---|---:|---:|---:|---:|
| Media por máquina | 0,1888 | [0,1869; 0,1906] | 0,2335 | 96/100 |
| Media global | 0,1992 | [0,1975; 0,2010] | 0,2454 | 3/100 |
| Aditivo referencia × máquina | 0,2066 | [0,2046; 0,2086] | 0,2553 | 1/100 |
| Media por referencia | 0,2180 | [0,2159; 0,2201] | 0,2691 | 0/100 |

Este resultado es una validación adicional. No sustituye el experimento histórico de una sola partición que ya aparece en la tesis; muestra que la superioridad de la media por máquina no depende de una única división aleatoria.

## Matriz publicada frente a reconstrucción desde datos crudos

La app mantiene ambas rutas separadas deliberadamente.

- MAE global entre ambas matrices: **0,0422 min/u**.
- MAE en celdas observadas: **0,0012 min/u**.
- MAE en celdas no observadas: **0,2385 min/u**.
- Celdas con diferencia superior a `1e-6`: **174**.

La matriz reconstruida con media por máquina no es idéntica a la matriz publicada. Al optimizar la reconstrucción, una validación independiente obtuvo **11.429,98 min**, aproximadamente 1,02% por debajo del objetivo publicado. Por esta razón la app no reemplaza silenciosamente una matriz por otra. El usuario elige explícitamente entre:

- `Matriz publicada (reproduce tesis)`
- `Reconstruida desde registros crudos`

Para la sustentación, el resultado canónico de la tesis sigue siendo el de la matriz publicada.

## Capacidad crítica con el MILP corregido

Con `delta = 30 min` y reoptimización exacta del mismo modelo:

| Máquina | Reducción del objetivo (min) | eta_j (min/min) |
|---|---:|---:|
| M8 | 5,4161 | 0,18054 |
| M7 | 3,2289 | 0,10763 |
| M9 | 0,3550 | 0,01183 |

Estos resultados coinciden con las cifras redondeadas documentadas en la tesis y confirman que el orden M8 > M7 > M9 no era un artefacto de la relajación continua del prototipo anterior.

## Nuevas métricas y visualizaciones

La app incorpora métricas que tienen interpretación operativa y metodológica:

- Pareto de demanda y número de referencias necesario para alcanzar 80% del volumen.
- Índice de Gini de la demanda.
- Cobertura observada de `C_ij` por máquina.
- Descomposición de capacidad en producción, setup y holgura.
- Dispersión de utilización entre máquinas.
- Residuos de factibilidad para demanda, capacidad, enlace e integralidad.
- Soporte observado/imputado de las asignaciones, medido por pares y por unidades.
- ARI K-Means vs Ward y Silhouette de ambos métodos.
- Kruskal-Wallis y epsilon-cuadrado como descripción del grado de separación de clusters.
- Sensibilidad de demanda con ajuste de discretización explícito.
- Sensibilidad de celdas imputadas.
- Valor marginal finito de capacidad `eta_j`.
- Monte Carlo con media, mediana, desviación estándar, error estándar, IC 95%, P5-P95, rango y proporción de mejora positiva.
- Heatmap de utilización día-máquina en el modelo multiperiodo.

## Sensibilidad de demanda corregida

Al discretizar la demanda escalada, la diferencia frente al valor teórico es pequeña pero debe declararse:

| Factor | Demanda teórica | Demanda entera | Ajuste |
|---:|---:|---:|---:|
| 0,8 | 11.350,4 | 11.353 | +2,6 |
| 0,9 | 12.769,2 | 12.773 | +3,8 |
| 1,0 | 14.188,0 | 14.188 | 0,0 |
| 1,1 | 15.606,8 | 15.614 | +7,2 |
| 1,2 | 17.025,6 | 17.023 | -2,6 |

La revalidación independiente obtuvo los siguientes valores: 9.508,938; 10.517,487; 11.547,413; 12.585,379 y 13.675,718 min para los factores 0,8 a 1,2. Los escenarios hasta 110% cerraron con brechas inferiores a 0,001%. Para 120% se obtuvo una solución factible con cota inferior aproximada de 13.589,463 min y GAP de 0,631%, por lo que no se declara optimalidad exacta.

## Multi-día

La ruta corregida conserva 200 referencias y un factor de calibración de **2,3743175**. Para tres días, la demanda teórica es 45.991,483 unidades y la demanda discretizada 45.987 unidades. Una corrida CBC 2.10.11 con `gapRel=0.005` obtuvo un incumbente de **34.234,8717 min** y una cota inferior de raíz de **34.071,197 min**. Esta cota garantiza conservadoramente un GAP no superior a **0,478%**. CBC se detuvo al satisfacer el criterio de GAP de 0,5%; el resultado se reporta como solución factible con cota, no como óptimo exacto.

## Interpretación recomendada

La aplicación debe presentarse como herramienta de apoyo a la decisión y como implementación reproducible de los análisis de la tesis. Los nuevos indicadores amplían la trazabilidad y la defensa metodológica, pero no convierten el estudio en un modelo estocástico ni en una política universal de producción. En particular, la matriz de compatibilidad física referencia-máquina y los setups dependientes de secuencia siguen siendo extensiones futuras porque no existe información validada para parametrizarlos.

## Cierre de la versión de aplicación

La versión final de la app evita ejecutar Monte Carlo automáticamente. El resultado N=1000 de la tesis se conserva en un archivo de referencia con metadatos y se distingue visualmente de cualquier nueva corrida. Las nuevas simulaciones usan el benchmark integral corregido y solo incorporan a los estadísticos inferenciales los escenarios con optimalidad certificada.

Se añadió una tabla maestra de trazabilidad en `results/TABLA_MAESTRA_RESULTADOS.xlsx`. Su objetivo es impedir que una cifra histórica sea reemplazada silenciosamente por una nueva formulación. Los resultados confirmados se conservan; los resultados que dependen de una corrida todavía no ejecutada con CBC quedan marcados como pendientes.

### Valores que permanecen confirmados

- Z* diario = 11.547,4126 min, que se reporta como 11.547,41 min.
- GAP diario = 0%.
- Mejora frente a la política greedy integral = 15,6168%, compatible con el 15,6% reportado.
- M8, M7 y M9 mantienen el orden de valor marginal de capacidad documentado.
- Cobertura 819/990 observada y 171/990 imputada.
- ARI K-Means/Ward = 1,000 y valores Silhouette documentados.

### Estado de los resultados al cierre numérico

- **Monte Carlo N=1000:** permanece como referencia histórica de la tesis. La nueva versión usa demanda entera y ejecución opcional; no se ha repetido la corrida completa corregida.
- **Sensibilidad de celdas imputadas +/-20%:** revalidada. Los extremos son incumbentes factibles con GAP aproximado de 0,251% y 0,122%; no se presentan como óptimos exactos.
- **Sensibilidad de demanda 120%:** revalidada como factible con GAP aproximado de 0,631%; los escenarios hasta 110% cerraron con GAP inferior a 0,001%.
- **Multiperiodo de tres días:** revalidado con CBC bajo demanda entera. UB = 34.234,8717 min y GAP conservador <= 0,478%.
- **Benchmark empírico bajo alta demanda:** en 110% y 120% deja 69 y 487 unidades sin atender, respectivamente. Por esa razón no se calcula una mejora porcentual de tiempo en esos dos escenarios.
