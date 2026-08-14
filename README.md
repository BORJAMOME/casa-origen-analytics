# Casa Origen — Segmentación de clientes y análisis de cesta de la compra

Proyecto de Machine Learning aplicado a un caso de negocio de retail/hostelería: dos cafeterías de especialidad en Madrid (Lavapiés y Malasaña), con datos sintéticos de 3 años (2023-2025) generados para simular un negocio real.

Este repositorio recoge la fase de Machine Learning del proyecto. El resultado final —un dashboard Power BI completo— puede verse aquí: **[borjamora.es/casa-origen.html](https://borjamora.es/casa-origen.html)**.

## De dónde vienen los datos

Antes de llegar al Machine Learning, el proyecto pasa por tres fases que resumo aquí porque el modelado depende de ellas, aunque el código de esa parte no está en este repo:

1. **Generación de datos sintéticos** (Python): 179.112 tickets, 37 productos en 6 categorías, 1.200 clientes, 2 locales, 3 canales de venta (in-store, Glovo, Uber Eats) — con reglas de negocio realistas (estacionalidad, franjas horarias, mermas, descuentos).
2. **Modelo de datos en estrella** (1 tabla de hechos + 6 dimensiones) e ingesta en **arquitectura Medallion** (Bronze → Silver → Gold) sobre SQL Server, con validaciones de calidad de dato en cada capa.
3. **Modelo Power BI y DAX**: métricas de negocio (RevPASH, ticket medio, margen, recencia de cliente) sobre el modelo Gold.

A partir de la capa Gold es donde arranca lo que hay en este repositorio.

## Los dos modelos

### 1. Segmentación de clientes (RFM + K-Means)

`notebooks/01_rfm_segmentacion.ipynb` · `src/rfm.py` · `src/clustering.py`

Cálculo del cubo RFM (Recencia, Frecuencia, Monetario) por cliente, transformación logarítmica y clustering K-Means. El modelo separa 5 segmentos con un 99,8% de varianza explicada en 2 componentes PCA:

| Segmento | % clientes | % revenue | Lectura |
|---|---|---|---|
| Champions | 15,0% | 30,8% | Compran esta semana, 2 veces/semana, triplican el ticket medio |
| Loyal | 37,7% | 39,9% | Habituales estables, ~1 visita/semana |
| At Risk | 22,3% | — | Mismo historial que Loyal pero casi un mes sin volver — máximo ROI de reactivación |
| Potential | 10,8% | — | Activos moderados, valor bajo |
| Lost | 14,2% | — | 9 meses de silencio |

**Titular de negocio:** el 52,7% de los clientes (Champions + Loyal) genera el 70,7% del revenue. La interpretación completa, con las decisiones accionables derivadas, está en [`outputs/interpretacion_h5.txt`](outputs/interpretacion_h5.txt).

### 2. Análisis de cesta de la compra (Market Basket / FP-Growth)

`notebooks/02_market_basket.ipynb` · `src/basket.py`

FP-Growth (min_support 0.005) sobre 179.112 tickets para identificar combinaciones de producto con lift significativo, segmentadas por franja horaria. Resultado: 20 itemsets frecuentes y 5 reglas cross-categoría con lift de hasta 3,46, traducidas a propuestas concretas de combo (p. ej. Kombucha + Tosta de Aguacate en horario de brunch, lift 3,46).

Interpretación completa en [`outputs/interpretacion_h2.txt`](outputs/interpretacion_h2.txt).

## Resultados visuales

Gráficas generadas por ambos modelos en [`results/`](results/): distribución RFM, codo/silhouette para elegir *k*, clusters en PCA, mapa de segmentos (Pareto), y tamaño/composición de cesta por franja horaria.

## Estructura del repo

```
notebooks/    → los dos notebooks de análisis, de principio a fin
src/          → funciones puras, testeadas, sin efectos secundarios (io, cálculo RFM, clustering, basket)
tests/        → pytest sobre src/
models/       → K-Means y scaler entrenados (.joblib)
outputs/      → CSVs de salida (segmentos, itemsets, reglas) e interpretación de negocio en texto
results/      → gráficas de resultado
```

## Cómo reproducirlo

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

## Limitaciones (declaradas en el propio análisis)

Dataset sintético con semilla fija (seed=42): los porcentajes exactos cambiarían con datos reales, pero los patrones (concentración de valor tipo Pareto, combos cross-categoría) se sostendrían. El silhouette del clustering es moderado (~0,3) por la homogeneidad propia de un dataset simulado.

---

*Proyecto Final de Bootcamp de Data Analytics — Borja Mora Méndez.*
[Portfolio](https://borjamora.es) · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/)
