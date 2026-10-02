# Casa Origen — Segmentación de clientes y análisis de cesta de la compra

[![tests](https://github.com/BORJAMOME/casa-origen-analytics/actions/workflows/tests.yml/badge.svg)](https://github.com/BORJAMOME/casa-origen-analytics/actions/workflows/tests.yml)

## Contexto de negocio

Casa Origen son dos cafeterías de especialidad en Madrid, una en Lavapiés (abierta en enero de 2023) y otra en Malasaña (julio de 2024). Perfil de cliente distinto en cada local —vecinos y foodies en Lavapiés, turismo gastronómico en Malasaña— con una carta de 37 productos repartidos en 6 categorías (café, bakery, pan, tostas, bebidas, retail) y venta por tres canales: en local, Glovo y Uber Eats.

Es un negocio ficticio, construido como caso práctico para el proyecto final del bootcamp, pero con datos generados siguiendo reglas de negocio reales: estacionalidad, franjas horarias, mermas por categoría, descuentos. Tres años de histórico (2023-2025), 179.112 tickets, 1.200 clientes.

El dueño de un negocio así tiene dos preguntas de fondo: **a qué clientes merece la pena dedicar presupuesto de retención**, y **qué productos conviene vender juntos**. Este repositorio responde a las dos con Machine Learning.

Antes de llegar aquí, el proyecto pasa por generación de datos en Python, un modelo en estrella con arquitectura Medallion (Bronze/Silver/Gold) en SQL Server y un modelo Power BI con las métricas de negocio. Lo dejo resumido porque el foco de este repo es la fase de Machine Learning; el resultado completo, con dashboard interactivo, está en [borjamora.es/casa-origen.html](https://borjamora.es/casa-origen.html).

## Modelo 1 — Segmentación de clientes (RFM + K-Means)

`notebooks/01_rfm_segmentacion.ipynb` · `src/rfm.py` · `src/clustering.py`

Cálculo del cubo RFM (Recencia, Frecuencia, Monetario) por cliente, transformación logarítmica para corregir la asimetría y clustering K-Means. Los 5 clusters quedan bien separados: un 99,8% de la varianza se explica con solo 2 componentes en el PCA.

| Segmento | % clientes | % revenue | Qué significa |
|---|---|---|---|
| Champions | 15,0% | 30,8% | Compraron esta semana, vienen dos veces por semana, gastan el triple de la media |
| Loyal | 37,7% | 39,9% | Habituales estables, una visita semanal aproximada |
| At Risk | 22,3% | — | Mismo perfil de gasto que Loyal, pero casi un mes sin volver |
| Potential | 10,8% | — | Compran poco y de forma irregular, valor bajo |
| Lost | 14,2% | — | Nueve meses sin actividad |

El dato que manda: Champions y Loyal son poco más de la mitad de la base de clientes (52,7%) y generan el 70,7% del revenue. At Risk es el segmento con más margen de recuperación, porque su comportamiento histórico es casi idéntico al de Loyal — solo dejaron de venir. El detalle completo, con las decisiones de negocio que se derivan de cada segmento, está en [`outputs/interpretacion_h5.txt`](outputs/interpretacion_h5.txt).

## Modelo 2 — Cesta de la compra (FP-Growth)

`notebooks/02_market_basket.ipynb` · `src/basket.py`

FP-Growth (soporte mínimo 0,005) sobre los 179.112 tickets para encontrar qué productos se compran juntos, cruzando el resultado con la franja horaria. Salen 20 itemsets frecuentes y 5 reglas cross-categoría con lift de hasta 3,46 — bastante por encima del umbral en el que una combinación deja de ser casualidad.

Se traducen en combos concretos: Kombucha + Tosta de Aguacate en horario de brunch (lift 3,46), Espresso + Tosta de Mantequilla y Miel a mediodía, Filter Coffee + Banana Bread en la apertura. El razonamiento completo, con las cuatro "personalidades" del negocio por franja horaria, está en [`outputs/interpretacion_h2.txt`](outputs/interpretacion_h2.txt).

## Resultados visuales

En [`results/`](results/): distribución RFM antes y después de la transformación, curva de codo y silhouette para elegir *k*, clusters proyectados en PCA, mapa de segmentos y su peso en el revenue (Pareto), y composición de la cesta por franja horaria.

## Archivos

```
notebooks/    los dos notebooks de análisis, de principio a fin
src/          funciones puras y testeadas: carga de datos, cálculo RFM, clustering, basket
tests/        pytest sobre src/
models/       K-Means y scaler ya entrenados (.joblib)
outputs/      CSVs de salida (segmentos, itemsets, reglas) e interpretación de negocio en texto
results/      gráficas de resultado
```

## Cómo reproducirlo

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

## Limitaciones

El dataset es sintético, con semilla fija (seed=42): en un negocio real los porcentajes exactos cambiarían, aunque el patrón de fondo —concentración de valor tipo Pareto, combos cross-categoría por franja— se mantendría. El silhouette del clustering es moderado (~0,3), algo esperable en un dataset simulado con menos ruido del que tendría uno real.

## Licencia

El código está bajo licencia [MIT](LICENSE). El dataset es sintético, generado para este proyecto.

---

**Autor:** Borja Mora Méndez · [Portfolio](https://borjamora.es) · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [GitHub](https://github.com/BORJAMOME)
