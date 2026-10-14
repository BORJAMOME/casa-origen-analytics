# Casa Origen — Segmentación de clientes y análisis de cesta de la compra

[![Tests](https://github.com/BORJAMOME/casa-origen-analytics/actions/workflows/tests.yml/badge.svg)](https://github.com/BORJAMOME/casa-origen-analytics/actions/workflows/tests.yml)

Proyecto final del bootcamp de Data Analyst en Neoland. He desarrollado una solución de análisis de datos para descubrir **qué clientes aportan más valor al negocio y qué productos tienen potencial para venderse juntos**, combinando Python, SQL Server, Power BI y Machine Learning.

**[Ver el proyecto completo y el dashboard interactivo](https://borjamora.es/casa-origen.html)**

## El proyecto

Casa Origen es un negocio ficticio formado por dos cafeterías de especialidad en Madrid, con distintos perfiles de clientes, 37 productos y tres canales de venta: en local, Glovo y Uber Eats.

Para construir este caso práctico, generé datos sintéticos basados en reglas de negocio realistas, como la estacionalidad, los hábitos de compra, las franjas horarias, los descuentos y las mermas.

El análisis abarca tres años de actividad (2023–2025), con 179.112 tickets y 1.200 clientes.

A partir de estos datos, desarrollé dos modelos de Machine Learning para responder a preguntas concretas del negocio:

- **Segmentación de clientes:** ¿qué grupos de clientes existen, cuánto aportan a las ventas y dónde hay oportunidades de fidelización?
- **Análisis de cesta de la compra:** ¿qué productos suelen comprarse juntos y qué combinaciones podrían convertirse en nuevas ofertas?

## 1. Segmentación de clientes — RFM + K-Means

**Archivos:** `notebooks/01_rfm_segmentacion.ipynb` · `src/rfm.py` · `src/clustering.py`

He aplicado el análisis RFM (Recencia, Frecuencia y Valor Monetario) para estudiar cuándo compró cada cliente por última vez, con qué frecuencia compra y cuánto gasta. Después, utilicé K-Means para identificar cinco segmentos con comportamientos de compra diferentes.

| Segmento | Clientes | Revenue | Interpretación |
|---|---:|---:|---|
| Champions | 15,0 % | 30,8 % | Clientes de alto valor, muy activos y con un gasto elevado |
| Loyal | 37,7 % | 39,9 % | Clientes habituales que mantienen un comportamiento estable |
| At Risk | 22,3 % | — | Clientes que antes compraban con regularidad, pero llevan tiempo sin volver |
| Potential | 10,8 % | — | Clientes con compras esporádicas y menor gasto |
| Lost | 14,2 % | — | Clientes que llevan aproximadamente nueve meses sin actividad |

### ¿Qué aporta al negocio?

Champions y Loyal representan el 52,7 % de los clientes y generan el 70,7 % de los ingresos.

Por otro lado, At Risk presenta una oportunidad de análisis: sus patrones históricos de gasto son similares a los de los clientes Loyal, pero llevan casi un mes sin regresar.

Esta información permite plantear estrategias diferenciadas de fidelización y recuperación, en lugar de aplicar las mismas acciones a toda la base de clientes.

## 2. Análisis de cesta de la compra — FP-Growth

**Archivos:** `notebooks/02_market_basket.ipynb` · `src/basket.py`

He utilizado FP-Growth para identificar productos que aparecen juntos con más frecuencia de la esperada. Después, relacioné los resultados con las franjas horarias para descubrir combinaciones con potencial comercial.

El análisis identifica 20 conjuntos frecuentes de productos y cinco reglas de asociación entre categorías, con un lift de hasta 3,46.

Algunos ejemplos:

- **Brunch:** Kombucha + Tosta de Aguacate.
- **Mediodía:** Espresso + Tosta de Mantequilla y Miel.
- **Apertura:** Filter Coffee + Banana Bread.

### ¿Qué aporta al negocio?

Estas asociaciones permiten explorar propuestas de combos, promociones y recomendaciones adaptadas a los hábitos de compra de cada franja horaria.

Las reglas identifican asociaciones, no causalidad: que dos productos se compren juntos no garantiza que una promoción conjunta aumente las ventas.

## 3. Resultados y visualizaciones

En [`results/`](results/) encontrarás las principales visualizaciones del análisis:

- Distribución de las variables RFM antes y después de la transformación.
- Curva de codo y coeficiente silhouette para evaluar el número de clusters.
- Segmentos de clientes representados mediante PCA.
- Distribución de los segmentos y su contribución a los ingresos.
- Composición de la cesta de la compra por franja horaria.

Los resultados completos, las métricas y las interpretaciones de negocio están disponibles en [`outputs/`](outputs/).

## 4. Cómo he construido la solución

Este repositorio se centra en la fase de Machine Learning, pero forma parte de un proyecto de Business Intelligence más amplio.

El flujo de trabajo completo incluye:

1. **Python:** generación y preparación de los datos.
2. **SQL Server:** transformación y organización mediante una arquitectura Medallion (Bronze, Silver y Gold) y un modelo dimensional en estrella.
3. **Power BI:** creación de métricas, análisis de negocio y dashboards interactivos.
4. **Machine Learning:** segmentación de clientes con RFM y K-Means, y análisis de asociaciones con FP-Growth.
5. **Validación:** pruebas automatizadas con pytest para comprobar el funcionamiento de las principales funciones del proyecto.

## 5. Estructura del repositorio

```text
notebooks/    Notebooks de análisis y experimentación
src/          Funciones para preparación de datos, RFM, clustering y cesta
tests/        Pruebas automatizadas con pytest
models/       Modelos K-Means y scaler entrenados (.joblib)
outputs/      Resultados, segmentos, itemsets, reglas e interpretaciones
results/      Gráficas y visualizaciones del análisis



## Tecnologías

- **Python:** pandas, NumPy y scikit-learn.
- **Machine Learning:** K-Means y FP-Growth.
- **Análisis de datos:** RFM, PCA y reglas de asociación.
- **Bases de datos:** SQL Server.
- **Business Intelligence:** Power BI, DAX y modelado dimensional.
- **Calidad del código:** pytest y GitHub Actions.

## Enlaces

- [Proyecto completo y dashboard interactivo](https://borjamora.es/casa-origen.html)
- [Repositorio de GitHub](https://github.com/BORJAMOME/casa-origen-analytics)
- [LinkedIn](https://www.linkedin.com/in/borjamoramendez/)

---

*Proyecto académico desarrollado como trabajo final del bootcamp de Data Analyst en Neoland. Casa Origen y sus datos son ficticios y se utilizan exclusivamente con fines formativos.*
