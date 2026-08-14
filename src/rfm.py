# -*- coding: utf-8 -*-
"""
Módulo RFM para Casa Origen — Fase 5 (Machine Learning).

Funciones puras para:
  - cargar el fact_ventas y filtrar clientes identificados,
  - calcular el cubo RFM (Recency, Frequency, Monetary),
  - transformar variables con log1p,
  - mapear etiquetas K-Means a segmentos narrativos.

Todas las funciones son deterministas y testeables. Los efectos secundarios
(guardar modelos, escribir CSVs) viven en el notebook, no aquí.

Autor : Borja Mora Méndez
Fecha : julio 2026
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes de negocio (aprobadas por el Consejo)
# ---------------------------------------------------------------------------
COLUMNAS_FACT_MINIMAS = ("id_ticket", "id_fecha", "id_cliente", "importe_neto")
COLUMNAS_RFM = ("recency", "frequency", "monetary")

ORDEN_SEGMENTOS: tuple[str, ...] = (
    "Champions", "Loyal", "At Risk", "Potential", "Lost"
)

# Orden de negocio (mejor → peor) para Power BI Sort by Column.
# Se materializa como columna int en el CSV de salida, de modo que
# el ordenamiento no depende de la etiqueta cruda de K-Means (que
# es arbitraria y cambia con la semilla del entrenamiento).
MAPA_ORDEN_SEGMENTO: dict[str, int] = {
    seg: i + 1 for i, seg in enumerate(ORDEN_SEGMENTOS)
}
#  → {'Champions': 1, 'Loyal': 2, 'At Risk': 3, 'Potential': 4, 'Lost': 5}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def cargar_fact_ventas(
    ruta: Path,
    columnas: Iterable[str] = COLUMNAS_FACT_MINIMAS,
) -> pd.DataFrame:
    """Lee ``fact_ventas.csv`` con dtypes eficientes y añade la columna ``fecha``.

    Args:
        ruta: Ruta al CSV canónico ``data/fact_ventas.csv``.
        columnas: Subconjunto de columnas a cargar. Deben incluir ``id_fecha``.

    Returns:
        DataFrame con las columnas pedidas más una columna ``fecha`` de tipo
        ``datetime64[ns]`` derivada de ``id_fecha`` (formato YYYYMMDD).

    Raises:
        FileNotFoundError: Si la ruta no existe.
        ValueError: Si falta alguna columna esencial (``id_cliente``, ``id_fecha``).
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No encuentro el fact_ventas en: {ruta}")

    dtypes = {
        "id_ticket":    "int32",
        "id_fecha":     "int32",
        "id_cliente":   "Int32",
        "id_local":     "int8",
        "importe_neto": "float32",
    }
    dtypes_usados = {c: dtypes[c] for c in columnas if c in dtypes}

    fact = pd.read_csv(ruta, usecols=list(columnas), dtype=dtypes_usados)

    faltan = {"id_cliente", "id_fecha"} - set(fact.columns)
    if faltan:
        raise ValueError(f"Faltan columnas esenciales en el CSV: {faltan}")

    fact["fecha"] = pd.to_datetime(fact["id_fecha"].astype(str), format="%Y%m%d")
    return fact


def filtrar_clientes_identificados(fact: pd.DataFrame) -> pd.DataFrame:
    """Deja solo las líneas con ``id_cliente`` no nulo.

    Args:
        fact: DataFrame del fact_ventas (salida de ``cargar_fact_ventas``).

    Returns:
        Subconjunto de ``fact`` con ``id_cliente`` casteado a ``int32``.
    """
    salida = fact.dropna(subset=["id_cliente"]).copy()
    salida["id_cliente"] = salida["id_cliente"].astype("int32")
    return salida


# ---------------------------------------------------------------------------
# Cubo RFM
# ---------------------------------------------------------------------------
def calcular_rfm(
    fact: pd.DataFrame,
    fecha_ref: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Calcula la tabla RFM (Recency, Frequency, Monetary) por cliente.

    Args:
        fact: fact_ventas ya filtrado a clientes identificados. Debe contener
            las columnas ``id_cliente``, ``id_ticket``, ``importe_neto`` y ``fecha``.
        fecha_ref: Fecha de referencia para calcular Recency. Si es ``None``,
            se usa ``fact["fecha"].max() + 1 día`` para evitar Recency = 0.

    Returns:
        DataFrame con las columnas ``id_cliente``, ``recency`` (int días),
        ``frequency`` (int tickets únicos) y ``monetary`` (float euros).

    Ejemplo:
        >>> fact = pd.DataFrame({
        ...     "id_ticket": [1, 1, 2],
        ...     "id_cliente": [10, 10, 10],
        ...     "importe_neto": [3.0, 2.5, 4.0],
        ...     "fecha": pd.to_datetime(["2025-12-01", "2025-12-01", "2025-12-15"]),
        ... })
        >>> calcular_rfm(fact, pd.Timestamp("2026-01-01")).iloc[0].to_dict()
        {'id_cliente': 10, 'recency': 17, 'frequency': 2, 'monetary': 9.5}
    """
    if fecha_ref is None:
        fecha_ref = fact["fecha"].max() + pd.Timedelta(days=1)

    frequency = (
        fact.groupby("id_cliente")["id_ticket"]
            .nunique()
            .rename("frequency")
    )
    monetary = (
        fact.groupby("id_cliente")["importe_neto"]
            .sum()
            .rename("monetary")
    )
    ultima = fact.groupby("id_cliente")["fecha"].max()
    recency = (fecha_ref - ultima).dt.days.rename("recency").astype("int32")

    rfm = pd.concat([recency, frequency, monetary], axis=1).reset_index()
    return rfm


def aplicar_log1p(rfm: pd.DataFrame,
                  columnas: Iterable[str] = COLUMNAS_RFM) -> pd.DataFrame:
    """Añade columnas ``<col>_log`` con ``log1p`` de las variables RFM.

    Args:
        rfm: Cubo RFM (salida de ``calcular_rfm``).
        columnas: Nombres de columnas a transformar.

    Returns:
        Copia de ``rfm`` con nuevas columnas ``recency_log``, ``frequency_log``,
        ``monetary_log``.

    Raises:
        ValueError: Si alguna columna a transformar contiene valores negativos.
    """
    salida = rfm.copy()
    for col in columnas:
        if (salida[col] < 0).any():
            raise ValueError(
                f"La columna '{col}' contiene valores negativos; "
                f"log1p no es aplicable."
            )
        salida[f"{col}_log"] = np.log1p(salida[col])
    return salida


# ---------------------------------------------------------------------------
# Mapeo narrativo cluster → segmento
# ---------------------------------------------------------------------------
def mapear_segmentos(
    rfm: pd.DataFrame,
    mapa: dict[int, str],
    columna_cluster: str = "cluster",
    columna_segmento: str = "segmento_rfm",
) -> pd.DataFrame:
    """Asigna nombres narrativos a las etiquetas K-Means.

    Args:
        rfm: Cubo RFM con la columna de cluster ya calculada.
        mapa: Diccionario ``{cluster_id: nombre_segmento}``.
        columna_cluster: Nombre de la columna de cluster origen.
        columna_segmento: Nombre de la columna de destino.

    Returns:
        Copia de ``rfm`` con la nueva columna categórica ordenada según
        ``ORDEN_SEGMENTOS``.

    Raises:
        ValueError: Si algún nombre de segmento del mapa no está en el
            canon ``ORDEN_SEGMENTOS`` (protege de errores de tipeo).
    """
    nombres_no_canon = set(mapa.values()) - set(ORDEN_SEGMENTOS)
    if nombres_no_canon:
        raise ValueError(
            f"Nombres de segmento fuera del canon RFM: {nombres_no_canon}. "
            f"Válidos: {ORDEN_SEGMENTOS}."
        )

    salida = rfm.copy()
    salida[columna_segmento] = pd.Categorical(
        salida[columna_cluster].map(mapa),
        categories=list(ORDEN_SEGMENTOS),
        ordered=True,
    )
    return salida


def resumen_por_segmento(
    rfm: pd.DataFrame,
    columna_segmento: str = "segmento_rfm",
) -> pd.DataFrame:
    """Tabla resumen ejecutiva por segmento (n, %, R/F/M medios, revenue).

    Args:
        rfm: Cubo RFM con la columna de segmento ya asignada.
        columna_segmento: Nombre de la columna de segmento.

    Returns:
        DataFrame indexado por segmento con columnas: ``n_clientes``,
        ``%_clientes``, ``R_media``, ``F_media``, ``M_media``, ``revenue_total``,
        ``%_revenue``.
    """
    grp = rfm.groupby(columna_segmento, observed=True)
    resumen = grp.agg(
        n_clientes=("id_cliente", "count"),
        R_media=("recency", "mean"),
        F_media=("frequency", "mean"),
        M_media=("monetary", "mean"),
        revenue_total=("monetary", "sum"),
    ).round(1)

    resumen["%_clientes"] = (resumen["n_clientes"] / resumen["n_clientes"].sum() * 100).round(1)
    resumen["%_revenue"] = (resumen["revenue_total"] / resumen["revenue_total"].sum() * 100).round(1)

    orden_cols = ["n_clientes", "%_clientes",
                  "R_media", "F_media", "M_media",
                  "revenue_total", "%_revenue"]
    return resumen[orden_cols]