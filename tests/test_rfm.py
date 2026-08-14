# -*- coding: utf-8 -*-
"""Tests unitarios del módulo RFM."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rfm import (
    ORDEN_SEGMENTOS,
    aplicar_log1p,
    calcular_rfm,
    filtrar_clientes_identificados,
    mapear_segmentos,
    resumen_por_segmento,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fact_mini() -> pd.DataFrame:
    """Un fact_ventas minúsculo pero completo para tests unitarios."""
    return pd.DataFrame({
        "id_ticket":    [1, 1, 2, 3, 4, 5, 6],
        "id_cliente":   pd.array([10, 10, 10, 20, 20, pd.NA, 30], dtype="Int32"),
        "importe_neto": [3.0, 2.5, 4.0, 5.0, 10.0, 7.0, 6.5],
        "fecha":        pd.to_datetime([
            "2025-12-01", "2025-12-01", "2025-12-15",
            "2025-11-01", "2025-12-25",
            "2025-11-30", "2025-12-30",
        ]),
    })


@pytest.fixture
def fecha_ref() -> pd.Timestamp:
    return pd.Timestamp("2026-01-01")


# ---------------------------------------------------------------------------
# filtrar_clientes_identificados
# ---------------------------------------------------------------------------
def test_filtrar_clientes_excluye_nulos(fact_mini):
    filtrado = filtrar_clientes_identificados(fact_mini)
    assert filtrado["id_cliente"].isna().sum() == 0
    assert filtrado["id_cliente"].dtype == np.int32
    assert len(filtrado) == 6  # se pierde 1 línea (la del cliente NaN)


# ---------------------------------------------------------------------------
# calcular_rfm
# ---------------------------------------------------------------------------
def test_calcular_rfm_columnas_y_forma(fact_mini, fecha_ref):
    fact_id = filtrar_clientes_identificados(fact_mini)
    rfm = calcular_rfm(fact_id, fecha_ref)
    assert set(rfm.columns) == {"id_cliente", "recency", "frequency", "monetary"}
    assert len(rfm) == 3  # tres clientes únicos: 10, 20, 30


def test_frequency_cuenta_tickets_unicos(fact_mini, fecha_ref):
    fact_id = filtrar_clientes_identificados(fact_mini)
    rfm = calcular_rfm(fact_id, fecha_ref)
    freq_10 = rfm.loc[rfm.id_cliente == 10, "frequency"].values[0]
    # Cliente 10 tiene 3 líneas pero solo 2 tickets únicos (1 y 2)
    assert freq_10 == 2


def test_monetary_suma_todas_las_lineas(fact_mini, fecha_ref):
    fact_id = filtrar_clientes_identificados(fact_mini)
    rfm = calcular_rfm(fact_id, fecha_ref)
    mon_10 = rfm.loc[rfm.id_cliente == 10, "monetary"].values[0]
    assert mon_10 == pytest.approx(9.5)   # 3 + 2.5 + 4


def test_recency_dias_desde_ultima_compra(fact_mini, fecha_ref):
    fact_id = filtrar_clientes_identificados(fact_mini)
    rfm = calcular_rfm(fact_id, fecha_ref)
    rec_10 = rfm.loc[rfm.id_cliente == 10, "recency"].values[0]
    # Última compra del cliente 10: 2025-12-15. Ref: 2026-01-01. Días: 17.
    assert rec_10 == 17


def test_calcular_rfm_fecha_ref_automatica(fact_mini):
    """Si no pasamos fecha_ref, debe ser max+1 y garantizar Recency>=1."""
    fact_id = filtrar_clientes_identificados(fact_mini)
    rfm = calcular_rfm(fact_id, fecha_ref=None)
    assert (rfm["recency"] >= 1).all()


# ---------------------------------------------------------------------------
# aplicar_log1p
# ---------------------------------------------------------------------------
def test_aplicar_log1p_anade_columnas(fact_mini, fecha_ref):
    fact_id = filtrar_clientes_identificados(fact_mini)
    rfm = calcular_rfm(fact_id, fecha_ref)
    rfm_log = aplicar_log1p(rfm)
    for col in ["recency_log", "frequency_log", "monetary_log"]:
        assert col in rfm_log.columns


def test_aplicar_log1p_rechaza_negativos():
    rfm_malo = pd.DataFrame({
        "id_cliente": [1], "recency": [-1], "frequency": [10], "monetary": [50.0],
    })
    with pytest.raises(ValueError, match="valores negativos"):
        aplicar_log1p(rfm_malo)


# ---------------------------------------------------------------------------
# mapear_segmentos
# ---------------------------------------------------------------------------
def test_mapear_segmentos_asigna_nombres():
    rfm = pd.DataFrame({
        "id_cliente": [1, 2, 3],
        "cluster":    [0, 1, 2],
    })
    mapa = {0: "Champions", 1: "Loyal", 2: "Lost"}
    salida = mapear_segmentos(rfm, mapa)
    assert list(salida["segmento_rfm"]) == ["Champions", "Loyal", "Lost"]


def test_mapear_segmentos_rechaza_nombres_no_canon():
    rfm = pd.DataFrame({"id_cliente": [1], "cluster": [0]})
    mapa = {0: "SúperClientes"}   # nombre inventado, no está en canon
    with pytest.raises(ValueError, match="fuera del canon"):
        mapear_segmentos(rfm, mapa)


# ---------------------------------------------------------------------------
# resumen_por_segmento
# ---------------------------------------------------------------------------
def test_resumen_por_segmento_porcentajes_suman_100():
    rfm = pd.DataFrame({
        "id_cliente":   [1, 2, 3, 4],
        "recency":      [1, 5, 30, 200],
        "frequency":    [100, 50, 20, 5],
        "monetary":     [1000.0, 500.0, 200.0, 50.0],
        "segmento_rfm": pd.Categorical(
            ["Champions", "Loyal", "Potential", "Lost"],
            categories=list(ORDEN_SEGMENTOS), ordered=True,
        ),
    })
    resumen = resumen_por_segmento(rfm)
    assert resumen["%_clientes"].sum() == pytest.approx(100.0, abs=0.5)
    assert resumen["%_revenue"].sum() == pytest.approx(100.0, abs=0.5)