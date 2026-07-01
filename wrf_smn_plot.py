#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 Ploteo del modelo WRF DET del SMN (Servicio Meteorologico Nacional, 4 km)
=============================================================================

Descarga los archivos NetCDF del pronostico deterministico WRF-ARW del SMN
desde el bucket publico de AWS y genera mapas con estilo "MetPy clasico":
mapa dentro de un recuadro blanco, colormap del producto al costado derecho,
titulo del producto arriba a la izquierda e informacion de la corrida
(inicializacion / validez) arriba a la derecha.

Guia oficial del formato de datos del SMN:
    https://odp-aws-smn.github.io/documentation_wrf_det/Formato_de_datos/

Bucket de AWS con los archivos .nc:
    https://smn-ar-wrf.s3-us-west-2.amazonaws.com/index.html#DATA/WRF/DET/2026/

Uso rapido:
    python wrf_smn_plot.py                       # viento 10 m, toda Argentina
    python wrf_smn_plot.py --var 10m_wind --region centro
    python wrf_smn_plot.py --date 20260630 --cycle 00 --lead 12 --region argentina

Autor: script generado para trabajar con el WRF DET del SMN.
=============================================================================
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import requests
import xarray as xr

import matplotlib
matplotlib.use("Agg")  # backend sin ventana (para guardar a archivo)
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------------
# 0) CONFIGURACION GENERAL
# ---------------------------------------------------------------------------

# URL base del bucket publico del SMN (acceso anonimo por HTTPS).
S3_BASE = "https://smn-ar-wrf.s3-us-west-2.amazonaws.com"

# Enlaces informativos que se muestran al pie del plot.
DOC_URL = "https://odp-aws-smn.github.io/documentation_wrf_det/Formato_de_datos/"
BUCKET_URL = "https://smn-ar-wrf.s3-us-west-2.amazonaws.com/index.html#DATA/WRF/DET/2026/"

# Carpeta donde se guardan los .nc descargados y las imagenes generadas.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Fuente: se prioriza Arial (pedido); si no esta instalada, se usa un
# sans-serif equivalente (DejaVu Sans) para conservar el estilo MetPy.
plt.rcParams["font.family"] = "sans-serif"
# Liberation Sans es metricamente compatible con Arial (mismo ancho/alto de
# glifos), por lo que el resultado es visualmente equivalente si Arial no esta.
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


def _font_note() -> str:
    """Devuelve un aviso sobre la fuente efectivamente usada."""
    try:
        names = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        names = set()
    if "Arial" in names:
        return ""
    if "Liberation Sans" in names:
        return ("[aviso] 'Arial' no esta instalada; se usa 'Liberation Sans' "
                "(metricamente identica a Arial).")
    return ("[aviso] 'Arial' no esta instalada; se usa un sans-serif "
            "equivalente (DejaVu Sans).")


# ---------------------------------------------------------------------------
# 1) REGIONES PARA HACER ZOOM  (lon_min, lon_max, lat_min, lat_max)
# ---------------------------------------------------------------------------
# El dominio del WRF cubre toda la Republica Argentina y alrededores.
# Para hacer zoom, simplemente elegir una region o agregar una nueva aca.

#   clave           :  (lon_min,  lon_max,  lat_min,  lat_max)
REGIONS: dict[str, tuple[float, float, float, float]] = {

    # -------------------- Dominio completo y macro-regiones --------------------
    "argentina":        (-76.0, -52.0, -56.0, -21.0),   # todo el pais + alrededores
    "noroeste":         (-70.0, -61.5, -31.5, -21.5),   # NOA
    "noreste":          (-63.5, -53.2, -31.0, -21.8),   # NEA / Litoral norte
    "cuyo":             (-71.0, -64.5, -37.8, -28.0),
    "centro":           (-68.5, -56.5, -41.5, -29.0),   # Region Centro / Pampeana
    "pampa_humeda":     (-64.0, -57.0, -39.0, -30.5),
    "litoral":          (-61.0, -53.2, -34.5, -25.5),
    "patagonia":        (-75.0, -62.0, -56.0, -38.0),
    "patagonia_norte":  (-72.0, -62.5, -42.5, -36.0),
    "patagonia_sur":    (-74.0, -63.5, -56.0, -42.0),
    "comahue":          (-72.0, -64.0, -42.0, -36.0),

    # -------------------- NOA (Noroeste) --------------------
    "noa_jujuy":        (-67.5, -64.0, -24.7, -21.6),
    "noa_salta":        (-68.7, -62.2, -26.5, -21.9),
    "noa_tucuman":      (-66.3, -64.4, -28.1, -25.9),
    "noa_catamarca":    (-69.2, -64.8, -30.2, -24.9),
    "noa_larioja":      (-69.7, -65.5, -32.0, -27.9),
    "noa_santiago":     (-65.7, -61.5, -30.8, -25.5),   # Santiago del Estero

    # -------------------- NEA (Noreste / Litoral) --------------------
    "nea_formosa":      (-62.6, -57.4, -27.0, -21.9),
    "nea_chaco":        (-63.6, -58.2, -28.2, -24.0),
    "nea_corrientes":   (-60.0, -55.5, -30.8, -27.1),
    "nea_misiones":     (-56.2, -53.5, -28.3, -25.4),

    # -------------------- Cuyo --------------------
    "cuyo_mendoza":     (-70.7, -66.4, -37.7, -31.9),
    "cuyo_sanjuan":     (-70.7, -66.8, -32.5, -28.4),
    "cuyo_sanluis":     (-67.3, -64.8, -36.1, -31.8),

    # -------------------- Centro / Pampeana --------------------
    "centro_cordoba":       (-65.9, -61.7, -35.1, -29.4),
    "centro_santafe":       (-63.1, -58.7, -34.1, -27.9),
    "centro_entrerios":     (-60.9, -57.7, -34.2, -30.0),
    "centro_lapampa":       (-68.4, -62.9, -39.6, -34.9),
    "centro_buenosaires":   (-63.5, -56.6, -41.2, -33.1),
    "centro_caba":          (-58.75, -58.15, -34.85, -34.40),  # CABA (zoom fino)
    "amba":                 (-59.6, -57.7, -35.4, -34.0),      # Area Metrop. Bs.As.

    # -------------------- Patagonia --------------------
    "pat_neuquen":          (-72.0, -67.9, -41.2, -35.9),
    "pat_rionegro":         (-72.0, -62.7, -42.1, -37.5),
    "pat_chubut":           (-72.1, -63.5, -46.1, -41.9),
    "pat_santacruz":        (-73.7, -65.6, -52.5, -45.9),
    "pat_tierradelfuego":   (-68.8, -63.4, -55.2, -52.5),

    # -------------------- Alias / compatibilidad --------------------
    "noa":              (-70.0, -61.5, -31.5, -21.5),   # = noroeste
    "nea":              (-63.5, -53.2, -31.0, -21.8),   # = noreste
    "norte":            (-70.0, -57.0, -30.0, -21.0),
    "buenos_aires":     (-63.5, -56.6, -41.2, -33.1),   # = centro_buenosaires
    "patagonia_n":      (-72.0, -62.5, -42.5, -36.0),   # = patagonia_norte
    "sur":              (-74.0, -63.5, -56.0, -49.0),
}

# Nombre descriptivo (para el listado). Solo informativo.
REGION_LABELS: dict[str, str] = {
    "argentina": "Republica Argentina (dominio completo)",
    "noroeste": "NOA - Noroeste argentino",
    "noreste": "NEA - Noreste argentino / Litoral norte",
    "cuyo": "Cuyo",
    "centro": "Region Centro / Pampeana",
    "pampa_humeda": "Pampa Humeda",
    "litoral": "Litoral (Entre Rios, Corrientes, Misiones, Santa Fe)",
    "patagonia": "Patagonia",
    "patagonia_norte": "Patagonia Norte",
    "patagonia_sur": "Patagonia Sur",
    "comahue": "Comahue (Neuquen, Rio Negro, oeste de La Pampa)",
    "noa_jujuy": "Jujuy", "noa_salta": "Salta", "noa_tucuman": "Tucuman",
    "noa_catamarca": "Catamarca", "noa_larioja": "La Rioja",
    "noa_santiago": "Santiago del Estero",
    "nea_formosa": "Formosa", "nea_chaco": "Chaco",
    "nea_corrientes": "Corrientes", "nea_misiones": "Misiones",
    "cuyo_mendoza": "Mendoza", "cuyo_sanjuan": "San Juan",
    "cuyo_sanluis": "San Luis",
    "centro_cordoba": "Cordoba", "centro_santafe": "Santa Fe",
    "centro_entrerios": "Entre Rios", "centro_lapampa": "La Pampa",
    "centro_buenosaires": "Buenos Aires (provincia)",
    "centro_caba": "Ciudad Autonoma de Buenos Aires (CABA)",
    "amba": "Area Metropolitana de Buenos Aires (AMBA)",
    "pat_neuquen": "Neuquen", "pat_rionegro": "Rio Negro",
    "pat_chubut": "Chubut", "pat_santacruz": "Santa Cruz",
    "pat_tierradelfuego": "Tierra del Fuego",
    "noa": "NOA (alias)", "nea": "NEA (alias)", "norte": "Norte (alias)",
    "buenos_aires": "Buenos Aires (alias)", "patagonia_n": "Patagonia Norte (alias)",
    "sur": "Sur / Patagonia austral (alias)",
}


def add_region(name: str, lon_min: float, lon_max: float,
               lat_min: float, lat_max: float, label: str | None = None) -> None:
    """Agrega o redefine una region de zoom en tiempo de ejecucion."""
    REGIONS[name] = (lon_min, lon_max, lat_min, lat_max)
    if label:
        REGION_LABELS[name] = label


def list_regions() -> None:
    """Imprime por pantalla todas las regiones disponibles con sus coordenadas."""
    print("\nRegiones disponibles (--region <clave>):\n")
    print(f"  {'clave':<22} {'lon_min':>8} {'lon_max':>8} "
          f"{'lat_min':>8} {'lat_max':>8}   descripcion")
    print("  " + "-" * 84)
    for key, (lo0, lo1, la0, la1) in REGIONS.items():
        label = REGION_LABELS.get(key, "")
        print(f"  {key:<22} {lo0:>8.2f} {lo1:>8.2f} {la0:>8.2f} {la1:>8.2f}   {label}")
    print(f"\n  Total: {len(REGIONS)} regiones.\n")


def list_products() -> None:
    """Imprime la lista de productos/variables disponibles (--var <clave>)."""
    print("\nProductos / variables disponibles (--var <clave>):\n")
    print(f"  {'clave':<10} {'freq':<5} {'unidad':<32} descripcion")
    print("  " + "-" * 92)
    for key, p in PRODUCTS.items():
        extra = f" [acum. {p.accum_hours} h]" if p.accum_hours else ""
        print(f"  {key:<10} {p.freq:<5} {p.units_label:<32} {p.title}{extra}")
    print(f"\n  Total: {len(PRODUCTS)} productos.")
    print("  Ejemplo: python wrf_smn_plot.py --var pp_24h --region centro "
          "--lead 24\n")


# ---------------------------------------------------------------------------
# 2) PALETAS DE COLOR (colormap del producto)
# ---------------------------------------------------------------------------

# Paleta de viento a 10 m (9 colores = 9 intervalos, de amarillo claro a rojo
# oscuro). Va acompañada de WIND_LEVELS (10 niveles en km/h).
WIND_COLORS = [
    "#ffffb4", "#fffb78", "#ffce36", "#ffa900", "#ff5d00",
    "#ff2a00", "#eb0600", "#c70000", "#a00000",
]
WIND_LEVELS = [30, 35, 40, 50, 60, 75, 90, 100, 120, 150]


def _wind_cmap():
    """Paleta para velocidad de viento a 10 m (km/h)."""
    return ListedColormap(WIND_COLORS)


def _temp_cmap():
    return plt.get_cmap("RdYlBu_r")


def _rh_cmap():
    return plt.get_cmap("BrBG")


def _pp_cmap():
    colors = [
        "#ffffff", "#c8e6c9", "#81c784", "#4caf50", "#2e7d32",
        "#64b5f6", "#1e88e5", "#0d47a1", "#8e24aa", "#4a148c",
        "#f06292", "#ad1457",
    ]
    return ListedColormap(colors)


# Paleta de precipitacion en 10 minutos (14 colores = 14 intervalos).
PP10M_COLORS = [
    "#006538", "#2ea355", "#75c678", "#c1e498", "#fbffc9", "#018087",
    "#66a8cf", "#b9c8dd", "#a43603", "#e5550c", "#fc8c3f", "#fcbe83",
    "#790276", "#7c0079",
]

# Si True, se muestra la INTENSIDAD en mm/h (= mm en 10 min x 6).
# Si False, se muestra la precipitacion ACUMULADA nativa en mm (por 10 min).
PP10M_AS_RATE = False

# Escala en mm/h pensada para un periodo de acumulacion de 10 minutos.
# 15 bordes -> 14 intervalos (coincide con los 14 colores). extend='neither'.
# Va desde llovizna debil hasta lluvia extrema (convectiva). El tope (200 mm/h)
# equivale a ~33 mm en 10 min, valor practicamente inalcanzable.
PP10M_LEVELS_RATE = [0.5, 1, 2, 4, 6, 10, 15, 20, 30, 45, 60, 90, 120, 160, 200]
# Escala en mm ACUMULADOS en 10 minutos (14 intervalos). Buena resolucion en
# valores debiles/moderados y tope alto (50 mm/10min) para eventos extremos.
PP10M_LEVELS_MM = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50]


def _pp10m_cmap():
    """Paleta para precipitacion en 10 minutos (14 colores)."""
    return ListedColormap(PP10M_COLORS)


def _pp_10min(ds: xr.Dataset) -> np.ndarray:
    """Campo de precipitacion en 10 min. Devuelve mm/h (intensidad) o mm nativo
    segun PP10M_AS_RATE."""
    pp = ds["PP"].isel(time=0).values  # mm acumulados en 10 minutos
    return pp * 6.0 if PP10M_AS_RATE else pp


# ---------------------------------------------------------------------------
# 3) REGISTRO DE PRODUCTOS / VARIABLES
# ---------------------------------------------------------------------------

@dataclass
class Product:
    """Define como se genera y rotula cada producto."""
    key: str
    # Titulo (arriba a la izquierda). Se admite doble idioma como pidio el user.
    title: str
    subtitle: str
    units_label: str          # etiqueta de la barra de color
    cmap: Callable            # funcion que devuelve el colormap
    levels: np.ndarray        # niveles del contourf / barra de color
    kind: str = "contourf"    # 'contourf' | 'barbs'
    extend: str = "max"
    # Funcion que, dado el dataset, devuelve el campo escalar a sombrear.
    field_fn: Callable[[xr.Dataset], np.ndarray] = None
    barbs: bool = False        # si ademas se dibujan barbas de viento
    freq: str = "01H"          # frecuencia del archivo: '01H' | '10M' | '24H'
    # Si accum_hours esta definido, el producto es una ACUMULACION de PP horaria
    # (suma de 'accum_hours' archivos 01H consecutivos que terminan en el plazo).
    accum_hours: int | None = None
    # Periodo de acumulacion en minutos (para armar el titulo automaticamente).
    period_min: int | None = None


def _wind_speed_knots(ds: xr.Dataset) -> np.ndarray:
    mag = ds["magViento10"].isel(time=0).values  # m/s
    return mag * 1.94384  # a nudos


def _wind_speed_kmh(ds: xr.Dataset) -> np.ndarray:
    mag = ds["magViento10"].isel(time=0).values  # m/s
    return mag * 3.6  # a km/h


def build_accum_title(period_min: int) -> str:
    """Arma el titulo de un producto de precipitacion acumulada segun el
    periodo (en minutos), respetando genero/numero en espanol.

    Ejemplos:
        10   -> 'Precipitacion Acumulada 10 minutos previos (mm, somb.)'
        60   -> 'Precipitacion Acumulada 1 hora previa (mm, somb.)'
        360  -> 'Precipitacion Acumulada 6 horas previas (mm, somb.)'
        1440 -> 'Precipitacion Acumulada 24 horas previas (mm, somb.)'
    """
    if period_min % 60 == 0:                       # expresable en horas
        h = period_min // 60
        unidad = "hora" if h == 1 else "horas"
        previo = "previa" if h == 1 else "previas"
        cantidad = f"{h} {unidad}"
    else:                                          # se deja en minutos
        unidad = "minuto" if period_min == 1 else "minutos"
        previo = "previo" if period_min == 1 else "previos"
        cantidad = f"{period_min} {unidad}"
    return f"Precipitaci\u00f3n Acumulada {cantidad} {previo} (mm, somb.)"


# Escalas (mm) para precipitacion acumulada. 15 bordes -> 14 intervalos (igual
# que los 14 colores de PP10M_COLORS). extend='neither'.
PP1H_LEVELS = [0.1, 0.25, 0.5, 1, 2, 4, 6, 10, 15, 20, 30, 45, 60, 90, 120]
PP6H_LEVELS = [0.5, 1, 2, 5, 10, 15, 25, 35, 50, 70, 90, 120, 150, 200, 250]
PP24H_LEVELS = [1, 5, 10, 20, 30, 50, 75, 100, 130, 160, 200, 250, 300, 400, 500]


PRODUCTS: dict[str, Product] = {
    "10m_wind": Product(
        key="10m_wind",
        title="Viento a 10 metros / Wind at 10 m (Wind Barbs)",
        subtitle="WRF DET SMN 4 km",
        units_label="Velocidad del viento a 10 m [km/h]",
        cmap=_wind_cmap,
        levels=np.array(WIND_LEVELS),
        kind="contourf",
        extend="neither",
        field_fn=_wind_speed_kmh,
        barbs=True,
    ),
    "t2": Product(
        key="t2",
        title="Temperatura a 2 metros / 2 m Temperature",
        subtitle="WRF DET SMN 4 km",
        units_label="Temperatura [\u00b0C]",
        cmap=_temp_cmap,
        levels=np.arange(-20, 44, 2),
        kind="contourf",
        extend="both",
        field_fn=lambda ds: ds["T2"].isel(time=0).values,
        barbs=False,
    ),
    "hr2": Product(
        key="hr2",
        title="Humedad relativa a 2 m / 2 m Relative Humidity",
        subtitle="WRF DET SMN 4 km",
        units_label="Humedad relativa [%]",
        cmap=_rh_cmap,
        levels=np.arange(0, 105, 5),
        kind="contourf",
        extend="neither",
        field_fn=lambda ds: ds["HR2"].isel(time=0).values,
        barbs=False,
    ),
    "pp": Product(
        key="pp",
        title=build_accum_title(60),
        subtitle="WRF DET SMN 4 km",
        units_label="Precipitaci\u00f3n acumulada 1 h [mm]",
        cmap=_pp10m_cmap,
        levels=np.array(PP1H_LEVELS),
        kind="contourf",
        extend="neither",
        field_fn=lambda ds: ds["PP"].isel(time=0).values,
        barbs=False,
        freq="01H",
        period_min=60,
    ),
    "pp_6h": Product(
        key="pp_6h",
        title=build_accum_title(360),
        subtitle="WRF DET SMN 4 km",
        units_label="Precipitaci\u00f3n acumulada 6 h [mm]",
        cmap=_pp10m_cmap,
        levels=np.array(PP6H_LEVELS),
        kind="contourf",
        extend="neither",
        field_fn=None,             # se calcula por acumulacion (ver make_plot)
        barbs=False,
        freq="01H",
        accum_hours=6,
        period_min=360,
    ),
    "pp_24h": Product(
        key="pp_24h",
        title=build_accum_title(1440),
        subtitle="WRF DET SMN 4 km",
        units_label="Precipitaci\u00f3n acumulada 24 h [mm]",
        cmap=_pp10m_cmap,
        levels=np.array(PP24H_LEVELS),
        kind="contourf",
        extend="neither",
        field_fn=None,             # se calcula por acumulacion (ver make_plot)
        barbs=False,
        freq="01H",
        accum_hours=24,
        period_min=1440,
    ),
    "pp_10m": Product(
        key="pp_10m",
        title=(build_accum_title(10) if not PP10M_AS_RATE
               else "Precipitaci\u00f3n 10 min - intensidad (mm/h, somb.)"),
        subtitle="WRF DET SMN 4 km",
        units_label=("Intensidad de precipitaci\u00f3n [mm/h]" if PP10M_AS_RATE
                     else "Precipitaci\u00f3n acumulada en 10 min [mm]"),
        cmap=_pp10m_cmap,
        levels=np.array(PP10M_LEVELS_RATE if PP10M_AS_RATE else PP10M_LEVELS_MM),
        kind="contourf",
        extend="neither",
        field_fn=_pp_10min,
        barbs=False,
        freq="10M",
    ),
    "psfc": Product(
        key="psfc",
        title="Presi\u00f3n en superficie / Surface Pressure",
        subtitle="WRF DET SMN 4 km",
        units_label="Presi\u00f3n [hPa]",
        cmap=lambda: plt.get_cmap("viridis"),
        levels=np.arange(960, 1046, 2),
        kind="contourf",
        extend="both",
        field_fn=lambda ds: ds["PSFC"].isel(time=0).values,
        barbs=False,
    ),
}


# ---------------------------------------------------------------------------
# 4) DESCARGA DE DATOS DESDE EL BUCKET DE AWS
# ---------------------------------------------------------------------------

def build_key(date: str, cycle: str, lead: int, freq: str = "01H") -> str:
    """Construye la key S3 del archivo para una fecha/ciclo/plazo dados.

    date : 'YYYYMMDD'   cycle : 'HH' (00/06/12/18)   lead : plazo (int)
    """
    yyyy, mm, dd = date[:4], date[4:6], date[6:8]
    fname = f"WRFDETAR_{freq}_{date}_{cycle}_{lead:03d}.nc"
    return f"DATA/WRF/DET/{yyyy}/{mm}/{dd}/{cycle}/{fname}"


def download_file(date: str, cycle: str, lead: int, freq: str = "01H",
                  force: bool = False) -> str:
    """Descarga (con cache local) el .nc solicitado y devuelve la ruta local."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = build_key(date, cycle, lead, freq)
    url = f"{S3_BASE}/{key}"
    local = os.path.join(CACHE_DIR, os.path.basename(key))

    if os.path.exists(local) and not force and os.path.getsize(local) > 0:
        print(f"[cache] Usando archivo local: {local}")
        return local

    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        if r.status_code == 404:
            raise FileNotFoundError(
                f"No se encontro el archivo en el bucket:\n  {url}\n"
                "Verificar fecha/ciclo/plazo. Los ciclos disponibles suelen ser "
                "00, 06, 12 y 18 UTC y los plazos 000..072."
            )
        r.raise_for_status()
        tmp = local + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, local)
    print(f"[ok] Guardado en {local} ({os.path.getsize(local) / 1e6:.1f} MB)")
    return local


# ---------------------------------------------------------------------------
# 5) PROYECCION Y HERRAMIENTAS GEOGRAFICAS
# ---------------------------------------------------------------------------

def wrf_projection(ds: xr.Dataset) -> ccrs.LambertConformal:
    """Devuelve la proyeccion Lambert Conforme exacta usada por el WRF DET."""
    lc = ds["Lambert_Conformal"].attrs
    sp = lc.get("standard_parallel", [-35.0, -35.0])
    try:
        sp = (float(sp[0]), float(sp[1]))
    except TypeError:
        sp = (float(sp), float(sp))
    lon0 = float(lc.get("longitude_of_central_meridian", -65.0))
    lat0 = float(lc.get("latitude_of_projection_origin", -35.0))
    radius = float(lc.get("earth_radius", 6370000.0))
    globe = ccrs.Globe(ellipse=None, semimajor_axis=radius, semiminor_axis=radius)
    # cutoff=30: necesario para que la proyeccion funcione en el hemisferio sur.
    return ccrs.LambertConformal(
        central_longitude=lon0, central_latitude=lat0,
        standard_parallels=sp, globe=globe, cutoff=30,
    )


def _province_feature():
    """Limites de provincias/estados (sin etiquetas de ciudades)."""
    return cfeature.NaturalEarthFeature(
        category="cultural", name="admin_1_states_provinces_lines",
        scale="10m", facecolor="none",
    )


def _country_feature():
    """Limites nacionales."""
    return cfeature.NaturalEarthFeature(
        category="cultural", name="admin_0_boundary_lines_land",
        scale="10m", facecolor="none",
    )


# ---------------------------------------------------------------------------
# 6) FUNCION PRINCIPAL DE PLOTEO
# ---------------------------------------------------------------------------

def _barb_step_for_extent(extent, x, y, target=26):
    """Calcula cada cuantos puntos de grilla dibujar barbas para ~'target'
    barbas a lo ancho del dominio visible."""
    lon_min, lon_max, lat_min, lat_max = extent
    # aproximacion: cuantos puntos de grilla (4 km) entran en el ancho visible
    deg_width = lon_max - lon_min
    # ~ km por grado de longitud a lat media
    lat_mid = (lat_min + lat_max) / 2.0
    km_per_deg = 111.0 * np.cos(np.deg2rad(lat_mid))
    npts = max(1, int(deg_width * km_per_deg / 4.0))
    return max(1, npts // target)


def plot_product(ds: xr.Dataset, product: Product, field: np.ndarray,
                 region: str, init_time: datetime, valid_time: datetime,
                 out_path: str, title_override: str | None = None) -> str:
    """Genera el mapa para el producto/region indicados y lo guarda en disco."""

    proj = wrf_projection(ds)
    data_crs = ccrs.PlateCarree()

    if region not in REGIONS:
        raise KeyError(f"Region '{region}' no definida. Opciones: {list(REGIONS)}")
    extent = REGIONS[region]

    # ---- Figura y ejes -----------------------------------------------------
    fig = plt.figure(figsize=(11, 11.5))
    # Ejes del mapa: dejamos margen arriba para los titulos, a la derecha para
    # la barra de color y abajo para el pie con los enlaces.
    ax = fig.add_axes([0.03, 0.075, 0.83, 0.845], projection=proj)
    ax.set_extent(extent, crs=data_crs)

    # Fondo blanco del mapa
    ax.set_facecolor("white")

    # ---- Campo escalar sombreado ------------------------------------------
    x = ds["x"].values
    y = ds["y"].values
    cmap = product.cmap()
    levels = product.levels
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend=product.extend) \
        if isinstance(cmap, ListedColormap) else None

    cf = ax.contourf(
        x, y, field, levels=levels, cmap=cmap, norm=norm,
        extend=product.extend, transform=proj,
    )

    # ---- Barbas de viento (si aplica) -------------------------------------
    if product.barbs:
        lon = ds["lon"].values
        lat = ds["lat"].values
        mag = ds["magViento10"].isel(time=0).values      # m/s
        dirg = ds["dirViento10"].isel(time=0).values      # grados (from)
        dir_rad = np.deg2rad(dirg)
        # Componentes tierra-relativas (este/norte) en nudos.
        kt = 1.94384
        u = -mag * np.sin(dir_rad) * kt
        v = -mag * np.cos(dir_rad) * kt

        step = _barb_step_for_extent(extent, x, y)
        sl = (slice(None, None, step), slice(None, None, step))
        ax.barbs(
            lon[sl], lat[sl], u[sl], v[sl],
            length=5.5, linewidth=0.55, transform=data_crs,
            regrid_shape=None, zorder=6,
        )

    # ---- Limites politicos -------------------------------------------------
    # Costas y oceano suave
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                   linewidth=0.6, edgecolor="black", zorder=4)
    # Provincias: linea fina
    ax.add_feature(_province_feature(), linewidth=0.4,
                   edgecolor="#3a3a3a", zorder=4)
    # Fronteras nacionales: linea mas gruesa
    ax.add_feature(_country_feature(), linewidth=1.3,
                   edgecolor="black", zorder=5)

    # Recuadro (marco negro alrededor del mapa) - estilo MetPy
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.3)
        spine.set_visible(True)
    ax.set_frame_on(True)

    # ---- Barra de color al costado derecho --------------------------------
    cax = fig.add_axes([0.885, 0.12, 0.024, 0.72])
    cb = fig.colorbar(cf, cax=cax, orientation="vertical",
                      ticks=levels, extend=product.extend)
    cb.set_label(product.units_label, fontsize=11)
    cb.ax.tick_params(labelsize=9)

    # ---- Titulos (arriba del recuadro) ------------------------------------
    # Izquierda: producto / variable (estilo del ejemplo)
    fig.text(0.03, 0.965, title_override or product.title, ha="left",
             va="bottom", fontsize=13.5, fontweight="normal",
             family="sans-serif")
    fig.text(0.03, 0.937, product.subtitle, ha="left", va="bottom",
             fontsize=12, fontweight="normal", family="sans-serif")

    # Derecha: inicializacion y validez de la corrida
    init_str = init_time.strftime("Inic.: %Y-%m-%d %H:%M UTC")
    valid_str = valid_time.strftime("Val.:  %Y-%m-%d %H:%M UTC")
    fig.text(0.86, 0.965, init_str, ha="right", va="bottom",
             fontsize=12, family="sans-serif")
    fig.text(0.86, 0.937, valid_str, ha="right", va="bottom",
             fontsize=12, family="sans-serif")

    # ---- Pie de pagina con los enlaces del SMN ----------------------------
    foot = (f"Guia SMN: {DOC_URL}\n"
            f"Datos (AWS): {BUCKET_URL}")
    fig.text(0.03, 0.012, foot, ha="left", va="bottom",
             fontsize=8, color="#555555", family="sans-serif")

    # ---- Guardado ----------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"[plot] Imagen guardada en: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 7) ORQUESTACION
# ---------------------------------------------------------------------------

def accumulate_pp(date: str, cycle: str, lead: int, accum_hours: int,
                  force: bool = False):
    """Suma la precipitacion horaria (PP de archivos 01H) sobre las
    'accum_hours' horas que terminan en el plazo 'lead'.

    Devuelve (campo_sumado, ds_ultimo, archivos_usados) donde ds_ultimo se usa
    para las coordenadas/tiempo y archivos_usados es la lista de rutas locales.
    """
    first = lead - accum_hours + 1
    if first < 1:
        raise SystemExit(
            f"[error] Para acumular {accum_hours} h se necesita --lead >= "
            f"{accum_hours} (se pidio {lead}). Elija un plazo mayor.")
    total = None
    last_ds = None
    used = []
    print(f"[accum] Sumando PP horaria de los plazos {first:03d}..{lead:03d} "
          f"({accum_hours} h)")
    for l in range(first, lead + 1):
        f = download_file(date, cycle, l, freq="01H", force=force)
        used.append(f)
        ds = xr.open_dataset(f)
        pp = ds["PP"].isel(time=0).values
        total = pp.copy() if total is None else total + pp
        if l == lead:
            last_ds = ds
        else:
            ds.close()
    return total, last_ds, used


def _valid_time_from_ds(ds: xr.Dataset, init_time: datetime,
                        lead: int) -> datetime:
    """Obtiene el tiempo de validez desde la coordenada 'time' del dataset."""
    try:
        ts = np.datetime64(ds["time"].isel(time=0).values, "s")
        return datetime.utcfromtimestamp(ts.astype("O").timestamp())
    except Exception:
        return init_time + timedelta(hours=lead)


def cleanup_files(paths) -> None:
    """Borra del disco los archivos .nc indicados (usado por --cleanup).
    Solo toca los .nc del cache; nunca borra las imagenes de salida."""
    borrados, liberado = 0, 0
    for p in dict.fromkeys(paths):          # evita duplicados, conserva orden
        try:
            if p and os.path.exists(p):
                liberado += os.path.getsize(p)
                os.remove(p)
                borrados += 1
        except OSError as e:
            print(f"[cleanup] No se pudo borrar {p}: {e}")
    if borrados:
        print(f"[cleanup] Borrados {borrados} archivo(s) .nc del cache "
              f"({liberado / 1e6:.1f} MB liberados).")
    else:
        print("[cleanup] No habia archivos .nc para borrar.")


def make_plot(date: str, cycle: str, lead: int, var: str = "10m_wind",
              region: str = "argentina", force_download: bool = False,
              cleanup: bool = False) -> str:
    """Descarga los datos y genera el plot del producto/region pedidos.

    Si cleanup=True, borra del cache los archivos .nc usados por esta corrida
    despues de guardar la imagen (el .png de salida se conserva).
    """
    if var not in PRODUCTS:
        raise KeyError(f"Producto '{var}' no definido. Opciones: {list(PRODUCTS)}")
    product = PRODUCTS[var]

    init_time = datetime.strptime(f"{date}{cycle}", "%Y%m%d%H")

    if product.accum_hours:
        # Producto de acumulacion: sumar varios archivos horarios.
        field, ds, used_files = accumulate_pp(
            date, cycle, lead, product.accum_hours, force=force_download)
    else:
        # Producto de un solo archivo.
        local = download_file(date, cycle, lead, freq=product.freq,
                              force=force_download)
        used_files = [local]
        ds = xr.open_dataset(local)
        field = product.field_fn(ds)

    valid_time = _valid_time_from_ds(ds, init_time, lead)

    tag = f"{var}_{region}_{date}_{cycle}_f{lead:03d}"
    out_path = os.path.join(OUTPUT_DIR, f"WRFDET_{tag}.png")
    result = plot_product(ds, product, field, region, init_time, valid_time,
                          out_path)

    ds.close()  # liberar el handle del archivo antes de un eventual borrado
    if cleanup:
        cleanup_files(used_files)
    return result


def _assemble_gif(frame_paths, gif_path, fps=2):
    """Une una lista de PNG en un GIF animado (usa Pillow)."""
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    duration = int(round(1000.0 / max(fps, 0.1)))  # ms por frame
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=duration, loop=0, disposal=2)
    for im in imgs:
        im.close()
    print(f"[gif] Animacion guardada en: {gif_path} "
          f"({len(frame_paths)} frames, {fps} fps)")
    return gif_path


def make_gif(date: str, cycle: str, lead: int, var: str = "pp_6h",
             region: str = "argentina", gif_mode: str = "building",
             frames: int | None = None, fps: float = 2.0,
             force_download: bool = False, cleanup: bool = False) -> str:
    """Genera un GIF animado del producto.

    - Productos acumulados (pp_6h, pp_24h):
        * gif_mode='building': la lluvia se ACUMULA cuadro a cuadro (1 h, 2 h,
          ... hasta el total). El ultimo frame es el acumulado total que
          termina en el plazo --lead. Nro de frames = accum_hours.
        * gif_mode='rolling': ventana movil completa (accum_hours) terminando
          en plazos consecutivos. Nro de frames = --frames (o accum_hours).
    - Productos instantaneos (viento, T2, etc.): anima los plazos consecutivos;
      por defecto los plazos 1..--lead (o los ultimos --frames).
    """
    if var not in PRODUCTS:
        raise KeyError(f"Producto '{var}' no definido. Opciones: {list(PRODUCTS)}")
    product = PRODUCTS[var]
    init_time = datetime.strptime(f"{date}{cycle}", "%Y%m%d%H")

    frame_dir = os.path.join(OUTPUT_DIR, f"_frames_{var}_{region}_{date}_"
                             f"{cycle}_f{lead:03d}")
    os.makedirs(frame_dir, exist_ok=True)
    frame_paths, used_files = [], []

    def render(ds, field, valid_time, idx, title_override=None):
        fp = os.path.join(frame_dir, f"frame_{idx:03d}.png")
        plot_product(ds, product, field, region, init_time, valid_time, fp,
                     title_override=title_override)
        frame_paths.append(fp)

    if product.accum_hours:
        accum = product.accum_hours
        if gif_mode == "building":
            first = lead - accum + 1
            if first < 1:
                raise SystemExit(
                    f"[error] Para el GIF 'building' de {accum} h se necesita "
                    f"--lead >= {accum} (se pidio {lead}).")
            running = None
            for k, l in enumerate(range(first, lead + 1), start=1):
                f = download_file(date, cycle, l, freq="01H", force=force_download)
                used_files.append(f)
                ds = xr.open_dataset(f)
                pp = ds["PP"].isel(time=0).values
                running = pp.copy() if running is None else running + pp
                vt = _valid_time_from_ds(ds, init_time, l)
                render(ds, running.copy(), vt, k,
                       title_override=build_accum_title(k * 60))
                ds.close()
        else:  # rolling
            n = frames or accum
            end_first = lead - n + 1
            need_first = end_first - accum + 1
            if need_first < 1:
                raise SystemExit(
                    f"[error] Para el GIF 'rolling' ({n} frames de {accum} h) "
                    f"se necesita --lead >= {n + accum - 1} (se pidio {lead}).")
            pp_by_lead, vt_by_lead = {}, {}
            for l in range(need_first, lead + 1):
                f = download_file(date, cycle, l, freq="01H", force=force_download)
                used_files.append(f)
                ds = xr.open_dataset(f)
                pp_by_lead[l] = ds["PP"].isel(time=0).values
                vt_by_lead[l] = _valid_time_from_ds(ds, init_time, l)
                if l == lead:
                    ds_coords = ds
                else:
                    ds.close()
            for k, end in enumerate(range(end_first, lead + 1), start=1):
                win = sum(pp_by_lead[l] for l in range(end - accum + 1, end + 1))
                render(ds_coords, win, vt_by_lead[end], k)
            ds_coords.close()
    else:
        # Producto instantaneo: animar plazos consecutivos.
        n = frames or lead
        start = max(1, lead - n + 1)
        for k, l in enumerate(range(start, lead + 1), start=1):
            f = download_file(date, cycle, l, freq=product.freq,
                              force=force_download)
            used_files.append(f)
            ds = xr.open_dataset(f)
            field = product.field_fn(ds)
            vt = _valid_time_from_ds(ds, init_time, l)
            render(ds, field, vt, k)
            ds.close()

    if not frame_paths:
        raise SystemExit("[error] No se genero ningun frame para el GIF.")

    gif_tag = gif_mode if product.accum_hours else "anim"
    gif_path = os.path.join(
        OUTPUT_DIR, f"WRFDET_{var}_{region}_{date}_{cycle}_{gif_tag}_"
        f"f{lead:03d}.gif")
    _assemble_gif(frame_paths, gif_path, fps=fps)

    # Limpieza: los PNG de frames son temporales; los .nc solo si --cleanup.
    for fp in frame_paths:
        try:
            os.remove(fp)
        except OSError:
            pass
    try:
        os.rmdir(frame_dir)
    except OSError:
        pass
    if cleanup:
        cleanup_files(used_files)
    return gif_path


def _parse_args():
    p = argparse.ArgumentParser(
        description="Ploteo del WRF DET del SMN (4 km) estilo MetPy.")
    p.add_argument("--date", default="20260630", help="Fecha del ciclo YYYYMMDD")
    p.add_argument("--cycle", default="00", help="Ciclo/hora de inicio HH (00/06/12/18)")
    p.add_argument("--lead", type=int, default=4, help="Plazo de pronostico (horas)")
    p.add_argument("--var", default="10m_wind", choices=list(PRODUCTS),
                   help="Producto/variable a plotear")
    p.add_argument("--region", default="argentina",
                   help="Region de zoom (ej: nea_misiones, cuyo_mendoza, "
                        "patagonia). Use --list-regions para ver todas.")
    p.add_argument("--list-regions", action="store_true",
                   help="Lista todas las regiones disponibles y termina.")
    p.add_argument("--list", "--list-vars", dest="list_vars",
                   action="store_true",
                   help="Lista todos los productos/variables disponibles y termina.")
    p.add_argument("--force", action="store_true", help="Forzar re-descarga")
    p.add_argument("--cleanup", action="store_true",
                   help="Borra del cache los archivos .nc usados por esta "
                        "corrida al terminar (conserva el .png de salida).")
    # ---- Modo GIF ----
    p.add_argument("--gif", action="store_true",
                   help="Genera un GIF animado en lugar de una imagen unica.")
    p.add_argument("--gif-mode", dest="gif_mode", default="building",
                   choices=["building", "rolling"],
                   help="Solo para productos acumulados. 'building': la lluvia "
                        "se acumula cuadro a cuadro hasta el total (default). "
                        "'rolling': ventana movil de igual duracion.")
    p.add_argument("--frames", type=int, default=None,
                   help="Nro de frames del GIF (opcional). Por defecto: "
                        "accum_hours en acumulados, o --lead en instantaneos.")
    p.add_argument("--fps", type=float, default=2.0,
                   help="Velocidad del GIF en cuadros por segundo (default 2).")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.list_vars:
        list_products()
        return
    if args.list_regions:
        list_regions()
        return
    if args.region not in REGIONS:
        raise SystemExit(
            f"[error] Region '{args.region}' no existe. "
            f"Use --list-regions para ver las {len(REGIONS)} disponibles.")
    note = _font_note()
    if note:
        print(note)
    if args.gif:
        make_gif(args.date, args.cycle, args.lead, args.var, args.region,
                 args.gif_mode, args.frames, args.fps, args.force, args.cleanup)
    else:
        make_plot(args.date, args.cycle, args.lead, args.var, args.region,
                  args.force, args.cleanup)


if __name__ == "__main__":
    main()
