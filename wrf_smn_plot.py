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

REGIONS: dict[str, tuple[float, float, float, float]] = {
    # Vista completa del dominio / toda Argentina
    "argentina":   (-76.0, -52.0, -56.0, -21.0),
    # Regiones
    "norte":       (-70.0, -57.0, -30.0, -21.0),
    "noa":         (-70.0, -62.0, -31.0, -21.0),   # Noroeste argentino
    "nea":         (-63.0, -53.0, -31.0, -22.0),   # Noreste argentino / Litoral
    "centro":      (-69.0, -57.0, -39.0, -29.0),   # Centro (Cordoba, Pampa)
    "cuyo":        (-71.0, -65.0, -37.0, -28.0),
    "buenos_aires":(-63.5, -56.0, -41.5, -33.0),
    "amba":        (-59.6, -57.7, -35.4, -34.0),   # Area Metropolitana Bs. As.
    "patagonia":   (-75.0, -62.0, -56.0, -38.0),
    "patagonia_n": (-72.0, -62.0, -44.0, -36.0),
    "sur":         (-75.0, -63.0, -56.0, -49.0),
    "cuyo_centro": (-71.0, -62.0, -38.0, -28.0),
}


def add_region(name: str, lon_min: float, lon_max: float,
               lat_min: float, lat_max: float) -> None:
    """Agrega o redefine una region de zoom en tiempo de ejecucion."""
    REGIONS[name] = (lon_min, lon_max, lat_min, lat_max)


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


def _wind_speed_knots(ds: xr.Dataset) -> np.ndarray:
    mag = ds["magViento10"].isel(time=0).values  # m/s
    return mag * 1.94384  # a nudos


def _wind_speed_kmh(ds: xr.Dataset) -> np.ndarray:
    mag = ds["magViento10"].isel(time=0).values  # m/s
    return mag * 3.6  # a km/h


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
        title="Precipitaci\u00f3n horaria / Hourly Precipitation",
        subtitle="WRF DET SMN 4 km",
        units_label="Precipitaci\u00f3n [mm]",
        cmap=_pp_cmap,
        levels=np.array([0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 60, 80, 120]),
        kind="contourf",
        extend="max",
        field_fn=lambda ds: ds["PP"].isel(time=0).values,
        barbs=False,
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


def plot_product(ds: xr.Dataset, product: Product, region: str,
                 init_time: datetime, valid_time: datetime,
                 out_path: str) -> str:
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
    field = product.field_fn(ds)
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
    fig.text(0.03, 0.965, product.title, ha="left", va="bottom",
             fontsize=13.5, fontweight="normal", family="sans-serif")
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

def make_plot(date: str, cycle: str, lead: int, var: str = "10m_wind",
              region: str = "argentina", force_download: bool = False) -> str:
    """Descarga los datos y genera el plot del producto/region pedidos."""
    if var not in PRODUCTS:
        raise KeyError(f"Producto '{var}' no definido. Opciones: {list(PRODUCTS)}")
    product = PRODUCTS[var]

    local = download_file(date, cycle, lead, force=force_download)
    ds = xr.open_dataset(local)

    # Tiempo de inicializacion (ciclo) y de validez (coordenada 'time').
    init_time = datetime.strptime(f"{date}{cycle}", "%Y%m%d%H")
    try:
        valid_time = np.datetime64(ds["time"].isel(time=0).values, "s").item()
        if not isinstance(valid_time, datetime):
            valid_time = datetime.utcfromtimestamp(
                np.datetime64(ds["time"].isel(time=0).values, "s").astype("O").timestamp()
            )
    except Exception:
        valid_time = init_time + timedelta(hours=lead)

    tag = f"{var}_{region}_{date}_{cycle}_f{lead:03d}"
    out_path = os.path.join(OUTPUT_DIR, f"WRFDET_{tag}.png")
    return plot_product(ds, product, region, init_time, valid_time, out_path)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Ploteo del WRF DET del SMN (4 km) estilo MetPy.")
    p.add_argument("--date", default="20260630", help="Fecha del ciclo YYYYMMDD")
    p.add_argument("--cycle", default="00", help="Ciclo/hora de inicio HH (00/06/12/18)")
    p.add_argument("--lead", type=int, default=4, help="Plazo de pronostico (horas)")
    p.add_argument("--var", default="10m_wind", choices=list(PRODUCTS),
                   help="Producto/variable a plotear")
    p.add_argument("--region", default="argentina",
                   help=f"Region de zoom. Opciones: {list(REGIONS)}")
    p.add_argument("--force", action="store_true", help="Forzar re-descarga")
    return p.parse_args()


def main():
    args = _parse_args()
    note = _font_note()
    if note:
        print(note)
    make_plot(args.date, args.cycle, args.lead, args.var, args.region, args.force)


if __name__ == "__main__":
    main()
