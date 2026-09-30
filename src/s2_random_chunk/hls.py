"""Random chunk sampling of NASA HLS v2.0 (L30 + S30) time series.

Extends the Sentinel-2 random chunk sampler to the Harmonized Landsat
Sentinel-2 (HLS) v2.0 collections hosted on Microsoft Planetary Computer
(``hls2-l30`` and ``hls2-s30``). Both sensors share the same 30 m MGRS tile
grid (3660 x 3660 pixels), so one set of random block-aligned chunk windows
is reused across every L30 and S30 acquisition, producing a unified
two-sensor time-series tensor per chunk.

The band orders below intentionally follow the within-season crop type
mapper convention (Zhang et al. 2025, ``HLS_io_chunks.py``):

- L30: B01 coastal, B02 blue, B03 green, B04 red, B05 NIR, B06 SWIR1,
  B07 SWIR2 (+ Fmask appended as the last band).
- S30: B01 coastal, B02 blue, B03 green, B04 red, B8A NIR-narrow,
  B11 SWIR1, B12 SWIR2, B05 red-edge1, B06 red-edge2, B07 red-edge3,
  B08 NIR-broad (+ Fmask appended as the last band).

Values are returned as raw digital numbers (reflectance DN, scale 0.0001,
fill -9999; Fmask is a uint8 bit field, fill 255) so downstream code can
apply classifier-specific scaling/normalization.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .chunks import (
    bounds_for_window,
    inspect_remote_geotiff,
    list_block_windows,
    pixel_size,
    select_random_windows,
)
from .providers import ProviderConfig
from .reader import read_chunks_threaded
from .stac import asset_href, item_id_value
from .utils import datetime_interval, normalize_mgrs_tile
from .xarray_output import coordinate_grids


HLS_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"

HLS_L30_COLLECTION = "hls2-l30"
HLS_S30_COLLECTION = "hls2-s30"

# Reflectance band order matches the in-season crop type mapper convention.
HLS_L30_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07")
HLS_S30_BANDS = ("B01", "B02", "B03", "B04", "B8A", "B11", "B12", "B05", "B06", "B07", "B08")

HLS_QA_BAND = "Fmask"
HLS_FILL = -9999
HLS_REFLECTANCE_SCALE = 0.0001

# HLS granule ids look like HLS.L30.T15TVG.2023152T165256.v2.0
_HLS_ID_RE = re.compile(
    r"HLS\.(?P<sensor>[LS]30)\.T(?P<tile>[0-9A-Z]{5})\.(?P<year>\d{4})(?P<doy>\d{3})T\d{6}"
)


def parse_hls_id(item_id: str) -> Optional[dict]:
    """Extract sensor, tile, year and day-of-year from an HLS granule id."""

    match = _HLS_ID_RE.search(item_id)
    if not match:
        return None
    return {
        "sensor": match.group("sensor"),
        "tile": match.group("tile"),
        "year": int(match.group("year")),
        "doy": int(match.group("doy")),
    }


def _open_client():
    try:
        from pystac_client import Client
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pystac-client is required to search STAC providers") from exc
    try:
        import planetary_computer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("planetary-computer is required to sign HLS asset URLs") from exc
    return Client.open(HLS_ENDPOINT, modifier=planetary_computer.sign_inplace)


def tile_centroid(mgrs_tile: str, client: Optional[Any] = None) -> Tuple[float, float]:
    """Resolve an MGRS tile's centroid (lon, lat) via a Sentinel-2 L2A lookup.

    HLS items on Planetary Computer expose no queryable MGRS property, so we
    resolve the tile footprint from the sentinel-2-l2a collection (any recent
    item) and search HLS collections spatially.
    """

    normalized = normalize_mgrs_tile(mgrs_tile)
    client = client or _open_client()
    search = client.search(
        collections=["sentinel-2-l2a"],
        query={"s2:mgrs_tile": {"eq": normalized}},
        max_items=1,
    )
    items = list(search.items())
    if not items:
        raise ValueError(f"Could not resolve MGRS tile {normalized!r} from sentinel-2-l2a")
    bbox = items[0].bbox
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def search_hls_items(
    mgrs_tile: str,
    start_date: str,
    end_date: str,
    client: Optional[Any] = None,
    collections: Sequence[str] = (HLS_L30_COLLECTION, HLS_S30_COLLECTION),
) -> List[Any]:
    """Search HLS L30+S30 items for one MGRS tile and date range.

    Items are found with a point-intersects search at the tile centroid and
    then filtered by granule id, because the Planetary Computer HLS items do
    not carry MGRS properties.
    """

    normalized = normalize_mgrs_tile(mgrs_tile)
    client = client or _open_client()
    lon, lat = tile_centroid(normalized, client)
    search = client.search(
        collections=list(collections),
        datetime=datetime_interval(start_date, end_date),
        intersects={"type": "Point", "coordinates": [lon, lat]},
        max_items=None,
    )
    wanted = f".T{normalized}."
    items = [item for item in search.items() if wanted in item_id_value(item)]
    items.sort(key=item_id_value)
    return items


def fetch_hls_random_chunks(
    mgrs_tile: str,
    start_date: str,
    end_date: str,
    n_chunks: int = 5,
    seed: Optional[int] = None,
    max_workers: int = 16,
    allow_partial_edges: bool = False,
    gdal_env: Optional[Mapping[str, str]] = None,
    items: Optional[Sequence[Any]] = None,
    chunks: Optional[Sequence[Any]] = None,
    on_error: str = "raise",
    windows_selector=None,
):
    """Fetch chunk-aligned HLS L30+S30 time series as an xarray Dataset.

    If ``chunks`` (a sequence of :class:`~s2_random_chunk.chunks.ChunkWindow`)
    is given, those exact block windows are fetched (skipping the random
    selection) — used to walk *every* block of a tile deterministically for
    dense full-tile mapping. Otherwise ``n_chunks`` random full-size blocks are
    selected as before.

    ``on_error='drop'`` skips scenes with an unreadable asset (corrupt archive
    granule, expired signed URL) instead of failing the whole fetch — see
    :func:`~s2_random_chunk.reader.read_chunks_threaded`. The dropped scene's
    time coordinate simply disappears from the output.

    Returns a Dataset with:

    - ``l30(time_l30, chunk, band_l30, y, x)`` raw DN float32 (last band Fmask)
    - ``s30(time_s30, chunk, band_s30, y, x)`` raw DN float32 (last band Fmask)
    - per-time ``year``/``doy``/``item_id`` coordinates for both sensors
    - chunk window offsets, bounds, projected and WGS84 coordinate grids
    - CRS/transform metadata shared by both sensors (same 30 m tile grid)
    """

    import xarray as xr

    normalized = normalize_mgrs_tile(mgrs_tile)
    if items is None:
        items = search_hls_items(normalized, start_date, end_date)
    if not items:
        raise ValueError(
            f"No HLS items found for tile {normalized} in {start_date}/{end_date}"
        )

    l30_items = [i for i in items if parse_hls_id(item_id_value(i)) and parse_hls_id(item_id_value(i))["sensor"] == "L30"]
    s30_items = [i for i in items if parse_hls_id(item_id_value(i)) and parse_hls_id(item_id_value(i))["sensor"] == "S30"]

    reference_item = (l30_items or s30_items)[0]
    reference_href = asset_href(reference_item, "B02")
    chunk_info = inspect_remote_geotiff(reference_href, gdal_env=gdal_env)
    if chunks is not None:
        chunks = tuple(chunks)
        if not chunks:
            raise ValueError("chunks was provided but is empty")
    else:
        windows = list_block_windows(reference_href, gdal_env=gdal_env)
        if windows_selector is not None:
            chunks = tuple(windows_selector(windows, chunk_info))
            if not chunks:
                raise ValueError("windows_selector returned no chunk windows")
        else:
            chunks = select_random_windows(
                windows,
                n_chunks=n_chunks,
                seed=seed,
                full_shape=chunk_info.reference_block_shape,
                allow_partial_edges=allow_partial_edges,
            )

    reference_resolution = max(pixel_size(chunk_info.transform))

    sensor_arrays = {}
    for sensor_key, sensor_items, bands in (
        ("l30", l30_items, HLS_L30_BANDS),
        ("s30", s30_items, HLS_S30_BANDS),
    ):
        if not sensor_items:
            sensor_arrays[sensor_key] = None
            continue
        asset_keys = tuple(bands) + (HLS_QA_BAND,)
        hrefs = [[asset_href(item, key) for key in asset_keys] for item in sensor_items]
        if on_error == "drop":
            data, kept = read_chunks_threaded(
                hrefs,
                chunks,
                scales=None,  # raw digital numbers
                max_workers=max_workers,
                masked=False,
                gdal_env=gdal_env,
                on_error="drop",
            )
            # keep the item list aligned with the (possibly reduced) time axis
            sensor_items = [sensor_items[i] for i in kept]
            if not sensor_items:
                sensor_arrays[sensor_key] = None
                continue
        else:
            data = read_chunks_threaded(
                hrefs,
                chunks,
                scales=None,  # raw digital numbers
                max_workers=max_workers,
                masked=False,
                gdal_env=gdal_env,
            )
        sensor_arrays[sensor_key] = (data.astype(np.float32), asset_keys, sensor_items)

    bounds = np.asarray([bounds_for_window(chunk_info.transform, chunk) for chunk in chunks])
    height, width = chunk_info.reference_block_shape
    projected_x, projected_y, longitude, latitude = coordinate_grids(
        chunk_info.transform,
        chunks,
        height,
        width,
        chunk_info.crs,
        reference_resolution,
        reference_resolution,
    )

    coords = {
        "chunk": ("chunk", np.arange(len(chunks))),
        "y": ("y", np.arange(height)),
        "x": ("x", np.arange(width)),
        "chunk_row_off": ("chunk", np.asarray([c.row_off for c in chunks], dtype=np.int64)),
        "chunk_col_off": ("chunk", np.asarray([c.col_off for c in chunks], dtype=np.int64)),
        "chunk_left": ("chunk", bounds[:, 0]),
        "chunk_bottom": ("chunk", bounds[:, 1]),
        "chunk_right": ("chunk", bounds[:, 2]),
        "chunk_top": ("chunk", bounds[:, 3]),
        "projected_x": (("chunk", "y", "x"), projected_x),
        "projected_y": (("chunk", "y", "x"), projected_y),
    }
    if longitude is not None and latitude is not None:
        coords["longitude"] = (("chunk", "y", "x"), longitude)
        coords["latitude"] = (("chunk", "y", "x"), latitude)

    data_vars = {}
    for sensor_key in ("l30", "s30"):
        entry = sensor_arrays[sensor_key]
        if entry is None:
            continue
        data, asset_keys, sensor_items = entry
        time_dim = f"time_{sensor_key}"
        band_dim = f"band_{sensor_key}"
        ids = [item_id_value(item) for item in sensor_items]
        parsed = [parse_hls_id(item_id) for item_id in ids]
        coords[f"{sensor_key}_item_id"] = (time_dim, np.asarray(ids, dtype=object))
        coords[f"{sensor_key}_year"] = (time_dim, np.asarray([p["year"] for p in parsed], dtype=np.int32))
        coords[f"{sensor_key}_doy"] = (time_dim, np.asarray([p["doy"] for p in parsed], dtype=np.int32))
        coords[band_dim] = (band_dim, np.asarray(asset_keys, dtype=object))
        data_vars[sensor_key] = (
            (time_dim, "chunk", band_dim, "y", "x"),
            data,
            {
                "long_name": f"HLS {sensor_key.upper()} raw digital numbers (+Fmask)",
                "reflectance_scale": HLS_REFLECTANCE_SCALE,
                "nodata": HLS_FILL,
                "qa_band": HLS_QA_BAND,
            },
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "title": "Random HLS v2.0 L30+S30 GeoTIFF chunk samples",
            "source": "planetary_computer",
            "stac_endpoint": HLS_ENDPOINT,
            "collections": json.dumps([HLS_L30_COLLECTION, HLS_S30_COLLECTION]),
            "requested_mgrs_tile": normalized,
            "requested_start_date": str(start_date),
            "requested_end_date": str(end_date),
            "n_chunks": len(chunks),
            "seed": seed if seed is not None else "",
            "reference_block_shape": tuple(chunk_info.reference_block_shape),
            "reference_resolution_m": reference_resolution,
            "crs": chunk_info.crs or "",
            "transform": tuple(chunk_info.transform),
        },
    )
    return ds


def hls_fmask_valid(fmask: np.ndarray) -> np.ndarray:
    """Good-quality mask following the in-season crop mapper convention.

    A pixel is invalid when it is fill (255) or any of Fmask bits
    1 (cloud), 2 (adjacent to cloud/shadow), 3 (cloud shadow), or
    4 (snow/ice) is set. Water (bit 5) is kept.
    """

    qa = fmask.astype(np.uint8)
    is_fill = qa == 255
    is_cloud = (qa >> 1) & 1
    is_adjacent = (qa >> 2) & 1
    is_shadow = (qa >> 3) & 1
    is_snow = (qa >> 4) & 1
    invalid = is_fill | (is_cloud == 1) | (is_adjacent == 1) | (is_shadow == 1) | (is_snow == 1)
    return ~invalid
