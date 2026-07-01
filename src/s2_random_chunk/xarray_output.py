"""Build xarray outputs with item and chunk metadata."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from .chunks import AssetChunkInfo, ChunkWindow, bounds_for_window
from .config import RandomChunkRequest
from .providers import ProviderConfig
from .scaling import AssetScale, scale_metadata_for_attrs
from .stac import item_collection_value, item_id_value, item_mgrs_tile, item_properties
from .utils import item_datetime, strip_url_query


COMMON_PROPERTY_COORDS = (
    "eo:cloud_cover",
    "s2:mgrs_tile",
    "grid:code",
    "mgrs:utm_zone",
    "mgrs:latitude_band",
    "mgrs:grid_square",
    "platform",
    "constellation",
    "s2:processing_baseline",
    "earthsearch:boa_offset_applied",
)


def build_dataset(
    data: np.ndarray,
    items: Sequence[Any],
    asset_keys: Sequence[str],
    asset_hrefs: Sequence[Sequence[str]],
    scales: Sequence[Sequence[AssetScale]],
    chunks: Sequence[ChunkWindow],
    chunk_info: AssetChunkInfo,
    provider: ProviderConfig,
    request: RandomChunkRequest,
    reference_asset: str,
) -> xr.Dataset:
    """Create an xarray Dataset from sampled data and STAC metadata."""

    if data.ndim != 5:
        raise ValueError("data must have dimensions time, chunk, band, y, x")

    times = [_datetime64_for_item(item) for item in items]
    coords = {
        "time": ("time", np.asarray(times)),
        "chunk": ("chunk", np.arange(len(chunks))),
        "band": ("band", np.asarray(tuple(asset_keys), dtype=object)),
        "y": ("y", np.arange(data.shape[-2])),
        "x": ("x", np.arange(data.shape[-1])),
        "item_id": ("time", np.asarray([item_id_value(item) for item in items], dtype=object)),
        "collection": (
            "time",
            np.asarray([item_collection_value(item) for item in items], dtype=object),
        ),
        "mgrs_tile": (
            "time",
            np.asarray([item_mgrs_tile(item) or request.mgrs_tile for item in items], dtype=object),
        ),
        "chunk_row_off": ("chunk", np.asarray([chunk.row_off for chunk in chunks], dtype=np.int64)),
        "chunk_col_off": ("chunk", np.asarray([chunk.col_off for chunk in chunks], dtype=np.int64)),
        "chunk_height": ("chunk", np.asarray([chunk.height for chunk in chunks], dtype=np.int64)),
        "chunk_width": ("chunk", np.asarray([chunk.width for chunk in chunks], dtype=np.int64)),
    }

    bounds = np.asarray([bounds_for_window(chunk_info.transform, chunk) for chunk in chunks])
    projected_x, projected_y, longitude, latitude = coordinate_grids(
        chunk_info.transform,
        chunks,
        data.shape[-2],
        data.shape[-1],
        chunk_info.crs,
    )
    coords.update(
        {
            "chunk_left": ("chunk", bounds[:, 0]),
            "chunk_bottom": ("chunk", bounds[:, 1]),
            "chunk_right": ("chunk", bounds[:, 2]),
            "chunk_top": ("chunk", bounds[:, 3]),
            "projected_x": (("chunk", "y", "x"), projected_x),
            "projected_y": (("chunk", "y", "x"), projected_y),
        }
    )
    if longitude is not None and latitude is not None:
        coords["longitude"] = (("chunk", "y", "x"), longitude)
        coords["latitude"] = (("chunk", "y", "x"), latitude)
    coordinate_names = "projected_x projected_y"
    if longitude is not None and latitude is not None:
        coordinate_names += " longitude latitude"

    properties_by_item = [item_properties(item) for item in items]
    for key in COMMON_PROPERTY_COORDS:
        values = [props.get(key) for props in properties_by_item]
        if any(value is not None for value in values):
            coord_name = key.replace(":", "_")
            coords[coord_name] = ("time", np.asarray(values, dtype=object))

    sanitized_hrefs = [
        [strip_url_query(href) for href in item_hrefs]
        for item_hrefs in asset_hrefs
    ]

    ds = xr.Dataset(
        data_vars={
            "data": (
                ("time", "chunk", "band", "y", "x"),
                data,
                {
                    "long_name": "Sampled Sentinel-2 asset pixels",
                    "source_band_index": request.band_index,
                    "scale_mode": request.scale_mode,
                    "units": data_units(scales),
                    "grid_mapping": "spatial_ref",
                    "coordinates": coordinate_names,
                    "nodata": chunk_info.nodata,
                },
            ),
            "asset_href": (
                ("time", "band"),
                np.asarray(sanitized_hrefs, dtype=object),
                {"description": "Asset URLs with query strings removed"},
            ),
            "scale": (
                ("time", "band"),
                np.asarray([[scale.scale for scale in item_scales] for item_scales in scales]),
                {"description": "Multiplicative scale applied to stored values"},
            ),
            "offset": (
                ("time", "band"),
                np.asarray([[scale.offset for scale in item_scales] for item_scales in scales]),
                {"description": "Additive offset applied after scale"},
            ),
            "spatial_ref": (
                (),
                0,
                {
                    "crs": chunk_info.crs or "",
                    "spatial_ref": chunk_info.crs or "",
                    "GeoTransform": " ".join(str(value) for value in chunk_info.transform),
                },
            ),
        },
        coords=coords,
        attrs={
            "title": "Random Sentinel-2 GeoTIFF chunk samples",
            "source": provider.name,
            "stac_endpoint": provider.endpoint,
            "collection": provider.collection,
            "requested_mgrs_tile": request.mgrs_tile,
            "requested_start_date": str(request.start_date),
            "requested_end_date": str(request.end_date),
            "n_chunks": request.n_chunks,
            "seed": request.seed if request.seed is not None else "",
            "reference_asset": reference_asset,
            "scale_mode": request.scale_mode,
            "scale_metadata": json.dumps(scale_metadata_for_attrs(scales)),
            "reference_block_shape": tuple(chunk_info.reference_block_shape),
            "reference_dtype": chunk_info.dtype,
            "reference_nodata": chunk_info.nodata if chunk_info.nodata is not None else "",
            "crs": chunk_info.crs or "",
            "transform": tuple(chunk_info.transform),
            "tiled": chunk_info.tiled,
            "overviews": tuple(chunk_info.overviews),
            "stac_items": json.dumps(_minimal_items(items, properties_by_item)),
        },
    )
    return ds


def coordinate_grids(
    transform: Sequence[float],
    chunks: Sequence[ChunkWindow],
    height: int,
    width: int,
    crs: str,
):
    a, b, c, d, e, f = transform
    projected_x = np.empty((len(chunks), height, width), dtype=np.float64)
    projected_y = np.empty((len(chunks), height, width), dtype=np.float64)
    row_template, col_template = np.meshgrid(
        np.arange(height, dtype=np.float64) + 0.5,
        np.arange(width, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    for chunk_index, chunk in enumerate(chunks):
        rows = row_template + chunk.row_off
        cols = col_template + chunk.col_off
        projected_x[chunk_index] = c + cols * a + rows * b
        projected_y[chunk_index] = f + cols * d + rows * e

    longitude = None
    latitude = None
    if crs:
        try:
            from rasterio.warp import transform as warp_transform

            lon, lat = warp_transform(
                crs,
                "EPSG:4326",
                projected_x.reshape(-1).tolist(),
                projected_y.reshape(-1).tolist(),
            )
            longitude = np.asarray(lon, dtype=np.float64).reshape(projected_x.shape)
            latitude = np.asarray(lat, dtype=np.float64).reshape(projected_y.shape)
        except Exception:
            longitude = None
            latitude = None
    return projected_x, projected_y, longitude, latitude


def data_units(scales: Sequence[Sequence[AssetScale]]) -> str:
    units = {
        scale.units
        for item_scales in scales
        for scale in item_scales
        if scale.applied
    }
    if len(units) == 1:
        return next(iter(units))
    if units:
        return "mixed"
    return "digital_number"


def _minimal_items(items: Sequence[Any], properties_by_item: Sequence[Mapping[str, Any]]):
    result = []
    for item, props in zip(items, properties_by_item):
        result.append(
            {
                "id": item_id_value(item),
                "collection": item_collection_value(item),
                "properties": {
                    key: _json_safe(value)
                    for key, value in props.items()
                    if _is_scalar_json_value(value)
                },
            }
        )
    return result


def _datetime64_for_item(item: Any) -> np.datetime64:
    dt = item_datetime(item)
    if dt is None:
        return np.datetime64("NaT", "ns")
    return np.datetime64(dt.replace(tzinfo=None), "ns")


def _is_scalar_json_value(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
