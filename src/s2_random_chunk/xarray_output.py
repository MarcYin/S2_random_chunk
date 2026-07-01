"""Build xarray outputs with item and chunk metadata."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from .chunks import AssetChunkInfo, ChunkWindow, bounds_for_window
from .config import RandomChunkRequest
from .providers import ProviderConfig
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
)


def build_dataset(
    data: np.ndarray,
    items: Sequence[Any],
    asset_keys: Sequence[str],
    asset_hrefs: Sequence[Sequence[str]],
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
    coords.update(
        {
            "chunk_left": ("chunk", bounds[:, 0]),
            "chunk_bottom": ("chunk", bounds[:, 1]),
            "chunk_right": ("chunk", bounds[:, 2]),
            "chunk_top": ("chunk", bounds[:, 3]),
        }
    )

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
                    "nodata": chunk_info.nodata,
                },
            ),
            "asset_href": (
                ("time", "band"),
                np.asarray(sanitized_hrefs, dtype=object),
                {"description": "Asset URLs with query strings removed"},
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
