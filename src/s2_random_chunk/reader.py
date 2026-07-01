"""Threaded Rasterio window reads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .chunks import ChunkWindow, bounds_for_window, merged_gdal_env
from .scaling import AssetScale


@dataclass(frozen=True)
class AssetReadTask:
    """Read all sampled chunks for one time/asset from one GeoTIFF open."""

    time_index: int
    band_index: int
    href: str
    chunks: Sequence[ChunkWindow]
    scale: AssetScale
    source_band_index: int
    masked: bool
    fill_value: Optional[float]
    gdal_env: Optional[Mapping[str, str]]
    reference_transform: Optional[Tuple[float, float, float, float, float, float]] = None
    reference_resolution: Optional[float] = None
    target_resolution: Optional[float] = None


@dataclass(frozen=True)
class ReadResult:
    time_index: int
    band_index: int
    arrays: Tuple[np.ndarray, ...]


def read_asset_chunks(task: AssetReadTask) -> ReadResult:
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window, from_bounds
    except ImportError as exc:  # pragma: no cover - exercised when dependency is absent
        raise ImportError("rasterio is required to read remote GeoTIFF chunks") from exc

    resample = task.target_resolution is not None
    arrays = []
    with rasterio.Env(**merged_gdal_env(task.gdal_env)):
        with rasterio.open(task.href, sharing=False) as src:
            nodata = task.scale.nodata if task.scale.nodata is not None else src.nodata
            resampling = None
            if resample:
                native_resolution = max(abs(src.transform.a), abs(src.transform.e))
                resampling = _resampling_for(native_resolution, task.target_resolution)
            for chunk in task.chunks:
                out_shape = None
                if resample:
                    bounds = bounds_for_window(task.reference_transform, chunk)
                    rasterio_window = from_bounds(*bounds, transform=src.transform)
                    out_shape = _output_shape(chunk, task.reference_resolution, task.target_resolution)
                else:
                    rasterio_window = Window(
                        col_off=chunk.col_off,
                        row_off=chunk.row_off,
                        width=chunk.width,
                        height=chunk.height,
                    )
                array = src.read(
                    task.source_band_index,
                    window=rasterio_window,
                    out_shape=out_shape,
                    resampling=resampling if resampling is not None else Resampling.nearest,
                    masked=task.masked or task.scale.applied,
                )
                arrays.append(_finalize_array(array, task.scale, nodata, task.fill_value, task.masked))
    return ReadResult(
        time_index=task.time_index,
        band_index=task.band_index,
        arrays=tuple(arrays),
    )


def _resampling_for(native_resolution: float, target_resolution: float):
    from rasterio.enums import Resampling

    if target_resolution < native_resolution:
        # Output pixels are finer than the source: nearest neighbour avoids
        # inventing new interpolated values.
        return Resampling.nearest
    if target_resolution > native_resolution:
        # Output pixels are coarser than the source: average downsamples
        # multiple source pixels into one.
        return Resampling.average
    return Resampling.nearest


def _output_shape(
    chunk: ChunkWindow,
    reference_resolution: float,
    target_resolution: float,
) -> Tuple[int, int]:
    ratio = reference_resolution / target_resolution
    height = max(1, round(chunk.height * ratio))
    width = max(1, round(chunk.width * ratio))
    return (height, width)


def _finalize_array(
    array,
    scale: AssetScale,
    nodata: Optional[float],
    fill_value: Optional[float],
    masked: bool,
) -> np.ndarray:
    if scale.applied:
        if np.ma.isMaskedArray(array):
            raw = np.asarray(array.data)
            mask = np.ma.getmaskarray(array)
            if nodata is not None:
                mask = np.logical_or(mask, raw == nodata)
            data = raw.astype("float32", copy=False)
            if np.any(mask):
                data = data.copy()
                data[mask] = np.nan
        else:
            raw = np.asarray(array)
            data = raw.astype("float32", copy=False)
            if nodata is not None:
                data = data.copy()
                data[raw == nodata] = np.nan
        data = data * np.float32(scale.scale) + np.float32(scale.offset)
        if fill_value is not None:
            data = np.where(np.isnan(data), fill_value, data)
        return np.asarray(data, dtype=np.float32)

    if masked:
        if fill_value is not None:
            fill = fill_value
        elif np.issubdtype(array.dtype, np.floating):
            fill = np.nan
        elif nodata is not None:
            fill = nodata
        else:
            fill = 0
        array = array.filled(fill)
    return np.asarray(array)


def read_chunks_threaded(
    hrefs: Sequence[Sequence[str]],
    chunks: Sequence[ChunkWindow],
    scales: Optional[Sequence[Sequence[AssetScale]]] = None,
    source_band_index: int = 1,
    max_workers: int = 8,
    masked: bool = False,
    fill_value: Optional[float] = None,
    gdal_env: Optional[Mapping[str, str]] = None,
    reference_transform: Optional[Tuple[float, float, float, float, float, float]] = None,
    reference_resolution: Optional[float] = None,
    target_resolution: Optional[float] = None,
) -> np.ndarray:
    """Read all ``time x chunk x band`` windows with a thread pool.

    Work is grouped by ``time x band`` so each remote GeoTIFF is opened once
    and all sampled windows for that asset are read from the same GDAL dataset.

    When ``target_resolution`` is provided, each chunk window is resolved from
    its real-world bounds (using ``reference_transform``/``reference_resolution``)
    against each asset's own grid, so assets with different native Sentinel-2
    resolutions (10 m/20 m/60 m) are resampled to a common output grid. Nearest
    neighbour resampling is used when ``target_resolution`` is finer than an
    asset's native resolution, and average resampling is used when it is
    coarser.
    """

    if not hrefs:
        raise ValueError("No asset hrefs were provided")
    if not chunks:
        raise ValueError("No chunks were provided")
    if target_resolution is not None and (reference_transform is None or reference_resolution is None):
        raise ValueError(
            "reference_transform and reference_resolution are required when target_resolution is set"
        )

    if scales is None:
        scales = tuple(
            tuple(AssetScale(asset_key=f"band-{band_index}") for band_index, _ in enumerate(item_hrefs))
            for item_hrefs in hrefs
        )

    tasks = []
    for time_index, item_hrefs in enumerate(hrefs):
        if len(item_hrefs) != len(scales[time_index]):
            raise ValueError("href and scale matrices must have matching shapes")
        for band_index, href in enumerate(item_hrefs):
            tasks.append(
                AssetReadTask(
                    time_index=time_index,
                    band_index=band_index,
                    href=href,
                    chunks=tuple(chunks),
                    scale=scales[time_index][band_index],
                    source_band_index=source_band_index,
                    masked=masked,
                    fill_value=fill_value,
                    gdal_env=gdal_env,
                    reference_transform=reference_transform,
                    reference_resolution=reference_resolution,
                    target_resolution=target_resolution,
                )
            )

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(read_asset_chunks, task): task for task in tasks}
        for future in as_completed(future_map):
            results.append(future.result())

    if not results:
        raise ValueError("No read results were produced")

    if target_resolution is not None:
        expected_shape = _output_shape(chunks[0], reference_resolution, target_resolution)
    else:
        expected_shape = chunks[0].shape
    for result in results:
        for chunk_index, array in enumerate(result.arrays):
            if array.shape != expected_shape:
                raise ValueError(
                    "Read window shapes differ. Use assets with the same resolution/grid "
                    f"or sample from one resolution at a time. Expected {expected_shape}, "
                    f"got {array.shape} for time={result.time_index}, "
                    f"chunk={chunk_index}, band={result.band_index}."
                )

    dtype = np.result_type(*(array.dtype for result in results for array in result.arrays))
    data = np.empty(
        (
            len(hrefs),
            len(chunks),
            len(hrefs[0]),
            expected_shape[0],
            expected_shape[1],
        ),
        dtype=dtype,
    )
    for result in results:
        for chunk_index, array in enumerate(result.arrays):
            data[result.time_index, chunk_index, result.band_index, :, :] = array
    return data
