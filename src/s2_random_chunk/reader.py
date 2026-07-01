"""Threaded Rasterio window reads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .chunks import ChunkWindow, merged_gdal_env


@dataclass(frozen=True)
class ReadTask:
    time_index: int
    chunk_index: int
    band_index: int
    href: str
    window: ChunkWindow
    source_band_index: int
    masked: bool
    fill_value: Optional[float]
    gdal_env: Optional[Mapping[str, str]]


@dataclass(frozen=True)
class ReadResult:
    time_index: int
    chunk_index: int
    band_index: int
    array: np.ndarray


def read_window(task: ReadTask) -> ReadResult:
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:  # pragma: no cover - exercised when dependency is absent
        raise ImportError("rasterio is required to read remote GeoTIFF chunks") from exc

    rasterio_window = Window(
        col_off=task.window.col_off,
        row_off=task.window.row_off,
        width=task.window.width,
        height=task.window.height,
    )

    with rasterio.Env(**merged_gdal_env(task.gdal_env)):
        with rasterio.open(task.href, sharing=False) as src:
            array = src.read(
                task.source_band_index,
                window=rasterio_window,
                masked=task.masked,
            )
            if task.masked:
                if task.fill_value is not None:
                    fill = task.fill_value
                elif np.issubdtype(array.dtype, np.floating):
                    fill = np.nan
                elif src.nodata is not None:
                    fill = src.nodata
                else:
                    fill = 0
                array = array.filled(fill)
            return ReadResult(
                time_index=task.time_index,
                chunk_index=task.chunk_index,
                band_index=task.band_index,
                array=np.asarray(array),
            )


def read_chunks_threaded(
    hrefs: Sequence[Sequence[str]],
    chunks: Sequence[ChunkWindow],
    source_band_index: int = 1,
    max_workers: int = 8,
    masked: bool = False,
    fill_value: Optional[float] = None,
    gdal_env: Optional[Mapping[str, str]] = None,
) -> np.ndarray:
    """Read all ``time x chunk x band`` windows with a thread pool."""

    if not hrefs:
        raise ValueError("No asset hrefs were provided")
    if not chunks:
        raise ValueError("No chunks were provided")

    tasks = []
    for time_index, item_hrefs in enumerate(hrefs):
        for band_index, href in enumerate(item_hrefs):
            for chunk_index, chunk in enumerate(chunks):
                tasks.append(
                    ReadTask(
                        time_index=time_index,
                        chunk_index=chunk_index,
                        band_index=band_index,
                        href=href,
                        window=chunk,
                        source_band_index=source_band_index,
                        masked=masked,
                        fill_value=fill_value,
                        gdal_env=gdal_env,
                    )
                )

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(read_window, task): task for task in tasks}
        for future in as_completed(future_map):
            results.append(future.result())

    if not results:
        raise ValueError("No read results were produced")

    expected_shape = chunks[0].shape
    for result in results:
        if result.array.shape != expected_shape:
            raise ValueError(
                "Read window shapes differ. Use assets with the same resolution/grid "
                f"or sample from one resolution at a time. Expected {expected_shape}, "
                f"got {result.array.shape} for time={result.time_index}, "
                f"chunk={result.chunk_index}, band={result.band_index}."
            )

    dtype = np.result_type(*(result.array.dtype for result in results))
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
        data[result.time_index, result.chunk_index, result.band_index, :, :] = result.array
    return data
