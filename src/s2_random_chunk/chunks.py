"""Remote GeoTIFF block inspection and random chunk selection."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Mapping, Optional, Sequence, Tuple


DEFAULT_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF,.TIFF",
}


@dataclass(frozen=True)
class ChunkWindow:
    """A block-aligned pixel window in row/column coordinates."""

    row_off: int
    col_off: int
    height: int
    width: int

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.row_off, self.col_off, self.height, self.width)


@dataclass(frozen=True)
class AssetChunkInfo:
    """Remote GeoTIFF layout metadata for one asset."""

    href: str
    width: int
    height: int
    count: int
    dtype: str
    nodata: Optional[float]
    crs: Optional[str]
    transform: Tuple[float, float, float, float, float, float]
    block_shapes: Tuple[Tuple[int, int], ...]
    tiled: bool
    bounds: Tuple[float, float, float, float]
    overviews: Tuple[int, ...]

    @property
    def reference_block_shape(self) -> Tuple[int, int]:
        if not self.block_shapes:
            return (self.height, self.width)
        return self.block_shapes[0]


def merged_gdal_env(gdal_env: Optional[Mapping[str, str]] = None) -> dict:
    env = dict(DEFAULT_GDAL_ENV)
    if gdal_env:
        env.update({str(key): str(value) for key, value in gdal_env.items()})
    return env


def inspect_remote_geotiff(
    href: str,
    band_index: int = 1,
    gdal_env: Optional[Mapping[str, str]] = None,
) -> AssetChunkInfo:
    """Open a remote GeoTIFF and return its internal block layout metadata."""

    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - exercised when dependency is absent
        raise ImportError("rasterio is required to inspect remote GeoTIFF chunks") from exc

    with rasterio.Env(**merged_gdal_env(gdal_env)):
        with rasterio.open(href, sharing=False) as src:
            transform = src.transform
            return AssetChunkInfo(
                href=href,
                width=src.width,
                height=src.height,
                count=src.count,
                dtype=src.dtypes[band_index - 1],
                nodata=src.nodata,
                crs=src.crs.to_string() if src.crs else None,
                transform=(
                    transform.a,
                    transform.b,
                    transform.c,
                    transform.d,
                    transform.e,
                    transform.f,
                ),
                block_shapes=tuple(tuple(shape) for shape in src.block_shapes),
                tiled=bool(src.is_tiled),
                bounds=tuple(src.bounds),
                overviews=tuple(src.overviews(band_index)),
            )


def list_block_windows(
    href: str,
    band_index: int = 1,
    gdal_env: Optional[Mapping[str, str]] = None,
) -> Tuple[ChunkWindow, ...]:
    """List the internal block windows for a remote GeoTIFF band."""

    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - exercised when dependency is absent
        raise ImportError("rasterio is required to list remote GeoTIFF chunks") from exc

    windows = []
    with rasterio.Env(**merged_gdal_env(gdal_env)):
        with rasterio.open(href, sharing=False) as src:
            for _, window in src.block_windows(band_index):
                windows.append(
                    ChunkWindow(
                        row_off=int(window.row_off),
                        col_off=int(window.col_off),
                        height=int(window.height),
                        width=int(window.width),
                    )
                )
    return tuple(windows)


def select_random_windows(
    windows: Iterable[ChunkWindow],
    n_chunks: int = 5,
    seed: Optional[int] = None,
    full_shape: Optional[Tuple[int, int]] = None,
    allow_partial_edges: bool = False,
) -> Tuple[ChunkWindow, ...]:
    """Select ``n_chunks`` random windows, preferring full internal blocks."""

    if n_chunks <= 0:
        raise ValueError("n_chunks must be greater than zero")

    candidates = tuple(windows)
    if not candidates:
        raise ValueError("No chunk windows are available for sampling")

    if full_shape and not allow_partial_edges:
        full_height, full_width = full_shape
        candidates = tuple(
            window
            for window in candidates
            if window.height == full_height and window.width == full_width
        )

    if not candidates:
        raise ValueError("No full-size chunk windows are available for sampling")

    if n_chunks > len(candidates):
        raise ValueError(
            f"Requested {n_chunks} chunks, but only {len(candidates)} candidate chunks are available"
        )

    rng = random.Random(seed)
    return tuple(rng.sample(list(candidates), n_chunks))


def bounds_for_window(
    transform: Sequence[float],
    window: ChunkWindow,
) -> Tuple[float, float, float, float]:
    """Return ``left, bottom, right, top`` bounds for a north-up affine window."""

    a, b, c, d, e, f = transform
    if b != 0 or d != 0:
        xs = []
        ys = []
        for row in (window.row_off, window.row_off + window.height):
            for col in (window.col_off, window.col_off + window.width):
                xs.append(c + col * a + row * b)
                ys.append(f + col * d + row * e)
        return (min(xs), min(ys), max(xs), max(ys))

    left = c + window.col_off * a
    right = c + (window.col_off + window.width) * a
    top = f + window.row_off * e
    bottom = f + (window.row_off + window.height) * e
    return (min(left, right), min(bottom, top), max(left, right), max(bottom, top))
