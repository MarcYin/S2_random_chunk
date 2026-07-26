"""User-facing request configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RandomChunkRequest:
    """Configuration for a Sentinel-2 random chunk sampling request."""

    mgrs_tile: str
    start_date: str
    end_date: str
    source: str = "planetary_computer"
    assets: Optional[Sequence[str]] = None
    n_chunks: int = 5
    chunk_batch: int = 0   # >0: stream this many windows per read (bounds peak memory)
    seed: Optional[int] = None
    max_items: Optional[int] = None
    cloud_cover: Optional[float] = None
    reference_asset: Optional[str] = None
    band_index: int = 1
    max_workers: int = 8
    scale_mode: str = "reflectance"
    allow_partial_edges: bool = False
    masked: bool = False
    fill_value: Optional[float] = None
    stac_endpoint: Optional[str] = None
    collection: Optional[str] = None
    gdal_env: Optional[Mapping[str, str]] = None
    target_resolution: Optional[float] = None
    # Optional hook to choose block windows: selector(windows, chunk_info) -> windows.
    # When set it replaces the default random selection (used for crop-dense sampling).
    windows_selector: Optional[Callable] = None
    # Optional pre-queried STAC items. When set, the internal STAC search is SKIPPED and these items are
    # used directly — lets a caller query once (with retry) and reuse across many fetches, avoiding
    # repeated Planetary Computer STAC queries (which are intermittently slow/timeout).
    items: Optional[Sequence[Any]] = None

    def asset_tuple(self, default_assets: Sequence[str]) -> Tuple[str, ...]:
        assets = self.assets if self.assets is not None else default_assets
        result = tuple(assets)
        if not result:
            raise ValueError("At least one asset must be requested")
        return result

    def validate(self) -> None:
        if self.n_chunks <= 0:
            raise ValueError("n_chunks must be greater than zero")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be greater than zero")
        if self.band_index <= 0:
            raise ValueError("band_index is 1-based and must be greater than zero")
        if self.cloud_cover is not None and not 0 <= self.cloud_cover <= 100:
            raise ValueError("cloud_cover must be between 0 and 100")
        if self.scale_mode not in {"raw", "stac", "reflectance"}:
            raise ValueError("scale_mode must be one of: raw, stac, reflectance")
        if self.target_resolution is not None and self.target_resolution <= 0:
            raise ValueError("target_resolution must be greater than zero")
