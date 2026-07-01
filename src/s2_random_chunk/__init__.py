"""Random chunk-aware Sentinel-2 sampling from STAC-hosted GeoTIFFs."""

from .chunks import AssetChunkInfo, ChunkWindow, inspect_remote_geotiff, select_random_windows
from .config import RandomChunkRequest
from .providers import ProviderConfig, get_provider
from .sampler import fetch_random_chunks, fetch_random_chunks_for_request

__all__ = [
    "AssetChunkInfo",
    "ChunkWindow",
    "ProviderConfig",
    "RandomChunkRequest",
    "fetch_random_chunks",
    "fetch_random_chunks_for_request",
    "get_provider",
    "inspect_remote_geotiff",
    "select_random_windows",
]
