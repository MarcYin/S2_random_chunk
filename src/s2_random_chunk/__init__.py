"""Random chunk-aware Sentinel-2 sampling from STAC-hosted GeoTIFFs."""

from .chunks import AssetChunkInfo, ChunkWindow, inspect_remote_geotiff, select_random_windows
from .config import RandomChunkRequest
from .hls import fetch_hls_random_chunks, hls_fmask_valid, parse_hls_id, search_hls_items
from .providers import ProviderConfig, get_provider
from .sampler import fetch_random_chunks, fetch_random_chunks_for_request
from .scaling import AssetScale

__all__ = [
    "AssetChunkInfo",
    "AssetScale",
    "ChunkWindow",
    "ProviderConfig",
    "RandomChunkRequest",
    "fetch_hls_random_chunks",
    "fetch_random_chunks",
    "fetch_random_chunks_for_request",
    "get_provider",
    "hls_fmask_valid",
    "inspect_remote_geotiff",
    "parse_hls_id",
    "search_hls_items",
    "select_random_windows",
]
