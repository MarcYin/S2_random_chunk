"""High-level Sentinel-2 random chunk sampler."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .chunks import inspect_remote_geotiff, list_block_windows, pixel_size, select_random_windows
from .config import RandomChunkRequest
from .providers import get_provider
from .reader import read_chunks_threaded
from .scaling import scale_matrix
from .stac import asset_href_matrix, search_s2_items
from .xarray_output import build_dataset


def fetch_random_chunks(
    mgrs_tile: str,
    start_date: str,
    end_date: str,
    source: str = "planetary_computer",
    assets: Optional[Sequence[str]] = None,
    n_chunks: int = 5,
    seed: Optional[int] = None,
    max_items: Optional[int] = None,
    cloud_cover: Optional[float] = None,
    reference_asset: Optional[str] = None,
    band_index: int = 1,
    max_workers: int = 8,
    scale_mode: str = "reflectance",
    allow_partial_edges: bool = False,
    masked: bool = False,
    fill_value: Optional[float] = None,
    stac_endpoint: Optional[str] = None,
    collection: Optional[str] = None,
    gdal_env: Optional[Mapping[str, str]] = None,
    target_resolution: Optional[float] = None,
    windows_selector=None,
    items=None,
    chunk_batch: int = 0,
):
    """Fetch random chunk-aligned Sentinel-2 time-series data as an xarray Dataset.

    ``chunk_batch`` > 0 returns a GENERATOR of datasets, each holding that many windows,
    instead of one dataset holding all ``n_chunks``. Window selection is identical either
    way -- only the read is split -- so results are unchanged. Use it whenever
    n_chunks x n_scenes is large: the single-dataset path allocates
    ``time x chunk x band x block_px x 4B``, linear in n_chunks.
    """

    request = RandomChunkRequest(
        mgrs_tile=mgrs_tile,
        start_date=start_date,
        end_date=end_date,
        source=source,
        assets=assets,
        n_chunks=n_chunks,
        seed=seed,
        max_items=max_items,
        cloud_cover=cloud_cover,
        reference_asset=reference_asset,
        band_index=band_index,
        max_workers=max_workers,
        scale_mode=scale_mode,
        allow_partial_edges=allow_partial_edges,
        masked=masked,
        fill_value=fill_value,
        stac_endpoint=stac_endpoint,
        collection=collection,
        gdal_env=gdal_env,
        target_resolution=target_resolution,
        windows_selector=windows_selector,
        items=items,
        chunk_batch=chunk_batch,
    )
    return fetch_random_chunks_for_request(request)


def fetch_random_chunks_for_request(request: RandomChunkRequest):
    """Execute a pre-built request."""

    request.validate()
    provider = get_provider(request.source, request.stac_endpoint, request.collection)
    asset_keys = request.asset_tuple(provider.default_assets)
    reference_asset = request.reference_asset or provider.reference_asset

    items = list(request.items) if request.items is not None else search_s2_items(request, provider)
    if not items:
        raise ValueError(
            "No Sentinel-2 items matched the requested MGRS tile/date range/provider filters"
        )

    reference_href = asset_href_matrix(items[:1], (reference_asset,))[0][0]
    chunk_info = inspect_remote_geotiff(
        reference_href,
        band_index=request.band_index,
        gdal_env=request.gdal_env,
    )
    windows = list_block_windows(
        reference_href,
        band_index=request.band_index,
        gdal_env=request.gdal_env,
    )
    if request.windows_selector is not None:
        chunks = tuple(request.windows_selector(windows, chunk_info))
        if not chunks:
            raise ValueError("windows_selector returned no chunk windows")
    else:
        chunks = select_random_windows(
            windows,
            n_chunks=request.n_chunks,
            seed=request.seed,
            full_shape=chunk_info.reference_block_shape,
            allow_partial_edges=request.allow_partial_edges,
        )

    hrefs = asset_href_matrix(items, asset_keys)
    scales = scale_matrix(items, asset_keys, provider, request.scale_mode)

    reference_resolution = max(pixel_size(chunk_info.transform))
    target_resolution = request.target_resolution or reference_resolution

    def _read_build(chunk_subset):
        data = read_chunks_threaded(
            hrefs,
            chunk_subset,
            scales=scales,
            source_band_index=request.band_index,
            max_workers=request.max_workers,
            masked=request.masked,
            fill_value=request.fill_value,
            gdal_env=request.gdal_env,
            reference_transform=chunk_info.transform,
            reference_resolution=reference_resolution,
            target_resolution=target_resolution,
        )
        return build_dataset(
            data=data,
            items=items,
            asset_keys=asset_keys,
            asset_hrefs=hrefs,
            scales=scales,
            chunks=chunk_subset,
            chunk_info=chunk_info,
            provider=provider,
            request=request,
            reference_asset=reference_asset,
            reference_resolution=reference_resolution,
            target_resolution=target_resolution,
        )

    # Reading every window at once allocates time x chunk x band x block_px x 4B, which is
    # linear in n_chunks and is what OOM-kills large fetches. chunk_batch>0 yields one
    # bounded dataset per batch instead, so peak memory is set by the batch, not n_chunks.
    batch = int(getattr(request, "chunk_batch", 0) or 0)
    if batch and batch < len(chunks):
        def _stream():
            for i in range(0, len(chunks), batch):
                yield _read_build(chunks[i:i + batch])
        return _stream()
    return _read_build(chunks)
