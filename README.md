# s2-random-chunk

Randomly sample internal Cloud Optimized GeoTIFF chunks from Sentinel-2 MGRS tiles and return a time-series `xarray.Dataset`.

The package searches Sentinel-2 L2A STAC items from either Microsoft Planetary Computer or Element84 Earth Search, inspects the remote GeoTIFF block layout before sampling, chooses random block-aligned chunks, and fetches the selected windows in parallel with Rasterio/GDAL HTTP range reads.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

## Quick start

```python
from s2_random_chunk import fetch_random_chunks

ds = fetch_random_chunks(
    mgrs_tile="06VVN",
    start_date="2024-06-01",
    end_date="2024-06-30",
    source="element84",          # or "planetary_computer"
    n_chunks=5,                  # default
    seed=42,
    max_items=10,
    scale_mode="reflectance",    # default; use "raw" for unchanged DN values
    target_resolution=10,        # optional; defaults to the reference asset's native resolution
)

print(ds)
```

By default, Planetary Computer uses the 10 reflectance bands `B02`-`B08`, `B8A`, `B11`, `B12` plus the `SCL` scene classification band; Earth Search uses `blue`, `green`, `red`, `rededge1`, `rededge2`, `rededge3`, `nir`, `nir08`, `swir16`, `swir22`, `scl`. Override with `assets=(...)` if you need a different band set.

## Mixed-resolution bands and `target_resolution`

Sentinel-2 bands are natively stored at different spatial resolutions (10 m, 20 m, and 60 m). Pass `target_resolution` (in the asset CRS units, meters for Sentinel-2 UTM tiles) to resample every requested band onto one common output grid:

- If `target_resolution` is finer (a smaller number) than an asset's native resolution, that asset is resampled with **nearest neighbour**.
- If `target_resolution` is coarser (a larger number) than an asset's native resolution, that asset is resampled with **average** resampling.
- If `target_resolution` is left unset, it defaults to the reference asset's native resolution (10 m for the default band lists), so mixing 10 m/20 m/60 m bands works out of the box.

The resolved `reference_resolution_m` and `target_resolution_m` are recorded in `ds.attrs`.

## Output

`fetch_random_chunks` returns an `xarray.Dataset` with:

- `data(time, chunk, band, y, x)`: sampled Sentinel-2 values. By default these are scaled to reflectance for optical bands.
- `time`, `item_id`, cloud cover, MGRS/grid coordinates, and provider metadata.
- chunk offsets, chunk bounds, chunk sizes, reference block shape, CRS, and affine transform metadata.
- `projected_x(chunk, y, x)` and `projected_y(chunk, y, x)` coordinates in the source CRS.
- `longitude(chunk, y, x)` and `latitude(chunk, y, x)` WGS84 coordinates when Rasterio can transform the source CRS.
- `scale(time, band)`, `offset(time, band)`, and serialized `scale_metadata` describing how stored values were converted.
- sanitized STAC item metadata in `ds.attrs["stac_items"]`.

The selected chunk windows are anchored to the first matching reference asset and reused across all times and requested bands. For Sentinel-2 assets on the same MGRS tile, all requested bands are resampled onto the `target_resolution` grid (defaulting to the reference asset's native resolution), so the default mixed-resolution 10/20/60 m band set produces one consistent local time-series tensor.

## Sentinel-2 scaling

The default `scale_mode="reflectance"` applies scale/offset before data are returned:

- Earth Search/Element84: uses each asset's STAC `raster:bands.scale`, `raster:bands.offset`, and `nodata` metadata.
- Planetary Computer: reads signed `product-metadata` XML and applies `BOA_ADD_OFFSET / BOA_QUANTIFICATION_VALUE` for optical bands. If XML cannot be read, it falls back to `scale=0.0001` and `offset=-0.1` for processing baselines `>= 04.00`, otherwise `offset=0`.
- `scale_mode="stac"` applies only STAC Raster metadata and does not use Planetary Computer XML fallbacks.
- `scale_mode="raw"` returns unchanged stored DN values.

This matters for Sentinel-2 L2A products after Processing Baseline 04.00, where Copernicus introduced BOA offsets. The output keeps no-data pixels as `NaN` when reflectance scaling is applied unless `fill_value` is provided.

## Provider notes

- `source="planetary_computer"` opens `https://planetarycomputer.microsoft.com/api/stac/v1` and signs asset URLs with the `planetary-computer` package.
- `source="element84"` or `source="earth_search"` opens `https://earth-search.aws.element84.com/v1`.
- Date-only inputs are normalized to RFC3339 intervals, which Earth Search requires.

## GDAL/Rasterio behavior

Remote reads use Rasterio windows over the GeoTIFF internal block grid. Reads are grouped by `time x band`, so each remote GeoTIFF is opened once and all selected chunk windows for that asset are read inside the same GDAL dataset. Those asset-level reads are parallelized with a thread pool.

On a live 4-band, 5-chunk, 1-item sample from MGRS tile `06VVN` on this development machine:

- Element84 Earth Search: 9.1 s with `max_workers=1`, 3.1 s with `max_workers=4`.
- Planetary Computer: 3.8 s with `max_workers=1`, 1.2 s with `max_workers=4`.

These timings are network- and cache-dependent, but they show the intended effect of parallelizing by remote asset while avoiding repeated opens for every chunk. The package sets conservative GDAL defaults:

- `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`
- `CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.tiff,.TIF,.TIFF`

Pass `gdal_env={...}` to override or extend those settings.
