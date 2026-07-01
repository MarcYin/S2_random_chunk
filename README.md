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
)

print(ds)
```

By default, Planetary Computer uses assets `B02`, `B03`, `B04`, `B08`; Earth Search uses `blue`, `green`, `red`, `nir`. Override with `assets=(...)` if you need a different band set.

## Output

`fetch_random_chunks` returns an `xarray.Dataset` with:

- `data(time, chunk, band, y, x)`: raw pixel values from each selected GeoTIFF window.
- `time`, `item_id`, cloud cover, MGRS/grid coordinates, and provider metadata.
- chunk offsets, chunk bounds, chunk sizes, reference block shape, CRS, and affine transform metadata.
- sanitized STAC item metadata in `ds.attrs["stac_items"]`.

The selected chunk windows are anchored to the first matching reference asset and reused across all times and requested bands. For Sentinel-2 assets on the same MGRS tile and resolution this gives a consistent local time-series tensor. If you mix assets with different resolutions, Rasterio will return different window sizes and the package will raise a shape error.

## Provider notes

- `source="planetary_computer"` opens `https://planetarycomputer.microsoft.com/api/stac/v1` and signs asset URLs with the `planetary-computer` package.
- `source="element84"` or `source="earth_search"` opens `https://earth-search.aws.element84.com/v1`.
- Date-only inputs are normalized to RFC3339 intervals, which Earth Search requires.

## GDAL/Rasterio behavior

Remote reads use Rasterio windows over the GeoTIFF internal block grid. The package sets conservative GDAL defaults:

- `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`
- `CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.tiff,.TIF,.TIFF`

Pass `gdal_env={...}` to override or extend those settings.
