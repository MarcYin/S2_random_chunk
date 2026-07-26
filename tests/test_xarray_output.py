import json
import importlib.util

import numpy as np

from s2_random_chunk.chunks import AssetChunkInfo, ChunkWindow
from s2_random_chunk.config import RandomChunkRequest
from s2_random_chunk.providers import get_provider
from s2_random_chunk.scaling import AssetScale
from s2_random_chunk.xarray_output import build_dataset


def test_build_dataset_populates_time_chunk_and_item_metadata():
    item = {
        "id": "S2A_6VVN_20240630_0_L2A",
        "collection": "sentinel-2-l2a",
        "properties": {
            "datetime": "2024-06-30T21:38:21.424000Z",
            "eo:cloud_cover": 12.5,
            "s2:processing_baseline": "05.10",
            "earthsearch:boa_offset_applied": True,
            "mgrs:utm_zone": 6,
            "mgrs:latitude_band": "V",
            "mgrs:grid_square": "VN",
            "grid:code": "MGRS-6VVN",
        },
    }
    data = np.ones((1, 1, 2, 4, 4), dtype=np.uint16)
    chunk_info = AssetChunkInfo(
        href="https://example.com/ref.tif?token=secret",
        width=10980,
        height=10980,
        count=1,
        dtype="uint16",
        nodata=0,
        crs="EPSG:32606",
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 7000000.0),
        block_shapes=((4, 4),),
        tiled=True,
        bounds=(500000.0, 6890200.0, 609800.0, 7000000.0),
        overviews=(2, 4, 8),
    )

    ds = build_dataset(
        data=data,
        items=[item],
        asset_keys=("blue", "green"),
        asset_hrefs=[["https://example.com/blue.tif?sig=1", "https://example.com/green.tif"]],
        scales=[
            [
                AssetScale("blue", scale=0.0001, offset=0.0, nodata=0, units="reflectance", source="test", applied=True),
                AssetScale("green", scale=0.0001, offset=0.0, nodata=0, units="reflectance", source="test", applied=True),
            ]
        ],
        chunks=[ChunkWindow(0, 0, 4, 4)],
        chunk_info=chunk_info,
        provider=get_provider("element84"),
        request=RandomChunkRequest("06VVN", "2024-06-01", "2024-06-30", source="element84"),
        reference_asset="blue",
        reference_resolution=10.0,
        target_resolution=10.0,
    )

    assert ds["data"].dims == ("time", "chunk", "band", "y", "x")
    assert ds["asset_href"].values[0, 0] == "https://example.com/blue.tif"
    assert ds.coords["grid_code"].values[0] == "MGRS-6VVN"
    assert ds.coords["mgrs_tile"].values[0] == "06VVN"
    assert ds.coords["projected_x"].values[0, 0, 0] == 500005.0
    assert ds.coords["projected_y"].values[0, 0, 0] == 6999995.0
    if importlib.util.find_spec("rasterio") is not None:
        assert "longitude" in ds.coords
        assert "latitude" in ds.coords
    assert ds["scale"].values[0, 0] == 0.0001
    assert ds["offset"].values[0, 0] == 0.0
    assert ds["data"].attrs["scale_mode"] == "reflectance"
    assert ds.attrs["reflectance_contract"] == "s2-l2a-provider-aware-reflectance-v2"
    assert ds.attrs["reference_block_shape"] == (4, 4)
    assert json.loads(ds.attrs["stac_items"])[0]["id"] == "S2A_6VVN_20240630_0_L2A"
