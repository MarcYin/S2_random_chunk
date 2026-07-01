import uuid

import numpy as np
import pytest

from s2_random_chunk.chunks import ChunkWindow
from s2_random_chunk.reader import _finalize_array, read_chunks_threaded
from s2_random_chunk.scaling import AssetScale


def test_finalize_array_masks_nodata_before_scaling():
    raw = np.asarray([[0, 2000]], dtype=np.uint16)
    scale = AssetScale("red", scale=0.0001, offset=-0.1, nodata=0, units="reflectance", applied=True)

    scaled = _finalize_array(raw, scale, nodata=0, fill_value=None, masked=False)

    assert np.isnan(scaled[0, 0])
    assert scaled[0, 1] == pytest.approx(0.1)


def _write_vsimem_raster(path, data, transform):
    rasterio = pytest.importorskip("rasterio")
    profile = dict(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:32606",
        transform=transform,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_read_chunks_threaded_resamples_mixed_resolution_bands():
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    transform10 = from_origin(500000, 7000000, 10, 10)
    transform20 = from_origin(500000, 7000000, 20, 20)

    data10 = np.arange(40 * 40, dtype=np.uint16).reshape(40, 40)
    data20 = np.arange(20 * 20, dtype=np.uint16).reshape(20, 20)

    prefix = f"/vsimem/{uuid.uuid4().hex}"
    href10 = f"{prefix}_10m.tif"
    href20 = f"{prefix}_20m.tif"
    try:
        _write_vsimem_raster(href10, data10, transform10)
        _write_vsimem_raster(href20, data20, transform20)

        chunks = [ChunkWindow(row_off=0, col_off=0, height=40, width=40)]
        data = read_chunks_threaded(
            hrefs=[[href10, href20]],
            chunks=chunks,
            reference_transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 7000000.0),
            reference_resolution=10.0,
            target_resolution=10.0,
        )

        assert data.shape == (1, 1, 2, 40, 40)
        # 10 m band (native resolution == target) is read unchanged.
        assert np.array_equal(data[0, 0, 0], data10)
        # 20 m band is nearest-neighbour upsampled to the 10 m target grid.
        assert np.array_equal(data[0, 0, 1, :2, :2], np.full((2, 2), data20[0, 0]))
        assert np.array_equal(data[0, 0, 1, :2, 2:4], np.full((2, 2), data20[0, 1]))
    finally:
        with rasterio.Env():
            for href in (href10, href20):
                try:
                    rasterio.shutil.delete(href)
                except Exception:
                    pass


def test_read_chunks_threaded_averages_when_target_resolution_is_coarser():
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    transform10 = from_origin(500000, 7000000, 10, 10)
    data10 = np.arange(40 * 40, dtype=np.float32).reshape(40, 40)

    href10 = f"/vsimem/{uuid.uuid4().hex}_10m.tif"
    try:
        _write_vsimem_raster(href10, data10, transform10)

        chunks = [ChunkWindow(row_off=0, col_off=0, height=40, width=40)]
        data = read_chunks_threaded(
            hrefs=[[href10]],
            chunks=chunks,
            reference_transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 7000000.0),
            reference_resolution=10.0,
            target_resolution=20.0,
        )

        assert data.shape == (1, 1, 1, 20, 20)
        expected_top_left = data10[:2, :2].mean()
        assert data[0, 0, 0, 0, 0] == pytest.approx(expected_top_left)
    finally:
        with rasterio.Env():
            try:
                rasterio.shutil.delete(href10)
            except Exception:
                pass
