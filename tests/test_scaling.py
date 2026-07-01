import pytest

from s2_random_chunk.providers import get_provider
from s2_random_chunk.scaling import asset_scale_for_item, s2_band_id


def test_earth_search_uses_stac_raster_scale_and_offset():
    item = {
        "id": "S2A_6VVN_20240630_0_L2A",
        "properties": {"s2:processing_baseline": "05.10"},
        "assets": {
            "red": {
                "href": "https://example.com/B04.tif",
                "raster:bands": [{"scale": 0.0001, "offset": -0.1, "nodata": 0}],
            }
        },
    }

    scale = asset_scale_for_item(item, "red", get_provider("element84"))

    assert scale.scale == pytest.approx(0.0001)
    assert scale.offset == pytest.approx(-0.1)
    assert scale.nodata == 0
    assert scale.source == "stac:raster:bands"
    assert scale.applied is True


def test_planetary_computer_falls_back_to_processing_baseline_when_xml_is_unavailable():
    item = {
        "id": "S2A_MSIL2A_20240630T213531_R086_T06VVN_20240701T215534",
        "properties": {"s2:processing_baseline": "05.10"},
        "assets": {"B04": {"href": "https://example.com/B04.tif"}},
    }

    scale = asset_scale_for_item(item, "B04", get_provider("planetary_computer"))

    assert scale.scale == pytest.approx(0.0001)
    assert scale.offset == pytest.approx(-0.1)
    assert scale.nodata == 0
    assert scale.source == "planetary-computer:processing-baseline-fallback"
    assert scale.applied is True


def test_raw_scale_mode_does_not_apply_scale():
    item = {
        "id": "S2A_6VVN_20240630_0_L2A",
        "assets": {
            "red": {
                "href": "https://example.com/B04.tif",
                "raster:bands": [{"scale": 0.0001, "offset": -0.1, "nodata": 0}],
            }
        },
    }

    scale = asset_scale_for_item(item, "red", get_provider("element84"), scale_mode="raw")

    assert scale.scale == 1.0
    assert scale.offset == 0.0
    assert scale.applied is False


def test_stac_scale_mode_applies_stac_metadata_without_provider_fallbacks():
    item = {
        "id": "S2A_6VVN_20240630_0_L2A",
        "assets": {
            "red": {
                "href": "https://example.com/B04.tif",
                "raster:bands": [{"scale": 0.0001, "offset": -0.1, "nodata": 0}],
            }
        },
    }

    scale = asset_scale_for_item(item, "red", get_provider("element84"), scale_mode="stac")

    assert scale.scale == pytest.approx(0.0001)
    assert scale.offset == pytest.approx(-0.1)
    assert scale.applied is True


def test_s2_common_name_band_mapping():
    assert s2_band_id("red") == s2_band_id("B04")
    assert s2_band_id("nir") == s2_band_id("B08")
