from s2_random_chunk.providers import get_provider, mgrs_query_strategies


def test_planetary_computer_provider_defaults():
    provider = get_provider("pc")

    assert provider.name == "planetary_computer"
    assert provider.default_assets == (
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
        "SCL",
    )
    assert provider.signs_assets is True


def test_element84_mgrs_query_uses_mgrs_extension_properties():
    provider = get_provider("earth_search")

    queries = mgrs_query_strategies(provider, "T06VVN")

    assert queries[0] == {
        "mgrs:utm_zone": {"eq": 6},
        "mgrs:latitude_band": {"eq": "V"},
        "mgrs:grid_square": {"eq": "VN"},
    }
    assert queries[1] == {"grid:code": {"eq": "MGRS-6VVN"}}
