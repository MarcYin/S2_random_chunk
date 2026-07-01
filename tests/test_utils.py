import pytest

from s2_random_chunk.utils import datetime_interval, mgrs_parts, normalize_mgrs_tile


def test_normalize_mgrs_tile_accepts_optional_t_prefix():
    assert normalize_mgrs_tile("T6vvn") == "06VVN"
    assert mgrs_parts("06VVN") == ("06", "V", "VN")


def test_normalize_mgrs_tile_rejects_invalid_zone():
    with pytest.raises(ValueError):
        normalize_mgrs_tile("99VVN")


def test_datetime_interval_expands_date_only_inputs_to_rfc3339():
    assert datetime_interval("2024-06-01", "2024-06-30") == (
        "2024-06-01T00:00:00Z/2024-06-30T23:59:59Z"
    )
