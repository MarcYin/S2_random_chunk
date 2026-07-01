import numpy as np
import pytest

from s2_random_chunk.reader import _finalize_array
from s2_random_chunk.scaling import AssetScale


def test_finalize_array_masks_nodata_before_scaling():
    raw = np.asarray([[0, 2000]], dtype=np.uint16)
    scale = AssetScale("red", scale=0.0001, offset=-0.1, nodata=0, units="reflectance", applied=True)

    scaled = _finalize_array(raw, scale, nodata=0, fill_value=None, masked=False)

    assert np.isnan(scaled[0, 0])
    assert scaled[0, 1] == pytest.approx(0.1)
