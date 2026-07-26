"""Sentinel-2 scale/offset metadata handling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from .providers import ProviderConfig
from .stac import asset_href, item_assets, item_properties


S2_OPTICAL_BAND_IDS = {
    "B01": 0,
    "B02": 1,
    "B03": 2,
    "B04": 3,
    "B05": 4,
    "B06": 5,
    "B07": 6,
    "B08": 7,
    "B8A": 8,
    "B09": 9,
    "B10": 10,
    "B11": 11,
    "B12": 12,
}

COMMON_NAME_TO_S2_BAND = {
    "coastal": "B01",
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "rededge1": "B05",
    "rededge2": "B06",
    "rededge3": "B07",
    "nir": "B08",
    "nir08": "B8A",
    "nir09": "B09",
    "swir16": "B11",
    "swir22": "B12",
}

VALID_SCALE_MODES = {"raw", "stac", "reflectance"}
REFLECTANCE_CONTRACT_VERSION = "s2-l2a-provider-aware-reflectance-v2"


@dataclass(frozen=True)
class AssetScale:
    """Scale/offset to convert one asset's stored values."""

    asset_key: str
    scale: float = 1.0
    offset: float = 0.0
    nodata: Optional[float] = None
    units: str = "dn"
    source: str = "raw"
    applied: bool = False


def scale_matrix(
    items: Sequence[Any],
    asset_keys: Sequence[str],
    provider: ProviderConfig,
    scale_mode: str = "reflectance",
) -> Tuple[Tuple[AssetScale, ...], ...]:
    """Return per-item, per-asset scaling metadata."""

    validate_scale_mode(scale_mode)
    return tuple(
        tuple(asset_scale_for_item(item, asset_key, provider, scale_mode) for asset_key in asset_keys)
        for item in items
    )


def validate_scale_mode(scale_mode: str) -> None:
    if scale_mode not in VALID_SCALE_MODES:
        valid = ", ".join(sorted(VALID_SCALE_MODES))
        raise ValueError(f"scale_mode must be one of: {valid}")


def asset_scale_for_item(
    item: Any,
    asset_key: str,
    provider: ProviderConfig,
    scale_mode: str = "reflectance",
) -> AssetScale:
    """Resolve the correct scale/offset for an item asset."""

    validate_scale_mode(scale_mode)
    if scale_mode == "raw":
        return AssetScale(asset_key=asset_key)

    raster = raster_band_metadata(item, asset_key)
    if raster is not None:
        scale = float(raster.get("scale", 1.0))
        offset = float(raster.get("offset", 0.0))
        nodata = _optional_float(raster.get("nodata"))
        source = "stac:raster:bands"
        # Earth Search's common-name COG assets are harmonised when this item
        # flag is true: post-PB04 pixels have already been shifted by -1000 DN.
        # Their raster metadata still advertises the original BOA offset, so
        # applying it in reflectance mode would subtract 0.1 a second time.
        #
        # Keep scale_mode="stac" literal for callers explicitly requesting raw
        # STAC metadata semantics.  The provider-aware correction belongs to
        # scale_mode="reflectance", which promises physical reflectance.
        if (
            scale_mode == "reflectance"
            and provider.name == "element84"
            and is_s2_optical_asset(asset_key)
            and earthsearch_boa_offset_applied(item)
        ):
            offset = 0.0
            source = "element84:boa-offset-already-applied"
        units = "reflectance" if scale != 1.0 or offset != 0.0 else "dn"
        return AssetScale(
            asset_key=asset_key,
            scale=scale,
            offset=offset,
            nodata=nodata,
            units=units,
            source=source,
            applied=scale_mode in {"stac", "reflectance"} and (scale != 1.0 or offset != 0.0),
        )

    if scale_mode == "stac":
        return AssetScale(asset_key=asset_key, source="missing-stac-raster:bands")

    if provider.name == "planetary_computer" and is_s2_optical_asset(asset_key):
        xml_scale = planetary_computer_xml_scale(item, asset_key)
        if xml_scale is not None:
            return xml_scale
        return planetary_computer_baseline_fallback(item, asset_key)

    return AssetScale(asset_key=asset_key, source="no-scale-metadata")


def raster_band_metadata(item: Any, asset_key: str, band_index: int = 1) -> Optional[Mapping[str, Any]]:
    assets = item_assets(item)
    if asset_key not in assets:
        return None
    asset = assets[asset_key]
    extra_fields = getattr(asset, "extra_fields", None)
    raster_bands = None
    if extra_fields:
        raster_bands = extra_fields.get("raster:bands")
    if raster_bands is None and isinstance(asset, Mapping):
        raster_bands = asset.get("raster:bands")
    if not raster_bands or len(raster_bands) < band_index:
        return None
    return raster_bands[band_index - 1]


def planetary_computer_xml_scale(item: Any, asset_key: str) -> Optional[AssetScale]:
    href = _metadata_href(item, "product-metadata")
    if not href:
        return None
    try:
        metadata = _read_l2a_product_scale_metadata(href)
    except Exception:
        return None

    quantification_value = metadata.get("boa_quantification_value")
    if not quantification_value:
        return None

    band_id = s2_band_id(asset_key)
    offset_dn = metadata.get("boa_offsets", {}).get(band_id, 0.0)
    return AssetScale(
        asset_key=asset_key,
        scale=1.0 / float(quantification_value),
        offset=float(offset_dn) / float(quantification_value),
        nodata=0,
        units="reflectance",
        source="planetary-computer:product-metadata",
        applied=True,
    )


def planetary_computer_baseline_fallback(item: Any, asset_key: str) -> AssetScale:
    baseline = processing_baseline(item)
    offset = -0.1 if baseline is not None and baseline >= 4.0 else 0.0
    return AssetScale(
        asset_key=asset_key,
        scale=0.0001,
        offset=offset,
        nodata=0,
        units="reflectance",
        source="planetary-computer:processing-baseline-fallback",
        applied=True,
    )


def processing_baseline(item: Any) -> Optional[float]:
    value = item_properties(item).get("s2:processing_baseline")
    if value is None:
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def earthsearch_boa_offset_applied(item: Any) -> bool:
    """Whether an Earth Search COG already contains the PB04 BOA shift."""

    value = item_properties(item).get("earthsearch:boa_offset_applied")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def is_s2_optical_asset(asset_key: str) -> bool:
    return s2_band_id(asset_key) is not None


def s2_band_id(asset_key: str) -> Optional[int]:
    normalized = asset_key.upper()
    if normalized in S2_OPTICAL_BAND_IDS:
        return S2_OPTICAL_BAND_IDS[normalized]
    band_name = COMMON_NAME_TO_S2_BAND.get(asset_key.lower())
    if band_name:
        return S2_OPTICAL_BAND_IDS[band_name]
    return None


def scale_metadata_for_attrs(scales: Sequence[Sequence[AssetScale]]):
    return [
        [
            {
                "asset": scale.asset_key,
                "scale": scale.scale,
                "offset": scale.offset,
                "nodata": scale.nodata,
                "units": scale.units,
                "source": scale.source,
                "applied": scale.applied,
            }
            for scale in item_scales
        ]
        for item_scales in scales
    ]


def _metadata_href(item: Any, asset_key: str) -> Optional[str]:
    try:
        return asset_href(item, asset_key)
    except Exception:
        return None


@lru_cache(maxsize=256)
def _read_l2a_product_scale_metadata(href: str) -> Dict[str, Any]:
    with urlopen(href, timeout=30) as response:
        root = ET.fromstring(response.read())
    quantification_value = None
    offsets: Dict[Optional[int], float] = {}
    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]
        text = (elem.text or "").strip()
        if not text:
            continue
        if tag == "BOA_QUANTIFICATION_VALUE":
            quantification_value = float(text)
        elif tag == "BOA_ADD_OFFSET":
            band_id = elem.attrib.get("band_id")
            offsets[int(band_id) if band_id is not None else None] = float(text)
    return {
        "boa_quantification_value": quantification_value,
        "boa_offsets": offsets,
    }


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)
