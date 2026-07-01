"""STAC provider definitions and MGRS query helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from .utils import mgrs_parts, normalize_mgrs_tile


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    endpoint: str
    collection: str
    default_assets: Tuple[str, ...]
    reference_asset: str
    signs_assets: bool = False


PROVIDERS: Dict[str, ProviderConfig] = {
    "planetary_computer": ProviderConfig(
        name="planetary_computer",
        endpoint="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="sentinel-2-l2a",
        default_assets=("B02", "B03", "B04", "B08"),
        reference_asset="B02",
        signs_assets=True,
    ),
    "element84": ProviderConfig(
        name="element84",
        endpoint="https://earth-search.aws.element84.com/v1",
        collection="sentinel-2-l2a",
        default_assets=("blue", "green", "red", "nir"),
        reference_asset="blue",
    ),
}

ALIASES = {
    "pc": "planetary_computer",
    "planetary": "planetary_computer",
    "planetary-computer": "planetary_computer",
    "microsoft": "planetary_computer",
    "earth_search": "element84",
    "earth-search": "element84",
    "earthsearch": "element84",
    "e84": "element84",
    "element-84": "element84",
}


def normalize_source(source: str) -> str:
    key = source.strip().lower()
    return ALIASES.get(key, key)


def get_provider(
    source: str,
    endpoint: Optional[str] = None,
    collection: Optional[str] = None,
) -> ProviderConfig:
    """Return provider defaults, applying optional endpoint/collection overrides."""

    key = normalize_source(source)
    if key not in PROVIDERS:
        valid = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unknown source {source!r}. Valid sources: {valid}")

    provider = PROVIDERS[key]
    if endpoint is None and collection is None:
        return provider
    return ProviderConfig(
        name=provider.name,
        endpoint=endpoint or provider.endpoint,
        collection=collection or provider.collection,
        default_assets=provider.default_assets,
        reference_asset=provider.reference_asset,
        signs_assets=provider.signs_assets,
    )


def mgrs_query_strategies(provider: ProviderConfig, tile: str) -> List[Mapping[str, Mapping[str, object]]]:
    """Build provider-specific MGRS query strategies."""

    normalized = normalize_mgrs_tile(tile)
    zone, latitude_band, grid_square = mgrs_parts(normalized)
    grid_code = f"MGRS-{int(zone)}{latitude_band}{grid_square}"

    if provider.name == "planetary_computer":
        return [
            {"s2:mgrs_tile": {"eq": normalized}},
            {"grid:code": {"eq": grid_code}},
        ]

    return [
        {
            "mgrs:utm_zone": {"eq": int(zone)},
            "mgrs:latitude_band": {"eq": latitude_band},
            "mgrs:grid_square": {"eq": grid_square},
        },
        {"grid:code": {"eq": grid_code}},
        {"s2:mgrs_tile": {"eq": normalized}},
    ]


def merge_query_parts(*parts: Mapping[str, Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    query: Dict[str, Mapping[str, object]] = {}
    for part in parts:
        query.update(part)
    return query
