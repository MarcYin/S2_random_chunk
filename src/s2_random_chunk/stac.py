"""STAC search and asset extraction."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .config import RandomChunkRequest
from .providers import ProviderConfig, merge_query_parts, mgrs_query_strategies
from .utils import datetime_interval, item_datetime, normalize_mgrs_tile


def search_s2_items(request: RandomChunkRequest, provider: ProviderConfig) -> List[Any]:
    """Search a provider for Sentinel-2 items matching the request."""

    try:
        from pystac_client import Client
    except ImportError as exc:  # pragma: no cover - exercised when dependency is absent
        raise ImportError("pystac-client is required to search STAC providers") from exc

    modifier = None
    if provider.signs_assets:
        try:
            import planetary_computer
        except ImportError as exc:  # pragma: no cover - exercised when dependency is absent
            raise ImportError(
                "planetary-computer is required for Planetary Computer signed asset URLs"
            ) from exc
        modifier = planetary_computer.sign_inplace

    client = Client.open(provider.endpoint, modifier=modifier)
    dt_interval = datetime_interval(request.start_date, request.end_date)
    errors = []
    successful_queries = 0

    for strategy in mgrs_query_strategies(provider, request.mgrs_tile):
        query_parts = [strategy]
        if request.cloud_cover is not None:
            query_parts.append({"eo:cloud_cover": {"lte": request.cloud_cover}})
        query = merge_query_parts(*query_parts)
        try:
            search = client.search(
                collections=[provider.collection],
                datetime=dt_interval,
                query=query,
                max_items=request.max_items,
            )
            items = list(search.items())
        except Exception as exc:  # pragma: no cover - provider-specific fallback path
            errors.append(f"{query}: {exc}")
            continue
        successful_queries += 1
        if items:
            return sorted(items, key=_sort_key)

    if errors and successful_queries == 0:
        joined = "; ".join(errors)
        raise RuntimeError(f"All STAC query strategies failed: {joined}")
    return []


def item_assets(item: Any) -> Mapping[str, Any]:
    assets = getattr(item, "assets", None)
    if assets is not None:
        return assets
    if isinstance(item, Mapping):
        return item.get("assets", {})
    return {}


def asset_href(item: Any, asset_key: str) -> str:
    assets = item_assets(item)
    if asset_key not in assets:
        item_id = item_id_value(item)
        available = ", ".join(sorted(assets.keys()))
        raise KeyError(f"Asset {asset_key!r} not found on item {item_id!r}. Available: {available}")

    asset = assets[asset_key]
    href = getattr(asset, "href", None)
    if href is None and isinstance(asset, Mapping):
        href = asset.get("href")
    if not href:
        raise ValueError(f"Asset {asset_key!r} on item {item_id_value(item)!r} has no href")
    return str(href)


def asset_href_matrix(items: Sequence[Any], asset_keys: Sequence[str]) -> List[List[str]]:
    return [[asset_href(item, key) for key in asset_keys] for item in items]


def item_id_value(item: Any) -> str:
    item_id = getattr(item, "id", None)
    if item_id is not None:
        return str(item_id)
    if isinstance(item, Mapping):
        return str(item.get("id", ""))
    return ""


def item_collection_value(item: Any) -> str:
    collection = getattr(item, "collection_id", None) or getattr(item, "collection", None)
    if collection is not None:
        return str(collection)
    if isinstance(item, Mapping):
        return str(item.get("collection", ""))
    return ""


def item_properties(item: Any) -> Dict[str, Any]:
    props = getattr(item, "properties", None)
    if props is not None:
        return dict(props)
    if isinstance(item, Mapping):
        return dict(item.get("properties", {}))
    return {}


def item_mgrs_tile(item: Any) -> Optional[str]:
    props = item_properties(item)
    if props.get("s2:mgrs_tile"):
        return normalize_mgrs_tile(str(props["s2:mgrs_tile"]))
    if props.get("grid:code"):
        code = str(props["grid:code"])
        if code.startswith("MGRS-"):
            return normalize_mgrs_tile(code[5:])
    zone = props.get("mgrs:utm_zone")
    band = props.get("mgrs:latitude_band")
    square = props.get("mgrs:grid_square")
    if zone is not None and band and square:
        return normalize_mgrs_tile(f"{int(zone):02d}{band}{square}")
    return None


def _sort_key(item: Any) -> datetime:
    dt = item_datetime(item)
    if dt is None:
        return datetime.min
    return dt.replace(tzinfo=None)
