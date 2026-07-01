"""Small pure helpers for MGRS and STAC metadata."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
import re
from typing import Any, Mapping, Optional, Tuple


MGRS_RE = re.compile(r"^(?:T)?(?P<zone>\d{1,2})(?P<band>[C-HJ-NP-X])(?P<square>[A-HJ-NP-Z]{2})$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_mgrs_tile(tile: str) -> str:
    compact = tile.strip().upper().replace(" ", "")
    match = MGRS_RE.match(compact)
    if not match:
        raise ValueError(
            "MGRS tile must look like '06VVN' or 'T06VVN' "
            "(zone, latitude band, two-letter grid square)"
        )
    zone = int(match.group("zone"))
    if not 1 <= zone <= 60:
        raise ValueError("MGRS UTM zone must be between 1 and 60")
    return f"{zone:02d}{match.group('band')}{match.group('square')}"


def mgrs_parts(tile: str) -> Tuple[str, str, str]:
    normalized = normalize_mgrs_tile(tile)
    return normalized[:2], normalized[2], normalized[3:]


def _rfc3339(value: Any, end_of_day: bool) -> str:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, time.max if end_of_day else time.min)
    else:
        text = str(value)
        if DATE_RE.match(text):
            suffix = "T23:59:59Z" if end_of_day else "T00:00:00Z"
            return f"{text}{suffix}"
        return text

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def datetime_interval(start_date: Any, end_date: Any) -> str:
    return f"{_rfc3339(start_date, end_of_day=False)}/{_rfc3339(end_date, end_of_day=True)}"


def item_datetime(item: Any) -> Optional[datetime]:
    dt = getattr(item, "datetime", None)
    if isinstance(dt, datetime):
        return dt

    props = getattr(item, "properties", None)
    if props is None and isinstance(item, Mapping):
        props = item.get("properties", {})
    if not props:
        return None

    value = props.get("datetime") or props.get("start_datetime")
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def strip_url_query(url: str) -> str:
    return url.split("?", 1)[0]
