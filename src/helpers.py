import hashlib
import numpy as np
import pandas as pd


def fmt_time(s: float) -> str:
    """Format seconds -> M:SS.mmm, clamp negatives to 0.0 to avoid odd display."""
    try:
        ss = float(s)
    except Exception:
        ss = 0.0
    if ss < 0:
        ss = 0.0
    minutes = int(ss // 60)
    seconds = ss % 60
    return f"{minutes}:{seconds:06.3f}"


def gen_color_from_string(s: str):
    """Deterministic RGB color (0-255) from a string."""
    h = hashlib.md5(str(s).encode("utf8")).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    # bias into brighter range
    r = 80 + r % 176
    g = 80 + g % 176
    b = 80 + b % 176
    return (int(r), int(g), int(b))


def get_telemetry_time_seconds(telemetry: pd.DataFrame) -> np.ndarray:
    """
    Convert telemetry timing (SessionTime / Time) into float seconds.
    """
    if telemetry is None or telemetry.empty:
        return None

    # Prefer SessionTime if present
    if 'SessionTime' in telemetry.columns:
        col = telemetry['SessionTime']
        if pd.api.types.is_timedelta64_dtype(col.dtype):
            return col.dt.total_seconds().to_numpy(dtype=float)
        if isinstance(col.iloc[0], pd.Timedelta):
            return pd.Series(col).dt.total_seconds().to_numpy(dtype=float)
        try:
            return col.astype(float).to_numpy(dtype=float)
        except Exception:
            pass

    # Fallback to Time column
    if 'Time' in telemetry.columns:
        col = telemetry['Time']
        if pd.api.types.is_timedelta64_dtype(col.dtype):
            return col.dt.total_seconds().to_numpy(dtype=float)
        if np.issubdtype(col.dtype, np.datetime64):
            base = col.iloc[0]
            return (col - base).dt.total_seconds().to_numpy(dtype=float)
        if isinstance(col.iloc[0], pd.Timestamp):
            base = col.iloc[0]
            return np.array([(t - base).total_seconds() for t in col], dtype=float)

    return None


def normalize_coords(xs: np.ndarray, ys: np.ndarray, avail_w: int, avail_h: int, padding: int = 40):
    """Map raw X/Y to integer screen coordinates (0..avail_w, 0..avail_h)."""
    min_x, max_x = float(np.nanmin(xs)), float(np.nanmax(xs))
    min_y, max_y = float(np.nanmin(ys)), float(np.nanmax(ys))

    dx = max_x - min_x if max_x != min_x else 1.0
    dy = max_y - min_y if max_y != min_y else 1.0

    effective_w = avail_w - padding * 2
    effective_h = avail_h - padding * 2

    scale = min(effective_w / dx, effective_h / dy)

    nx = ((xs - min_x) * scale) + padding
    ny = ((ys - min_y) * scale) + padding

    # invert y for screen coordinates
    ny = avail_h - ny

    return nx.astype(int), ny.astype(int)


def to_epoch_seconds(seq):
    """
    Convert an iterable of possible datetime/timedelta/float-like objects to POSIX seconds (float).
    Falls back gracefully for timedeltas or floats.
    """
    res = []
    for t in seq:
        try:
            # Works for pd.Timestamp, np.datetime64, datetime.datetime
            res.append(float(pd.Timestamp(t).timestamp()))
        except Exception:
            try:
                # For Timedelta-like objects
                if hasattr(t, "total_seconds"):
                    res.append(float(t.total_seconds()))
                else:
                    res.append(float(t))
            except Exception:
                res.append(np.nan)
    return np.array(res, dtype=float)
