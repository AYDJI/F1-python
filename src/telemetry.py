import numpy as np
import pandas as pd

from .helpers import get_telemetry_time_seconds


def collect_session_telemetry(session):
    """
    Iterate laps for every driver, collect X/Y and time arrays
    and return a concatenated DataFrame with columns:
      ['driver', 'time', 'x', 'y', 'lap', 'distance', 'abs_time' (optional)]
    """
    laps = session.laps
    if laps is None or laps.empty:
        return None

    drivers = laps['Driver'].unique()
    rows = []

    print("Collecting telemetry from laps for drivers:", drivers.tolist())

    for drv in drivers:
        driver_laps = laps.pick_drivers(drv)
        for _idx, lap in driver_laps.iterlaps():
            try:
                tel = lap.get_telemetry()
            except Exception:
                continue
            if tel is None or tel.empty:
                continue
            if 'X' not in tel.columns or 'Y' not in tel.columns:
                continue
            t_seconds = get_telemetry_time_seconds(tel)
            if t_seconds is None:
                continue
            mask = tel['X'].notna() & tel['Y'].notna()
            if not mask.any():
                continue
            xs = tel.loc[mask, 'X'].to_numpy(dtype=float)
            ys = tel.loc[mask, 'Y'].to_numpy(dtype=float)
            ts = t_seconds[mask]

            # lap number (safe fallback)
            lap_num = 0
            try:
                if hasattr(lap, 'LapNumber') and not pd.isna(lap['LapNumber']):
                    lap_num = int(lap['LapNumber'])
                else:
                    lap_num = 0
            except Exception:
                lap_num = 0

            # distance: prefer telemetry Distance column if available, else approximate by intra-lap time
            if 'Distance' in tel.columns:
                dists = tel.loc[mask, 'Distance'].to_numpy(dtype=float)
            else:
                lap_start = ts.min() if len(ts) > 0 else 0.0
                dists = (ts - lap_start).astype(float)

            # Absolute time: attempt to compute POSIX seconds if telemetry contains an absolute 'Time'
            abs_times = None
            if 'Time' in tel.columns:
                abs_list = []
                for t in tel.loc[mask, 'Time']:
                    try:
                        # Try interpreting as absolute timestamp
                        abs_list.append(float(pd.Timestamp(t).timestamp()))
                    except Exception:
                        try:
                            # If it's a timedelta, use total_seconds
                            if hasattr(t, 'total_seconds'):
                                abs_list.append(float(t.total_seconds()))
                            else:
                                abs_list.append(float(t))
                        except Exception:
                            abs_list.append(np.nan)
                # If we have at least one non-NaN absolute time, keep the array
                if not all(np.isnan(abs_list)):
                    abs_times = np.array(abs_list, dtype=float)

            df = pd.DataFrame({
                'driver': [drv] * len(ts),
                'time': ts,
                'x': xs,
                'y': ys,
                'lap': [lap_num] * len(ts),
                'distance': dists
            })
            if abs_times is not None:
                # align length — it should match mask
                if len(abs_times) == len(df):
                    df['abs_time'] = abs_times
                else:
                    # safe fallback: drop abs_times if lengths mismatch
                    df['abs_time'] = np.nan

            rows.append(df)

    if not rows:
        return None

    full = pd.concat(rows, ignore_index=True)

    # normalize global time so earliest sample becomes 0.0
    global_min = float(full['time'].min())
    full['time'] = full['time'] - global_min

    # sort by time (keeps timeline ordered)
    full = full.sort_values('time').reset_index(drop=True)

    return full
