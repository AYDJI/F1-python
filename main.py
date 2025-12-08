#!/usr/bin/env python3
"""
main.py - Simple F1 telemetry viewer using fastf1 + pygame

This version fixes:
 - leaderboard correctness at start & finish
 - performance: removes heavy per-frame DataFrame ops (no more 1 FPS)
 - simulation time driven by real wall-clock time for smooth playback

Usage:
    python main.py
"""

import os
import hashlib
import time
import datetime
import threading

import numpy as np
import pandas as pd

try:
    import fastf1 as ff1
except Exception:
    print("fastf1 is required. Install with: pip install fastf1")
    raise

try:
    import pygame
except Exception:
    print("pygame is required. Install with: pip install pygame")
    raise

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:
    print("tkinter is required for GUI selection")
    raise


# ---------------------------
# Configuration
# ---------------------------
MAIN_W = 600
INFO_W = 300
SIDEBAR_W = 240
SCREEN_W = MAIN_W + INFO_W + SIDEBAR_W
SCREEN_H = 800
FPS = 60
DEFAULT_POINT_SIZE = 6


# ---------------------------
# Helpers
# ---------------------------
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


# ---------------------------
# Telemetry collection (per-lap) - returns DataFrame with lap & distance
# ---------------------------
def collect_session_telemetry(session):
    """
    Iterate laps for every driver, collect X/Y and time arrays
    and return a concatenated DataFrame with columns:
      ['driver', 'time', 'x', 'y', 'lap', 'distance']
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

            df = pd.DataFrame({
                'driver': [drv] * len(ts),
                'time': ts,
                'x': xs,
                'y': ys,
                'lap': [lap_num] * len(ts),
                'distance': dists
            })
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


# ---------------------------
# Viewer / Playback (optimized)
# ---------------------------
def run_viewer(telemetry: pd.DataFrame, session):
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("F1 Telemetry Viewer")
    clock = pygame.time.Clock()
    font_small = pygame.font.SysFont(None, 18)
    font_big = pygame.font.SysFont(None, 24)

    # Prepare telemetry + drivers
    drivers = telemetry['driver'].unique().tolist()
    colors = {d: gen_color_from_string(d) for d in drivers}

    # Get driver positions from laps (live position at end of last completed lap)
    lap_df = session.laps
    driver_positions = {}
    for drv in drivers:
        pos_df = lap_df[lap_df.Driver == drv]
        if not pos_df.empty:
            last_row = pos_df.iloc[-1]
            driver_positions[drv] = last_row['Position']
        else:
            driver_positions[drv] = 99  # no laps, unknown position

    total_laps = int(lap_df['LapNumber'].max()) if not lap_df.empty else 0

    # Precompute session info
    fastest_df = session.laps[session.laps['LapTime'].notna()].nsmallest(1, 'LapTime')
    if not fastest_df.empty:
        fastest_driver = fastest_df.iloc[0]['Driver']
        fastest_time = fastest_df.iloc[0]['LapTime'].total_seconds()
        fastest_str = f"{fastest_driver} {fmt_time(fastest_time)}"
    else:
        fastest_str = "N/A"

    # Times arrays for timing
    global_min = float(telemetry['time'].min()) if not telemetry.empty else 0.0
    def convert_to_seconds(t):
        if hasattr(t, 'total_seconds'):
            return t.total_seconds()
        elif hasattr(t, 'timestamp'):
            return t.timestamp()
        else:
            return float(t)

    status_times = np.array([convert_to_seconds(t) for t in session.status_data['Time']]) - global_min if hasattr(session, 'status_data') and not session.status_data.empty else np.array([])
    status_codes = session.status_data['Status'].values if hasattr(session, 'status_data') and not session.status_data.empty else np.array([])
    message_times = np.array([convert_to_seconds(t) for t in session.race_control_messages['Time']]) - global_min if hasattr(session, 'race_control_messages') and not session.race_control_messages.empty else np.array([])
    messages = session.race_control_messages['Message'].values if hasattr(session, 'race_control_messages') and not session.race_control_messages.empty else np.array([])

    # Normalize coords to screen space once
    nx, ny = normalize_coords(telemetry['x'].values, telemetry['y'].values, MAIN_W, SCREEN_H)
    telemetry = telemetry.copy()
    telemetry['xn'] = nx
    telemetry['yn'] = ny

    # Draw track outline using first driver's telemetry points
    track_color = (220, 220, 220)  # white-ish
    if drivers:
        track_driver = drivers[0]
        track_telem = telemetry[telemetry['driver'] == track_driver].sort_values('time')
        track_xn = track_telem['xn'].values
        track_yn = track_telem['yn'].values
    else:
        track_xn = np.array([])
        track_yn = np.array([])

    # global times array for bounds
    times = np.sort(telemetry['time'].unique())
    if times.size == 0:
        print("No time points found.")
        return

    # Pre-extract per-driver numpy arrays for fast per-frame lookup
    driver_data = {}
    for drv in drivers:
        g = telemetry[telemetry['driver'] == drv].sort_values('time')
        driver_data[drv] = {
            'time': g['time'].to_numpy(dtype=float),
            'xn': g['xn'].to_numpy(dtype=int),
            'yn': g['yn'].to_numpy(dtype=int),
            'lap': g['lap'].to_numpy(dtype=int) if 'lap' in g.columns else np.zeros(len(g), dtype=int),
            'distance': g['distance'].to_numpy(dtype=float) if 'distance' in g.columns else g['time'].to_numpy(dtype=float)
        }

    # lap margin to turn lap into dominant factor in progress score
    all_distances = telemetry['distance'].values if 'distance' in telemetry.columns else telemetry['time'].values
    max_dist = float(np.nanmax(all_distances)) if len(all_distances) else 0.0
    lap_distance_margin = max(1000.0, max_dist + 100.0)

    # Playback state: use continuous simulation time driven by real dt for smoothness
    sim_time = float(times[0])
    playing = True
    speed = 4.0  # replay real-time x speed (faster for better visibility)
    point_size = DEFAULT_POINT_SIZE
    show_labels = True

    last_frame_time = time.time()

    running = True
    while running:
        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    playing = not playing
                elif ev.key == pygame.K_ESCAPE or ev.key == pygame.K_q:
                    running = False
                elif ev.key == pygame.K_UP:
                    speed = min(128.0, speed * 2.0)
                elif ev.key == pygame.K_DOWN:
                    speed = max(0.125, speed / 2.0)
                elif ev.key == pygame.K_PLUS or ev.key == pygame.K_EQUALS:
                    point_size = min(30, point_size + 1)
                elif ev.key == pygame.K_MINUS or ev.key == pygame.K_UNDERSCORE:
                    point_size = max(2, point_size - 1)
                elif ev.key == pygame.K_l:
                    show_labels = not show_labels
                elif not playing:
                    # small scrub while paused (1 second)
                    if ev.key == pygame.K_RIGHT:
                        sim_time = min(sim_time + 1.0, float(times[-1]))
                    elif ev.key == pygame.K_LEFT:
                        sim_time = max(sim_time - 1.0, float(times[0]))

        if playing:
            sim_time += dt * speed
            # clamp to available telemetry range
            if sim_time > float(times[-1]):
                sim_time = float(times[-1])
            if sim_time < float(times[0]):
                sim_time = float(times[0])

        # draw background
        screen.fill((18, 18, 20))
        for gx in range(0, MAIN_W, 200):
            pygame.draw.line(screen, (28, 28, 28), (gx, 0), (gx, SCREEN_H))
        for gy in range(0, SCREEN_H, 200):
            pygame.draw.line(screen, (28, 28, 28), (0, gy), (MAIN_W, gy))

        # draw track line
        track_color = (220, 220, 220)  # white-ish
        for i in range(1, len(track_xn)):
            pygame.draw.line(screen, track_color, (track_xn[i-1], track_yn[i-1]), (track_xn[i], track_yn[i]))

        # Build driver snapshot fast using numpy searchsorted (cheap)
        driver_stats = {}
        drivers_no_telemetry = []
        for drv, data in driver_data.items():
            t_arr = data['time']
            if t_arr.size == 0:
                # no telemetry at all
                drivers_no_telemetry.append(drv)
                continue
            # find rightmost index with time <= sim_time
            idx = np.searchsorted(t_arr, sim_time, side='right') - 1
            if idx < 0:
                # before first sample: show earliest sample (grid/start)
                idx = 0
                # note: we do not treat as no telemetry, we show earliest
            elif idx >= t_arr.size:
                idx = t_arr.size - 1

            tm = float(t_arr[idx])
            lap = int(data['lap'][idx]) if len(data['lap']) > 0 else 0
            dist = float(data['distance'][idx]) if len(data['distance']) > 0 else 0.0
            xn = int(data['xn'][idx]) if len(data['xn']) > 0 else 0
            yn = int(data['yn'][idx]) if len(data['yn']) > 0 else 0

            progress_score = lap * lap_distance_margin + dist

            driver_stats[drv] = {
                'time': tm,
                'lap': lap,
                'distance': dist,
                'xn': xn,
                'yn': yn,
                'progress_score': progress_score
            }

        # Draw driver dots (always from driver_stats so we have positions even at t=0)
        for drv, st in driver_stats.items():
            pygame.draw.circle(screen, colors.get(drv, (200, 200, 200)), (st['xn'], st['yn']), point_size)
            if show_labels:
                label = font_small.render(str(drv), True, (230, 230, 230))
                screen.blit(label, (st['xn'] + point_size + 3, st['yn'] - point_size - 3))

        # UI overlay
        header = font_big.render(
            f"Time: {sim_time:.1f}s  Speed: {speed}x  Playing: {'Yes' if playing else 'No'}  Drivers: {len(drivers)}",
            True, (220, 220, 220))
        screen.blit(header, (8, 6))
        help_line = font_small.render("SPACE play/pause | ←/→ scrub (paused) | ↑/↓ speed | +/- size | L labels | ESC quit",
                                     True, (160, 160, 160))
        screen.blit(help_line, (8, 36))

        # progress bar
        bar_w = MAIN_W - 260
        bx = 120; by = SCREEN_H - 40; bh = 10
        pygame.draw.rect(screen, (40, 40, 40), (bx, by, bar_w, bh))
        frac = (sim_time - float(times[0])) / (float(times[-1]) - float(times[0])) if float(times[-1]) > float(times[0]) else 0.0
        pygame.draw.rect(screen, (200, 80, 80), (bx, by, int(bar_w * frac), bh))

        # Compute session info for current time
        current_mask = telemetry['time'] <= sim_time
        if current_mask.any():
            current_lap = int(telemetry.loc[current_mask, 'lap'].max())
        else:
            current_lap = 0
        current_status_idx = np.searchsorted(status_times, sim_time, side='right') - 1
        current_status = status_codes[current_status_idx] if current_status_idx >= 0 and current_status_idx < len(status_codes) else '1'
        if current_status == '3':
            safety_str = "Safety Car Deployed"
        elif current_status == '7':
            safety_str = "VSC Deployed"
        elif current_status == '2':
            safety_str = "Safety Car Reported"
        elif current_status == '5':
            safety_str = "Yellow Flag"
        elif current_status == '4':
            safety_str = "Red Flag"
        elif current_status == '6' or current_status == '8':
            safety_str = "Safety Car Ending"
        else:
            safety_str = "Green Flag"
        message_idx = np.searchsorted(message_times, sim_time, side='right') - 1
        last_msg = str(messages[message_idx]) if message_idx >= 0 and message_idx < len(messages) else ""
        if len(last_msg) > 40:
            last_msg = last_msg[:37] + "..."

        # Info panel
        info_x = MAIN_W
        pygame.draw.rect(screen, (32, 28, 28), (info_x, 0, INFO_W, SCREEN_H))
        title_info = font_big.render("Session Info", True, (220, 220, 220))
        screen.blit(title_info, (info_x + 10, 10))
        y_pos_info = 50
        lap_render = font_small.render(f"Lap Count: {current_lap}", True, (230, 230, 230))
        screen.blit(lap_render, (info_x + 10, y_pos_info))
        y_pos_info += 24
        fast_render = font_small.render(f"Fastest Lap: {fastest_str}", True, (230, 230, 230))
        screen.blit(fast_render, (info_x + 10, y_pos_info))
        y_pos_info += 24
        safety_render = font_small.render(f"Safety Car: {safety_str}", True, (230, 230, 230))
        screen.blit(safety_render, (info_x + 10, y_pos_info))
        y_pos_info += 24
        msg_render = font_small.render(f"Race Control: {last_msg}", True, (200, 230, 200))
        screen.blit(msg_render, (info_x + 10, y_pos_info))

        # Sidebar & leaderboard
        sidebar_x = MAIN_W + INFO_W
        pygame.draw.rect(screen, (28, 28, 32), (sidebar_x, 0, SIDEBAR_W, SCREEN_H))
        title = font_big.render("Positions", True, (220, 220, 220))
        screen.blit(title, (sidebar_x + 10, 10))

        # sort by current progress (higher first) or final positions if finished
        if current_lap >= total_laps:
            sorted_drivers = sorted(
                [(drv, info) for drv, info in driver_stats.items()],
                key=lambda x: driver_positions.get(x[0], 99)
            )
        else:
            sorted_drivers = sorted(
                [(drv, info) for drv, info in driver_stats.items()],
                key=lambda x: x[1]['progress_score'],
                reverse=True
            )

        y_pos = 50
        if sorted_drivers:
            leader_info = sorted_drivers[0][1]
            leader_time = leader_info['time']
            leader_lap = leader_info['lap']
        else:
            leader_time = 0.0
            leader_lap = 0

        for pos, (drv, info) in enumerate(sorted_drivers, 1):
            tm = info['time']
            lap_num = info['lap']
            time_str = fmt_time(tm)

            if lap_num == leader_lap:
                if pos == 1:
                    gap_str = ""
                else:
                    gap_sec = leader_time - tm
                    if gap_sec < 0 and gap_sec > -0.001:
                        gap_sec = 0.0
                    gap_sec = max(0.0, gap_sec)
                    gap_str = f" +{fmt_time(gap_sec)}"
            else:
                lap_diff = leader_lap - lap_num
                if lap_diff <= 0:
                    gap_str = ""
                elif lap_diff == 1:
                    gap_str = " +1 lap"
                else:
                    gap_str = f" +{lap_diff} laps"

            text = font_small.render(f"{pos}. {drv} {time_str}{gap_str}", True, (230, 230, 230))
            screen.blit(text, (sidebar_x + 10, y_pos))
            y_pos += 24

        # drivers with no telemetry at all -> DNF (bottom)
        for drv in [d for d in drivers if d not in driver_stats]:
            text = font_small.render(f"-. {drv} DNF", True, (230, 120, 120))
            screen.blit(text, (sidebar_x + 10, y_pos))
            y_pos += 24

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


# ---------------------------
# GUI for selection
# ---------------------------
class F1Selector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("F1 Telemetry Selector")
        self.geometry("420x520")

        years = [str(y) for y in range(2018, datetime.datetime.now().year + 1)]

        tk.Label(self, text="Select Year:").pack(pady=5)
        self.year_var = tk.StringVar(value=str(datetime.datetime.now().year))
        self.year_cb = ttk.Combobox(self, textvariable=self.year_var, values=years, state="readonly")
        self.year_cb.pack()
        self.year_cb.bind("<<ComboboxSelected>>", self.on_year_select)

        tk.Label(self, text="Select Grand Prix:").pack(pady=5)
        self.gp_list = tk.Listbox(self, height=15)
        self.gp_list.pack(fill=tk.BOTH, expand=True)

        self.gps = []  # list of (official_name, round_num)

        tk.Label(self, text="Select Session:").pack(pady=5)
        self.session_var = tk.StringVar(value="R")
        self.session_cb = ttk.Combobox(self, textvariable=self.session_var,
                                       values=["R", "Q", "FP1", "FP2", "FP3"], state="readonly")
        self.session_cb.pack()

        self.btn = tk.Button(self, text="Launch Viewer", command=self.launch, height=2)
        self.btn.pack(pady=10)

        self.on_year_select(None)

    def on_year_select(self, event):
        year = int(self.year_var.get())
        self.gps = []
        try:
            sched = ff1.get_event_schedule(year)
            for idx, row in sched.iterrows():
                rn = row.get('RoundNumber', row.get('Round Number', row.get('roundNumber', 0)))
                try:
                    rn_val = int(rn)
                except Exception:
                    rn_val = 0
                if rn_val > 0:
                    name = row.get('OfficialName', row.get('EventName', row.get('Event Name', 'Unknown')))
                    self.gps.append((name, rn_val))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load schedule for {year}: {e}")

        self.gp_list.delete(0, tk.END)
        for name, rnd in self.gps:
            self.gp_list.insert(tk.END, f"{name} ({rnd})")

    def launch(self):
        sel = self.gp_list.curselection()
        if not sel or not self.gps:
            return
        name, rnd = self.gps[sel[0]]
        year = int(self.year_var.get())
        sess = self.session_var.get()

        # Create loading progress window
        self.loading_window = tk.Toplevel(self)
        self.loading_window.title("Loading F1 Telemetry")
        self.loading_window.geometry("400x150")
        self.loading_window.transient(self)

        tk.Label(self.loading_window, text="Loading session data...", font=("Arial", 12)).pack(pady=10)
        self.progress_var = tk.StringVar(value="Initializing...")
        self.progress_label = tk.Label(self.loading_window, textvariable=self.progress_var, font=("Arial", 10))
        self.progress_label.pack()
        self.progress_bar = ttk.Progressbar(self.loading_window, mode="determinate", length=300, maximum=100)
        self.progress_bar.pack(pady=10)

        # Make loading window modal
        self.loading_window.grab_set()
        self.loading_window.focus_set()

        # Start loading in background thread
        self.loading_thread = threading.Thread(target=self.load_data, args=(name, rnd, year, sess))
        self.loading_thread.start()

    def load_data(self, name, rnd, year, sess):
        try:
            self.update_progress("Setting up cache...", progress=10)
            cache = "ff1_cache"
            os.makedirs(cache, exist_ok=True)
            ff1.Cache.enable_cache(cache)

            self.update_progress(f"Loading session: {year} {name} {sess}...", progress=20)
            session = ff1.get_session(year, rnd, sess)

            self.update_progress("Loading telemetry data...", progress=60)
            session.load(telemetry=True, weather=False, messages=True)

            self.update_progress("Processing telemetry...", progress=100)
            telemetry = collect_session_telemetry(session)

            if telemetry is None:
                self.update_progress("No telemetry data found", error=True)
                return

            # Close loading window and destroy main window in main thread
            self.after(100, lambda: self.finish_loading(telemetry, session))

        except Exception as e:
            self.update_progress(f"Error: {str(e)[:50]}...", error=True)

    def update_progress(self, message, error=False, progress=None):
        def _update():
            self.progress_var.set(f"{message}{' (' + str(progress) + '%)' if progress is not None else ''}")
            if progress is not None:
                self.progress_bar['value'] = progress
            if error:
                self.progress_label.config(fg="red")
        self.after(0, _update)

    def finish_loading(self, telemetry, session):
        self.loading_window.destroy()
        self.destroy()
        run_viewer(telemetry, session)


def main():
    app = F1Selector()
    app.mainloop()


if __name__ == "__main__":
    main()
