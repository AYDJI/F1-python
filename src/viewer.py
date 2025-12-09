import time
import numpy as np
import pandas as pd
import pygame

from .config import MAIN_W, INFO_W, SIDEBAR_W, SCREEN_W, SCREEN_H, FPS, DEFAULT_POINT_SIZE
from .helpers import to_epoch_seconds, fmt_time, gen_color_from_string, normalize_coords


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

    # Helper: convert arrays of status/message times into epoch seconds and align to telemetry if possible
    telemetry_has_abs = 'abs_time' in telemetry.columns and telemetry['abs_time'].notna().any()
    telemetry_abs_min = float(telemetry['abs_time'].min()) if telemetry_has_abs else None

    # Convert status_data times -> relative seconds in telemetry timebase (preferred) or session-relative fallback
    if hasattr(session, 'status_data') and not session.status_data.empty:
        sd_times = session.status_data['Time'].values
        sd_epoch = to_epoch_seconds(sd_times)
        if telemetry_has_abs:
            status_times = sd_epoch - telemetry_abs_min
        else:
            # fallback: make status times relative to first status timestamp
            if np.all(np.isnan(sd_epoch)):
                status_times = np.array([])
            else:
                status_times = sd_epoch - sd_epoch[0]
        status_codes = session.status_data['Status'].values
    else:
        status_times = np.array([])
        status_codes = np.array([])

    # Convert race_control_messages similarly
    if hasattr(session, 'race_control_messages') and not session.race_control_messages.empty:
        rm_times = session.race_control_messages['Time'].values
        rm_epoch = to_epoch_seconds(rm_times)
        if telemetry_has_abs:
            message_times = rm_epoch - telemetry_abs_min
        else:
            if np.all(np.isnan(rm_epoch)):
                message_times = np.array([])
            else:
                message_times = rm_epoch - rm_epoch[0]
        messages = session.race_control_messages['Message'].values
    else:
        message_times = np.array([])
        messages = np.array([])

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

        # Determine current status index using the aligned status_times
        current_status_idx = np.searchsorted(status_times, sim_time, side='right') - 1
        current_status = status_codes[current_status_idx] if current_status_idx >= 0 and current_status_idx < len(status_codes) else '1'
        current_status = str(current_status)

        if current_status == '3' or current_status == '3.0':
            safety_str = "Safety Car Deployed"
        elif current_status == '7' or current_status == '7.0':
            safety_str = "VSC Deployed"
        elif current_status == '2' or current_status == '2.0':
            safety_str = "Safety Car Reported"
        elif current_status == '5' or current_status == '5.0':
            safety_str = "Yellow Flag"
        elif current_status == '4' or current_status == '4.0':
            safety_str = "Red Flag"
        elif current_status == '6' or current_status == '6.0' or current_status == '8' or current_status == '8.0':
            safety_str = "Safety Car Ending"
        else:
            safety_str = "Green Flag"

        # Race control last message (aligned)
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
