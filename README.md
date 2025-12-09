# F1 Telemetry Viewer

A real-time F1 telemetry visualization tool built with Python, FastF1, and Pygame. This application allows you to replay F1 race sessions with live-updating driver positions, track visualization, and session information.

## Features

- **Real-time Telemetry Playback**: Replay F1 sessions with synchronized telemetry data
- **Live Position Tracking**: Dynamic leaderboard that updates driver positions during the race based on progress
- **Track Visualization**: White track outline drawn from telemetry data
- **Session Information Panel**: Displays lap count, fastest lap, safety car status, and race control messages
- **Final Race Results**: Automatically displays official final positions from FastF1 data once the race is complete
- **Interactive Controls**: Play/pause, speed control, scrubbing, point size adjustment, label toggling

## Project Structure

```
F1-python/
├── main.py          # Entry point for the application
├── src/
│   ├── config.py    # Configuration variables (screen dimensions, FPS, etc.)
│   ├── helpers.py   # Utility functions (time formatting, color generation, coordinate normalization)
│   ├── telemetry.py # Functions for collecting and processing F1 telemetry data
│   ├── viewer.py    # Pygame-based telemetry viewer and playback engine
│   └── selector.py  # Tkinter GUI for selecting F1 sessions
├── requirements.txt # Python dependencies
├── README.md        # This file
└── .gitignore       # Git ignore rules
```

The codebase is organized into a `src/` directory for better modularity and maintainability.

## Requirements

### System Requirements
- Python 3.11.9 (tested and working 100% - other versions not tested)
- Windows/macOS/Linux

### Dependencies
- `fastf1` - F1 data API
- `pygame` - Graphics and GUI
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `tkinter` - Selection interface

## Installation

1. Clone or download the repository
2. Install Python 3.11.9 if not already installed
3. Install the required packages:

```bash
pip install fastf1 pygame numpy pandas
```

Note: tkinter is typically included with Python installations.

### Installing from requirements file

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python main.py
```

### Interface Overview

1. **Session Selection**: Choose year, Grand Prix, and session type (R for Race, Q for Qualifying, etc.)
2. **"Launch Viewer"**: Start the telemetry viewer for the selected session

### Controls

- **SPACE**: Play/Pause
- **ESC** or **Q**: Quit
- **↑/↓**: Increase/Decrease playback speed
- **←/→** (when paused): Scrub 1 second forward/backward
- **+/-**: Adjust driver dot size
- **L**: Toggle driver labels

### Layout

- **Left Panel (600px)**: Track visualization with driver positions and white track outline
- **Middle Panel (300px)**: Session information (lap count, fastest lap, safety car, race control)
- **Right Panel (240px)**: Live position leaderboard

## How It Works

### Data Collection
1. FastF1 API loads session data including telemetry, lap times, positions, and messages
2. Telemetry data from all drivers is collected and synchronized by time
3. Track coordinates are normalized and scaled for screen display

### Playback Engine
1. Telemetry is pre-processed into driver-specific data arrays for fast lookup
2. Real-time simulation driven by wall-clock time for smooth playback
3. Driver positions calculated using progress scores (lap * margin + distance)
4. Dynamic switching between live progress tracking and official race results

### Real-time Updates
- **During Race**: Leaderboard shows current positions based on race progress
- **At Race End**: Automatically switches to official final positions once all drivers complete the race
- **Session Info**: Safety car status, race control messages, lap counts update in real-time based on session data

### Performance Optimizations
- Pre-normalized coordinates for efficient rendering
- Numpy arrays for fast telemetry lookups
- Progress score-based sorting avoids expensive dataframe operations per frame

## Data Sources

- **Driver Positions**: FastF1 session.laps with position data per lap
- **Telemet ry**: X/Y coordinates from car telemetry systems
- **Timing**: Synchronized using session time offsets
- **Status Messages**: Safety car deployments, yellow flags from status_data
- **Race Control**: Official messages from race control team

## Technical Details

### Configuration
Screen layout can be adjusted in the `main.py` file:
- `MAIN_W = 600`  # Track area width
- `INFO_W = 300`  # Session info panel width
- `SIDEBAR_W = 240` # Positions panel width
- `SCREEN_H = 800` # Screen height
- `FPS = 60`      # Target frame rate

### Data Synchronization
- All timing data is normalized to simulation time (0 = race start)
- Status messages and race control messages are time-filtered for relevance
- Track coordinates are projected onto 2D screen space with padding

### Error Handling
- Graceful handling of missing telemetry data
- Defaults for drivers without telemetry (shown as "DNF")
- Caching system reduces API load for repeated sessions

## Limitations

- Requires active internet connection for FastF1 data (first-time load)
- Telemetry quality depends on session data availability
- Designed primarily for race sessions; qualifying sessions have limited data
- Python 3.11.9 confirmed working - other versions may have compatibility issues

## License

This project is for educational and personal use only. F1 data is provided through the FastF1 API with proper attribution to Formula 1.
