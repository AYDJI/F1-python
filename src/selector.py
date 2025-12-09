import datetime
import threading
import os

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:
    print("tkinter is required for GUI selection")
    raise

try:
    import fastf1 as ff1
except Exception:
    print("fastf1 is required. Install with: pip install fastf1")
    raise

from .viewer import run_viewer
from .telemetry import collect_session_telemetry


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
            self.update_progress(f"Error: {str(e)[:200]}...", error=True)

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
