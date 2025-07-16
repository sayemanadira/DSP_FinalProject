import tkinter as tk
from tkinter import (
    Toplevel, Label, Scale, HORIZONTAL, Button, Frame,
    LEFT, BOTTOM, BOTH, DISABLED, NORMAL, TOP, CENTER
)
import random
import math
import numpy as np

from audio import OLAEngine, PVEngine, HybridEngine, OPTEngine
from userinfo import get_user_info

# Mapping engine names to classes
engine_map = {
    "OLA": OLAEngine,
    "PV": PVEngine,
    "Hybrid": HybridEngine,
    "OPT0.1": OPTEngine,
    "OPT0.2": OPTEngine,
    "OPT0.25": OPTEngine,
    "OPT0.3": OPTEngine,
    "OPT0.35": OPTEngine,
    "OPT0.4": OPTEngine,
    "OPT0.5": OPTEngine,
    "OPT0.6": OPTEngine,
    "OPT0.7": OPTEngine,
    # "OPT0.25": OPTEngine,
    # "OPT0.9": OPTEngine,
    # "OPT0.1": OPTEngine,
    # "OPT0.08": OPTEngine,
    # "OPT0.075": OPTEngine,
    # "OPT0.125": OPTEngine,
}


class Player:
    def __init__(self, master, user_id, on_close=None):
        self.user_id = user_id
        self.on_close = on_close

        self.window = Toplevel(master)
        self.window.title("Music Control")
        self.window.geometry("900x500")
        self.window.protocol("WM_DELETE_WINDOW", self.handle_close)

        self.init_everything()

    def init_everything(self):
        self.players = []
        self.userinfo = get_user_info(self.user_id)

        self.current_playing = -1
        self.current_choice = -1
        self.same_selected = False
        self.engine = None

        self.setup_instruction()
        task = self.userinfo.get_next_task()

        if task is None:
            self.handle_close()
            return

        self.filename, engine_pair = task
        self.engines, self.engine_names = self.prepare_engines(engine_pair)
        self.alphas = [1.0] * len(self.engines)

        self.choice_vars = []
        self.setup_players()

    def setup_instruction(self):
        instruction_text = (
            "Which player has better quality?\n"
            "1. Try to find the option with the fewest audio artifacts\n"
            "2. Move the 'Adjust Tempo' slider in real-time while listening\n"
            "3. Check the box to indicate which player has the best quality, or whether they both sound the same\n"
            "4. Click 'Submit' to save your choice and move to the next pair."
        )

        frame = Frame(self.window)
        frame.pack(padx=10, pady=5, fill="x")
        Label(frame, text=instruction_text, justify=CENTER).pack(side=TOP)

    def prepare_engines(self, engine_pair):
        engines = []
        names = engine_pair.copy()

        for name in names:
            if name not in engine_map:
                raise ValueError(f"Unknown engine name: {name}")
            engine_class = engine_map[name]
            if name.startswith("OPT"):
                alpha = float(name[3:])
                engines.append(engine_class(self.filename, beta=alpha))
            else:
                engines.append(engine_class(self.filename))

        perm = np.random.permutation(len(engines))
        return [engines[i] for i in perm], [names[i] for i in perm]

    def setup_players(self):
        container = Frame(self.window)
        container.pack(padx=10, pady=10)

        num_players = len(self.engines)
        for i in range(num_players):
            self.create_player_section(container, i + 1)
            self.engines[i].on_complete = self.make_on_complete(self.players[i], i, self.window)
            self.set_frame_background(self.players[i]["frame"], "grey")

        bottom = Frame(self.window)
        bottom.pack(side=BOTTOM, fill=BOTH, expand=True)

        # controls = Frame(bottom)
        # controls.pack()
        # Create radio-style checkboxes
        self.choice_vars = [tk.BooleanVar() for _ in range(3)]

        # Container for choices
        choice_frame = Frame(bottom)
        choice_frame.pack(pady=10)

        self.choice_checkbuttons = []

        labels = ["Player 1", "Player 2", "Sound the Same"]
        for i, label in enumerate(labels):
            cb = tk.Checkbutton(
                choice_frame,
                text=label,
                variable=self.choice_vars[i],
                command=lambda idx=i: self.handle_choice_selection(idx)
            )
            cb.pack(side=LEFT, padx=20)
            self.choice_checkbuttons.append(cb)

        # Submit button
        self.submit_btn = Button(bottom, text="Submit", command=self.submit_choice, state=DISABLED)
        self.submit_btn.pack(pady=10)

    def create_player_section(self, parent, index):
        frame = Frame(parent, bd=2, relief="groove", padx=10, pady=10)  # ← no bg argument
        frame.pack(side=LEFT, padx=10, pady=10)

        Label(frame, text=f"Player {index}", font=("Arial", 12, "bold")).pack()

        factor_label = Label(frame, text="Factor: 1.0")
        factor_label.pack()

        slider = Scale(
            frame,
            from_=math.log(2),
            to=math.log(0.5),
            resolution=0.001,
            orient=HORIZONTAL,
            label="Adjust Tempo",
            showvalue=0,
            command=lambda val, i=index, fl=factor_label: self.update_alpha(i, math.exp(float(val)), fl)
        )
        slider.set(math.log(1.0))
        slider.pack()

        play_btn = Button(frame, text="Play", command=lambda i=index: self.toggle_stop(i))
        play_btn.pack(pady=5)

        self.players.append({
            "frame": frame,
            "factor_slider": slider,
            "stop_button": play_btn,
            "chose": False,
            "stopped": True,
        })


    def update_alpha(self, player_idx, value, label):
        label.config(text=f"Factor: {(1 / value):.1f}")
        self.alphas[player_idx - 1] = value
        if self.current_playing == player_idx - 1 and self.engine:
            self.engine.set_alpha(value)

    def set_frame_background(self, frame, color):
        frame.config(bg=color)
        for widget in frame.winfo_children():
            try:
                widget.config(bg=color)
            except:
                pass  # Some widgets may not support bg config (like Scale label), safe to skip
    def toggle_stop(self, player_idx):
        player = self.players[player_idx - 1]

        if self.current_playing == player_idx - 1:
            self.stop_engine(player)
            return

        if self.current_playing != -1:
            self.stop_engine(self.players[self.current_playing])

        player["stopped"] = False
        player["stop_button"].config(text="Stop")
        # player["frame"].config(bg="green")
        self.set_frame_background(player["frame"], "green")

        self.current_playing = player_idx - 1
        self.engine = self.engines[self.current_playing]
        self.engine.set_alpha(self.alphas[self.current_playing])
        self.engine.start()

    def stop_engine(self, player):
        player["stopped"] = True
        player["stop_button"].config(text="Play")
        # player["frame"].config(bg="grey")
        self.set_frame_background(player["frame"], "grey")
        if self.engine:
            self.safe_stop_engine()
            self.engine = None
        self.current_playing = -1
    
    def handle_choice_selection(self, selected_idx):
        for i, var in enumerate(self.choice_vars):
            var.set(i == selected_idx)
        
        if selected_idx == 2:
            self.same_selected = True
            self.current_choice = -1
        else:
            self.same_selected = False
            self.current_choice = selected_idx
        
        self.submit_btn.config(state=NORMAL)

    def toggle_choice(self, player_idx, var):
        idx = player_idx - 1

        if var.get():
            # Uncheck all others
            for i, v in enumerate(self.choice_vars):
                if i != idx:
                    v.set(False)
            self.current_choice = idx
            self.same_selected = False
            self.same_btn.config(text="Sound the Same")
            self.submit_btn.config(state=NORMAL)
        else:
            self.current_choice = -1
            self.submit_btn.config(state=DISABLED)

    def toggle_same(self):
        self.same_selected = not self.same_selected

        if self.same_selected:
            self.same_btn.config(text="Not the Same")
            for var in self.choice_vars:
                var.set(False)
            self.current_choice = -1
            self.submit_btn.config(state=NORMAL)
        else:
            self.same_btn.config(text="Sound the Same")
            self.submit_btn.config(state=DISABLED)


    def submit_choice(self):
        choice = "SAME" if self.same_selected else self.engine_names[self.current_choice]
        self.userinfo.log(self.filename, sorted(self.engine_names), choice)
        if self.engine:
            self.safe_stop_engine()

        for widget in self.window.winfo_children():
            widget.destroy()

        self.players.clear()
        self.engine = None
        self.engines.clear()
        self.engine_names.clear()
        self.alphas.clear()
        self.current_choice = -1
        self.current_playing = -1
        self.same_selected = False
        self.same_btn = None

        self.init_everything()


    def make_on_complete(self, player, idx, window):
        return lambda: window.after(0, lambda: self._on_engine_end(player, idx))

    def _on_engine_end(self, player, idx):
        try:
            player["stopped"] = True
            player["stop_button"].config(text="Play")
            # player["frame"].config(bg="grey")
            self.set_frame_background(player["frame"], "grey")
            if idx == self.current_playing:
                self.safe_stop_engine()
                self.current_playing = -1
        except Exception as e:
            print(f"⚠️ Error on playback end: {e}")

    def safe_stop_engine(self):
        if self.engine:
            import threading
            threading.Thread(target=self.engine.stop).start()

    def handle_close(self):
        if self.engine:
            self.safe_stop_engine()
        if self.on_close:
            self.on_close()
        self.window.destroy()
