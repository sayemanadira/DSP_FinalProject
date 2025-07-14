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
    "OPT0.2": OPTEngine,
    "OPT0.3": OPTEngine,
    "OPT0.4": OPTEngine,
    "OPT0.5": OPTEngine,
    "OPT0.6": OPTEngine,
    "OPT0.7": OPTEngine,
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

        self.setup_players()

    def setup_instruction(self):
        instruction_text = (
            "Thank you for participating in our study! \n"
            "Your task is to compare two audio algorithms that change music speed.\n\n"
            "1. Use the 'Play' buttons and 'Adjust Speed' sliders to listen to each option.\n"
            "2. Click 'Pick' for the one that sounds better to you.\n"
            "3. If they sound the same, use the 'Sound the Same' button.\n"
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
                engines.append(engine_class(self.filename, min_alpha=alpha))
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

            # if i == num_players // 2 - 1:
            #     self.same_btn = Button(container, text="Models Sound the Same", command=self.toggle_same)
            #     self.same_btn.pack(side=LEFT, padx=20, pady=10)

        bottom = Frame(self.window)
        bottom.pack(side=BOTTOM, fill=BOTH, expand=True)
        
        controls = Frame(bottom)
        controls.pack()

        self.pick_buttons = []
        i=0
        pick_btn = Button(controls, text=f"Pick Player {i + 1}", command=lambda i=i: self.toggle_choice(i + 1))
        pick_btn.pack(side=LEFT, padx=10)
        self.pick_buttons.append(pick_btn)
        self.same_btn = Button(controls, text="Sound the Same", command=self.toggle_same)
        self.same_btn.pack(side=LEFT, padx=20)
        i=1
        pick_btn = Button(controls, text=f"Pick Player {i + 1}", command=lambda i=i: self.toggle_choice(i + 1))
        pick_btn.pack(side=LEFT, padx=10)
        self.pick_buttons.append(pick_btn)

        self.submit_btn = Button(bottom, text="Submit", command=self.submit_choice, state=DISABLED)
        self.submit_btn.pack(pady=10)

    def create_player_section(self, parent, index):
        frame = Frame(parent, bd=2, relief="groove", padx=10, pady=10, bg="grey")
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
            label="Adjust Speed",
            showvalue=0,
            command=lambda val, i=index, fl=factor_label: self.update_alpha(i, math.exp(float(val)), fl)
        )
        slider.set(math.log(1.0))
        slider.pack()

        play_btn = Button(frame, text="Play", command=lambda i=index: self.toggle_stop(i))
        play_btn.pack(pady=5)

        # pick_btn = Button(frame, text="Pick", command=lambda i=index: self.toggle_choice(i))
        # pick_btn.pack(pady=5)

        self.players.append({
            "frame": frame,
            "factor_slider": slider,
            "stop_button": play_btn,
            # "choice_button": pick_btn,
            "chose": False,
            "stopped": True,
        })

    def update_alpha(self, player_idx, value, label):
        label.config(text=f"Factor: {(1 / value):.1f}")
        self.alphas[player_idx - 1] = value
        if self.current_playing == player_idx - 1 and self.engine:
            self.engine.set_alpha(value)

    def toggle_stop(self, player_idx):
        player = self.players[player_idx - 1]

        if self.current_playing == player_idx - 1:
            self.stop_engine(player)
            return

        if self.current_playing != -1:
            self.stop_engine(self.players[self.current_playing])

        player["stopped"] = False
        player["stop_button"].config(text="Stop")
        player["frame"].config(bg="green")

        self.current_playing = player_idx - 1
        self.engine = self.engines[self.current_playing]
        self.engine.set_alpha(self.alphas[self.current_playing])
        self.engine.start()

    def stop_engine(self, player):
        player["stopped"] = True
        player["stop_button"].config(text="Play")
        player["frame"].config(bg="grey")
        if self.engine:
            self.safe_stop_engine()
            self.engine = None
        self.current_playing = -1

    def toggle_choice(self, player_idx):
        idx = player_idx - 1
        player = self.players[idx]


        if self.current_choice == idx:
            player["chose"] = False
            self.pick_buttons[player_idx - 1].config(text=f"Pick Player {player_idx}")
            self.current_choice = -1
        else:
            if self.current_choice != -1:
                prev = self.players[self.current_choice]
                prev["chose"] = False
                self.pick_buttons[self.current_choice].config(text=f"Pick Player {self.current_choice+1}")

            player["chose"] = True
            self.pick_buttons[player_idx - 1].config(text=f"Unpick Player {player_idx}")
            self.current_choice = idx

        if self.same_selected:
            self.same_selected = False
            self.same_btn.config(text="Sound the Same")

        self.submit_btn.config(state=NORMAL if self.current_choice != -1 else DISABLED)

    def toggle_same(self):
        self.same_selected = not self.same_selected

        if self.same_selected:
            if self.current_choice != -1:
                prev = self.players[self.current_choice]
                prev["chose"] = False
                # prev["choice_button"].config(text="Pick")
                self.pick_buttons[self.current_choice].config(text=f"Pick Player {self.current_choice+1}")
                self.current_choice = -1

            self.same_btn.config(text="Not the Same")
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
            player["frame"].config(bg="grey")
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
