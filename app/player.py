from tkinter import Toplevel, Label, Scale, HORIZONTAL, Button, Frame, LEFT, BOTTOM, BOTH, DISABLED, NORMAL
from audio import OLAEngine,PVEngine, HybridEngine

import random
import numpy as np
import math


engine_map = {
    "OLA": OLAEngine,
    "PVEngine": PVEngine,
    "Hybrid": HybridEngine
}
class Player:
    def __init__(self, master, on_close=None, userinfo = None):
        self.on_close = on_close
        self.window = Toplevel(master)
        self.window.title("Music Control")
        self.window.geometry('900x350')  # Wider to fit all three players
        self.window.protocol("WM_DELETE_WINDOW", self.handle_close)

        self.players = []

        # Horizontal container for players
        container = Frame(self.window)
        container.pack(padx=10, pady=10)
        
        self.current_playing = -1
        self.current_choice = -1
        
        # Create 3 horizontally-aligned players
        self.userinfo = userinfo
        self.filename, engine_pair = userinfo.get_next_task()  # Placeholder for audio file
            
        self.engines = [engine_map[name](self.filename) for name in engine_pair]
        self.engine_names = engine_pair.copy()
        
        permute = np.random.permutation(len(self.engines))
        self.engines = [self.engines[i] for i in permute]
        self.engine_names = [self.engine_names[i] for i in permute]
        
        self.alphas = [1.0] * len(self.engines)  # Initial alpha values for each engine
        self.engine = None
        for i in range(len(self.engines)):
            self.create_player_section(container, i + 1)
            self.engines[i].on_complete = self.make_on_complete(self.players[i], i, self.window)
        
        # submit
        
        bottom = Frame(
            self.window,
            # bg='#FFFFFF'
        )
        bottom.pack(side=BOTTOM, fill=BOTH, expand=True)
        self.submit_btn = Button(bottom, text="Submit", command=lambda: self.toggle_submit(self.current_choice))
        self.submit_btn.config(state=DISABLED)
        self.submit_btn.pack()
        
    def create_player_section(self, parent, index):
        frame = Frame(parent, bd=2, relief='groove', padx=10, pady=10, bg="grey")
        frame.pack(side=LEFT, padx=10, pady=10)

        Label(frame, text=f"Player {index}", font=("Arial", 12, 'bold')).pack()

        # Volume Slider
        factor_label = Label(frame, text="Factor: 1.0")
        factor_label.pack()
        factor_slider = Scale(frame, from_=math.log(2), to=math.log(0.5), orient=HORIZONTAL, 
                        resolution=0.001,label="Adjust Speed",showvalue=0,  # This controls the actual step size
                        # tickinterval=0.1, # This controls tick mark display (optional)
                        command=lambda val, i=index, fl=factor_label: self.update_volume(i, math.exp(float(val)),fl))
        factor_slider.set(math.log(1.0))
        factor_slider.pack()

        # stop Button
        stop_btn = Button(frame, text="Play", command=lambda i=index: self.toggle_stop(i))
        stop_btn.pack(pady=5)

        # # Rating Slider
        # rating = 5
        # Label(frame, text="Rating (1–10)").pack()
        # rating_slider = Scale(frame, from_=1, to=10, orient=HORIZONTAL, command=lambda val,i=index: self.update_rating(i,val))
        # rating_slider.pack()
        
        # stop Button
        choice_btn = Button(frame, text="Pick", command=lambda i=index: self.toggle_choice(i))
        choice_btn.pack(pady=5)

        self.players.append({
            'frame':frame,
            'factor_slider': factor_slider,
            'stop_button': stop_btn,
            # 'rating_slider': rating_slider,
            # 'rating': rating,
            'choice_button': choice_btn,
            'chose': False,
            'stopped': True,
        })

    def update_volume(self, player_idx, value, factor_label):
        print(f"Player {player_idx}: ALPHA = {value}")
        factor_label.config(text=f"Factor: {(1/value):.1f}")
        self.alphas[player_idx-1] = float(value)
        if self.current_playing == player_idx - 1:
            if self.engine:
                self.engine.set_alpha(float(value))
        # TODO: Connect to audio backend
        
    def update_rating(self,player_idx, value):
        print(f"Player {player_idx}: Rating = {value}")

    def toggle_submit(self, choice):
        # print('clicked_submit',choice, self.current_choice)
        if choice == -1:
            return
        self.userinfo.log(
            self.filename,
            sorted(self.engine_names),
            self.engine_names[choice]
        )
        self.handle_close()
    
    def safe_stop_engine(self):
        if self.engine:
            import threading
            threading.Thread(target=self.engine.stop).start()
            self.engine = None
        
    def make_on_complete(self, player, player_idx, window):
        # return None
        # return None
        return lambda: window.after(0, lambda: self._on_engine_end(player, player_idx))
        # return lambda: self._on_engine_end(player_idx)

    def _on_engine_end(self, player, player_idx):
        print(f"Playback ended callback called, player {player} will be called")
        try:
            if str(player['stop_button']) not in self.window.tk.call("winfo", "children", self.window):
                print("Button was destroyed, skipping callback.")
                return
            player['stopped'] = True
            player['stop_button'].config(text="Play")
            player['frame'].config(bg="grey")
            if player_idx == self.current_playing:
                if self.engine:
                    self.safe_stop_engine()
                self.current_playing = -1
        except Exception as e:
            print(f"⚠️ Exception in _on_engine_end: {e}")
    
    def toggle_choice(self, player_idx):
        player = self.players[player_idx - 1]
        player['chose'] = not player['chose']
        state = "Unpick" if player['chose'] else "Pick"
        # print(f"Player {player_idx}: {state}")
        player['choice_button'].config(text="Unpick" if player['chose'] else "Pick")
        
        if self.current_choice == player_idx - 1:
            # Unpick current player
            player['chose'] = False
            player['choice_button'].config(text="Pick")
            self.current_choice = -1
        else:
            # Unpick previous choice
            if self.current_choice != -1:
                prev_player = self.players[self.current_choice]
                prev_player['chose'] = False
                prev_player['choice_button'].config(text="Pick")

            # Pick new player
            player['chose'] = True
            player['choice_button'].config(text="Unpick")
            self.current_choice = player_idx - 1

        # Enable or disable submit button
        if self.current_choice != -1:
            self.submit_btn.config(state=NORMAL)
        else:
            self.submit_btn.config(state=DISABLED)
    
    def toggle_stop(self, player_idx):
        player = self.players[player_idx - 1]
        is_stopped = player['stopped']

        if self.current_playing == player_idx - 1:
            player['stopped'] = True
            player['stop_button'].config(text="Play")
            player['frame'].config(bg="grey")
            if self.engine:
                self.safe_stop_engine()
                self.engine = None
            self.current_playing = -1
            return

        # Stop previously playing engine if any
        if self.current_playing != -1:
            prev_player = self.players[self.current_playing]
            prev_player['stopped'] = True
            prev_player['stop_button'].config(text="Play")
            prev_player['frame'].config(bg="grey")
            if self.engine:
                self.safe_stop_engine()
                # self.engine.stop()
                # self.engine = None

        # Start new engine for this player
        player['stopped'] = False
        player['stop_button'].config(text="Stop")
        player['frame'].config(bg="green")
        self.current_playing = player_idx - 1
        self.engine = self.engines[self.current_playing]
        self.engine.set_alpha(self.alphas[self.current_playing])
        self.engine.start()
        

    def handle_close(self):
        if self.engine:
            self.safe_stop_engine()
            # self.engine.stop()
        if self.on_close:
            self.on_close()
        self.window.destroy()
