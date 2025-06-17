from tkinter import Toplevel, Label, Scale, HORIZONTAL, Button, Frame, LEFT
from audio import OLAEngine,PVEngine, HybridEngine

import random
import numpy as np

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
        
        self.current = -1
        self.engines = [OLAEngine, PVEngine, HybridEngine]
        self.engine_names = ["OLA", "Phase Vocoder", "Hybrid"]
        
        # Create 3 horizontally-aligned players
        self.userinfo = userinfo
        self.filename = userinfo.assigned[random.choice(userinfo.todo)]  # Placeholder for audio file
        print(userinfo.assigned)
        
        permute = np.random.permutation(len(self.engines))
        self.engines = [self.engines[i] for i in permute]
        self.engine_names = [self.engine_names[i] for i in permute]
        
        self.alphas = [1.0, 1.0, 1.0]  # Initial alpha values for each engine
        self.engine = None
        for i in range(3):
            # self.engines[i].start()  # Start each engine
            # self.engines[i].set_stopped(True)
            self.create_player_section(container, i + 1)
            

    def create_player_section(self, parent, index):
        frame = Frame(parent, bd=2, relief='groove', padx=10, pady=10)
        frame.pack(side=LEFT, padx=10, pady=10)

        Label(frame, text=f"Player {index}", font=("Arial", 12, 'bold')).pack()

        # Volume Slider
        Label(frame, text="Factor").pack()
        factor_slider = Scale(frame, from_=0.5, to=2, orient=HORIZONTAL, 
                        resolution=0.05,  # This controls the actual step size
                        # tickinterval=0.1, # This controls tick mark display (optional)
                        command=lambda val, i=index: self.update_volume(i, val))
        factor_slider.set(1.0)
        factor_slider.pack()

        # stop Button
        stop_btn = Button(frame, text="Play", command=lambda i=index: self.toggle_stop(i))
        stop_btn.pack(pady=5)

        # Rating Slider
        rating = 5
        Label(frame, text="Rating (1–10)").pack()
        rating_slider = Scale(frame, from_=1, to=10, orient=HORIZONTAL, command=lambda val,i=index: self.update_rating(i,val))
        rating_slider.pack()

        self.players.append({
            'factor_slider': factor_slider,
            'stop_button': stop_btn,
            'rating_slider': rating_slider,
            'rating': rating,
            'stopped': True,
        })

    def update_volume(self, player_idx, value):
        print(f"Player {player_idx}: ALPHA = {value}")
        self.alphas[player_idx-1] = float(value)
        if self.current == player_idx - 1:
            if self.engine:
                self.engine.set_alpha(float(value))
        # TODO: Connect to audio backend
        
    def update_rating(self,player_idx, value):
        print(f"Player {player_idx}: Rating = {value}")

    def toggle_stop(self, player_idx):
        player = self.players[player_idx - 1]
        player['stopped'] = not player['stopped']
        state = "stopped" if player['stopped'] else "Playing"
        print(f"Player {player_idx}: {state}")
        player['stop_button'].config(text="Play" if player['stopped'] else "stop")
        # self.engines[player_idx - 1].set_stopped(player['stopped'])
        # TODO: Control audio stream
        
        # self.players[self.current]['stop_button'].config(text="Play")
        if self.current==player_idx - 1:
            self.engine.stop()
            self.current=-1
            return
        self.current = player_idx - 1
        self.engine = self.engines[self.current](self.filename)
        self.engine.set_alpha(self.alphas[self.current])
        self.engine.start()

    def handle_close(self):
        if self.on_close:
            self.on_close()
        self.window.destroy()
