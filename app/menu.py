from tkinter import Tk, Label, Entry, Button, Menu
from player import Player
from userinfo import get_user_info
from tkinter import Menu, Label, Entry, Button, Text, END, DISABLED, NORMAL

class MainMenu:
    def __init__(self, root):
        self.user_info = None
        
        self.root = root
        self.root.title("Main Menu")
        self.root.geometry('700x400')  # Add some height for the text box
        self.player_window = None  # Track player window

        # Menu bar
        self.menu = Menu(self.root)
        self.file_menu = Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label='File', menu=self.file_menu)
        self.root.config(menu=self.menu)

        # UI Elements
        self.lbl = Label(self.root, text="User ID:")
        self.lbl.grid(column=0, row=0, padx=10, pady=10)

        # Instruction Text box (read-only)
        self.instructions = Text(self.root, height=40, width=80, wrap='word')
        self.instructions.grid(column=0, row=1, columnspan=3, padx=10, pady=10)

        # Insert your commentary/instructions
        instruction_text = (
            "Welcome! Your job is to evaluate the quality of several music signal procesing algorithms.\n\n"
            "Time-scale modification (TSM) is the process of changing the tempo of a music recording, ideally without affecting other musical aspects like pitch or sound quality.\n\n"
            "We would like you to evaluate pairs of TSM algorithms by playing with the speed slider, and then select which one sounds better.\n\n"
            "Note that the selection of algorithms is different on each page.\n\n"
            "Use the play buttons to listen and the pick buttons to make your choice.\n\n"
            "Once you've selected which option you think is better, press Submit to continue."
        )
        self.instructions.insert(END, instruction_text)
        self.instructions.config(state=DISABLED)  # Make it read-only
        
        self.txt = Entry(self.root, width=15)
        self.txt.grid(column=1, row=0, padx=5)

        self.btn = Button(self.root, text="Enter", fg="black", command=self.login_clicked)
        self.btn.grid(column=2, row=0)

    def login_clicked(self):
        self.root.withdraw()  # Hide the main menu window
        # text = self.txt.get()
        # self.lbl.config(text=f"You wrote: {text}")
        self.open_music_control(self.txt.get())

    def open_music_control(self, user_id):
        if self.player_window is None or not self.player_window.window.winfo_exists():
            self.player_window = Player(self.root,  user_id = user_id,on_close=self.on_player_close)

    def on_player_close(self):
        self.root.deiconify()  # Show the main menu again when player is closed

