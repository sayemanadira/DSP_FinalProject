from tkinter import Tk, Label, Entry, Button, Menu
from player import Player
from userinfo import get_user_info

class MainMenu:
    def __init__(self, root):
        self.user_info = None
        
        self.root = root
        self.root.title("Main Menu")
        self.root.geometry('350x200')
        self.player_window = None  # Track player window

        # Menu bar
        self.menu = Menu(self.root)
        self.file_menu = Menu(self.menu, tearoff=0)
        # self.file_menu.add_command(label='Open Music Control', command=self.open_music_control)
        # self.file_menu.add_separator()
        # self.file_menu.add_command(label='Exit', command=self.root.quit)
        self.menu.add_cascade(label='File', menu=self.file_menu)
        self.root.config(menu=self.menu)

        # UI Elements
        self.lbl = Label(self.root, text="User ID:")
        self.lbl.grid(column=0, row=0, padx=10, pady=10)

        self.txt = Entry(self.root, width=15)
        self.txt.grid(column=1, row=0, padx=5)

        self.btn = Button(self.root, text="Enter", fg="black", command=self.login_clicked)
        self.btn.grid(column=2, row=0)

    def login_clicked(self):
        self.root.withdraw()  # Hide the main menu window
        # text = self.txt.get()
        # self.lbl.config(text=f"You wrote: {text}")
        self.user_info = get_user_info(self.txt.get())
        print(f"User ID: {self.user_info}")
        self.open_music_control()

    def open_music_control(self):
        if self.player_window is None or not self.player_window.window.winfo_exists():
            self.player_window = Player(self.root, on_close=self.on_player_close, userinfo = self.user_info)

    def on_player_close(self):
        self.root.deiconify()  # Show the main menu again when player is closed

