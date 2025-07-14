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
            "Thank you for participating in our study! \n"
            "Please put your first and last name as your User ID, e.g. \"michaeljackson\" then press \"Enter\" to start.\n\n"
            "Your task is to compare two audio algorithms that change music speed.\n\n"
            "1. Use the 'Play' buttons and 'Adjust Speed' sliders to listen to each option.\n"
            "2. Click 'Pick' for the one that sounds better to you.\n"
            "3. If they sound the same, use the 'Sound the Same' button.\n"
            "4. Click 'Submit' to save your choice and move to the next pair."
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

