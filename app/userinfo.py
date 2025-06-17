import numpy as np
import glob

class UserInfo:
    def __init__(self, id, total_assigned):
        self.id = id
        self.todo = np.arange(len(total_assigned)).tolist()
        self.assigned=total_assigned
    def login(self, id):
        self.todo.remove(id)
    def log(self, id):
        self.todo.remove(id)
    def get_seen(self):
        #query database to remove todo
        pass
    
def get_user_info(id):
    return UserInfo(id, glob.glob('../samples/*.wav'))