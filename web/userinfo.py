import numpy as np
import glob

from pymongo import MongoClient
from pymongo.server_api import ServerApi
uri = "mongodb+srv://clarkipeng:s2eNNVECeTuRQN4L@cluster0.bkjusqg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    print("Collections available:", client.stuff.list_collection_names())
except Exception as e:
    print(e)

users_collection = client.stuff.users

class UserInfo:
    def __init__(self, user_id, total_assigned):
        self.id = user_id
        self.assigned = total_assigned
        self.todo = list(range(len(total_assigned)))
        self.collected_data = {}
        self._load_or_create_user()

    def _load_or_create_user(self):
        if self.get_seen():
            pass
        else:
            users_collection.insert_one({
                "id": self.id,
                "assigned": self.assigned,
                "todo": self.todo,
                "collected_data": self.collected_data
            })
            print(f"🆕 New user '{self.id}' created.")

    def log(self, filename, result):
        if filename not in self.assigned:
            raise ValueError(f"'{filename}' is not in assigned list for user '{self.id}'")

        index = self.assigned.index(filename)
        if index in self.todo:
            self.todo.remove(index)
        else:
            print(f"⚠️ Warning: File '{filename}' was already logged for user '{self.id}'")

        self.collected_data[filename] = result

        users_collection.update_one(
            {"id": self.id},
            {"$set": {
                "assigned": self.assigned,
                "todo": self.todo,
                "collected_data": self.collected_data
            }},
            upsert=True
        )
        print(f"📥 Logged result for '{filename}' under user '{self.id}'.")

    def get_seen(self):
        row = users_collection.find_one({"id": self.id})
        if row:
            self.todo = row.get("todo", self.todo)
            self.assigned = row.get("assigned", self.assigned)
            self.collected_data = row.get("collected_data", self.collected_data)
        return not (row is None)

def get_user_info(user_id):
    total_files = sorted(glob.glob('../samples/*.wav'))
    user = UserInfo(user_id, total_files)
    user.get_seen()
    return user