import glob
import random
import datetime
import time
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# === MongoDB Setup ===
uri = "mongodb+srv://clarkipeng:s2eNNVECeTuRQN4L@cluster0.bkjusqg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB!")
    print("Collections:", client.stuff.list_collection_names())
except Exception as e:
    print(e)

users_collection = client.stuff.users

# === Generate All Possible Tasks (filename + engine pairs) ===

engine_pairs = [("Hybrid","PV"),
                ("Hybrid","OPT0.25"),
                ("Hybrid","OPT0.5"),
                ("Hybrid","OPT1.0"),
                ("Hybrid","OPT2.0")]
def get_all_tasks():
    files = sorted(glob.glob('../samples/genres_original/*/*.wav'))
    return [(f, sorted(pair)) for f in files for pair in engine_pairs]


def get_user_info(user_id):
    total_files = sorted(glob.glob('../samples/genres_original/*/*.wav'))
    user = UserInfo(user_id, total_files)
    user.get_seen()
    return user

class UserInfo:
    def __init__(self, user_id):
        self.id = user_id
        self.completed_tasks = {}  # task_id -> {chosen_engine, timestamp, duration}
        self._current_task_id = None
        self._current_task_start = None  # 🕒 per-task timer
        self._load_or_create_user()

    def _load_or_create_user(self):
        row = users_collection.find_one({"id": self.id})
        if row:
            self.completed_tasks = row.get("completed_tasks", {})
        else:
            users_collection.insert_one({
                "id": self.id,
                "completed_tasks": {}
            })
            print(f"🆕 Created new user '{self.id}'")

    def _make_task_id(self, filename, engines):
        return f"{filename}|{'|'.join(sorted(engines))}"

    def log(self, filename, engines, chosen_engine):
        task_id = self._make_task_id(filename, engines)
        duration = None

        if self._current_task_id == task_id and self._current_task_start:
            duration = round(time.time() - self._current_task_start, 2)  # ⏱ seconds, rounded

        self.completed_tasks[task_id] = {
            "chosen_engine": chosen_engine,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "duration_sec": duration  # 💾 Save time spent
        }

        users_collection.update_one(
            {"id": self.id},
            {"$set": {"completed_tasks": self.completed_tasks}}
        )
        print(f"📥 Logged: {task_id} → {chosen_engine} (⏱ {duration}s)")


    def get_next_task(self):
        all_tasks = get_all_tasks()
        random.shuffle(all_tasks)
        for filename, engines in all_tasks:
            task_id = self._make_task_id(filename, engines)
            if task_id not in self.completed_tasks:
                self._current_task_id = task_id
                self._current_task_start = time.time()  # ⏱ Start timer
                print(filename,engines)
                return filename, engines
        return None

    def get_seen(self):
        return list(self.completed_tasks.keys())
    
def get_user_info(user_id):
    user = UserInfo(user_id)
    user.get_seen()
    return user