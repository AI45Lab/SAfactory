from .environment import TradingGym
import threading
import os
import pandas as pd

class TradingWrapper:
    def __init__(self, data_dir="/mnt/shared-storage-user/chenxinquan/ai_sandbox/env/data/trading"):
        self._max_id = 0
        self.env = {}
        self.info = {}
        self.ls = []
        self._lock = threading.Lock()
        self.price_list = ['AMZN.csv']
        self.tweet_list = ['amzn_stockmo.csv']
        self.data_dir = data_dir
        self.price_df_list, self.tweet_df_list = self.init_dataset()

    def init_dataset(self):
        price_df_list = []
        tweet_df_list = []
        for price, tweet in zip(self.price_list, self.tweet_list):
            price_path = os.path.join(self.data_dir, price)
            tweet_path = os.path.join(self.data_dir, tweet)
            price_df = pd.read_csv(price_path)
            tweet_df = pd.read_csv(tweet_path)
            price_df_list.append(price_df)
            tweet_df_list.append(tweet_df)
        return price_df_list, tweet_df_list
    
    def create(self):
        try:
            with self._lock:
                id = self._max_id
                self._max_id += 1
            new_env = TradingGym(
                self.price_df_list[0], window_size=7, senti_df=self.tweet_df_list[0]
            )
            obs, info = new_env.reset()
            print(f"-------Env {id} created--------")
            self.ls.append(id)
            payload = {"id": id, "observation": obs, "done": False, "reward": 0}
            self.env[id] = new_env
            self.info[id] = {
                "observation": obs,
                "done": False,
                "reward": 0,
                "deleted": False,
            }
        
        except Exception as e:
            payload = {"error": f"{e}"}

        return payload
    
    def step(self, id, action):
        try:
            self._check_id(id)
            (ob, reward, done, _, _) = self.env[id].step(action)
            payload = {"observation": ob, "reward": reward, "done": done}
            self.info[id].update(payload)
        except Exception as e:
            payload = {"error": f"{e}"}
        return payload
    
    def reset(self, id):
        try:
            self._check_id(id)
            ob, _ = self.env[id].reset()
            payload = {"id": id, "observation": ob, "done": False, "reward": 0}
            self.info[id].update(
                {"observation": ob, "done": False, "reward": 0, "deleted": False}
            )
        except Exception as e:
            payload = {"error": str(e)}
        return payload
    
    def get_observation(self, id: int):
        try:
            self._check_id(id)
            return self.info[id]["observation"]
        except Exception as e:
            return {"error": str(e)}
        
    def get_detailed_info(self, id: int):
        try:
            self._check_id(id)
            return self.info[id]
        except Exception as e:
            return {"error": str(e)}
            
    def _check_id(self, id: int):
        if id not in self.info:
            raise NameError(f"The id {id} is not valid.")
        if self.info[id]["deleted"]:
            raise NameError(f"The task with environment {id} has been deleted.")
        
    def close(self, id):
        try:
            self.ls.remove(id)
            self.env[id].close() 
            del self.info[id] 
            del self.env[id] 
            print(f"-------Env {id} closed--------")
            return True
        except KeyError:
            print(f"--------Env {id} not exist--------")
            return False
        except Exception as e:
            print(f"Error while closing Env {id}: {e}")
            return False
    
    
    def __del__(self):
        for idx in self.ls:
            self.env[idx].close()
            print(f"-------Env {idx} closed--------")

server = TradingWrapper()