from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self):
        pass

    def reset(self, item_count):
        self.available_items = set()
        self.t = 0
        self.reset_hook(item_count)

    def act(self, k):
        if len(self.available_items) < 1:
            res = []
        elif len(self.available_items) <= k:
            res = list(self.available_items)
        else:
            res = self.choose_items(k)

        return res

    def start_episode(self, new_items, expired_items, t):
        self.t = t
        self.start_episode_hook(new_items, expired_items)
        self.available_items = (self.available_items - expired_items) | new_items

    @property
    def label(self):
        return type(self).__name__

    def reset_hook(self, item_count):
        pass

    def start_episode_hook(self, new_items, expired_items):
        pass

    @abstractmethod
    def estimate_ctr(self, item):
        return 0

    @abstractmethod
    def choose_items(self, k):
        return []

    @abstractmethod
    def learn(self, item, reward):
        pass
