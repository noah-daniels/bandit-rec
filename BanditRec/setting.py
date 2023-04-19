from abc import ABC, abstractmethod
from dataclasses import dataclass
from matplotlib import pyplot as plt

import numpy as np

from BanditRec.utils import my_argmax, my_argmax_dict


@dataclass
class SettingConfig:
    episode_count: int
    item_count: int
    k: int = 1
    episode_length: int = 1


class Episode:
    def __init__(self, setting):
        self.t = 0
        self.available_items = set()
        self.new_items = set()
        self.expired_items = set()
        self.__setting = setting

    def recommend(self, item):
        ctr = self.__setting.get_ctr(item, self.t)
        reward = np.random.binomial(1, ctr)
        return reward, ctr


class Setting(ABC):
    def __init__(self, config: SettingConfig):
        self.episode_count = config.episode_count
        self.episode_length = config.episode_length
        self.item_count = config.item_count
        self.k = config.k

    def reseed(self, seed):
        pass

    def start(self):
        episode = Episode(self)
        for t in range(self.episode_count):
            episode.t = t
            episode.new_items = self.new_items(episode)
            episode.expired_items = self.expired_items(episode)
            episode.available_items = (
                episode.available_items - episode.expired_items
            ) | episode.new_items
            yield episode

    def optimal_items(self, t, k, available_items=None):
        if available_items is None:
            a = np.array([self.get_ctr(i, t) for i in range(self.item_count)])
            return my_argmax(a, k)
        else:
            a = {i: self.get_ctr(i, t) for i in available_items}
            return my_argmax_dict(a, k)

    def visualize(self, **kwargs):
        fig, ax = plt.subplots()
        self.plot(ax, **kwargs)

        if self.item_count <= 10:
            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

        fig.suptitle("Ground Truth CTR", fontsize=14, fontweight="bold")
        ax.set_title(self.label)
        ax.set_xlabel("Time")
        ax.set_ylabel("CTR")

        plt.show()

    def plot(self, ax, tmin=0, tmax=None):
        tmax = self.episode_count if tmax is None else tmax

        x = np.arange(self.episode_count)
        y = np.empty((self.episode_count, self.item_count))
        y[:] = np.nan

        for ep in self.start():
            for i in ep.new_items:
                y[ep.t - 1][i] = 0
            for i in ep.expired_items:
                y[ep.t][i] = 0

            for i in ep.available_items:
                y[ep.t][i] = self.get_ctr(i, ep.t)

        for i in range(self.item_count):
            ax.plot(x[tmin:tmax], y[tmin:tmax, i], label=f"arm {i}")

    @property
    def label(self):
        """
        Textual representation of the setting.
        """
        return f"T={self.episode_count}x{self.episode_length}, K={self.item_count}/{self.k}"

    @abstractmethod
    def get_ctr(self, item, t):
        """
        Gives the true ctr of a given item at a given timepoint (episode).
        Should always return the same results for the same arguments. (Except after reseeding)
        """
        return 0

    @abstractmethod
    def new_items(self, episode):
        """
        Gives the missing items that should be preset among the available items at time t.
        Should always return the same results for the same arguments. (Except after reseeding)
        """
        return set()

    @abstractmethod
    def expired_items(self, episode):
        """
        Gives the expired items that should be removed from the available items at time t.
        Should always return the same results for the same arguments. (Except after reseeding)
        """
        return set()
