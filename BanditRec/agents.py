import numpy as np

from BanditRec.utils import my_argmax_dict
from BanditRec.agent import Agent


class OracleAgent(Agent):
    def __init__(self, setting):
        super().__init__()
        self.setting = setting

    @property
    def label(self):
        return f"Oracle"

    def estimate_ctr(self, item):
        return self.setting.get_ctr(item, self.t)

    def choose_items(self, k):
        return self.setting.optimal_items(self.t, k, self.available_items)

    def learn(self, item, reward):
        pass


class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    @property
    def label(self):
        return f"Random"

    def estimate_ctr(self, item):
        return 0

    def choose_items(self, k):
        return np.random.choice(list(self.available_items), k)

    def learn(self, item, reward):
        pass


class EpsGreedyAgent(Agent):
    def __init__(self, eps, lr, q0=None, ws=None):
        super().__init__()

        self.eps = eps
        self.lr = lr
        self.q0 = q0
        self.ws = ws

        if self.q0 is not None and self.lr is None:
            raise ValueError("Cannot use q0 with sample average updates.")

    @property
    def label(self):
        prefix = "" if self.ws is None else "SW-"
        learning_rate = "" if self.lr is None else f", lr={self.lr}, q0={self.q0}"
        window_size = "" if self.ws is None else f", ws={self.ws}"
        return f"{prefix}EG(eps={self.eps}{learning_rate}{window_size})"

    def reset_hook(self, item_count):
        self.averages = dict()
        self.impressions = dict()
        if self.ws is not None:
            self.update_queue = dict()

    def start_episode_hook(self, new_items, expired_items):
        for item in new_items:
            self.impressions[item] = 0
            if self.q0 is None:
                self.averages[item] = 0
            else:
                self.averages[item] = self.q0

        for item in expired_items:
            del self.averages[item]
            del self.impressions[item]

    def choose_items(self, k):
        s = np.random.uniform(0, 1)
        act_randomly = s < self.eps

        if act_randomly:
            items = np.random.choice(list(self.available_items), k)
        else:
            items = my_argmax_dict(self.averages, k)

        return items

    def learn(self, item, reward):
        if item is None:
            return

        self.impressions[item] += 1
        if self.lr is None:
            self.averages[item] += (reward - self.averages[item]) / self.impressions[
                item
            ]
        else:
            self.averages[item] += (reward - self.averages[item]) * self.lr

        # update window
        if self.ws is not None:
            self.update_queue.append((item, reward))
            if len(self.update_queue) > self.ws:
                i, r = self.update_queue.pop(0)

                if self.lr is None:
                    self.averages[i] -= r / self.impressions[i]
                    if self.impressions[i] > 1:
                        self.averages[i] *= self.impressions[i] / (
                            self.impressions[i] - 1
                        )
                    else:
                        self.averages[i] = 0
                    self.impressions[i] -= 1
                else:
                    self.impressions[i] -= 1
                    self.averages[i] -= (
                        (1 - self.lr) ** self.impressions[i] * self.lr * r
                    )

    def estimate_ctr(self, item):
        return self.averages[item]


class CTRBasedAgent(Agent):
    def __init__(self, ws=None):
        super().__init__()
        self.ws = ws

    def reset_hook(self, item_count):
        self.clicks = dict()
        self.impressions = dict()
        if self.ws is not None:
            self.update_queue = []

    def start_episode_hook(self, new_items, expired_items):
        for item in new_items:
            self.clicks[item] = 0
            self.impressions[item] = 0

        for item in expired_items:
            del self.clicks[item]
            del self.impressions[item]

    def estimate_ctr(self, item):
        return (
            0
            if self.impressions[item] < 1
            else self.clicks[item] / self.impressions[item]
        )

    def choose_items(self, k):
        ctrs = {item: self.estimate_ctr(item) for item in self.available_items}
        return my_argmax_dict(ctrs, k)

    def learn(self, item, reward):
        self.impressions[item] += 1
        self.clicks[item] += reward

        if self.ws is not None:
            self.update_queue.append((item, reward))
            if len(self.update_queue) > self.ws:
                i, r = self.update_queue.pop(0)
                if i in self.available_items:
                    self.impressions[i] -= 1
                    self.clicks[i] -= r


class ThompsonAgent(CTRBasedAgent):
    def __init__(self, ws=None, prior=None):
        super().__init__(ws)

        if prior is None:
            self.prior = (1, 1)
        else:
            self.prior = prior

    @property
    def label(self):
        args = []
        prefix = ""
        if self.prior != (1, 1):
            args.append(f"prior={self.prior}")
        if self.ws is not None:
            args.append(f"ws={self.ws}")
            prefix = "SW-"

        return f"{prefix}TS({', '.join(args)})"

    def choose_items(self, k):
        ctrs = {
            item: np.random.beta(
                self.prior[0] + self.clicks[item],
                self.prior[1] + self.impressions[item] - self.clicks[item],
            )
            for item in self.available_items
        }
        return my_argmax_dict(ctrs, k)


class FroomleAgent(CTRBasedAgent):
    def __init__(self, ws=None, boost_denominator=1, boost_rank=1):
        super().__init__(ws)

        self.boost_denominator = boost_denominator
        self.boost_rank = boost_rank

    @property
    def label(self):
        args = []
        args.append(f"d={self.boost_denominator}")
        args.append(f"r={self.boost_rank}")

        if self.ws is None:
            prefix = ""
        else:
            prefix = "SW-"
            args.append(f"ws={self.ws}")

        return f"{prefix}FR({', '.join(args)})"

    def start_episode_hook(self, new_items, expired_items):
        if len(self.available_items) > 0:
            rank = min(self.boost_rank, len(self.available_items))
            boost_ctr = np.partition(
                [
                    0
                    if self.impressions[i] == 0
                    else self.clicks[i] / self.impressions[i]
                    for i in self.available_items
                ],
                -rank,
            )[-rank]
        else:
            boost_ctr = 0

        for item in new_items:
            self.impressions[item] = self.boost_denominator
            self.clicks[item] = boost_ctr * self.boost_denominator

        for item in expired_items:
            del self.impressions[item]
            del self.clicks[item]
