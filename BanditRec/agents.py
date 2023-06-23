from abc import abstractmethod
import math
import random
import numpy as np

from BanditRec.utils import my_argmax_dict
from BanditRec.agent import Agent


class OracleAgent(Agent):
    def __init__(self, setting, **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def label(self):
        return f"Random"

    def estimate_ctr(self, item):
        return 0

    def choose_items(self, k):
        return np.random.choice(list(self.available_items), k, replace=False)

    def learn(self, item, reward):
        pass


class CTREstimator:
    def __init__(self, numpy=False, **kwargs):
        self.np = numpy

    def reset(self, setting=None):
        if self.np and setting is not None:
            self.ctrs = np.zeros(setting.item_count) + np.nan
            self.clicks = np.zeros(setting.item_count) + np.nan
            self.impressions = np.zeros(setting.item_count) + np.nan
        else:
            self.ctrs = dict()
            self.clicks = dict()
            self.impressions = dict()

    def update(self, item, reward):
        self.impressions[item] += 1
        self.clicks[item] += reward
        self.ctrs[item] = self.clicks[item] / self.impressions[item]

    def register_item(self, item):
        self.ctrs[item] = 0
        self.clicks[item] = 0
        self.impressions[item] = 0

    def unregister_item(self, item):
        if self.np:
            self.ctrs[item] = np.nan
            self.clicks[item] = np.nan
            self.impressions[item] = np.nan
        else:
            del self.ctrs[item]
            del self.clicks[item]
            del self.impressions[item]

    def is_valid_item(self, item):
        if self.np:
            return np.isnan(self.ctrs[item])
        else:
            return item in self.ctrs.keys()

    @property
    def params(self):
        return []


class WindowedCTREstimator(CTREstimator):
    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def reset(self):
        super().reset()
        self.window_queue = []

    def update(self, item, reward):
        super().update(item, reward)

        self.window_queue.append((item, reward))
        while len(self.window_queue) > self.window_size:
            i, r = self.window_queue.pop(0)
            if self.is_valid_item(i):
                self.impressions[i] -= 1
                self.clicks[i] -= r
                self.ctrs[i] = (
                    0
                    if self.impressions[i] < 1
                    else self.clicks[i] / self.impressions[i]
                )

    @property
    def params(self):
        return [f"ws={self.window_size}"]


class ImpressionWindowedCTREstimator(CTREstimator):
    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def reset(self):
        super().reset()
        self.windows = dict()

    def register_item(self, item):
        super().register_item(item)
        self.windows[item] = []

    def unregister_item(self, item):
        super().unregister_item(item)
        del self.windows[item]

    def update(self, item, reward):
        self.impressions[item] += 1
        self.clicks[item] += reward
        self.windows[item].append(reward)
        if len(self.windows[item]) < self.window_size:
            self.impressions[item] -= 1
            self.clicks[item] -= self.windows[item].pop(0)

        self.ctrs[item] = self.clicks[item] / self.impressions[item]

    @property
    def params(self):
        return [f"iws={self.window_size}"]


class WeightedCTREstimator(CTREstimator):
    def __init__(self, learning_rate, starting_value=None, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.staring_value = starting_value

    def register_item(self, item):
        super().register_item(item)
        if self.staring_value is not None:
            self.ctrs[item] = self.staring_value

    def update(self, item, reward):
        self.impressions[item] += 1
        self.clicks[item] += reward
        self.ctrs[item] += self.learning_rate * (reward - self.ctrs[item])

    @property
    def params(self):
        params = []
        params.append(f"lr={self.learning_rate}")
        if self.staring_value is not None:
            params.append(f"q0={self.staring_value}")
        return params


class DelayedWeightedCTREstimator(CTREstimator):
    def __init__(self, learning_rate, update_interval, starting_value=None, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.update_interval = update_interval
        self.staring_value = starting_value

    def reset(self, setting=None):
        super().reset(setting)
        self.updates_since_last_update = 0
        if self.np:
            self.temp_clicks = np.zeros(setting.item_count) + np.nan
            self.temp_impressions = np.zeros(setting.item_count) + np.nan
        else:
            self.temp_clicks = dict()
            self.temp_impressions = dict()

    def register_item(self, item):
        super().register_item(item)
        if self.staring_value is not None:
            self.ctrs[item] = self.staring_value

        self.temp_clicks[item] = 0
        self.temp_impressions[item] = 0

    def unregister_item(self, item):
        super().unregister_item(item)
        if self.np:
            self.temp_clicks[item] = np.nan
            self.temp_impressions[item] = np.nan
        else:
            del self.temp_clicks[item]
            del self.temp_impressions[item]

    def update(self, item, reward):
        self.temp_impressions[item] += 1
        self.temp_clicks[item] += reward
        self.updates_since_last_update += 1

        if self.updates_since_last_update >= self.update_interval:
            self.updates_since_last_update = 0
            if self.np:
                self.impressions += self.temp_impressions
                self.clicks += self.temp_clicks
                lrs = 1 - np.power(1 - self.learning_rate, self.temp_impressions)
                targets = np.divide(
                    self.temp_clicks,
                    self.temp_impressions,
                    out=np.zeros_like(self.temp_clicks),
                    where=self.temp_impressions != 0,
                )
                self.ctrs += lrs * (targets - self.ctrs)
                self.temp_clicks = np.zeros_like(self.temp_clicks)
                self.temp_impressions = np.zeros_like(self.temp_impressions)
            else:
                for item in self.ctrs.keys():
                    self.impressions[item] += self.temp_impressions[item]
                    self.clicks[item] += self.temp_clicks[item]

                    lr = 1 - (1 - self.learning_rate) ** self.temp_impressions[item]
                    target = (
                        self.temp_clicks[item] / self.temp_impressions[item]
                        if self.temp_impressions[item] > 0
                        else 0
                    )

                    self.ctrs[item] += lr * (target - self.ctrs[item])

                    self.temp_clicks[item] = 0
                    self.temp_impressions[item] = 0

    @property
    def params(self):
        params = []
        params.append(f"lr={self.learning_rate}, dt={self.update_interval}")
        if self.staring_value is not None:
            params.append(f"q0={self.staring_value}")
        return params


class EstimatorAgent(Agent):
    def __init__(self, estimator=None, **kwargs):
        super().__init__(**kwargs)

        if estimator is None:
            self.estimator = CTREstimator()
        else:
            self.estimator = estimator

    def reset_hook(self, setting):
        self.estimator.reset(setting)

    def start_episode_hook(self, new_items, expired_items):
        for item in new_items:
            self.estimator.register_item(item)

        for item in expired_items:
            self.estimator.unregister_item(item)

    def learn(self, item, reward):
        self.estimator.update(item, reward)

    def estimate_ctr(self, item):
        return self.estimator.ctrs[item]

    def choose_items(self, k):
        ctrs = {item: self.estimate_ctr(item) for item in self.available_items}
        return my_argmax_dict(ctrs, k)

    @property
    def params(self):
        return self.estimator.params

    @property
    def ctrs(self):
        return self.estimator.ctrs

    @property
    def clicks(self):
        return self.estimator.clicks

    @property
    def impressions(self):
        return self.estimator.impressions


class EpsGreedyAgent(EstimatorAgent):
    def __init__(self, eps, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    @property
    def label(self):
        params = []
        params.append(f"eps={self.eps}")
        params.extend(self.params)
        return f"EG({', '.join(params)})"

    def choose_items(self, k):
        s = np.random.uniform(0, 1)
        act_randomly = s < self.eps

        if act_randomly:
            items = np.random.choice(list(self.available_items), k)
        else:
            items = my_argmax_dict(self.ctrs, k)

        return items


class UCBAgent(EstimatorAgent):
    def __init__(self, c=1, **kwargs):
        super().__init__(**kwargs)
        self.c = c

    @property
    def label(self):
        params = []
        params.append(f"c={self.c}")
        params.extend(self.params)

        return f"{super().label}UCB({', '.join(params)})"

    def choose_items(self, k):
        scores = {
            item: self.ctrs[item]
            + self.c * math.sqrt(math.log(self.t) / (self.impressions[item] + 0.000001))
            for item in self.available_items
        }
        return my_argmax_dict(scores, k)


class ThompsonAgent(EstimatorAgent):
    def __init__(self, prior=None, **kwargs):
        super().__init__(**kwargs)

        if prior is None:
            self.prior = (1, 1)
        else:
            self.prior = prior

    @property
    def label(self):
        params = []
        if self.prior != (1, 1):
            params.append(f"prior={self.prior}")
        params.extend(self.estimator.params)

        return f"TS({', '.join(params)})"

    def choose_items(self, k):
        ctrs = {
            item: np.random.beta(
                self.prior[0] + self.impressions[item] * self.ctrs[item],
                self.prior[1] + self.impressions[item] * (1 - self.ctrs[item]),
            )
            for item in self.available_items
        }
        return my_argmax_dict(ctrs, k)


class FroomleAgent(EstimatorAgent):
    def __init__(
        self,
        boost_n=None,
        boost_ctrx=1,
        boost_nx=1,
        boost_ctr=None,
        boost_rank=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.boost_n = boost_n
        self.boost_nx = boost_nx
        self.boost_ctr = boost_ctr
        self.boost_ctrx = boost_ctrx
        self.boost_rank = boost_rank

    @property
    def label(self):
        params = []

        if self.boost_n is None or self.boost_ctr is None:
            params.append(f"r={self.boost_rank}")

        if self.boost_n is not None:
            params.append(f"n={self.boost_n}")
        else:
            params.append(f"n=x{self.boost_nx}")

        if self.boost_ctr is not None:
            params.append(f"ctr={self.boost_ctr}")
        elif self.boost_ctrx != 1:
            params.append(f"ctr=x{self.boost_ctrx}")

        params.extend(super().params)

        return f"FR({', '.join(params)})"

    def start_episode_hook(self, new_items, expired_items):
        super().start_episode_hook(new_items, expired_items)

        rank = min(self.boost_rank, len(self.available_items))
        if rank > 0:
            top_ctr = np.partition(
                list(self.ctrs.values()),
                -rank,
            )[-rank]
        else:
            top_ctr = 0.5

        if self.boost_ctr is None:
            boost_ctr = self.boost_ctrx * top_ctr
        else:
            boost_ctr = self.boost_ctr

        if self.boost_n is None:
            boost_n = self.boost_nx / max(top_ctr, 0.0001)
        else:
            boost_n = self.boost_n

        for item in new_items:
            self.impressions[item] = boost_n
            self.clicks[item] = boost_ctr * boost_n
            self.ctrs[item] = boost_ctr


class DynamicFroomleAgent(EstimatorAgent):
    def __init__(
        self,
        boost_n=None,
        boost_nx=1,
        boost_ctr=None,
        boost_ctrx=1,
        boost_rank=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.boost_n = boost_n
        self.boost_nx = boost_nx
        self.boost_ctr = boost_ctr
        self.boost_ctrx = boost_ctrx
        self.boost_rank = boost_rank

    @property
    def label(self):
        params = []

        if self.boost_n is None or self.boost_ctr is None:
            params.append(f"r={self.boost_rank}")

        if self.boost_n is not None:
            params.append(f"n={self.boost_n}")
        else:
            params.append(f"n=x{self.boost_nx}")

        if self.boost_ctr is not None:
            params.append(f"ctr={self.boost_ctr}")
        elif self.boost_ctrx != 1:
            params.append(f"ctr=x{self.boost_ctrx}")

        params.extend(super().params)

        return f"DYN-FR({', '.join(params)})"

    def reset_hook(self, setting):
        super().reset_hook(setting)
        self.top_ctr = 0.5

    def learn(self, item, reward):
        super().learn(item, reward)
        rank = min(self.boost_rank, len(self.available_items))
        if rank > 0:
            self.top_ctr = np.partition(
                [self.ctrs[i] for i in self.available_items],
                -rank,
            )[-rank]
        else:
            self.top_ctr = 0.5

    def estimate_ctr(self, item):
        if self.boost_ctr is None:
            boost_ctr = self.boost_ctrx * self.top_ctr
        else:
            boost_ctr = self.boost_ctr

        if self.boost_n is None:
            boost_n = self.boost_nx / max(self.top_ctr, 0.0001)
        else:
            boost_n = self.boost_n

        i = self.impressions[item]
        f = max(boost_n - i, 0) / boost_n

        return (1 - f) * self.ctrs[item] + f * boost_ctr


class CheatingFroomleAgent(Agent):
    def __init__(self, b0, b1, t, ws=None, **kwargs):
        super().__init__(**kwargs)

        self.b0 = b0
        self.b1 = b1
        self.t = t
        self.ws = ws

    @property
    def label(self):
        params = []
        params.append(f"b0={self.b0}")
        params.append(f"b1={self.b1}")
        params.append(f"t={self.t}")
        params.append(f"ws={self.ws}")
        # params.append(f"q={self.q0}/{self.q1}")

        return f"CF({', '.join(params)})"

    def reset_hook(self, setting):
        self.ctrs = dict()
        self.clicks = dict()
        self.impressions = dict()
        self.max_ctr = 0
        self.windows = dict()

        self.aux_info = setting.get_aux_info()

    def start_episode_hook(self, new_items, expired_items):
        for item in new_items:
            self.ctrs[item] = 0
            self.clicks[item] = 0
            self.impressions[item] = 0
            self.windows[item] = []

        for item in expired_items:
            del self.ctrs[item]
            del self.clicks[item]
            del self.impressions[item]
            del self.windows[item]

    def choose_items(self, k):
        return my_argmax_dict(
            {i: self.estimate_ctr(i) for i in self.available_items}, k
        )
        # return my_argmax_dict(
        #     {i: -self.aux_info[self.t][i] for i in self.available_items}, k
        # )

    def learn(self, item, reward):
        self.clicks[item] += reward
        self.impressions[item] += 1
        self.ctrs[item] = self.clicks[item] / self.impressions[item]

        if self.ws is not None:
            self.windows[item].append(reward)
            while len(self.windows[item]) > self.ws:
                self.impressions[item] -= 1
                self.clicks[item] -= self.windows[item].pop(0)

        self.max_ctr = max(self.ctrs.values())

    def estimate_ctr(self, item):
        rank = float(self.aux_info[self.t][item])
        boost_ctr_modifier = self.b0 * np.exp(-self.t * np.power(rank, 2))

        boost_ctr = boost_ctr_modifier * self.max_ctr
        if boost_ctr <= self.ctrs[item] or boost_ctr <= 0.0000001:
            return self.ctrs[item]

        boost_base = self.b1 / boost_ctr

        i = self.impressions[item]
        f = max(boost_base - i, 0) / boost_base

        return (1 - f) * self.ctrs[item] + f * boost_ctr


class PopAgent(Agent):
    def __init__(self, noise=3, **kwargs):
        super().__init__(**kwargs)
        self.noise = noise

    @property
    def label(self):
        # return f"POP(n={self.noise})"
        return "POP"

    def reset_hook(self, setting):
        self.extra_info = setting.get_extra_info(noise=self.noise)

    def choose_items(self, k):
        return my_argmax_dict(
            {i: -self.extra_info[self.t][i] for i in self.available_items}, k
        )

    def learn(self, item, reward):
        pass

    def estimate_ctr(self, item):
        return 0


class OPGAgent(EstimatorAgent):
    def __init__(self, lr, q0, q1, gamma, noise=3, update_interval=None, **kwargs):
        if update_interval is None:
            est = WeightedCTREstimator(learning_rate=lr, numpy=True)
        else:
            est = DelayedWeightedCTREstimator(
                learning_rate=lr, numpy=True, update_interval=update_interval
            )
        super().__init__(est, **kwargs)
        self.q0 = q0
        self.q1 = q1
        self.gamma = gamma
        self.noise = noise
        self.lr = lr

    @property
    def label(self):
        params = []
        params.extend(self.params)
        params.append(f"q={self.q0}/{self.q1}")
        params.append(f"γ={self.gamma}")
        # params.append(f"n={self.noise}")

        return f"OPG({', '.join(params)})"

    def reset_hook(self, setting):
        super().reset_hook(setting)
        self.extra_info = setting.get_extra_info(self.noise)

    def choose_items(self, k):
        ranks = self.extra_info[self.t]
        scores = self.q0 + (self.q1 - self.q0) * np.exp(-ranks / self.gamma)
        self.estimates = self.ctrs + scores * np.power(1 - self.lr, self.impressions)
        return my_argmax_dict({i: self.estimates[i] for i in self.available_items}, k)
