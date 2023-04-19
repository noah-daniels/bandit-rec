import numpy as np
import scipy as sp
import math

from BanditRec.setting import Setting
from BanditRec.utils import my_argmax


class SimpleSetting(Setting):
    def __init__(self, config, evolution_speed=1e-3):
        super().__init__(config)

        self.evolution_speed = evolution_speed

    @property
    def label(self):
        return f"{super().label}(es={self.evolution_speed})"

    def get_ctr(self, item, t):
        K = self.item_count
        w = 1 + (K - 1) * (1 + math.sin(t * self.evolution_speed)) / 2
        return (K - 1) / K - abs(w - item - 1) / K

    def new_items(self, episode):
        if len(episode.available_items) == 0:
            return set(range(self.item_count))
        else:
            return set()

    def expired_items(self, episode):
        return set()


class PublicationSetting(Setting):
    def __init__(self, config, lifetime=None):
        super().__init__(config)

        self.lifetime = self.episode_count if lifetime is None else lifetime

    def reseed(self, seed):
        np.random.seed(seed)

        T = self.episode_count
        K = self.item_count
        self.publication_dates = (
            np.sort(
                np.random.choice(np.arange(0, K, 0.1), K),
            )
            * T
            // K
        )

        np.random.seed()

    def new_items(self, episode):
        result = set()
        if not hasattr(episode, "item_count"):
            episode.item_count = 0

        while (
            episode.item_count < self.item_count
            and episode.t >= self.publication_dates[episode.item_count]
        ):
            result.add(episode.item_count)
            episode.item_count += 1
        return result

    def expired_items(self, episode):
        result = set()
        for item in episode.available_items:
            if episode.t - self.publication_dates[item] >= self.lifetime:
                result.add(item)

        return result


class NewsSimulationSetting(PublicationSetting):
    def __init__(
        self,
        config,
        lifetime=None,
        seed=0,
        falloff_speed=0.2,
        ctr_base=0.005,
        ctr_fuzz=0.003,
        variance=0.7,
    ):
        super().__init__(config, lifetime)

        self.falloff_speed = falloff_speed

        self.ctr_base = ctr_base
        self.ctr_fuzz = ctr_fuzz

        self.variance = variance

        self.reseed(seed)

    def reseed(self, seed):
        super().reseed(seed)
        # predetermine item trajectories
        np.random.seed(seed)

        self.base_ctrs = self.ctr_base + self.ctr_fuzz * np.random.uniform(
            -1, 1, size=self.item_count
        )

        self.falloffs = (
            self.episode_count
            * self.falloff_speed
            * (
                1
                + np.random.uniform(-self.variance, self.variance, size=self.item_count)
            )
        )
        self.smoothness = 0.5 * (
            1 + np.random.uniform(-self.variance, self.variance, size=self.item_count)
        )
        self.peaks = 2 * (
            1 + np.random.uniform(-self.variance, self.variance, size=self.item_count)
        )
        np.random.seed()

    @property
    def label(self):
        return f"NS({super().label}, fs={self.falloff_speed}, ctr={self.ctr_base}/{self.ctr_fuzz}, v={self.variance})"

    def get_ctr(self, item, t):
        start_time = self.publication_dates[item]
        t = t - start_time

        if t < 0:
            return 0

        falloff = self.falloffs[item]
        base = self.base_ctrs[item]
        smoothness = self.smoothness[item]
        peaks = self.peaks[item]

        tt = t / falloff
        f = math.exp(-(tt**2) - math.log(1 / base))
        return f + smoothness * f * math.sin(math.pi * peaks * tt) * (1 - base)


class NewsSimulationSetting2(PublicationSetting):
    def __init__(self, config, lifetime=None, seed=0, **kwargs):
        super().__init__(config, lifetime)

        self.ctr_base = kwargs.get("ctr_base", 0.7)
        self.ctr_fuzz = kwargs.get("ctr_fuzz", 0.1)
        self.falloff_rate = kwargs.get("falloff_rate", 0)
        self.falloff_popularity_factor = kwargs.get("falloff_popularity_factor", 0.5)
        self.rank_factor = kwargs.get("rank_factor", 0.5)
        self.smoothing = kwargs.get("smoothing", 10)

        self.reseed(seed)

    def reseed(self, seed):
        super().reseed(seed)

        np.random.seed(seed)

        T = self.episode_count
        N = self.item_count

        base_ctrs = self.ctr_base + self.ctr_fuzz * np.random.uniform(-1, 1, size=N)
        popularity = (base_ctrs - self.ctr_base + self.ctr_fuzz) / (2 * self.ctr_fuzz)
        falloffs = (
            T
            * self.falloff_rate
            * (
                self.falloff_popularity_factor * (1 - popularity)
                + (1 - self.falloff_popularity_factor) * np.random.uniform(0, 1, size=N)
            )
        )

        # calculate item relevances
        t = np.linspace([0] * N, [T] * N, num=T, endpoint=False, dtype=int)
        tt = (t - self.publication_dates) / falloffs
        # R = np.clip(self.base_ctrs - tt, 0, 1)
        R = base_ctrs * np.exp(-np.power(tt, 2))

        # calculate items ranks
        RR = R.copy()
        RR[tt < 0] = np.nan
        R_ranks = np.argsort(np.argsort(-RR))

        # calculate item CTR
        CTR = R / np.log(self.rank_factor * R_ranks + math.e)

        # smooth item CTR curves
        kernel_size = self.episode_count // self.smoothing
        kernel = sp.stats.norm.pdf(np.linspace(-3, 3, num=kernel_size))
        kernel = kernel / np.sum(kernel)
        padded = np.pad(
            CTR, (((kernel_size - 1) // 2, kernel_size // 2), (0, 0)), mode="edge"
        )
        self.CTR = np.apply_along_axis(
            lambda r: np.convolve(r, kernel, mode="valid"), axis=0, arr=padded
        )

        np.random.seed()

    @property
    def label(self):
        return f"NS2({super().label}, ctr={self.ctr_base}/{self.ctr_fuzz}, fr={self.falloff_rate}, s={self.smoothing})"

    def get_ctr(self, item, t):
        return self.CTR[t][item]
