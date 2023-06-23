import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import math

from BanditRec.setting import Setting
from BanditRec.utils import my_argmax


class SimpleSetting(Setting):
    def __init__(self, config, evolution_speed=1e-3, seed=0):
        super().__init__(config)

        self.evolution_speed = evolution_speed
        self.reseed(seed)

    @property
    def label(self):
        return f"S1({super().label}, \u03C3={self.evolution_speed})"

    def get_ctr(self, item, t):
        K = self.item_count
        w = 1 + (K - 1) * (1 + math.sin((t + self.offset) * self.evolution_speed)) / 2
        return (K - 1) / K - abs(w - item - 1) / K

    def new_items(self, episode):
        if len(episode.available_items) == 0:
            return set(range(self.item_count))
        else:
            return set()

    def expired_items(self, episode):
        return set()

    def reseed(self, seed):
        np.random.seed(seed)
        self.offset = np.random.randint(0, self.episode_count)
        np.random.seed()


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
        ctr_base=0.3,
        ctr_fuzz=0.1,
        evolution_base=5e-4,
        evolution_fuzz=1e-4,
    ):
        super().__init__(config, lifetime)

        self.ctr_base = ctr_base
        self.ctr_fuzz = ctr_fuzz

        self.evolution_base = evolution_base
        self.evolution_fuzz = evolution_fuzz

        self.smoothness_base = 0.5
        self.smoothness_fuzz = 0.3

        self.peak_base = 2
        self.peak_fuzz = 1.5

        self.reseed(seed)

    def reseed(self, seed):
        super().reseed(seed)
        # predetermine item trajectories
        np.random.seed(seed)

        K = self.item_count

        ctr_fuzz = self.ctr_fuzz * np.random.uniform(-1, 1, size=K)
        self.base_ctrs = self.ctr_base + ctr_fuzz

        evolution_fuzz = self.evolution_fuzz * np.random.uniform(-1, 1, size=K)
        self.evolution_speeds = self.evolution_base + evolution_fuzz

        smoothness_fuzz = self.smoothness_fuzz * np.random.uniform(-1, 1, size=K)
        self.smoothness = self.smoothness_base + smoothness_fuzz

        peak_fuzz = self.peak_fuzz * np.random.uniform(-1, 1, size=K)
        self.peaks = self.peak_base + peak_fuzz

        np.random.seed()

    @property
    def label(self):
        return f"S2({super().label}, b={self.ctr_base}\u00B1{self.ctr_fuzz}, \u03C3={self.evolution_base}\u00B1{self.evolution_fuzz})"

    def get_ctr(self, item, t):
        start_time = self.publication_dates[item]
        t = t - start_time

        if t < 0:
            return 0

        evolution_speed = self.evolution_speeds[item]
        base = self.base_ctrs[item]
        smoothness = self.smoothness[item]
        peaks = self.peaks[item]

        f = base * math.exp(-((evolution_speed * t) ** 2))
        return f + smoothness * f * math.sin(math.pi * peaks * evolution_speed * t) * (
            1 - base
        )


class NewsSimulationSetting2(PublicationSetting):
    def __init__(
        self,
        config,
        lifetime=None,
        seed=0,
        ctr_base=0.5,
        ctr_fuzz=0.2,
        sigma_base=5e-4,
        sigma_fuzz=2e-4,
        smoothing=200,
        global_evolution=None,
    ):
        super().__init__(config, lifetime)

        self.ctr_base = ctr_base
        self.ctr_fuzz = ctr_fuzz

        self.sigma_base = sigma_base
        self.sigma_fuzz = sigma_fuzz

        self.smoothing = smoothing
        self.rank_factor = 0.5

        self.global_evolution = global_evolution

        self.reseed(seed)

    def reseed(self, seed):
        super().reseed(seed)

        np.random.seed(seed)

        T = self.episode_count
        N = self.item_count

        if self.global_evolution is None:
            ctr_fuzz = self.ctr_fuzz * np.random.uniform(-1, 1, size=N)
        else:
            scale = self.global_evolution
            influence = 0.5

            x = np.linspace(0, 1, N)
            offset = np.random.uniform(0, 2 * math.pi)
            global_noise = (
                np.sin(3 * scale * x - offset) + np.sin(7 * scale * x + offset)
            ) / 2
            local_noise = np.random.uniform(-1, 1, size=N)
            ctr_fuzz = self.ctr_fuzz * (
                influence * global_noise + (1 - influence) * local_noise
            )

        base_ctrs = self.ctr_base + ctr_fuzz

        sigma_fuzz = self.sigma_fuzz * np.random.uniform(-1, 1, size=N)
        evolution_speeds = self.sigma_base + sigma_fuzz

        # calculate item relevances
        t = np.linspace([0] * N, [T] * N, num=T, endpoint=False, dtype=int)
        TT = evolution_speeds * (t - self.publication_dates)
        F = base_ctrs * np.exp(-np.power(TT, 2))

        # calculate items ranks
        FF = F.copy()
        FF[TT < 0] = np.nan
        F_ranks = np.argsort(np.argsort(-FF))

        # calculate item CTR
        CTR = F / np.log(self.rank_factor * F_ranks + math.e)

        # smooth item CTR curves
        if self.smoothing is None:
            self.CTR = CTR
        else:
            kernel_size = self.smoothing
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
        params = []
        if self.lifetime is not None:
            params.append(f"L={self.lifetime}")
        params.append(f"b={self.ctr_base}\u00B1{self.ctr_fuzz}")
        params.append(f"\u03C3={self.sigma_base}\u00B1{self.sigma_fuzz}")
        if self.smoothing is not None:
            params.append(f"s={self.smoothing}")
        if self.global_evolution is not None:
            params.append(f"g={self.global_evolution}")

        return f"S2({super().label}, {', '.join(params)})"

    def get_ctr(self, item, t):
        return self.CTR[t][item]

    def get_extra_info(self, noise=3):
        ctrs = self.CTR
        ctr_fuzz = noise * self.ctr_fuzz

        noise = np.tile(
            ctr_fuzz * np.random.uniform(-1, 1, size=(1, ctrs.shape[1])),
            (ctrs.shape[0], 1),
        ) + 0.5 * ctr_fuzz * np.random.uniform(-1, 1, size=ctrs.shape)

        ctrs2 = ctrs + noise

        pd = self.publication_dates.astype(int)
        for i, column in enumerate(ctrs2.T):
            column[: pd[i]] = np.nan
            column[pd[i] + self.lifetime :] = np.nan

        # return ctrs2
        args = np.argsort(np.argsort(-ctrs2))

        for i, column in enumerate(args.T):
            column[: pd[i]] = -1
            column[pd[i] + self.lifetime :] = -1
        return args

    def plot_extra_info(self, noise=3, tmin=0, tmax=None):
        if tmax is None:
            tmax = self.episode_count

        ctrs = self.get_extra_info(noise)
        for i in range(ctrs.shape[1]):
            cm = plt.colormaps.get_cmap("tab20c")
            y = ctrs[:, i].astype(float)
            pd = int(self.publication_dates[i])
            y[:pd] = np.nan
            y[pd + self.lifetime :] = np.nan

            plt.plot(
                np.arange(tmin, tmax), y[tmin:tmax], color=cm.colors[i % len(cm.colors)]
            )
            plt.xlabel("Time")
            plt.ylabel("Rank")


class ConstantsSetting(Setting):
    def __init__(self, config, events):
        super().__init__(config)

        items = dict()
        for t in range(self.episode_count):
            processed_items = set()
            if t in events:
                for i, v in events[t].items():
                    if i not in items:
                        items[i] = [None, None, []]

                    if v is None:
                        items[i][1] = t
                        items[i][2].append(0)
                    else:
                        if items[i][0] is None:
                            items[i][0] = t
                        items[i][2].append(v)

                    processed_items.add(i)

            for i in items.keys():
                if i not in processed_items:
                    items[i][2].append(items[i][2][-1])

        ctrs = np.zeros((self.episode_count, self.item_count))
        for i, (start, end, values) in items.items():
            ctrs[start:, i] = values

        times = dict()
        for i, (start, end, values) in items.items():
            if start not in times:
                times[start] = [set(), set()]
            if end not in times:
                times[end] = [set(), set()]

            times[start][0].add(i)
            times[end][1].add(i)

        self.ctrs = ctrs
        self.times = times
        self.valid_ranges = [(start, end) for start, end, _ in items.values()]

    @property
    def label(self):
        return f"ES({super().label})"

    def get_ctr(self, item, t):
        return self.ctrs[t][item]

    def new_items(self, episode):
        return self.times[episode.t][0] if episode.t in self.times else set()

    def expired_items(self, episode):
        return self.times[episode.t][1] if episode.t in self.times else set()

    def get_aux_info(self, n0=0.2, n1=0.1):
        ctrs = self.ctrs

        noise = np.tile(
            n0 * np.random.uniform(-1, 1, size=(1, ctrs.shape[1])), (ctrs.shape[0], 1)
        ) + n1 * np.random.uniform(-1, 1, size=ctrs.shape)
        ctrs2 = ctrs + noise

        vr = self.valid_ranges
        for i, column in enumerate(ctrs2.T):
            s, e = vr[i]
            column[:s] = np.nan
            if e is not None:
                column[e:] = np.nan

        args = np.argsort(np.argsort(-ctrs2))

        return args
