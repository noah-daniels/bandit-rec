from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp

from BanditRec.utils import bin_linear, smooth_fast


def with_bounds(array, confidence=0.95):
    N = array.shape[0]
    mean = array.mean(axis=0)
    if confidence is None:
        iqr = sp.stats.iqr(array, axis=0)
        return mean, iqr
    else:
        sem = sp.stats.sem(array, axis=0)
        st = sp.stats.t.ppf((confidence + 1) / 2, N - 1)
        confidence_bound = sem * st
        return mean, confidence_bound


def calculate_regret(rewards, oracle_rewards, confidence=0.95, tmin=0, tmax=None):
    if tmax is None:
        tmax = rewards.shape[1]

    cum_regret = (oracle_rewards[:, tmin:tmax] - rewards[:, tmin:tmax]).cumsum(axis=1)
    return with_bounds(cum_regret, confidence)


def calcualte_ctr(rewards, confidence=0.95, tmin=0, tmax=None):
    if tmax is None:
        tmax = rewards.shape[1]

    return with_bounds(rewards[:, tmin:tmax].mean(axis=1), confidence)


class Result:
    def __init__(self, reward, estimate, impression, setting, label):
        self.reward_array = reward
        self.estimate_array = estimate
        self.impression_array = impression

        self.label = label
        self.setting = setting
        self.setting_label = setting.label


class Results:
    def __init__(self, simulator):
        self.results = defaultdict(lambda: dict())
        self.oracles = dict()

        self.simulator = simulator

    def setting_iter(self, labels=None, require_oracle=False, skip_oracle=False):
        settings = []
        if labels is None:
            settings = self.results.items()
        else:
            for l in labels:
                if l in self.results:
                    settings = [(l, self.results[l])]

        for setting_label, agent_results in settings:
            if require_oracle and not self.oracle_exists(setting_label):
                continue
            result_list = filter(
                lambda r: not skip_oracle or r.label != "Oracle", agent_results.values()
            )
            yield setting_label, result_list

    def agent_iter(self, labels=None, require_oracle=False, skip_oracle=False):
        results = defaultdict(lambda: list())
        for setting_label, agent_results in self.results.items():
            if require_oracle and not self.oracle_exists(setting_label):
                continue

            if labels is None:
                for agent_label, result in agent_results.items():
                    if skip_oracle and result.label == "Oracle":
                        continue
                    results[agent_label].append(result)
            else:
                for l in labels:
                    if l in agent_results:
                        results[l].append(agent_results[l])

        return results.items()

    def results_iter(self, setting=None, agent=None):
        if setting is None and agent is None:
            for _, agents in self.results.items():
                for _, result in agents.items():
                    yield result
        elif setting is None:
            for _, agents in self.results.items():
                if agent in agents:
                    yield agents[agent]
        elif agent is None:
            for _, result in self.results[setting].items():
                yield result
        else:
            yield self.results[setting][agent]

    def set_oracle(self, setting, bundle):
        if self.oracle_exists(setting.label):
            o = self.oracles[setting.label]
            o.reward_array = bundle[0]
            o.estimate_array = bundle[1]
            o.impression_array = bundle[2]
        else:
            r = Result(*bundle, setting, "Oracle")
            self.oracles[setting.label] = r
            self.results[setting.label]["Oracle"] = r

    def oracle_exists(self, setting_label):
        return setting_label in self.oracles

    def add(self, setting, label, bundle):
        self.results[setting.label][label] = Result(*bundle, setting, label)

    def delete(self, setting_label, agent_label=None):
        if agent_label is None:
            del self.results[setting_label]
            del self.oracles[setting_label]
        else:
            del self.results[setting_label][agent_label]

    def ctrs(
        self, setting_labels=None, relative=False, sort=True, latex=False, **kwargs
    ):
        for label, results in self.setting_iter(
            setting_labels, require_oracle=True, skip_oracle=True
        ):
            oracle = self.oracles[label]
            setting = oracle.setting

            if relative:
                ora = oracle.reward_array
                ora = np.maximum(ora, 0.0001)
                ctrs = {
                    r.label: calcualte_ctr(r.reward_array / ora, **kwargs)
                    for r in results
                }
            else:
                ctrs = {
                    r.label: calcualte_ctr(r.reward_array / setting.k, **kwargs)
                    for r in results
                }
            self.__print_ctr(ctrs, label, sort, latex)

    def ctrs_per_agent(self, agent_labels=None, relative=False, sort=True, **kwargs):
        for label, results in self.agent_iter(
            agent_labels, require_oracle=True, skip_oracle=True
        ):
            if relative:
                ctrs = {
                    r.setting_label: calcualte_ctr(
                        r.reward_array
                        / np.maximum(
                            self.oracles[r.setting_label].reward_array, 0.0001
                        ),
                        **kwargs,
                    )
                    for r in results
                }
            else:
                ctrs = {
                    r.setting_label: calcualte_ctr(
                        r.reward_array / r.setting.k, **kwargs
                    )
                    for r in results
                }
            self.__print_ctr(ctrs, label, sort)

    def __print_ctr(self, ctrs, title, sort, latex):
        sorted_ = (
            sorted(ctrs.items(), key=lambda item: -item[1][0]) if sort else ctrs.items()
        )
        print(f"=={'='*len(title)}\n {title} \n=={'='*len(title)}")
        for label, ctr in sorted_:
            if latex:
                print(f"${ctr[0]:.3f} \pm {ctr[1]:.3f}$ & {label} ")
            else:
                print(f"{ctr[0]:10.3f} ±{ctr[1]:<10.3f} - {label}")
        print("")

    ### RANKING ###
    def rank(self, setting_labels=None, sort=True, **kwargs):
        for label, results in self.setting_iter(
            setting_labels, require_oracle=True, skip_oracle=True
        ):
            o = self.oracles[label]
            regrets = {
                r.label: calculate_regret(r.reward_array, o.reward_array, **kwargs)
                for r in results
            }
            self.__print_ranking(regrets, label, sort)

    def rank_per_agent(self, agent_labels=None, sort=True, **kwargs):
        for label, results in self.agent_iter(
            agent_labels, require_oracle=True, skip_oracle=True
        ):
            regrets = {
                r.setting_label: calculate_regret(
                    r.reward_array, self.oracles[r.setting_label].reward_array, **kwargs
                )
                for r in results
            }
            self.__print_ranking(regrets, label, sort)

    def __print_ranking(self, regrets, title, sort):
        regrets = {l: (r[0][-1], r[1][-1]) for l, r in regrets.items()}
        sorted_ = (
            sorted(regrets.items(), key=lambda item: item[1][0])
            if sort
            else regrets.items()
        )
        print(f"=={'='*len(title)}\n {title} \n=={'='*len(title)}")
        for label, regret in sorted_:
            print(f"{regret[0]:10.1f} ±{regret[1]:<10.1f} - {label}")
        print("")

    ### REGRET ###
    def create_regret_plot(self, setting_labels=None, **kwargs):
        for label, results in self.setting_iter(
            setting_labels, require_oracle=True, skip_oracle=True
        ):
            o = self.oracles[label]
            regrets = {
                r.label: calculate_regret(r.reward_array, o.reward_array, **kwargs)
                for r in results
            }
            self.__create_regret_plot(
                regrets, f"{label}, {self.simulator.label}", kwargs.get("tmin", 0)
            )

    def create_regret_plot_per_agent(self, agent_labels=None, **kwargs):
        for label, results in self.agent_iter(
            agent_labels, require_oracle=True, skip_oracle=True
        ):
            regrets = {
                r.setting_label: calculate_regret(
                    r.reward_array, self.oracles[r.setting_label].reward_array, **kwargs
                )
                for r in results
            }
            self.__create_regret_plot(
                regrets, f"{label}, {self.simulator.label}", kwargs.get("tmin", 0)
            )

    def __create_regret_plot(self, regrets, subtitle, tmin):
        fig, ax = plt.subplots()

        for label, regret in regrets.items():
            r, s = regret
            x = np.arange(len(r)) + tmin
            ax.plot(x, r, label=label)
            ax.fill_between(x, r + s, r - s, alpha=0.2)

        fig.suptitle(f"Average Cumulative Regret", fontsize=14, fontweight="bold")
        ax.set_title(subtitle)
        ax.set_xlabel("Time")
        ax.set_ylabel("Regret")
        ax.legend(loc="upper left")

        plt.show()

    ### IMPRESSIONS ###
    def create_impressions_plot(
        self, agent=None, setting=None, show_setting=True, **kwargs
    ):
        if self.simulator.seed_change_interval > 0:
            raise ValueError("Cannot plot impressions with a changing seed.")

        for r in self.results_iter(setting, agent):
            if r.label == "Oracle":
                continue

            if show_setting:
                fig, axs = plt.subplots(2, 1, sharex=True, height_ratios=[1, 3])
                fig.subplots_adjust(hspace=0)
                ax0 = axs[0]
                ax1 = axs[1]
                r.setting.plot(ax0, kwargs.get("tmin", 0), kwargs.get("tmax", None))
                ax0.set_title(f"{r.setting_label}, {self.simulator.label}")
                ax0.set_ylabel("CTR")
                ax1.set_title("")
            else:
                fig, axs = plt.subplots(2, 1, sharex=True, height_ratios=[1, 3])
                fig.subplots_adjust(hspace=0)
                ax0 = axs[0]
                ax1 = axs[1]
                oracle = self.oracles[r.setting_label]
                self.__plot_impressions(ax0, oracle, **kwargs)
                ax0.set_yticks([])
                ax0.set_title(f"{r.setting_label}, {self.simulator.label}")
                ax1.set_title("")

            self.__plot_impressions(ax1, r, **kwargs)
            ax1.set_ylabel("Impression probability")
            ax1.set_xlabel("Time")

            fig.suptitle(
                f"Arm Distribution for {r.label}", fontsize=14, fontweight="bold"
            )

            plt.show()
            plt.close()

    def __plot_impressions(
        self, ax, result, resolution=100, sort_by_id=True, tmin=0, tmax=None
    ):
        tmax = result.impression_array.shape[1] if tmax is None else tmax

        cm = plt.cm.get_cmap("tab20c")

        ia = result.impression_array[:, tmin:tmax]
        episode_count = ia.shape[1]
        item_count = ia.shape[0]

        bin_size = episode_count / resolution
        x = tmin + np.arange(resolution) * bin_size

        aandeel = np.zeros((item_count, resolution))
        for i, d in enumerate(ia):
            dd = bin_linear(d, resolution)
            aandeel[i] = dd / bin_size

        if sort_by_id:
            for i in range(len(aandeel)):
                offset = aandeel[:i, :].sum(axis=0)
                ax.bar(
                    x,
                    aandeel[i],
                    width=bin_size,
                    label=i,
                    bottom=offset,
                    color=cm.colors[i % len(cm.colors)],
                )
        else:
            sorted_arg = np.argsort(
                np.flip(np.argsort(aandeel, axis=0), axis=0), axis=0
            )
            sorted_sums = np.flip(np.sort(aandeel, axis=0), axis=0).cumsum(axis=0)
            sorted_sums[-1] = np.zeros(resolution)
            sorted_sums = np.roll(sorted_sums, 1, axis=0)
            for i in range(len(aandeel)):
                offset = sorted_sums[sorted_arg[i], np.arange(resolution)]
                ax.bar(
                    x,
                    aandeel[i],
                    width=bin_size,
                    label=i,
                    bottom=offset,
                    color=cm.colors[i % len(cm.colors)],
                )

        if item_count <= 10:
            ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ### ESTIMATES ###
    def create_estimates_plot(self, agent=None, setting=None, **kwargs):
        if self.simulator.seed_change_interval > 0:
            raise ValueError("Cannot plot estimates with a changing seed.")

        for r in self.results_iter(setting, agent):
            fig, ax = plt.subplots()

            self.__plot_estimates(ax, r, **kwargs)

            fig.suptitle(f"Estimated CTRs by {r.label}", fontsize=14, fontweight="bold")
            ax.set_title(f"{r.setting_label}, {self.simulator.label}")
            ax.set_xlabel("Time")
            ax.set_ylabel("CTR")

            plt.show()
            plt.close()

    def __plot_estimates(self, ax, result, show_truth=True, tmin=0, tmax=None):
        tmax = result.estimate_array.shape[1] if tmax is None else tmax

        x = np.arange(tmin, tmax)

        cm = plt.cm.get_cmap("tab20c")

        max1 = 0
        for item, estimate in enumerate(result.estimate_array):
            y = estimate[tmin:tmax]
            ax.plot(
                x,
                y,
                label=f"item {item}",
                alpha=0.3,
                color=cm.colors[item % len(cm.colors)],
            )
            max1 = max(max1, 1.1 * y.max())

        if show_truth and self.oracle_exists(result.setting_label):
            max2 = 0
            for item, estimate in enumerate(
                self.oracles[result.setting_label].estimate_array
            ):
                y = estimate[tmin:tmax]
                ax.plot(
                    x,
                    y,
                    label=f"true {item}",
                    alpha=1,
                    color=cm.colors[item % len(cm.colors)],
                )
                max2 = max(max2, 1.1 * y.max())

            ax.set_ylim(0, min(max(max1, max2), 3 * max2))

        if result.estimate_array.shape[0] <= 10:
            ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    # def delete_regex(self, regex):
    #     for label in list(self.rewards.keys()):
    #         if re.search(regex, label) is not None:
    #             self.delete(label)

    # def save(self, filename):
    #     result = {
    #         **{f"r:{l}": r for l, r in self.rewards.items()},
    #         **{f"e:{l}": e for l, e in self.estimates.items()},
    #         **{f"i:{l}": i for l, i in self.impressions.items()},
    #         "oracle0": self.oracle[0],
    #         "oracle1": self.oracle[1],
    #         "oracle2": self.oracle[2],
    #     }
    #     np.savez(filename, **result)

    # def load(self, filename):
    #     data = np.load(filename)
    #     for key, value in data.items():
    #         if key.startswith("r:"):
    #             self.rewards[key[2:]] = value
    #         elif key.startswith("e:"):
    #             self.estimates[key[2:]] = value
    #         elif key.startswith("i:"):
    #             self.impressions[key[2:]] = value

    #     self.oracle = (data["oracle0"], data["oracle1"], data["oracle2"])

    # def save(self, base_path):
    #     # get paths and make sure directories exist
    #     base_path = Path(base_path)
    #     impressions = base_path / "impressions"
    #     estimates = base_path / "estimates"
    #     impressions.mkdir(parents=True, exist_ok=True)
    #     estimates.mkdir(parents=True, exist_ok=True)

    #     # regret plot
    #     fig = self.create_regret_plot(show=False)
    #     fig.savefig(base_path / "regret.png", bbox_inches="tight")

    #     # impression plots
    #     for a in self.results.impressions.keys():
    #         fig = self.create_impressions_plot(a, show=False)[0]
    #         fig.savefig(impressions / f"{a}.png", bbox_inches="tight")

    #     # estimate plots
    #     for a in self.results.estimates.keys():
    #         fig = self.create_estimates_plot(a, show=False)[0]
    #         fig.savefig(estimates / f"{a}.png")

    #     plt.close("all")
