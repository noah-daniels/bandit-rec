import numpy as np
from tqdm.notebook import tqdm
from multiprocess import Pool
from copy import copy


from BanditRec.results import Results
from BanditRec.agents import OracleAgent, ThompsonAgent


class Simulator:
    def __init__(self, setting=None, run_count=None, seed_change_interval=0):
        if seed_change_interval > 0 and run_count % seed_change_interval != 0:
            raise ValueError("run_count must be divisible by seed_change_interval.")

        self.run_count = run_count
        self.seed_change_interval = seed_change_interval

        if setting is not None:
            self.set_setting(setting)

        self.results = Results(self)

    @property
    def label(self):
        return f"N={self.run_count}/{self.seed_change_interval}"

    @property
    def single_seed(self):
        return self.seed_change_interval < 1

    def set_setting(self, setting, seed=0):
        if self.single_seed:
            setting.reseed(seed)
            self.settings = [setting]
        else:
            self.settings = []
            for i in range(self.run_count // self.seed_change_interval):
                new_s = copy(setting)
                new_s.reseed(seed + i)
                self.settings.append(new_s)

    def set_episode_length(self, episode_length):
        for s in self.settings:
            s.episode_length = episode_length

    def run_oracle(self):
        schedule = (
            (
                s,
                OracleAgent(s),
                self.single_seed,
                True,
            )
            for s in self.settings
        )

        r, e, i = self.__run_schedule(schedule, len(self.settings), False)
        r = np.repeat(r, max(self.seed_change_interval, 1), axis=0)
        if self.single_seed:
            e *= self.run_count
            i *= self.run_count

        self.results.set_oracle(self.settings[0], (r, e, i))
        return (r, e, i)

    def run_agent(
        self, agent_ctor, agent_params, label=None, progress=True, multiprocess=True
    ):
        schedule = (
            (
                self.settings[
                    0 if self.single_seed else i // self.seed_change_interval
                ],
                agent_ctor(**agent_params),
                self.single_seed,
                False,
            )
            for i in range(self.run_count)
        )

        if multiprocess:
            bundle = self.__run_schedule(schedule, self.run_count, progress)
        else:
            bundle = self.__run_schedule_single(schedule, self.run_count, progress)

        label = agent_ctor(**agent_params).label if label is None else label
        self.results.add(self.settings[0], label, bundle)
        return bundle

    def __run_schedule(self, schedule, length, progress):
        with Pool() as pool:
            if progress:
                results = list(
                    tqdm(
                        pool.imap(
                            single_run,
                            schedule,
                        ),
                        total=length,
                    )
                )
            else:
                results = list(
                    pool.imap(
                        single_run,
                        schedule,
                    )
                )

        rewards = np.stack([r[0] for r in results])
        if self.single_seed:
            estimates = np.sum([r[1] for r in results], axis=0) / self.run_count
            impressions = np.sum([r[2] for r in results], axis=0) / self.run_count
        else:
            estimates = None
            impressions = None

        return rewards, estimates, impressions

    def __run_schedule_single(self, schedule, length, progress):
        if progress:
            results = list(
                tqdm(
                    map(
                        single_run,
                        schedule,
                    ),
                    total=length,
                )
            )
        else:
            results = list(
                map(
                    single_run,
                    schedule,
                )
            )

        rewards = np.stack([r[0] for r in results])
        if self.single_seed:
            estimates = np.sum([r[1] for r in results], axis=0) / self.run_count
            impressions = np.sum([r[2] for r in results], axis=0) / self.run_count
        else:
            estimates = None
            impressions = None

        return rewards, estimates, impressions


def single_run(args):
    setting, agent, record_ei, is_oracle = args
    np.random.seed()

    # init statistics
    rewards = np.zeros(setting.episode_count)
    estimates = np.zeros((setting.item_count, setting.episode_count))
    impressions = np.zeros((setting.item_count, setting.episode_count))

    # prepare agent and setting
    agent.reset(setting)
    epsiode_generator = setting.start()

    el = 1 if is_oracle else setting.episode_length

    # run episodes
    for episode in epsiode_generator:
        # init agent for new episode
        agent.start_episode(episode.new_items, episode.expired_items, episode.t)

        # record agent estimates
        for item in agent.available_items:
            if record_ei:
                estimates[item, episode.t] += agent.estimate_ctr(item)

        # let agent act
        for _ in range(el):
            items = agent.act(setting.k)

            for item in items:
                reward, ctr = episode.recommend(item)

                # let agent learn
                agent.learn(item, reward)

                # record rewards and impression statistics
                rewards[episode.t] += ctr
                if record_ei:
                    impressions[item, episode.t] += 1

    return rewards / el, estimates, impressions / el
