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

    def set_setting(self, setting):
        if self.single_seed:
            self.settings = [setting]
        else:
            self.settings = []
            for i in range(self.run_count // self.seed_change_interval):
                new_s = copy(setting)
                new_s.reseed(i)
                self.settings.append(new_s)

    def run_oracle(self):
        schedule = (
            (
                s,
                OracleAgent(s),
                self.single_seed,
            )
            for s in self.settings
        )

        r, e, i = self.__run_schedule(schedule, len(self.settings))
        r = np.repeat(r, max(self.seed_change_interval, 1), axis=0)
        if self.single_seed:
            e *= self.run_count
            i *= self.run_count

        self.results.set_oracle(self.settings[0], (r, e, i))

    def run_agent(self, agent_ctor, agent_params, label=None):
        schedule = (
            (
                self.settings[
                    0 if self.single_seed else i // self.seed_change_interval
                ],
                agent_ctor(**agent_params),
                self.single_seed,
            )
            for i in range(self.run_count)
        )

        bundle = self.__run_schedule(schedule, self.run_count)
        label = agent_ctor(**agent_params).label if label is None else label
        self.results.add(self.settings[0], label, bundle)

    def __run_schedule(self, schedule, length):
        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(
                        single_run,
                        schedule,
                    ),
                    total=length,
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

    # def run_agent(self, agent, label=None):
    #     setting = self.settings[0]
    #     label = agent.label if label is None else label

    #     bundle = self.__run_agent(agent, setting)
    #     self.results.add(setting, label, bundle)

    # def __run_agent(self, agent, setting):
    #     # what is the reward during each epsiode
    #     rewards = np.zeros((self.run_count, setting.episode_count))
    #     # per item: what does the agent think the estimated ctr is during each episode?
    #     estimates = np.zeros((setting.item_count, setting.episode_count))
    #     # per item: how many times was the item recommended during each episode?
    #     impressions = np.zeros((setting.item_count, setting.episode_count))

    #     setting.reseed(0)

    #     # do N independent runs
    #     for run in tqdm(range(self.run_count)):
    #         if self.seed_change_interval > 0 and run % self.seed_change_interval == 0:
    #             setting.reseed(run // self.seed_change_interval)

    #         agent.reset(setting.item_count)
    #         for episode in setting.start():
    #             agent.start_episode(episode.new_items, episode.t)

    #             for item in agent.available_items:
    #                 estimates[item, episode.t] += agent.estimate_ctr(item)

    #             items = []
    #             for _ in range(self.batch_size):
    #                 items.extend(agent.act(setting.k))

    #             for item in items:
    #                 reward, ctr = episode.recommend(item)
    #                 agent.learn(item, reward)

    #                 rewards[run, episode.t] += ctr
    #                 impressions[item, episode.t] += 1

    #     # end N independent runs

    #     # compute statistics
    #     rewards = rewards
    #     estimates = estimates / self.run_count
    #     impressions = impressions / self.run_count
    #     return rewards, estimates, impressions


def single_run(args):
    setting, agent, record_ei = args
    np.random.seed()

    # init statistics
    rewards = np.zeros(setting.episode_count)
    if record_ei:
        estimates = np.zeros((setting.item_count, setting.episode_count))
        impressions = np.zeros((setting.item_count, setting.episode_count))
    else:
        estimates = None
        impressions = None

    # prepare agent and setting
    agent.reset(setting.item_count)
    epsiode_generator = setting.start()

    # run episodes
    for episode in epsiode_generator:
        # init agent for new episode
        agent.start_episode(episode.new_items, episode.expired_items, episode.t)

        # record agent estimates
        for item in agent.available_items:
            if record_ei:
                estimates[item, episode.t] += agent.estimate_ctr(item)

        # let agent act
        for _ in range(setting.episode_length):
            items = agent.act(setting.k)

            for item in items:
                reward, ctr = episode.recommend(item)

                # let agent learn
                agent.learn(item, reward)

                # record rewards and impression statistics
                rewards[episode.t] += ctr
                if record_ei:
                    impressions[item, episode.t] += 1

    return rewards, estimates, impressions
