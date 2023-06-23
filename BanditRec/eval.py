import numpy as np
from tqdm.notebook import tqdm


def replay(log, agent, K, k, N=1, use_bootstrap=False):
    episodes = 0
    clicks = 0
    impressions = 0

    for _ in tqdm(range(N)):
        if use_bootstrap:
            log_indices = np.random.choice(len(log), size=(K * len(log)), replace=True)
        else:
            log_indices = range(len(log))

        agent.reset(K)

        ep = 0
        agent.start_episode(set(range(K)), set(), ep)
        recs = set(agent.act(k))

        for i in log_indices:
            item, clicked = log[i]
            if item in recs:
                agent.learn(item, clicked)
                recs.remove(item)

                clicks += clicked
                impressions += 1

            if len(recs) == 0:
                ep += 1
                episodes += 1
                agent.start_episode(set(), set(), ep)
                recs = set(agent.act(k))

    episodes /= N
    clicks /= N
    impressions /= N

    return episodes, clicks, impressions, clicks / impressions
