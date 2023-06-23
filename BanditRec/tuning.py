from BanditRec.simulator import Simulator
from BanditRec.results import calcualte_ctr

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import functools, itertools, operator


def plot_grid(values, x_labels, y_labels, xlabel="", ylabel="", title=""):
    fig, ax = plt.subplots()
    im = ax.imshow(values, cmap=plt.cm.Greens)

    y_range, x_range = values.shape
    plt.xticks(np.arange(x_range), x_labels)
    plt.yticks(np.arange(y_range), y_labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    maxv = values.max()
    minv = values.min()

    half_point = minv + (maxv - minv) / 2

    for i in range(y_range):
        for j in range(x_range):
            s = (i, j)
            v = values[i, j]
            if v < 0.1:
                t = f"{v:1.4f}"
            else:
                t = f"{v:1.3f}"
            c = "white" if v > half_point else "black"
            text = ax.text(j, i, t, ha="center", va="center", color=c, fontsize=8)


def grid_parameters(parameters):
    for params in itertools.product(*parameters):
        yield params


def run_grid(setting, agent_ctor, params, experiment_name, run_count=15):
    param_keys = params.keys()
    param_values = params.values()

    sizes = list(map(len, param_values))
    total_size = functools.reduce(operator.mul, sizes, 1)

    sim = Simulator(run_count=run_count, seed_change_interval=1)
    sim.set_setting(setting)
    results = np.zeros(sizes)
    for i, p in tqdm(
        enumerate(grid_parameters(param_values)), total=total_size, smoothing=0
    ):
        ii = []
        for j, k in enumerate(param_keys):
            ii.append(i // functools.reduce(operator.mul, sizes[j + 1 :], 1) % sizes[j])

        pdict = dict(zip(param_keys, p))

        if "episode_length" in pdict:
            sim.set_episode_length(pdict["episode_length"])

        rewards, _, _ = sim.run_agent(agent_ctor, pdict, progress=False)
        results[*ii] = calcualte_ctr(rewards, tmin=1000)[0] / setting.k

    np.savez(
        f"results/{experiment_name}.npz",
        data=results,
        meta={
            "param_keys": list(param_keys),
            "param_values": list(param_values),
            "agent": agent_ctor.__name__,
            "name": experiment_name,
        },
    )


def update_grid(setting, agent_ctor, extra_params, experiment_name, run_count=15):
    with np.load(f"results/{experiment_name}.npz", allow_pickle=True) as file:
        data = file["data"]
        meta = file["meta"][()]

        pks = meta["param_keys"]

        params_dict = dict(zip(pks, meta["param_values"]))

        for k, plist in extra_params.items():
            new_params_dict = params_dict.copy()
            params_dict[k].extend(plist)
            new_params_dict[k] = plist

            axis = pks.index(k)

            sizes = list(map(len, new_params_dict.values()))
            total_size = functools.reduce(operator.mul, sizes, 1)
            sim = Simulator(run_count=run_count, seed_change_interval=1)
            sim.set_setting(setting)
            results = np.zeros(sizes)
            for i, p in tqdm(
                enumerate(grid_parameters(new_params_dict.values())),
                total=total_size,
                smoothing=0,
            ):
                ii = []
                for j, k in enumerate(pks):
                    ii.append(
                        i
                        // functools.reduce(operator.mul, sizes[j + 1 :], 1)
                        % sizes[j]
                    )

                pdict = dict(zip(pks, p))

                if "episode_length" in pdict:
                    sim.set_episode_length(pdict["episode_length"])

                rewards, _, _ = sim.run_agent(agent_ctor, pdict, progress=False)
                results[*ii] = calcualte_ctr(rewards)[0] / setting.k

            data = np.concatenate((data, results), axis=axis)

    np.savez(
        f"results/{experiment_name}.npz",
        data=data,
        meta={
            "param_keys": pks,
            "param_values": list(params_dict.values()),
            "agent": meta["agent"],
            "name": meta["name"],
        },
    )


def show_grid(experiment_name, axes, fixed=None):
    with np.load(f"results/{experiment_name}.npz", allow_pickle=True) as file:
        data = file["data"]
        meta = file["meta"][()]

        pks = meta["param_keys"]
        pvs = meta["param_values"]

        ax0 = axes[0]
        ax1 = axes[1]
        pv0 = pvs[ax0]
        pv1 = pvs[ax1]
        pk0 = pks[ax0]
        pk1 = pks[ax1]

        if fixed is None:
            fixed = []
        fixed = sorted(fixed, key=lambda x: -x[0])

        args = []
        axf = []
        d = data
        for ax, idx in fixed:
            args.append(f"{pks[ax]}={pvs[ax][idx]}")
            axf.append(ax)
            d = np.take(d, idx, axis=ax - sum(ax > i for i in axf))

        max_axes = tuple(
            ax - sum(ax > i for i in axf)
            for ax in set(range(len(pks))) - {ax0, ax1, *axf}
        )
        d = d.max(axis=max_axes)
        if ax0 < ax1:
            d = d.T

        plot_grid(d, pv0, pv1, pk0, pk1, f"{meta['name']} - {', '.join(args)}")


def calculate_prior(mean, variance):
    alpha = -(mean**3) / variance + mean**2 / variance - mean
    beta = alpha / mean - alpha
    return alpha, beta
