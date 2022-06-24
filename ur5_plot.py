import os
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
# from matplotlib import rc
import numpy as np
import csv

def bin_episodes(t, G, bin_wid, interval=[0, 10000000]):
    t_tails, G_means, G_stds, G_stderrs, t_bins, G_bins = [], [], [], [], [], []
    interval_idx = np.logical_and((t > interval[0]), (t <= interval[1]))
    t, G = t[interval_idx], G[interval_idx]
    for i in range(bin_wid, t[-1] + bin_wid, bin_wid):
        bin_idx = np.logical_and((t > i - bin_wid), (t <= i))
        if not np.any(bin_idx):
            continue

        t_bin = t[bin_idx]
        G_bin = G[bin_idx]
        t_tails.append(i)
        G_means.append(np.mean(G_bin))
        G_stds.append(np.std(G_bin))
        G_stderrs.append(np.std(G_bin) / np.sqrt(G_bin.shape[0]))
        t_bins.append(t_bin)
        G_bins.append(G_bin)

    return np.array(t_tails), np.array(G_means), np.array(G_stds), np.array(G_stderrs), t_bins, G_bins

def load_returns(filename):
    timesteps = []
    returns = []
    with open(filename, 'rt', encoding='utf-8') as return_file:
        reader = csv.reader(return_file)
        for i, row in enumerate(reader):
            timesteps.append(int(row[0]))
            returns.append(float(row[1]))

    timesteps = np.array(timesteps)
    returns = np.array(returns)

    return timesteps, returns

def qlan_ur5_learning_curve():
    colors = [plt.get_cmap('tab10')(i) for i in np.arange(10)]
    color_cycler = cycler('color', colors)
    fig, ax = plt.subplots()
    fig.set_size_inches(7.767, 4.8)
    clr_id = 0

    seeds = np.arange(5)
    # seeds = np.array([4])
    algos = ['RPG', 'PPO']
    bin_wid = 2000

    for algo in algos:
        t_seeds, G_seeds = [], []
        for i_seed, seed in np.ndenumerate(seeds):
            fname = f'ur5_logs/{seed}/{algo}.csv'
            print(fname)
            ts, Gs = load_returns(fname)
            t_tails, G_means, G_stds, G_stderrs, t_bins, G_bins = bin_episodes(ts, Gs, bin_wid, interval=[0, 90000])
            t_seeds.append(t_tails)
            G_seeds.append(G_means)
        t_seeds, G_seeds = np.array(t_seeds), np.array(G_seeds)
        # print(t_seeds.shape, G_seeds.shape)

        x = t_seeds[0]
        y = G_seeds.mean(axis=0)
        y_err = G_seeds.std(axis=0) / np.sqrt(G_seeds.shape[0])
        # print(x.shape, y.shape, y_err.shape)

        ax.plot(x, y, color=colors[clr_id], label=algo)
        ax.fill_between(x, y - y_err, y + y_err, color=colors[clr_id], alpha=0.3)
        clr_id += 3

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Step')
    ax.set_ylabel('Return\naveraged\nover 5 runs', labelpad=25, verticalalignment='center').set_rotation(0)
    ax.set_title('Learning Curves on the Real-Robot Reacher Task')
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    # plt.savefig('ur5_figs/rpg_ppo3.pdf', dpi=fig.dpi)

if __name__=='__main__':
    qlan_ur5_learning_curve()
