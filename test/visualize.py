import os, pickle
import numpy as np
import pandas as pd

from virtual_rodent.utils import stats_to_dataframe
from virtual_rodent.visualization import plot_stats, plot_smooth_training_curve, plot_rewards

paths = [
        'out_mlp_0',
        'out_mlp_1',
        ]

if __name__ == '__main__':
    exclude = ['policy_weight', 'value_weight', 'entropy_weight']
    for i in (0, 1):
        try:
            with open(os.path.join(paths[i], 'training_stats.pkl'), 'rb') as f:
                d = pickle.load(f)

            exclude = ['critic_weight', 'actor_weight', 'entropy_weight']
            if len(df['learning_rate']) == 0:
                exclude.append('learning_rate')
            df = stats_to_dataframe(d, exclude=exclude)
            plot_stats(df, os.path.join(paths[i], 'training_stats.png'), plot=plot_smooth_training_curve)

            rewards = np.load(os.path.join(paths[i], 'rewards.npy'))
            plot_rewards(rewards, os.path.join(paths[i], 'rewards.png'))

        except FileNotFoundError as e:
            print(e)
            continue
