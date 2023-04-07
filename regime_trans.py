import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

ct = 2

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'
result_path = os.path.join(box_path, 'HierarchyWisdom', 'results',
                           'modevo_ct{}_results_sim1.csv'.format(ct))

result_df = pd.read_csv(result_path)
result_df['group_id'] = result_df['group_id'].astype(str)

# trajectories of alpha mean and skewness by groups
for i in range(50):
    group_result = result_df[result_df['group_id'] == str(i)].sort_values(by='step')
    x = group_result['avg_alpha'].to_numpy()
    y = group_result['alpha_skewness'].to_numpy()

    x_pos = x[:-1]
    y_pos = y[:-1]

    x_dirs = x[1:] - x[:-1]
    y_dirs = y[1:] - y[:-1]

    # Using the quiver() function to plot multiple arrows
    plt.quiver(x_pos, y_pos, x_dirs, y_dirs, width=0.002,
               angles='xy', scale_units='xy', scale=1,
               color='green', alpha=0.3)

plt.xlim(0,1)
plt.ylim(-6,6)
plt.show()


# iterate over different values of C parameter (time cost)
cbPalette = ['#999999', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
ct_list = [0, 1, 2]

for i in range(3):
    result_path = os.path.join(box_path, 'HierarchyWisdom', 'results',
                               'modevo_ct{}_results_sim1.csv'.format(ct_list[i]))
    result_df = pd.read_csv(result_path)
    result_df['group_id'] = result_df['group_id'].astype(str)

    # trajectories of aggregated alpha mean and skewness
    agg_results = result_df.groupby('step').agg({'avg_alpha': 'mean', 'alpha_skewness': 'mean'})
    x = agg_results['avg_alpha'].to_numpy()
    y = agg_results['alpha_skewness'].to_numpy()

    x_first_last = x[-1] - x[0]
    y_first_last = y[-1] - y[0]

    x_pos = x[:-1]
    y_pos = y[:-1]

    x_dirs = x[1:] - x[:-1]
    y_dirs = y[1:] - y[:-1]

    # Using the quiver() function to plot multiple arrows
    plt.quiver(x_pos, y_pos, x_dirs, y_dirs, width=0.003,
               angles='xy', scale_units='xy', scale=1,
               color=cbPalette[i], alpha=0.3)
    plt.quiver(x_pos[0], y_pos[0], x_first_last, y_first_last, width=0.003,
               angles='xy', scale_units='xy', scale=1,
               color='black')
plt.xlim(0,1)
plt.ylim(-2,2)
plt.xlabel('Average influence')
plt.ylabel('Skewness of influence')
# plt.show()
plt.savefig('results/regime_transition.png')