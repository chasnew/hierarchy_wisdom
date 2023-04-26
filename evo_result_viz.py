import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

ct = 0
criterion = 'prop'

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'
result_path = os.path.join(box_path, 'HierarchyWisdom', 'results',
                           'modevo_ct{}_{}_results_sim1.csv'.format(ct, criterion))
alpha_path = os.path.join(box_path, 'HierarchyWisdom', 'results',
                          'modpool_ct{}_{}_alpha_sim1.npy'.format(ct, criterion))

result_df = pd.read_csv(result_path)
result_df['group_id'] = result_df['group_id'].astype(str)

alpha_array = np.load(alpha_path)

alpha_array = alpha_array[~np.isnan(alpha_array).any(axis=1),:]

cum_alpha = alpha_array.cumsum(axis=1)

# create a heatmap to replicate figure 2
alpha_bins = np.linspace(0, 1, num=alpha_array.shape[1])
prop_bins = np.linspace(0, 1, num=250)
heatmap_array = []
for alpha_row in cum_alpha:
    prev_prop = 0
    cur_prop = 0
    gen_alpha_prop = []

    for i, alpha_cumprob in enumerate(alpha_row):
        # problems with the first (0 always excluded) and last (non-precise probs) bins
        cur_prop = alpha_cumprob

        if prev_prop == 0 and cur_prop > 0:
            alpha_num = np.sum((prop_bins >= prev_prop) & (prop_bins <= cur_prop))
        elif i == len(alpha_row) - 1:
            alpha_num = np.sum((prop_bins > prev_prop) & (prop_bins <= 1))
        else:
            alpha_num = np.sum((prop_bins > prev_prop) & (prop_bins <= cur_prop))

        gen_alpha_prop += [alpha_bins[i]] * alpha_num

        prev_prop = cur_prop

    heatmap_array.append(gen_alpha_prop)

heatmap_array = np.array(heatmap_array)

sns.set(rc={'figure.figsize': (15, 6)})

# visualize alpha proportions
new_yticks = list(range(0,251,25))
ytick_labels = np.round(np.linspace(0,1,num=len(new_yticks)), 2)

new_xticks = list(range(0,10001, 1500))

ax = sns.heatmap(heatmap_array.T, cmap='coolwarm',
                 vmin=0, vmax=1,
                 cbar_kws={'shrink': 0.5, 'label': 'influence'})
ax.set_xlabel('generation', fontsize=18)
ax.set_ylabel('proportion of individuals', fontsize=18)
ax.set_yticks(new_yticks)
ax.set_yticklabels(ytick_labels)
ax.set_xticks(new_xticks)
ax.set_xticklabels(new_xticks, rotation=0)
plt.title('pooled alpha distribution of 50 patches over time when C = {}'.format(ct))
plt.savefig('results/evoalpha_ct{}_{}_distribution1.png'.format(ct, criterion))
plt.close()

# zoomed-in heatmap
new_yticks = list(range(0,101,25))
ytick_labels = np.round(np.linspace(0.6,1,num=len(new_yticks)), 2)

new_xticks = list(range(0,101,25))
xtick_labels = list(range(9900,10001, 25))

ax = sns.heatmap(heatmap_array[9900:,150:].T, cmap='coolwarm',
                 vmin=0, vmax=1,
                 cbar_kws={'shrink': 0.5, 'label': 'influence'})

ax.set_xlabel('generation', fontsize=18)
ax.set_ylabel('proportion of individuals', fontsize=18)
ax.set_yticks(new_yticks)
ax.set_yticklabels(ytick_labels)
ax.set_xticks(new_xticks)
ax.set_xticklabels(xtick_labels, rotation=0)
plt.savefig('results/fig2_3_zoomed.png')
plt.close()

# skewness plot
ax = sns.lineplot(x='step', y='alpha_skewness', data=result_df, ci='sd')
ax.set_xlabel('generation', fontsize=18)
ax.set_ylabel('skewness of influence', fontsize=18)
plt.savefig('results/evo_ct{}_{}_alpha_skewness1.png'.format(ct, criterion))
plt.close()