import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'
result_path = os.path.join(box_path, 'HierarchyWisdom', 'results', 'replicated_evo_results_sim1.csv')
alpha_path = os.path.join(box_path, 'HierarchyWisdom', 'results', 'replicated_pool_alpha_sim1.npy')

result_df = pd.read_csv(result_path)
result_df['group_id'] = result_df['group_id'].astype(str)

alpha_array = np.load(alpha_path)
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
# plt.title('pooled alpha distribution of 50 patches over time')
plt.savefig('results/fig2.png')
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
plt.savefig('results/fig2_zoomed.png')
plt.close()

# skewness plot
ax = sns.lineplot(x='step', y='alpha_skewness', data=result_df)
ax.set_xlabel('generation', fontsize=18)
ax.set_ylabel('skewness of influence', fontsize=18)
plt.savefig('results/evo_alpha_skewness.png')
plt.close()

# sample_groups = np.random.choice(np.arange(50), size=5, replace=True).astype(str)
# sample_results = result_df[result_df['group_id'].isin(sample_groups)]
#
# ax = sns.lineplot(x='step', y='n_event', hue='group_id',
#                   estimator=None, lw=1, data=result_df)
# ax.get_legend().remove()
# plt.show()