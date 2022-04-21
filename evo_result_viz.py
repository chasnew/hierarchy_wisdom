import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'
result_path = os.path.join(box_path, 'HierarchyWisdom', 'results', 'test_evo_results.csv')
alpha_path = os.path.join(box_path, 'HierarchyWisdom', 'results', 'alpha_array.npy')

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

# visualize alpha proportions
new_yticks = list(range(0,251,25))
ytick_labels = np.round(np.linspace(0,1,num=len(new_yticks)), 2)

ax = sns.heatmap(heatmap_array.T, cmap='coolwarm')
ax.set_yticks(new_yticks)
ax.set_yticklabels(ytick_labels)
plt.title('pooled alpha distribution of 50 patches over time')
plt.show()

ax = sns.lineplot(x='step', y='group_size', hue='group_id', data=result_df)
ax.get_legend().remove()
plt.title('population size over time of different patches')
plt.show()

sns.lineplot(x='step', y='alpha_skewness', hue='group_id', data=result_df)
plt.show()

sns.lineplot(x='step', y='n_event', hue='group_id', data=result_df)
plt.show()
