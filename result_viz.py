import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'
result_path = os.path.join(box_path, 'HierarchyWisdom', 'results', 'consensus_results.csv')

result_df = pd.read_csv(result_path)

min_results = result_df[result_df['nlead'].isin([0,1,10])]
agg_results = min_results.groupby(['N', 'nlead'], as_index=False)[['n_event']].mean()

sns.lineplot(x='N', y='n_event', hue='nlead', data=agg_results)
plt.show()