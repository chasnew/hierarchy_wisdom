import os
import numpy as np
import pandas as pd

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'
start_gen_list = [1, 1001, 2001, 4001, 5001, 6001, 7001]

result_list = []
pool_alpha_list = []
sep_alpha_list = []
for start_gen in start_gen_list:
    filename = 'replicated_evo_results{}.csv'.format(start_gen)
    result_path = os.path.join(box_path, 'HierarchyWisdom', 'results', filename)

    filename = 'replicated_pool_alpha{}.npy'.format(start_gen)
    pool_alpha_path = os.path.join(box_path, 'HierarchyWisdom', 'results', filename)

    filename = 'replicated_sep_alpha{}.npy'.format(start_gen)
    sep_alpha_path = os.path.join(box_path, 'HierarchyWisdom', 'results', filename)

    result_list.append(pd.read_csv(result_path))
    pool_alpha_list.append(np.load(pool_alpha_path))
    sep_alpha_list.append(np.load(sep_alpha_path))

agg_result = pd.concat(result_list).reset_index(drop=True)
agg_poolalpha = np.vstack(pool_alpha_list)
agg_sepalpha = np.vstack(sep_alpha_list)

agg_result.to_csv(os.path.join(box_path, 'HierarchyWisdom', 'results', 'replicated_evo_results.csv'),
                  index=False)

with open(os.path.join(box_path, 'HierarchyWisdom', 'results', 'replicated_pool_alpha.npy'), 'wb') as file:
    np.save(file, agg_poolalpha)

with open(os.path.join(box_path, 'HierarchyWisdom', 'results', 'replicated_sep_alpha.npy'), 'wb') as file:
    np.save(file, agg_sepalpha)