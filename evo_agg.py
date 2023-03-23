import os
import numpy as np
import pandas as pd

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'
start_gen_list = [1, 8000]
sim = 1
sa = 1

result_list = []
pool_alpha_list = []
sep_alpha_list = []
for start_gen in start_gen_list:
    filename = 'modevo_ct{}_results{}_sim{}.csv'.format(sa, start_gen, sim)
    result_path = os.path.join(box_path, 'HierarchyWisdom', 'results', filename)

    filename = 'modpool_ct{}_alpha{}_sim{}.npy'.format(sa, start_gen, sim)
    pool_alpha_path = os.path.join(box_path, 'HierarchyWisdom', 'results', filename)

    filename = 'modsep_ct{}_alpha{}_sim{}.npy'.format(sa, start_gen, sim)
    sep_alpha_path = os.path.join(box_path, 'HierarchyWisdom', 'results', filename)

    result_list.append(pd.read_csv(result_path))
    pool_alpha_list.append(np.load(pool_alpha_path))
    sep_alpha_list.append(np.load(sep_alpha_path))

agg_result = pd.concat(result_list).reset_index(drop=True)
agg_poolalpha = np.vstack(pool_alpha_list)
agg_sepalpha = np.vstack(sep_alpha_list)

agg_result.to_csv(os.path.join(box_path, 'HierarchyWisdom', 'results',
                               'modevo_ct{}_results_sim{}.csv'.format(sa, sim)),
                  index=False)

with open(os.path.join(box_path, 'HierarchyWisdom', 'results',
                       'modpool_ct{}_alpha_sim{}.npy'.format(sa, sim)), 'wb') as file:
    np.save(file, agg_poolalpha)

with open(os.path.join(box_path, 'HierarchyWisdom', 'results',
                       'modsep_ct{}_alpha_sim{}.npy'.format(sa, sim)), 'wb') as file:
    np.save(file, agg_sepalpha)