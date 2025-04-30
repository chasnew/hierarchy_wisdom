import os
import numpy as np
import pandas as pd
import yaml

start_gen_list = [1, 1500]
sim = 14
ct = -1
criterion = 'sd'
track_fitness = True

with open('evo_opf_config.yaml') as file:
    config_params = yaml.safe_load(file)

box_path = config_params['result_path']

result_list = []
pool_alpha_list = []
fitness_list = []
sep_alpha_list = []
for i, start_gen in enumerate(start_gen_list):
    filename = 'modevo_ct{}_{}_results{}_sim{}.csv'.format(ct, criterion, start_gen, sim)
    result_path = os.path.join(box_path, filename)

    filename = 'modpool_ct{}_{}_alpha{}_sim{}.npy'.format(ct, criterion, start_gen, sim)
    pool_alpha_path = os.path.join(box_path, filename)

    # filename = 'modsep_ct{}_{}_alpha{}_sim{}.npy'.format(ct, criterion, start_gen, sim)
    # sep_alpha_path = os.path.join(box_path, 'HierarchyWisdom', 'results', filename)

    if track_fitness:
        filename = 'evow_ct{}_{}_sim{}_{}.csv'.format(ct, criterion, sim, i)
        fitness_path = os.path.join(box_path, filename)
        fitness_df = pd.read_csv(fitness_path)
        if start_gen > 1:
            fitness_df['step'] = fitness_df['step'] + start_gen

        fitness_list.append(fitness_df)

    result_list.append(pd.read_csv(result_path))
    pool_alpha_list.append(np.load(pool_alpha_path))
    # sep_alpha_list.append(np.load(sep_alpha_path))

agg_result = pd.concat(result_list).reset_index(drop=True)
agg_poolalpha = np.vstack(pool_alpha_list)
# agg_sepalpha = np.vstack(sep_alpha_list)

agg_result.to_csv(os.path.join(box_path,
                               'modevo_ct{}_{}_results_sim{}.csv'.format(ct, criterion, sim)),
                  index=False)

with open(os.path.join(box_path,
                       'modpool_ct{}_{}_alpha_sim{}.npy'.format(ct, criterion, sim)), 'wb') as file:
    np.save(file, agg_poolalpha)

# with open(os.path.join(box_path, 'HierarchyWisdom', 'results',
#                        'modsep_ct{}_{}_alpha_sim{}.npy'.format(ct, criterion, sim)), 'wb') as file:
#     np.save(file, agg_sepalpha)

if track_fitness:
    fitness_result = pd.concat(fitness_list).reset_index(drop=True)
    fitness_result.to_csv(os.path.join(box_path,
                                       'evow_ct{}_{}_sim{}.csv'.format(ct, criterion, sim)),
                          index=False)