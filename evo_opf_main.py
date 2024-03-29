import os
import numpy as np
import pandas as pd
import pickle
import yaml
from evo_hierarchy_model import EvoOpinionModel
import sys

with open('evo_opf_config.yaml') as file:
    config_params = yaml.safe_load(file)

result_path = config_params['result_path']
start_gen = config_params['start_gen']
sim_num = config_params['sim_num']
fixed_params = config_params['fixed_params']
sephist_step = config_params['sephist_step']

crit_abbv = fixed_params['criterion'].split('_')[0]

# load stored communities in case of on-going simulation
pkl_path = os.path.join(result_path, 'modevo_ct{}_{}_communities{}.pkl'.format(fixed_params['Ct'],
                                                                               crit_abbv, sim_num))

if start_gen > 1:
    with open(pkl_path, 'rb') as file:
        load_communities = pickle.load(file)

    fixed_params['load_communities'] = load_communities
else:
    fixed_params['load_communities'] = None

iter_num = config_params['iter_num']
process_num = 1 #int(sys.argv[1])

# initializing model
evo_model = EvoOpinionModel(**fixed_params)

# prepare alpha-value collection
alpha_pool = []

alpha_sep_hists = []
alpha_pool_hists = []

if start_gen <= 1:
    # extract init alpha proportions
    for c in evo_model.communities:
        alpha_list = [agent.alpha for agent in c.population]
        alpha_pool.extend(alpha_list)

        alpha_hist = np.histogram(np.array(alpha_list), bins=50, range=(0, 1))[0]
        alpha_hist = alpha_hist / np.sum(alpha_hist)
        alpha_sep_hists.append(alpha_hist)

    alpha_hist = np.histogram(np.array(alpha_pool), bins=50, range=(0, 1))[0]
    alpha_hist = alpha_hist/np.sum(alpha_hist)
    alpha_pool_hists.append(alpha_hist)

# evolutionary iterations
for i in range(iter_num):
    print('step', i)
    evo_model.step(verbose=False, process_num=process_num)

    # extract alpha proportions
    alpha_pool = []

    for c in evo_model.communities:
        alpha_list = [agent.alpha for agent in c.population]
        alpha_pool.extend(alpha_list)

        if (i % sephist_step) == 0:
            alpha_hist = np.histogram(np.array(alpha_list), bins=50, range=(0, 1))[0]
            alpha_hist = alpha_hist / np.sum(alpha_hist)
            alpha_sep_hists.append(alpha_hist)

    alpha_hist = np.histogram(np.array(alpha_pool), bins=50, range=(0, 1))[0]
    alpha_hist = alpha_hist / np.sum(alpha_hist)
    alpha_pool_hists.append(alpha_hist)

if start_gen == 0:
    filename = 'modsep_ct{}_{}_alpha_sim{}.npy'.format(fixed_params['Ct'],
                                                       crit_abbv, sim_num)
    with open(os.path.join(result_path, filename), 'wb') as file:
        np.save(file, np.array(alpha_sep_hists))

    filename = 'modpool_ct{}_{}_alpha_sim{}.npy'.format(fixed_params['Ct'],
                                                        crit_abbv, sim_num)
    with open(os.path.join(result_path, filename), 'wb') as file:
        np.save(file, np.array(alpha_pool_hists))
else:
    filename = 'modsep_ct{}_{}_alpha{}_sim{}.npy'.format(fixed_params['Ct'],
                                                         crit_abbv, start_gen, sim_num)
    with open(os.path.join(result_path, filename), 'wb') as file:
        np.save(file, np.array(alpha_sep_hists))

    filename = 'modpool_ct{}_{}_alpha{}_sim{}.npy'.format(fixed_params['Ct'],
                                                          crit_abbv, start_gen, sim_num)
    with open(os.path.join(result_path, filename), 'wb') as file:
        np.save(file, np.array(alpha_pool_hists))

# retrieve model-level results
community_data = pd.DataFrame(evo_model.datacollector)
if start_gen > 0:
    community_data['step'] = community_data['step'] + (start_gen-1)
# print(community_data.info())
# print(community_data.iloc[:5, :])

# save data
if start_gen == 0:
    filename = 'modevo_ct{}_{}_results_sim{}.csv'.format(fixed_params['Ct'],
                                                         crit_abbv, sim_num)
else:
    filename = 'modevo_ct{}_{}_results{}_sim{}.csv'.format(fixed_params['Ct'],
                                                           crit_abbv, start_gen, sim_num)

result_file = os.path.join(result_path, filename)
community_data.to_csv(result_file, index=False)

# save communities
evo_model.save_communities(filepath=pkl_path)

print('simulation is complete.')