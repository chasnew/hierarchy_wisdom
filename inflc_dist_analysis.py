import os
import numpy as np
import pandas as pd
import scipy
import yaml

sim = 1
ct = 3
fixed_popsize = "all_groups"
criterion = 'prop'
init_cond = 'most_followers'

with open('evo_opf_config.yaml') as file:
    config_params = yaml.safe_load(file)

for sim in range(1,6):
    print('sim', sim)

    box_path = config_params['result_path']
    result_path = os.path.join(box_path, 'modevo_ct{}_{}_{}_results_sim{}.csv'.format(ct, criterion, init_cond, sim))
    alpha_path = os.path.join(box_path, 'modpool_ct{}_{}_{}_alpha_sim{}.npy'.format(ct, criterion, init_cond, sim))

    result_df = pd.read_csv(result_path)
    ng = result_df['group_id'].nunique()

    alpha_raw = np.load(alpha_path, allow_pickle=True)
    print('loaded alpha array.')

    # fitting beta distribution
    beta_results = {'a': [], 'b': []}
    for i in range(alpha_raw.shape[0]):
        a, b, loc, scale = scipy.stats.beta.fit(alpha_raw[i], floc=0, fscale=1.000000001)
        beta_results['a'].append(a)
        beta_results['b'].append(b)

    print('complete fitting beta distribution.')

    beta_df = pd.DataFrame(beta_results)
    beta_df['step'] = list(range(alpha_raw.shape[0]))

    beta_df.to_csv(os.path.join(box_path, 'global_beta_ct{}_{}_{}_sim{}.csv'.format(ct, criterion, init_cond, sim)),
                   index=False)
    print('complete saving.')




# alpha distributions for each separate group
alphasep_path = os.path.join(box_path, 'modsep_ct{}_{}_{}_alpha_sim{}.npy'.format(ct, criterion, init_cond, sim))

concat_group_alpha = np.load(alphasep_path, allow_pickle=True)

print('alpha array by groups')

group_alpha_list = []

# extract array of alpha of each group from the array
for i in range(ng):
    inds = list(range(i, concat_group_alpha.shape[0], ng))
    group_alpha_list.append(concat_group_alpha[inds])

step_num = group_alpha_list[0].shape[0]
step_gap = int(alpha_raw.shape[0]/step_num)
step_list = list(range(0, step_num, step_gap))

gbeta_results = {'step': [], 'group_id': [], 'a': [], 'b':[]}

# fitting beta distribution to alpha for each group in each time step
for i in range(ng):
    print('group {}'.format(i))

    galpha_array = group_alpha_list[i]
    gbeta_results['step'].extend(step_list)
    gbeta_results['group_id'].extend([i] * step_num)

    for j in range(step_num):
        a, b, loc, scale = scipy.stats.beta.fit(galpha_array[j], floc=0, fscale=1.000000001)
        gbeta_results['a'].append(a)
        gbeta_results['b'].append(b)

gbeta_df = pd.DataFrame(gbeta_results)

gbeta_df.to_csv(os.path.join(box_path, 'group_beta_ct{}_{}_sim{}.csv'.format(ct, criterion, sim)),
                index=False)

print('complete processing and saving.')