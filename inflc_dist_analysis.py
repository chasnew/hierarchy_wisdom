import os
import numpy as np
import pandas as pd
import scipy

sim = 1
ct = -3
fixed_popsize = "all_groups"
criterion = 'sd'

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'
result_path = os.path.join(box_path, 'HierarchyWisdom', 'results',
                           'modevo_ct{}_{}_results_sim{}.csv'.format(ct, criterion, sim))
alpha_path = os.path.join(box_path, 'HierarchyWisdom', 'results',
                          'modpool_ct{}_{}_alpha_sim{}.npy'.format(ct, criterion, sim))

result_df = pd.read_csv(result_path)
ng = result_df['group_id'].nunique()

alpha_raw = np.load(alpha_path, allow_pickle=True)

# fitting beta distribution
beta_results = {'a': [], 'b': []}
for i in range(alpha_raw.shape[0]):
    a, b, loc, scale = scipy.stats.beta.fit(alpha_raw[i], floc=0, fscale=1.001)
    beta_results['a'].append(a)
    beta_results['b'].append(b)

beta_df = pd.DataFrame(beta_results)
beta_df['step'] = list(range(alpha_raw.shape[0]))

beta_df.to_csv(os.path.join(box_path, 'HierarchyWisdom', 'results',
                            'global_beta_ct{}_{}_sim{}.csv'.format(ct, criterion, sim)),
               index=False)


# alpha distributions for each separate group
alphasep_path = os.path.join(box_path, 'HierarchyWisdom', 'results',
                             'modsep_ct{}_{}_alpha_sim{}.npy'.format(ct, criterion, sim))

concat_group_alpha = np.load(alphasep_path, allow_pickle=True)

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
    galpha_array = group_alpha_list[i]
    gbeta_results['step'].extend(step_list)
    gbeta_results['group_id'].extend([i] * step_num)

    for j in range(step_num):
        a, b, loc, scale = scipy.stats.beta.fit(galpha_array[j], floc=0, fscale=1.001)
        gbeta_results['a'].append(a)
        gbeta_results['b'].append(b)

print('complete fitting beta distributions.')

gbeta_df = pd.DataFrame(gbeta_results)

gbeta_df.to_csv(os.path.join(box_path, 'HierarchyWisdom', 'results',
                             'group_beta_ct{}_{}_sim{}.csv'.format(ct, criterion, sim)),
                index=False)