import os
import sys
import itertools
import numpy as np
import pandas as pd
from efficient_hierarchy_model import OpinionModel
import multiprocessing as mp

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'

def simulate_opf(params, model_keys, max_step=10000):
    # change seed of each sub-process
    np.random.seed()

    # initialize OPF model
    opf_model = OpinionModel(**params)

    # opinion formation simulation
    for k in range(max_step):

        if opf_model.running == False:
            break

        opf_model.step()

    # collecting model results
    datacollector = opf_model.report_model(model_keys)

    return datacollector

run_data = []
def log_result(result):
    # a callback function for simulate_opf
    run_data.append(result)

# set up parameters for model and simulation
n_sim = 10 # number of simulations per parameter combination
max_step = 10000
variable_params = {'N': list(range(50, 101, 50)),
                   'nlead': list(range(2))}
combo_vparams = [dict(zip(variable_params.keys(), a))
                 for a in itertools.product(*variable_params.values())]

fixed_params = {'x_threshold': 0.05,
                'k': 4,
                'lead_alpha': 0.75,
                'follw_alpha': 0.25,
                'lim_listeners': 30,
                'criterion': 'sd_threshold',
                'update_coef': None,
                'speak_prob': 'non-uniform'}

# keys for model reporter
model_keys = ['mean_opinion', 'sd_opinion', 'N', 'nlead', 'n_event']

# initiate multicore-processing pool
process_num = int(sys.argv[1])
if process_num == -1:
    process_num = mp.cpu_count()

if process_num > 1:
    pool = mp.Pool(processes=process_num)

# simulation
for i in range(len(combo_vparams)):

    # pre-processing parameters
    params = fixed_params.copy()
    params.update(combo_vparams[i])
    print('variable parameters: ', combo_vparams[i])

    # simulate multiple iterations
    if process_num == 1:
        for j in range(n_sim):
            print('simulation:', j)
            sim_result = simulate_opf(params, model_keys, max_step=max_step)
            run_data.append(sim_result)
    else:
        for j in range(n_sim):
            # alternative to callback is using get()
            pool.apply_async(simulate_opf,
                             args=(params, model_keys, max_step,),
                             callback=log_result)

pool.close()
pool.join()

result_df = pd.DataFrame(run_data)
print(result_df.info())

# save data
result_file = os.path.join(box_path, 'HierarchyWisdom', 'results', 'catvote75_results.csv')
if os.path.exists(result_file):
    result_df.to_csv(result_file, index=False, header=False, mode='a')
else:
    result_df.to_csv(result_file, index=False)