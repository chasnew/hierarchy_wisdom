import os
import numpy as np
import pandas as pd
import yaml
from evo_hierarchy_model import EvoOpinionModel
import multiprocessing as mp

def simulate_consensus(params, n_iter=5000, print_interval=None):
    np.random.seed()

    sim_result = {'a': [], 'b': [], 'n_event': []}
    print('initial condition =', params['init_cond'])
    for i in range(n_iter):

        # initializing model
        evo_model = EvoOpinionModel(**params)

        if print_interval != None and (i % print_interval == 0):
            print('iteration: {}'.format(i), flush=True)

        evo_model.form_decision()

        # collecting model results
        sim_result['n_event'].append(evo_model.communities[0].n_event)
        sim_result['a'].append(evo_model.init_cond[0])
        sim_result['b'].append(evo_model.init_cond[1])

        evo_model.reset_opinion()

    print('complete:', params['init_cond'])
    return sim_result

def iterate_seq(param_list, n_iter=1000, print_interval=500):

    tmp_results = []

    for i in range(len(param_list)):

        # pre-processing parameters
        params = param_list[i]
        print('variable parameters: ', params, flush=True)

        sim_result = simulate_consensus(params, n_iter, print_interval)
        tmp_results.append(sim_result)

    return tmp_results


def iterate_parallel(param_list, pool, process_num,
                     n_iter=1000, print_interval=500):

    tmp_results = []

    print('simulations with {} processes'.format(process_num))
    sim_results = list(pool.apply_async(simulate_consensus,
                                        args=(param_list[i], n_iter, print_interval,))
                   for i in range(len(param_list)))
    sim_results = [r.get() for r in sim_results]
    tmp_results.extend(sim_results)

    return tmp_results

if __name__ == '__main__':
    with open('evo_opf_config.yaml') as file:
        config_params = yaml.safe_load(file)

    result_path = config_params['result_path']
    start_gen = config_params['start_gen']
    sim_num = config_params['sim_num']
    fixed_params = config_params['fixed_params']
    sephist_step = config_params['sephist_step']
    consensus_dist = config_params['consensus_dist']
    track_fitness = config_params['track_fitness']
    fixed_popsize = config_params['fixed_popsize']

    crit_abbv = fixed_params['criterion'].split('_')[0]

    iter_num = config_params['iter_num']
    print_interval = 1
    process_num = 1  # int(sys.argv[1])

    a_beta = [0.1] #np.linspace(0.1, 2, num=5)
    b_beta = [2] #np.linspace(0.1, 2, num=5)

    param_list = []
    for a in a_beta:
        for b in b_beta:
            fixed_params['init_cond'] = [a, b]
            param_list.append(fixed_params.copy())

    # evolutionary iterations
    # initiate multicore-processing pool
    if process_num == -1:
        process_num = mp.cpu_count()

    if process_num > 1:
        pool = mp.Pool(processes=process_num)

    # activate consensus building for each community
    if process_num == 1:
        print('sequential simulation')
        tmp_results = iterate_seq(param_list, iter_num, print_interval)
    else:
        print('parallel simulation')
        tmp_results = iterate_parallel(param_list, pool, process_num,
                                       iter_num, print_interval)

        pool.close()
        pool.join()


    # retrieve model-level results
    print(tmp_results)
    print(len(tmp_results))
    combined_results = {'a': [], 'b': [], 'n_event': []}
    for i in range(len(tmp_results)):
        combined_results['a'].extend(tmp_results[i]['a'])
        combined_results['b'].extend(tmp_results[i]['b'])
        combined_results['n_event'].extend(tmp_results[i]['n_event'])

    step_data = pd.DataFrame(combined_results)
    agg_results = step_data.groupby(['a', 'b']).agg({'n_event': ['mean', 'std']})

    agg_results.columns = ['avg_n_event', 'sd_n_event']
    decisiont_df = agg_results.reset_index()
    print(decisiont_df)

    # save data
    filename = 'opfsp{}_{}_beta_decision_time{}.csv'.format(fixed_params['lim_speakers'],
                                                            crit_abbv, sim_num)

    result_file = os.path.join(result_path, filename)
    decisiont_df.to_csv(result_file, index=False)

    print('simulation is complete.')