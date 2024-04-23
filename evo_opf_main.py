import os
import numpy as np
import pandas as pd
import joblib
import yaml
from evo_hierarchy_model import EvoOpinionModel
import sys


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

    if track_fitness:
        consensus_dist = False

    crit_abbv = fixed_params['criterion'].split('_')[0]

    # load stored communities in case of on-going simulation
    pkl_path = os.path.join(result_path, 'modevo_ct{}_{}_communities{}.pkl'.format(fixed_params['Ct'],
                                                                                   crit_abbv, sim_num))

    if start_gen > 1:
        # with open(pkl_path, 'rb') as file:
        load_communities = joblib.load(pkl_path)

        fixed_params['load_communities'] = load_communities
    else:
        fixed_params['load_communities'] = None

    iter_num = config_params['iter_num']
    process_num = 1  # int(sys.argv[1])


    # initializing model
    evo_model = EvoOpinionModel(**fixed_params)

    # prepare alpha-value collection
    alpha_pool = []

    alpha_sep_hists = []
    alpha_pool_hists = []
    cum_cd_pool = []
    cum_w_pool = []

    cum_mv_pool = []

    if start_gen <= 1:
        # extract init alpha proportions
        for c in evo_model.communities:
            alpha_list = [agent.alpha for agent in c.population]
            alpha_pool.extend(alpha_list)

            if fixed_popsize == "all_groups":
                alpha_sep_hists.append(np.array(alpha_list))
            else:
                alpha_hist = np.histogram(np.array(alpha_list), bins=50, range=(0, 1))[0]
                alpha_hist = alpha_hist / np.sum(alpha_hist)
                alpha_sep_hists.append(alpha_hist)

        if fixed_popsize == "all_groups":
            alpha_pool_hists.append(np.array(alpha_pool))
        else:
            alpha_hist = np.histogram(np.array(alpha_pool), bins=50, range=(0, 1))[0]
            alpha_hist = alpha_hist/np.sum(alpha_hist)
            alpha_pool_hists.append(alpha_hist)

    # evolutionary iterations
    for i in range(iter_num):
        print('step', i, flush=True)
        # evo_model.step(verbose=False, capped_pop=capped_popsize, process_num=process_num)
        evo_model.form_decision(verbose=False, process_num=process_num)
        evo_model.step_count += 1

        # collecting model results
        for key, collect_func in evo_model.agent_reporter.items():
            evo_model.datacollector[key].extend(list(map(collect_func, evo_model.communities)))

        evo_model.datacollector['step'].extend([evo_model.step_count] * evo_model.ng)

        # extract individual-level aggregate data
        alpha_pool = []
        cd_pool = []
        w_pool = []
        mv_pool = []

        for c in evo_model.communities:

            if consensus_dist:
                alpha_list = []
                cd_list = []
                mv_list = []
                for agent in c.population:
                    alpha_list.append(agent.alpha)
                    cd_list.append(agent.con_dist)
                    mv_list.append(agent.mv_dist)

                # print(cd_list)
                alpha_pool.extend(alpha_list)
                cd_pool.extend(cd_list)
                mv_pool.extend(mv_list)
            elif track_fitness:
                alpha_list = []
                w_list = []
                for agent in c.population:
                    alpha_list.append(agent.alpha)
                    w_list.append(agent.w)

                # print(cd_list)
                alpha_pool.extend(alpha_list)
                w_pool.extend(w_list)
            else:
                alpha_list = [agent.alpha for agent in c.population]
                alpha_pool.extend(alpha_list)



            if (i % sephist_step) == 0:
                if fixed_popsize == "all_groups":
                    alpha_sep_hists.append(np.array(alpha_list))
                else:
                    alpha_hist = np.histogram(np.array(alpha_list), bins=50, range=(0, 1))[0]
                    alpha_hist = alpha_hist / np.sum(alpha_hist)
                    alpha_sep_hists.append(alpha_hist)



        if consensus_dist:
            tmp_cd = pd.DataFrame({'alpha': alpha_pool, 'con_dist': cd_pool, 'mv_dist': mv_pool, 'step': i})
            tmp_cd['alpha_bin'] = tmp_cd['alpha'].map(lambda alpha: int(np.floor(alpha / 0.1)) if alpha != 1 else 9)
            tmp_agg_cd = tmp_cd.groupby('alpha_bin').agg({'con_dist': 'mean', 'mv_dist': 'mean',
                                                          'step': 'first'}).reset_index()
            cum_cd_pool.append(tmp_agg_cd)
        elif track_fitness:
            # need to track fitness in parent's generation
            tmp_w = pd.DataFrame({'alpha': alpha_pool, 'w': w_pool, 'step': i})
            tmp_w['alpha_bin'] = tmp_w['alpha'].map(lambda alpha: int(np.floor(alpha / 0.1)) if alpha != 1 else 9)
            tmp_agg_w = tmp_w.groupby('alpha_bin').agg({'w': ['mean', 'count'], 'step': 'first'}).reset_index()
            tmp_agg_w.columns = ['alpha_bin', 'avg_w', 'n', 'step']
            cum_w_pool.append(tmp_agg_w)

        if fixed_popsize == "all_groups":
            alpha_pool_hists.append(np.array(alpha_pool))
        else:
            alpha_hist = np.histogram(np.array(alpha_pool), bins=50, range=(0, 1))[0]
            alpha_hist = alpha_hist / np.sum(alpha_hist)
            alpha_pool_hists.append(alpha_hist)

        # Reproduction and Migration
        # evo_model.reset_opinion()

        if fixed_popsize:
            evo_model.rescale_pop(process_num=process_num)
        else:
            evo_model.reproduce(process_num=process_num)

        # fixed_popsize can be "all_groups" to keep all groups at a constant size
        evo_model.migrate(fixed_popsize)

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

    if consensus_dist:
        cum_cd_df = pd.concat(cum_cd_pool).reset_index(drop=True)
        print(cum_cd_df.head())

        filename = 'evocondist_ct{}_{}_sim{}.csv'.format(fixed_params['Ct'],
                                                         crit_abbv, sim_num)
        result_file = os.path.join(result_path, filename)
        cum_cd_df.to_csv(result_file, index=False)
    elif track_fitness:
        cum_w_df = pd.concat(cum_w_pool).reset_index(drop=True)
        print(cum_w_df.head())

        filename = 'evow_ct{}_{}_sim{}.csv'.format(fixed_params['Ct'],
                                                         crit_abbv, sim_num)
        result_file = os.path.join(result_path, filename)
        cum_w_df.to_csv(result_file, index=False)

    # retrieve model-level results
    community_data = pd.DataFrame(evo_model.datacollector)
    if start_gen > 0:
        community_data['step'] = community_data['step'] + (start_gen-1)
    print(community_data.info())
    print(community_data.iloc[:10, :])

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