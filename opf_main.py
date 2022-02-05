import os
from efficient_hierarchy_model import OpinionModel
from mesa.batchrunner import BatchRunner, BatchRunnerMP

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'

# need group size from 50 to 1000 (50 incremental step)
variable_params = {'N': range(50, 1001, 50),
                   'nlead': range(51)}
fixed_params = {'x_threshold': 0.05,
                'k': 4,
                'lead_alpha': 0.75,
                'follw_alpha': 0.25,
                'lim_listeners': 30}
n_sim = 100 # number of simulations per parameter combination

batch_run = BatchRunnerMP(OpinionModel,
                          nr_processes=None,
                          variable_parameters=variable_params,
                          fixed_parameters=fixed_params,
                          iterations=n_sim,
                          max_steps=10000,
                          model_reporters={'mean_opinion': lambda m: m.mean_opinion(),
                                           'sd_opinion': lambda m: m.sd_opinion(),
                                           'n_event': lambda m: m.n_event},
                          agent_reporters={'opinion': 'opinion',
                                           'influence': 'alpha'}
                          )
batch_run.run_all()

# retrieve model-level results
run_data = batch_run.get_model_vars_dataframe()

# drop unnecessary columns
drop_cols = list(fixed_params.keys()) + ['Run']
run_data.drop(columns=drop_cols, inplace=True)

print(run_data.info())

# save data
result_file = os.path.join(box_path, 'HierarchyWisdom', 'results', 'consensus_results.csv')
if os.path.exists(result_file):
    run_data.to_csv(result_file, index=False, header=False, mode='a')
else:
    run_data.to_csv(result_file, index=False)