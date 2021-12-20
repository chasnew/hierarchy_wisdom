import os
from efficient_hierarchy_model import OpinionModel
from mesa.batchrunner import BatchRunner

# group_size_list = list(range(100, 401, 50))
# n_sim = 100
# nlead_list = list(range(51))
# x_threshold = 0.05
# k = 4
# lead_alpha = 0.75
# follw_alpha = 0.25
# lim_listeners = 30
variable_params = {'N': range(50, 101, 50)}
fixed_params = {'x_threshold': 0.05,
                'k': 4,
                'nlead': 0,
                'lead_alpha': 0.75,
                'follw_alpha': 0.25,
                'lim_listeners': 20}
n_sim = 3 # number of simulations per parameter combination

batch_run = BatchRunner(OpinionModel,
                        variable_parameters=variable_params,
                        fixed_parameters=fixed_params,
                        iterations=n_sim,
                        max_steps=1000,
                        model_reporters={'mean_opinion': lambda m: m.mean_opinion(),
                                         'sd_opinion': lambda m: m.sd_opinion(),
                                         'n_event': lambda m: m.n_event},
                        agent_reporters={'opinion': 'opinion'}
                        )
batch_run.run_all()

# retrieve model-level results
run_data = batch_run.get_model_vars_dataframe()
# print(run_data.info())
print(run_data['n_event'])
run_data.to_csv(os.path.join('results', 'test.csv'), index=False)

# data_collector_agents = batch_run.get_collector_agents()
# agent_data = data_collector_agents[(50,1)]
# print(agent_data.info())
# print(agent_data)