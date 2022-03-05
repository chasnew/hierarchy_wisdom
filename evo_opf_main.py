import os
import pandas as pd
from evo_hierarchy_model import EvoOpinionModel

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'

# might try varying d and lim_listeners
# variable_params = {'Ct': [2, 5]}
fixed_params = {'init_n': 50,
                'x_threshold': 0.05,
                'k': 4,
                'lim_listeners': 30,
                'np': 1,
                'mu': 0.01,
                'mu_var': 0.01,
                'K': 50,
                'ra': 2,
                'gammar': 0.025,
                'betar': 3,
                'gammab': 0.005,
                'betab': 10000,
                'S': 0.9,
                'b_mid': 500,
                'Ct': 2,
                'd': 1}

n_sim = 1 # number of simulations per parameter combination
max_step = 5

evo_model = EvoOpinionModel(**fixed_params)
for i in range(n_sim):
    for j in range(max_step):
        print('step', j)
        evo_model.step(verbose=False)

# retrieve model-level results
agent_data = pd.DataFrame(evo_model.datacollector)
print(agent_data.info())
print(agent_data.iloc[:, 1:7])

# save data
result_file = os.path.join(box_path, 'HierarchyWisdom', 'results', 'test_evo_results.csv')
# if os.path.exists(result_file):
#     agent_data.to_csv(result_file, index=False, header=False, mode='a')
# else:
agent_data.to_csv(result_file, index=False)


# scratch note

# x_list = []
# alpha_list = []
#
# for agent in c.population:
#     x_list.append(agent.opinion)
#     alpha_list.append(agent.alpha)

