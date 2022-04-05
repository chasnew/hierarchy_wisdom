import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from efficient_hierarchy_model import OpinionModel

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'

fixed_params = {'N': 100,
                'nlead': 2,
                'x_threshold': 0.90,
                'k': 4,
                'lead_alpha': 0.75,
                'follw_alpha': 0.25,
                'lim_listeners': 30,
                'criterion': 'prop_threshold',
                'update_coef': None,
                'speak_prob': 'non-uniform'}

opf_model = OpinionModel(**fixed_params)

max_step = 2000
opi_arrays = list()
opi_arrays.append(np.array([agent.opinion for agent in opf_model.schedule.agents]))

for i in range(max_step):

    print('step', i)
    if opf_model.running == False:
        break

    opf_model.step()

    opi_arrays.append(np.array([agent.opinion for agent in opf_model.schedule.agents]))

opi_arrays = np.array(opi_arrays)

opi_hists = []
for i in range(opi_arrays.shape[0]):
    opi_hists.append(np.histogram(opi_arrays[i], bins=20, range=(0,1))[0])

opi_hists = np.array(opi_hists)

sns.heatmap(opi_hists.T, cmap='coolwarm')
plt.show()