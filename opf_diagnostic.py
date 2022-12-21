import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from efficient_hierarchy_model import OpinionModel
from matplotlib import animation

plt.style.use('seaborn')

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'

fixed_params = {'N': 100,
                'nlead': 1,
                'x_threshold': 0.05,
                'k': 4,
                'lead_alpha': 0.75,
                'follw_alpha': 0.25,
                'lim_listeners': 30,
                'criterion': 'sd_threshold',
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

# opi_hists = []
# for i in range(opi_arrays.shape[0]):
#     opi_hists.append(np.histogram(opi_arrays[i], bins=20, range=(0,1))[0])
#
# opi_hists = np.array(opi_hists)
#
# sns.heatmap(opi_hists.T, cmap='coolwarm')
# plt.show()


# opinion dynamics animation
y_pos = np.random.uniform(0.05, 0.3, size=opi_arrays.shape[1])
lead = ['leader']*fixed_params['nlead'] + ['follower']*(fixed_params['N']-fixed_params['nlead'])
fig, ax = plt.subplots()

def animate_func(t):
    ax.clear()
    ax.set_xlim(0,1)
    sns.scatterplot(x=opi_arrays[t], y=y_pos, hue=lead, style=lead,
                    size=lead, sizes={'leader': 200, 'follower': 25},
                    palette="Set2", alpha=0.75, ax=ax)
    sns.kdeplot(x=opi_arrays[t], clip=(0, 1), ax=ax)
    # plt.show()

opi_anim = animation.FuncAnimation(fig, animate_func, interval=10,
                                   frames=opi_arrays.shape[0])

f = "results/opinion_dynamics.gif"
writergif = animation.PillowWriter(fps=opi_arrays.shape[0]/6)
opi_anim.save(f, writer=writergif)