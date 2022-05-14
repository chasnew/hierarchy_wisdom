import os
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

box_path = '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'
result_path = os.path.join(box_path, 'HierarchyWisdom', 'results', 'replicated_opf_results.csv')

result_df = pd.read_csv(result_path)

# figure 1a
sub_nleads = [0, 1, 10]
sub_df = result_df[result_df['nlead'].isin(sub_nleads)]
agg_table = pd.pivot_table(sub_df, values='n_event', index='N',
                           columns='nlead', aggfunc=['mean', 'std'])
lower_table = agg_table['mean'] - 2*agg_table['std']
upper_table = agg_table['mean'] + 2*agg_table['std']

tmp_cmap = [plt.get_cmap('Set2')(i) for i in range(len(sub_nleads))]

plt.figure(figsize=(15,6))
ax = sns.scatterplot(x='N', y='n_event', hue='nlead',
                     data=sub_df[sub_df['nlead'].isin(sub_nleads)], palette=tmp_cmap)
ax.set_xlim(0,1050)
ax.set_ylim(-200, 6500)

for i in range(len(sub_nleads)):
    agg_table['mean'][sub_nleads[i]].plot(ax=ax, color=tmp_cmap[i])
    ax.fill_between(agg_table.index.to_list(), lower_table[sub_nleads[i]],
                    upper_table[sub_nleads[i]], color=tmp_cmap[i], alpha=0.5)

# ax.set(xlabel=None, ylabel=None)
# plt.legend([],[], frameon=False)
plt.title(' Time to consensus as a function of group size and number of leaders ')
plt.savefig('results/fig1.png')

# figure 1b
nlead_list = list(range(51))
model_results = []
for nlead in nlead_list:
    tmp_df = result_df[result_df['nlead'] == nlead]
    model = smf.ols('n_event ~ N', data=tmp_df)
    mresult = model.fit()

    mdict = {'nlead': nlead, 'scalar_stress': mresult.params['N'],
             'lower': mresult.conf_int().loc['N', 0],
             'upper': mresult.conf_int().loc['N', 1]}
    model_results.append(mdict)

model_results = pd.DataFrame(model_results)

plt.figure(figsize=(15,6))
ax = model_results['scalar_stress'].plot(style='-o', figsize=(15,5))
ax.fill_between(nlead_list, model_results['lower'], model_results['upper'], alpha=0.5)
plt.ylabel('scalar stress (regression slope)')
plt.xlabel('number of leaders')
plt.title(' scalar stress as a function number of leader ')
plt.savefig('results/catvote_scalar_stress.png')

# Time to consensus while increasing number of leaders
# ax = sns.scatterplot(x='nlead', y='n_event', data=result_df,
#                      size=0.01, color='orange', alpha=0.3, legend=False)
# sns.lineplot(x='nlead', y='n_event', data=result_df,
#              ci=False, ax=ax)
# plt.xlabel('number of leaders')
# plt.ylabel('time to consensus')
# plt.title('Time to consensus as a function of leader number (proportion threshold = 75%)')
# plt.savefig('results/catvote75_inclead_time.png')