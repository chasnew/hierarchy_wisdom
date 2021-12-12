from efficient_hierarchy_model import OpinionModel

group_size_list = list(range(100, 401, 50))
n_sim = 100
nlead_list = list(range(51))
x_threshold = 0.05
k = 4
lead_alpha = 0.75
follw_alpha = 0.25
lim_listeners = 30

test_model = OpinionModel(group_size_list[0], nlead_list[0],
                          x_threshold, k, lead_alpha, follw_alpha,
                          lim_listeners)
test_model.step()
opinion_stats = test_model.datacollector.get_model_vars_dataframe()
print(opinion_stats)