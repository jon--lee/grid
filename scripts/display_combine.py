import numpy as np
from analysis import Analysis
import scenarios
from gridworld import Grid
import os

H = 15
W = 15
ITER = 25

rewards = scenarios.tower['rewards']
sinks = scenarios.tower['sinks']
grid = Grid(15, 15, 15)
grid.reward_states = rewards
grid.sink_states = sinks


comparisons_directory = 'comparisons/tower_combine_1ld_3d_comparisons/'
slv_dir = 'comparisons/tower_1ld_3d_data/'
lv_dir = 'comparisons/towerhvis_1ld_3d_data/'
hv_dir = 'comparisons/towerhvis2_1ld_3d_data/'

if not os.path.exists(comparisons_directory):
    os.makedirs(comparisons_directory)

value_iter_data = np.load(hv_dir + 'sup_data.npy')
slv_as_data = np.load(slv_dir + 'ada_data.npy')
slv_ad_data = np.load(slv_dir + 'adadagger_data.npy')
lv_as_data = np.load(lv_dir + 'ada_data.npy')
lv_ad_data = np.load(lv_dir + 'adadagger_data.npy')
hv_as_data = np.load(hv_dir + 'ada_data.npy')
hv_ad_data = np.load(hv_dir + 'adadagger_data.npy')


slv_as_acc = np.load(hv_dir + 'ada_acc.npy')
slv_ad_acc = np.load(hv_dir + 'adadagger_acc.npy')
lv_as_acc = np.load(lv_dir + 'ada_acc.npy')
lv_ad_acc = np.load(lv_dir + 'adadagger_acc.npy')
hv_as_acc = np.load(hv_dir + 'ada_acc.npy')
hv_ad_acc = np.load(hv_dir + 'adadagger_acc.npy')

slv_as_loss = np.load(hv_dir + 'ada_loss.npy')
slv_ad_loss = np.load(hv_dir + 'adadagger_loss.npy')
lv_as_loss = np.load(lv_dir + 'ada_loss.npy')
lv_ad_loss = np.load(lv_dir + 'adadagger_loss.npy')
hv_as_loss = np.load(hv_dir + 'ada_loss.npy')
hv_ad_loss = np.load(hv_dir + 'adadagger_loss.npy')

analysis = Analysis(H, W, ITER, rewards=rewards, sinks=sinks, desc="General comparison")
analysis.get_perf(value_iter_data)
analysis.get_perf(slv_as_data)
analysis.get_perf(slv_ad_data)
analysis.get_perf(lv_as_data)
analysis.get_perf(lv_ad_data)
analysis.get_perf(hv_as_data)
analysis.get_perf(hv_ad_data)

analysis.plot(names = ['Value iter', 'SLV AS', 'SLV AD', 'LV AS', 'LV AD', 'HV AS', 'HV AD'], filename=comparisons_directory + 'reward_comparison.eps')#, ylims=[-50, 110])

acc_analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks=grid.sink_states, desc="Accuracy comparison")
acc_analysis.get_perf(slv_as_acc)
acc_analysis.get_perf(slv_ad_acc)
acc_analysis.get_perf(lv_as_acc)
acc_analysis.get_perf(lv_ad_acc)
acc_analysis.get_perf(hv_as_acc)
acc_analysis.get_perf(hv_ad_acc)

acc_analysis.plot(names = ['SLV AS Acc.', 'SLV AD Acc.', 'LV AS Acc.', 'LV AD Acc.', 'HV AS Acc.', 'HV AD Acc.'], label='Accuracy', filename=comparisons_directory + 'acc_comparison.eps', ylims=[0,1])

loss_analysis = Analysis(H, W, ITER, rewards=rewards, sinks=sinks, desc="Loss plot")
loss_analysis.get_perf(slv_as_loss)
loss_analysis.get_perf(slv_ad_loss)
loss_analysis.get_perf(lv_as_loss)
loss_analysis.get_perf(lv_ad_loss)
loss_analysis.get_perf(hv_as_loss)
loss_analysis.get_perf(hv_ad_loss)

loss_analysis.plot(names = ['SLV AS Loss', 'SLV AD Loss', 'LV AS Loss', 'LV AD Loss', 'HV AS Loss', 'HV AD Loss'], label='Loss', filename=comparisons_directory + 'loss_plot.eps', ylims=[0, 1])


