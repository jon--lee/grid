import numpy as np
from analysis import Analysis
import scenarios
import os
from gridworld import Grid
H = 15
W = 15
ITER = 35

rewards = scenarios.tower['rewards']
sinks = scenarios.tower['sinks']
grid = Grid(H, W)
grid.reward_states = rewards
grid.sink_states = sinks


comparisons_directory = 'comparisons/deter_scen1/real/scen1.p_1ld_7d_35m_comparisons/'
hv_dir = 'comparisons/deter_scen1/real/scen1.p_1ld_7d_35m_data/'

if not os.path.exists(comparisons_directory):
    os.makedirs(comparisons_directory)

value_iter_data = np.load(hv_dir + 'sup_data.npy')
# classic_il_data = np.load(hv_dir + 'classic_il_data.npy')
# dagger_data = np.load(hv_dir + 'dagger_data.npy')
ada_data = np.load(hv_dir + 'ada_data.npy')
adadagger_data = np.load(hv_dir + 'adadagger_data.npy')


# classic_il_acc = np.load(hv_dir + 'classic_il_acc.npy')
# dagger_acc = np.load(hv_dir + 'dagger_acc.npy')
ada_acc = np.load(hv_dir + 'ada_acc.npy')
adadagger_acc = np.load(hv_dir + 'adadagger_acc.npy')


# classic_il_loss = np.load(hv_dir + 'classic_il_loss.npy')
# dagger_loss = np.load(hv_dir + 'dagger_loss.npy')
ada_loss = np.load(hv_dir + 'ada_loss.npy')
adadagger_loss = np.load(hv_dir + 'adadagger_loss.npy')

analysis = Analysis(H, W, ITER, rewards=rewards, sinks=sinks, desc="General comparison")
analysis.get_perf(value_iter_data)
# analysis.get_perf(classic_il_data)
# analysis.get_perf(dagger_data)
analysis.get_perf(ada_data, color='c')
analysis.get_perf(adadagger_data, color='m')

print "Plotting rewards"
analysis.plot(names = ['Value iter', 'Adaboost Supervised', 'Adaboost DAgger'], filename=comparisons_directory + 'reward_comparison.eps')#, ylims=[-50, 110])
# analysis.plot(names = ['Value iter', 'Supervised', 'DAgger', 'Adaboost Supervised', 'Adaboost DAgger'], filename=comparisons_directory + 'reward_comparison.eps')#, ylims=[-50, 500])

acc_analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks=grid.sink_states, desc="Accuracy comparison")
# acc_analysis.get_perf(classic_il_acc)
# acc_analysis.get_perf(dagger_acc)
acc_analysis.get_perf(ada_acc, color='c')
acc_analysis.get_perf(adadagger_acc, color='m')

acc_analysis.plot(names = ['Adaboost Acc.', 'Adaboost DAgger Acc.'], label='Accuracy', filename=comparisons_directory + 'acc_comparison.eps', ylims=[0,1])
# acc_analysis.plot(names = ['Supervised Acc.', 'DAgger Acc.', 'Adaboost Acc.', 'Adaboost DAgger Acc.'], label='Accuracy', filename=comparisons_directory + 'acc_comparison.eps', ylims=[0,1])

loss_analysis = Analysis(H, W, ITER, rewards=rewards, sinks=sinks, desc="Loss plot")
# loss_analysis.get_perf(classic_il_loss)
# loss_analysis.get_perf(dagger_loss)
loss_analysis.get_perf(ada_loss, color='c')
loss_analysis.get_perf(adadagger_loss, color='m')

loss_analysis.plot(names = ['Adaboost loss', 'Adaboost DAgger loss'], label='Loss', filename=comparisons_directory + 'loss_plot.eps', ylims=[0, 1])
# loss_analysis.plot(names = ['Supervised loss', 'DAgger loss', 'Adaboost loss', 'Adaboost DAgger loss'], label='Loss', filename=comparisons_directory + 'loss_plot.eps', ylims=[0, 1])


