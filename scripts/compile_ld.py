"""
Script for plotting normalized averages together
same as original compile but with different data plotted.
this was created to avoid cluttering the old one. either can be used.

designed to be used to show multiple trajectories per iteration
"""

import numpy as np
import matplotlib.pyplot as plt
import IPython

max_traj = 61

dagger_10ld_names = ['compilations/svm_6init/dagger_10ld.npy']
dagger_20ld_names = ['compilations/svm_6init/dagger_20ld.npy']
supervise_10ld_names = ['compilations/svm_6init/supervise_10ld.npy']
supervise_20ld_names = ['compilations/svm_6init/supervise_20ld.npy']

dagger_data10 = np.load(dagger_10ld_names[0])
dagger_data20 = np.load(dagger_20ld_names[0])
supervise_data10 = np.load(supervise_10ld_names[0])
supervise_data20 = np.load(supervise_20ld_names[0])

ax10 = np.arange(0, max_traj, 10) + 1
ax20 = np.arange(0, max_traj, 20) + 1
axes = [ax10, ax20, ax10, ax20]

data = [dagger_data10, dagger_data20, supervise_data10, supervise_data20]
colors = ['orange', 'salmon', 'steelblue', 'cadetblue']
names = ['DAgger 10 traj', 'DAgger 20 trajs', 'Supervise 10 trajs', 'Supervise 20 trajs']

means = []
ses = []
plots = []

for dat in data:
    mean = np.mean(dat, axis=0)[:7]
    se = (np.std(dat, axis=0) / np.sqrt(dat.shape[0]))[:7]
    means.append(mean)
    ses.append(se)

for mean, se, color, name, ax in zip(means, ses, colors, names, axes):
    print name
    print len(mean), len(ax)
    a = plt.errorbar(ax, mean, se, linewidth=2.0, color=color, marker='o', ecolor='white', elinewidth=1.0, markeredgecolor=color, markeredgewidth=2.5, capsize=0, markerfacecolor='white')
    plt.errorbar(ax, mean, se, linewidth=1.0, color=color, marker='o', ecolor='black', elinewidth=1.0, markeredgecolor=color, markeredgewidth=1, markerfacecolor='white')
    plots.append(a)
    

# x_axis = [1, 21, 41, 61, 81, 101, 121, 141]
# length = len(x_axis)

# plots = []
# figure = plt.subplots(figsize=(15, 10))

# means []
# ses = []



# #mean10 = np.mean(dagger_data10, axis=0)[::2] # get every other since ones at 10 are not needed
# #mean20 = np.mean(dagger_data20, axis=0)[:8]  # limit data since 20 per iteration > 10 per iteration
# #sup_mean20 = np.mean(supervise_data20, axis=0)[:8]
# #se10 = (np.std(dagger_data10, axis=0) / np.sqrt(dagger_data10.shape[0]))[::2]
# #se20 = (np.std(dagger_data20, axis=0) / np.sqrt(dagger_data20.shape[0]))[:8]
# #sup_se20 = (np.std(supervise_data20, axis=0) / np.sqrt(supervise_data20.shape[0]))[:8]

# #means = [mean10, mean20, sup_mean20]
# #ses = [se10, se20, sup_se20]


# #for mean, se, color, name in zip(means, ses, colors, names):
# #    a = plt.errorbar(x_axis, mean[:25], se[:25], linewidth=2.0, color=color, marker='o', ecolor='white', elinewidth=1.0, markeredgecolor=color, markeredgewidth=2.5, capsize=0, markerfacecolor='white')
# #    plt.errorbar(x_axis, mean[:25], se[:25], linewidth=1.0, color=color, marker='o', ecolor='black', elinewidth=1.0, markeredgecolor=color, markeredgewidth=1, markerfacecolor='white')

# #    plots.append(a)


# # all_sups = [np.load(name) for name in dt_comparison_names]
# # colors = ['cadetblue', 'steelblue', 'orange']
# # plots = []

# # figure = plt.subplots(figsize=(15, 10))
# # for data, color, name in zip(all_sups, colors, dt_comparison_names):
# #     print name
# #     mean = np.mean(data, axis=0)
# #     se = np.std(data, axis=0) / np.sqrt(data.shape[0])



# #     a = plt.errorbar(x_axis, mean[:25], se[:25], linewidth=2.0, color=color, marker='o', ecolor='white', elinewidth=1.0, markeredgecolor=color, markeredgewidth=2.5, capsize=0, markerfacecolor='white')
# #     plt.errorbar(x_axis, mean[:25], se[:25], linewidth=1.0, color=color, marker='o', ecolor='black', elinewidth=1.0, markeredgecolor=color, markeredgewidth=1, markerfacecolor='white')

# #     plots.append(a)

IPython.embed()

mini = 1.0
for mean in means:
    mini = min(np.min(mean), mini)

# mini = max(mini, 0.0)
# plt.ylim(mini, 1)
plt.xlabel('Iterations')
plt.legend(plots, names, loc='upper center',prop={'size':12}, bbox_to_anchor=(.5, 1.12), fancybox=True, ncol=len(names))
plt.savefig('compilations/dt_batch_comparison.eps', format='eps', dpi=1000)
plt.show()
