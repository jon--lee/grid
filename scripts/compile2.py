"""
Script for plotting normalized averages together
same as original compile but with different data plotted.
this was created to avoid cluttering the old one. either can be used.
"""

import numpy as np
import matplotlib.pyplot as plt
import IPython

dt_comparison_names = ['compilations/eps_sweep_dt/supervise_0.0eps.npy', 'compilations/eps_sweep_dt/supervise_0.2eps.npy',
                        'compilations/beta_sweep_dt/dagger_0.0beta.npy']
# svm_comparison_names = ['compilations/eps_sweep/supervise_0.0eps.npy', 'compilations/eps_sweep/supervise_0.5eps.npy',
#                     'compilations/beta_sweep/dagger_0.1beta.npy']
names = ['Supervised', 'Stochastic Supervised', 'DAgger']
all_sups = [np.load(name) for name in dt_comparison_names]
colors = ['cadetblue', 'steelblue', 'orange']
plots = []

figure = plt.subplots(figsize=(15, 10))
for data, color, name in zip(all_sups, colors, dt_comparison_names):
    print name
    mean = np.mean(data, axis=0)
    se = np.std(data, axis=0) / np.sqrt(data.shape[0])

    x1 = range(len(mean))

    a = plt.errorbar(x1[:25], mean[:25], se[:25], linewidth=2.0, color=color, marker='o', ecolor='white', elinewidth=1.0, markeredgecolor=color, markeredgewidth=2.5, capsize=0, markerfacecolor='white')
    plt.errorbar(x1[:25], mean[:25], se[:25], linewidth=1.0, color=color, marker='o', ecolor='black', elinewidth=1.0, markeredgecolor=color, markeredgewidth=1, markerfacecolor='white')

    plots.append(a)


mini = max(np.min(all_sups), 0.0)
plt.ylim(mini, 1)
plt.xlabel('Iterations')
plt.legend(plots, names, loc='upper center',prop={'size':12}, bbox_to_anchor=(.5, 1.12), fancybox=True, ncol=len(names))
plt.savefig('compilations/dt_comparison.eps', format='eps', dpi=1000)
plt.show()
