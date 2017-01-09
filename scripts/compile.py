"""
Script for plotting normalized averages together
"""

import numpy as np
import matplotlib.pyplot as plt
import IPython
all_sup_names = ['compilations/classic_il_data0.npy', 'compilations/classic_il_data1.npy',
                 'compilations/classic_il_data3.npy', 'compilations/classic_il_data5.npy',
                 'compilations/classic_il_data6.npy', 'compilations/classic_il_data9.npy',
                 'compilations/classic_il_data10.npy']
names = ['0.0', '0.1', '0.3', '0.5', '0.6', '0.9', '1.0']
all_sups = [np.load(name) for name in all_sup_names]
colors = ['blue', 'green', 'red', 'purple', 'orange', 'steelblue', 'black']
plots = []
for data, color, name in zip(all_sups, colors, all_sup_names):
    print name
    mean = np.mean(data, axis=0)
    se = np.std(data, axis=0) / np.sqrt(data.shape[0])

    x1 = range(len(mean))

    a = plt.errorbar(x1[:25], mean[:25], se[:25], linewidth=2.0, color=color, marker='o', ecolor='white', elinewidth=1.0, markeredgecolor=color, markeredgewidth=2.5, capsize=0, markerfacecolor='white')
    plt.errorbar(x1[:25], mean[:25], se[:25], linewidth=1.0, color=color, marker='o', ecolor='black', elinewidth=1.0, markeredgecolor=color, markeredgewidth=1, markerfacecolor='white')

    plots.append(a)

plt.ylim(0, 1)
plt.xlabel('Iterations')
plt.legend(plots, names, loc='upper center',prop={'size':12}, bbox_to_anchor=(.5, 1.12), fancybox=True, ncol=len(names))
plt.savefig('compilations/compile.eps', format='eps', dpi=1000)
plt.show()
