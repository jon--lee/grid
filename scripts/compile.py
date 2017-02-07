"""
Script for plotting normalized averages together
"""

import numpy as np
import matplotlib.pyplot as plt
import IPython

# all_supervise_eps_names = [ 'compilations/eps_sweep_dt/supervise_0.0eps.npy',
#                             'compilations/eps_sweep_dt/supervise_0.1eps.npy',
#                             'compilations/eps_sweep_dt/supervise_0.2eps.npy',
#                             'compilations/eps_sweep_dt/supervise_0.3eps.npy',
#                             'compilations/eps_sweep_dt/supervise_0.4eps.npy',
#                             'compilations/eps_sweep_dt/supervise_0.5eps.npy',
#                             'compilations/eps_sweep_dt/supervise_0.6eps.npy',
#                             'compilations/eps_sweep_dt/supervise_0.7eps.npy',
#                             'compilations/eps_sweep_dt/supervise_0.8eps.npy',
#                             'compilations/eps_sweep_dt/supervise_0.9eps.npy',
#                             'compilations/eps_sweep_dt/supervise_1.0eps.npy']
# all_dagger_beta_names = ['compilations/beta_sweep/dagger_0.0beta.npy', 
#                             'compilations/beta_sweep/dagger_0.1beta.npy',
#                             'compilations/beta_sweep/dagger_0.2beta.npy',
#                             'compilations/beta_sweep/dagger_0.3beta.npy',
#                             'compilations/beta_sweep/dagger_0.4beta.npy',
#                             'compilations/beta_sweep/dagger_0.5beta.npy',
#                             'compilations/beta_sweep/dagger_0.6beta.npy',
#                             'compilations/beta_sweep/dagger_0.7beta.npy',
#                             'compilations/beta_sweep/dagger_0.8beta.npy',
#                             'compilations/beta_sweep/dagger_0.9beta.npy',
#                             'compilations/beta_sweep/dagger_1.0beta.npy']
# all_dagger_beta_dt_names = ['compilations/beta_sweep_dt/dagger_0.0beta.npy', 
#                             'compilations/beta_sweep_dt/dagger_0.1beta.npy',
#                             'compilations/beta_sweep_dt/dagger_0.2beta.npy',
#                             'compilations/beta_sweep_dt/dagger_0.3beta.npy',
#                             'compilations/beta_sweep_dt/dagger_0.4beta.npy',
#                             'compilations/beta_sweep_dt/dagger_0.5beta.npy',
#                             'compilations/beta_sweep_dt/dagger_0.6beta.npy',
#                             'compilations/beta_sweep_dt/dagger_0.7beta.npy',
#                             'compilations/beta_sweep_dt/dagger_0.8beta.npy',
#                             'compilations/beta_sweep_dt/dagger_0.9beta.npy',
#                             'compilations/beta_sweep_dt/dagger_1.0beta.npy']
# names = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

# all_dagger_batches = ['compilations/trajs_sweep_dagger_dt/dagger_1ld.npy',
#                         'compilations/trajs_sweep_dagger_dt/dagger_10ld.npy',
#                         'compilations/trajs_sweep_dagger_dt/dagger_20ld.npy']
# all_supervise_batches = ['compilations/trajs_sweep_supervise_dt/supervise_1ld.npy',
#                         'compilations/trajs_sweep_supervise_dt/supervise_10ld.npy',
#                         'compilations/trajs_sweep_supervise_dt/supervise_20ld.npy']

# all_dagger_batches = ['compilations/trajs_sweep_dagger/dagger_1ld.npy',
#                         'compilations/trajs_sweep_dagger/dagger_10ld.npy',
#                         'compilations/trajs_sweep_dagger/dagger_20ld.npy']
all_supervise_batches = ['compilations/trajs_sweep_supervise/supervise_1ld.npy',
                        'compilations/trajs_sweep_supervise/supervise_10ld.npy',
                        'compilations/trajs_sweep_supervise/supervise_20ld.npy']

names = ['1 traj', '10 trajs', '20 trajs']
# all_sup_names = ['compilations/classic_il_data0.npy', 'compilations/classic_il_data1.npy',
#                  'compilations/classic_il_data3.npy', 'compilations/classic_il_data5.npy',
#                  'compilations/classic_il_data6.npy', 'compilations/classic_il_data9.npy',
#                  'compilations/classic_il_data10.npy']
# names = ['0.0', '0.1', '0.3', '0.5', '0.6', '0.9', '1.0']
all_sups = [np.load(name) for name in all_supervise_batches]
colors = ['blue', 'green', 'red', 'purple', 'magenta', 'cadetblue', 'orange', 'steelblue', 'black', 'brown', 'blueviolet']
plots = []

figure = plt.subplots(figsize=(15, 10))
for data, color, name in zip(all_sups, colors, all_supervise_batches):
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
