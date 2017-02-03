import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = 'comparisons/revisited/test_loss/5'
ref_root = 'comparisons/random2d_deter_sparse_linear'
substr = '_1ld_-1d_70m_data'


def normalize(directory):
    ref_dir = directory.replace(base_dir, ref_root)
    value_iter_data = np.load(ref_dir + 'sup_data.npy')
    classic_il_data = np.load(directory + 'classic_il_data.npy')
    dagger_data = np.load(directory + 'dagger_data.npy')

    mean_value_iter = np.mean(value_iter_data, axis=0)
    mean_classic_il = np.mean(classic_il_data, axis=0)
    mean_dagger = np.mean(dagger_data, axis=0)

    maximum = np.mean(mean_value_iter)
    minimum = min(np.amin(mean_classic_il), np.amin(mean_dagger))
    rang = maximum - minimum

    norm_value_iter = (mean_value_iter - minimum) / float(rang)
    norm_classic_il = (mean_classic_il - minimum) / float(rang)
    norm_dagger = (mean_dagger - minimum) / float(rang)


    return norm_classic_il, norm_dagger

def plot(classic_il_data, dagger_data, label='Reward', filename='tmp.eps'):
    mean_classic_il = np.mean(classic_il_data, axis=0)
    mean_dagger = np.mean(dagger_data, axis=0)

    se_classic_il = np.std(classic_il_data, axis=0) / np.sqrt(classic_il_data.shape[0])
    se_dagger = np.std(dagger_data, axis=0) / np.sqrt(dagger_data.shape[0])

    # save the normalized data for comparisons with others if needed
    #np.save('compilations/classic_il_data3.npy', classic_il_data)
    #np.save('compilations/dagger_data3.npy', dagger_data)
    
    x1 = range(len(mean_classic_il))
    x2 = range(len(mean_dagger))

    plt.errorbar(x1[:25], mean_classic_il[:25], se_classic_il[:25], linewidth=2.0, color='orange', marker='o', ecolor='white', elinewidth=1.0, markeredgecolor='orange', markeredgewidth=2.5, capsize=0, markerfacecolor='white')
    plt.errorbar(x2[:25], mean_dagger[:25], se_dagger[:25], linewidth=2.0, color='steelblue', marker='o', ecolor='white', elinewidth=1.0, markeredgecolor='steelblue', markeredgewidth=2.5, capsize=0, markerfacecolor='white')
    plt.errorbar(x1[:25], mean_classic_il[:25], se_classic_il[:25], linewidth=1.0, color='orange', marker='o', ecolor='black', elinewidth=1.0, markeredgecolor='orange', markeredgewidth=1, markerfacecolor='white')
    plt.errorbar(x2[:25], mean_dagger[:25], se_dagger[:25], linewidth=1.0, color='steelblue', marker='o', ecolor='black', elinewidth=1.0, markeredgecolor='steelblue', markeredgewidth=1, markerfacecolor='white')

    plt.ylim(0, 1)
    plt.ylabel(label)
    plt.xlabel('Iterations')
    names = ['Supervised Learning', 'DAgger']
    plt.legend(names,loc='upper center',prop={'size':15}, bbox_to_anchor=(.5, 1.12), fancybox=True, ncol=len(names))
    plt.savefig(filename, format='eps', dpi=1000)
    plt.show()



def aggregate():
    classic_il_data = []
    dagger_data = []
    for sub_dir in next(os.walk(base_dir))[1]:
        if substr in sub_dir:
            dire = base_dir + '/' + sub_dir + '/'
            norm_classic_il, norm_dagger = normalize(dire)
            classic_il_data.append(norm_classic_il)
            dagger_data.append(norm_dagger)
    plot(np.array(classic_il_data), np.array(dagger_data), label='Reward', filename='images/tmp_reward.eps')

def aggregate_loss():
    classic_il_data = []
    dagger_data = []
    for sub_dir in next(os.walk(base_dir))[1]:
        if substr in sub_dir:
            dire = base_dir + '/' + sub_dir + '/'
            try:
                classic_il_data.append(np.mean(np.load(dire + 'classic_il_loss.npy'), axis=0))
                dagger_data.append(np.mean(np.load(dire + 'dagger_loss.npy'), axis=0))
                print sub_dir            
            except:
                pass
    plot(np.array(classic_il_data), np.array(dagger_data), label='Loss', filename='images/tmp_loss.eps')

def aggregate_test_loss():
    classic_il_data = []
    dagger_data = []
    for sub_dir in next(os.walk(base_dir))[1]:
        if substr in sub_dir:
            dire = base_dir + '/' + sub_dir + '/'
            try:
                classic_il_data.append(np.mean(np.load(dire + 'classic_il_test_loss.npy'), axis=0))
                dagger_data.append(np.mean(np.load(dire + 'dagger_test_loss.npy'), axis=0))
                print sub_dir            
            except:
                pass
    plot(np.array(classic_il_data), np.array(dagger_data), label='Loss', filename='images/tmp_test_loss.eps')
    

if __name__ == '__main__':
    aggregate()
    aggregate_loss()
    aggregate_test_loss()



