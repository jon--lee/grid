import numpy as np
import matplotlib.pyplot as plt
import os

def normalize(directory):
    # uncomment the two following lines in case of noisy supervisor
    #new_dir = directory.replace('random2d_deter_sparse_linear_noisy', 'random2d_deter_sparse_linear')
    #value_iter_data = np.load(new_dir + 'sup_data.npy')
    value_iter_data = np.load(directory + 'sup_data.npy')
    classic_il_data = np.load(directory + 'classic_il_data.npy')
    dagger_data = np.load(directory + 'dagger_data.npy')

    mean_value_iter = np.mean(value_iter_data, axis=0)
    mean_classic_il = np.mean(classic_il_data, axis=0)
    mean_dagger = np.mean(dagger_data, axis=0)

    maximum = np.mean(mean_value_iter)
    #maximum = np.amax(mean_value_iter)
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
    
    x1 = range(len(mean_classic_il))
    x2 = range(len(mean_dagger))

    plt.errorbar(x1, mean_classic_il, se_classic_il, linewidth=4.0, color='orange')
    plt.errorbar(x2, mean_dagger, se_dagger, linewidth=4.0, color='steelblue')

    """
    plt.plot(mean_classic_il, linewidth=4.0, color='orange')
    plt.plot(mean_dagger, linewidth=4.0, color='steelblue')

    plt.fill_between(range(len(mean_classic_il)), 
            mean_classic_il - se_classic_il, mean_classic_il + se_classic_il,
            facecolor='orange', alpha='.3')
    plt.fill_between(range(len(mean_dagger)), mean_dagger - se_dagger, mean_dagger + se_dagger,
            facecolor='steelblue', alpha='.3')
    """
    plt.ylim(0, 1)
    plt.ylabel(label)
    plt.xlabel('Iterations')
    names = ['Supervised Learning', 'DAgger']
    plt.legend(names,loc='upper center',prop={'size':15}, bbox_to_anchor=(.5, 1.12), fancybox=True, ncol=len(names))
    plt.savefig(filename, format='eps', dpi=1000)
    plt.show()


base_dir = 'comparisons/random2d_deter_sparse_linear'
#base_dir = 'comparisons/random2d_deter_sparse_linear_noisy'
#base_dir = 'comparisons/sparse/deter_linear'
substr = '_1ld_100d_30m_data'

def aggregate():
    #base_dir = 'comparisons/random2d'
    #base_dir = 'comparisons/sparse/deter_random'
    #base_dir = 'comparisons/random2d_deter_sparse'
    #base_dir ='comparisons/random2d_deter_sparse_linear'    
    #base_dir = 'comparisons/sparse/deter_linear'
    classic_il_data = []
    dagger_data = []
    for sub_dir in next(os.walk(base_dir))[1]:
        if substr in sub_dir:
        #if '_1ld_-1d_30m_data' in sub_dir: #and 'scen17' not in sub_dir:
        #if '_1ld_100d_50m_data' in sub_dir:
            dire = base_dir + '/' + sub_dir + '/'
            try:
                norm_classic_il, norm_dagger = normalize(dire)
                classic_il_data.append(norm_classic_il)
                dagger_data.append(norm_dagger)
                print sub_dir                
            except:
                pass
    plot(np.array(classic_il_data), np.array(dagger_data), label='Reward')#, filename='../deep_noisy_2d_reward.eps')

def aggregate_loss():
    #base_dir = 'comparisons/sparse/deter_random'    
    #base_dir = 'comparisons/random2d_deter_sparse'
    #base_dir ='comparisons/random2d_deter_sparse_linear'
    #base_dir = 'comparisons/sparse/deter_linear'    
    classic_il_data = []
    dagger_data = []
    for sub_dir in next(os.walk(base_dir))[1]:
        if substr in sub_dir:
        #if '_1ld_-1d_30m_data' in sub_dir: # and 'scen17' not in sub_dir:
        #if '_1ld_100d_50m_data' in sub_dir:
            dire = base_dir + '/' + sub_dir + '/'
            try:
                classic_il_data.append(np.mean(np.load(dire + 'classic_il_loss.npy'), axis=0))
                dagger_data.append(np.mean(np.load(dire + 'dagger_loss.npy'), axis=0))
                print sub_dir            
            except:
                pass
    plot(np.array(classic_il_data), np.array(dagger_data), label='Loss')#, filename='../deep_noisy_2d_loss.eps')


if __name__ == '__main__':
    aggregate()
    #aggregate_loss()
    #normalize('comparisons/sparse/deter_random/scen103.p_1ld_4d_50m_data/')




