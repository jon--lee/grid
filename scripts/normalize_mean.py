import numpy as np
import matplotlib.pyplot as plt

def normalize(directory):
    value_iter_data = np.load(directory + 'sup_data.npy')
    classic_il_data = np.load(directory + 'classic_il_data.npy')
    dagger_data = np.load(directory + 'dagger_data.npy')

    mean_value_iter = np.mean(value_iter_data, axis=0)
    mean_classic_il = np.mean(classic_il_data, axis=0)
    mean_dagger = np.mean(dagger_data, axis=0)

    se_value_iter = np.std(value_iter_data, axis=0) / np.sqrt(value_iter_data.shape[0])
    se_classic_il = np.std(classic_il_data, axis=0) / np.sqrt(classic_il_data.shape[0])
    se_dagger = np.std(dagger_data, axis=0) / np.sqrt(dagger_data.shape[0])

    plt.errorbar(mean_classic_il)
    plt.fill_between(mean_classic_il, mean_classic_il - se_classic_il, 
            mean_classic_il + se_classic_il, facecolor='r', alpha=.5)
    plt.show()

    """
    maximum = np.amax(mean_value_iter)
    minimum = min(np.amin(mean_classic_il), np.amin(mean_dagger))
    print maximum, minimum
    rang = maximum - minimum
    print rang

    norm_value_iter = (mean_value_iter - minimum) / float(rang)
    norm_classic_il = (mean_classic_il - minimum) / float(rang)
    norm_dagger = (mean_dagger - minimum) / float(rang)

    plt.plot(norm_classic_il, linewidth=4.0, color='orange')
    plt.plot(norm_dagger, linewidth=4.0, color='steelblue')
    plt.ylim(0, 1)
    plt.show()
    ""

    return norm_classic_il, norm_dagger

    """
if __name__ == '__main__':
    normalize('comparisons/sparse/deter_random/scen103.p_1ld_4d_50m_data/')




