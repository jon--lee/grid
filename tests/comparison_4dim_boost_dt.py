from policy import Action, Policy
from gridworld import Grid
from state import State
from mdp import ClassicMDP
from svm_dagger import SVMDagger
from scikit_supervise import ScikitSupervise
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from analysis import Analysis
import scenarios
import numpy as np

comparisons_directory = 'comparisons/boost_4dim_dt_comparison/'
data_directory = 'comparisons/boost_4dim_dt_data/'

if not os.path.exists(comparisons_directory):
    os.makedirs(comparisons_directory)
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

ITER = 40
TRIALS = 50
SAMP = 30
LIMIT_DATA = 1

grid = Grid(15, 15, 15, 15)
rewards = scenarios.scenario5['rewards']
sinks = scenarios.scenario5['sinks']
grid.reward_states = rewards
grid.sink_states = sinks

mdp = ClassicMDP(Policy(grid), grid)
mdp.load_policy('4dim_policy.p')

value_iter_pi = mdp.pi

value_iter_data = np.zeros([TRIALS, ITER])
classic_il_data = np.zeros([TRIALS, ITER])
classic_il_acc = np.zeros([TRIALS, ITER])

for t in range(TRIALS):
    print "IL trial: " + str(t)
    mdp.load_policy('4dim_policy.p')
    dt = DecisionTreeClassifier(max_depth=3)
    boost = AdaBoostClassifier(base_estimator=dt, n_estimators=10, algorithm='SAMME')
    sup = ScikitSupervise(grid, mdp, Classifier=boost, moves=80)

    sup.sample_policy()
    
    value_iter_r = np.zeros(ITER)
    classic_il_r = np.zeros(ITER)
    acc = np.zeros(ITER)
    for i in range(ITER):
        print "     Iteration: " + str(i)
        mdp.pi = value_iter_pi
        sup.record = True
        for _ in range(SAMP):
            if _ >= LIMIT_DATA:
                sup.record = False
            sup.rollout()
            value_iter_r[i] += sup.get_reward() / float(SAMP)

        sup.record = False
        print "     Training on " + str(len(sup.net.data)) + ' examples'
        sup.train()
        acc[i] = sup.svm.acc()
        for _ in range(SAMP):
            sup.rollout()
            classic_il_r[i] += sup.get_reward() / SAMP

    value_iter_data[t,:] = value_iter_r
    classic_il_data[t,:] = classic_il_r
    classic_il_acc[t,:] = acc 
    
# DAGGER

dagger_data = np.zeros((TRIALS, ITER))
dagger_acc = np.zeros((TRIALS, ITER))
for t in range(TRIALS):
    print "DAgger trial: " + str(t)
    mdp.load_policy('4dim_policy.p')
    dagger = SVMDagger(grid, mdp, moves=80)
    dagger.svm.nonlinear=False
    dagger.rollout()
    r = np.zeros(ITER)
    acc = np.zeros(ITER)
    for _ in range(ITER):
        print "     Iteration: " + str(_)
        print "     Retraining with " + str(len(dagger.net.data))
        dagger.retrain()
        acc[_] = dagger.svm.acc()
        iteration_states = []
        dagger.record = True
        for i in range(SAMP):
            if i >= LIMIT_DATA:
                dagger.record=False
            dagger.rollout()
            iteration_states += dagger.get_recent_rollout_states().tolist()
            r[_] += dagger.get_reward() / float(SAMP)
        
    dagger_data[t,:] = r
    dagger_acc[t,:] = acc




np.save(data_directory + '4dim_boost_dt_sup_data.npy', value_iter_data)
np.save(data_directory + '4dim_boost_dt_classic_il_data.npy', classic_il_data)
np.save(data_directory + '4dim_boost_dt_dagger_data.npy', dagger_data)

np.save(data_directory + '4dim_boost_dt_dagger_acc.npy', dagger_acc)
np.save(data_directory + '4dim_boost_dt_classic_il_acc.npy', classic_il_acc)

analysis = Analysis(15, 15, ITER, rewards=rewards, sinks=sinks, desc="General reward comparison")

analysis.get_perf(value_iter_data)
analysis.get_perf(classic_il_data)
analysis.get_perf(dagger_data)
analysis.plot(names=['Value Iteration', 'AdaBoost IL', 'RBF SVM DAgger'], 
        filename=comparisons_directory+'boost_dim_dt_reward_comparison.png', ylims=[-60, 100])



acc_analysis = Analysis(15, 15, ITER, rewards = rewards, sinks=sinks, desc='Accuracy comparison')
acc_analysis.get_perf(classic_il_acc)
acc_analysis.get_perf(dagger_acc)
acc_analysis.plot(names = ['AdaBoost IL Acc.', 'RBF SVM DAgger'], label='Accuracy',
        filename=comparisons_directory+'boost_4dim_dt_acc_comparison.png', ylims=[0,1])




