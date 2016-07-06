from policy import Action, Policy
from gridworld import Grid
from state import State
from mdp import ClassicMDP
from svm_dagger import SVMDagger
from scikit_supervise import ScikitSupervise
from scikit_dagger import ScikitDagger
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from analysis import Analysis
from sklearn.svm import SVC
import scenarios
import numpy as np
import os

comparisons_directory = 'comparisons/test_comparisons/'
data_directory = 'comparisons/test_data/'

if not os.path.exists(comparisons_directory):
    os.makedirs(comparisons_directory)
if not os.path.exists(data_directory):
    os.makedirs(data_directory)


ITER = 10
TRIALS = 10
SAMP = 20
LIMIT_DATA = 1
DEPTH = 5

H = 15
W = 15

grid = Grid(15, 15, 15, 15)
rewards = scenarios.scenario6['rewards']
sinks = scenarios.scenario6['sinks']
grid.reward_states = rewards
grid.sink_states = sinks

mdp = ClassicMDP(Policy(grid), grid)
#mdp.value_iteration()
#mdp.save_policy('4dim_policy6.p')
mdp.load_policy('4dim_policy6.p')

value_iter_pi = mdp.pi

value_iter_data = np.zeros([TRIALS, ITER])
classic_il_data = np.zeros([TRIALS, ITER])
classic_il_acc = np.zeros([TRIALS, ITER])
classic_il_loss = np.zeros([TRIALS, ITER])

for t in range(TRIALS):
    print 'IL Trial: ' + str(t)
    mdp.load_policy('4dim_policy6.p')
    dt = DecisionTreeClassifier(max_depth = DEPTH)
    sup = ScikitSupervise(grid, mdp, value_iter_pi, classifier=dt, moves=80)
    
    value_iter_r = np.zeros(ITER)
    classic_il_r = np.zeros(ITER)
    acc = np.zeros(ITER)
    loss = np.zeros(ITER)
    
    for i in range(ITER):
        print "     Iteration: " + str(i)
        mdp.pi = value_iter_pi
        sup.record = True
        for _ in range(SAMP):
            if _ >= LIMIT_DATA:
                sup.record=False
            sup.rollout()
            value_iter_r[i] += sup.get_reward() / float(SAMP)
        
        sup.record = False
        print "     Training on " + str(len(sup.learner.data)) + ' examples'
        sup.train()
        acc[i] = sup.learner.acc()

        for _ in range(SAMP):
            sup.record = False
            sup.rollout()
            classic_il_r[i] += sup.get_reward() / float(SAMP)
            loss[i] += sup.get_loss() / float(SAMP)
    
    value_iter_data[t,:] = value_iter_r
    classic_il_data[t,:] = classic_il_r
    classic_il_loss[t,:] = loss
    classic_il_acc[t,:] = acc


#DAgger

dagger_data = np.zeros((TRIALS, ITER))
dagger_acc = np.zeros((TRIALS, ITER))
dagger_loss = np.zeros((TRIALS, ITER))
for t in range(TRIALS):
    print "DAgger Trial: " + str(t)
    mdp.load_policy('4dim_policy6.p')
    dt = DecisionTreeClassifier(max_depth=DEPTH)
    dagger = ScikitDagger(grid, mdp, value_iter_pi, dt, moves=80)
    dagger.record = True

    for _ in range(LIMIT_DATA):
        dagger.rollout()

    r = np.zeros(ITER)
    acc = np.zeros(ITER)
    loss = np.zeros(ITER)
    for i in range(ITER):
        print "     Iteration: " + str(i)
        print "     Retraining with " + str(len(dagger.learner.data)) + ' examples'
        dagger.retrain()
        acc[i] = dagger.learner.acc()
        dagger.record = True
        for _ in range(SAMP):
            if _ >= LIMIT_DATA: 
                dagger.record = False
            dagger.rollout()
            loss[i] += dagger.get_loss() / float(SAMP)
            r[i] += dagger.get_reward() / SAMP

    dagger_data[t,:] = r
    dagger_acc[t, :] = acc
    dagger_loss[t,:] = loss




np.save(data_directory + 'sup_data.npy', value_iter_data)
np.save(data_directory + 'classic_il_data.npy', classic_il_data)
np.save(data_directory + 'dagger_data.npy', dagger_data)

np.save(data_directory + 'dagger_acc.npy', dagger_acc)
np.save(data_directory + 'classic_il_acc.npy', classic_il_acc)

np.save(data_directory + 'dagger_loss.npy', dagger_loss)
np.save(data_directory + 'classic_il_loss.npy', classic_il_loss)


analysis = Analysis(H, W, ITER, rewards=rewards, sinks=sinks, desc="General comparison")
analysis.get_perf(value_iter_data)
analysis.get_perf(classic_il_data)
analysis.get_perf(dagger_data)

analysis.plot(names = ['Value iteration', 'DT IL', 'DT DAgger'], filename=comparisons_directory + 'reward_comparison.png', ylims=[-60, 100])

acc_analysis = Analysis(H, W, ITER, rewards = grid.reward_states, sinks=grid.sink_states, desc="Accuracy comparison")
acc_analysis.get_perf(classic_il_acc)
acc_analysis.get_perf(dagger_acc)


acc_analysis.plot(names = ['DT IL Acc.', 'DT DAgger Acc.'], label='Accuracy', filename=comparisons_directory + 'acc_comparison.png', ylims=[0,1])
loss_analysis = Analysis(H, W, ITER, rewards=rewards, sinks=sinks, desc="Loss plot")
loss_analysis.get_perf(classic_il_loss)
loss_analysis.get_perf(dagger_loss)

loss_analysis.plot(names = ['DT IL loss', 'DAgger loss'], label='Loss', filename=comparisons_directory + 'loss_plot.png', ylims=[0, 1])


