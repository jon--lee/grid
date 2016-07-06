from policy import Action, Policy
from gridworld import Grid
from state import State
from mdp import ClassicMDP
from svm_dagger import SVMDagger
from scikit_supervise import ScikitSupervise
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from analysis import Analysis
from sklearn import svm
import itertools
import scenarios
import numpy as np
import os

def make_names(ITER, TRIALS, SAMP, LIMIT_DATA, INITIAL_DATA, depth):
    short = 'comparisons/gs_4dim_dt_'
    short += str(ITER) + 'i' + str(TRIALS) + 't' + str(SAMP) + 's' + str(LIMIT_DATA) + 'ld' + str(INITIAL_DATA) + 'id' + str(depth) + 'd'
    return short + '_comparison/', short + '_data/'


def run(ITER, TRIALS, SAMP, LIMIT_DATA, INITIAL_DATA, depth):
    comparisons_directory, data_directory = make_names(ITER, TRIALS, SAMP, LIMIT_DATA, INITIAL_DATA, depth)
    
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
        dt = DecisionTreeClassifier(max_depth=depth)
        #svc = svm.SVC(kernel='rbf', gamma=0.1, C=1.0)
        #boost = AdaBoostClassifier(base_estimator=dt, n_estimators=10, algorithm='SAMME')
        sup = ScikitSupervise(grid, mdp, Classifier=dt, moves=80)

        sup.sample_policy()
        
        for _ in range(INITIAL_DATA):
            sup.record=True
            sup.rollout()

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
                sup.record=False
                sup.rollout()
                classic_il_r[i] += sup.get_reward() / float(SAMP)


        value_iter_data[t,:] = value_iter_r
        classic_il_data[t,:] = classic_il_r
        classic_il_acc[t,:] = acc 

    # DAGGER
    dagger_data = np.zeros((TRIALS, ITER))
    dagger_acc = np.zeros((TRIALS, ITER))
    for t in range(TRIALS):
        print "DAgger trial: " + str(t)
        mdp.load_policy('4dim_policy.p')
        dagger = SVMDagger(grid, mdp, moves=80, depth=depth)
        dagger.svm.nonlinear=False
        dagger.super_pi = value_iter_pi
        for _ in range(INITIAL_DATA):
            dagger.record=True
            dagger.rollout()
        r = np.zeros(ITER)
        acc = np.zeros(ITER)
        for i in range(ITER):
            print "     Iteration: " + str(i)
            print "     Retraining with " + str(len(dagger.net.data))
            dagger.retrain()
            acc[i] = dagger.svm.acc()
            dagger.record = True
            for _ in range(SAMP):
                if _ >= LIMIT_DATA:
                    dagger.record=False
                dagger.super_pi = value_iter_pi        
                dagger.rollout()
                r[i] = r[i] + dagger.get_reward() / float(SAMP)
            
        dagger_data[t,:] = r
        dagger_acc[t,:] = acc

    if not os.path.exists(comparisons_directory):
        os.makedirs(comparisons_directory)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)


    np.save(data_directory + '4dim_boost_dt_sup_data.npy', value_iter_data)
    np.save(data_directory + '4dim_boost_dt_classic_il_data.npy', classic_il_data)
    np.save(data_directory + '4dim_boost_dt_dagger_data.npy', dagger_data)

    np.save(data_directory + '4dim_boost_dt_dagger_acc.npy', dagger_acc)
    np.save(data_directory + '4dim_boost_dt_classic_il_acc.npy', classic_il_acc)

    analysis = Analysis(15, 15, ITER, rewards=rewards, sinks=sinks, desc="General reward comparison")

    analysis.get_perf(value_iter_data)
    analysis.get_perf(classic_il_data)
    analysis.get_perf(dagger_data)
    analysis.plot(names=['Value Iteration', 'DT IL', 'DT DAgger'], 
            filename=comparisons_directory+'boost_dim_dt_reward_comparison.png', ylims=[-20, 100])


    acc_analysis = Analysis(15, 15, ITER, rewards = rewards, sinks=sinks, desc='Accuracy comparison')
    acc_analysis.get_perf(classic_il_acc)
    acc_analysis.get_perf(dagger_acc)
    acc_analysis.plot(names = ['DT IL Acc.', 'DT DAgger'], label='Accuracy',
            filename=comparisons_directory+'boost_4dim_dt_acc_comparison.png', ylims=[0,1])


    return



iter_set = [25]
trials_set = [40]
samp_set = [25]
limit_data_set = [1, 5, 10, 15]
initial_data_set = [1, 5, 10, 15]
depth_set = [5, 10, 15, 20]

params = list(itertools.product(iter_set, trials_set, samp_set,
                limit_data_set, initial_data_set, depth_set))


for i in range(19, len(params)):
    param = params[i]
    print "PROCESSING " + str(i) + " OF " + str(len(params))
    run(*param)

"""
ITER = 15
TRIALS = 2
SAMP = 20
LIMIT_DATA = 1
INITIAL_DATA = 3
depth = 5
run(ITER, TRIALS, SAMP, LIMIT_DATA, INITIAL_DATA, depth)

depth = 10
run(ITER, TRIALS, SAMP, LIMIT_DATA, INITIAL_DATA, depth)

INITIAL_DATA = 5
run(ITER, TRIALS, SAMP, LIMIT_DATA, INITIAL_DATA, depth)

"""
