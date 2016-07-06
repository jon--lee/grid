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

comparisons_directory = 'comparisons/dagger/'
data_directory = 'comparisons/dagger/'

ITER = 15
TRIALS = 15
SAMP = 15
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
for t in range(TRIALS):
    print "Value iteration: " + str(t)
    mdp.load_policy('4dim_policy.py')

    sup = ScikitSupervise(grid, mdp, moves=80)
    sup.sample_policy()

    value_iter_r = np.zeros(ITER)
    for i in range(ITER):
        print "     Iteration: " + str(i)
        mdp.pi = value_iter_pi
        for _ in range(SAMP):
            sup.rollout()
            value_iter_r[i] += sup.get_reward() / float(SAMP)
    
    value_iter_data[t, :] = value_iter_r


dagger_data = np.zeros((TRIALS, ITER))
for t in range(TRIALS):
    print "Dagger trial: " +str(t)
    mdp.load_policy('4dim_policy.p')
    dagger = SVMDagger(grid, mdp, moves=80)
    dagger.svm.nonlinear=True
    dagger.rollout()

    r = np.zeros(ITER)
    for _ in range(ITER):
        print "     Iteration: " + str(_)
        print "     Retraining with " + str(len(dagger.net.data))
        dagger.retrain()
        dagger.record = True
