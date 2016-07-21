from gridworld import Grid, HighVarInitStateGrid
from state import State
import scenarios
from policy import Policy, SKPolicy
from scikit_supervise import ScikitSupervise
from mdp import ClassicMDP
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import IPython
import pickle


if __name__ == '__main__':
    grid = HighVarInitStateGrid(15, 15, 15)
    scen = scenarios.tower
    grid.set_reward_states(scen['rewards'])
    grid.set_sink_states(scen['sinks'])
    mdp = ClassicMDP(Policy(grid), grid)
    policy = 'policies/tower.p'
    mdp.load_policy(policy)
    super_pi = mdp.pi

    dt = DecisionTreeClassifier(max_depth=3)
    boost = AdaBoostClassifier(dt, n_estimators=50)
    sup = ScikitSupervise(grid, mdp, super_pi, classifier=boost, moves=80)
    sup.record = True
    sup.rollout()
    print sup.get_reward()
    print len(sup.learner.data)
    
