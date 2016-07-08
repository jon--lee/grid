from gridworld import Grid
from state import State
import scenarios
from policy import Policy, SKPolicy
from scikit_supervise import ScikitSupervise
from mdp import ClassicMDP
import IPython
import pickle


if __name__ == '__main__':
    grid = Grid(15, 15, 15)
    mdp = ClassicMDP(Policy(grid), grid)
    policy = 'policies/tower.p'
    mdp.load_policy(policy)
    super_pi = mdp.pi

    dt = DecisionTreeClassifier(max_depth=3)
    boost = AdaBoostClassifier(dt, n_estimators=50)
    sup = ScikitSupervise(grid, mdp, super_pi, classifier=boost, moves=80)
    
