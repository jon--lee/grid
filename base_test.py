import os
from gridworld import Grid
from policy import Policy
from mdp import ClassicMDP
from scikit_supervise import ScikitSupervise
from scikit_dagger import ScikitDagger
import numpy as np
import plot_class
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import scenarios
from analysis import Analysis
import itertools
import os
import plot_class
import numpy as np




class BaseTest():

    def __init__(self, base_name, moves, ITER, TRIALS, SAMP):
        self.ITER = ITER
        self.TRIALS = TRIALS
        self.SAMP = SAMP
        self.moves = moves
        self.base_name = base_name
        return

    def make_dirs(self, args, ids):
        ld, d = args
        ld_id, d_id = ids
        short = 'comparisons/' + str(self.base_name) + '_' + str(ld) + str(ld_id) + '_' + str(d) + str(d_id) + '_'
        return short + 'comparisons/', short + 'data/'

    def execute(self, value_iter_pi, grid):
        self.value_iter_pi = value_iter_pi
        self.grid = grid
        for i in range(self.parameters):
            param = self.parameters[i]
            run(*param)
        return

    def supervise_trial(self, mdp, learner):
        mdp.load_policy(self.policy)
        sup = ScikitSupervise(self.grid, mdp, self.value_iter_pi, classifier=learner, moves=self.moves)
        
        value_iter_r = np.zeros(self.ITER)
        classic_il_r = np.zeros(self.ITER)
        acc = np.zeros(self.ITER)
        loss = np.zeros(self.ITER)
        for i in range(self.ITER):
            print "     Iteration: " + str(i)
            mdp.pi = self.value_iter_pi
            sup.record = True
            for _ in range(self.SAMP):
                if _ >= self.LIMIT_DATA:
                    sup.record = False
                sup.rollout()
                value_iter_r[i] += sup.get_reward() / float(self.SAMP)

            sup.record = False
            print "     Training on " + str(len(sup.learner.data)) + ' examples'
            sup.train()
            acc[i] = sup.learner.acc()
            for _ in range(self.SAMP):
                sup.record = False
                sup.rollout()
                loss[i] += sup.get_loss() / float(self.SAMP)
                classic_il_r[i] += sup.get_reward() / float(self.SAMP)
        return value_iter_r, classic_il_r, acc, loss



    def dagger_trial(self, mdp, learner):
        mdp.load_policy(self.policy)
        dagger = ScikitDagger(self.grid, mdp, self.value_iter_pi, learner, moves=self.moves)
        dagger.record = True
        
        for _ in range(self.LIMIT_DATA):
            dagger.rollout()
        
        r = np.zeros(self.ITER)
        acc = np.zeros(self.ITER)
        loss = np.zeros(self.ITER)
        for i in range(self.ITER):
            print "     Iteration: " + str(i)
            print "     Retraining with " + str(len(dagger.learner.data)) + ' examples'
            dagger.retrain()
            acc[i] = dagger.learner.acc()
            dagger.record = True
            for _ in range(self.SAMP):
                if _ >= self.LIMIT_DATA: 
                    dagger.record = False
                dagger.rollout()
                loss[i] += dagger.get_loss() / float(self.SAMP)
                r[i] += dagger.get_reward() / float(self.SAMP)
         
        return r, acc, loss
                


    def run(self, *args):
        raise NotImplementedError 
        