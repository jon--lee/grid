import os
from gridworld import Grid
from policy import Policy
from mdp import ClassicMDP
from scikit_supervise import ScikitSupervise
from scikit_dagger import ScikitDagger
from scikit_beta_dagger import ScikitBetaDagger
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

    def __init__(self, base_name, moves, ITER, TRIALS, SAMP, INIT_TEST):
        self.ITER = ITER
        self.TRIALS = TRIALS
        self.SAMP = SAMP
        self.INIT_TEST = INIT_TEST
        # self.INITIAL_RO = None
        self.moves = moves
        self.p_beta = -1.0
        self.base_name = base_name
        self.pi_actual = None
        return

    def make_dirs(self, args, ids):
        short = 'comparisons/' + str(self.base_name) + '_'
        for arg, id_ in zip(args, ids):
            short += str(arg) + str(id_) + '_'
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
        sup = ScikitSupervise(self.grid, mdp, self.value_iter_pi, classifier=learner, moves=self.moves, super_pi_actual=None)
        
        value_iter_r = np.zeros(self.ITER)
        classic_il_r = np.zeros(self.ITER)
        acc = np.zeros(self.ITER)
        loss = np.zeros(self.ITER)
        sup_dist_loss = np.zeros(self.ITER)
        
        for i in range(self.ITER):
            print "     Iteration: " + str(i)
            mdp.pi = self.value_iter_pi
            sup.record = True
            for _ in range(self.SAMP):
                if _ >= self.LIMIT_DATA or (i == 0 and _ >= 1):
                    sup.record=False

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

            for _ in range(self.SAMP):
                sup.record = False
                sup.rollout_sup()
                sup_dist_loss[i] += sup.get_sup_loss() / float(self.SAMP)
                
        return value_iter_r, classic_il_r, acc, loss, sup_dist_loss


    def init_trial(self, mdp, learner,eps):
        mdp.load_policy(self.policy)
        sup = ScikitSupervise(self.grid, mdp, self.value_iter_pi, classifier=learner, moves=self.moves, super_pi_actual=None)
        sup.super_pi.EPS = eps
        value_iter_r = np.zeros(self.ITER)
        classic_il_r = np.zeros(self.ITER)
        eps_played = np.zeros(self.ITER)
        acc = np.zeros(self.ITER)
        loss = np.zeros(self.ITER)
        sup_dist_loss = np.zeros(self.ITER)
        print "Starting trial with eps: " + str(sup.super_pi.EPS) + ", " + str(self.value_iter_pi)
        for i in range(self.ITER):
            print "     Iteration: " + str(i)
            mdp.pi = self.value_iter_pi
            sup.record = True
            for _ in range(self.SAMP):
                if _ >= self.LIMIT_DATA or (i == 0 and _ >= 1):
                    sup.record=False

                sup.rollout()
                value_iter_r[i] += sup.get_reward() / float(self.SAMP)

            sup.record = False
            print "     Training on " + str(len(sup.learner.data)) + ' examples'
            sup.train()

            for _ in range(self.INIT_TEST):
                sup.get_current_test()

            
            eps_played[i] = sup.compute_epsilon()
            print "eps played iteration " + str(i) + ": " + str(eps_played[i])

            acc[i] = sup.learner.acc()
            for _ in range(self.SAMP):
                sup.record = False
                sup.rollout()
                loss[i] += sup.get_loss() / float(self.SAMP)
                classic_il_r[i] += sup.get_reward() / float(self.SAMP)

            for _ in range(self.SAMP):
                sup.record = False
                sup.rollout_sup()
                sup_dist_loss[i] += sup.get_sup_loss() / float(self.SAMP)
                
        return value_iter_r, classic_il_r, acc, loss, sup_dist_loss, eps_played




    def dagger_trial(self, mdp, learner):
        mdp.load_policy(self.policy)
        dagger = ScikitDagger(self.grid, mdp, self.value_iter_pi, learner, moves=self.moves, super_pi_actual=None)
        dagger.record = True
        
        for _ in range(1):
            dagger.rollout()

        r = np.zeros(self.ITER)
        acc = np.zeros(self.ITER)
        loss = np.zeros(self.ITER)
        sup_dist_loss = np.zeros(self.ITER)

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
         
            for _ in range(self.SAMP):
                dagger.record = False
                dagger.rollout_sup()
                sup_dist_loss[i] += dagger.get_sup_loss() / float(self.SAMP)

        return r, acc, loss, sup_dist_loss
                

    def beta_dagger_trial(self, mdp, learner):
        if self.p_beta < 0:
            raise Exception("Negative p_beta");
        mdp.load_policy(self.policy)
        dagger = ScikitBetaDagger(self.grid, mdp, self.value_iter_pi, learner, moves=self.moves, super_pi_actual=self.pi_actual)
        dagger.record = True
        
        for _ in range(1):
            dagger.rollout(1.0)

        r = np.zeros(self.ITER)
        acc = np.zeros(self.ITER)
        loss = np.zeros(self.ITER)
        sup_dist_loss = np.zeros(self.ITER)

        for i in range(self.ITER):
            print "     Iteration: " + str(i)
            print "     Retraining with " + str(len(dagger.learner.data)) + ' examples'
            beta = self.p_beta ** (i+1)
            dagger.retrain()
            acc[i] = dagger.learner.acc()

            dagger.record = False
            for _ in range(self.SAMP):
                dagger.record = False
                dagger.eval_rollout()
                loss[i] += dagger.get_loss() / float(self.SAMP)
                r[i] += dagger.get_reward() / float(self.SAMP)

            dagger.record = True
            for _ in range(self.LIMIT_DATA):
                dagger.record = True
                dagger.rollout(beta)

            for _ in range(self.SAMP):
                dagger.record = False
                dagger.rollout_sup()
                sup_dist_loss[i] += dagger.get_sup_loss() / float(self.SAMP)

        return r, acc, loss, sup_dist_loss



    def run(self, *args):
        raise NotImplementedError 
        
