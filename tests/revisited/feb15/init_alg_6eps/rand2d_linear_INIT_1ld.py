
import sys
sys.path.append('/Users/michael/LfD/jon_grid/grid')
from base_test import BaseTest

import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
import os
from analysis import Analysis
import plot_class
import random
import scenarios
from gridworld import HighVarInitStateGrid, LowVarInitStateGrid, Grid
from policy import NoisyPolicy6 as EpsPolicy         # type of policy (plain, noisy, etc.)
import numpy as np
from mdp import ClassicMDP
import random_scen
import os
import uniform_param

class RandomTest(BaseTest):

    
    def init_supervise(self):
        mdp = ClassicMDP(EpsPolicy(self.grid), self.grid)
        if not os.path.isfile(self.policy):
            mdp.value_iteration()
            mdp.save_policy(self.policy)
        mdp.load_policy(self.policy)
        self.original_EPS = mdp.pi.EPS
        self.value_iter_pi = mdp.pi

        value_iter_data =   np.zeros([self.TRIALS, self.ITER])
        classic_il_data =   np.zeros([self.TRIALS, self.ITER])
        classic_il_acc  =   np.zeros([self.TRIALS, self.ITER])
        classic_il_loss =   np.zeros([self.TRIALS, self.ITER])
        classic_il_test_loss = np.zeros([self.TRIALS, self.ITER])
        classic_il_eps = np.zeros([self.TRIALS, self.ITER])
        
        for t in range(self.TRIALS):
            print "IL Trial: " + str(t)
            print "***DO NOT TERMINATE THIS PROGRAM***"
            mdp.load_policy(self.policy)
            if self.DEPTH == -1:
                dt = LinearSVC()
            else:
                dt = DecisionTreeClassifier(max_depth=self.DEPTH)
            print self.original_EPS
            value_iter_r, classic_il_r, acc, loss, test_loss, eps_played = self.init_trial(mdp, dt,self.original_EPS)
            
            value_iter_data[t,:] = value_iter_r            
            classic_il_data[t,:] = classic_il_r
            classic_il_acc[t,:] = acc
            classic_il_loss[t,:] = loss
            classic_il_test_loss[t,:] = test_loss
            classic_il_eps[t,:] = eps_played


        return value_iter_data, classic_il_data, classic_il_acc, classic_il_loss, classic_il_test_loss, classic_il_eps
            

    def vanilla_dagger(self):
        mdp = ClassicMDP(EpsPolicy(self.grid), self.grid)
        mdp.load_policy(self.policy)
        
        self.value_iter_pi = mdp.pi

        dagger_data = np.zeros((self.TRIALS, self.ITER))
        dagger_acc = np.zeros((self.TRIALS, self.ITER))
        dagger_loss = np.zeros((self.TRIALS, self.ITER))
        dagger_test_loss = np.zeros((self.TRIALS, self.ITER))
 


        for t in range(self.TRIALS):
            print "DAgger Trial: " + str(t)
            print "***DO NOT TERMINATE THIS PROGRAM***"
            mdp.load_policy(self.policy)         
            if self.DEPTH == -1:
                dt = LinearSVC()
            else:
                dt = DecisionTreeClassifier(max_depth=self.DEPTH)
            r, acc, loss, test_loss = self.beta_dagger_trial(mdp, dt)
            
            dagger_data[t,:] = r
            dagger_acc[t, :] = acc
            dagger_loss[t,:] = loss
            dagger_test_loss[t,:] = test_loss
    
    
        return dagger_data, dagger_acc, dagger_loss, dagger_test_loss


    def run(self, LIMIT_DATA, DEPTH, MOVES, p_beta, scen, policy):
        self.LIMIT_DATA = LIMIT_DATA
        self.DEPTH = DEPTH  
        self.moves = MOVES
        self.p_beta = p_beta
        self.comparisons_directory, self.data_directory = self.make_dirs([LIMIT_DATA, DEPTH, MOVES, p_beta], ['ld', 'd', 'm', 'pb'])
        
        H = 15
        W = 15

        rewards = scen['rewards']
        sinks = scen['sinks']
        self.grid = Grid(15, 15)                # can specify which type of grid you want here (e.g. high var, low var)
        self.grid.set_reward_states(rewards)
        self.grid.set_sink_states(sinks)
        self.policy = policy
        
        if not os.path.isfile(self.data_directory + 'sup_data.npy'):
            if not os.path.exists(self.comparisons_directory):
                os.makedirs(self.comparisons_directory)
                os.makedirs(self.data_directory)
        else:
            return
        


        value_iter_data, classic_il_data, classic_il_acc, classic_il_loss, classic_il_test_loss, classic_il_eps = self.init_supervise()
        dagger_data, dagger_acc, dagger_loss, dagger_test_loss = self.vanilla_dagger()

        np.save(self.data_directory + 'eps_data.npy', classic_il_eps)
        print classic_il_eps

        np.save(self.data_directory + 'sup_data.npy', value_iter_data)
        np.save(self.data_directory + 'classic_il_data.npy', classic_il_data)
        np.save(self.data_directory + 'dagger_data.npy', dagger_data)
        
        np.save(self.data_directory + 'dagger_acc.npy', dagger_acc)
        np.save(self.data_directory + 'classic_il_acc.npy', classic_il_acc)

        np.save(self.data_directory + 'dagger_loss.npy', dagger_loss)
        np.save(self.data_directory + 'classic_il_loss.npy', classic_il_loss)

        np.save(self.data_directory + 'classic_il_test_loss.npy', classic_il_test_loss)
        np.save(self.data_directory + 'dagger_test_loss.npy', dagger_test_loss)       

        analysis = Analysis(H, W, self.ITER, rewards=rewards, sinks=sinks, desc="General comparison")
        analysis.get_perf(value_iter_data)
        analysis.get_perf(classic_il_data)
        analysis.get_perf(dagger_data)

        analysis.plot(names = ['Value iteration', 'Supervise', 'DAgger'], filename=self.comparisons_directory + 'reward_comparison.eps')#, ylims=[-60, 100])

        acc_analysis = Analysis(H, W, self.ITER, rewards = self.grid.reward_states, sinks=self.grid.sink_states, desc="Accuracy comparison")
        acc_analysis.get_perf(classic_il_acc)
        acc_analysis.get_perf(dagger_acc)

        acc_analysis.plot(names = ['Supervise Acc.', 'DAgger Acc.'], label='Accuracy', filename=self.comparisons_directory + 'acc_comparison.eps', ylims=[0,1])
        
        loss_analysis = Analysis(H, W, self.ITER, rewards=rewards, sinks=sinks, desc="Loss plot")
        loss_analysis.get_perf(classic_il_loss)
        loss_analysis.get_perf(dagger_loss)

        loss_analysis.plot(names = ['Supervise loss', 'DAgger loss'], label='Loss', filename=self.comparisons_directory + 'loss_plot.eps', ylims=[0, 1])

        test_loss_analysis = Analysis(H, W, self.ITER, rewards=rewards, sinks=sinks, desc='Sup Loss plot')
        test_loss_analysis.get_perf(classic_il_test_loss)
        test_loss_analysis.get_perf(dagger_test_loss)

        test_loss_analysis.plot(names = ['Supervise Test Loss', 'DAgger Test loss'], label='Test Loss', filename=self.comparisons_directory + 'test_loss_plot.eps', ylims=[0, 1])
        
        
        return
        


if __name__ == '__main__':

    # full trial (overnight)
    #ITER = 45
    #TRIALS = 10
    #SAMP = 15
    
    # partial trial (several hours when number of scenarios is limited)
    ITER = 15
    #TRIALS = 10
    #SAMP = 10

    # debugging     (several minutes)
    #ITER = 2
    #TRIALS = 1
    #SAMP = 2

    # specify settings (i.e. you can add moves = [30, 70] to get tests with 30 and 70 moves)
    ld_set = [1]    # (limit_data) number of trajectories per iteration added to dataset
    #d_set = [-1, 4]    # depth of decision tree (-1 for LinearSVC)
    #moves = [70]    # number of steps in each trajectory
    
    #ITER = uniform_param.ITER
    TRIALS = uniform_param.TRIALS
    SAMP = uniform_param.SAMP
    INTIT_TEST = uniform_param.INIT_SAMPLES 

    #ld_set = uniform_param.ld_set
    d_set = uniform_param.d_set
    moves = uniform_param.moves
    p_beta_set = uniform_param.p_beta_set
    
    params = list(itertools.product(ld_set, d_set, moves, p_beta_set)) # cartesian product of all settings

    for i, filename in enumerate(sorted(os.listdir('scenarios_sparse2d/'))):    
        print filename 
        print i                     # path to pickled scenarios to test on
        if i >= 20:                                                                                    # in case you want to only test on subset of scenarios (recommended because faster)
            break
        for i in range(len(params)):
            if filename.endswith('.p'):
                policy = 'policies/rand2d_linear_noisy6_' + str(filename)                            # path to optimal policy: convention for policy paths is 'policies/[description] + [scenario name]'
                scenario = random_scen.load('scenarios_sparse2d/' + filename)
                test = RandomTest('revisited/init_full/6/' + filename, 40, ITER, TRIALS, SAMP,INTIT_TEST)      # (ignore '40'), first parameter is where data/plots are saved within ./comparisons/
                print "Param " + str(i) + " of " + str(len(params))
                param = list(params[i])
                param.append(scenario)
                param.append(policy)
                test.run(*param)


