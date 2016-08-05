"""
A series of experiments on random scenarios in 3D
using the typical learning methods such as those used
in the tower experiments.
"""


from base_test import BaseTest
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import os
from analysis import Analysis
import plot_class
import random
import scenarios
from gridworld import LowVarInitStateGrid
from policy import Policy
import numpy as np
from mdp import ClassicMDP
import random_scen

class RandomTest(BaseTest):
    
    def vanilla_supervise(self):
        mdp = ClassicMDP(Policy(self.grid), self.grid)
        #mdp.value_iteration()
        #mdp.save_policy(self.policy)
        mdp.load_policy(self.policy)
        self.value_iter_pi = mdp.pi

        value_iter_data =   np.zeros([self.TRIALS, self.ITER])
        classic_il_data =   np.zeros([self.TRIALS, self.ITER])
        classic_il_acc  =    np.zeros([self.TRIALS, self.ITER])
        classic_il_loss =   np.zeros([self.TRIALS, self.ITER])

        for t in range(self.TRIALS):
            print "IL Trial: " + str(t)
            print "*** DO NOT TERMINATE THIS PROGRAM"
            mdp.load_policy(self.policy)
            dt = DecisionTreeClassifier(max_depth=self.DEPTH)
            value_iter_r, classic_il_r, acc, loss = self.supervise_trial(mdp, dt)
            
            value_iter_data[t,:] = value_iter_r            
            classic_il_data[t,:] = classic_il_r
            classic_il_acc[t,:] = acc
            classic_il_loss[t,:] = loss


        return value_iter_data, classic_il_data, classic_il_acc, classic_il_loss
            

    def vanilla_dagger(self):
        mdp = ClassicMDP(Policy(self.grid), self.grid)
        #mdp.value_iteration()
        #mdp.save_policy(self.policy)
        mdp.load_policy(self.policy)
        
        self.value_iter_pi = mdp.pi

        dagger_data = np.zeros((self.TRIALS, self.ITER))
        dagger_acc = np.zeros((self.TRIALS, self.ITER))
        dagger_loss = np.zeros((self.TRIALS, self.ITER))

        for t in range(self.TRIALS):
            print "DAgger Trial: " + str(t)
            print "***DO NOT TERMINATE THIS PROGRAM***"
            mdp.load_policy(self.policy)            
            dt = DecisionTreeClassifier(max_depth=self.DEPTH)
            r, acc, loss = self.dagger_trial(mdp, dt)
            
            dagger_data[t,:] = r
            dagger_acc[t, :] = acc
            dagger_loss[t,:] = loss
    
    
        return dagger_data, dagger_acc, dagger_loss

    def boosted_supervise(self):
        mdp = ClassicMDP(Policy(self.grid), self.grid)
        #mdp.value_iteration()
        #mdp.save_policy(self.policy)
        mdp.load_policy(self.policy)
        self.value_iter_pi = mdp.pi

        value_iter_data =   np.zeros([self.TRIALS, self.ITER])
        classic_il_data =   np.zeros([self.TRIALS, self.ITER])
        classic_il_acc  =   np.zeros([self.TRIALS, self.ITER])
        classic_il_loss =   np.zeros([self.TRIALS, self.ITER])

        for t in range(self.TRIALS):
            print "Boosted IL Trial: " + str(t)
            print "***DO NOT TERMINATE THIS PROGRAM***"
            mdp.load_policy(self.policy)
            dt = DecisionTreeClassifier(max_depth=self.DEPTH)
            boost = AdaBoostClassifier(dt, n_estimators=50)
            value_iter_r, classic_il_r, acc, loss = self.supervise_trial(mdp, boost)
            
            value_iter_data[t,:] = value_iter_r            
            classic_il_data[t,:] = classic_il_r
            classic_il_acc[t,:] = acc
            classic_il_loss[t,:] = loss


        return classic_il_data, classic_il_acc, classic_il_loss
        

    def boosted_dagger(self):
        mdp = ClassicMDP(Policy(self.grid), self.grid)
        #mdp.value_iteration()
        #mdp.save_policy(self.policy)
        mdp.load_policy(self.policy)
        
        self.value_iter_pi = mdp.pi

        dagger_data = np.zeros((self.TRIALS, self.ITER))
        dagger_acc = np.zeros((self.TRIALS, self.ITER))
        dagger_loss = np.zeros((self.TRIALS, self.ITER))

        for t in range(self.TRIALS):
            print "Boosted DAgger Trial: " + str(t)
            print "***DO NOT TERMINATE THIS PROGRAM***"
            mdp.load_policy(self.policy)            
            dt = DecisionTreeClassifier(max_depth=self.DEPTH)
            boost = AdaBoostClassifier(dt, n_estimators=50)
            r, acc, loss = self.dagger_trial(mdp, boost)
            
            dagger_data[t,:] = r
            dagger_acc[t, :] = acc
            dagger_loss[t,:] = loss
    
        return dagger_data, dagger_acc, dagger_loss



    def run(self, LIMIT_DATA, DEPTH, MOVES, scen):
        self.LIMIT_DATA = LIMIT_DATA
        self.DEPTH = DEPTH
        self.moves = MOVES
        self.comparisons_directory, self.data_directory = self.make_dirs([LIMIT_DATA, DEPTH, MOVES], ['ld', 'd', 'm'])
        if not os.path.exists(self.comparisons_directory):
            os.makedirs(self.comparisons_directory)
        # else:
        #     return
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        # else:
        #     return 


        H = 15
        W = 15

        rewards = scen['rewards']
        sinks = scen['sinks']
        self.grid = LowVarInitStateGrid(15, 15, 15)
        self.grid.set_reward_states(rewards)
        self.grid.set_sink_states(sinks)
        self.policy = 'policies/overreal3d.p'
    
        value_iter_data, classic_il_data, classic_il_acc, classic_il_loss = self.vanilla_supervise()
        dagger_data, dagger_acc, dagger_loss = self.vanilla_dagger()
        #ada_data, ada_acc, ada_loss = self.boosted_supervise()
        #adadagger_data, adadagger_acc, adadagger_loss = self.boosted_dagger()

        
        np.save(self.data_directory + 'sup_data.npy', value_iter_data)
        np.save(self.data_directory + 'classic_il_data.npy', classic_il_data)
        np.save(self.data_directory + 'dagger_data.npy', dagger_data)
        #np.save(self.data_directory + 'ada_data.npy', ada_data)
        #np.save(self.data_directory + 'adadagger_data.npy', adadagger_data)
        
        np.save(self.data_directory + 'dagger_acc.npy', dagger_acc)
        np.save(self.data_directory + 'classic_il_acc.npy', classic_il_acc)
        #np.save(self.data_directory + 'ada_acc.npy', ada_acc)
        #np.save(self.data_directory + 'adadagger_acc.npy', adadagger_acc)    

        np.save(self.data_directory + 'dagger_loss.npy', dagger_loss)
        np.save(self.data_directory + 'classic_il_loss.npy', classic_il_loss)
        #np.save(self.data_directory + 'ada_loss.npy', ada_loss)    
        #np.save(self.data_directory + 'adadagger_loss.npy', adadagger_loss)    

        analysis = Analysis(H, W, self.ITER, rewards=rewards, sinks=sinks, desc="General comparison")
        analysis.get_perf(value_iter_data, 'b')
        analysis.get_perf(classic_il_data, 'g')
        analysis.get_perf(dagger_data, 'r')
        #analysis.get_perf(ada_data, 'c')
        #analysis.get_perf(adadagger_data, 'm')

        analysis.plot(names = ['Value iteration', 'Supervised', 'DAgger'], filename=self.comparisons_directory + 'reward_comparison.eps')#, ylims=[-60, 100])

        acc_analysis = Analysis(H, W, self.ITER, rewards = self.grid.reward_states, sinks=self.grid.sink_states, desc="Accuracy comparison")
        acc_analysis.get_perf(classic_il_acc, 'g')
        acc_analysis.get_perf(dagger_acc, 'r')
        #acc_analysis.get_perf(ada_acc, 'c')
        #acc_analysis.get_perf(adadagger_acc, 'm')

        acc_analysis.plot(names = ['Supervised Acc.', 'DAgger Acc.'], label='Accuracy', filename=self.comparisons_directory + 'acc_comparison.eps', ylims=[0,1])
        
        loss_analysis = Analysis(H, W, self.ITER, rewards=rewards, sinks=sinks, desc="Loss plot")
        loss_analysis.get_perf(classic_il_loss, 'g')
        loss_analysis.get_perf(dagger_loss, 'r')
        #loss_analysis.get_perf(ada_loss, 'c')
        #loss_analysis.get_perf(adadagger_loss, 'm')    

        loss_analysis.plot(names = ['Supervised loss', 'DAgger loss'], label='Loss', filename=self.comparisons_directory + 'loss_plot.eps', ylims=[0, 1])
            

        

        return
        


if __name__ == '__main__':

    # ITER = 25
    # TRIALS = 15
    # SAMP = 15

    ITER = 30
    TRIALS = 30
    SAMP = 15
    #ITER = 2
    #TRIALS = 1
    #SAMP = 2
    

    #test = RandomTest('random/random', 80, ITER, TRIALS, SAMP)

    # ld_set = [5]
    ld_set = [1]
    d_set = [4]
    steps = [50]

    params = list(itertools.product(ld_set, d_set, steps))



    for i in range(len(params)):
        for filename in sorted(os.listdir('scenarios/')):
            if filename.endswith('.p') and filename == 'scen1.p':
                scenario = random_scen.load('scenarios/' + filename)
                test = RandomTest('lvis_scen1/unreal/' + filename, 50, ITER, TRIALS, SAMP)
                print "Param " + str(i) + " of " + str(len(params))
                param = list(params[i])
                param.append(scenario)
                test.run(*param)


