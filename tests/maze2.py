from base_test import BaseTest
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import os
from analysis import Analysis
import plot_class
import scenarios
from gridworld import Grid
from policy import Policy
import numpy as np
from mdp import ClassicMDP

class TestTest(BaseTest):

    def vanilla_supervise(self):
        mdp = ClassicMDP(Policy(self.grid), self.grid)
        mdp.value_iteration()
        mdp.save_policy(self.policy)
        mdp.load_policy(self.policy)
        self.value_iter_pi = mdp.pi
        self.plotter.plot_state_actions(self.value_iter_pi, rewards = self.grid.reward_states, sinks = self.grid.sink_states,
                filename=self.comparisons_directory + 'value_iter_state_action.eps')

        value_iter_data =   np.zeros([self.TRIALS, self.ITER])
        classic_il_data =   np.zeros([self.TRIALS, self.ITER])
        classic_il_acc  =    np.zeros([self.TRIALS, self.ITER])
        classic_il_loss =   np.zeros([self.TRIALS, self.ITER])

        for t in range(self.TRIALS):
            print "IL Trial: " + str(t)
            mdp.load_policy(self.policy)
            dt = DecisionTreeClassifier(max_depth=self.DEPTH)
            value_iter_r, classic_il_r, acc, loss = self.supervise_trial(mdp, dt)
            
            value_iter_data[t,:] = value_iter_r            
            classic_il_data[t,:] = classic_il_r
            classic_il_acc[t,:] = acc
            classic_il_loss[t,:] = loss

            if t == 0:
                self.plotter.plot_state_actions(mdp.pi, rewards=self.grid.reward_states, sinks=self.grid.sink_states,
                        filename=self.comparisons_directory + 'classic_il_state_action.eps')

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
            mdp.load_policy(self.policy)            
            dt = DecisionTreeClassifier(max_depth=self.DEPTH)
            r, acc, loss = self.dagger_trial(mdp, dt)
            
            dagger_data[t,:] = r
            dagger_acc[t, :] = acc
            dagger_loss[t,:] = loss
    
            if t == 0:
                self.plotter.plot_state_actions(mdp.pi, rewards=self.grid.reward_states, sinks=self.grid.sink_states,
                        filename=self.comparisons_directory + 'dagger_state_action.eps')
    
        return dagger_data, dagger_acc, dagger_loss

    def boosted_supervise(self):
        mdp = ClassicMDP(Policy(self.grid), self.grid)
        #mdp.value_iteration()
        #mdp.save_policy(self.policy)
        mdp.load_policy(self.policy)
        self.value_iter_pi = mdp.pi
        self.plotter.plot_state_actions(self.value_iter_pi, rewards = self.grid.reward_states, sinks = self.grid.sink_states,
                filename=self.comparisons_directory + 'value_iter_state_action.eps')

        value_iter_data =   np.zeros([self.TRIALS, self.ITER])
        classic_il_data =   np.zeros([self.TRIALS, self.ITER])
        classic_il_acc  =    np.zeros([self.TRIALS, self.ITER])
        classic_il_loss =   np.zeros([self.TRIALS, self.ITER])

        for t in range(self.TRIALS):
            print "Boosted IL Trial: " + str(t)
            mdp.load_policy(self.policy)
            dt = DecisionTreeClassifier(max_depth=self.DEPTH)
            boost = AdaBoostClassifier(dt, n_estimators=5)
            value_iter_r, classic_il_r, acc, loss = self.supervise_trial(mdp, boost)
            
            value_iter_data[t,:] = value_iter_r            
            classic_il_data[t,:] = classic_il_r
            classic_il_acc[t,:] = acc
            classic_il_loss[t,:] = loss

            if t == 0:
                self.plotter.plot_state_actions(mdp.pi, rewards=self.grid.reward_states, sinks=self.grid.sink_states,
                        filename=self.comparisons_directory + 'adaboost_state_action.eps')

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
            mdp.load_policy(self.policy)            
            dt = DecisionTreeClassifier(max_depth=self.DEPTH)
            boost = AdaBoostClassifier(dt, n_estimators=5)
            r, acc, loss = self.dagger_trial(mdp, boost)
            
            dagger_data[t,:] = r
            dagger_acc[t, :] = acc
            dagger_loss[t,:] = loss
            if t == 0:
                self.plotter.plot_state_actions(mdp.pi, rewards=self.grid.reward_states, sinks=self.grid.sink_states,
                        filename=self.comparisons_directory + 'adadagger_state_action.eps')
    
        return dagger_data, dagger_acc, dagger_loss



    def run(self, LIMIT_DATA, DEPTH):
        self.LIMIT_DATA = LIMIT_DATA
        self.DEPTH = DEPTH
        self.comparisons_directory, self.data_directory = self.make_dirs([LIMIT_DATA, DEPTH], ['ld', 'd'])
        if not os.path.exists(self.comparisons_directory):
            os.makedirs(self.comparisons_directory)
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        H = 15
        W = 15

        self.plotter = plot_class.Plotter()
        scen = scenarios.maze2

        rewards = scen['rewards']
        sinks = scen['sinks']
        self.grid = Grid(15, 15)
        self.grid.reward_states = rewards
        self.grid.sink_states = sinks
        self.policy = 'policies/maze2.p'
    
        value_iter_data, classic_il_data, classic_il_acc, classic_il_loss = self.vanilla_supervise()
        dagger_data, dagger_acc, dagger_loss = self.vanilla_dagger()
        ada_data, ada_acc, ada_loss = self.boosted_supervise()
        adadagger_data, adadagger_acc, adadagger_loss = self.boosted_dagger()

        
        np.save(self.data_directory + 'sup_data.npy', value_iter_data)
        np.save(self.data_directory + 'classic_il_data.npy', classic_il_data)
        np.save(self.data_directory + 'dagger_data.npy', dagger_data)
        np.save(self.data_directory + 'ada_data.npy', ada_data)
        np.save(self.data_directory + 'adadagger_data.npy', adadagger_data)
        
        np.save(self.data_directory + 'dagger_acc.npy', dagger_acc)
        np.save(self.data_directory + 'classic_il_acc.npy', classic_il_acc)
        np.save(self.data_directory + 'ada_acc.npy', ada_acc)
        np.save(self.data_directory + 'adadagger_acc.npy', adadagger_acc)    

        np.save(self.data_directory + 'dagger_loss.npy', dagger_loss)
        np.save(self.data_directory + 'classic_il_loss.npy', classic_il_loss)
        np.save(self.data_directory + 'ada_loss.npy', ada_loss)    
        np.save(self.data_directory + 'adadagger_loss.npy', adadagger_loss)    

        analysis = Analysis(H, W, self.ITER, rewards=rewards, sinks=sinks, desc="General comparison")
        analysis.get_perf(value_iter_data)
        analysis.get_perf(classic_il_data)
        analysis.get_perf(dagger_data)
        analysis.get_perf(ada_data)
        analysis.get_perf(adadagger_data)

        analysis.plot(names = ['Value iteration', 'DT IL', 'DT DAgger', 'Adaboost IL', 'Adaboost DAgger'], filename=self.comparisons_directory + 'reward_comparison.eps', ylims=[-60, 100])

        acc_analysis = Analysis(H, W, self.ITER, rewards = self.grid.reward_states, sinks=self.grid.sink_states, desc="Accuracy comparison")
        acc_analysis.get_perf(classic_il_acc)
        acc_analysis.get_perf(dagger_acc)
        acc_analysis.get_perf(ada_acc)
        acc_analysis.get_perf(adadagger_acc)

        acc_analysis.plot(names = ['DT IL Acc.', 'DT DAgger Acc.', 'Adaboost Acc.', 'Adaboost DAgger Acc.'], label='Accuracy', filename=self.comparisons_directory + 'acc_comparison.eps', ylims=[0,1])
        loss_analysis = Analysis(H, W, self.ITER, rewards=rewards, sinks=sinks, desc="Loss plot")
        loss_analysis.get_perf(classic_il_loss)
        loss_analysis.get_perf(dagger_loss)
        loss_analysis.get_perf(ada_loss)
        loss_analysis.get_perf(adadagger_loss)    

        loss_analysis.plot(names = ['DT IL loss', 'DAgger loss', 'Adaboost loss', 'Adaboost DAgger loss'], label='Loss', filename=self.comparisons_directory + 'loss_plot.eps', ylims=[0, 1])
            

        

        return
        


if __name__ == '__main__':


    #ITER = 25
    #TRIALS = 30
    #SAMP = 30
    ITER = 3
    TRIALS = 2
    SAMP = 3
    


    test = TestTest('maze2', 80, ITER, TRIALS, SAMP)

    ld_set = [1]
    d_set = [3]
    params = list(itertools.product(ld_set, d_set))

    for i in range(len(params)):
        print "Param " + str(i) + " of " + str(len(params))
        param = params[i]
        test.run(*param)
