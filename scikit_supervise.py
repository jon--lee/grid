from scikit_est import SKEst
from policy import SKPolicy
from state import State
from Optimize_Eps import INIT_OPT
import numpy as np
import IPython


class ScikitSupervise():

    def __init__(self, grid, mdp, super_pi, classifier=None, moves=40, super_pi_actual=None):
        self.grid = grid
        self.mdp = mdp
        self.super_pi = super_pi
        self.learner = SKEst(grid, mdp, classifier, learn_trajs=True)
        self.moves = moves
        self.reward = np.zeros(self.moves)
        self.record = False
        self.recent_rollout_states = []
        self.comp_eps_traj = []
        if super_pi_actual is None:
            super_pi_actual = super_pi
        self.super_pi_actual = super_pi_actual

    def rollout(self):
        # print "\t\tsuper_pi: " + self.super_pi.__class__.__name__ + ", mdp.pi: "  + self.mdp.pi.__class__.__name__ + \
        #             "super_pi_actual: " + self.super_pi_actual.__class__.__name__

        self.grid.reset_mdp()
        self.recent_rollout_states = [self.mdp.state]
        self.reward = np.zeros(self.moves)
        self.mistakes = 0
        traj = []
        for t in range(self.moves):
            
            x_t = self.mdp.state
            self.compare_policies(x_t)

            a_t = self.grid.step(self.mdp)
            if self.record:
                actual_action = self.super_pi.get_actual_next(x_t)
                traj.append((x_t, a_t, actual_action, self.super_pi.EPS))
                self.learner.add_datum(x_t, actual_action)

            x_t_1 = self.mdp.state

            self.reward[t] = self.grid.reward(x_t, a_t, x_t_1)
            self.recent_rollout_states.append(self.mdp.state)
    
        if self.record:
            self.learner.add_traj(traj)


    def rollout_sup(self):
        """
            Take actions according to the supervisor (whether it is noisy or not)
            Run analytics to get the loss of the learner on the supervisor's distr
        """
        self.grid.reset_mdp()
        self.sup_mistakes = 0
        traj = []
        for t in range(self.moves):
            if self.record:
                raise Exception("Should not be collecting data on test rollout")
            x_t = self.mdp.state
            self.compare_sup_policies(x_t)

            #sample
            tmp_pi = self.mdp.pi
            self.mdp.pi = self.super_pi
            a_t = self.grid.step(self.mdp)
            self.mdp.pi = tmp_pi

            x_t_1 = self.mdp.state

    def get_current_test(self):
        """
            Take actions according to the supervisor (whether it is noisy or not)
            Run analytics to get the loss of the learner on the supervisor's distr
        """
        self.grid.reset_mdp()
        self.sup_mistakes = 0
        traj = []
        
        for t in range(self.moves):
            if self.record:
                raise Exception("Should not be collecting data on test rollout")
            
            #sample
            x_t = self.mdp.state

            tmp_pi = self.mdp.pi
            self.mdp.pi = self.super_pi
            a_t = self.grid.step(self.mdp)
            self.mdp.pi = tmp_pi

            actual_action = self.super_pi.get_actual_next(x_t)
            r_a = self.learner.predict([list(x_t.pos)])
            traj.append((x_t, a_t, actual_action,r_a, self.super_pi.EPS))

            

            x_t_1 = self.mdp.state
        
        self.comp_eps_traj.append(traj)


    def compute_epsilon(self):
        int_opt = INIT_OPT(70,5)
        eps = int_opt.grid_search_eps(self.comp_eps_traj)
        self.comp_eps_traj = []

        self.super_pi.EPS = eps
        

    def compare_policies(self, x):
        sup_a = self.super_pi.get_actual_next(x)
        act_a = self.mdp.pi.get_actual_next(x)
        if sup_a != act_a:
            self.mistakes += 1

    def compare_sup_policies(self, x):
        sup_a = self.super_pi.get_actual_next(x)
        act_a = self.mdp.pi.get_actual_next(x)
        if sup_a != act_a:
            self.sup_mistakes += 1


    def get_loss(self):
        return float(self.mistakes) / float(self.moves)

    def get_sup_loss(self):
        return float(self.sup_mistakes) / float(self.moves)

    def train(self):
        self.learner.fit()
        self.mdp.pi = SKPolicy(self.learner)
        self.record = False

    def get_reward(self):
        return np.sum(self.reward)

    def get_states(self):
        return self.learner.get_states()

    def get_recent_rollout_states(self):
        N = len(self.recent_rollout_states)
        m = len(self.recent_rollout_states[0].pos)
        states = np.zeros([N, m])
        for i in range(N):
            x = self.recent_rollout_states[i].toArray()
            states[i, :] = x
        return states

