from scikit_est import SKEst
from policy import SKPolicy
from state import State
import numpy as np

class ScikitDagger():
    def __init__(self, grid, mdp, super_pi, classifier=None, moves=40):
        self.grid = grid
        self.mdp = mdp
        self.super_pi = super_pi
        self.learner = SKEst(grid, mdp, classifier)
        self.moves = moves
        self.reward = np.zeros(self.moves)
        self.record = False
        self.recent_rollout_states = []

    def rollout(self):
        self.grid.reset_mdp()
        self.recent_rollout_states = [self.mdp.state]
        self.reward = np.zeros(self.moves)
        self.mistakes = 0
        for t in range(self.moves):
            if self.record:
                self.learner.add_datum(self.mdp.state, self.super_pi.get_next(self.mdp.state))

            x_t = self.mdp.state
            self.compare_policies(x_t)

            a_t = self.grid.step(self.mdp)
            x_t_1 = self.mdp.state

            self.reward[t] = self.grid.reward(x_t, a_t, x_t_1)
            self.recent_rollout_states.append(self.mdp.state)

    def compare_policies(self, x):
        sup_a = self.super_pi.get_next(x)
        act_a = self.mdp.pi.get_next(x)
        if sup_a != act_a:
            self.mistakes += 1
            
    def get_loss(self):
        return float(self.mistakes) / float(self.moves)

    def retrain(self):
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
    
