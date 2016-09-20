from partials.scikit_est_partial import SKEstPartial
from policy import SKPolicy
from state import State
import numpy as np
import IPython

class ScikitDaggerPartial():
    def __init__(self, grid, mdp, super_pi, classifier=None, moves=40, super_pi_actual=None):
        self.grid = grid
        self.mdp = mdp
        self.super_pi = super_pi
        self.learner = SKEstPartial(grid, mdp, classifier)
        self.moves = moves
        self.reward = np.zeros(self.moves)
        self.record = False
        self.recent_rollout_states = []
        if super_pi_actual is None:
            super_pi_actual = super_pi
        self.super_pi_actual = super_pi_actual
        self.COV = 0.5

    def noisy(self, s):
        n = len(s.pos)
        rand = np.random.multivariate_normal([0] * n, np.identity(n) * self.COV)
        pos = tuple(s.pos + rand)
        return State(*pos)    
        

    def rollout(self):
        self.grid.reset_mdp()
        self.recent_rollout_states = [self.mdp.state]
        self.reward = np.zeros(self.moves)
        self.mistakes = 0
        for t in range(self.moves):
            obs = self.noisy(self.mdp.state)
            x_t = self.mdp.state
            if self.record:
                self.learner.add_datum(obs, self.super_pi.get_next(self.mdp.state))
            
            
            if self.mdp.pi.__class__.__name__ == 'Policy':
                self.compare_policies(x_t, x_t)
                a_t = self.grid.step_partial(self.mdp, x_t)
            else:
                self.compare_policies(x_t, obs)
                a_t = self.grid.step_partial(self.mdp, obs)
            
            x_t_1 = self.mdp.state

            self.reward[t] = self.grid.reward(x_t, a_t, x_t_1)
            self.recent_rollout_states.append(self.mdp.state)

    def compare_policies(self, x, obs):
        sup_a = self.super_pi_actual.get_next(x)
        act_a = self.mdp.pi.get_next(obs)
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
    
