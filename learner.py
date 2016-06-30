from state import State

class Learner():
    def __init__(self, grid, mdp, moves=40):
        self.grid = grid
        self.mdp = mdp
        self.learner = None
        self.super_pi = mdp.pi
        self.reward = np.zeros(self.moves)
        self.animate = False
    

    def rollout(self):
        self.grid.reset_mdp()
        self.recent_rollout_states = [self.mdp.state]
        self.reward = np.zeros(self.moves)
        for t in range(self.moves):
            if self.record:
                self.learner.add_datum(self.mdp.state, self.super_pi.get_next(self.mdp.state))
                
            x_t = self.mdp.state
            a_t = self.mdp.pi.get_next(x_t)
            self.grid.step(self.mdp)
            x_t_1 = self.mdp.state

            self.reward[t] = self.grid.reward(x_t, a_t, x_t_1)
            self.recent_rollout_states.append(self.mdp.state)

        
