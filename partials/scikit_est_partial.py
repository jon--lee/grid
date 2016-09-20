import numpy as np
import IPython
class SKEstPartial():

    def __init__(self, grid, mdp, learner):
        self.mdp = mdp
        self.grid = grid
        self.data = []
        self.learner = learner
        self.COV = 0.0

    def noisy(self, s):
        n = len(s.pos)
        rand = np.random.multivariate_normal([0] * n, np.identity(n) * self.COV)
        pos = tuple(s.pos + rand)
        return State(*pos)
        
        
    def add_datum(self, state, action):
        self.data.append((state, action))

    def fit(self):
        X = []
        Y = []
        for state, action in self.data:
            X.append(list(state.pos))
            Y.append(action)
        self.learner.fit(X, Y)


    def predict(self, s):
        return self.learner.predict(s)

    def get_states(self):
        N = len(self.data)
        states = np.zeros([N,self.grid.dim])
        for i in range(N):
            x = self.data[i][0].toArray()
            states[i,:] = x
        return states


    def clear_data(self):
        self.data = []


    def acc(self):
        X = []
        Y = []
        for state, action in self.data:
            X.append(list(state.pos))
            Y.append(action)
        return self.learner.score(X, Y)
        """results = []
        for s, a in self.data:
            pred = self.predict([list(s.pos)])[0]
            results.append(pred == a)
        return float(sum(results)) / float(len(self.data))"""
            
    

