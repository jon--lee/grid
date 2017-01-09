
class SKEst():

    def __init__(self, grid, mdp, learner):
        self.mdp = mdp
        self.grid = grid
        self.data = []
        self.learner = learner
        self.failed_class = False


    def add_datum(self, state, action):
        self.data.append((state, action))



    def fit(self):
        X = []
        Y = []
        for state, action in self.data:
            X.append(list(state.pos))
            Y.append(action)
        try:
            self.learner.fit(X, Y)
        except ValueError:
            self.failed_class = True


    def predict(self, a):
        if self.failed_class:
            return self.data[0][1]
        return self.learner.predict(a)[0]

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
        results = []
        for s, a in self.data:
            pred = self.predict([list(s.pos)])
            results.append(pred == a)
        return float(sum(results)) / float(len(self.data))
    

