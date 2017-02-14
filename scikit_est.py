
class SKEst():

    def __init__(self, grid, mdp, learner, learn_trajs=False):
        self.mdp = mdp
        self.grid = grid
        self.data = []
        self.trajs = []
        self.learner = learner
        self.failed_class = False
        self.learn_trajs = learn_trajs


    def add_datum(self, state, action):
        self.data.append((state, action))

    def add_traj(self, traj):
        """ 
            format of traj is [(state, action_taken, sup_intended_action, eps), ...]
        """
        self.trajs.append(traj)

    def fit(self):
        if self.learn_trajs:
            self.fit_trajs()
        else:
            self.fit_states()


    def fit_trajs(self):
        self.failed_class = False
        X = []
        Y = []
        for traj in self.trajs:
            for x_t, a_t, sup_a_t, eps in traj:
                X.append(list(x_t.pos))
                Y.append(sup_a_t)
        try:
            self.learner.fit(X, Y)
        except ValueError:
            self.failed_class = True

    def fit_states(self):
        self.failed_class = False
        X = []
        Y = []
        for state, action in self.data:
            X.append(list(state.pos))
            Y.append(action)
        try:
            self.learner.fit(X, Y)
        except ValueError:
            self.failed_class = True


    def traj_predictions():
        if not self.learn_trajs:
            raise Exception("Learner is not tracking trajectories");
        
        trajs_preds = []
        for traj in self.trajs:
            for x_t, a_t, sup_a_t, eps in traj:
                learn_a_t = self.predict()
                trajs_preds.append(x_t, a_t, sup_a_t, learn_a_t, eps)

        return trajs_preds




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
    

