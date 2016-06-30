import numpy as np
# all states s' with hamming_dist(s, s') < 2 are adjacent

class State():
    NONE = 0
    def __init__(self, *args):
        self.pos = args
        self.available_actions = range(len(self.pos) * 2 + 1)


    def toString(self):
        s = "(" + str(self.pos[0])
        for i in range(1, len(self.pos)):
            s += "," + str(self.pos[i])
        s += ")"
        return s

    def add(self, state):
        lst = list(self.pos)
        for i in range(len(lst)):
            lst[i] = lst[i] + state.pos[i]
        return State(*lst)

    def get_action(self, dif):
        lst = dif.pos
        if all(v == 0 for v in lst):
            return State.NONE
        else:
            lst = np.array(lst)
            index = np.argmax(np.abs(lst))            
            lst = (lst + 1) / 2
            action = index * 2 + lst[index] + 1
            return action

    def difference(self, action):
        """
            Compute difference state based on given action
            which should be in >= 0 and <= len(pos) * 2
        """
        if action == State.NONE:
            return State(*([0] * len(self.pos)))
        action = action - 1
        index = int(action / 2)
        sign = (action % 2) * 2 - 1
        lst = [0] * len(self.pos)
        lst[index] = sign
        return State(*lst)

    def subtract(self, state):
        lst = list(self.pos)
        for i in range(len(lst)):
            lst[i] = lst[i] - state.pos[i]
        return State(*lst)

    def __repr__(self):
        return self.toString()

    def __str__(self):
        return self.toString()

    def equals(self, other):
        lst = [i == j for i, j in zip(other.pos, self.pos)]
        return all(lst)

    def __eq__(self, other):
        return self.equals(other)

    def toArray(self):
        return list(self.pos)



