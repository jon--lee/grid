from state import State

def notch(i):
    states3D = []
    states = (vertical(3, 9, 5) + vertical(3, 9, 6) + vertical(3, 9, 7) + vertical(3, 9, 9)
        + vertical(3, 9, 10) + vertical(3, 9, 11) + vertical(3, 6, 8))
    for state in states:
        states3D.append(State(state.pos[0], state.pos[1], i))
    return states3D

def tower_const():
    states = []
    for i in range(15):
        states += notch(i)
    return states
 

def horizontal(n, m, y):
    states = [ State(i, y) for i in range(n, m) ]
    return states

def vertical(n, m, x):
    states = [State(x, i) for i in range(n, m) ]
    return states
        

def inverse(states):
    states = [(state.pos[0], state.pos[1]) for state in states]
    sinks = []
    for i in range(15):
        for j in range(15):
            if not (i, j) in states:
                sinks.append(State(i, j))
    return sinks

def horizontal4(n, m, y):
    states = [ State(8, i, y, 7) for i in range(n, m) ]
    return states

def vertical4(n, m, x):
    states = [State(8, x, i, 7) for i in range(n, m) ]
    return states
        

def inverse4(states):
    states = [(8, state.pos[1], state.pos[2], 7) for state in states]
    sinks = []
    for i in range(15):
        for j in range(15):
            for k in range(15):
                for l in range(15):
                    if not (i, j, k, l) in states:
                        sinks.append(State(i, j, k, l))
    return sinks


scenario0 = {
    'rewards': [State(7, 7)],
    'sinks': [State(4, 3)]
}

scenario1 = {    
    'rewards': [State(11, 0), State(13, 0), State(7, 7)],
    'sinks': [State(1, 1), State(1, 0), State(2, 0), State(2, 1), State(2, 2), State(3, 1),
    State(1, 2), State(3, 2), State(4, 1), State(4, 0), State(3, 0), State(2, 0), State(4, 3),
    State(4, 2), State(1,3), State(2, 3), State(3, 3), State(5, 0), State(6, 0), State(7, 0), State(8, 0), State(9, 0),
    State(10, 0)]
}

scenario2 = {
    'rewards': [State(14, 14)],
    'sinks': [State(12, 13), State(13, 12), State(13, 13), State(13, 11), State(11, 13),
    State(12, 11), State(11, 12), State(11, 11), State(7, 7), State(8, 8), State(7, 8), State(8,7),
    State(12, 10), State(10, 12), State(10, 11), State(11, 10)]
}


scenario3 = {
    'rewards': [State(9, 8, 7)],
    'sinks': [State(9, 2, 4), State(10, 9, 8), State(10, 10, 10), State(12, 3, 8)]
}

scenario_classic = { # extra state to block fork-in-road situation
    'rewards': [State(14, 14)],
    'sinks': [State(12, 13), State(13, 12), State(13, 13), State(13, 11), State(11, 13),
    State(12, 11), State(11, 12), State(11, 11), State(7, 7), State(8, 8), State(7, 8), State(8,7),
    State(12, 10), State(10, 12), State(10, 11), State(11, 10), State(14, 13)]
}


scenario4 = {
    'rewards': [State(14, 14, 14, 14, 14)],
    'sinks': [State(13, 13, 13, 13, 13), State(12, 13, 13, 13, 13), State(13, 12, 13, 13, 13),
        State(13, 13, 12, 13, 13), State(13, 13, 13, 12, 13), State(13, 13, 13, 13, 12),
        State(12, 12, 13, 13, 13), State(12, 13, 12, 13, 13), State(12, 13, 13, 12, 13), State(12, 13, 13, 13, 12),
        State(13, 12, 12, 13, 13), State(13, 12, 13, 12, 13), State(13, 12, 13, 13, 12), State(13, 13, 12, 12, 13),
        State(13, 13, 12, 13, 12), State(13, 13, 13, 12, 12),  # surrounding

        State(7, 7, 7, 7, 7), State(8, 8, 8, 8, 8), State(7, 8, 8, 8, 8), State(8, 7, 8, 8, 8),
        State(8, 8, 7, 8, 8), State(8, 8, 8, 7, 8), State(8, 8, 8, 8, 7),# middle
        
        State(8, 8, 4, 11, 7), State(7, 12, 8, 6, 14), State(4, 12, 9, 12, 3) #rand
        
        ]
    }
scenario5 = {
    'rewards':  [State(14, 14, 14, 14)],
    'sinks':    [State(13, 13, 13, 13), State(12, 13, 13, 13), State(13, 12, 13, 13), State(13, 13, 12, 13),
                State(13, 13, 13, 12), State(12, 12, 12, 12), State(12, 12, 13, 13), State(12, 13, 12, 13),
                State(12, 13, 13, 12), State(13, 12, 12, 13), State(13, 12, 13, 12), State(13, 13, 12, 12), # surrounding
                State(7, 7, 7, 7), State(8, 8, 8, 8), State(7, 8, 8, 8), State(8, 7, 8, 8), State(8, 8, 7, 8),
                State(8, 8, 8, 7)]  # middle
}


scenario6 = {
    'rewards':  [State(14, 14, 14, 14)],
    'sinks':    [
                State(14, 14, 13, 14), State(14, 13, 14, 14), State(13, 14, 14, 14),
                State(13, 13, 13, 13), State(12, 13, 13, 13), State(13, 12, 13, 13), State(13, 13, 12, 13),
                State(13, 13, 13, 12), State(12, 12, 12, 12), State(12, 12, 13, 13), State(12, 13, 12, 13),
                State(12, 13, 13, 12), State(13, 12, 12, 13), State(13, 12, 13, 12), State(13, 13, 12, 12), # surrounding
                State(7, 7, 7, 7), State(8, 8, 8, 8), State(7, 8, 8, 8), State(8, 7, 8, 8), State(8, 8, 7, 8),
                State(8, 8, 8, 7)]  # middle
}




maze2 = {
    'rewards': [ State(6, 12) ],
    'sinks': inverse(([State(0, 0), State(0, 1)] + horizontal(0, 3, 1) + 
        vertical(1, 5, 2) + horizontal(2, 12, 4) + vertical(4, 9, 11) + horizontal(9, 12, 8)
        + vertical(8, 13, 9) + horizontal(6, 10, 12)
        ))
}


maze4 = {
    'rewards': [ State(8, 6, 12, 7) ],
    'sinks': inverse4(([State(8, 0, 0, 7), State(8, 0, 1, 7)] + horizontal4(0, 3, 1) + 
        vertical4(1, 5, 2) + horizontal4(2, 12, 4) + vertical4(4, 9, 11) + horizontal4(9, 12, 8)
        + vertical4(8, 13, 9) + horizontal4(6, 10, 12)
        ))
}

   

tower = {
    'rewards': [ State(8, 6, 14) ],
    'sinks': tower_const()
}


