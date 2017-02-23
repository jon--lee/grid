import os
import numpy as np


parent_direc = 'comparisons/revisited/feb16/6'


def add_values(d, mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not val in d:
                d[val] = 1
            else:
                d[val] += 1
d = {}
for direc in os.listdir(parent_direc):
    if '_20ld_-3d_70m_0.0pb_data' in direc:
        print direc
        data = np.load(parent_direc + '/' + direc + '/eps_data.npy')
        print "\n\n"
        print data
        add_values(d, data)

s = 0
for key in sorted(d.keys()):
    s += d[key]
for key in sorted(d.keys()):
    d[key] = float(d[key]) / float(s)
for key in sorted(d.keys()):
    print "Eps:" + str(key), "Freq: " + str(d[key])
        
 
    
        
