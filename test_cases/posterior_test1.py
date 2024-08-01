
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from treestruct import *
from nodestruct import *
from posterior import *
from buildmtrx import *
from MCMCMoves import *

"""# Posterior calc"""

# relative rates matrix
R = np.array([0, 0.5, 2, 0.5,
              0.5, 0, 0.5, 2,
              2, 0.5, 0, 0.5,
              0.5, 2, 0.5, 0]).reshape(4,4)
# initial distribution
Pi = [0.295, 0.205, 0.205, 0.295]
# rate matrix
Q1 = R@np.diag(Pi)
# need to adjust so rows sum to 0
Q1 -= np.diag(Q1@np.ones(4))

alpha = np.array([0.5,0.5]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)


nodeStr = "2A.1T.17C.8G.9G.9T.18A." # nodes and branch lengths
t=20 # obs time
trees= []
trees.append(Tree(1))
trees[0].str2tree(nodeStr,t,by='df')
trees[0].disp()

mystr = "(4 / 'T'(2 / 'G'(5 / 'A')(5 / 'T'))(7 / 'C'))"
t2 = Tree(1)
t2.str2tree(mystr,4+7, by='io')
trees.append(t2)
t2.disp()

nodeStr = "2T.3G.5A.5T.7C.1A.1T" # nodes and branch lengths
t2 = Tree(1)
t2.str2tree(nodeStr,10,by='df')
trees.append(t2)
t2.disp()

nodeStr = "2T.5G.5A" # nodes and branch lengths
t2 = Tree(1)
t2.str2tree(nodeStr,7,by='df')
trees.append(t2)
t2.disp()



for tree in trees:
	print(tree_posterior(tree, alpha, d, D0, B, Q1, Pi))