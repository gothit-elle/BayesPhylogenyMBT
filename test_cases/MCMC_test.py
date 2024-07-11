
import sys
sys.path.insert(0, '../thesis_likelihood')
import numpy as np
from treestruct import *
from nodestruct import *
from prior import *
from buildmtrx import *
from MCMCMoves import *

alpha = np.array([0.5,0.5]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)

nodeStr = "2A.1B.17C.8D.9E.9F.18G."
t=20
t2 = Tree(1)
t2.str2tree(nodeStr,t,by='df')
t2.disp()

debug = 0
move_type=3
step = 1

move = propose_move(t2, alpha, d, D0, B, step, move_type, debug)
if move != EXIT_FAILURE:
  t2p, Q, alpha, d, D0, B = move

d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)


print(Q)
t2p.disp() 