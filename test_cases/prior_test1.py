import sys
sys.path.insert(0, '../thesis_likelihood')
import numpy as np
from treestruct import *
from nodestruct import *
from prior import *
from buildmtrx import *

alpha = np.array([0.5,0.5]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)

t2 = Tree(1)
t = 20
mystr = "(0.1 / 'T'(0.2 / 'G'(1 / 'A')(0.5 / 'T'(0.5 / 'T')(0.5 / 'T')))(1.1 / 'C'(0.1 / 'A')(0.1 / 'A')))"
t2.str2tree(mystr, t, by='io')
t2.disp()
lik = tree_prior(t2, alpha, d, D0, B)
print(lik)