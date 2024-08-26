import sys
sys.path.insert(0, '../thesis_likelihood')
import numpy as np
from treestruct import *
from nodestruct import *
from prior import *
from buildmtrx import *

alpha = np.array([0.5,0.5]).astype(object)
lambda_a = [1, 0, 0, 0, 0,0,0,0.099]
d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda_a=lambda_a)

block = (np.linalg.inv(-D0.astype(np.float64))@B@np.transpose(np.kron(np.ones(2), np.identity(2)) + np.kron(np.identity(2), np.ones(2)))).astype(np.float64)
evals, evecs = np.linalg.eigh(block)
print(evals) # lambda < 1 so E is maximum of 1
print(np.linalg.det(block))
print(np.linalg.inv(block))