import sys
sys.path.insert(0, '../thesis_likelihood')
import numpy as np
from treestruct import *
from nodestruct import *
from prior import *
from buildmtrx import *
from MCMCMoves import *
import random
import csv

# random.seed(26111994)

# we start a tree
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

N = 10000

debug = 0

t = 20
alpha = np.array([0.5,0.5]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)
s = "(0.15100139284528913 / 'G'(16.33881754412946 / 'A'(3.270634985083973 / 'T'(0.2395460779412808 / 'A')(0.2395460779412808 / 'G'))(3.510181063025252 / 'T'))(19.848998607154712 / 'C'))"
with open('log2.txt', "a") as f:
	successes, chaina, chainb, chainc = run_chain(s, N, t, Q1, alpha, d, D0, B, Pi, by='io', fname=f)

print("acceptance rate", successes/len(chaina))

i = 1
with open(f"../thesis_likelihood\csv\c{i+1}a.csv", 'w', newline = '') as csvfile:
	my_writer = csv.writer(csvfile, delimiter = 'Y')
	my_writer.writerow(chaina)

with open(f"../thesis_likelihood\csv\c{i+1}b.csv", 'w', newline = '') as csvfile:
	my_writer = csv.writer(csvfile, delimiter = 'Y')
	my_writer.writerow(chainb)
	
with open(f"../thesis_likelihood\csv\c{i+1}c.csv", 'w', newline = '') as csvfile:
	my_writer = csv.writer(csvfile, delimiter = 'Y')
	my_writer.writerow(chainc)

