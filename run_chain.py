
import sys
sys.path.insert(0, '../thesis_likelihood')
import numpy as np
from treestruct import *
from nodestruct import *
from prior import *
from buildmtrx import *
from MCMCMoves import *
import random



random.seed(26111994)

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

N = 1500

debug = 0

t = 20
alpha = np.array([0.5,0.5]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)
s = "2A.1T.17C.8G.9G.9T.18A."
successes1, chain1a, chain1b, chain1c = run_chain(s, N, t, Q1, alpha, d, D0, B, Pi, by='df')

alpha = np.array([0.5,0.5]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)
s = "(0.15100139284528913 / 'G'(16.33881754412946 / 'A'(3.270634985083973 / 'T'(0.2395460779412808 / 'A')(0.2395460779412808 / 'G'))(3.510181063025252 / 'T'))(19.848998607154712 / 'C'))"
successes2, chain2a, chain2b, chain2c = run_chain(s, N, t, Q1, alpha, d, D0, B, Pi, by='io')

alpha = np.array([0.5,0.5]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)
s = "(2.0 / 'A'(8.0 / 'G'(10.0 / 'G')(10.0 / 'T'))(6.747867925981902 / 'T'(11.2521320740181 / 'C')(11.2521320740181 / 'A')))"
successes3, chain3a, chain3b, chain3c = run_chain(s, N, t, Q1, alpha, d, D0, B, Pi, by='io')

alpha = np.array([0.5,0.5]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.2, mu1 = 0.099, q01 = 0.2, q10 =0.001, lambda0 = 1, lambda1 = 1)
s = "(3.616051029646276 / 'A'(8.832241477452936 / 'T'(7.551707492900789 / 'C')(7.471726162738409 / 'G'(0.0799813301623793 / 'G')(0.0799813301623793 / 'T')))(16.383948970353725 / 'A'))"
successes4, chain4a, chain4b, chain4c = run_chain(s, N, t, Q1, alpha, d, D0, B, Pi, by='io')

print(len(chain1a))
print("acceptance rate", successes1/len(chain1a))
print("acceptance rate", successes2/len(chain2a))
print("acceptance rate", successes3/len(chain3a))
print("acceptance rate", successes4/len(chain4a))

import csv

chaina = [chain1a, chain2a, chain3a, chain4a]
chainb = [chain1b, chain2b, chain3b, chain4b]
chainc = [chain1c, chain2c, chain3c, chain4c]
for i in range(4): 
	with open(f"../thesis_likelihood\csv\c{i+1}a.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chaina[i])

	with open(f"../thesis_likelihood\csv\c{i+1}b.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainb[i])
		
	with open(f"../thesis_likelihood\csv\c{i+1}c.csv", 'w', newline = '') as csvfile:
		my_writer = csv.writer(csvfile, delimiter = 'Y')
		my_writer.writerow(chainc[i])

