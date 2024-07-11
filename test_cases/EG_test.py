import sys
sys.path.insert(0, '../thesis_likelihood')

from prior import *
from buildmtrx import *
import numpy as np

toler = 1e-7
plot=0
alpha = np.array([0.5,0.5]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)

E = get_E(5, alpha, d, D0, B)
# should get E(5) = [0.2083, 0.3336]
print(E)

debug = 0
if debug:
  print("G(b1,x1) is ", G_bkxk(2,18, alpha, d, D0, B))
  print("\nG(b2,x2) is ", G_bkxk(1,17, alpha, d, D0, B))
  print("\nG(b2,x3) is ", G_bkxk(8,9, alpha, d, D0, B))

plot = 0
if plot:
  scale=20
  alpha = np.array([0.5,0.5]).astype(object)
  d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)
  a = G_bkxk(scale, 0, alpha, d, D0, B, plot)
  print([a[0,0] + a[0,1], a[1,0] + a[1,1]])
