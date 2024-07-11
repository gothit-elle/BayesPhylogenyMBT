from prior import *
from likelihood import *

def tree_posterior(tree, alpha, d, D0, B, Q1, Pi, debug=False):
  a = tree_prior(tree, alpha, d, D0, B)
  b = log_lik(tree, Q1, Pi)
  if debug: print("p: ", a, "L:", b)
  return  a + b
