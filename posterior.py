from prior import *
from likelihood import *

def tree_posterior(tree, alpha, d, D0, B, Q1, Pi, debug=False, fname = None):
  a = tree_prior(tree, alpha, d, D0, B, log=True, fname=fname)
  b = log_lik(tree, Q1, Pi)
  if a == 1: # impossible state
	  return 1
  return  a + b
