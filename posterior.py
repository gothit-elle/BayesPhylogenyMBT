from prior import *
from likelihood import *


def tree_posterior(tree, alpha, d, D0, B, Q1, Pi, debug=False, fname = None, multip=False):

  a = tree_prior(tree, alpha, d, D0, B, log=True, fname=fname, multip=multip)

  b = log_lik(tree, Q1, Pi, debug=False, multip=multip)

  if a == 1: # impossible state
	  return 1
  return  a + b
