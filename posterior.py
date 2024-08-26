from prior import *
from likelihood import *
import time

def tree_posterior(tree, alpha, d, D0, B, Q1, Pi, debug=False, fname = None, multip=False):
  start = time.time()
  a = tree_prior(tree, alpha, d, D0, B, logit=True, fname=fname, multip=multip)
  end = time.time()
  print('prior time: ', end-start)
  start= time.time()
  b = log_lik(tree, Q1, Pi, debug=False, multip=multip)
  end = time.time()
  print('lik time: ', end-start)
  if a == 1: # impossible state
	  return [0.5, 0.5]
  return  [a, b]
