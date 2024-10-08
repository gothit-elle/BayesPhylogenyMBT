from scipy.linalg import fractional_matrix_power, expm
from multiprocessing import Pool, freeze_support, cpu_count
from itertools import repeat
import numpy as np
from treestruct import *

BASES = ["A","C", "G", "T"]

# if its a tip, have a 0 for all s except actaully observed
def base2int(base):
  return [int(base==char) for char in BASES]

def prob(mtrx, sk, si): #sk, si, vi):
  # prob that a lineage in state k will be in state i after vi units of time have elapsed
  # can speed up using ep 7 in felenstein 1981
  #result = sk@fractional_matrix_power(P, vi)@si
  result = sk@mtrx@si
  return result

def cond_likelihood(cur, P, index, Pi, debug=False):
  # calculates the likelihood of a tree given data (embedded in the struct)

  if cur.isLeaf(): # if leaf return obs
    if debug: print("leaf at", cur.seq, index)
    cur.lik = cur.base2int(index)
    if cur.lik == -1:
      cur.lik = Pi
  else:
    # otherwise we want to sum across the leaves
    cur.right.lik = cond_likelihood(cur.right, P, index, Pi, debug)
    cur.left.lik = cond_likelihood(cur.left, P, index, Pi, debug)
    # we have the likelihoods.
    # pick a base:
    L = []
    for sk in BASES:
      left = 0
      right = 0
      for j in range(len(BASES)):
          si = BASES[j]
          left += prob(P, base2int(sk), base2int(si), cur.right.time)*cur.right.lik[j]
          right += prob(P, base2int(sk), base2int(si), cur.left.time)*cur.left.lik[j]
      L.append(left*right)
    cur.lik = L
  if debug: print("printing...", cur.seq, cur.time, cur.lik)
  return cur.lik

def cond_likelihood2(tree, P, index, Pi, debug=False):
  leaves, parents = find_levels(tree)
  for leaf in leaves:
    leaf.lik = leaf.base2int(index)
    if leaf.lik == -1:
      leaf.lik = Pi
	  
  for parent in parents:
    #print(parent.map, parent.Qt, flush=True)
    # we have the likelihoods.
    # pick a base:
    L = []
    for sk in BASES:
      left = 0
      right = 0
      for j in range(len(BASES)):
          si = BASES[j]
          left += prob(parent.right.Qt, base2int(sk), base2int(si))*parent.right.lik[j]
          right += prob(parent.left.Qt, base2int(sk), base2int(si))*parent.left.lik[j]
      L.append(left*right)
    parent.lik = L
  if debug: print("printing...", tree.head.seq, tree.head.time, tree.head.lik)
  return tree.head.lik


def sub_lik(k, new_tree, P, Pi, debug=False):
  res = 0
  for w in range(k,k+50):
    #if w != k:
    #  new_tree.head.mark_c_sites(w)
    res += np.log(sum([i*j for (i,j) in zip(Pi,  cond_likelihood2(new_tree, P, w, Pi, False))]))
  return res
 

def log_lik(new_tree, P, Pi, debug=False, multip=True):
  # returns the loglik of the tree
  new_tree.lik = 0
  seq_len = new_tree.seq_len
  if debug: print('seq len is', seq_len)
  if multip:
    if __name__ == 'likelihood':
      pool = Pool()
      results = pool.starmap(sub_lik, zip(range(0,seq_len,50), repeat(new_tree), repeat(P), repeat(Pi)))
      pool.close()
      new_tree.lik = sum(results)
  else: 

    for k in range(seq_len):
      new_tree.lik += sub_lik(k, new_tree, P, Pi, debug)
  
  return new_tree.lik
