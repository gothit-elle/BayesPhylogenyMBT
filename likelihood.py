from scipy.linalg import fractional_matrix_power, expm
import numpy as np
BASES = ["A","C", "G", "T"]

# if its a tip, have a 0 for all s except actaully observed
def base2int(base):
  return [int(base==char) for char in BASES]

def prob(Q, sk, si, t): #sk, si, vi):
  # prob that a lineage in state k will be in state i after vi units of time have elapsed
  # can speed up using ep 7 in felenstein 1981
  #result = sk@fractional_matrix_power(P, vi)@si
  result = sk@expm(Q*t)@si
  return result

def cond_likelihood(tree, P, index, Pi, debug=False):
  # calculates the likelihood of a tree given data (embedded in the struct)

  cur=tree

  if cur.isLeaf(): # if leaf return obs
    cur.lik = cur.base2int(index)
  else:
    # otherwise we want to sum across the leaves
    if (cur.right is not None):
      cur.right.lik = cond_likelihood(cur.right, P, index, debug)
    if (cur.left is not None):
      cur.left.lik = cond_likelihood(cur.left, P, index, debug)
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
  if debug: print(cur.seq, cur.time, cur.lik)
  return cur.lik


def log_lik(new_tree, P, Pi, debug=False):
  # returns the loglik of the tree
  new_tree.lik = 0
  seq_len = new_tree.seq_len
  for k in range(seq_len):
    new_tree.lik += np.log(sum([i*j for (i,j) in zip(Pi,cond_likelihood(new_tree.head, P, k, Pi, debug))]))
  return new_tree.lik
