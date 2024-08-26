import numpy as np
from multiprocessing import Pool
from itertools import repeat
from math import factorial
def G_bd(z,x, lambda0, mu0):
  c = lambda0-mu0
  num = np.exp(c*z)*(mu0-lambda0*np.exp(c*x))**2
  denom = (mu0-lambda0*np.exp(c*(z+x)))**2
  return num/denom

# handles the external branches likelihood calc
def ext_bd(t, lambda0, mu0):
  c = lambda0-mu0
  num = np.exp(c*t)*(c**2)
  denom = (mu0-lambda0*np.exp(c*t))**2
  return num/denom


def int_bd(cur, lambda0, mu0, adj=True):
  if (cur.isLeaf()):
    lik = cur.prior

  else:
    t_left = int_bd(cur.left, lambda0, mu0)
    t_right = int_bd(cur.right, lambda0, mu0)
    G_val = cur.gval
    if adj==False: G_val = 1
    prod = 2*t_left*t_right
    lik = G_val*lambda0*prod

  return lik

def bd_lik(tree, lambda0, mu0, log=True, adj = True):
  cur = tree.head
  parents = cur.find_parents()
  leaves = cur.find_leaves()

  for leaf in leaves:
    leaf.prior = ext_bd(leaf.time, lambda0, mu0)

 
  pool = Pool()
  results = pool.starmap(G_bd, zip([parent.time for parent in parents], [parent.dist_from_tip() for parent in parents], repeat(lambda0), repeat(mu0)))
  pool.close()
  for i in range(len(results)):
    parents[i].gval = results[i]

	  
  # doesnt work with fractional lengths?
  t_left = int_bd(cur.left, lambda0, mu0, adj)
  t_right = int_bd(cur.right, lambda0, mu0, adj)
  G_val = cur.gval
  if adj == False: G_val = 1
  alpha = 1
  prod = 2*t_left*t_right
  val = G_val*lambda0*prod
  if log:
    val = np.log(val)
  return val
  
def lin_bd_lik(lambda0, mu0, tree):
  cur = tree.head
  r = lambda0-mu0
  a = mu0/lambda0
  leaves = cur.find_leaves()
  n = len(leaves)
  adj = 0
  parents = cur.right.find_parents() + cur.left.find_parents()
  sum1 = 0
  prod1 = 1
  for node in parents +leaves:
    sum1 += node.dist_from_tip()
  for node in parents+leaves+[cur]:
    x = node.dist_from_tip()
    prod1 *= 1/(np.exp(r*x)-a)**2
  adj = factorial(n-1)*r**(n-2)*(1-a)**n*prod1*(np.exp(r*sum1))
  return np.log(adj)