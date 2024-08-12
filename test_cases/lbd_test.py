
import sys
sys.path.insert(0, '../thesis_likelihood')
import numpy as np
from treestruct import *
from nodestruct import *
from prior import *
from buildmtrx import *
from MCMCMoves import *
from math import factorial

"""# Hiding Linear birth death model

calc from
Estimating a Binary Character's Effect on Speciation and Extinction WAYNE P. MADDISON
"""

lambda0 = 0.7 # birth rate
mu0 = 0.3 # death rate
nodeStr = "2A.1B.17C.8D.9E.9F.18G." # nodes and branch lengths
t=20 # obs time
t2 = Tree(1)
t2.str2tree(nodeStr,t,by='df')
t2.disp()

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
  adj = factorial(n-1)*r**(n-2)*np.exp(r*sum)*(1-a)**n*prod1
  return adj
lin_bd_lik(lambda0, mu0, t2)

lambda0 = 0.7 # birth rate
mu0 = 0.3 # death rate
nodeStr = "2A.1B.17C.8D.9E.9F.18G." # nodes and branch lengths
t=20 # obs time
t2 = Tree(1)
t2.str2tree(nodeStr,t,by='df')
t2.disp()

def G_bd(z,x):
  c = lambda0-mu0
  num = np.exp(c*z)*(mu0-lambda0*np.exp(c*x))**2
  denom = (mu0-lambda0*np.exp(c*(z+x)))**2
  return num/denom

# handles the external branches likelihood calc
def ext_bd(t):
  c = lambda0-mu0
  num = np.exp(c*t)*(c**2)
  denom = (mu0-lambda0*np.exp(c*t))**2
  return num/denom


def int_bd(cur, adj=True):
  if (cur.isLeaf()):
    lik = ext_bd(cur.time)

  else:
    t_left = int_bd(cur.left)
    t_right = int_bd(cur.right)
    G_val = G_bd(cur.time, cur.dist_from_tip())
    if adj==False: G_val = 1
    prod = 2*t_left*t_right
    lik = G_val*lambda0*prod

  return lik

def bd_lik(tree, log=True, adj = True):
  cur = tree.head
  # doesnt work with fractional lengths?
  t_left = int_bd(cur.left, adj)
  t_right = int_bd(cur.right, adj)
  G_val = np.array(G_bd(cur.time,cur.dist_from_tip()))
  if adj == False: G_val = 1
  alpha = 1
  prod = 2*t_left*t_right
  val = G_val*lambda0*prod
  if log:
    val = np.log(val)
  return val

p1 = ext_bd(9)*ext_bd(9)*ext_bd(17)*ext_bd(18)
#print(ext_bd(9))
#print(ext_bd(17))
#print(ext_bd(18))

p2 = G_bd(8,9)*G_bd(1,17)*G_bd(2,18)
#print(np.log(p1*p2*(2*lambda0)**3))

plot = 0
alpha = [0.5,0.5]
mu0 = mu1 = 0.3
lambda0 = lambda1 = 0.7
q10 = q01 = 0

d, D0, D1, B = build_mtrx(mu0, mu1, q01, q10, lambda0, lambda1)
print(bd_lik(t2))

lik = tree_prior(t2, alpha, d, D0, B, logit=True, fname = None, multip=False)
print(lik)



nodeStr = "2A.1B.17C.8D.9E.9F.18G." # nodes and branch lengths
t=20 # obs time
trees= []
trees.append(Tree(1))
trees[0].str2tree(nodeStr,t,by='df')
trees[0].disp()

mystr = "(4 / 'T'(2 / 'G'(5 / 'A')(5 / 'T'))(7 / 'C'))"
t2 = Tree(1)
t2.str2tree(mystr,4+7, by='io')
trees.append(t2)
t2.disp()

nodeStr = "2T.3G.5A.5T.7C.1A.1T" # nodes and branch lengths
t2 = Tree(1)
t2.str2tree(nodeStr,10,by='df')
trees.append(t2)
t2.disp()

nodeStr = "2T.5G.5A" # nodes and branch lengths
t2 = Tree(1)
t2.str2tree(nodeStr,7,by='df')
trees.append(t2)
t2.disp()

for tree in trees:
  a = bd_lik(tree)
  b = tree_prior(tree, alpha, d, D0, B)
  print("lin bd llik: ", a)
  print("MBT llik: ", b)
  #print("\tratio", a/b)
  #print("\tdifference", a-b)

nodeStr = "2A.1T.17C.8G.9G.9T.18A." # nodes and branch lengths
t=20 # obs time
trees= []
t2=Tree(1)
t2.str2tree(nodeStr,t,by='df')
t2.disp()

t_current = t2
t_new= q_ratio = -1
for i in range(20):
  move = propose_move(t_current,alpha, d, D0, B, i)

  if move != EXIT_FAILURE:
    t_new, q_ratio, alpha_n, d_n, D0_n, B_n = move
    trees.append(t_new)
print(len(trees))

l1 = []
l2 = []
l3=[]
for i in range(len(trees)):
  print(i)
  tree = trees[i]
  tree.disp()
  a = bd_lik(tree, True, True)
  c = bd_lik(tree, True, False)
  b = tree_prior(tree, alpha, d, D0, B)
  l1.append((i, a))
  l2.append((i,b))
  l3.append((i,c))

c = sorted(l1, key=lambda x: x[1])

f = sorted(l2, key=lambda x: x[1])

g = sorted(l3, key=lambda x: x[1])
print(c)
print(f)
print(g)


from ete3 import Tree as Tree_n


for i in range(len(trees)):
  s_ml = trees[i].head.to_newick()
  t = Tree_n(s_ml, format = 3)
  # t.convert_to_ultrametric()
  t.show()
# the order of conclusions are the same, my model works!!!
