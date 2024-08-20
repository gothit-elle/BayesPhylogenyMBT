
import sys
sys.path.insert(0, '../thesis_likelihood')
import numpy as np
from treestruct import *
from nodestruct import *
from prior import *
from buildmtrx import *
from MCMCMoves import *
from math import factorial
from sim import *

"""# Hiding Linear birth death model

calc from
Estimating a Binary Character's Effect on Speciation and Extinction WAYNE P. MADDISON
"""

nodeStr = "2A.1B.17C.8D.9E.9F.18G." # nodes and branch lengths
#t=20 # obs time
#t2 = Tree(1)
#t2.str2tree(nodeStr,t,by='df')
#t2.disp()

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

#print('test1 is', lin_bd_lik(lambda0, mu0, t2))
"""
lambda0 = 0.7 # birth rate
mu0 = 0.3 # death rate
nodeStr = "2A.1B.17C.8D.9E.9F.18G." # nodes and branch lengths
t=20 # obs time
t2 = Tree(1)
t2.str2tree(nodeStr,t,by='df')
t2.disp()
"""
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

#p1 = ext_bd(9)*ext_bd(9)*ext_bd(17)*ext_bd(18)
#print(ext_bd(9))
#print(ext_bd(17))
#print(ext_bd(18))

#p2 = G_bd(8,9)*G_bd(1,17)*G_bd(2,18)
#print(np.log(p1*p2*(2*lambda0)**3))



"""
#print(bd_lik(t2))

#lik = tree_prior(t2, alpha, d, D0, B, logit=True, fname = None, multip=False)
#print(lik)
"""

mu0 = mu1 = 0.1
lambda0 = lambda1 = 0.7
alpha = np.array([0.5,0.5]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.1, q10 =0.1, lambda0 = 0.7, lambda1 = 0.7)

nodeStr = "2A.1B.17C.8D.9E.9F.18G." # nodes and branch lengths
t=20 # obs time
trees= []
trees.append(Tree(1))
trees[0].str2tree(nodeStr,t,by='df')
trees[0].disp()

mystr = "(0.5253663846117584 / 'B'(19.474633615388242 / 'C')(4.2029310768940675 / 'D'(5.81510761548252 / 'A'(9.456594923011655 / 'E')(9.456594923011652 / 'G'))(15.271702538494171 / 'F')))"
t2 = Tree(1)
t2.str2tree(mystr,20, by='io')
trees.append(t2)
t2.disp()

nodeStr = "(2.0 / 'A'(1.0 / 'B'(17.0 / 'C')(17.0 / 'G'))(8.0 / 'D'(10.0 / 'E')(10.0 / 'F')))" # nodes and branch lengths
t2 = Tree(1)
t2.str2tree(nodeStr,20,by='io')
trees.append(t2)
t2.disp()

nodeStr = "(2.0 / 'A'(1.0 / 'B'(17.0 / 'C')(2.2648043922303245 / 'D'(14.735195607769676 / 'E')(14.735195607769676 / 'F')))(18.0 / 'G'))" # nodes and branch lengths
t2 = Tree(1)
t2.str2tree(nodeStr,20,by='io')
trees.append(t2)
t2.disp()



		


l1 = []
l2 = []
l3 = []
for i in range(len(trees)):
  tree = trees[i]
  a = bd_lik(tree)
  b = tree_prior(tree, alpha, d, D0, B, True, None, False)
  c = lin_bd_lik(lambda0, mu0, tree)
  l1.append((i, a))
  l2.append((i,c))
  l3.append((i,b))
  print("lin bd llik: ", a)
  print("Nee94 llik: ", c)
  print("MBT llik: ", b)
  print("\n")
  #print("\tratio", a/b)
  #print("\tdifference", a-b)

c = sorted(l1, key=lambda x: x[1])

f = sorted(l2, key=lambda x: x[1])

g = sorted(l3, key=lambda x: x[1])
print('bd T T', c)
print('nee94', f)
print('mbt ', g)

from ete3 import Tree as Tree_n


for i in range(len(trees)):
  s_ml = trees[i].head.to_newick()
  t = Tree_n(s_ml, format = 3)
  # t.convert_to_ultrametric()
  t.show()
# the order of conclusions are the same, my model works!!!

#nodeStr = "2A.1T.17C.8G.9G.9T.18A." # nodes and branch lengths
#t=20 # obs time
"""
from multiprocessing import freeze_support
if __name__ == '__main__':
	freeze_support()
	plot = 0

	mu0 = mu1 = 0.1
	lambda0 = lambda1 = 0.7

	alpha = np.array([0.5,0.5]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.1, q10 =0.1, lambda0 = 0.7, lambda1 = 0.7)
	trees= []

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
	stopping_time = 10
	t2=sim_tree(alpha, D0, d, B, Q1, Pi, stopping_time, min_leaves = 5, seq_len = 1)

	nodeStr = "2A.1B.17C.8D.9E.9F.18G." # nodes and branch lengths
	t=20 # obs time
	trees.append(Tree(1))
	trees[0].str2tree(nodeStr,t,by='df')
	trees[0].disp()
	
	t_current = trees[0]
	t_new= q_ratio = -1
	for i in range(2):
		move = propose_move(t_current,alpha, d, D0, B, i)

		if move != EXIT_FAILURE:
			t_new, q_ratio, alpha_n, d_n, D0_n, B_n, j = move
			if j != 4: trees.append(t_new)
	print(len(trees))

	l1 = []
	l2 = []
	l3 = []
	l4 = []
	for i in range(len(trees)):
		print(i)
		tree = trees[i]
		tree.disp()
		print(tree.toStr())
		a = bd_lik(tree, True, True)
		#b = bd_lik(tree, True, False)
		#c = lin_bd_lik(lambda0, mu0, tree)
		d = tree_prior(tree, alpha, d, D0, B,True, None, False)
		l1.append((i, a))
		#l2.append((i,b))
		#l3.append((i,c))
		l4.append((i,d))

	c = sorted(l1, key=lambda x: x[1])

	#f = sorted(l2, key=lambda x: x[1])

	#g = sorted(l3, key=lambda x: x[1])

	h = sorted(l4, key=lambda x: x[1])
	print('bd T T', c)
	#print('bd w/o', f)
	#print('nee94 ', g)
	print('mbt pr', h)
"""

"""
from ete3 import Tree as Tree_n


for i in range(len(trees)):
  s_ml = trees[i].head.to_newick()
  t = Tree_n(s_ml, format = 3)
  # t.convert_to_ultrametric()
  t.show()
# the order of conclusions are the same, my model works!!!
"""