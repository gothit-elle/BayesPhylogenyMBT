import sys
sys.path.insert(0, '../thesis_likelihood')

import numpy as np
from treestruct import *
from nodestruct import *
from likelihood import *
from buildmtrx import *
from multiprocessing import freeze_support
if __name__ == '__main__':
	freeze_support()
	debug = False
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


	plot=0
	alpha = np.array([0.5,0.5]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)

	"""# Testing displays, likelihood"""

	new_tree = Tree(1)
	new_tree.head = node("T", None, 0.1)
	new_tree.head.right = node("N", new_tree.head, 1.2)
	new_tree.head.left = node("N", new_tree.head, 0.2)
	new_tree.head.left.right = node("T", new_tree.head.left, 0.5)
	new_tree.head.left.left = node("A", new_tree.head.left, 1)
	new_tree.head.left.right.right = node("T", new_tree.head.left, 0.5)
	new_tree.head.left.right.left = node("T", new_tree.head.left, 0.5)
	new_tree.disp()

	print("tree likelihood is: ", log_lik(new_tree, Q1, Pi, debug))
	print(new_tree.head.to_newick())

	"""# Testing displays, likelihood"""

	# could have branch dep substitution rates like file:///C:/Users/User/Downloads/meyer-et-al-2019-simultaneous-bayesian-inference-of-phylogeny-and-molecular-coevolution.pdf

	t2 = Tree(1)
	print(new_tree.toStr())
	t2.str2tree(new_tree.toStr(), by='io')
	t2.disp()

	# longer seqs test
	nodeStr = "2TAT.1TTG.17AAC.8GTG.9TAC.9TAA.18CAT."
	# enter time
	t=20
	t2 = Tree(1)
	t2.str2tree(nodeStr,t,by='df')
	t2.disp()
	print("tree likelihood is: ", log_lik(t2, Q1, Pi, debug))