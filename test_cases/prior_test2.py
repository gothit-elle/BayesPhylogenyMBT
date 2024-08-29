import sys
sys.path.insert(0, '../thesis_likelihood')
import numpy as np
from treestruct import *
from nodestruct import *
from prior import *
from buildmtrx import *
from multiprocessing import freeze_support

def get_D1h(b):
  def f(state, t, d, D0, B):
    # partition state. NB: will need to use a lookup for >2 states
    E = np.array(state[0:2]).astype(object)
    D_1 = np.array(state[2:4]).astype(object)
    # our ODEs
    dE = d + D0@E + B@(np.kron(E, E))
    dD_1 = D0@D_1 + B@(np.kron(E, D_1) + np.kron(D_1, E))
    return dE.tolist() +dD_1.tolist()

  state0 = [0,0, 1,1] # E(0) = 0, D_1 = 1 initial conds
  times = np.linspace(0,b)
  x_sol = odeint(f, state0, times, (d, D0, B))
  if plot:
    plt.plot(times, x_sol[:,0],'g--', times, x_sol[:,1],'b--', times, x_sol[:,2], 'm-', times, x_sol[:,3], 'c-')
    plt.axis([0, 20, 0, 1])
    params = {'mathtext.default': 'regular' }
    plt.rcParams.update(params)
    plt.xlabel('Time')
    plt.ylabel('Function value')
    plt.title("E(t) and D$^{(1)}$(t) vs Time")
    plt.legend(["E$_0$(t)", "E$_1$(t)", "D$^{(1)}$$_0$(t)", "D$^{(1)}$$_1$(t)"], loc="upper right")
  return np.array(x_sol[-1,2:4]).astype(object) # need to return D_1 at time bm_hat
  # TODO: double check getting to the right index

if __name__ == '__main__':
	alpha = np.array([0.5,0.5]).astype(object)
	lambda_a = np.array([1, 0,0,0,0,0,0,0.099]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda_a = lambda_a)


	plot = 0
	new_tree = Tree(1)
	alpha = np.array([0.5,0.5]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda_a = lambda_a)
	b1 = 2
	b2 = 1
	b3 = 8
	b3h = x2 = 17
	b4h = x1 = 18
	x3 = b1h = b2h = 9
	new_tree.head = node("T", None, b1)
	new_tree.head.right = node("C", new_tree.head, b4h)
	new_tree.head.left = node("G", new_tree.head, b2)
	new_tree.head.left.right = node("T", new_tree.head.left, b3)
	new_tree.head.left.left = node("A", new_tree.head.left, b3h)
	new_tree.head.left.right.right = node("T", new_tree.head.left,b1h)
	new_tree.head.left.right.left = node("T", new_tree.head.left, b2h)
	new_tree.disp()

	liks = [0,0,0,0]
	lambda_a = np.array([1, 0,0,0,0,0,0,0.099]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda_a = lambda_a)
	liks[0] = tree_prior(new_tree,alpha, d, D0, B, True, None, True)

	lambda_a = np.array([0.3, 0,0,0,0,0,0,0.099]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.1, mu1 = 0.1, q01 = 0.9, q10 =0.001, lambda_a = lambda_a)
	liks[1] = tree_prior(new_tree,alpha, d, D0, B, True, None, True)
	
	lambda_a = np.array([1, 0,0,0,0,0,0,1]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.999, mu1 = 0.099, q01 = 0.2, q10 =0.001, lambda_a = lambda_a)
	liks[2] = tree_prior(new_tree,alpha, d, D0, B, True, None, True)

	lambda_a = np.array([1, 0,0,0,0,0,0,1]).astype(object)
	d, D0, D1, B = build_mtrx(mu0= 0.2, mu1 = 0.099, q01 = 0.2, q10 =0.001, lambda_a = lambda_a)
	liks[3] = tree_prior(new_tree,alpha, d, D0, B, True, None, True)
	
	print("the tree priors are")
	print(liks)

"""
print("the tree prior is")
lik = tree_prior(new_tree, alpha, d, D0, B)
print(lik)

#"by hand" method on page 16 of paper



p1 = get_D1h(b1h)
print("p1", p1)
p2 = get_D1h(b2h)
print("p2", p2)
p3 = get_D1h(b3h)
print("p3", p3)
G_val = G_bkxk(b3, x3, alpha, d, D0, B)
#print("G is", G_val, "with", b3, x3)
print("val is", G_val@D1@(2*p1*p2))
p4 = 2*G_val@D1@(2*p1*p2) * p3
p5 = get_D1h(b4h)
print("p5", p5)
G_val = G_bkxk(b2, x2, alpha, d, D0, B)
#print("G is", G_val, "with", b2, x2)
print("next val is", G_val@D1@(p4))
p6 = 2*G_val@D1@(p4)*p5
G_val = G_bkxk(b1, x1, alpha, d, D0, B)
xm2 = alpha@G_val@D1@(p6)
print(xm2)
print(np.log(xm2))

print("node A, 17", p3)
print("node T, 9", p2)
print("node T, 9", p1)
print("node T, 8", p4) # the internal calcs r wrong
print("node G, 1", p6)  # the internal calcs r wrong
print("node C, 18", p5)

# again this is different and i dont know why ;w;
# its like... different to botht he paper and my answer.

d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.001, lambda0 = 1, lambda1 = 0.099)
mystr = "(4 / 'T'(2 / 'G'(5 / 'A')(5 / 'T'))(7 / 'C'))"
t2 = Tree(1)
t2.str2tree(mystr, by='io')
t2.disp()
lik = tree_prior(t2, alpha, d, D0, B)
print(lik)

b1 = 4
b2 = 2
b3h = x1 = 7
x2 = b1h = b2h = 5

p1 = get_D1h(b1h)
p2 = get_D1h(b2h)
p3 = get_D1h(b3h)
print(p1, p2, p3)
print(2*p1*p2)

G_val = G_bkxk(b2, x2, alpha, d, D0, B)
print("g", G_val@D1@(2*p1*p2))
p4 = G_val@D1@(2*p1*p2)*p3
print(p4)
p5 = alpha@G_bkxk(b1, x1, alpha, d, D0, B)@D1@(p4)
print(np.log(2*p5))

"""


