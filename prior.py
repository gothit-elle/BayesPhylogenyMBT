
"""Example from https://dunnlab.org/phylogenetic_biology/inferring-phylogenies-from-data.html

Create a function to solve G(bk,xk) from the paper

# Prior Calc

"""
import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("error")

toler = 1e-7

def get_E(end, alpha, d, D0, B, plot=0):
    # @constrain([0, 1]) # limits growth at end points. doesnt impact unless t is big
    def E_sol(t, E):
      # we need this loop, otherwise in big time steps E(t) might be > 1
      E = np.array(E).astype(object)
      for i in range(len(E)):
        if E[i] > 1:
          E[i] = 1 # hack-y solution to do this
      dE = d + D0@E + B@np.kron(E, E)
      return dE.astype(object)

    stateE0 = np.array([0,0]) # E(0) = 0 initial cond
    if plot:
      times = np.linspace(0, end)
      x_sol = solve_ivp(E_sol, [0,end], stateE0, t_eval=times, atol=toler, rtol=toler, method='RK45')
    else:
      x_sol = solve_ivp(E_sol, [0,end], stateE0, atol=toler, rtol=toler, method='RK45')
    return  x_sol.y[:, -1]


def G_bkxk(z,x, alpha, d, D0, B, plot=0):
  #@constrain([0,1])
  def f(t, state):
    G = np.array(state).astype(object)
    G.shape = (2,2)
    if t+x == 0:
      E = get_E(t+x+toler, alpha, d, D0, B)
    else:
      E = get_E(t+x, alpha, d, D0, B)
    E.shape = (2,1) # force the shape - 2 rows 1 column

    # need to typecast for accuracy?
    #dG = D0@G + B@(np.kron(E,G.astype(np.float64)).astype(object) + np.kron(G.astype(np.float64), E).astype(object))
    p1 = np.kron(E,G) + np.kron(G,E)
    dG = D0@G + B@(p1)
    dG = dG.astype(object).ravel()
    return dG

  state0 = np.identity(2).ravel()#initially the identity
  if plot:
    times = np.linspace(0,z)
    dataG = solve_ivp(fun=f, t_span=[0,z], y0=np.array(state0), t_eval= times, atol=toler, rtol=toler, method = 'RK45')
  else:
    dataG = solve_ivp(fun=f, t_span=[0,z], y0=np.array(state0), atol=toler, rtol=toler, method = 'RK45')



  if plot:
    def get_D1h(b, d, D0, B):

      def f(t, state):
        # partition state. NB: will need to use a lookup for >2 states
        for i in range(len(state)):
          if state[i] > 1:
            state[i] = 1 # hack-y solution to do this
        E = np.array(state[0:2]).astype(object)
        D_1 = np.array(state[2:4]).astype(object)
        # our ODEs
        dE = d + D0@E + B@(np.kron(E, E).astype(object))
        dD_1 = D0@D_1 + B@(np.kron(E, D_1).astype(object) + np.kron(D_1, E).astype(object))
        return np.concatenate((dE, dD_1)).astype(np.float64)

      state0 = np.array([0,0, 1,1]) # E(0) = 0, D_1 = 1 initial conds
      times = np.linspace(0,b, 50)
      x_sol = solve_ivp(f, [0,b], state0, atol=1e-12, rtol = 1e-12)
      # print("2nd E is ", x_sol.y[0:2, -1])
      # print("D^(1) is ", x_sol.y[2:4, -1])
      return (x_sol.y[2:4,:], x_sol.t)
    ee, ts = get_D1h(z, d, D0, B)
    plt.plot(dataG.t, dataG.y[0,:]+dataG.y[1,:] ,'m-', dataG.t, dataG.y[2,:]+dataG.y[3,:], 'r-', ts, ee[0], 'c--', ts, ee[1], 'b--')
    params = {'mathtext.default': 'regular' }
    plt.rcParams.update(params)
    plt.axis([0, z, 0, 1])
    plt.xlabel('t')
    plt.ylabel('Function value')
    plt.title("G(t,0) vs D$^{(1)}$(t)")
    plt.legend(["G(z,x)$_{00}$+G(z,x)$_{01}$", "G(z,x)$_{10}$+G(z,x)$_{11}$", "D$^{(1)}$$_0$(t)", "D$^{(1)}$$_1$(t)"], loc="upper right")
    plt.savefig("../thesis_likelihood/plots/gvd.png")
  mtrx = dataG.y[:,-1].reshape(2,2) # will always be last
  return mtrx


# handles the external branches likelihood calc
def ext_f(bm_hat, alpha, d, D0, B):

  def f(t, state):
    # partition state. NB: will need to use a lookup for >2 states
    for i in range(len(state)):
      if state[i] > 1:
        state[i] = 1 # hack-y solution to do this
    E = np.array(state[0:2]).astype(object)
    D_1 = np.array(state[2:4]).astype(object)
    # our ODEs
    dE = d + D0@E + B@(np.kron(E, E).astype(object))
    dD_1 = D0@D_1 + B@(np.kron(E, D_1).astype(object) + np.kron(D_1, E).astype(object))
    return np.concatenate((dE, dD_1)).astype(object)

  state0 = np.array([0,0, 1,1]) # E(0) = 0, D_1 = 1 initial conds
  #times = np.linspace(0,bm_hat)
  x_sol = solve_ivp(f, [0, bm_hat], state0,  atol=toler, rtol = toler)

  return np.array(x_sol.y[2:4, -1]).astype(object) # need to return D_1 at time bm_hat


def int_f(cur, alpha, d, D0, B):
  if (cur.isLeaf()):
    lik = ext_f(cur.time, alpha, d, D0, B) # this is correct.

  else:
    t_left = int_f(cur.left, alpha, d, D0, B)
    t_right = int_f(cur.right, alpha, d, D0, B)
    G_val = np.array(G_bkxk(cur.time, cur.dist_from_tip(), alpha, d, D0, B)).astype(object)

    prod = np.kron(t_left, t_right).astype(object) + np.kron(t_right, t_left).astype(object)
    lik = G_val@B@prod

  return np.array(lik).astype(object)


def tree_prior(tree, alpha, d, D0, B, log=True, fname = None):
  cur = tree.head
  # doesnt work with fractional lengths?
  t_left = int_f(cur.left, alpha, d, D0, B)
  t_right = int_f(cur.right, alpha, d, D0, B)
  G_val = np.array(G_bkxk(cur.time,cur.dist_from_tip(), alpha, d, D0, B)).astype(object)
  alpha = np.array(alpha).astype(object)
  prod = np.array(np.kron(t_right, t_left) + np.kron(t_left, t_right)).astype(object)
  val = alpha@G_val@B@prod
  try:
    if log:
      val = np.log(val)
  except RuntimeWarning as rw:
    print("\n", rw, "val=", val)
    tree.disp(log, fname = fname)
    print(tree.toStr(), alpha, d, D0, B, file = fname)
    val = 1 # impossible as probs are < 1
  return val
