"""# 3d plots

"""
import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from prior import *
from buildmtrx import *

alpha = np.array([0.5,0.5]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.1, mu1= 0.1, q01 = 0.9, q10 =0.01, lambda0 = 1, lambda1 = 0.099)

plot = 1
toler = 1e-7
def G_bkxk2(z,x):
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
    dG = dG.astype(np.float64).ravel()
    return dG

  times = np.linspace(0,z)
  state0 = np.identity(2).ravel()#initially the identity

  dataG = solve_ivp(fun=f, t_span=[0,z], y0=np.array(state0), t_eval= times, atol=toler, rtol=toler, method = 'RK45')

  mtrx = dataG.y # will always be last
  return mtrx

 # either this calc is wrong or the calc of D_1 is wrong.

def build_3d_G(z,x):
  x = np.linspace(0,x,11)
  y = np.linspace(0,z)
  Z00 = []
  Z01 = []
  Z10 = []
  Z11 = []

  for i in range(len(x)):
    elem = G_bkxk2(z, x[i])
    Z00.append(list(elem[0,:]))
    Z01.append(list(elem[1, :]))
    Z10.append(list(elem[2, :]))
    Z11.append(list(elem[3, :]))
  X, Y = np.meshgrid(x,y)
  return X, Y, Z00, Z01, Z10, Z11

if plot: X,Y,Z00, Z01, Z10, Z11 = build_3d_G(5,10)

colours = ['#CC2F00', '#DB6600', '#E39E00', '#76B80D', '#007668', '#006486', '#007CB5', '#465AB2', '#6D47B1', '#873B9C', '#000000' ]
colours.reverse()
from itertools import cycle
cols = cycle(colours)

for elem in Z00:
  plt.plot(np.linspace(0,5), elem, next(cols))
  params = {'mathtext.default': 'regular' }
  plt.rcParams.update(params)
  plt.axis([0, 5, 0, 1])
  plt.xlabel('t')
  plt.ylabel('G(z,x)$_{00}$')
  plt.title("G(z,x)$_{00}$ for different x values")
  plt.legend(['x='+str(i) for i in np.linspace(0,10,11)], loc="upper right")
  plt.savefig("../thesis_likelihood/plots/g00.png")
plt.clf()
for elem in Z01:
  plt.plot(np.linspace(0,5), elem, next(cols))
  params = {'mathtext.default': 'regular' }
  plt.rcParams.update(params)
  plt.axis([0, 5, 0, 1])
  plt.xlabel('t')
  plt.ylabel('G(z,x)$_{01}$')
  plt.title("G(z,x)$_{01}$ for different x values")
  plt.legend(['x='+str(i) for i in np.linspace(0,10,11)], loc="lower right")
  plt.savefig("../thesis_likelihood/plots/g01.png")
plt.clf()
for elem in Z10:
  plt.plot(np.linspace(0,5), elem, next(cols))
  params = {'mathtext.default': 'regular' }
  plt.rcParams.update(params)
  plt.axis([0, 5, 0, 0.01])
  plt.xlabel('t')
  plt.ylabel('G(z,x)$_{10}$')
  plt.title("G(z,x)$_{10}$ for different x values")
  plt.legend(['x='+str(i) for i in np.linspace(0,10,11)], loc="lower right")
  plt.savefig("../thesis_likelihood/plots/g10.png")
plt.clf()
for elem in Z11:
  plt.plot(np.linspace(0,5), elem, next(cols))
  params = {'mathtext.default': 'regular' }
  plt.rcParams.update(params)
  plt.axis([0, 5, 0, 1])
  plt.xlabel('t')
  plt.ylabel('G(z,x)$_{11}$')
  plt.title("G(z,x)$_{11}$ for different x values")
  plt.legend(['x='+str(i) for i in np.linspace(0,10,11)], loc="upper right")
  plt.savefig("../thesis_likelihood/plots/g11.png")
plt.clf()
if plot:
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.dist = 13
  ax.plot_surface(X, Y, np.transpose(np.array(Z00)), cmap='cool')
  ax.set_xlabel('x')
  ax.set_ylabel('z')
  ax.set_zlabel('G(z,x)$_{00}$');
  plt.title("Plot of G(z,x)$_{00}$")
  plt.tight_layout()
  plt.savefig("../thesis_likelihood/plots/g003d.png")
	
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.dist = 13
  ax.plot_surface(X, Y, np.transpose(np.array(Z01)), cmap='cool')
  ax.set_xlabel('x')
  ax.set_ylabel('z')
  ax.set_zlabel('G(z,x)$_{01}$');
  plt.title("Plot of G(z,x)$_{01}$")
  plt.tight_layout()
  plt.savefig("../thesis_likelihood/plots/g013d.png")

  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.dist = 13
  ax.plot_surface(X, Y, np.transpose(np.array(Z10)), cmap='cool')
  ax.set_xlabel('x')
  ax.set_ylabel('z')
  ax.set_zlabel('G(z,x)$_{10}$');
  plt.title("Plot of G(z,x)$_{10}$")
  plt.tight_layout()
  plt.savefig("../thesis_likelihood/plots/g103d.png")

  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.dist = 13
  ax.plot_surface(X, Y, np.transpose(np.array(Z11)), cmap='cool')
  ax.set_xlabel('x')
  ax.set_ylabel('z')
  ax.set_zlabel('G(z,x)$_{11}$');
  plt.title("Plot of G(z,x)$_{11}$")
  plt.tight_layout()
  plt.savefig("../thesis_likelihood/plots/g113d.png")