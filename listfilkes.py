
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from prior import *
from buildmtrx import *
import numpy as np
from scipy.linalg import expm
LIMIT = 30
STEPS = 100

alpha = np.array([0.7,0.3]).astype(object)
lambda_a = np.array([1, 0,0,0,0,0,0,0.1]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.3, mu1= 0.1, q01 = 0.9, q10 =0.1, lambda_a = lambda_a)


def TTE(end, alpha, d, D0, B):
	res = get_E(end, alpha, d, D0, B, plot=1).y
	return alpha@res

def MTPS(end, alpha, D0, B):

	times = np.linspace(0,end)
	N = len(B)
	prod = np.transpose(np.kron(np.ones(N),np.identity(N)) + np.kron(np.identity(N),np.ones(N))) # this transpose is once again sus
	omega = D0 + B@(prod)

	return [alpha@expm(omega*t)@np.ones(N) for t in times]

print("TTE")
extinction_time_true = TTE(LIMIT, alpha, d, D0, B)


print("MTPS")
mean_total_pop_true = MTPS(10, alpha, D0, B)

 
alpha = np.array([0.53965628, 0.46034372]).astype(object)
lambda_a = np.array([0.79414076, 0.67004163, 0.93875172, 0.86574251, 0.4882603, 0.97083113, 0.94799957, 0.7425331]).astype(object)
d, D0, D1, B = build_mtrx(mu0= 0.49382398, mu1= 0.9328768, q01 = 0.44993805, q10 = 0.71900588, lambda_a = lambda_a)
print(alpha)
print(d)
print(D0)
print(B)
print("TTE")
extinction_time_MAP = TTE(LIMIT, alpha, d, D0, B)


print("MTPS")
mean_total_pop_MAP = MTPS(10, alpha, D0, B)
print(mean_total_pop_true)
i = 0
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
plt.xlabel('t')
plt.ylabel('M(t)')
plt.title("Theoretical Mean Total Population Size")
plt.plot(np.linspace(0,10), mean_total_pop_true, label="True")
plt.plot(np.linspace(0,10), mean_total_pop_MAP, label="MAP")
plt.legend()
plt.savefig(f"../thesis_likelihood/plots/meanpop_theoretical_{i}.png")
plt.clf()


params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
plt.xlabel('t')
plt.ylabel(r'$\alpha \mathbf{E}(t)$')
plt.title("Theoretical Time to Extinction")
plt.plot(np.linspace(0,LIMIT,STEPS), extinction_time_true, label="True")
plt.plot(np.linspace(0,LIMIT,STEPS), extinction_time_MAP, label="MAP")
plt.legend()
plt.savefig(f"../thesis_likelihood/plots/extinction_theoretical_{i}.png")