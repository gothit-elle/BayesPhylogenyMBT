import csv
import matplotlib.pyplot as plt
import numpy as np

chaina = []
chainb = []
chainc = []

for i in range(4): 
	with open(f"../thesis_likelihood\csv\c{i+1}a.csv", newline='') as f:
		reader = csv.reader(f, delimiter='Y')
		data = list(reader)
	chaina += data
	
	with open(f"../thesis_likelihood\csv\c{i+1}b.csv", newline='') as f:
		data= []
		reader = csv.reader(f,  delimiter='Y')
		for line in reader:
			data.append([float(x) for x in line])
	chainb += data
	
	with open(f"../thesis_likelihood\csv\c{i+1}c.csv", newline='') as f:
		reader = csv.reader(f,  delimiter='Y')
		data = list(reader)
	chainc += data

print(chainb[0])
N = len(chainb[0])
plt.plot(range(N), chainb[0], range(N), chainb[1], range(N), chainb[2], range(N), chainb[3])
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
plt.xlabel('Step')
plt.ylabel('Posterior Log-Likelihood')
plt.title("Posterior Log-likelihood vs MCMC Step")
plt.legend(["Chain 1", "Chain 2", "Chain 3", "Chain 4"], loc="lower left")
plt.savefig("../thesis_likelihood/plots/MCMC.png")