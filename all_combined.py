import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
# from scipy.spatial.distance import pdist, squareform
# from scipy.stats import poisson
# import sys
# import time
# import argparse
# import os
# import warnings
# from matplotlib import pyplot as plt
# from collections import Counter
# from collections import OrderedDict

from exactClusteringFunction import incompbeta, beta

from create_girg import localClusteringCoefficient, makeBigClus
# from calculate_H import create_H_n
# from create_girg import zipfClustering
# from sampling_H import sample_H
# from get_input import get_input, create_girg_
from retreive_xi import run_tail_estimation
from sampling_CDF import sample_from_CDF
# from getEstimates import get_estimates
from calculate_H_2 import CDF


# from sampling_H import simulate3
# from calculate_gamma import create_pmf_slow

# dic_real = {}
# with open("./Examples/realClusteringFunction_alpha0.8_nu1.dat", 'r') as f:
# 	lines = f.readlines()
# 	for line in lines:
# 		fields = line.split("\t")
# 		dic_real[int(fields[0])] = float(fields[1].strip())
# print(dic_real)

# def makeLocClus(n_list, ple):
# 	big_clusDict = {}
# 	for n in n_list:
# 		create_girg_(n=str(n), d=1, ple=ple, alpha="inf", deg=10, wseed=12, pseed=130, sseed=1400, threads=1, file="graph_" + str(n), dot=0, edge=1)
# 		get_input("graph_" + str(n) + ".txt")
# 		# #
# 		girg = generate_graph(firstTime=True, edgeFile="./input/graph_" + str(n) + ".txt")
# 		locClus = localClusteringCoefficient(girg, n)
# 		big_clusDict[n] = locClus
# 	return big_clusDict

def run_all(n, nruns, ple, bigLocClus):
	# locClus = np.load('./clusteringCoefficients/n_{}.npy'.format(n), allow_pickle=True).item()

	# locClus = zipfClustering(ple, n)
	# # print(locClus)
	#
	# H_n = create_H_n(clus_dict=locClus, dict_length=len(locClus.keys()), n=n, ple=ple)
	if bigLocClus.get(n):
		locClus = bigLocClus[n]
		k_c = int(n**(1/ple))
		k_c = len(locClus.keys())
		print(k_c)
		H_n_CDF = CDF(clusDict=locClus, k_c=k_c)
		#

		# sample_ = sample_H(dist=H_n, nruns=nruns)
		sample_ = sample_from_CDF(H_n_CDF, nruns=nruns)

		# run_tail_estimation(dimension=1, n=n, nruns=nruns, ple=ple, sample=sample_, k_c=k_c, testing=1)

		estimates_dict = run_tail_estimation(dimension=1, n=n, nruns=nruns, ple=ple, sample=sample_, k_c=k_c, testing=1)
		results = {"hill_ple": None, "moments_ple": None, "kernel_ple": None}
		# print(estimates_dict)
		if ple == 2.5:
			for key, value in estimates_dict.items():
				results[key] = value
				estimates_dict[key] = abs(value - 1) if value != None else None
		elif ple > 2.5:
			for key, value in estimates_dict.items():
				results[key] = value
				estimates_dict[key] = abs(value - 1) if value != None else None
		elif 2 < ple and ple < 2.5:
			for key, value in estimates_dict.items():
				results[key] = value
				estimates_dict[key] = abs(value - (2 * ple - 4)) if value != None else None
		else:
			print("error: ple not in correct range")
			return None
		# get_estimates(dimension=1, n=n, nruns=nruns, ple=ple, sample=sample_)
		# print(estimates_dict)
		nu = 1
		alpha = (ple - 1) / 2
		xi = (4 * alpha * nu) / (np.pi * (2 * alpha - 1))
		x_axis = [i for i in range(2, k_c + 1)]
		hill_ple = results["hill_ple"]
		moments_ple = results["moments_ple"]
		kernel_ple = results["kernel_ple"]
		# plot comparing with the exact clustering function
		if ple == 2.5:
			real_clusList = [6 * nu / np.pi * np.log(k) / k for k in range(2, k_c + 1)]
			scaling = 1
		elif ple > 2.5:
			real_clusList = [8 * alpha * nu / (np.pi * (4 * alpha - 3)) * k ** (-1) for k in range(2, k_c + 1)]
			scaling = 1
		elif 2 < ple and ple < 2.5:
			real_clusList = [(((3 * alpha - 1) / (2 ** (4 * alpha + 1) * alpha * (alpha - 1) ** 2) + (
					(alpha - 0.5) * incompbeta(0.5, 2 * alpha + 1, 2 * alpha - 2)) / (2 * (alpha - 1) * alpha) - (
								   beta(2 * alpha, 3 * alpha - 4)) / (4 * (alpha - 1))) * xi ** (4 * alpha - 2)) *
							 k ** (2 - 4 * alpha) for k in range(2, k_c + 1)]
			scaling = 4*alpha-2

		plt.plot(x_axis, real_clusList, label=r"real function $(\sim \eta=1)$", color="blue", lw=1.5)
		plt.plot(x_axis, [k ** (-scaling) for k in range(2, k_c +1)], ls="-.", lw=1.5, color="black",
				 label=r"real scaling $(\eta=" + str(scaling) + r")$")
		plt.plot(x_axis, [k ** (-hill_ple) for k in range(2, k_c + 1)], ls='--', lw=2,
				 label=r"Adj. Hill Scaling $(\eta=" + str(hill_ple) + r")$", color="red")
		plt.plot(x_axis, [k ** (-moments_ple) for k in range(2, k_c + 1)], ls='--', lw=2,
				 label=r"Moments Scaling $(\eta=" + str(moments_ple) + r")$", color="cyan")
		plt.plot(x_axis, [k ** (-kernel_ple) for k in range(2, k_c + 1)], ls='--', lw=2,
				 label=r"Kernel Scaling $(\eta=" + str(kernel_ple) + r")$", color="orange")
		plt.xlabel(r" $k$")
		plt.ylabel(r" $\gamma(k)")
		plt.title("clustering function and estimators for ple {}".format(str(ple)))
		plt.yscale("log")
		plt.xscale("log")
		plt.legend()
		plt.savefig('./Figures/Plots/ple_{}_n{}_nruns{}.png'.format(str(ple), str(n), str(nruns)))
		plt.show()

	return estimates_dict


# NOTE: ple of both functions should be equal.
bigLoc = makeBigClus(n_list=[200000], ple=2.6)
run_all(n=200000, nruns=100000, ple=2.6, bigLocClus=bigLoc)
