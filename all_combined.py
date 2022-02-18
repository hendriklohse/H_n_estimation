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
from sklearn.metrics import mean_squared_error

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
	big_labdas_eta_dict = {}
	big_labdas_rrms_dict = {}
	# H_n = create_H_n(clus_dict=locClus, dict_length=len(locClus.keys()), n=n, ple=ple)
	if bigLocClus.get(n):
		locClus = bigLocClus[n]
		labdaList = [v/4 for v in range(5, 15)]
		# k_c_list = [int(n**(1/(ple+(c/10)))) for c in range(10)]
		k_c_list = [int(n**(1/(ple+1))), int(n**(1/(ple+0.5))), int(n**(1/(ple))), len(locClus.keys())]
		# k_c_list = [len(locClus.keys())]
		# k_c = len(locClus.keys())
		print(k_c_list)
		results_k_c = []
		for i in range(len(k_c_list)):
			big_labdas_eta_dict[k_c_list[i]] = {}
			big_labdas_rrms_dict[k_c_list[i]] = {}
			for labda in labdaList:

				H_n_CDF = CDF(clusDict=locClus, k_c=k_c_list[i],labda=labda)
				#
				print(big_labdas_eta_dict)
				print(big_labdas_eta_dict[k_c_list[i]])
				# sample_ = sample_H(dist=H_n, nruns=nruns)
				sample_ = sample_from_CDF(H_n_CDF, nruns=nruns)

				# run_tail_estimation(dimension=1, n=n, nruns=nruns, ple=ple, sample=sample_, k_c=k_c, testing=1)

				estimates_dict = run_tail_estimation(dimension=1, n=n, nruns=nruns, ple=ple, sample=sample_, k_c=k_c_list[i], labda=labda, testing=1)
				rrms_estimates_dict = {"hill_ple": 10000, "moments_ple": 10000, "kernel_ple": 10000}
				# print(estimates_dict)
				if ple == 2.5:
					for key, value in estimates_dict.items():
						# estimates_dict[key] = value
						rrms = np.sqrt((1 - value)**2) / 1 if value is not None else None
						# rrms = mean_squared_error(y_true =1, y_pred=value, squared=False) / 1 if value is not None else None
						rrms_estimates_dict[key] = rrms
				elif ple > 2.5:
					for key, value in estimates_dict.items():
						# estimates_dict[key] = value
						rrms = np.sqrt((1 - value)**2) / 1 if value is not None else None
						# rrms = mean_squared_error(y_true =1, y_pred=value, squared=False) / 1 if value is not None else None
						rrms_estimates_dict[key] = rrms
				elif 2 < ple < 2.5:
					for key, value in estimates_dict.items():
						# estimates_dict[key] = value
						rrms = np.sqrt((2*ple-4 - value)**2) / (2*ple-4) if value is not None else None
						# rrms = mean_squared_error(y_true =2*ple-4, y_pred=value, squared=False) / (2*ple-4) if value is not None else None
						rrms_estimates_dict[key] = rrms
				else:
					print("error: ple not in correct range")
					return None
				# get_estimates(dimension=1, n=n, nruns=nruns, ple=ple, sample=sample_)
				# print(estimates_dict)

				big_labdas_eta_dict[k_c_list[i]][labda] = estimates_dict
				big_labdas_rrms_dict[k_c_list[i]][labda] = rrms_estimates_dict #plots the rrms

				# Below: for plotting.
				nu = 1
				alpha = (ple - 1) / 2
				# print(alpha)
				xi = (4 * alpha * nu) / (np.pi * (2 * alpha - 1))
				x_axis = [i for i in range(2, k_c_list[i] + 1)]
				hill_ple = estimates_dict["hill_ple"]
				moments_ple = estimates_dict["moments_ple"]
				kernel_ple = estimates_dict["kernel_ple"]
				# plot comparing with the exact clustering function
				if ple == 2.5:
					# real_clusList = [6 * nu / np.pi * np.log(k) / k for k in range(2, k_c_list[i] + 1)]
					scaling = 1
				elif ple > 2.5:
					# real_clusList = [8 * alpha * nu / (np.pi * (4 * alpha - 3)) * k ** (-1) for k in range(2, k_c_list[i] + 1)]
					scaling = 1
				elif 2 < ple < 2.5:
					# # real_clusList = [(((3 * alpha - 1) / (2 ** (4 * alpha + 1) * alpha * (alpha - 1) ** 2) + (
					# 		(alpha - 0.5) * incompbeta(0.5, 2 * alpha + 1, 2 * alpha - 2)) / (2 * (alpha - 1) * alpha) - (
					# 					   beta(2 * alpha, 3 * alpha - 4)) / (4 * (alpha - 1))) * xi ** (4 * alpha - 2)) *
					# 				 k ** (2 - 4 * alpha) for k in range(2, k_c_list[i] + 1)]
					scaling = 4*alpha-2
				else:
					print("error: invalid ple")
					return None

				# plt.plot(x_axis, real_clusList, label=r"real function $(\sim \eta=1)$", color="blue", lw=1.5)
				# plt.plot(x_axis, [k ** (-scaling) for k in range(2, k_c_list[i] +1)], ls="-.", lw=1.5, color="black",
				# 		 label=r"real scaling $(\eta=" + str(scaling) + r")$")
				# plt.plot(x_axis, [k ** (-hill_ple) for k in range(2, k_c_list[i] + 1)], ls='--', lw=2,
				# 		 label=r"Adj. Hill Scaling $(\eta=" + str(hill_ple) + r")$", color="red")
				# plt.plot(x_axis, [k ** (-moments_ple) for k in range(2, k_c_list[i] + 1)], ls='--', lw=2,
				# 		 label=r"Moments Scaling $(\eta=" + str(moments_ple) + r")$", color="cyan")
				# plt.plot(x_axis, [k ** (-kernel_ple) for k in range(2, k_c_list[i] + 1)], ls='--', lw=2,
				# 		 label=r"Kernel Scaling $(\eta=" + str(kernel_ple) + r")$", color="orange")
				# plt.xlabel(r" $k$")
				# plt.ylabel(r" $\gamma(k)")
				# plt.title("clustering function and estimators for ple {} labda {}".format(str(ple), str(labda)))
				# plt.yscale("log")
				# plt.xscale("log")
				# plt.legend()
				# plt.savefig('./Figures/Plots/labdas/ple_{}_n{}_nruns{}_k_c{}_labda{}.png'.format(str(ple), str(n), str(nruns), str(k_c_list[i]), str(labda)))
				# plt.show()

				results_k_c.append(estimates_dict)

			# plt.plot([elt for elt in k_c_list], [1 for _ in range(len(k_c_list))], label=r"coefficient $\eta = 1$", color="black")
			# plt.plot([elt for elt in k_c_list], [results_k_c[i]["hill_ple"] for i in range(len(results_k_c))], 'or' , label="hill result", color="red")
			# plt.plot([elt for elt in k_c_list], [results_k_c[i]["moments_ple"] for i in range(len(results_k_c))],'oc' , label="moments result", color="cyan")
			# plt.plot([elt for elt in k_c_list], [results_k_c[i]["kernel_ple"] for i in range(len(results_k_c))], 'og', label="kernel result", color="orange")
			# plt.title("estimator performace for different k_c's for ple={}".format(str(ple)))
			# plt.xlabel("k_c")
			# plt.ylabel(r"$\gamma(k)$")
			# plt.legend()
			# plt.savefig('./Figures/k_c/ple_{}_n{}_nruns{}.png'.format(str(ple), str(n), str(nruns)))
			# plt.show()
		print("labdas_eta_dict:")
		print(big_labdas_eta_dict)

		print("labdas_rrms_dict:")
		print(big_labdas_rrms_dict)

		for i in range(len(k_c_list)):
			labdas_kernel_rrms_list = [v["kernel_ple"] for v in big_labdas_rrms_dict[k_c_list[i]].values()]
			labdas_kernel_list = [k for k in big_labdas_rrms_dict[k_c_list[i]].keys()]
			if i == 0:
				label = r"$k_c=n^{1/(ple + 1)}$"
			elif i == 1:
				label = r"$k_c=n^{1/(ple + 1/2)}$"
			elif i == 2:
				label = r"$k_c=n^{1/ple}$"
			elif i ==3:
				label = r"$k_c = n$"
			else:
				print("length of k_c_list exceeds 4")
				break
			plt.plot(labdas_kernel_list, labdas_kernel_rrms_list, label=label)

		plt.title("kernel estimator error for n = {}, nruns = {}, ple = {}".format(str(n), str(nruns), str(ple)))
		plt.xlabel(r"$\lambda$")
		plt.ylabel("RRMS")
		plt.ylim(0,3)
		plt.legend()
		plt.savefig("./Figures/kernelPlots/error_n{}_nruns{}_ple{}.png".format(str(n), str(nruns), str(ple)))
		plt.show()

		return big_labdas_rrms_dict

	else:
		print("error: GIRG with " + str(n) + " nodes not sampled")
		return None


# NOTE: ple of both functions should be equal.
bigLoc = makeBigClus(n_list=[200000], ple=2.6, nr_graphs=5)
run_all(n=200000, nruns=100000, ple=2.6, bigLocClus=bigLoc)
