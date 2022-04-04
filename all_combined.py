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
# from sklearn.metrics import mean_squared_error

# from exactClusteringFunction import incompbeta, beta

from create_girg import makeBigClus
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
		# labdaList = [v/7 for v in range(7, 28)]
		labdaList = [1, 2, 3]
		# labdaList= [2]
		# k_c_list = [int(n**(1/(ple+(c/10)))) for c in range(10)]
		# k_c_list = [int(n**(1/(ple+1))), int(n**(1/(ple+0.5))), int(n**(1/(ple))), len(locClus.keys())]
		k_c_list = [len(locClus.keys())]
		# print(k_c_list)
		results_k_c = []

		cusum = {}
		ccdf_samples = {}

		for i in range(len(k_c_list)):
			big_labdas_eta_dict[k_c_list[i]] = {}
			big_labdas_rrms_dict[k_c_list[i]] = {}
			for labda in labdaList:

				H_n_CDF = CDF(clusDict=locClus, k_c=k_c_list[i],labda=labda)
				#
				print(big_labdas_eta_dict)
				print(big_labdas_eta_dict[k_c_list[i]])
				# sample_ = sample_H(dist=H_n, nruns=nruns)
				sample_ = sample_from_CDF(H_n_CDF, nruns=nruns, k_c = k_c_list[i], labda=labda)

				# CALCULATE CDF

				cusum[labda] = np.cumsum(list(zip(*sample_))[1])
				cdf_samples = cusum[labda] / cusum[labda][-1]
				ccdf_samples[labda] = [1-elt for elt in cdf_samples]

				# run_tail_estimation(dimension=1, n=n, nruns=nruns, ple=ple, sample=sample_, k_c=k_c, testing=1)

				estimates_dict = run_tail_estimation(dimension=1, n=n, nruns=nruns, ple=ple, sample=sample_, k_c=k_c_list[i], labda=labda, testing=1, graphname=None)
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


		# PLOT SAMPLED CCDF LOGLOG
		fig = plt.figure(figsize=(5*1.618, 5*1))
		ax = fig.add_subplot(1, 1, 1)
		for labda in labdaList:
			if labda == 1:
				clr = "darkgray"
			elif labda==2:
				clr = "dimgrey"
			else:
				clr = "black"
			plt.plot(ccdf_samples[labda], '.', color=clr, label=r"$\lambda = $" + str(labda))
			ax.set_yscale('log')
			ax.set_xscale('log')
			ax.set_title('sampled loglog CCDF of H_n')
			ax.legend()
		plt.savefig('./Figures/CDFsampled_H_n_CCDF_loglog_ple{}_k_c{}_labda{}.png'.format(str(ple),str(k_c_list[i]),str(labda)))
		plt.show()


		print("labdas_eta_dict:")
		print(big_labdas_eta_dict)

		print("labdas_rrms_dict:")
		print(big_labdas_rrms_dict)

		# for i in range(len(k_c_list)):
		#
		# 	labdas_hill_rrms_list = [v["hill_ple"] for v in big_labdas_rrms_dict[k_c_list[i]].values()]
		#
		# 	labdas_moments_rrms_list = [v["moments_ple"] for v in big_labdas_rrms_dict[k_c_list[i]].values()]
		#
		# 	labdas_kernel_rrms_list = [v["kernel_ple"] for v in big_labdas_rrms_dict[k_c_list[i]].values()]
		# 	labdas_list = [k for k in big_labdas_rrms_dict[k_c_list[i]].keys()]
		#
		# 	if i == 0:
		# 		k_c_label = r"$k_c=n^{1/(ple+1)}$"
		# 	elif i == 1:
		# 		k_c_label = r"$k_c=n^{1/(ple+0.5)}$"
		# 	elif i == 2:
		# 		k_c_label = r"$k_c=n^{1/ple}$"
		# 	elif i ==3:
		# 		k_c_label = r"$k_c = k_{max}$"
		# 	else:
		# 		print("length of k_c_list exceeds 4")
		# 		break
		#
		# 	fig = plt.figure(figsize=(5*1.618, 5*1))
		# 	ax = fig.add_subplot(1, 1, 1)
		# 	plt.plot(labdas_list, labdas_hill_rrms_list, label="Hill", color="firebrick", linestyle=(0, (5, 1)))
		# 	plt.plot(labdas_list, labdas_moments_rrms_list, label="Moments", color="skyblue", linestyle="solid")
		# 	plt.plot(labdas_list, labdas_kernel_rrms_list, label="Kernel", color="orange", linestyle=(0, (3, 1, 1, 1)))
		# 	ax.set_title("Root MSE for n = {}, nruns = {}, ".format(str(n), str(nruns), str(ple)) + r"$\beta=$" + " {}, ".format(str(ple)) + k_c_label)
		# 	ax.set_xlabel(r"$\lambda$")
		# 	ax.set_ylabel("Root MSE")
		# 	ax.set_ylim(0,1.5)
		# 	ax.legend()
		# 	fig.savefig("./Figures/simulationFigures/old_k_c/error_n{}_nruns{}_ple{}_k_c{}.png".format(str(n), str(nruns), str(ple),str(k_c_list[i])))
		# 	plt.show()

		return big_labdas_rrms_dict

	else:
		print("error: GIRG with " + str(n) + " nodes not sampled")
		return None


# NOTE: ple of both functions should be equal.
# bigLoc = makeBigClus(n_list=[300000], ple=2.6, nr_graphs=2)
# run_all(n=300000, nruns=600000, ple=2.6, bigLocClus=bigLoc)
