from collections import Counter
import networkx as nx
import time
import numpy as np
from matplotlib import pyplot as plt
from create_girg import localClusteringCoefficient
from calculate_H_2 import CDF
from sampling_CDF import sample_from_CDF
from retreive_xi import run_tail_estimation

def readGraph(edgeFile, comments, delimiter):
	graph = nx.read_edgelist(edgeFile, comments=comments, delimiter=delimiter)
	return graph

def run_realNetworks(nruns, locClus, graphname):


	big_labdas_eta_dict = {}
	labdaList = [2]
	k_c_list = [len(locClus.keys())]

	for i in range(len(k_c_list)):
		big_labdas_eta_dict[k_c_list[i]] = {}
		for labda in labdaList:
			H_n_CDF = CDF(clusDict=locClus, k_c=k_c_list[i],labda=labda)
			sample_ = sample_from_CDF(H_n_CDF, nruns=nruns, k_c = k_c_list[i], labda=labda)
			estimates_dict = run_tail_estimation(dimension=1, n=None, nruns=nruns, ple=None, sample=sample_, k_c=k_c_list[i], labda=labda, testing=3, graphname=graphname)
			print(estimates_dict)
			big_labdas_eta_dict[k_c_list[i]][labda] = estimates_dict
	print("labdas_eta_dict:")
	print(big_labdas_eta_dict)

	for labda in labdaList:
		hill_scaling = big_labdas_eta_dict[len(locClus.keys())][labda]["hill_ple"]
		moments_scaling = big_labdas_eta_dict[len(locClus.keys())][labda]["moments_ple"]
		kernel_scaling = big_labdas_eta_dict[len(locClus.keys())][labda]["kernel_ple"]

		#plot locClus
		x_axis = [k for k in range(list(sorted(locClus.keys()))[-1])]
		samples = sorted(locClus.items())
		fig = plt.figure(figsize=(5*1.618, 5*1))
		ax = fig.add_subplot(1, 1, 1)
		plt.plot(*zip(*samples), '.', color="black")
		if 0 < hill_scaling < 4:
			plt.plot(x_axis, [k**(-hill_scaling) for k in range(len(x_axis))], label="Hill (" + str(round(hill_scaling,3)) + ")", color="firebrick", linestyle=(0, (5, 1)))
		if 0 < moments_scaling < 4:
			plt.plot(x_axis, [k**(-moments_scaling) for k in range(len(x_axis))], label="Moments (" + str(round(moments_scaling,3)) + ")", color="skyblue", linestyle="solid")
		if 0 < kernel_scaling < 4:
			plt.plot(x_axis, [k**(-kernel_scaling) for k in range(len(x_axis))], label="Kernel (" + str(round(kernel_scaling,3)) + ")", color="orange", linestyle=(0, (3, 1, 1, 1)))
		ax.set_yscale('log')
		ax.set_xscale('log')
		ax.set_xlabel('Degree')
		ax.set_ylabel(r"$\gamma_n(k)$")
		ax.legend()
		plt.savefig("./Figures/realNetworks/" + graphname + "_labda_" + str(labda) + ".png")
		plt.show()


	return big_labdas_eta_dict

# g = "./realNetworks/stegehuis/as20000102.txt"
# graphname_ = g[25:]
# graphname = graphname_[:-4]
# graph = readGraph(g, comments="#", delimiter="\t")
# locClus = localClusteringCoefficient(graph)
#
# run_realNetworks(nruns=200000, locClus=locClus, graphname=graphname)


# googlePlus = nx.read_edgelist("./Examples/g_plusAnonymized/g_plusAnonymized.csv", delimiter=",") #degree works
# academiaEU = nx.read_edgelist("./Examples/academiaAnonymized/academia2Anonymized.csv", delimiter=",") #gamma works a bit
#
# wordnet = nx.read_edgelist("./Examples/wordnet-words/wordnet-words.edges", delimiter=",") #works kind of for both
#
# webgoogle = nx.read_edgelist("./Examples/web-Google.txt",comments="#",delimiter="\t") #definatly works.

