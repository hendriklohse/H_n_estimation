import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
#
# from get_input import get_input, create_girg_
# from create_girg import generate_graph, localClusteringCoefficient, zipfClustering


def c_n_sum_(clusDict, limit):
	c_n_sum = 0
	for i in range(2, limit + 1):
		if clusDict.get(i):
			c_n_sum += i ** (-1) * clusDict[i]
	return c_n_sum


def H_n(clusDict, k: int, c_n_sum):
	"""create the pmf"""
	# c_n_sum = 0
	# for i in range(2, limit+1):
	# 	if clusDict.get(i):
	# 		c_n_sum += i ** (-2) * clusDict[i]

	if k > 0 and c_n_sum > 0 and clusDict.get(k) and clusDict.get(k) > 0:
		return (1 / c_n_sum) * k ** (-3) * clusDict[k]
	else:
		return 0.0


def PMF(clusDict, k_c):
	dicLength = len(clusDict.keys())
	cSum = c_n_sum_(clusDict=clusDict, limit=k_c)
	pmf_list = []
	for i in range(k_c):
		pmf_list.append(H_n(clusDict=clusDict, k=i, c_n_sum=cSum))

	# plt.plot(pmf_list)
	# plt.title("normal pmf")
	# plt.show()
	#
	# plt.plot(pmf_list)
	# plt.yscale('log')
	# plt.xscale('log')
	# plt.title("loglog pmf")
	# plt.show()

	return pmf_list

def CDF(clusDict, k_c):
	# print(k_c)
	pmf = PMF(clusDict=clusDict, k_c=k_c)
	cusum = np.cumsum(pmf)
	cdf_list = cusum / cusum[-1]

	# plt.plot(cdf_list)
	# plt.title("normal cdf")
	# plt.show()
	#
	# plt.plot(cdf_list)
	# plt.yscale('log')
	# plt.xscale('log')
	# plt.title("loglog cdf")
	# plt.show()

	return cdf_list

#run
# n = 500000
# ple=2.5
# k_c = int(n**(1/(ple+1)))
# print(k_c)
# create_girg_(n=str(n), d=1, ple=2.5, alpha="inf", deg=10, wseed=12, pseed=130, sseed=1400, threads=1, file="graph_" + str(n), dot=0, edge=1)
# get_input("graph_" + str(n) + ".txt")
#
#
# girg = generate_graph(firstTime=True, edgeFile="./input/graph_" + str(n) + ".txt")
# locClus = localClusteringCoefficient(girg)
#
# # locClus = zipfClustering(ple, n)
#
# CDF(locClus, k_c)