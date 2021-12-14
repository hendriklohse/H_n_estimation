from scipy import stats
import matplotlib.pyplot as plt
from get_input import get_input, create_girg_
from create_girg import generate_graph, localClusteringCoefficient
import numpy as np

def H_n(clusDict, k: int, limit):
	"""create the pmf tuple"""
	c_n_sum = 0
	for i in range(2, limit+1):
		if clusDict.get(i):
			c_n_sum += i ** (-2) * clusDict[i]

	if k > 0 and c_n_sum > 0 and clusDict.get(k) and clusDict.get(k) > 0:
		return k, (1 / c_n_sum) * k ** (-2) * clusDict[k]
	else:
		return k, 0.0

def create_CDF(clus_dict, dict_length, n, ple):

	# SET k_c
	k_c = int(n**(1/(ple)))

	# CALCULATE H_n
	H_tuple_k = []
	H_tuple_prb_nn = []
	for j in range(dict_length):
		H_tuple_k.append(H_n(clus_dict, j, dict_length)[0])
		H_tuple_prb_nn.append(H_n(clus_dict, j, dict_length)[1])

	# DELETE FROM FIRST ZERO
	# for i in range(2, len(H_tuple_prb_nn)):
	# 	if H_tuple_prb_nn[i] == 0.0:
	# 		k_c_zero = i
	# 		break
	# k_c = int(k_c_zero)
	# rest = dict_length - k_c
	# del H_tuple_k[-rest:]
	# del H_tuple_prb_nn[-rest:]

	# CORRECT FOR FIRST TWO VALUES OF H_n
	H_tuple_prb_nn[0] = 0.0
	H_tuple_prb_nn[1] = 0.0

	# NORMALIZE H_n PROBABILITIES
	s = sum(H_tuple_prb_nn)
	H_tuple_prb = [prb / s for prb in H_tuple_prb_nn]

	# PLOT H_n loglog
	# plt.plot(H_tuple_k, H_tuple_prb, 'o', color='black')
	# plt.title('pmf prb loglog')
	# plt.yscale('log')
	# plt.xscale('log')
	# plt.savefig('./Figures/pmf_prb_loglog')
	# plt.show()

	# PLOT H_n
	# plt.plot(H_tuple_k, H_tuple_prb, 'o', color='black')
	# plt.title('pmf prb')
	# plt.savefig('./Figures/pmf_prb')
	# plt.show()

	# CALCULATE CDF
	cusum = np.cumsum(H_tuple_prb_nn)
	cdf_prb = cusum / cusum[-1]

	# CALCULATE CCDF
	ccdf_prb = [1-elt for elt in cdf_prb]

	# CUT UNTIL k_c
	rest = dict_length - k_c
	del ccdf_prb[-rest:]
	cdf_prb = [1-elt for elt in ccdf_prb]

	# APPEND 1.0 for sampling
	cdf_prb.append(1.0)

	# PLOT CCDF LOGLOG
	# kList = [i for i in range(len(ccdf_prb))]
	# plt.plot(kList, ccdf_prb, 'o', color='black')
	# plt.title('ccdf prb loglog')
	# plt.yscale('log')
	# plt.xscale('log')
	# plt.savefig('./Figures/ccdf_prb_loglog')
	# plt.show()

	# PLOT CCDF
	# plt.plot(kList, ccdf_prb, 'o', color='black')
	# plt.title('ccdf prb')
	# plt.savefig('./Figures/ccdf_prb')
	# plt.show()

	return cdf_prb

	# CREATE CUSTOM SCIPY.STATS DISCRETE DISTRIBUTION FROM H_n
	# custm = stats.rv_discrete(name='custm', values=(H_tuple_k, H_tuple_prb))

	# CALCULATE CDF
	# cdf_list = [custm.cdf(k) for k in range(dict_length)]
	# cdf_list[0] = 0.0
	# cdf_list[1] = 0.0

	# CALCULATE CCDF
	# ccdf_list = [1-elt for elt in cdf_list]

	# CUT UNTIL k_c
	# rest = dict_length - k_c
	# del ccdf_list[-rest:]

	# PLOT CCDF loglog
	# k_list = [i for i in range(len(ccdf_list))]
	# plt.plot(k_list, ccdf_list, 'o', color='black')
	# plt.title('ccdf prb loglog)
	# plt.yscale('log')
	# plt.xscale('log')
	# plt.savefig('./Figures/ccdf_prb_loglog')
	# plt.show()

	# CREATE NEW CDF FROM CUTOFF CCDF
	# cdf_list2 = [1 - elt for elt in ccdf_list]
	# cdf_list2.append(1.0)
	# cdf_list2.insert(0,0.0)
	# cdf_list2[0] = 0.0
	# cdf_list2[1] = 0.0
	# return cdf_list2

	# CREATE NEW PMF H_n FROM CUTOFF CDF
	# pmf_list2_nn = np.ediff1d(cdf_list2, to_begin=cdf_list2[0])
	# pmf_list2_nn[0] = 0.0
	# pmf_list2_nn[1] = 0.0

	# NORMALIZE PMF
	# norm = 1-sum(pmf_list2_nn)
	# pmf_list2 = [prb + norm / len(pmf_list2_nn) for prb in pmf_list2_nn]

	# CREATE CUSTOM SCIPY.STATS DISCRETE DISTRIBUTION FROM NEW PMF
	# k_list2 = [i for i in range(len(pmf_list2))]
	# custm2 = stats.rv_discrete(name='custm2', values=(k_list2, pmf_list2))

	# return custm2

# IMPORT REAL CLUSTERING FUNCTION
# dic_real = {}
# with open("./Examples/realClusteringFunction_alpha0.8_nu1.dat", 'r') as f:
# 	lines = f.readlines()
# 	for line in lines:
# 		fields = line.split("\t")
# 		dic_real[int(fields[0])] = float(fields[1].strip())
# print(dic_real)
#
# H_n_real = create_PMF(clus_dict=dic_real, dict_length=len(dic_real.keys()), ple=2.6, n=10000)


# # # # # # #
# RUN SIMULATION
n = 100000
ple=2.5
create_girg_(n=str(n), d=1, ple=2.5, alpha="inf", deg=10, wseed=12, pseed=130, sseed=1400, threads=1, file="graph_" + str(n), dot=0, edge=1)
get_input("graph_" + str(n) + ".txt")


girg = generate_graph(firstTime=True, edgeFile="./input/graph_" + str(n) + ".txt")
locClus = localClusteringCoefficient(girg)

H_n = create_CDF(clus_dict=locClus, dict_length=len(locClus.keys()), n=n, ple=2.5)
# # # # # # #

# PLOT LOCAL CLUSTERING COEFFICIENT
# clusList = []
# for i in range(2, len(locClus.keys())+1):
# 	if locClus.get(i):
# 		clusList.append(locClus[i])
#
# plt.plot(clusList)
# plt.title("clustering function")
# plt.savefig('./Figures/locClus')
# plt.show()
# #
# plt.plot(clusList)
# plt.yscale('log')
# plt.xscale('log')
# plt.title("clustering function loglog")
# plt.savefig('./Figures/locClus_loglog')
# plt.show()


# CALCULATE CDF AND CCDF of H_n
# cdf_list = [H_n.cdf(k) for k in range(k_c)]
# ccdf_list = [1-elt for elt in cdf_list]

# CUT OFF CCDF UNTIL k_c
# k_c = int(n**(1/ple))
# rest = len(locClus.keys()) - k_c
# del ccdf_list[-rest:]

# CALCULATE NEW CUTOFF CDF AND PMF
# cdf_list2 = [1 - elt for elt in ccdf_list]
# pmf_list = [H_n.pmf(k) for k in range(k_c)]

# PLOT PMF
# plt.plot(pmf_list)
# plt.title("PMF2 of H_n")
# plt.savefig('./Figures/pmf_list2')
# plt.show()

# PLOT PMF LOGLOG
# plt.plot(pmf_list)
# plt.yscale('log')
# plt.xscale('log')
# plt.title("loglog PMF2 of H_n")
# plt.savefig('./Figures/pmf_list2_loglog')
# plt.show()

# PLOT CDF
# plt.plot(cdf_list)
# plt.title("CDF2 of H_n")
# plt.savefig('./Figures/cdf_list2')
# plt.show()

# PLOT CCDF LOGLOG
# x_list = [x for x in range(len(ccdf_list))]
# plt.plot(x_list, ccdf_list, 'o', color='black')
# plt.yscale('log')
# plt.xscale('log')
# plt.title("loglog CCDF2 of H_n")
# plt.savefig('./Figures/ccdf_list2_loglog')
# plt.show()

