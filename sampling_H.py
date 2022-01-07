import time
from collections import Counter
import matplotlib.pyplot as plt

import numpy as np
# from calculate_H import H_n
from scipy.stats import zipf, pareto
import scipy

def sample_H(dist, nruns):
	t1 = time.time()
	res = dist.rvs(size=nruns)

	# TEST WITH PARETO DISTRIBUTION, works.
	# res = pareto.rvs(2.5, size=nruns)

	# TEST WITH ZIPF DISTRIBUTION, works.
	# res = zipf.rvs(2.5, size=nruns)

	samples = sorted(Counter(res).items())
	t2 = time.time()
	print("time elapsed to sample " + str(nruns) + " times from H_n(k): " + str(t2 - t1))

	# CALCULATE CDF
	cusum = np.cumsum(list(zip(*samples))[1])
	cdf_samples = cusum / cusum[-1]

	# CALCULATE CCDF
	ccdf_samples = [1-elt for elt in cdf_samples]

	# PLOT SAMPLED CCDF LOGLOG
	plt.plot(ccdf_samples)
	plt.yscale('log')
	plt.xscale('log')
	plt.title('sampled loglog CCDF of H_n')
	plt.savefig('./Figures/k-1/CDFsampled_H_n_CCDF_loglog_')
	plt.show()

	return samples #list of tuples

# RUN SAMPLING BASED ON SCIPY.STATS DISCRETE DISTRIBUTION H_n FROM calculate_H
# n = 200000
# nruns = 1000000
# sam = sample_H(H_n, nruns)

# DO CHISQUARE TEST TO TEST IF SAMPLING IS DONE CORRECTLY, answer=yes.
# s = sum(list(zip(*sam))[1])
# norm = [float(i)/s for i in list(zip(*sam))[1]]
# delete_list = []
# pmf_list_new = pmf_list[2:]
# for i in range(len(pmf_list_new)):
# 	if pmf_list_new[i] == 0:
# 		delete_list.append(i)
# print(delete_list)
# cor = 0
# for index in delete_list:
# 	pmf_list_new.pop(index - cor)
# 	cor += 1
# print(scipy.stats.chisquare(f_obs=norm, f_exp=pmf_list_new)) #p-value high means we can say they are equal.

# DO KSTEST TO TEST IF SAMPLING IS DONE CORRECTLY
# cusum = np.cumsum(list(zip(*sam))[1])
# print(scipy.stats.kstest(list(zip(*sam))[1], H_n.cdf))
# print(scipy.stats.kstest(cusum / cusum[-1], H_n.cdf))

# CALCULATE CDF
# cusum = np.cumsum(list(zip(*sam))[1])
# cdf_samples = cusum / cusum[-1]

# CALCULATE CCDF
# ccdf_samples = [1-elt for elt in cdf_samples]

# PLOT SAMPLED PMF
# plt.plot(*zip(*sam))
# plt.title("sampled PMF of H_n for " + str(n) + " nodes")
# plt.savefig('./Figures/CDFsampled_H_n_' + str(n) + "n" + str(nruns) + "nruns")
# plt.show()

# PLOT SAMPLED CDF
# plt.plot(cusum / cusum[-1])
# plt.title("sampled CDF of H_n for " + str(n) + " nodes")
# plt.savefig('./Figures/CDFsampled_H_n_CDF_' + str(n) + "n" + str(nruns) + "nruns")
# plt.show()

# PLOT SAMPLED CCDF LOGLOG
# plt.plot(ccdf_samples)
# plt.yscale('log')
# plt.xscale('log')
# plt.title('sampled loglog CCDF of H_n cut to n^(1/ple) with ' + str(n) + ' nodes')
# plt.savefig('./Figures/CDFsampled_H_n_CCDF_loglog_' + str(n) + "n" + str(nruns) + "nruns")
# plt.show()




