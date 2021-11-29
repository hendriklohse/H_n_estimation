import time
from collections import Counter
import matplotlib.pyplot as plt

import numpy as np
from calculate_gamma_faster import H_n
from scipy.stats import zipf
import scipy


def sample_H(dist, nruns):
	t1 = time.time()
	res = dist.rvs(size=nruns)
	samples = sorted(Counter(res).items())
	t2 = time.time()
	print("time elapsed to sample " + str(nruns) + " times from H_n(k): " + str(t2 - t1))
	return samples #list of tuples


# sam = sample_H(H_n_real,100000)


# n = 200000
# nruns = 1000000
# sam = sample_H(H_n, nruns)
#
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
# print(len(pmf_list_new))
#
# print(norm)
# print(pmf_list_new)
# print(scipy.stats.chisquare(f_obs=norm, f_exp=pmf_list_new)) #p-value high means we can say they are equal.
#
# cusum = np.cumsum(list(zip(*sam))[1])
# print(scipy.stats.kstest(list(zip(*sam))[1], H_n.cdf))
# print(scipy.stats.kstest(cusum / cusum[-1], H_n.cdf))

# plt.plot(*zip(*sam))
# plt.title("sampled PMF of H_n for " + str(n) + " nodes")
# plt.savefig('./Figures/sampled_H_n_' + str(n) + "n" + str(nruns) + "nruns")
# plt.show()
# #
# cusum = np.cumsum(list(zip(*sam))[1])
# cdf_samples = cusum / cusum[-1]
# ccdf_samples = [1-elt for elt in cdf_samples]
# plt.plot(cusum / cusum[-1])
# plt.title("sampled CDF of H_n for " + str(n) + " nodes")
# plt.savefig('./Figures/sampled_H_n_CDF' + str(n) + "n" + str(nruns) + "nruns")
# plt.show()

# plt.plot(ccdf_samples)
# plt.yscale('log')
# plt.xscale('log')
# plt.title('sampled CCDF of H_n for ' + str(n) + ' nodes')
# plt.savefig('./Figures/sampled_H_n_CCDF_loglog' + str(n) + "n" + str(nruns) + "nruns")
# plt.show()

#
# plt.plot(*zip(*sam))
# plt.yscale('log')
# plt.xscale('log')
# plt.title("sampled loglog PMF of H_n for " + str(n) + " nodes")
# plt.savefig('./Figures/sampled_H_N_loglog_' + str(n) + "n" + str(nruns) + "nruns")
# plt.show()



# print(zipf.rvs(3.5, size=10000))
# print(np.mean(zipf.rvs(3.5, size=10000)))
#
# print(scipy.special.zeta(2.5) / scipy.special.zeta(3.5))