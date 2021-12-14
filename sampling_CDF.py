import numpy as np
from collections import Counter
import time
import matplotlib.pyplot as plt
from scipy.stats import zipf
import scipy
# from calculate_H import H_n, n


def find_interval(u, cdf_list):
	""""""
	# print(u)
	k = 1
	while True:
		if k <= len(cdf_list):

			(left, right) = (cdf_list[k-1], cdf_list[k])
			if left < u <= right:
				if k == len(cdf_list) or k == len(cdf_list)-1:
					print(u)
				return k
			k += 1
		else:
			print("else case")
			return None

def sample_from_CDF(cdf_list, nruns):
	""""""
	t1 = time.time()
	res = []
	for _ in range(nruns):
		u = np.random.random_sample()
		res.append(find_interval(u, cdf_list))
	samples = sorted(Counter(res).items())
	# print(samples)
	del samples[-1]
	# print(samples)
	t2 = time.time()
	print("time elapsed to sample " + str(nruns) + " times from CDF of H_n(k): " + str(t2 - t1))
	return samples

#
# cdf_lst = H_n
# print(len(cdf_lst))
# print(cdf_lst)
# nruns = 1000000
# sam = sample_from_CDF(cdf_lst, nruns)
#
#
#
# cusum = np.cumsum(list(zip(*sam))[1])
#
# plt.plot(*zip(*sam))
# plt.title("sampled PMF of H_n for " + str(n) + " nodes")
# plt.savefig('./Figures/CDFsampled_H_n_' + str(n) + "n" + str(nruns) + "nruns")
# plt.show()
# #
# cusum = np.cumsum(list(zip(*sam))[1])
# cdf_samples = cusum / cusum[-1]
# ccdf_samples = [1-elt for elt in cdf_samples]
# plt.plot(cusum / cusum[-1])
# plt.title("sampled CDF of H_n for " + str(n) + " nodes")
# plt.savefig('./Figures/CDFsampled_H_n_CDF' + str(n) + "n" + str(nruns) + "nruns")
# plt.show()
#
# plt.plot(ccdf_samples)
# plt.yscale('log')
# plt.xscale('log')
# plt.title('sampled loglog CCDF of H_n cut to n^(1/ple) with ' + str(n) + ' nodes')
# plt.savefig('./Figures/CDFsampled_H_n_CCDF_loglog_cut_' + str(n) + "n" + str(nruns) + "nruns")
# plt.show()

