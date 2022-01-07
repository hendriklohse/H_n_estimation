import numpy as np
from collections import Counter
import time
import matplotlib.pyplot as plt
from scipy.stats import zipf
import scipy
# from calculate_H import H_n_list, n


def find_interval(u, cdf_list):
	""""""
	# print(u)
	k = 1
	while True:
		if k <= len(cdf_list):
			(left, right) = (cdf_list[k-1], cdf_list[k])
			if left < u <= right:
				if k == len(cdf_list):
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
	samples.insert(0,(1,0))
	samples.insert(0,(0,0))
	# print(samples)
	# DELETE LAST VALUE FROM SAMPLES
	del samples[-1]

	t2 = time.time()
	print("time elapsed to sample " + str(nruns) + " times from CDF of H_n(k): " + str(t2 - t1))

	# CALCULATE CDF
	cusum = np.cumsum(list(zip(*samples))[1])
	cdf_samples = cusum / cusum[-1]
	ccdf_samples = [1-elt for elt in cdf_samples]

	# PLOT SAMPLED CCDF LOGLOG
	plt.plot(ccdf_samples)
	plt.yscale('log')
	plt.xscale('log')
	plt.title('sampled loglog CCDF of H_n cut to n^(1/ple)')
	plt.savefig('./Figures/k-1/CDFsampled_H_n_CCDF_loglog')
	plt.show()

	return samples

# RUN SAMPLING BASED ON CDF FROM calculate_H
# cdf_lst = H_n_list
# nruns = 100000
# sam = sample_from_CDF(cdf_lst, nruns)

# # CALCULATE CDF
# cusum = np.cumsum(list(zip(*sam))[1])
# cdf_samples = cusum / cusum[-1]
# print(cdf_samples)
#
# # CALCULATE CCDF
# ccdf_samples = [1-elt for elt in cdf_samples]
# print(ccdf_samples)
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
#
