import numpy as np
import networkx as nx
import networkit as nk
from collections import Counter
import matplotlib.pyplot as plt
import scipy

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

data = np.loadtxt('./Examples/CAIDA_KONECT.dat')
ax1.plot(data[:,0], data[:,1])
ax1.set_title('CAIDA_KONECT')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set(xlim=(1, 10**4), ylim=(1, 10**5))
ax1.grid(True)

# data = np.loadtxt('./output/k-1/girg_1D_200000n_100000nruns_2.5ple_cut_372.dat')
# ax2.plot(data[:,0], data[:,1])
# ax2.set_title('girg_200000n+100000nruns cut to 0.8*limit')
# ax2.set_yscale('log')
# ax2.set_xscale('log')
# ax2.set(xlim=(1, 10**4), ylim=(1, 10**5))
# ax2.grid(True)

data = np.loadtxt('./output/k-2/girg_1D_200000n_100000nruns_2.5ple_cut_372.dat')
ax2.plot(data[:,0], data[:,1])
ax2.set_title('k-2 girg_200000n+100000nruns cut to 0.8*limit')
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set(xlim=(1, 10**4), ylim=(1, 10**5))
ax2.grid(True)

data = np.loadtxt('./output/k-1/girg_1D_200000n_100000nruns_2.5ple.dat')
ax3.plot(data[:,0], data[:,1])
ax3.set_title('girg_200000n+100000nruns uncut')
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set(xlim=(1, 10**4), ylim=(1, 10**5))
ax3.grid(True)


data = np.loadtxt('./output/k-1_zipf_ple25.dat')
ax4.plot(data[:,0], data[:,1])
ax4.set_title('zipf ple2.5')
ax4.set_yscale('log')
ax4.set_xscale('log')
ax4.set(xlim=(1, 10**4), ylim=(1, 10**5))
ax4.grid(True)

fig.tight_layout(pad=3.0)

plt.show()

# print(zipf.rvs(3.5, size=10000))
# print(np.mean(zipf.rvs(3.5, size=10000)))
# print(scipy.special.zeta(2.5) / scipy.special.zeta(3.5))

# print(scipy.stats.chisquare([16, 16, 16, 16, 12, 11], f_exp=[16, 16, 16, 16, 12, 12]))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# from calculate_H import cdf_list2

# class MyRandomVariableClass(stats.rv_discrete):
# 	def __init__(self, moment_tol=1e-8, seed=None):
# 		super().__init__(a=0, moment_tol=moment_tol, seed=seed)
#
# 	def _cdf(self, k):
# 		return cdf_list2[k]
#
#
#
# my_rv = MyRandomVariableClass()
#
# # sample distribution
# samples = my_rv.rvs(size = 1000)
#
# # plot histogram of samples
# fig, ax1 = plt.subplots()
# ax1.hist(list(samples), bins=50)
#
# # plot PDF and CDF of distribution
# pts = np.linspace(0, 5)
# ax2 = ax1.twinx()
# ax2.set_ylim(0,1.1)
# ax2.plot(pts, my_rv.pdf(pts), color='red')
# ax2.plot(pts, my_rv.cdf(pts), color='orange')
#
# fig.tight_layout()
# plt.show()

# z = np.cumsum( [ 0, 1, 2, 6, 9 ] )
# print(z)
# x = np.ediff1d(z, to_begin=z[0])
# print(x)


