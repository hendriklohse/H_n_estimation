import numpy as np
# import networkx as nx
# import networkit as nk
# from collections import Counter
import matplotlib.pyplot as plt
# import scipy
#
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

data = np.loadtxt('./output/k-3/girg_1D_100000n_1000000nruns_2.5ple.dat')
ax2.plot(data[:,0], data[:,1])
ax2.set_title('k-3 girg_1D_100000n_1000000nruns_2.5ple.dat')
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set(xlim=(1, 10**4), ylim=(1, 10**5))
ax2.grid(True)

data = np.loadtxt('./output/k-3/girg_1D_500000n_1000000nruns_2.5ple.dat')
ax3.plot(data[:,0], data[:,1])
ax3.set_title('k-3 girg_1D_500000n_1000000nruns_2.5ple.dat')
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
#
#
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
#
# N_step = 1
# N = 10
# Nruns_step = 10
# Nruns = 100
#
# rows = [i for i in range(N_step,N + N_step,N_step)] #n * n_step
# cols = [i for i in range(Nruns_step,Nruns + Nruns_step,Nruns_step)] #nruns * nruns_step
#
# df1 = pd.DataFrame(2*np.random.random((10,10,)), rows, cols)
# df2 = pd.DataFrame(np.random.random((10,10,)), rows, cols)
# df3 = pd.DataFrame(np.random.random((10,10,)), rows, cols)
#
#
# fig, axn = plt.subplots(nrows=1, ncols=3, figsize=(15,6))
# fig.tight_layout(rect=[0.025, 0.025, .9, .9]) #where the 3 heatmaps are
# cbar_ax = fig.add_axes([.91, .3, .03, .4]) #dimensions: [left, bottom, width, height] in fractions of figure width and height.
#
# for i, ax in enumerate(axn.flat):
# 	if i == 0:
# 		df = df1
# 		ax.set_title("df1")
# 	elif i == 1:
# 		df = df2
# 		ax.set_title("df2")
# 	elif i == 2:
# 		df = df3
# 		ax.set_title("df3")
# 	sns.heatmap(df, ax=ax,
# 				vmin= 0,
# 				cmap= sns.color_palette("Reds"),
# 				xticklabels= True, yticklabels=True,
# 				annot = True,
# 				cbar= i == 0,
# 				cbar_kws={'label': 'estimator error'},
# 				cbar_ax=None if i else cbar_ax)
#
# ple = 2.5
# plt.setp(axn, xlabel='Nruns')
# plt.setp(axn[0], ylabel='N')
# plt.suptitle('Estimator performance for ple = {}'.format(str(ple)), weight="bold", size= 'x-large')
# # plt.savefig('./Figures/k-3/Heatmap')
# plt.show()

