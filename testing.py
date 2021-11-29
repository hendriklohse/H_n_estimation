import numpy as np
import networkx as nx
import networkit as nk

import matplotlib.pyplot as plt
import scipy

# plt.loglog.plotfile('./output/girg_1D_49999n_500000nruns_2.5ple.dat', delimiter=' ', cols=(0, 1),
# 			 names=('k', 'H_n(k)'), marker='o')
# plt.show()
#
# plt.plotfile('./output/girg_1D_50000n_500000nruns_2.5ple.dat', delimiter=' ', cols=(0, 1),
# 			 names=('k_faster', 'H_n(k)_faster'), marker='o')
# plt.show()

# print(scipy.stats.chisquare([16, 16, 16, 16, 12, 11], f_exp=[16, 16, 16, 16, 12, 12]))

print(scipy.special.gamma(4))
print(scipy.special.gamma(-4))