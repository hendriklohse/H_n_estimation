import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
import time
from create_girg import makeBigClus
from all_combined import run_all

#res_dict = {"hill_ple": None, "moments_ple": None, "kernel_ple": None}


def fill_data(N, N_step, Nruns ,Nruns_step, est, ple):
	"""N_step and Nruns_step must divide N and Nruns respectively"""

	rows = [i for i in range(N_step,N + N_step,N_step)] #n * n_step
	cols = [i for i in range(Nruns_step,Nruns + Nruns_step,Nruns_step)] #nruns * nruns_step
	dat_hill = [[None] * len(cols) for _ in range(len(rows))] #distance between estimate and real value of ple
	dat_moments = [[None] * len(cols) for _ in range(len(rows))]
	dat_kernel = [[None] * len(cols) for _ in range(len(rows))]

	t1 = time.time()
	bigClus = makeBigClus(n_list=rows, ple=ple)
	t2 = time.time()
	print("######## makeBigClus done in {} seconds #########".format(t2-t1))
	for n in rows:
		for nruns in cols:
			results = run_all(n=n, nruns=nruns, ple=ple, bigLocClus=bigClus)
			dat_hill[rows.index(n)][cols.index(nruns)] = results[est[0]]
			dat_moments[rows.index(n)][cols.index(nruns)] = results[est[1]]
			dat_kernel[rows.index(n)][cols.index(nruns)] = results[est[2]]


	df_hill = pd.DataFrame(dat_hill, rows, cols)
	df_moments = pd.DataFrame(dat_moments, rows, cols)
	df_kernel = pd.DataFrame(dat_kernel, rows, cols)

	df_hill.to_csv('./Dataframes/df_hill_ple{}_rows{}cols{}.csv'.format(str(ple), len(rows), len(cols)))
	df_moments.to_csv('./Dataframes/df_moments_ple{}_rows{}cols{}.csv'.format(str(ple), len(rows), len(cols)))
	df_kernel.to_csv('./Dataframes/df_kernel_ple{}_rows{}cols{}.csv'.format(str(ple), len(rows), len(cols)))

	print(df_hill)
	print(df_moments)
	print(df_kernel)

	fig, axn = plt.subplots(nrows=1, ncols=3, figsize=(15,6))
	fig.tight_layout(rect=[0.025, 0.025, .9, .9]) #where the 3 heatmaps are
	cbar_ax = fig.add_axes([.91, .3, .03, .4]) #dimensions: [left, bottom, width, height] in fractions of figure width and height.

	for i, ax in enumerate(axn.flat):
		if i == 0:
			df = df_hill
			ax.set_title("Hill estimator")
		elif i == 1:
			df = df_moments
			ax.set_title("Moments estimator")
		elif i == 2:
			df = df_kernel
			ax.set_title("Kernel estimator")
		sns.heatmap(df, ax=ax,
					vmin= 0,
					cmap= sns.color_palette("Reds"),
					xticklabels= True, yticklabels=True,
					annot = True,
					cbar= i == 0,
					cbar_kws={'label': 'estimator error'},
					cbar_ax=None if i else cbar_ax)


	plt.setp(axn, xlabel='Nruns')
	plt.setp(axn[0], ylabel='N')
	plt.suptitle('Estimator performance for ple = {}'.format(str(ple)), weight="bold", size= 'x-large')
	plt.savefig('./Figures/k-3/Heatmap_ple{}_rows{}cols{}.csv'.format(str(ple), len(rows), len(cols)))
	plt.show()


fill_data(N=1000000, N_step=500000, Nruns=1000000, Nruns_step=500000, est=["hill_ple", "moments_ple", "kernel_ple"], ple=2.5)


