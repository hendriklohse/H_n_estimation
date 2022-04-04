import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
import time
from create_girg import makeBigClus
from all_combined import run_all

#res_dict = {"hill_ple": None, "moments_ple": None, "kernel_ple": None}


def fill_data(N, N_step, Nruns ,Nruns_step, est, ple, nr_graphs):
	"""N_step and Nruns_step must divide N and Nruns respectively"""

	rows = [i for i in range(N_step,N + N_step,N_step)] #n * n_step
	cols = [i for i in range(Nruns_step,Nruns + Nruns_step,Nruns_step)] #nruns * nruns_step
	dat_hill = [[[[None] * len(cols) for _ in range(len(rows))] for _ in range(3)] for _ in range(4)] #distance between estimate and real value of ple
	dat_moments = [[[[None] * len(cols) for _ in range(len(rows))] for _ in range(3)] for _ in range(4)] #distance between estimate and real value of ple
	dat_kernel = [[[[None] * len(cols) for _ in range(len(rows))] for _ in range(3)] for _ in range(4)]
	t1 = time.time()
	bigClus = makeBigClus(n_list=rows, ple=ple, nr_graphs=nr_graphs)
	t2 = time.time()
	print("######## makeBigClus done in {} seconds #########".format(t2-t1))

	for n in rows:
		for nruns in cols:
			results = run_all(n=n, nruns=nruns, ple=ple, bigLocClus=bigClus)
			for i in range(4): #
				for l in range(1, 3+1):
					dat_hill[i][l-1][rows.index(n)][cols.index(nruns)] = results[i][l][est[0]]
					dat_moments[i][l-1][rows.index(n)][cols.index(nruns)] = results[i][l][est[1]]
					dat_kernel[i][l-1][rows.index(n)][cols.index(nruns)] = results[i][l][est[2]]

	df_hill_dict = {i : {l : None for l in range(1, 3+1)} for i in range(4)}
	df_moments_dict = {i : {l : None for l in range(1, 3+1)} for i in range(4)}
	df_kernel_dict = {i : {l : None for l in range(1, 3+1)} for i in range(4)}


	for i in range(4):
		for l in range(1, 3+1):
			df_hill_dict[i][l] = pd.DataFrame(dat_hill[i][l-1], rows, cols).iloc[::-1].round(2)
			df_moments_dict[i][l] = pd.DataFrame(dat_moments[i][l-1], rows, cols).iloc[::-1].round(2)
			df_kernel_dict[i][l] = pd.DataFrame(dat_kernel[i][l-1], rows, cols).iloc[::-1].round(2)

			df_hill_dict[i][l].to_csv('./Dataframes/df_hill_ple{}_rows{}cols{}k_c={}labda={}.csv'.format(str(ple), str(len(rows)), str(len(cols)), str(i), str(l)))
			df_moments_dict[i][l].to_csv('./Dataframes/df_moments_ple{}_rows{}cols{}k_c={}labda={}.csv'.format(str(ple), str(len(rows)), str(len(cols)), str(i), str(l)))
			df_kernel_dict[i][l].to_csv('./Dataframes/df_kernel_ple{}_rows{}cols{}k_c={}labda={}.csv'.format(str(ple), str(len(rows)), str(len(cols)), str(i), str(l)))

	print(df_hill_dict)
	print(df_moments_dict)
	print(df_kernel_dict)

	reds = sns.color_palette("Reds", as_cmap=True)
	reds.set_over('k')

	for i in range(4):
		for l in range(1, 3+1):
			fig, axn = plt.subplots(nrows=1, ncols=3, figsize=(15,6))
			fig.tight_layout(rect=[0.025, 0.025, .9, .9]) #where the 3 heatmaps are
			cbar_ax = fig.add_axes([.91, .3, .03, .4]) #dimensions: [left, bottom, width, height] in fractions of figure width and height.

			if l == 1:
				label_labda = ", " + r"$\lambda = 1$"
			elif l == 2:
				label_labda = ", " + r"$\lambda = 2$"
			elif l == 3:
				label_labda = ", " + r"$\lambda = 3$"
			else:
				label_labda = ""
				print("wrong l")

			if i == 0:
				label_k_c = ", " + r"$k_c = n^{1 / (\beta + 1)}$"
			elif i == 1:
				label_k_c = ", " + r"$k_c = n^{1 / (\beta + 0.5)}$"
			elif i == 2:
				label_k_c = ", " + r"$k_c = n^{1 / \beta}$"
			elif i == 3:
				label_k_c = ", " + r"$k_c = k_{max}$"
			else:
				label_k_c = ""
				print("wrong i")

			for j, ax in enumerate(axn.flat):
				if j == 0:
					df = df_hill_dict[i][l]
					ax.set_title("Hill estimator" + label_k_c + label_labda)
				elif j == 1:
					df = df_moments_dict[i][l]
					ax.set_title("Moments estimator" + label_k_c + label_labda)
				elif j == 2:
					df = df_kernel_dict[i][l]
					ax.set_title("Kernel estimator" + label_k_c + label_labda)
				sns.heatmap(df, ax=ax,
							vmin= 0,
							vmax= 1,
							cmap= reds,
							xticklabels= [elt // 100000 for elt in cols], yticklabels=[elt // 100000 for elt in rows][::-1],
							annot = True,
							cbar= j == 0,
							cbar_kws={'label': 'root mean squared error', 'extend' : 'max'},
							cbar_ax=None if j else cbar_ax)

			plt.setp(axn, xlabel='Nruns' + r"$(\cdot 10^5)$")
			plt.setp(axn[0], ylabel='N' + r"$(\cdot 10^5)$")
			plt.suptitle('Estimator performance for '  r"$\beta$ = {}".format(str(ple)), weight="bold", size= 'x-large')
			plt.savefig('./Figures/labdas/Heatmap_ple{}_rows{}cols{}k_c={}labda={}.png'.format(str(ple), str(len(rows)), str(len(cols)), str(i), str(l)))


fill_data(N=2000000, N_step=200000, Nruns=1000000, Nruns_step=100000, est=["hill_ple", "moments_ple", "kernel_ple"], ple=2.5, nr_graphs=5)


