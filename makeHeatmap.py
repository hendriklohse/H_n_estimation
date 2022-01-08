import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from all_combined import run_all

#res_dict = {"hill_ple": None, "moments_ple": None, "kernel_ple": None}


def fill_data(N, N_step, Nruns ,Nruns_step, est, ple):
	"""N_step and Nruns_step must divide N and Nruns respectively"""

	rows = [i for i in range(N_step,N + N_step,N_step)] #n * n_step
	cols = [i for i in range(Nruns_step,Nruns + Nruns_step,Nruns_step)] #nruns * nruns_step
	dat_hill = [[None] * len(cols) for _ in range(len(rows))] #distance between estimate and real value of ple
	dat_moments = [[None] * len(cols) for _ in range(len(rows))]
	dat_kernel = [[None] * len(cols) for _ in range(len(rows))]
	nruns = Nruns_step
	n = N_step
	for i in range(len(rows)):
		for j in range(len(cols)):
			results = run_all(n=n, nruns=nruns, ple=ple)
			dat_hill[i][j] = results[est[0]]
			dat_moments[i][j] = results[est[1]]
			dat_kernel[i][j] = results[est[2]]
			n += N_step
			nruns += Nruns_step
	df_hill = pd.DataFrame(dat_hill, rows, cols)
	df_moments = pd.DataFrame(dat_moments, rows, cols)
	df_kernel = pd.DataFrame(dat_kernel, rows, cols)

	print(df_hill)
	print(df_moments)
	print(df_kernel)

fill_data(N=100000, N_step=50000, Nruns=200000, Nruns_step=100000, est=["hill_ple", "moments_ple", "kernel_ple"], ple=2.5)


# ax = sns.heatmap(data=df)