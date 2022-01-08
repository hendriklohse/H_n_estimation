import os
import numpy as np
# from get_input import inputMaker
from subprocess import Popen, PIPE

import tail_estimation


def run_tail_estimation(dimension, n, nruns, ple, sample, k_c, testing):
	if testing == 1:
		outputFile = "girg_" + str(dimension) + "D_" + str(n) + "n_" + str(nruns) + "nruns_" + str(ple) + "ple" + ".dat"
		#print('/output/' + outputFile)
		plotFile = "girg_" + str(dimension) + "D_" + str(n) + "n_" + str(nruns) + "nruns_" + str(ple) + "ple" + ".pdf"
		with open('./output/k-3/' + outputFile, 'w') as f:
			for duo in sample:
				f.write(str(duo[0]) + " " + str(duo[1]) + "\n")
		# + '--pnoise 1' + ' --noise 1'
		command = 'python tail_estimation.py ./output/k-3/' + outputFile + ' ./plots/k-3/' + plotFile
		os.system(command)
		# print('end os.system(command)')
		# stream = os.popen(command)
		# print('end os.popen(command)')
		# output = stream.read()
		# print('end stream.read()')
		# print(output)

		# stdout = Popen(command, shell=True, stdout=PIPE).stdout
		# output = stdout.read()
		# print(output)
		# print(type(output))

		# print('output printed, now loading res_dict')
		res_dict_read = np.load('./result_dicts/res_dict.npy', allow_pickle=True).item()
		# print(type(res_dict_read))
		return res_dict_read

	elif testing == 0:
		with open('./output/' + "realClusteringFunction_alpha0.8_nu1_samples.dat", 'w') as f:
			for duo in sample:
				f.write(str(duo[0]) + " " + str(duo[1]) + "\n")
		command = 'python tail_estimation.py ./output/' + "realClusteringFunction_alpha0.8_nu1_samples.dat" + ' ./plots/' + "realClusteringPlots"
		os.system(command)
		stream = os.popen(command)
		output = stream.read()
		print(output)
	elif testing == 2:
		with open('./output/' + 'k-1_zipf_ple25.dat', 'w') as f:
			for duo in sample:
				f.write(str(duo[0]) + " " + str(duo[1]) + "\n")
		command = 'python tail_estimation.py ./output/' + 'k-1_zipf_ple25.dat' + ' ./plots/' + "k-1_zipf_ple25Plots"
		os.system(command)
		stream = os.popen(command)
		output = stream.read()
		print(output)



