import os
import numpy as np
# from get_input import inputMaker


def run_tail_estimation(dimension, n, nruns, ple, sample, testing):
	if not testing:
		outputFile = "girg_" + str(dimension) + "D_" + str(n) + "n_" + str(nruns) + "nruns_" + str(ple) + "ple" + ".dat"
		#print('/output/' + outputFile)
		plotFile = "./plots" + "girg_" + str(dimension) + "D_" + str(n) + "n_" + str(nruns) + "nruns_" + str(ple) + "ple" + ".pdf"
		with open('./output/cut/' + outputFile, 'w') as f:
			for duo in sample:
				f.write(str(duo[0]) + " " + str(duo[1]) + "\n")

		command = 'python tail_estimation.py ./output/cut/' + outputFile + ' ./plots/' + plotFile
		os.system(command)
		stream = os.popen(command)
		output = stream.read()
		print(output)
		res_dict_read = np.load('./result_dicts/res_dict.npy',allow_pickle='TRUE').item()
		return res_dict_read
	else:
		with open('./output/' + "realClusteringFunction_alpha0.8_nu1_samples.dat", 'w') as f:
			for duo in sample:
				f.write(str(duo[0]) + " " + str(duo[1]) + "\n")
		command = 'python tail_estimation.py ./output/' + "realClusteringFunction_alpha0.8_nu1_samples.dat" + ' ./plots/' + "realClusteringPlots"
		os.system(command)
		stream = os.popen(command)
		output = stream.read()
		print(output)




