# from tail_estimation import hill_estimator
# from tail_estimation import moments_estimator
# from tail_estimation import kernel_type_estimator


import numpy as np

res_dict_read = np.load('./result_dicts/res_dict.npy',allow_pickle='TRUE').item()


#
# def get_estimates(dimension, n, nruns, ple, sample):
# 	""""""
#
# 	outputFile = "girg_" + str(dimension) + "D_" + str(n) + "n_" + str(nruns) + "nruns_" + str(ple) + "ple" + ".dat"
# 	sequence_file_path = './output/' + outputFile
# 	delimiter = " "
# 	# check for number of entries
# 	N = 0
# 	with open(sequence_file_path, "r") as f:
# 		for line in f:
# 			degree, count = line.strip().split(delimiter)
# 			N += int(count)
# 	with open(sequence_file_path, 'w') as f:
# 		for duo in sample:
# 			f.write(str(duo[0]) + " " + str(duo[1]) + "\n")
#
#
# 	ordered_data = np.zeros(N)
# 	current_index = 0
# 	with open(sequence_file_path, "r") as f:
# 		for line in f:
# 			degree, count = line.strip().split(delimiter)
# 			ordered_data[current_index:current_index + int(count)] = float(degree)
# 			current_index += int(count)
#
# 	hill_xi = hill_estimator(ordered_data=ordered_data)[3]
# 	moments_xi = moments_estimator(ordered_data=ordered_data)[3]
# 	kernel_xi = kernel_type_estimator(ordered_data=ordered_data, hsteps=200)[3]
# 	hill_ple_estimate = 1./hill_xi - 1 if hill_xi != 0 else print("hill_xi = 0")
# 	moments_ple_estimate = 1./moments_xi - 1 if moments_xi != 0 else print("moments_xi = 0")
# 	kernel_ple_estimate = 1./kernel_xi - 1 if kernel_xi != 0 else print("kernel_xi = 0")
# 	print(res_dict)
# 	return {"hill_ple_estimate" : hill_ple_estimate, "moments_ple_estimate" : moments_ple_estimate, "kernel_ple_estimate" : kernel_ple_estimate}