import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import poisson
import sys
import time
import argparse
import os
import warnings
from matplotlib import pyplot as plt
from collections import Counter
from collections import OrderedDict


from create_girg_faster import localClusteringCoefficient, generate_graph
from calculate_gamma_faster import create_PMF
from sampling_H_faster import sample_H
from get_input import get_input, create_girg_
from retreive_xi import run_tail_estimation
# from getEstimates import get_estimates

# from sampling_H import simulate3
# from calculate_gamma import create_pmf_slow

# dic_real = {}
# with open("./Examples/realClusteringFunction_alpha0.8_nu1.dat", 'r') as f:
# 	lines = f.readlines()
# 	for line in lines:
# 		fields = line.split("\t")
# 		dic_real[int(fields[0])] = float(fields[1].strip())
# print(dic_real)

def run_all(n, nruns, ple):
	create_girg_(n=str(n), d=1, ple=ple, alpha="inf", deg=10, wseed=12, pseed=130, sseed=1400, threads=1, file="graph_" + str(n), dot=0, edge=1)
	get_input("graph_" + str(n) + ".txt")
	#
	girg = generate_graph(firstTime=True, edgeFile="./input/graph_" + str(n) + ".txt")
	locClus = localClusteringCoefficient(girg)
	# # locClus = dic_real
	#
	# # H_n = create_pmf_slow(clusDict=locClus, dict_length=len(locClus.keys())) #returns a list H_n
	H_n = create_PMF(clus_dict=locClus, dict_length=len(locClus.keys()), n=n, ple=ple)
	#
	# # sample_ = simulate3(nruns=nruns, probability_list=H_n)
	sample_ = sample_H(dist=H_n, nruns=nruns)

	estimates_dict = run_tail_estimation(dimension=1, n=n, nruns=nruns, ple=2.5, sample=sample_, testing=False)
	print(estimates_dict)
	for key, value in estimates_dict.items():
		estimates_dict[key] = abs(value - ple) if value != None else None
	# get_estimates(dimension=1, n=n, nruns=nruns, ple=ple, sample=sample_)
	print(estimates_dict)
	return estimates_dict

run_all(n=100000, nruns=100000, ple=2.5)

# create_girg_(n=50000, d=1, ple=2.5, alpha="inf", deg=10, wseed=12, pseed=130, sseed=1400, threads=1, file="graph_" + str(50000), dot=0, edge=1)