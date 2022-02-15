from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
import time
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zipf
from collections import Counter
from get_input import get_input, create_girg_

def makeEdgeFile(edgeFile):
	"""Deletes first row from edge file"""

	lines = open(str(edgeFile)).readlines()
	del lines[0]
	ef_new = open(edgeFile, "w+")
	for line in lines:
		ef_new.write(line)
	ef_new.close()

def generate_graph(firstTime, edgeFile):
	if firstTime:
		makeEdgeFile(edgeFile)
		edgeList = edgeFile
	else:
		edgeList = edgeFile
	t1 = time.time()
	graph = nx.read_edgelist(edgeList)
	t2 = time.time()
	print("time to generate nx graph: " + str(t2-t1))
	return graph

def localClusteringCoefficient(graph):
	t1 = time.time()
	deg_tupleList = nx.degree(graph)
	deg_dict_ = dict((deg_tupleList))
	deg_dict = deg_dict_
	inv_deg_dict = {}
	for k, v in deg_dict.items():
		inv_deg_dict[v] = inv_deg_dict.get(v, []) + [k]
	loc_clustering_dict = {}
	for key, value in inv_deg_dict.items():
		total_clustering = 0
		for v in value:
			total_clustering += nx.clustering(graph, v)
		avg_clustering = total_clustering / len(value)
		loc_clustering_dict[key] = avg_clustering
	t2 = time.time()
	print("time to calculate loc clus coef: " + str(t2 - t1))
	# np.save('./clusteringCoefficients/big_clusDict.npy', big_clusDict)
	return loc_clustering_dict

def makeBigClus(n_list, ple, nr_graphs):
	big_clusDict = {}
	for n in n_list:
		locClus_sum = {i : 0.0 for i in range(n)}
		for _ in range(nr_graphs):
			create_girg_(n=str(n), d=1, ple=ple, alpha="inf", deg=10, wseed=12, pseed=130, sseed=1400, threads=1, file="graph_" + str(n), dot=0, edge=1)
			get_input("graph_" + str(n) + ".txt")
			girg = generate_graph(firstTime=True, edgeFile="./input/graph_" + str(n) + ".txt")
			loc_Clus = localClusteringCoefficient(girg)
			# print(loc_Clus)
			for key, value in loc_Clus.items():
				if loc_Clus.get(key):
					locClus_sum[key] += value
		max_key = max(k for k, v in locClus_sum.items() if v != 0.0)
		# print(max_key)
		# print(locClus_sum)
		locClus_sum = dict(list(locClus_sum.items())[:max_key+1])
		# print(locClus_sum)
		locClus = {k: v / nr_graphs for k, v in locClus_sum.items()}
		# print(locClus)
		big_clusDict[n] = locClus
	return big_clusDict

def zipfClustering(a, n):
	res = zipf.rvs(a, size=n) #a is pmf coefficient
	samples = sorted(Counter(res).items()) #list of tuples
	dic = dict()
	dic[0] = 0.0
	dic[1] = 0.0
	for key, value in samples:
		key = key + 1
		dic[key] = value / n
	# print(dic)
	return dic

# zipfClustering(2.5, 100000)
