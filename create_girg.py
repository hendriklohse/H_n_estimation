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


def degreeDistributionPlot(graph):
	degree_freq = nx.degree_histogram(graph)
	s = sum(degree_freq)
	norm_degree_freq = [float(i)/s for i in degree_freq]
	degrees = range(len(degree_freq))
	plt.plot(degrees, norm_degree_freq,'.', color="black")
	# plt.gca().set_aspect(1/1.61803398875)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('Degree')
	plt.ylabel(r"P(k)")
	plt.savefig("./Figures/IntroductionFigures/youtubeGraphGoldenRatio.png")
	plt.show()
	# deg_tupleList = nx.degree(graph)
	# deg_dict_ = dict((deg_tupleList))
	# deg_dict = deg_dict_ #node:degree
	# inv_deg_dict = {}
	# for k, v in deg_dict.items():
	# 	inv_deg_dict[v] = inv_deg_dict.get(v, []) + [k]
	# degree_distribution_dict = {}
	# for key, value in inv_deg_dict.items():
	# 	degree_distribution_dict[key] = len(value)
	# samples = sorted(degree_distribution_dict.items())
	# plt.plot(*zip(*samples), '.')
	# plt.gca().set_aspect(1/1.61803398875)
	# plt.yscale('log')
	# plt.xscale('log')
	# plt.show()

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
	# samples = sorted(loc_clustering_dict.items())
	# fig = plt.figure(figsize=(5*1.618, 5*1))
	# ax = fig.add_subplot(1, 1, 1)
	# plt.plot(*zip(*samples), '.', color="black")
	# ax.set_yscale('log')
	# ax.set_xscale('log')
	# ax.set_xlabel('Degree')
	# ax.set_ylabel(r"$\gamma_n(k)$")
	# plt.savefig("./Figures/300000n_clusdictPlot.png")
	# plt.show()
	return loc_clustering_dict

def makeBigClus(n_list, ple, nr_graphs):
	big_clusDict = {}
	for n in n_list:
		locClus_sum = {i : 0.0 for i in range(n)}
		for dummy in range(nr_graphs):
			create_girg_(n=str(n), d=1, ple=ple, alpha="inf", deg=10, wseed=12, pseed=130, sseed=1400, threads=1, file="graph_" + str(n), dot=0, edge=1)
			get_input("graph_" + str(n) + ".txt")
			girg = generate_graph(firstTime=True, edgeFile="./input/graph_" + str(n) + ".txt")
			loc_Clus = localClusteringCoefficient(girg)
			# print(loc_Clus)
			for key, value in loc_Clus.items():
				if loc_Clus.get(key):
					locClus_sum[key] += value
		max_key = max(k for k, v in locClus_sum.items() if v != 0.0)
		k_c = int(n**(1/ple))
		# print(max_key)
		# print(locClus_sum)
		locClus_sum = dict(list(locClus_sum.items())[:k_c+1])
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

#
# youtubeGraph = generate_graph(firstTime=False, edgeFile='./Examples/com-youtube.ungraph.txt')
#
# googlePlus = nx.read_edgelist("./Examples/g_plusAnonymized/g_plusAnonymized.csv", delimiter=",") #degree works
# academiaEU = nx.read_edgelist("./Examples/academiaAnonymized/academia2Anonymized.csv", delimiter=",") #gamma works a bit

# wordnet = nx.read_edgelist("./Examples/wordnet-words/wordnet-words.edges", delimiter=",") #works kind of for both

# webgoogle = nx.read_edgelist("./Examples/web-Google.txt",comments="#",delimiter="\t") #definatly works.

# import scipy as sp
# import scipy.io  # for mmread() and mmwrite()
# import io  # Use BytesIO as a stand-in for a Python file object
# fh = io.BytesIO()
# # Read from file
# fh.seek(0)
# H = nx.from_scipy_sparse_matrix(sp.io.mmread(fh))

# hrg = nx.read_edgelist("./Examples/hrg_500000.txt") #ple = 2.5
#
# n = 300000
# girg = generate_graph(firstTime=True, edgeFile="./input/graph_" + str(n) + ".txt")

# degree_list = girg.degree()
# count = 0
# for elt in degree_list:
# 	count += elt[1]
# print(count / len(degree_list))

# pl = nx.powerlaw_cluster_graph(n, 3, 0.5)
# pref_attach = nx.barabasi_albert_graph(n, 3) #n = 500k, m=3
# degreeDistributionPlot(youtubeGraph)
# localClusteringCoefficient(girg)
