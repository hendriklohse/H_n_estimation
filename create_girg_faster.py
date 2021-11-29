from scipy.spatial.distance import pdist, squareform
import time
from matplotlib import pyplot as plt
import time
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


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
	# print(graph.edges())
	deg_tupleList = nx.degree(graph)
	# print(nx.degree(graph))
	deg_dict_ = dict((deg_tupleList))
	deg_dict = deg_dict_
	#print(deg_dict)
	inv_deg_dict = {}
	for k, v in deg_dict.items():
		inv_deg_dict[v] = inv_deg_dict.get(v, []) + [k]
	#print(inv_deg_dict)
	loc_clustering_dict = {}
	for key, value in inv_deg_dict.items():
		total_clustering = 0
		for v in value:
			total_clustering += nx.clustering(graph, v)
		avg_clustering = total_clustering / len(value)
		loc_clustering_dict[key] = avg_clustering
	# print(loc_clustering_dict)
	t2 = time.time()
	print("time to calculate loc clus coef: " + str(t2 - t1))
	return loc_clustering_dict

