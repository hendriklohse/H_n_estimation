import networkit as nk
import pandas as pd
import time

def makeEdgeFile(edgeFile):
	"""Deletes first row from edge file"""

	lines = open(str(edgeFile)).readlines()
	del lines[0]
	ef_new = open(edgeFile, "w+")
	for line in lines:
		ef_new.write(line)
	ef_new.close()

def make_NK_Graph(firstTime, edgeFile):
	""""""
	if firstTime:
		makeEdgeFile(edgeFile)
		edgeList = edgeFile
	else:
		edgeList = edgeFile
	t1 = time.time()
	graph = nk.readGraph(edgeList, nk.Format.EdgeListSpaceOne)
	t2 = time.time()
	print("time to generate nk graph: " + str(t2-t1))
	return graph

def localClustering_NK(graph):
	t1 = time.time()
	maxNodeID = graph.upperNodeIdBound()
	deg_dict = {i : graph.degree(i) for i in range(0, maxNodeID)}
	print(deg_dict)
	inv_deg_dict = {}
	for k, v in deg_dict.items():
		inv_deg_dict[v] = inv_deg_dict.get(v, []) + [k]
	print(inv_deg_dict)
	lcc = nk.centrality.LocalClusteringCoefficient(graph)
	print(lcc.run())
	# loc_clustering_dict = {}
	# for key, value in inv_deg_dict.items():
	# 	total_clustering = 0
	# 	for v in value:
	# 		total_clustering += nk.centrality.LocalClusteringCoefficient(graph, turbo=True)
	# 	avg_clustering = total_clustering / len(value)
	# 	loc_clustering_dict[key] = avg_clustering
	# # print(loc_clustering_dict)
	# t2 = time.time()
	# print("time to calculate loc clus coef: " + str(t2 - t1))
	# return loc_clustering_dict


# graph = nk.readGraph("./graph.txt", nk.Format.EdgeListSpaceOne)


localClustering_NK(make_NK_Graph(firstTime=False, edgeFile="./input/graph_5000.txt"))