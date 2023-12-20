import argparse, os, math
import numpy as np 
from heapq import heapify, heappop

parser = argparse.ArgumentParser(description='Computes self similarity join (kNN) of a given ser of points')

parser.add_argument('input_numpy', type=str, help='the numpy vectors')
parser.add_argument('output_folder', type=str, help='the directory where the results must be stored')
#parser.add_argument('--batch', dest='batch', type=int, default='100', help='the percentage of data to be used')

args = parser.parse_args()

class Res:
	def __init__(self, obj, dist):
		self.dist = -dist
		self.obj = obj

	def __lt__(self, other):
		return self.dist < other.dist

	def __eq__(self, other):
		return self.dist == other.dist and self.obj == other.obj

	def __str__(self):
		return str(self.obj) + "; " + str(self.dist)

k = 32
dataset = np.load(args.input_numpy)

print("starting bruteforce for %d" % dataset.shape[0])
with open(os.path.join(args.output_folder, "random.csv"), "w") as outfile:	
	idx = np.arange(len(dataset)).reshape(len(dataset), 1)
	matrix = np.hstack((idx, dataset))	
	D = np.abs((matrix[:,1:,None]-matrix[:,1:,None].T)).sum(1)
	idx_knn = np.argpartition(D,k, axis=1)[:, :k+1]
	for j, el in enumerate(matrix):
		knn_dist = [Res(int(nn), d) for nn, d in zip(idx_knn[j], D[j, idx_knn[j]])]
		heapify(knn_dist)
		sorted_knn = []
		while len(knn_dist) > 0 :
			x = heappop(knn_dist)
			if x.obj!=el[0]:
				sorted_knn.append(x.obj)
		txt = "%d,%s\n" % (el[0], sorted_knn)
		outfile.write(txt)
		