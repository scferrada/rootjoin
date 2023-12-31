import rootjoin as rj
import numpy as np
import argparse, os

parser = argparse.ArgumentParser(description='Runs the approximated self similarity join (kNN) algorithm of a given ser of points several times')

parser.add_argument('input_matrix', type=str, help='The numpy vector storing file')
parser.add_argument('output_folder', type=str, help='The directory where the results must be stored')
#parser.add_argument('--N', dest='iter', type=int, default=1, help='The number of times the experiment must be repeated. 1 by default.')
#parser.add_argument('--k', dest='k', type=int, default=1, help='The number of nearest neighbors to retrieve. 1 by default.')

N = 1000

args = parser.parse_args()
data = np.load(args.input_matrix)#np.random.rand(N, 2) #
#np.save("out/random.npy", data)

import time

for c in [2,3,5,7,10]:
	for k in [1, 4, 8, 16, 32]:
		print(c, k)
		for i in range(10):
			print("running %d experiment "%i)
			start = time.time()
			results = rj.rootjoin(data, k, c)
			end = time.time()
			f1 = open(os.path.join(args.output_folder, str(c), str(k), "%d.time"%i), "w")
			f1.write(end-start)
			f1.close()
			f = open(os.path.join(args.output_folder, str(c), str(k), "%d.res"%i), "w")
			for x in results:
				f.write("%d,%s\n"%(x, ",".join([str(y.obj) for y in results[x]])))
			f.close()