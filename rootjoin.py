import numpy as np
import math
from heapq import heappush, heappushpop, heapify

class Res:
    def __init__(self, obj, dist):
        self.dist = dist
        self.obj = obj

    def __lt__(self, other):
        return self.dist > other.dist
        
    def __lte__(self, other):
        return self.dist >= other.dist

    def __eq__(self, other):
        return self.dist == other.dist and self.obj == other.obj

    def __str__(self):
        return str(self.obj) + "; " + str(self.dist)
    
class Group:
    def __init__(self, mask, center, id, r=0):
        self.mask = mask
        self.center = center
        self.id = id
        self.r = r
    
    def add(self, i, r):
        self.mask[i] = True
        if r > self.r:
            self.r = r
    
    def size(self):
        return np.sum(self.mask)
    
def get_centers(data, c):
    h, _ = data.shape
    c_idx = np.random.choice(h, size=int(np.floor(np.sqrt(len(data)))), replace=False)
    c_mask = np.zeros(h, dtype=bool)
    c_mask[c_idx] = True
    return c_mask

def make_groups(data, c_mask, k, c, results):
    groups = [Group(np.zeros(len(data), dtype=bool), c, i) for i, c in enumerate(data[c_mask])]
    max_size= c*np.sqrt(len(data))
    for row in data[~c_mask]:
        D = np.sum(np.abs(data[c_mask,1:]-row[1:]), axis=1)
        for i, cent in enumerate(data[c_mask]):
            r = D[i]
            if len(results[cent[0]]) < k:
                heappush(results[cent[0]], Res(int(row[0]), r))
            elif results[cent[0]][0].dist > r:
                heappushpop(results[cent[0]], Res(int(row[0]), r))
            if len(results[row[0]]) < k :
                heappush(results[row[0]], Res(int(cent[0]), r))
            elif results[row[0]][0].dist > r:
                heappushpop(results[row[0]], Res(int(cent[0]), r))
        j = 0
        while True:
            closest_group = np.argpartition(D, j)[j]
            if np.sum(groups[closest_group].size()) < max_size:
                groups[closest_group].add(int(row[0]), r)
                break
            j += 1
    return groups

def rootjoin(data, k, c):
    idx = np.arange(len(data)).reshape(len(data), 1)
    data = np.hstack((idx, data))
    results = {x[0]: [] for x in data}
    c_mask = get_centers(data, c)
    groups = make_groups(data, c_mask, k, c, results)
    R = np.array([group.r for group in groups])
    for group in groups:
        for row in data[~c_mask & group.mask]:
            dist_to_groups = np.sum(np.abs(data[c_mask, 1:] - row[1:]), axis=1) - R
            j = 0
            other_groups = []
            nb_of_candidates = group.size()-1
            while True:
                closest = np.argpartition(dist_to_groups, j)[j]
                if groups[closest].id != group.id:
                    other_groups.append(groups[closest])
                    nb_of_candidates += groups[closest].size()
                    if nb_of_candidates >= k:
                        break
                j += 1
            target = group.mask
            for g in other_groups:
                target = target | g.mask
            distances = np.sum(np.abs(data[target, 1:] - row[1:]), axis=1)
            idx = np.argpartition(distances, k+1)[:k+1]
            knn = data[target][idx]
            for d, e in enumerate(knn):
                if int(e[0])==int(row[0]):
                    continue
                if len(results[row[0]])< k:
                    heappush(results[row[0]], Res(int(e[0]), distances[idx][d]))
                elif distances[idx][d] < results[row[0]][0].dist:
                    heappushpop(results[row[0]], Res(int(e[0]), distances[idx][d]))
    return results
