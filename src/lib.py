#!/usr/bin/env python3

from numpy import append, arange, empty, full, int16, sqrt, unique


class Dbscan:
    def __init__(self, points, epsilon, min_size):
        assert points.ndim == 2
        self.UNDEFINED = -2
        self.NOISE = -1
        self.points = points
        self.epsilon = epsilon
        self.min_size = min_size
        self.n = len(points)
        assert self.n < 0x7FFF
        self.ix = arange(self.n, dtype=int16)

    def get_euclidean_neighbor_indices(self, i):
        delta = self.points - self.points[i]
        distance = sqrt((delta * delta).sum(axis=1))
        return self.ix[distance <= self.epsilon]

    def get_labels(self):
        labels = full(self.n, self.UNDEFINED, dtype=int16)
        cluster = 0
        for i in range(self.n):
            if labels[i] != self.UNDEFINED:
                continue
            neighbors = self.get_euclidean_neighbor_indices(i)
            if len(neighbors) < self.min_size:
                labels[i] = self.NOISE
                continue
            cluster += 1
            labels[i] = cluster
            seeds = neighbors[neighbors != i]
            while len(seeds) != 0:
                new_seeds = empty(0, dtype=int16)
                for j in seeds:
                    if labels[j] == self.NOISE:
                        labels[j] = cluster
                    if labels[j] != self.UNDEFINED:
                        continue
                    labels[j] = cluster
                    neighbors = self.get_euclidean_neighbor_indices(j)
                    if self.min_size <= len(neighbors):
                        new_seeds = unique(append(new_seeds, neighbors))
                seeds = new_seeds
        assert (labels != self.UNDEFINED).all()
        return labels
