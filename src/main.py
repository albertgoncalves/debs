#!/usr/bin/env python3

from numpy import append, arange, empty, full, int32, sqrt, unique


class Dbscan:
    def __init__(self, points, epsilon, min_size):
        assert points.ndim == 2
        self.UNDEFINED = -2
        self.NOISE = -1
        self.points = points
        self.epsilon = epsilon
        self.min_size = min_size
        self.n = len(points)
        self.ix = arange(self.n, dtype=int32)

    def get_euclidean_neighbor_indices(self, i):
        delta = self.points - self.points[i]
        distance = sqrt((delta * delta).sum(axis=1))
        return self.ix[distance <= self.epsilon]

    def get_labels(self):
        labels = full(self.n, self.UNDEFINED)
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
                new_seeds = empty(0, dtype=int32)
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


def main():
    from os import environ
    from os.path import join

    from matplotlib.pyplot import close, savefig, subplots, tight_layout
    from seaborn import scatterplot, set_style
    from sklearn.datasets import make_blobs, make_circles, make_moons

    epsilon = 0.2
    min_size = 5

    n = 500
    data = []
    for (i, (points, _)) in enumerate([
        make_blobs(n, centers=4),
        make_circles(n, factor=0.5, noise=0.0375),
        make_moons(n, noise=0.05),
    ]):
        points = (points - points.mean()) / points.std()
        labels = Dbscan(points, epsilon, min_size).get_labels()
        data.append((i, points, labels))

    set_style("dark")
    (_, axs) = subplots(len(data), figsize=(7, 12), dpi=85)
    for (i, points, labels) in data:
        scatterplot(
            x=points[:, 0],
            y=points[:, 1],
            hue=labels,
            palette="Dark2",
            ax=axs[i],
        )
        axs[i].set_aspect("equal")
    tight_layout()
    savefig(join(environ["WD"], "out", "plot.png"))
    close()


if __name__ == "__main__":
    main()
