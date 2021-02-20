#!/usr/bin/env python3

from os import environ
from os.path import join

from matplotlib.pyplot import close, savefig, subplots, tight_layout
from seaborn import scatterplot, set_style
from sklearn.datasets import make_blobs, make_circles, make_moons

from lib import Dbscan


def main():
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
