from collections import namedtuple
from typing import List, Tuple, Dict, Union, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.spatial.distance
import scipy.io
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.graph_shortest_path import graph_shortest_path


def make_affinity_matrix(X: np.ndarray, e: float, method: str = 'euclidean') -> np.ndarray:
    """
    Constructs an affinity matrix given a set of points, X, and epsilon value, e.

    The ε-neighborhood graph:
    Here we connect all points whose pairwise distances are smaller than ε.
    As the distances between all connected points are roughly of the same scale (at most ε),
    weighting the edges would not incorporate more information about the data to the graph.
    Hence, the ε-neighborhood graph is usually considered as an unweighted graph.

    :param X: An array of points
    :param e: Threshold value for point similarity
    :param method: Distance measure to use {'euclidean' or 'cityblock'}
    :return: An affinity matrix (array)
    """
    distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, metric=method))
    aff_matrix = (distances < e).astype(int)
    return aff_matrix


def make_weighted_matrix(unweighted_matrix: np.ndarray) -> np.ndarray:
    """Uses pairwise distances to add weightings to unweighted affinity matrix"""
    distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(unweighted_matrix))
    return distances * unweighted_matrix


def make_distance_matrix(weighted_matrix: np.ndarray, inf: float = 1e7) -> np.ndarray:
    """
    Uses a weighted matrix to create a matrix of distances between pairs of nodes.
    Replaces any A_ij=0 (representing no connection between nodes) with infinity (or proxy).

    :param weighted_matrix: An affinity matrix, weighted by pairwise distance
    :param inf: Value to assign nodes with no connection
    :return: An array of distances between node pairs
    """
    distance_matrix = np.where(weighted_matrix > 0, weighted_matrix, inf)
    reverse_diags = np.nan_to_num(np.identity(weighted_matrix.shape[0]) * inf)
    return distance_matrix - reverse_diags


def get_shortest_paths(weighted_matrix: np.ndarray, inf: float = 1e6) -> np.ndarray:
    """Perform a shortest-path graph search on a positive directed or undirected graph."""
    return graph_shortest_path(make_distance_matrix(weighted_matrix, inf))


def make_centering_matrix(graph: np.ndarray) -> np.ndarray:
    """Creates a centering matrix"""
    identity = np.identity(graph.shape[0])
    ones = np.ones(graph.shape[0])
    return identity - ((1 / graph.shape[0]) * ones * ones.T)


def make_tau_matrix(graph: np.ndarray) -> np.ndarray:
    """Uses a centering matrix to create a tau matrix"""
    centering = make_centering_matrix(graph)
    return (-1 / 2) * centering * (graph ** 2) * centering


def get_eigvals_eigvecs(matrix: np.ndarray, ascending: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Returns sorted eigenvalues and eigenvectors"""
    eig_vals, eig_vecs = scipy.linalg.eig(matrix)
    if ascending:
        sort_idx = np.argsort(eig_vals)
    else:
        sort_idx = np.argsort(eig_vals)[::-1]
    sorted_vecs = eig_vecs[:, sort_idx]
    sorted_vals = eig_vals[sort_idx]
    return sorted_vals.real, sorted_vecs.real


def get_projected_data(eigvals: np.ndarray, eigvecs: np.ndarray, dims: int = 2) -> np.ndarray:
    """"""
    # return (np.sqrt(eigvals[:dims].reshape(-1, 1)) * eigvecs[eigvecs[:, :dims]]).T
    return (np.sqrt(np.diag(eigvals[:dims])) @ eigvecs[:, :dims].T).T


def get_edges_weights(distance_matrix: np.ndarray) -> \
        List[Tuple[Union[int, Dict[str, Any]], ...]]:
    """
    Creates an iterable of weighted edges, given an array of shortest paths.
    Used for `add_edges_from()` method from NetworkX library.

    :param distance_matrix: An array of shortest paths between nodes.
    :return: A list of tuples [(X, Y {'weight': float}), ..]
    """
    edges = []
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            edges.append(tuple([i + 1, j + 1, {'weight': distance_matrix[i, j]}]))
    return edges


def annotate_points_with_images(ax: plt.axes, indices: Union[List[int], np.ndarray],
                                plot_data: np.ndarray, image_data: np.ndarray,
                                image_size: Tuple[int, int],
                                xybox: Tuple[float, float] = (-40., 40.)) -> None:
    """
    Annotates a plot of points with converted image data.

    :param ax: A matplotlib axes object
    :param indices: A list of indices to index image data and plot data.
    :param plot_data: A 2D projection of image data.
    :param image_data: Original image data, where each row corresponds to an array of image
    data.
    :param image_size: A tuple representing image size in pixels.
    :return: None
    """
    for i in indices:
        ab = AnnotationBbox(
            OffsetImage(
                image_data[i, :].reshape(image_size[0], image_size[1]),
                zoom=.8,
                cmap='gray'
            ),
            tuple(plot_data[i, :]),
            xybox=xybox,
            xycoords='data',
            boxcoords="offset points",
            pad=0.3,
            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)


if __name__ == '__main__':
    contents = scipy.io.loadmat('homework2/data/isomap.mat')
    data = contents['images'].T

    for eps in np.linspace(22, 23, 10):
        affinity_matrix = make_affinity_matrix(data, eps)
        print(f'Epsilon={eps}; '
              f'Clusters under 100={(affinity_matrix.sum(axis=1) < 100).sum()},'
              f'Avg cluster size={np.mean(affinity_matrix.sum(axis=1))}')

    A = make_affinity_matrix(data, e=22.5)
    weighted_A = make_weighted_matrix(A)
    distances = make_distance_matrix(weighted_A)

    # --- Using NetworkX --- #
    G = nx.Graph()
    edges = get_edges_weights(distances)
    G.add_edges_from(edges)

    nx.draw(G)
    nx.draw_networkx(G, node_size=25, edge_color='white', with_labels=False)

    # --- Exporting to CSV --- #
    Edge = namedtuple('Edge', ['source', 'target', 'weight'])
    edges = []
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            edges.append(Edge(i + 1, j + 1, distances[i, j]))
    edge_df = pd.DataFrame(edges)

    # --- Projecting data into 2 dimensions via PCA--- #
    graph = graph_shortest_path(distances)
    sc = StandardScaler()
    pc = PCA(2)
    projected = pc.fit_transform(sc.fit_transform(graph))
    plt.scatter(projected[:, 0], projected[:, 1], s=5, alpha=.5)

    # --- Showing numpy array as image --- #
    img1 = data[0, :].reshape(64, 64)
    plt.imshow(img1, cmap='gray')

    # --- Full ISOMAP --- #
    A = make_affinity_matrix(data, e=22.5)
    weighted_A = make_weighted_matrix(A)
    distances = make_distance_matrix(weighted_A)
    graph = graph_shortest_path(distances)
    tau = make_tau_matrix(graph)
    eigvals, eigvecs = get_eigvals_eigvecs(tau, ascending=False)
    projected = get_projected_data(eigvals, eigvecs)

    sorted = pd.DataFrame(projected).sort_values(0)
    # --- Plotting similarity matrix (shortest paths) projected into 2D --- #
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(projected[:, 0], projected[:, 1], s=5, alpha=.5)
    indices = [105, 175, 147]
    offset = 0
    for i in indices:
        ab = AnnotationBbox(
            OffsetImage(
                data[i, :].reshape(64, 64),
                zoom=.8,
                cmap='gray'
            ),
            tuple(projected[i, :]),
            xybox=(0. - offset, 50.),
            xycoords='data',
            boxcoords="offset points",
            pad=0.3,
            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)
        offset += 50
    ax.set_title('Similarity Matrix, Projected into 2D, with image overlay')

    # --- Using citblock --- #
    A = make_affinity_matrix(data, e=1012, method='cityblock')
    weighted_A = make_weighted_matrix(A)
    distances = make_distance_matrix(weighted_A)
    graph = graph_shortest_path(distances)
    tau = make_tau_matrix(graph)
    eigvals, eigvecs = get_eigvals_eigvecs(tau, ascending=False)
    projected = get_projected_data(eigvals, eigvecs)

    sorted = pd.DataFrame(projected).sort_values(0)
    # --- Plotting similarity matrix (shortest paths) projected into 2D --- #
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(projected[:, 0], projected[:, 1], s=5, alpha=.5)
    indices = [361, 442, 193]
    offset = 0
    for i in indices:
        ab = AnnotationBbox(
            OffsetImage(
                data[i, :].reshape(64, 64),
                zoom=.8,
                cmap='gray'
            ),
            tuple(projected[i, :]),
            xybox=(0. - offset, 50.),
            xycoords='data',
            boxcoords="offset points",
            pad=0.3,
            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)
        offset += 50
    ax.set_title('Similarity Matrix, Projected into 2D, with image overlay')

