import itertools

import numpy as np
import pandas as pd


def calc_d(corr):
    """ Transform a correlation matrix to a distance matrix

    :param corr: correlation matrix
    :return: matrix of distances
    """
    return ((1 - corr) / 2.) ** 0.5


def calc_d_bar(d):
    """ Calculate "d_bar"

    :param d: distance matrix
    :return: the "d_bar" matrix
    """
    d_bar = d * 0
    for i, j in itertools.permutations(d.columns, 2):
        d_bar[i][j] = np.linalg.norm(d[i] - d[j])

    return d_bar


def calc_link_matrix(d_bar):
    """ Calculate the linkage matrix

    :param d_bar: the "d_bar" matrix
    :return: matrix describing the linkages
    """
    n = d_bar.shape[0]
    link_mat = np.zeros((n-1, 4))
    cluster_sizes = {_: 1 for _ in d_bar.columns}
    cluster_indices = dict(zip(d_bar.columns, range(n)))

    for it in range(n - 1):
        col_names = d_bar.columns
        idx = np.tril_indices(d_bar.shape[0], -1)
        min_idx = np.argmin(d_bar.values[idx])
        i_int, j_int = idx[1][min_idx], idx[0][min_idx]
        i, j = col_names[i_int], col_names[j_int]
        min_val = d_bar[i][j]

        new_name = f"({i}, {j})"
        new_col = d_bar[[i, j]].min(axis=1)

        d_bar[new_name] = new_col
        new_col[new_name] = 0
        d_bar = d_bar.append(pd.DataFrame({new_name: new_col}).T)
        d_bar.drop([i, j], axis=0, inplace=True)
        d_bar.drop([i, j], axis=1, inplace=True)

        cluster_sizes[new_name] = cluster_sizes[i] + cluster_sizes[j]
        cluster_indices[new_name] = n + it
        link_mat[it, :] = [cluster_indices[i], cluster_indices[j],
                           min_val, cluster_sizes[new_name]]

    return link_mat


def calc_ordering_index(link_mat, tickers):
    """

    :param link_mat:
    :param tickers:
    :return:
    """

    link_mat = link_mat.astype(int)
    ordering = pd.Series([link_mat[-1, 0], link_mat[-1, 1]])
    n = link_mat[-1, 3]

    while ordering.max() >= n:
        ordering.index = range(0, len(ordering) * 2, 2)
        clusters = ordering[ordering >= n]
        indices = clusters.index
        rows = clusters.values - n
        ordering[indices] = link_mat[rows, 0]
        clusters = pd.Series(link_mat[rows, 1], index=indices + 1)
        ordering = ordering.append(clusters)
        ordering = ordering.sort_index()

    return pd.Series(tickers)[ordering.values]


def split_indices(indices):
    """ Split sets of indices into "left" and "right" halves

    :param indices: list of sets of indices
    :return: list of tuples, each containing the left and right halves of
        a set of indices
    """
    splits = []
    for i in indices:
        if len(i) <= 1:
            continue
        splits.extend([i[0:(len(i) // 2)], i[(len(i) // 2):len(i)]])

    return splits


def calc_cluster_variance(cov, assets):
    """ Calculate the variance of an equally-weighted subset of assets

    :param cov: covariance matrix
    :param assets: subset of assets to include in the portfolio
    :return: a variance value
    """
    sub_cov = cov.loc[assets, assets]
    w = (1. / np.diag(sub_cov)).reshape(-1, 1)
    w /= w.sum()
    return (w.T @ sub_cov @ w)[0][0]


def calc_hrp_weights(cov, ordering):
    """

    :param cov:
    :param ordering:
    :return:
    """
    weights = pd.Series(1, index=ordering)
    indices = [ordering.index]
    while len(indices) > 0:
        indices = split_indices(indices)  # A
        for i in range(0, len(indices), 2):
            i_left, i_right = indices[i], indices[i + 1]
            left_var = calc_cluster_variance(cov, ordering[i_left])  # B
            right_var = calc_cluster_variance(cov, ordering[i_right])  # C
            alpha = left_var / (left_var + right_var)
            weights[ordering[i_left]] *= 1 - alpha
            weights[ordering[i_right]] *= alpha

    return weights


def calc_hrp_portfolio(cov, corr) -> pd.Series:
    """ Calculate the weights for a Hierarchical Risk Parity portfolio

    :param cov: asset-level covariance matrix
    :param corr: asset-level correlation matrix
    :return: weights of the hierarchical risk parity portfolio
    """

    d = calc_d(corr)
    tickers = d.columns.values
    d_bar = calc_d_bar(d)
    link_mat = calc_link_matrix(d_bar)
    ordering = calc_ordering_index(link_mat, tickers)
    weights = calc_hrp_weights(cov, ordering)[tickers]

    return weights
