import numpy as np
import pandas as pd


def compute_equilibrium_ers(weights, sigma, risk_aversion):
    return risk_aversion * sigma @ weights


def compute_posterior_mean(mu, C, A, b, omega):
    c_inv = np.linalg.inv(C)
    a_t_oinv = A.T @ np.linalg.inv(omega)

    er = np.linalg.inv(c_inv + a_t_oinv @ A) @ (c_inv @ mu + a_t_oinv @ b)

    return pd.Series(er, index=mu.index)


def compute_posterior_cov(sigma, C, A, omega):
    return sigma + np.linalg.inv((np.linalg.inv(C) + A.T @ omega @ A))
