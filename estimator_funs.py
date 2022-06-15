import numpy as np
from scipy.stats import norm
from scipy.stats.distributions import chi2


# Exact formula used for case: n=1
def eu_exact(K, r, s0, sigma):
    d1 = 1 / sigma * (np.log(s0 / K) + r + sigma ** 2 / 2)
    d2 = d1 - sigma
    res = s0 * norm(0, 1).cdf(d1) - K * np.exp(-r) * norm(0, 1).cdf(d2)
    return res * np.exp(-r)


# Geometric Brownian motion
def gbm(x, s0, r, sigma):
    mis = r - sigma ** 2 / 2
    st = s0 * np.exp(mis + sigma * x)
    return st


# Creating covariance matrix
def create_sigma(n):
    sigma = np.full((n, n), 0)
    for i in np.arange(n):
        for j in np.arange(n):
            sigma[i, j] = np.minimum(i, j) + 1

    sigma = sigma / n
    return sigma


# CMC estimator
#n=1
def eu_option_cmc(K, r, s0, sigma, R, seed=2022):
    np.random.seed(seed)
    x = np.random.normal(0, 1, R)
    An = gbm(x, s0, r, sigma)
    Yk = np.maximum(An - K, 0)
    Yhat = np.mean(Yk) * np.exp(-r)
    return Yhat


# general n
def eu_option_cmc_n(K, r, s0, sigma, R, n, seed=2022):
    np.random.seed(seed)
    u = np.random.random(int(R))
    D = np.sqrt(chi2.ppf(u, df=n))
    ksi = np.random.multivariate_normal(mean=np.zeros(n), cov=np.identity(n), size=int(R))
    ksi_len = np.sqrt(np.sum(ksi ** 2, axis=1))
    X = (ksi.T / ksi_len).T
    Z = (X.T * D).T
    sigma_matrix = create_sigma(n)
    A = np.linalg.cholesky(sigma_matrix)
    B = np.dot(A, Z.T).T
    St = gbm(B, s0, r, sigma)
    An = np.sum(St, axis=1) / n
    Yk = np.maximum(An - K, 0)
    Yhat = np.mean(Yk) * np.exp(-r)
    return Yhat


# Antithetic estimator
def eu_option_ant(K, r, s0, sigma, R, method='standard', seed=2022):
    if R % 2 != 0:
        raise ValueError('Value of R must be even!')
    np.random.seed(seed)
    x = np.zeros(R)
    # generating antithetic variables
    if method == "standard":
        z = np.random.normal(0, 1, int(R / 2))
        x[range(0, R, 2)] = z
        x[range(1, R + 1, 2)] = -z
    elif method == "inverse":
        u = np.random.uniform(0, 1, int(R / 2))
        x[range(0, R, 2)] = u
        x[range(1, R + 1, 2)] = 1 - u
        x = norm(0, 1).ppf(x)
    else:
        raise ValueError('method should be either "standard" or "inverse"')
    An = gbm(x, s0, r, sigma)
    Yk = np.maximum(An - K, 0)
    Yhat = np.mean(Yk) * np.exp(-r)
    return Yhat


# Control variate estimator
def eu_option_cv(K, r, s0, sigma, R, seed=2022):
    np.random.seed(seed)
    mis = r - sigma ** 2 / 2
    y = np.random.normal(0, 1, R)
    x = np.random.normal(0, 1, R)
    # Calculating c(from script)
    c = - np.cov(x, y)[0, 1] / np.var(x)
    # Calculating Y^CMC
    An = gbm(x, s0, r, sigma)
    Yk = np.maximum(An - K, 0)
    Yhat = np.mean(Yk) * np.exp(-r) + c * np.mean(x)

    return Yhat


# Stratified estimator
def eu_option_strat_prop(K, r, s0, sigma, R, nr_of_strata=4, n=1, seed=2022):
    if R % nr_of_strata != 0:
        raise ValueError('Value of R must be dividable by nr_of_strata!')
    np.random.seed(seed)
    Yhat = 0
    for j in np.arange(nr_of_strata) + 1:
        u = np.random.random(int(R / nr_of_strata))
        v = u / nr_of_strata + (j - 1) / nr_of_strata
        D = np.sqrt(chi2.ppf(v, df=n))
        ksi = np.random.multivariate_normal(mean=np.zeros(n), cov=np.identity(n), size=int(R / nr_of_strata))
        ksi_len = np.sqrt(np.sum(ksi ** 2, axis=1))
        X = (ksi.T / ksi_len).T
        Z = (X.T * D).T
        sigma_matrix = create_sigma(n)
        A = np.linalg.cholesky(sigma_matrix)
        B = np.dot(A, Z.T).T
        St = gbm(B, s0, r, sigma)
        An = np.sum(St, axis=1) / n
        Yk = np.maximum(An - K, 0)
        Yhat = Yhat + np.mean(Yk) * np.exp(-r)
    Yhat = Yhat / nr_of_strata
    return Yhat


def eu_option_strat_opt(K, r, s0, sigma, R, nr_of_strata=4, n=1, seed=2022):
    if R % nr_of_strata != 0:
        raise ValueError('Value of R must be dividable by nr_of_strata!')
    np.random.seed(seed)
    Yhat = 0
    Rprim = 100
    sds = np.zeros(nr_of_strata)
    for j in np.arange(nr_of_strata) + 1:
        u = np.random.random(int(Rprim / nr_of_strata))
        v = u / nr_of_strata + (j - 1) / nr_of_strata
        D = np.sqrt(chi2.ppf(v, df=n))
        ksi = np.random.multivariate_normal(mean=np.zeros(n), cov=np.identity(n), size=int(Rprim / nr_of_strata))
        ksi_len = np.sqrt(np.sum(ksi ** 2, axis=1))
        X = (ksi.T / ksi_len).T
        Z = (X.T * D).T
        sigma_matrix = create_sigma(n)
        A = np.linalg.cholesky(sigma_matrix)
        B = np.dot(A, Z.T).T
        St = gbm(B, s0, r, sigma)
        An = np.sum(St, axis=1) / n
        Yk = np.maximum(An - K, 0)
        sds[j - 1] = np.sqrt(np.var(Yk))

    Rj = np.ceil(R * sds / np.sum(sds))
    for j in np.arange(nr_of_strata) + 1:
        u = np.random.random(int(Rj[j - 1]))
        v = u / nr_of_strata + (j - 1) / nr_of_strata
        D = np.sqrt(chi2.ppf(v, df=n))
        ksi = np.random.multivariate_normal(mean=np.zeros(n), cov=np.identity(n), size=int(Rj[j - 1]))
        ksi_len = np.sqrt(np.sum(ksi ** 2, axis=1))
        X = (ksi.T / ksi_len).T
        Z = (X.T * D).T
        sigma_matrix = create_sigma(n)
        A = np.linalg.cholesky(sigma_matrix)
        B = np.dot(A, Z.T).T
        St = gbm(B, s0, r, sigma)
        An = np.sum(St, axis=1) / n
        Yk = np.maximum(An - K, 0)
        Yhat = Yhat + np.mean(Yk) * np.exp(-r)
    Yhat = Yhat / nr_of_strata
    return Yhat

