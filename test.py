import numpy as np
from scipy.stats import norm
from scipy.stats.distributions import chi2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# m = 4
# R = 10000
# fig3 = plt.figure(1)
# ax3 = fig3.add_subplot(111, projection='3d')
# for j in np.arange(m) + 1:
#     u = np.random.random(int(R / m))
#     v = u / m + (j - 1) / m
#     D = np.sqrt(chi2.ppf(v, df=3))
#     x = np.random.normal(0, 1, int(R / m))
#     y = np.random.normal(0, 1, int(R / m))
#     z = np.random.normal(0, 1, int(R / m))
#     d_vec = np.sqrt(x ** 2 + y ** 2 + z ** 2)
#     x = x / d_vec
#     y = y / d_vec
#     z = z / d_vec
#     ax3.scatter(x * D, y * D, z * D, s=1, label=j)
#
# plt.show()
def create_sigma(n):
    sigma = np.full((n, n), 0)
    for i in np.arange(n):
        for j in np.arange(n):
            sigma[i, j] = np.minimum(i, j) + 1

    sigma = sigma / n
    return sigma


m = 4
R = 10000
fig3 = plt.figure(1)
ax3 = fig3.add_subplot(111, projection='3d')
for j in np.arange(m) + 1:
    u = np.random.random(int(R / m))
    v = u / m + (j - 1) / m
    D = np.sqrt(chi2.ppf(v, df=3))
    Z = np.array([np.random.normal(0, 1, int(R / m)), np.random.normal(0, 1, int(R / m)), np.random.normal(0, 1, int(R / m))]).T
    d_vec = np.sqrt(Z[:,0] ** 2 + Z[:,1] ** 2 + Z[:,2] ** 2)
    Z = (Z.T * 1 / d_vec).T
    sigma = np.array([1,1,1,1,2,2,1,2,3]) / 3
    sigma.shape = (3,3)
    A = np.linalg.cholesky(sigma)
    X = np.dot(A, Z.T).T
    ax3.scatter(X[:, 0] * D, X[:, 1] * D, X[:, 2] * D, s=1, label=j)

plt.show()

#eu_option_strat_prop(100, 0.05, 100, 0.25, 1000, seed=2)
# m = 4
# R = 10000
# fig3 = plt.figure(1)
# ax3 = fig3.add_subplot(111, projection='3d')
# nr_of_strata = m
# n = 3
# for j in np.arange(m) + 1:
#     u = np.random.random(int(R / m))
#     v = u / m + (j - 1) / m
#     D = np.sqrt(chi2.ppf(v, df=3))
#     ksi = np.random.multivariate_normal(mean=np.zeros(n), cov=np.identity(n), size=int(R / nr_of_strata))
#     ksi_len = np.sqrt(np.sum(ksi ** 2, axis=0))
#     X = ksi / ksi_len
#     Z = (X.T * D).T
#     sigma_matrix = create_sigma(n)
#     A = np.linalg.cholesky(sigma_matrix)
#     B = np.dot(A, Z.T).T
#     ax3.scatter(B[:, 0], B[:, 1], B[:, 2], s=1, label=j)
#
# plt.show()