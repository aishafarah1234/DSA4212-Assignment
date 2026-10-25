import numpy as np

class TwoBasinGaussian:
    def __init__(self, d=5, sigma_l=1.0, sigma_g=1.0,
                 A_g=2.0, A_l=1.0, separation=5.0, domain=10.0):
        self.d = d
        self.sigma_l = sigma_l
        self.sigma_g = sigma_g
        self.A_g = A_g
        self.A_l = A_l
        self.domain = domain
        # Place basins along the first axis, symmetric about origin
        self.mu_g = np.zeros(d); self.mu_g[0] = -separation / 2
        self.mu_l = np.zeros(d); self.mu_l[0] = +separation / 2

    def f_global(self, X):
        X = np.atleast_2d(X)
        sq = np.sum((X - self.mu_g) ** 2, axis=1)
        return -self.A_g * np.exp(-sq / (2 * self.sigma_g ** 2))

    def f_local(self, X):
        X = np.atleast_2d(X)
        sq = np.sum((X - self.mu_l) ** 2, axis=1)
        return -self.A_l * np.exp(-sq / (2 * self.sigma_l ** 2))

    def __call__(self, X):
        X = np.atleast_2d(X)
        return np.minimum(self.f_global(X), self.f_local(X))

    def basin_of(self, X):
        X = np.atleast_2d(X)
        return np.where(self.f_global(X) <= self.f_local(X), 0, 1)  # 0=global, 1=local

    @property
    def bounds(self):
        return np.array([[-self.domain, self.domain]] * self.d)

    def global_optimum_value(self):
        return -self.A_g  # attained at mu_g

    def global_optimum_point(self):
        return self.mu_g.copy()
