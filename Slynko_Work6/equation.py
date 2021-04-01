import numpy as np


class Equation:
    epsilon = 0.0001
    a = 0.0
    b = 1.0

    @staticmethod
    def k(x):
        return x + 1

    @staticmethod
    def q(x):
        return np.exp(x)

    @staticmethod
    def f(x):
        return np.exp(-1.0 * x * x)


def get_matrices(N, h):
    M = np.zeros(N * N).reshape(N, N)
    d = np.zeros(N)

    M[0][0] = -1.0
    M[0][1] = 1.0
    M[N - 1][N - 2] = -1.0 * Equation.k(1.0)
    M[N - 1][N - 1] = h + Equation.k(1.0)
    for i in range(1, N - 1):
        M[i][i - 1] = Equation.k((float(i) - 0.5) * h) / h / h
        M[i][i] = -1.0 * (
                    Equation.k((float(i) - 0.5) * h) + Equation.k((float(i) + 0.5) * h)) / h / h - \
                  Equation.q(i * h)
        M[i][i + 1] = Equation.k((float(i) + 0.5) * h) / h / h
        d[i] = -1.0 * Equation.f(i * h)

    return M, d


def get_auxiliary_matrices(M, d):
    if M.shape[0] != M.shape[1] or M.shape[0] != d.shape[0]:
        raise ValueError("invalid matrices")

    N = d.shape[0]
    p = np.zeros(N)
    r = np.zeros(N)

    p[0] = M[0][1] / M[0][0]
    r[0] = d[0] / M[0][0]

    for i in range(1, N - 1):
        p[i] = M[i][i + 1] / (M[i][i] - M[i][i - 1] * p[i - 1])
        r[i] = (d[i] - M[i][i - 1] * r[i - 1]) / (M[i][i] - M[i][i - 1] * p[i - 1])
    r[N - 1] = (d[N - 1] - M[N - 1][N - 2] * r[N - 2]) / (M[N - 1][N - 1] - M[N - 1][N - 2] * p[N - 2])

    return p, r
