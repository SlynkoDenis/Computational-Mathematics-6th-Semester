import numpy as np


def solve_seidel(A, f, debug=False):
    if debug:
        print(f"Determinant in Seidel method equals {np.linalg.det(A)}")
    # return np.linalg.inv(A).dot(f)
    epsilon = np.float64(1.0e-20)

    dim = A.shape[0]

    L = np.tril(A, k=-1)
    D = np.zeros(dim * dim, dtype=np.float64).reshape(dim, dim)
    for i in range(dim):
        D[i][i] = A[i][i]
    U = A - L - D
    inversed = np.linalg.inv(L + D)

    init = np.zeros(dim, dtype=np.float64)
    res = np.float64(-1.0) * inversed.dot(U).dot(init) + inversed.dot(f)
    counter = 0
    while np.linalg.norm(res - init, ord=1) >= epsilon and counter < 50:
        counter += 1
        init = res
        res = np.float64(-1.0) * inversed.dot(U).dot(init) + inversed.dot(f)

    if debug:
        print(f"Difference equals {np.linalg.norm(np.linalg.inv(A).dot(f) - res)}")

    return res


# in our case a = const = (I - h * alpha[0][0] * df/dy)^-1
def solve_newton(a, y_n, h, alpha, calculate_f, maxiter=50):
    if a.shape[0] != a.shape[1]:
        raise ValueError("Calculation matrix of the system must be square")

    epsilon = np.float64(1.0e-20)

    dim = a.shape[0]
    initial = np.zeros(dim, dtype=np.float64)
    res = a.dot(calculate_f(y_n))
    counter = 0
    while np.linalg.norm(res - initial) >= epsilon and counter <= maxiter:
        counter += 1
        initial = res
        res = initial - a.dot(initial - calculate_f(y_n + h * alpha * initial))

    return res
