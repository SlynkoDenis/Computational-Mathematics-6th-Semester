import numpy as np


class Equation:
    a = np.float64(0.0)
    b = np.float64(1.0)

    @staticmethod
    def get_boundary_values():
        return [(np.float64(0.0), np.float64(1.0)), (np.float64(1.0), np.float64(0.0))]

    @staticmethod
    def exact_solution(x, e=np.float64(0.01)):
        return -1.0 * e * np.log(x + np.exp(-1.0 / e) * (np.float64(1.0) - x))


def shooting_method(n, epsilon, precision=0.001):
    bv = Equation.get_boundary_values()
    right_bv = bv[1][1]

    def right_boundary_dev(right_y_value, rbv):
        return right_y_value - rbv

    step = np.float64(0.001)
    while True:
        alpha1 = -1.0 * epsilon * (np.exp(1.0 / epsilon) - 1) + step
        tmp1 = get_auxiliary_solution(n, epsilon, alpha1)
        dev1 = right_boundary_dev(tmp1[n - 1], right_bv)
        if abs(dev1) < precision:
            return tmp1

        alpha2 = -1.0 * epsilon * (np.exp(1.0 / epsilon) - 1) - step
        tmp2 = get_auxiliary_solution(n, epsilon, alpha2)
        dev2 = right_boundary_dev(tmp2[n - 1], right_bv)
        if abs(dev2) < precision:
            return tmp2

        if dev1 * dev2 < 0.0:
            while True:
                alpha = (alpha1 + alpha2) / 2
                tmp = get_auxiliary_solution(n, epsilon, alpha)
                dev = right_boundary_dev(tmp[n - 1], right_bv)
                if abs(dev) < precision:
                    return tmp

                if dev * dev1 < 0.0:
                    alpha2 = alpha
                elif dev * dev2 < 0.0:
                    alpha1 = alpha
                else:
                    print("Unforeseen!")
        else:
            step *= 10.0
            print(f"Step equals {step}")


def get_auxiliary_solution(n, epsilon, alpha):
    h = (Equation.b - Equation.a) / n
    bv = Equation.get_boundary_values()

    y = np.zeros(n)
    y[0] = bv[0][1]
    y[1] = y[0] + alpha * h
    for i in range(2, n):
        b = y[i - 2] + 2.0 * epsilon
        d = 4.0 * epsilon * (epsilon + 2.0 * y[i - 2] - 2.0 * y[i - 1])
        y1 = b - np.sqrt(d)
        y2 = b + np.sqrt(d)
        if abs(y[i - 1] - y1) < abs(y[i - 1] - y2):
            y[i] = y1
        else:
            y[i] = y2

    return y
