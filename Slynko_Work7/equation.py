import numpy as np


class Equation:
    a = 0.0
    b = 1.0

    @staticmethod
    def get_boundary_values():
        return [(0.0, 1.0), (1.0, 0.0)]

    @staticmethod
    def exact_solution(x, e=0.01):
        return -1.0 * e * np.log(x + np.exp(-1.0 / e) * (1 - x))


def shooting_method(n, epsilon, precision=0.001):
    bv = Equation.get_boundary_values()
    right_bv = bv[1]

    def right_boundary_dev(right_y_value):
        return right_y_value - right_bv[1]

    step = 0.5
    while True:
        alpha1 = -1.0 * epsilon * (np.exp(1.0 / epsilon) - 1) + step
        tmp1 = get_auxiliary_solution(n, epsilon, alpha1)
        dev1 = right_boundary_dev(tmp1[n - 1])
        if abs(dev1) < precision:
            return tmp1

        alpha2 = -1.0 * epsilon * (np.exp(1.0 / epsilon) - 1) - step
        tmp2 = get_auxiliary_solution(n, epsilon, alpha2)
        dev2 = right_boundary_dev(tmp2[n - 1])
        if abs(dev2) < precision:
            return tmp2

        if dev1 * dev2 < 0.0:
            while True:
                print(dev1, dev2)
                alpha = (alpha1 + alpha2) / 2
                tmp = get_auxiliary_solution(n, epsilon, alpha)
                print(f"In iteration right boundary equals {tmp[n - 1]}")
                dev = right_boundary_dev(tmp[n - 1])
                if abs(dev) < precision:
                    return tmp

                if dev * dev1 < 0.0:
                    alpha2 = alpha
                elif dev * dev2 < 0.0:
                    alpha1 = alpha
                else:
                    print("Unforeseen!")
        else:
            print(f"Step equals {step}")
            step += 0.1


def get_auxiliary_solution(n, epsilon, alpha):
    h = (Equation.b - Equation.a) / n
    bv = Equation.get_boundary_values()

    y = np.zeros(n)
    y[0] = bv[0][1]
    y[1] = y[0] + alpha * h
    for i in range(2, n):
        b = y[i - 2] + 2.0 * epsilon
        c = y[i - 2] * y[i - 2] + 8.0 * epsilon * y[i - 1] - 4.0 * epsilon * y[i - 2]
        print(f"D = {b * b - c}")
        y1 = b - np.sqrt(b * b - c)
        y2 = b + np.sqrt(b * b - c)
        if abs(y[i - 1] - y1) < abs(y[i - 1] - y2):
            y[i] = y1
        else:
            y[i] = y2

    return y
