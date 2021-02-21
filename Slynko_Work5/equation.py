import numpy as np


class Equation:
    num_of_equations = 0
    equations = []
    initial_conditions = None
    A = 0.0
    B = 0.0
    C = 0.0
    M = 0.0

    def __init__(self):
        self.num_of_equations = 4

        self.equations.append("y1' = -A * y1 - B * y1 * y3")
        self.equations.append("y2' = A * y1 - M * C * y2 * y3")
        self.equations.append("y3' = A * y1 - B * y1 * y3 - M * C * y2 * y3 + C * y4")
        self.equations.append("y4' = B * y1 * y3 - C * y4")

        self.initial_conditions = np.array([1.76e-03, 0.0, 0.0, 0.0], dtype=np.float64)

        self.A = np.float64(7.89e-10)
        self.B = np.float64(1.1e+07)
        self.C = np.float64(1.13e+03)
        self.M = np.float64(1.0e+06)

    def get_equations(self):
        for eq in self.equations:
            print(eq)

        for i in range(len(self.initial_conditions)):
            print(f"y{i + 1}({self.initial_conditions[i][0]}) = {self.initial_conditions[i][1]}")

    def calculate_f(self, y):
        res = np.zeros(4, dtype=np.float64)
        res[0] = np.float64(-1.0) * self.A * y[0] - self.B * y[0] * y[2]
        res[1] = self.A * y[0] - self.M * self.C * y[1] * y[2]
        res[3] = self.B * y[0] * y[2] - self.C * y[3]
        res[2] = res[1] - res[3]

        return res

    def calculate_der_f(self, y):
        res = np.zeros(self.num_of_equations * self.num_of_equations, dtype=np.float64).reshape(self.num_of_equations,
                                                                                                self.num_of_equations)
        res[0][0] = np.float64(-1.0) * self.A - self.B * y[2]
        res[0][2] = np.float64(-1.0) * self.B * y[0]
        res[1][0] = self.A
        res[1][1] = res[2][1] = np.float64(-1.0) * self.M * self.C * y[2]
        res[1][2] = np.float64(-1.0) * self.M * self.C * y[1]
        res[2][0] = self.A - self.B * y[2]
        res[2][2] = res[0][2] + res[1][2]
        res[2][3] = self.C
        res[3][0] = self.B * y[2]
        res[3][2] = self.B * y[0]
        res[3][3] = np.float64(-1.0) * self.C

        return res
