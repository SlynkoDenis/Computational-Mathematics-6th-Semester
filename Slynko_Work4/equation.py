import numpy as np


class Equation:
    num_of_equations = 0
    equations = []
    initial_conditions = None

    def __init__(self):
        self.num_of_equations = 2

        self.equations.append("x1' = 1 + x1^2 * v - (B - 1) * x2")
        self.equations.append("x2' = B * x1 - x1^2 * x2")

        self.initial_conditions = np.array([1.0, 1.0])

    def get_equations(self):
        for eq in self.equations:
            print(eq)

        for i in range(len(self.initial_conditions)):
            print(f"x{i + 1}({self.initial_conditions[i][0]}) = {self.initial_conditions[i][1]}")

    def calculate_f(self, u, b):
        if b < 1.0 or b > 5.0:
            raise ValueError(f"b = {b} doesn't match the task range [1, 5]")

        return np.array([1.0 + u[0] * u[0] * u[1] - (b + 1) * u[1], b * u[0] - u[0] * u[0] * u[1]])
