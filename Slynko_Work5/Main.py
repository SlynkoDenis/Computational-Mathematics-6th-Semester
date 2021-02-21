import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
import ssolver
import subprocess
import equation
import argparse
import time
import sys
import os


def get_func(y_n, h, alpha, func):
    def f(x):
        return x - func(y_n + h * alpha * x)
    return f


def get_prim(y_n, h, alpha, func):
    def f(x):
        return np.identity(4) - h * alpha * func(y_n + h * alpha * x)
    return f


class Main:
    system = None
    right_end = 0
    debug = False

    def __init__(self, right_end=1.0e+03):
        self.system = equation.Equation()
        self.right_end = right_end

        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action="store_true",
                            help="Debug mode, print step results")
        cmd_args = parser.parse_args()
        self.debug = cmd_args.debug

    @staticmethod
    def create_plots_dir(name="Plots"):
        if not os.path.isdir(os.path.join(os.path.dirname(sys.argv[0]), name)):
            subprocess.check_output("mkdir Plots", shell=True)

    @staticmethod
    def __initiate__method__():
        num_of_steps = 4

        alpha = np.zeros(num_of_steps * num_of_steps, dtype=np.float64).reshape(num_of_steps, num_of_steps)
        for i in range(num_of_steps):
            alpha[i][i] = np.float64(0.5)
        alpha[1][0] = np.float64(1.0 / 6.0)
        alpha[2][0] = np.float64(-0.5)
        alpha[2][1] = alpha[3][2] = np.float64(0.5)
        alpha[3][0] = np.float64(1.5)
        alpha[3][1] = np.float64(-1.5)
        c = np.array([0.5, 2.0 / 3.0, 0.5, 1.0], dtype=np.float64)
        b = np.array([1.5, -1.5, 0.5, 0.5], dtype=np.float64)

        return num_of_steps, alpha, b, c

    def __solve_runge_kutte__(self, y_n, h):
        num_of_steps, alpha, b, c = self.__initiate__method__()

        k = np.zeros(num_of_steps * self.system.num_of_equations, dtype=np.float64) \
            .reshape(num_of_steps, self.system.num_of_equations)

        # alpha[i][i] is the same for all i
        # we calculate derivative matrix in y_n for the sake of fast execution
        k_matrix = np.linalg.inv(np.identity(self.system.num_of_equations, dtype=np.float64) -\
                                 h * alpha[0][0] * self.system.calculate_der_f(y_n))
        for i in range(num_of_steps):
            k[i] = ssolver.solve_newton(k_matrix, y_n + h * sum(alpha[i][j] * k[j] for j in range(i)), h, alpha[i][i],
                                        self.system.calculate_f)

        tmp = y_n + b.dot(k)
        tmp[2] = tmp[1] - tmp[3]
        return tmp

    def __solve_rosenbrock__(self, y_n, h):
        num_of_steps, alpha, b, c = self.__initiate__method__()

        k = np.zeros(num_of_steps * self.system.num_of_equations, dtype=np.float64)\
            .reshape(num_of_steps, self.system.num_of_equations)
        k[0] = h * self.system.calculate_f(y_n)

        # alpha[i][i] is the same for all i
        tmp_matrix = np.identity(self.system.num_of_equations, dtype=np.float64) -\
                     h * alpha[0][0] * self.system.calculate_der_f(y_n)
        for i in range(1, num_of_steps):
            tmp = sum(alpha[i][j] * k[j] for j in range(i))
            # k[i] = tmp_matrix.dot(h * self.system.calculate_f(y_n + tmp))
            k[i] = ssolver.solve_seidel(tmp_matrix, h * self.system.calculate_f(y_n + tmp), debug=self.debug)

        tmp = y_n + b.dot(k)
        tmp[2] = tmp[1] - tmp[3]
        return tmp

    def solve(self, h):
        x_axis = []
        y_values = ([], [], [], [])

        x = np.float64(0.0)
        y = self.system.initial_conditions
        while x <= self.right_end:
            try:
                y_tmp = self.__solve_rosenbrock__(y, h)
                y = y_tmp
            except np.linalg.LinAlgError as e:
                print(f"An error occurred while calculating Rosenbrock iteration for x = {x}")
                print(f"Previous y = {y}")
                print(e)

                x += h
                y = self.__solve_rosenbrock__(y, np.float64(2.0) * h)

            x_axis.append(x)
            x += h
            for i in range(self.system.num_of_equations):
                y_values[i].append(y[i])

            if self.debug:
                print("\n", x, y, "\n")

        return x_axis, y_values

    def main(self, h=0.5):
        Main.create_plots_dir()

        fig = plt.figure()

        start = time.time()
        x_axis, y_values = self.solve(h)
        end = time.time()
        print(f"Elapsed time = {end - start}")

        ax = fig.add_subplot(111)
        for i in range(self.system.num_of_equations):
            ax.plot(x_axis, y_values[i], label=f"y{i + 1}", marker='o', markersize=1)
        ax.set_title("Rosenbrock", fontsize=20)
        ax.set_xlabel("x", fontsize=16)
        ax.set_ylabel("y", fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=20)
        ax.grid()

        fig.set_figheight(20)
        fig.set_figwidth(25)
        fig.savefig(os.path.join("Plots", "Result.png"))


if __name__ == '__main__':
    main = Main(right_end=np.float64(1.0e+04))
    main.main(h=np.float64(0.5))
