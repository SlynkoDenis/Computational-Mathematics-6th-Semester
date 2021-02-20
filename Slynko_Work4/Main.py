import numpy as np
import matplotlib.pyplot as plt
import subprocess
import equation
import sys
import os


class Main:
    system = None
    num_of_steps = 0
    right_end = 0

    def __init__(self, right_end=1.0):
        self.system = equation.Equation()
        self.num_of_steps = 4
        self.right_end = right_end

    @staticmethod
    def create_plots_dir(name="Plots"):
        if not os.path.isdir(os.path.join(os.path.dirname(sys.argv[0]), name)):
            subprocess.check_output("mkdir Plots", shell=True)

    def __sub_solve__(self, u_n, h, b, p):
        if p == 1:
            alpha = np.zeros(self.num_of_steps * self.num_of_steps).reshape(self.num_of_steps, self.num_of_steps)
            alpha[1][0] = alpha[2][1] = alpha[3][2] = 0.5
            # deprecated
            c = np.array([0, 0.5, 0.5, 0.5])
            beta = np.array([0.25, 0.25, 0.25, 0.25])
        elif p == 4:
            alpha = np.zeros(self.num_of_steps * self.num_of_steps).reshape(self.num_of_steps, self.num_of_steps)
            alpha[1][0] = alpha[2][1] = 0.5
            alpha[3][2] = 1.0
            # deprecated
            c = np.array([0, 0.5, 0.5, 1])
            beta = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
            # Another possible method - '3/8 rule'
            # beta = np.array([1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0])
            # alpha[1][0] = 1.0 / 3.0
            # alpha[2][0] = -1.0 / 3.0
            # alpha[2][1] = alpha[3][0] = alpha[3][2] = 1.0
            # alpha[3][1] = -1.0
        else:
            raise ValueError(f"Unsupported order: p = {p}")

        k = np.zeros(self.num_of_steps * self.system.num_of_equations).reshape(self.num_of_steps,
                                                                               self.system.num_of_equations)
        for i in range(self.num_of_steps):
            tmp = alpha[i][0] * k[0]
            if i > 0:
                for j in range(1, i):
                    tmp += alpha[i][j] * k[j]
            k[i] = self.system.calculate_f(u_n + h * tmp, b)

        # print(f"Step for {x_n}: {u_n + h * c * sum(x for x in k)}")
        return u_n + h * beta.dot(k)

    def solve(self, h, b, p=4):
        x_axis = []
        u_values = ([], [])

        x = 0.0
        u = self.system.initial_conditions
        while x <= self.right_end:
            x_axis.append(x)
            u = self.__sub_solve__(u, h, b, p)
            x += h
            for i in range(self.system.num_of_equations):
                u_values[i].append(u[i])

        return x_axis, u_values

    def main(self, b, h=0.02):
        Main.create_plots_dir()

        fig = plt.figure()

        x_axis, u_values = self.solve(h, b, p=1)
        x_axis4, u_values4 = self.solve(h, b, p=4)

        ax = fig.add_subplot(121)
        ax.plot(x_axis, u_values[0], label='p = 1', marker='o')
        ax.plot(x_axis4, u_values4[0], label='p = 4', marker='o')
        ax.set_title("b = 2.0", fontsize=20)
        ax.set_xlabel("x", fontsize=16)
        ax.set_ylabel("u", fontsize=16)
        ax.legend(fontsize=20)
        ax.grid()

        ax = fig.add_subplot(122)
        ax.plot(x_axis, u_values[1], label='p = 1', marker='o')
        ax.plot(x_axis4, u_values4[1], label='p = 4', marker='o')
        ax.set_title("b = 2.0", fontsize=20)
        ax.set_xlabel("x", fontsize=16)
        ax.set_ylabel("v", fontsize=16)
        ax.legend(fontsize=20)
        ax.grid()

        fig.set_figheight(20)
        fig.set_figwidth(25)
        fig.savefig(os.path.join("Plots", f"{str(b).replace('.', '_')}.png"))

        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.plot(u_values[0], u_values[1], label='p = 1', marker='o')
        ax.plot(u_values4[0], u_values4[1], label='p = 4', marker='o')
        ax.set_title(f"Phase for b = {b}", fontsize=20)
        ax.set_xlabel("u", fontsize=16)
        ax.set_ylabel("v", fontsize=16)
        ax.legend(fontsize=20)
        ax.grid()

        fig.set_figheight(20)
        fig.set_figwidth(25)
        fig.savefig(os.path.join("Plots", f"Phase_{str(b).replace('.', '_')}.png"))


if __name__ == '__main__':
    main = Main(10.0)
    b = 1.0
    while b <= 5.0:
        main.main(b)
        b += 0.5
