#!/usr/bin/python3.8

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
import os
import equation


def create_plots_dir(name="Plots"):
    if not os.path.isdir(os.path.join(os.path.dirname(sys.argv[0]), name)):
        subprocess.check_output(f"mkdir {name}", shell=True)


epsilon = 0.0001
result_values = np.zeros(11)

dirname = "Plots"
create_plots_dir(dirname)
cumulative_fig = plt.figure()
cumulative_ax = cumulative_fig.add_subplot(111)

N = 11
while True:
    h = (equation.Equation.b - equation.Equation.a) / float(N - 1)

    M, d = equation.get_matrices(N, h)

    p, r = equation.get_auxiliary_matrices(M, d)

    u = np.zeros(N)
    u[N - 1] = r[N - 1]
    for i in range(N - 2, -1, -1):
        u[i] = r[i] - p[i] * u[i + 1]
    x = np.array([h * t for t in range(N)], dtype=float)

    markersz = 2 if N < 300 else 1
    cumulative_ax.plot(x, u, marker='o', markersize=markersz, label=f"h={h}")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, u, marker='o', markersize=2)
    ax.set_title(f"Tridiagonal matrix solution for h = {h}", fontsize=20)
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("u", fontsize=16)
    ax.grid()

    fig.set_figheight(20)
    fig.set_figwidth(25)

    fig.savefig(os.path.join(dirname, f"with_{N}_points.png"))
    plt.close(fig)

    step = int(N / 11.0)
    print(h, abs(result_values[0] - u[0]))
    if all([abs(result_values[i] - u[step * i]) < epsilon for i in range(11)]):
        print(f"The solution with the error = {epsilon} was gained with {N} points")
        print([u[step * i] for i in range(11)])

        cumulative_ax.set_title("Tridiagonal matrix solutions", fontsize=20)
        cumulative_ax.set_xlabel("x", fontsize=16)
        cumulative_ax.set_ylabel("u", fontsize=16)
        cumulative_ax.legend()
        cumulative_ax.grid()

        cumulative_fig.set_figheight(30)
        cumulative_fig.set_figwidth(40)

        cumulative_fig.savefig(os.path.join(dirname, "cumulative.png"))
        break
    else:
        result_values = np.array([u[step * i] for i in range(11)])
    N *= 3
