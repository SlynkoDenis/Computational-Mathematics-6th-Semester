#!/usr/bin/python3.8

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import time
import sys
import os
import equation


def create_plots_dir(name="Plots"):
    if not os.path.isdir(os.path.join(os.path.dirname(sys.argv[0]), name)):
        subprocess.check_output(f"mkdir {name}", shell=True)


def get_max_deviation(y, exact_y):
    return max(np.abs(np.array(y, dtype=np.float64) - exact_y))


n = 1000000
precision = np.float64(0.001)

h = (equation.Equation.b - equation.Equation.a) / n
x = np.arange(0.0, n * h, step=h, dtype=np.float64)

fig = plt.figure()
ax = fig.add_subplot(111)

eps = [np.float64(0.07), np.float64(0.1), np.float64(0.35), np.float64(0.7)]
for i in eps:
    epsilon = i

    exact_solution = np.float64(-1.0) * epsilon * np.log(x + np.exp(-1.0 / epsilon) * (np.float64(1.0) - x))

    start_time = time.time()
    u = equation.shooting_method(n, epsilon, precision=precision)
    end_time = time.time()
    print(f"Time elapsed for epsilon = {epsilon}: {end_time - start_time}s")
    print("Maximum deviation comparing to the exact solution equals", get_max_deviation(u, exact_solution))
    ax.plot(x, u, marker='o', label=f"e={epsilon}", markersize=1)

ax.set_title(f"Shooting method solution", fontsize=20)
ax.set_xlabel("x", fontsize=16)
ax.set_ylabel("u", fontsize=16)
ax.legend(fontsize=20)
ax.grid()

fig.set_figheight(20)
fig.set_figwidth(25)

dirname = "Plots"
create_plots_dir(dirname)
fig.savefig(os.path.join(dirname, "graph.png"))
