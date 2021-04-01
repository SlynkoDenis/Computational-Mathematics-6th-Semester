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


N = 11
h = (equation.Equation.b - equation.Equation.a) / float(N)

M, d = equation.get_matrices(N, h)

p, r = equation.get_auxiliary_matrices(M, d)

u = np.zeros(N)
u[N - 1] = r[N - 1]
for i in range(N - 2, -1, -1):
    u[i] = r[i] - p[i] * u[i + 1]
x = np.arange(0.0, N * h, step=h, dtype=float)


fig = plt.figure()

ax = fig.add_subplot(111)
ax.plot(x, u, marker='o', markersize=3)
ax.set_title("Tridiagonal matrix solution", fontsize=20)
ax.set_xlabel("x", fontsize=16)
ax.set_ylabel("u", fontsize=16)
ax.grid()

fig.set_figheight(20)
fig.set_figwidth(25)

dirname = "Plots"
create_plots_dir(dirname)
fig.savefig(os.path.join(dirname, "plot.png"))
