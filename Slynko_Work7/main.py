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


n = 100
precision = 0.01
epsilon = 0.1
u = equation.shooting_method(n, epsilon, precision=precision)
h = (equation.Equation.b - equation.Equation.a) / n
x = np.arange(0.0, n * h, step=h, dtype=float)
print(u)


fig = plt.figure()

ax = fig.add_subplot(111)
ax.plot(x, u, marker='o', label=f"e={epsilon}", markersize=3)
ax.set_title(f"Shooting method solution for epsilon = {epsilon}", fontsize=20)
ax.set_xlabel("x", fontsize=16)
ax.set_ylabel("u", fontsize=16)
ax.grid()

fig.set_figheight(20)
fig.set_figwidth(25)

dirname = "Plots"
create_plots_dir(dirname)
fig.savefig(os.path.join(dirname, f"{str(epsilon).replace('.', '_')}.png"))
