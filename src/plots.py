import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch
from collections import Counter


def sigmoid(z, other=0):
    return 1/(1 + np.exp(-z+other))


def draw_arrow(ax, x1, y1, x2, y2, size, type=None):
    ax.annotate("",
                xy=(x1, y1), xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="<-", color="black",
                                shrinkA=12, shrinkB=12,
                                patchA=None, patchB=None,
                                connectionstyle="angle3,angleA=90,angleB=0" if type is None else "arc3,rad=0.",
                                lw=size*10)
                )


def compute_traj_mut(traj_mut, norm=True):
    tot_mut = {}
    for mut_l in traj_mut:
        counts_m = {k: v for k, v in Counter(mut_l).items() if k is not None}
        for mut, c in counts_m.items():
            if mut in tot_mut:
                tot_mut[mut] += c
            else:
                tot_mut[mut] = c
    if norm:
        norm_c = {}
        for mut, c in tot_mut.items():
            if mut[0] in norm_c:
                norm_c[mut[0]] += c
            else:
                norm_c[mut[0]] = c
        tot_mut_ = {k: c/norm_c[k[0]] for k, c in tot_mut.items()}
        return tot_mut_
    else:
        return tot_mut


def plot_landscape(traj_mut, pheno_dic, parms, land, y_lab=True, x_lab=True, norm=True):
    model = lambda x, y, p: p[0]*x + (1-p[0])*sigmoid((y-p[1])*p[2])
    MUTANTS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    NAMES = {(-1, -1): "WT", (-1, 1): "$\\Delta$L", (1, -1): "$\\Delta$G", (1, 1): "$\\Delta$G+$\\Delta$L"}

    growth, swim = zip(*list(pheno_dic.values()))
    min_swim, max_swim = min(swim), max(swim)
    min_growth, max_growth = min(growth), max(growth)
    # x = np.linspace(-0.2, 1.2, 100)  # Adjust the range and resolution as needed
    # y = np.linspace(-1.6, 0.2, 100)
    x = np.linspace(min_growth-(max_growth-min_growth)*0.1, max_growth + (max_growth-min_growth)*0.1, 100)  # Adjust the range and resolution as needed
    y = np.linspace(min_swim-(max_swim-min_swim)*0.1, max_swim + (max_swim-min_swim)*0.1, 100)
    # land.set_xlim([0-0.1, 1+0.1])
    land.set_xlim([min_growth-(max_growth-min_growth)*0.1, max_growth + (max_growth-min_growth)*0.1])
    land.set_ylim([min_swim-(max_swim-min_swim)*0.1, max_swim + (max_swim-min_swim)*0.1])

    X, Y = np.meshgrid(x, y)
    Z = model((X-min_growth)/(max_growth-min_growth), Y, parms)
    # Z = model(X, Y, parms)

    cax = land.contourf(X, Y, Z, cmap="Blues", levels=4)
    if x_lab:
        land.set_xlabel("Growth ($h^{-1})$")
    if y_lab:
        land.set_ylabel("Swimming (cm)")

    coords = []
    for el in MUTANTS:
        coords += [pheno_dic[el]]
    coords = np.array(coords)
    land.scatter(coords[:, 0], coords[:, 1], marker="x", c="orangered", s=20)
    for i, el in enumerate(MUTANTS):
        land.text(coords[i, 0], coords[i, 1], NAMES[el], rotation=45, ha="center", va="center")

    mut_traj_count = compute_traj_mut(traj_mut, norm=norm)
    mut_list = [el for el, c in mut_traj_count.items()]
    mut_traj_count = {el: mut_traj_count[el]/sum(mut_traj_count.values()) for el in mut_traj_count}
    in_c = {-1: 0, 1: 0}
    out_c = {-1: 0, 1: 0}
    for (eli, elj), prob in mut_traj_count.items():
        xi, yi = pheno_dic[eli]
        xj, yj = pheno_dic[elj]
        draw_arrow(land, xi, yi, xj, yj, prob)

    for el in ["top", "right"]:
        land.spines[el].set_visible(False)
    return cax
