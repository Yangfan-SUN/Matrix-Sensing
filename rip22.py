import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# -----------------------------
# Helpers
# -----------------------------
def vec2x2(X):
    return np.array([X[0,0], X[0,1], X[1,0], X[1,1]], float)

def A_apply(A_list, X):
    return np.array([np.sum(Ai * X) for Ai in A_list], float)

def clean_axis_keep_labels(ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


# -----------------------------
# Rank-1 manifold
# -----------------------------
def rank1_curve(theta):
    u = np.array([np.cos(theta), np.sin(theta)])
    v = np.array([1.0, 0.0])
    return np.outer(u, v)

# -----------------------------
# Operators
# -----------------------------
E11 = np.array([[1,0],[0,0]], float)
E12 = np.array([[0,1],[0,0]], float)
E21 = np.array([[0,0],[1,0]], float)
E22 = np.array([[0,0],[0,1]], float)

A_good = [E11, E12, E21, E22]   # Example A
A_bad  = [E11, E12]             # Example B

# -----------------------------
# Projection R^4 -> R^2 (for plotting only)
# -----------------------------
P = np.array([[1.0, 0.2, -0.3, 0.0],
              [0.0, 0.7,  0.6, 0.2]])

def proj(v):
    return P @ v

# -----------------------------
# Plot routine
# -----------------------------
def plot_example(axL, axR, A_list, title_L, title_R, m_is_4=False):
    T = 500
    thetas = np.linspace(0, 2*np.pi, T)

    X_vecs = np.array([vec2x2(rank1_curve(t)) for t in thetas])
    L_curve = np.array([proj(v) for v in X_vecs])

    # ----- Left: rank-1 manifold -----
    axL.plot(L_curve[:,0], L_curve[:,1], lw=2)
    axL.set_title(title_L, fontsize=11)
    axL.set_xlabel("projection dim 1")
    axL.set_ylabel("projection dim 2")
    clean_axis_keep_labels(axL)
    axL.set_aspect("equal")

    # labeled points
    K = 8
    sel = np.linspace(0.3, 2*np.pi-0.3, K)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i/(K-1)) for i in range(K)]

    L_pts, R_pts = [], []

    for i, t in enumerate(sel):
        X = rank1_curve(t)
        v = vec2x2(X)
        y = A_apply(A_list, X)

        Lp = proj(v)
        Rp = proj(y) if m_is_4 else y

        L_pts.append(Lp)
        R_pts.append(Rp)

        axL.scatter(*Lp, s=45, color=colors[i],
                    edgecolors="white", zorder=3)
        axL.text(Lp[0], Lp[1], f"{i+1}",
                 color=colors[i], fontsize=10, fontweight="bold")

        axR.scatter(*Rp, s=45, color=colors[i],
                    edgecolors="white", zorder=3)
        axR.text(Rp[0], Rp[1], f"{i+1}",
                 color=colors[i], fontsize=10, fontweight="bold")

    # ----- Right: measurement space -----
    if m_is_4:
        R_curve = np.array([proj(A_apply(A_list, rank1_curve(t))) for t in thetas])
        axR.plot(R_curve[:,0], R_curve[:,1], lw=2)
        axR.set_xlabel("projection dim 1")
        axR.set_ylabel("projection dim 2")
    else:
        R_curve = np.array([A_apply(A_list, rank1_curve(t)) for t in thetas])
        axR.plot(R_curve[:,0], R_curve[:,1], lw=2)
        axR.set_xlabel(r"$y_1$")
        axR.set_ylabel(r"$y_2$")

    axR.set_title(title_R, fontsize=11)
    clean_axis_keep_labels(axR)
    axR.set_aspect("equal")


# -----------------------------
# Main figure
# -----------------------------
def main():
    fig = plt.figure(figsize=(11.5, 7), dpi=200)
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.35)

    axA_L = fig.add_subplot(gs[0,0])
    axA_R = fig.add_subplot(gs[0,1])
    axB_L = fig.add_subplot(gs[1,0])
    axB_R = fig.add_subplot(gs[1,1])

    plot_example(
        axA_L, axA_R, A_good,
        title_L=r"Rank-1 set in $\mathbb{R}^4$ (2D projection)",
        title_R=r"Example A: $\mathcal{A}(X)$ in $\mathbb{R}^4$ (2D projection)",
        m_is_4=True
    )

    plot_example(
        axB_L, axB_R, A_bad,
        title_L=r"Rank-1 set in $\mathbb{R}^4$ (2D projection)",
        title_R=r"Example B: $\mathcal{A}(X)$ in $\mathbb{R}^2$ (collapsed)",
        m_is_4=False
    )


    # Big arrows
    def add_arrow(axL, axR):
        pL, pR = axL.get_position(), axR.get_position()
        x0, x1 = pL.x1+0.01, pR.x0-0.01
        y = (pL.y0+pL.y1)/2
        arrow = FancyArrowPatch((x0,y), (x1,y),
                                transform=fig.transFigure,
                                arrowstyle="-|>", linewidth=2)
        fig.patches.append(arrow)
        fig.text((x0+x1)/2, y+0.02, r"$\mathcal{A}$",
                 ha="center", fontsize=14, fontweight="bold")

    add_arrow(axA_L, axA_R)
    add_arrow(axB_L, axB_R)

    fig.suptitle(
        "Visualizing the action of $\\mathcal{A}$ on a rank-1 manifold",
        fontsize=15, fontweight="bold", y=0.97
    )

    out = "fig_viz_A_examples_clean.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
