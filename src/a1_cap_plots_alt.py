import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[1]
TAB_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"


def load_cap_table():
    p = TAB_DIR / "A1_CAP_Sensitivity_USA.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run src/a1_cap_sensitivity.py first.")
    df = pd.read_csv(p)
    return df


def get_base(df):
    base_total = float(df["USA_Total_host0"].iloc[0])
    base_gold = float(df["USA_Gold_host0"].iloc[0])
    return base_total, base_gold


def tornado_plot(df, base_ct, base_cg, outpath):
    """
    Fix baseline (base_ct, base_cg). Vary one cap at a time and plot ranges for:
    - Total_host1, Gold_host1
    """
    base_total, base_gold = get_base(df)

    caps_total = np.sort(df["cap_total"].unique())
    caps_gold = np.sort(df["cap_gold"].unique())

    # baseline values
    total0 = base_total * base_ct
    gold0 = base_gold * base_cg

    # vary cap_total only
    total_var = base_total * caps_total
    gold_hold = np.full_like(caps_total, gold0)

    # vary cap_gold only
    gold_var = base_gold * caps_gold
    total_hold = np.full_like(caps_gold, total0)

    # compute deltas from baseline
    d_total_by_ct = total_var - total0
    d_gold_by_cg = gold_var - gold0

    labels = (
        [f"cap_total: {ct:.2f}" for ct in caps_total] +
        [f"cap_gold: {cg:.2f}" for cg in caps_gold]
    )
    deltas = np.concatenate([d_total_by_ct, d_gold_by_cg])

    # sort by absolute impact
    order = np.argsort(np.abs(deltas))[::-1]
    labels = [labels[i] for i in order]
    deltas = deltas[order]

    plt.figure()
    y = np.arange(len(labels))
    plt.barh(y, deltas)
    plt.axvline(0)
    plt.yticks(y, labels)
    plt.xlabel("Change from baseline (host1 medals)")
    plt.title(f"Tornado Sensitivity (baseline cap_total={base_ct}, cap_gold={base_cg})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def heatmap_composite(df, outpath, mode="H"):
    """
    Heatmap over (cap_total, cap_gold). Use composite index:
      H = sqrt((ΔT/T0)^2 + (ΔG/G0)^2)
    """
    base_total, base_gold = get_base(df)

    # grid
    caps_total = np.sort(df["cap_total"].unique())
    caps_gold = np.sort(df["cap_gold"].unique())

    grid = np.zeros((len(caps_gold), len(caps_total)))

    for i, cg in enumerate(caps_gold):
        for j, ct in enumerate(caps_total):
            sub = df[(df["cap_total"] == ct) & (df["cap_gold"] == cg)].iloc[0]
            dT = float(sub["delta_total"])
            dG = float(sub["delta_gold"])
            if mode == "H":
                val = np.sqrt((dT / base_total) ** 2 + (dG / base_gold) ** 2)
            elif mode == "delta_total":
                val = dT
            else:
                val = dG
            grid[i, j] = val

    plt.figure()
    plt.imshow(grid, aspect="auto", origin="lower")
    plt.xticks(np.arange(len(caps_total)), [f"{x:.2f}" for x in caps_total])
    plt.yticks(np.arange(len(caps_gold)), [f"{x:.2f}" for x in caps_gold])
    plt.xlabel("CAP_TOTAL")
    plt.ylabel("CAP_GOLD")
    title = "Heatmap of Composite Host Boost Index H" if mode == "H" else f"Heatmap of {mode}"
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def add_confidence_ellipse(x, y, ax, n_std=2.0):
    """
    Draw an ellipse representing n_std confidence region of (x,y).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    mean_x, mean_y = np.mean(x), np.mean(y)
    ellipse = Ellipse((mean_x, mean_y), width=width, height=height, angle=theta, fill=False)
    ax.add_patch(ellipse)


def correlated_mc_cloud(df, base_ct, base_cg, rho, nsim, outpath):
    """
    Sample correlated caps using Gaussian copula:
    - cap_total ~ Normal(mean=base_ct, sd=(max-min)/6), clipped to [min,max]
    - cap_gold  ~ Normal(mean=base_cg, sd=(max-min)/6), clipped to [min,max]
    correlation = rho
    """
    base_total, base_gold = get_base(df)
    caps_total = np.sort(df["cap_total"].unique())
    caps_gold = np.sort(df["cap_gold"].unique())

    ct_min, ct_max = float(caps_total.min()), float(caps_total.max())
    cg_min, cg_max = float(caps_gold.min()), float(caps_gold.max())

    sd_ct = (ct_max - ct_min) / 6.0 if ct_max > ct_min else 0.01
    sd_cg = (cg_max - cg_min) / 6.0 if cg_max > cg_min else 0.01

    rng = np.random.default_rng(42)
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z = rng.normal(size=(nsim, 2)) @ L.T

    ct = base_ct + sd_ct * z[:, 0]
    cg = base_cg + sd_cg * z[:, 1]
    ct = np.clip(ct, ct_min, ct_max)
    cg = np.clip(cg, cg_min, cg_max)

    gold_host1 = base_gold * cg
    total_host1 = base_total * ct

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(gold_host1, total_host1, s=12)
    add_confidence_ellipse(gold_host1, total_host1, ax, n_std=2.0)
    ax.set_xlabel("USA Gold (host1)")
    ax.set_ylabel("USA Total (host1)")
    ax.set_title(f"Joint Uncertainty Cloud (rho={rho}, nsim={nsim})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def vector_field(df, outpath):
    """
    Arrow plot on cap grid: arrows represent (ΔGold, ΔTotal) normalized.
    """
    caps_total = np.sort(df["cap_total"].unique())
    caps_gold = np.sort(df["cap_gold"].unique())

    X, Y = np.meshgrid(caps_total, caps_gold)
    U = np.zeros_like(X, dtype=float)  # ΔGold
    V = np.zeros_like(Y, dtype=float)  # ΔTotal

    for i, cg in enumerate(caps_gold):
        for j, ct in enumerate(caps_total):
            sub = df[(df["cap_total"] == ct) & (df["cap_gold"] == cg)].iloc[0]
            U[i, j] = float(sub["delta_gold"])
            V[i, j] = float(sub["delta_total"])

    # normalize for better arrow visibility
    mag = np.sqrt(U**2 + V**2)
    mag[mag == 0] = 1.0
    U2 = U / mag
    V2 = V / mag

    plt.figure()
    plt.quiver(X, Y, U2, V2, angles="xy")
    plt.xlabel("CAP_TOTAL")
    plt.ylabel("CAP_GOLD")
    plt.title("Direction of Host Boost (normalized arrows)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ct", type=float, default=1.25)
    ap.add_argument("--base_cg", type=float, default=1.35)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--nsim", type=int, default=2000)
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cap_table()

    tornado_plot(
        df, args.base_ct, args.base_cg,
        FIG_DIR / "Fig_A1_CAP_Tornado.png"
    )
    heatmap_composite(
        df, FIG_DIR / "Fig_A1_CAP_Heatmap_CompositeH.png", mode="H"
    )
    correlated_mc_cloud(
        df, args.base_ct, args.base_cg, args.rho, args.nsim,
        FIG_DIR / "Fig_A1_CAP_MC_Cloud.png"
    )
    vector_field(
        df, FIG_DIR / "Fig_A1_CAP_VectorField.png"
    )

    print("Saved figures:")
    print(" - Fig_A1_CAP_Tornado.png")
    print(" - Fig_A1_CAP_Heatmap_CompositeH.png")
    print(" - Fig_A1_CAP_MC_Cloud.png")
    print(" - Fig_A1_CAP_VectorField.png")


if __name__ == "__main__":
    main()