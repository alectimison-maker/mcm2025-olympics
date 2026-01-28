import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TAB_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
RAW_DIR = ROOT / "data" / "raw"

def latest_file(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern} in {folder}")
    return files[-1]

def make_rank_table_mu(df_pred, topn=20):
    df = df_pred.copy()
    df = df.sort_values(["pred_Gold", "pred_Total"], ascending=[False, False])
    df["Rank"] = np.arange(1, len(df) + 1)
    show = df[[
        "Rank", "NOC",
        "pred_Gold", "mu_low_Gold", "mu_high_Gold",
        "pred_Total", "mu_low_Total", "mu_high_Total"
    ]].head(topn)
    return show

def plot_top15_gold_mu_errorbar(df_pred, outpath: Path):
    df = df_pred.sort_values(["pred_Gold", "pred_Total"], ascending=[False, False]).head(15).copy()
    x = np.arange(len(df))
    y = df["pred_Gold"].values
    yerr_low = y - df["mu_low_Gold"].values
    yerr_high = df["mu_high_Gold"].values - y
    plt.figure()
    plt.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o")
    plt.xticks(x, df["NOC"].values, rotation=45, ha="right")
    plt.ylabel("Expected Gold (2028)")
    plt.title("Top-15 Expected Gold with 95% Bootstrap CI (Mean Band)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_gold_vs_total(df_pred, outpath: Path):
    plt.figure()
    plt.scatter(df_pred["pred_Gold"], df_pred["pred_Total"])
    plt.xlabel("Expected Gold (2028)")
    plt.ylabel("Expected Total (2028)")
    plt.title("Gold vs Total (2028, Expected Values)")
    top = df_pred.sort_values(["pred_Gold", "pred_Total"], ascending=[False, False]).head(10)
    for _, r in top.iterrows():
        plt.text(r["pred_Gold"], r["pred_Total"], str(r["NOC"]))
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_delta_total(df_merge, outpath: Path):
    df = df_merge.copy()
    df["delta_total"] = df["pred_Total"] - df["Total_2024"]
    up = df.sort_values("delta_total", ascending=False).head(10)
    down = df.sort_values("delta_total", ascending=True).head(10)
    show = pd.concat([up, down], axis=0).sort_values("delta_total", ascending=False)

    plt.figure()
    x = np.arange(len(show))
    plt.bar(x, show["delta_total"].values)
    plt.xticks(x, show["NOC"].values, rotation=45, ha="right")
    plt.ylabel("ΔTotal = Expected(2028) - Actual(2024)")
    plt.title("Most Improved / Declined (Total Medals)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_uncertainty_vs_level_mu(df_pred, outpath: Path):
    df = df_pred.copy()
    df["w_total_mu"] = df["mu_high_Total"] - df["mu_low_Total"]
    plt.figure()
    plt.scatter(df["pred_Total"], df["w_total_mu"])
    plt.xlabel("Expected Total (2028)")
    plt.ylabel("CI Width (Total, mean band)")
    plt.title("Uncertainty vs Expected Level (Mean Band)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def process_one(pred_path: Path, tag: str):
    # create subfolders so host0 and host1 won't overwrite each other
    fig_sub = FIG_DIR / f"A1_{tag}"
    tab_sub = TAB_DIR / f"A1_{tag}"
    fig_sub.mkdir(parents=True, exist_ok=True)
    tab_sub.mkdir(parents=True, exist_ok=True)

    df_pred = pd.read_csv(pred_path)

    # merge 2024 actual
    medals = pd.read_csv(RAW_DIR / "summerOly_medal_counts.csv")
    medals["Year"] = pd.to_numeric(medals["Year"], errors="coerce").astype(int)
    m2024 = medals[medals["Year"] == 2024].copy()
    m2024 = m2024[["NOC", "Gold", "Total"]].rename(columns={"Gold": "Gold_2024", "Total": "Total_2024"})

    df_merge = df_pred.merge(m2024, on="NOC", how="left")
    df_merge["Gold_2024"] = df_merge["Gold_2024"].fillna(0)
    df_merge["Total_2024"] = df_merge["Total_2024"].fillna(0)

    # ---- Tables (MU bands) ----
    t1 = make_rank_table_mu(df_pred, topn=20)
    t1.to_csv(tab_sub / "Table_A1_Top20_2028_MU.csv", index=False)

    df_merge["delta_gold"] = df_merge["pred_Gold"] - df_merge["Gold_2024"]
    df_merge["delta_total"] = df_merge["pred_Total"] - df_merge["Total_2024"]
    t2_up = df_merge.sort_values("delta_total", ascending=False)[
        ["NOC", "Total_2024", "pred_Total", "delta_total", "Gold_2024", "pred_Gold", "delta_gold"]
    ].head(10)
    t2_dn = df_merge.sort_values("delta_total", ascending=True)[
        ["NOC", "Total_2024", "pred_Total", "delta_total", "Gold_2024", "pred_Gold", "delta_gold"]
    ].head(10)
    pd.concat([t2_up, t2_dn], axis=0).to_csv(tab_sub / "Table_A1_DeltaTopBottom_2028_vs_2024.csv", index=False)

    df_pred["w_total_mu"] = df_pred["mu_high_Total"] - df_pred["mu_low_Total"]
    t3 = df_pred.sort_values("w_total_mu", ascending=False)[
        ["NOC", "pred_Total", "mu_low_Total", "mu_high_Total", "w_total_mu"]
    ].head(15)
    t3.to_csv(tab_sub / "Table_A1_UncertaintyTop15_Total_MU.csv", index=False)

    # ---- Figures (MU bands) ----
    plot_top15_gold_mu_errorbar(df_pred, fig_sub / "Fig_A1_Top15_Gold_MU_CI.png")
    plot_gold_vs_total(df_pred, fig_sub / "Fig_A1_Gold_vs_Total.png")
    plot_delta_total(df_merge, fig_sub / "Fig_A1_DeltaTotal_2028_vs_2024.png")
    plot_uncertainty_vs_level_mu(df_pred, fig_sub / "Fig_A1_Uncertainty_vs_Level_MU.png")

    print(f"[{tag}] Done. Output -> {tab_sub} and {fig_sub}")
    print(f"Using prediction file: {pred_path.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["both", "host0", "host1"], default="both")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ["both", "host0"]:
        p0 = latest_file(TAB_DIR, "a1_pred_2028_host0_B*_*.csv")
        process_one(p0, "host0")

    if args.mode in ["both", "host1"]:
        p1 = latest_file(TAB_DIR, "a1_pred_2028_host1USA_B*_*.csv")
        process_one(p1, "host1USA")

if __name__ == "__main__":
    main()