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

def load_pred(which: str) -> pd.DataFrame:
    if which == "host0":
        p = latest_file(TAB_DIR, "a1_pred_2028_host0_B*_*.csv")
    elif which == "host1":
        p = latest_file(TAB_DIR, "a1_pred_2028_host1USA_B*_*.csv")
    else:
        raise ValueError("which must be host0 or host1")
    df = pd.read_csv(p)
    return df

def guess_usa_noc(noc_series: pd.Series) -> str:
    nocs = [x for x in noc_series.dropna().unique() if isinstance(x, str)]
    s = set(nocs)
    for c in ["USA", "United States", "United States of America"]:
        if c in s:
            return c
    for x in nocs:
        xl = x.lower()
        if ("united" in xl) and ("states" in xl):
            return x
    raise ValueError("Cannot identify USA NOC label from predictions.")

def load_avg(years) -> pd.DataFrame:
    medals = pd.read_csv(RAW_DIR / "summerOly_medal_counts.csv")
    medals["Year"] = pd.to_numeric(medals["Year"], errors="coerce").astype(int)
    base = medals[medals["Year"].isin(list(years))].copy()
    avg = base.groupby("NOC", as_index=False).agg(
        Gold_avg=("Gold", "mean"),
        Total_avg=("Total", "mean"),
    )
    return avg

def delta_vs_avg(pred: pd.DataFrame, avg: pd.DataFrame) -> pd.DataFrame:
    m = pred.merge(avg, on="NOC", how="left")
    m["Gold_avg"] = m["Gold_avg"].fillna(0.0)
    m["Total_avg"] = m["Total_avg"].fillna(0.0)
    m["delta_total_vs_avg"] = m["pred_Total"] - m["Total_avg"]
    m["delta_gold_vs_avg"] = m["pred_Gold"] - m["Gold_avg"]
    return m

def pick_top_bottom(df: pd.DataFrame, k=10, col="delta_total_vs_avg") -> pd.DataFrame:
    up = df.sort_values(col, ascending=False).head(k)
    dn = df.sort_values(col, ascending=True).head(k)
    out = pd.concat([up, dn], axis=0)
    return out

def plot_bar(df: pd.DataFrame, ycol: str, title: str, ylabel: str, outpath: Path):
    df2 = df.copy().sort_values(ycol, ascending=False)
    plt.figure()
    x = np.arange(len(df2))
    plt.bar(x, df2[ycol].values)
    plt.xticks(x, df2["NOC"].values, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["host0", "host1"], default="host1")
    parser.add_argument("--baseline", choices=["2016_2020_2024", "2016_2024", "2012_2016_2020_2024"],
                        default="2016_2020_2024")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    if args.baseline == "2016_2020_2024":
        years = (2016, 2020, 2024)
    elif args.baseline == "2016_2024":
        years = (2016, 2024)
    else:
        years = (2012, 2016, 2020, 2024)

    pred = load_pred(args.which)
    avg = load_avg(years)
    m = delta_vs_avg(pred, avg)

    usa = guess_usa_noc(m["NOC"])

    # ----- Version A: include USA (show the shock) -----
    outA = pick_top_bottom(m, k=args.k, col="delta_total_vs_avg")
    outA_path = TAB_DIR / f"A1_Delta_vs_Avg_{args.which}_{args.baseline}_includeUSA.csv"
    outA.to_csv(outA_path, index=False)

    plot_bar(
        outA,
        ycol="delta_total_vs_avg",
        title=f"Most Improved / Declined vs Avg{years} ({args.which}, include USA)",
        ylabel=f"ΔTotal = Pred(2028) - Avg{years}",
        outpath=FIG_DIR / f"Fig_A1_Delta_vs_Avg_{args.which}_{args.baseline}_includeUSA.png",
    )

    # ----- Version B: exclude USA (better readability for others) -----
    m_no = m[m["NOC"] != usa].copy()
    outB = pick_top_bottom(m_no, k=args.k, col="delta_total_vs_avg")
    outB_path = TAB_DIR / f"A1_Delta_vs_Avg_{args.which}_{args.baseline}_excludeUSA.csv"
    outB.to_csv(outB_path, index=False)

    plot_bar(
        outB,
        ycol="delta_total_vs_avg",
        title=f"Most Improved / Declined vs Avg{years} ({args.which}, exclude USA)",
        ylabel=f"ΔTotal = Pred(2028) - Avg{years}",
        outpath=FIG_DIR / f"Fig_A1_Delta_vs_Avg_{args.which}_{args.baseline}_excludeUSA.png",
    )

    print("Saved:")
    print(" -", outA_path.name)
    print(" -", outB_path.name)
    print(" -", f"Fig_A1_Delta_vs_Avg_{args.which}_{args.baseline}_includeUSA.png")
    print(" -", f"Fig_A1_Delta_vs_Avg_{args.which}_{args.baseline}_excludeUSA.png")
    print("USA label detected as:", usa)

if __name__ == "__main__":
    main()