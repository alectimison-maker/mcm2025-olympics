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

def load_scenarios():
    p0 = latest_file(TAB_DIR, "a1_pred_2028_host0_B*_*.csv")
    p1 = latest_file(TAB_DIR, "a1_pred_2028_host1USA_B*_*.csv")
    d0 = pd.read_csv(p0)
    d1 = pd.read_csv(p1)
    return p0, p1, d0, d1

def load_baseline_avg(years=(2016, 2020, 2024)):
    medals = pd.read_csv(RAW_DIR / "summerOly_medal_counts.csv")
    medals["Year"] = pd.to_numeric(medals["Year"], errors="coerce").astype(int)
    base = medals[medals["Year"].isin(list(years))].copy()
    avg = base.groupby("NOC", as_index=False).agg(
        Gold_avg=("Gold", "mean"),
        Total_avg=("Total", "mean"),
    )
    return avg

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
    raise ValueError("Cannot identify USA NOC label from predictions. Inspect df['NOC'].unique().")

def host_effect_usa_only(d0, d1, out_csv: Path):
    host_noc = guess_usa_noc(d0["NOC"])
    print(f"[INFO] Detected USA NOC label as: {host_noc}", flush=True)

    m = d0.merge(d1, on=["NOC", "Year"], suffixes=("_host0","_host1"))

    others = m[m["NOC"] != host_noc].copy()
    if len(others):
        others["abs_delta_total"] = (others["pred_Total_host1"] - others["pred_Total_host0"]).abs()
        others["abs_delta_gold"]  = (others["pred_Gold_host1"] - others["pred_Gold_host0"]).abs()
        max_other_total = float(others["abs_delta_total"].max())
        max_other_gold  = float(others["abs_delta_gold"].max())
    else:
        max_other_total = 0.0
        max_other_gold = 0.0

    usa = m[m["NOC"] == host_noc].copy()
    if usa.empty:
        raise ValueError(f"{host_noc} not found after merge. Check prediction files.")

    usa["delta_host_gold"]  = usa["pred_Gold_host1"]  - usa["pred_Gold_host0"]
    usa["delta_host_total"] = usa["pred_Total_host1"] - usa["pred_Total_host0"]

    # mean-band CI delta (conservative)
    for t in ["Gold", "Total"]:
        lo0, hi0 = f"mu_low_{t}_host0", f"mu_high_{t}_host0"
        lo1, hi1 = f"mu_low_{t}_host1", f"mu_high_{t}_host1"
        if all(c in usa.columns for c in [lo0, hi0, lo1, hi1]):
            usa[f"delta_ci_low_{t}"]  = usa[lo1] - usa[hi0]
            usa[f"delta_ci_high_{t}"] = usa[hi1] - usa[lo0]

    out_cols = [c for c in [
        "NOC","Year",
        "pred_Gold_host0","mu_low_Gold_host0","mu_high_Gold_host0",
        "pred_Total_host0","mu_low_Total_host0","mu_high_Total_host0",
        "pred_Gold_host1","mu_low_Gold_host1","mu_high_Gold_host1",
        "pred_Total_host1","mu_low_Total_host1","mu_high_Total_host1",
        "delta_host_gold","delta_host_total",
        "delta_ci_low_Gold","delta_ci_high_Gold",
        "delta_ci_low_Total","delta_ci_high_Total"
    ] if c in usa.columns]

    out = usa[out_cols].copy()
    out.to_csv(out_csv, index=False)

    if max_other_total > 1e-6 or max_other_gold > 1e-6:
        print(
            f"[WARN] Non-USA rows differ between host0 and host1 "
            f"(max |ΔTotal|={max_other_total:.4g}, max |ΔGold|={max_other_gold:.4g}).",
            flush=True
        )
    else:
        print("[OK] Only USA changed between host0 and host1.", flush=True)

    return out

def plot_host_effect_usa(out_usa: pd.DataFrame, outpath: Path):
    r = out_usa.iloc[0]
    vals = [float(r["delta_host_gold"]), float(r["delta_host_total"])]
    labels = ["ΔGold (host1-host0)", "ΔTotal (host1-host0)"]

    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, vals)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Change in expected medals")
    plt.title("USA Host Effect (Expected Medals)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def table_delta_vs_avg(d_pred, avg, outpath):
    m = d_pred.merge(avg, on="NOC", how="left")
    m["Gold_avg"] = m["Gold_avg"].fillna(0)
    m["Total_avg"] = m["Total_avg"].fillna(0)
    m["delta_total_vs_avg"] = m["pred_Total"] - m["Total_avg"]
    m["delta_gold_vs_avg"] = m["pred_Gold"] - m["Gold_avg"]

    up = m.sort_values("delta_total_vs_avg", ascending=False).head(10)
    dn = m.sort_values("delta_total_vs_avg", ascending=True).head(10)
    out = pd.concat([up, dn], axis=0)[
        ["NOC","Total_avg","pred_Total","delta_total_vs_avg","Gold_avg","pred_Gold","delta_gold_vs_avg"]
    ]
    out.to_csv(outpath, index=False)
    return out

def plot_delta_vs_avg_bar(out, outpath, years_label="Avg(2016,2020,2024)"):
    df = out.copy().sort_values("delta_total_vs_avg", ascending=False)
    plt.figure()
    x = np.arange(len(df))
    plt.bar(x, df["delta_total_vs_avg"].values)
    plt.xticks(x, df["NOC"].values, rotation=45, ha="right")
    plt.ylabel(f"ΔTotal = Pred(2028) - {years_label}")
    plt.title("Most Improved / Declined vs Multi-Games Average")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["host0","host1"], default="host0")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    p0, p1, d0, d1 = load_scenarios()
    avg = load_baseline_avg()

    out_csv = TAB_DIR / "A1_HostEffect_USA_only.csv"
    usa_out = host_effect_usa_only(d0, d1, out_csv)
    plot_host_effect_usa(usa_out, FIG_DIR / "Fig_A1_HostEffect_USA_only.png")

    d = d0 if args.which == "host0" else d1
    tag = args.which
    out_delta = table_delta_vs_avg(d, avg, TAB_DIR / f"A1_Delta_vs_Avg_{tag}.csv")
    plot_delta_vs_avg_bar(out_delta, FIG_DIR / f"Fig_A1_Delta_vs_Avg_{tag}.png")

    print("Diagnostics generated:")
    print(" -", out_csv.name)
    print(" - Fig_A1_HostEffect_USA_only.png")
    print(" -", f"A1_Delta_vs_Avg_{tag}.csv")
    print(" -", f"Fig_A1_Delta_vs_Avg_{tag}.png")
    print("Using:", p0.name, "and", p1.name)

if __name__ == "__main__":
    main()