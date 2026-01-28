import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TAB_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"

def latest_file(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern} in {folder}")
    return files[-1]

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
    raise ValueError("Cannot identify USA label from predictions (check NOC values).")

def parse_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps_total", type=str, default="1.15,1.20,1.25,1.30")
    parser.add_argument("--caps_gold", type=str, default="1.20,1.30,1.40,1.50")
    args = parser.parse_args()

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # baseline from latest host0 predictions (USA row)
    p0 = latest_file(TAB_DIR, "a1_pred_2028_host0_B*_*.csv")
    host0 = pd.read_csv(p0)

    usa = guess_usa_noc(host0["NOC"])
    row = host0[host0["NOC"] == usa].iloc[0]

    base_total = float(row["pred_Total"])
    base_gold = float(row["pred_Gold"])

    caps_total = sorted(parse_list(args.caps_total))
    caps_gold = sorted(parse_list(args.caps_gold))

    # build grid table
    records = []
    for ct in caps_total:
        for cg in caps_gold:
            total_host1 = base_total * ct
            gold_host1 = base_gold * cg
            records.append({
                "cap_total": ct,
                "cap_gold": cg,
                "USA_Total_host0": base_total,
                "USA_Gold_host0": base_gold,
                "USA_Total_host1": total_host1,
                "USA_Gold_host1": gold_host1,
                "delta_total": total_host1 - base_total,
                "delta_gold": gold_host1 - base_gold,
            })
    out = pd.DataFrame(records).sort_values(["cap_total", "cap_gold"])
    out_csv = TAB_DIR / "A1_CAP_Sensitivity_USA.csv"
    out.to_csv(out_csv, index=False)

    # ----------------------------
    # Fig 1: 1D sensitivity (Gold)
    # ----------------------------
    # use mid/first cap_total just for table lookup; delta_gold depends only on cap_gold
    gold_curve = pd.DataFrame({
        "cap_gold": caps_gold,
        "delta_gold": [base_gold * (cg - 1.0) for cg in caps_gold],
        "USA_Gold_host1": [base_gold * cg for cg in caps_gold],
    }).sort_values("cap_gold")

    plt.figure()
    plt.plot(gold_curve["cap_gold"].values, gold_curve["delta_gold"].values, marker="o")
    plt.xlabel("CAP_GOLD")
    plt.ylabel("ΔGold (USA host1 - host0)")
    plt.title("Sensitivity of USA Gold Host Effect to CAP_GOLD (1D)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig_A1_CAP_Sensitivity_USA_Gold_1D.png", dpi=200)
    plt.close()

    # -----------------------------
    # Fig 2: 1D sensitivity (Total)
    # -----------------------------
    total_curve = pd.DataFrame({
        "cap_total": caps_total,
        "delta_total": [base_total * (ct - 1.0) for ct in caps_total],
        "USA_Total_host1": [base_total * ct for ct in caps_total],
    }).sort_values("cap_total")

    plt.figure()
    plt.plot(total_curve["cap_total"].values, total_curve["delta_total"].values, marker="o")
    plt.xlabel("CAP_TOTAL")
    plt.ylabel("ΔTotal (USA host1 - host0)")
    plt.title("Sensitivity of USA Total Host Effect to CAP_TOTAL (1D)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig_A1_CAP_Sensitivity_USA_Total_1D.png", dpi=200)
    plt.close()

    # -------------------------------------------
    # Fig 3: Trade-off map (Total_host1 vs Gold_host1)
    # show multiple non-overlapping lines by fixing cap_gold
    # -------------------------------------------
    plt.figure()
    for cg in caps_gold:
        df = out[out["cap_gold"] == cg].sort_values("cap_total")
        plt.plot(df["USA_Gold_host1"].values, df["USA_Total_host1"].values, marker="o", label=f"cap_gold={cg}")
    plt.xlabel("USA Gold (host1)")
    plt.ylabel("USA Total (host1)")
    plt.title("CAP Grid Trade-off: USA Total vs Gold (host1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig_A1_CAP_Tradeoff_USA_Total_vs_Gold.png", dpi=200)
    plt.close()

    print("Saved:")
    print(" -", out_csv.name)
    print(" - Fig_A1_CAP_Sensitivity_USA_Gold_1D.png")
    print(" - Fig_A1_CAP_Sensitivity_USA_Total_1D.png")
    print(" - Fig_A1_CAP_Tradeoff_USA_Total_vs_Gold.png")
    print("USA label:", usa)
    print("host0 baseline: Total=", base_total, "Gold=", base_gold)

if __name__ == "__main__":
    main()