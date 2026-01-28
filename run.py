import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import RAW_DIR, PROCESSED_DIR, AUDIT_DIR, FIG_DIR, TAB_DIR, stamp
from src.io_utils import read_csv_flex
from src.clean_data import clean_hosts, clean_medals, clean_athletes, extract_events, write_audit_report
from src.build_features import build_country_year_features, build_country_sport_year_medals, build_sport_share
from src.eda_plots import plot_events_vs_gold, plot_host_boost, plot_medal_hist_2024
from src.model_a1_nb import fit_nb_glm, bootstrap_prediction_interval

def ensure_dirs():
    for p in [PROCESSED_DIR, AUDIT_DIR, FIG_DIR, TAB_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def stage_clean():
    ensure_dirs()
    ts = stamp()

    hosts = read_csv_flex(RAW_DIR / "summerOly_hosts.csv")
    medals = read_csv_flex(RAW_DIR / "summerOly_medal_counts.csv")
    athletes = read_csv_flex(RAW_DIR / "summerOly_athletes.csv")
    programs = read_csv_flex(RAW_DIR / "summerOly_programs.csv")

    hosts_c = clean_hosts(hosts)
    medals_c = clean_medals(medals)

    valid_countries = set()
    if "Team" in medals_c.columns:
        valid_countries |= set(medals_c["Team"].dropna().unique().tolist())
    valid_countries |= set(hosts_c["Host_country"].dropna().unique().tolist())

    athletes_c = clean_athletes(athletes, valid_countries)
    total_events, events_by_sport_year = extract_events(programs)

    # save
    hosts_path = PROCESSED_DIR / f"hosts_clean_{ts}.csv"
    medals_path = PROCESSED_DIR / f"medals_clean_{ts}.csv"
    athletes_path = PROCESSED_DIR / f"athletes_clean_{ts}.csv"
    te_path = PROCESSED_DIR / f"total_events_{ts}.csv"
    es_path = PROCESSED_DIR / f"events_by_sport_year_{ts}.csv"

    hosts_c.to_csv(hosts_path, index=False)
    medals_c.to_csv(medals_path, index=False)
    athletes_c.to_csv(athletes_path, index=False)
    total_events.to_csv(te_path, index=False)
    events_by_sport_year.to_csv(es_path, index=False)

    # audit
    lines = []
    lines.append(f"[timestamp] {ts}")
    lines.append(f"hosts rows: {len(hosts)} -> {len(hosts_c)}")
    lines.append(f"medals rows: {len(medals)} -> {len(medals_c)}")
    lines.append(f"athletes rows: {len(athletes)} -> {len(athletes_c)}")
    if "Total_mismatch" in medals_c.columns:
        lines.append(f"medal total mismatches: {int(medals_c['Total_mismatch'].sum())}")
    lines.append("note: Team_norm created for athletes where applicable (e.g., Germany-1 -> Germany).")
    write_audit_report(AUDIT_DIR / f"clean_audit_{ts}.txt", lines)

    print("Saved:")
    print(hosts_path, medals_path, athletes_path, te_path, es_path, sep="\n")

def stage_features(latest_ts=None):
    ensure_dirs()

    # pick latest clean files automatically
    def latest(prefix):
        files = sorted([p for p in PROCESSED_DIR.glob(f"{prefix}_*.csv")])
        if not files:
            raise FileNotFoundError(f"No processed files for prefix {prefix}. Run: python run.py clean")
        return files[-1]

    hosts = pd.read_csv(latest("hosts_clean"))
    medals = pd.read_csv(latest("medals_clean"))
    athletes = pd.read_csv(latest("athletes_clean"))
    total_events = pd.read_csv(latest("total_events"))
    events_by_sport_year = pd.read_csv(latest("events_by_sport_year"))

    country_year = build_country_year_features(medals, hosts, total_events)
    country_sport_year = build_country_sport_year_medals(athletes)
    sport_share = build_sport_share(country_sport_year, window_years=5)  # last 5 editions

    ts = stamp()
    cy_path = PROCESSED_DIR / f"country_year_features_{ts}.csv"
    csy_path = PROCESSED_DIR / f"country_sport_year_medals_{ts}.csv"
    share_path = PROCESSED_DIR / f"sport_share_{ts}.csv"

    country_year.to_csv(cy_path, index=False)
    country_sport_year.to_csv(csy_path, index=False)
    sport_share.to_csv(share_path, index=False)
    events_by_sport_year.to_csv(PROCESSED_DIR / f"events_by_sport_year_{ts}.csv", index=False)

    print("Saved:")
    print(cy_path, csy_path, share_path, sep="\n")

def stage_eda():
    ensure_dirs()

    def latest(prefix):
        files = sorted([p for p in PROCESSED_DIR.glob(f"{prefix}_*.csv")])
        if not files:
            raise FileNotFoundError(f"No processed files for prefix {prefix}. Run stages first.")
        return files[-1]

    medals = pd.read_csv(latest("medals_clean"))
    total_events = pd.read_csv(latest("total_events"))

    year_stats = medals.groupby("Year", as_index=False).agg(
        GoldMedals=("Gold","sum"),
        TotalMedals=("Total","sum"),
        Countries=("NOC","nunique"),
    ).merge(total_events, on="Year", how="left")

    # host boost proxy: requires Team and host mapping; if missing, skip
    # Here we compute host boost using medal table 'Team' and latest hosts file if present
    hosts = pd.read_csv(latest("hosts_clean"))
    if "Team" in medals.columns:
        host_map = hosts[["Year","Host_country"]].copy()
        host_map = host_map[host_map["Host_country"] != "Cancelled"]
        m2 = medals.merge(host_map, on="Year", how="left")
        m2["is_host"] = (m2["Team"] == m2["Host_country"]).astype(int)

        host_years = m2[m2["is_host"]==1][["Year","NOC","Total"]].drop_duplicates().sort_values("Year")
        def prev2_mean(noc, year):
            prev = m2[(m2["NOC"]==noc) & (m2["Year"]<year)].sort_values("Year").tail(2)
            return prev["Total"].mean() if len(prev) else np.nan

        host_boost = host_years.copy()
        host_boost["Prev2Mean"] = [prev2_mean(n,y) for y,n,_ in host_years.itertuples(index=False)]
        host_boost["HostBoostTotal"] = host_boost["Total"] - host_boost["Prev2Mean"]
    else:
        host_boost = pd.DataFrame(columns=["Year","HostBoostTotal"])

    ts = stamp()
    plot_events_vs_gold(year_stats, save_path=FIG_DIR / f"eda_events_vs_gold_{ts}.png")
    if len(host_boost):
        plot_host_boost(host_boost, save_path=FIG_DIR / f"eda_host_boost_{ts}.png")
    m2024 = medals[medals["Year"]==2024].copy()
    plot_medal_hist_2024(m2024, save_path=FIG_DIR / f"eda_2024_hist_{ts}.png")

    year_stats.to_csv(TAB_DIR / f"year_stats_{ts}.csv", index=False)
    if len(host_boost):
        host_boost.to_csv(TAB_DIR / f"host_boost_{ts}.csv", index=False)

    print("EDA saved to outputs/figures and outputs/tables.")

def stage_a1():
    ensure_dirs()

    def latest(prefix):
        files = sorted([p for p in PROCESSED_DIR.glob(f"{prefix}_*.csv")])
        if not files:
            raise FileNotFoundError(f"No processed files for prefix {prefix}. Run: python run.py features")
        return files[-1]

    df = pd.read_csv(latest("country_year_features"))
    # baseline: train on all years <=2024, predict 2028 scenario using 2024 events as default
    # build 2028 placeholder rows for each NOC
    last_year = 2024
    base = df[df["Year"]==last_year].copy()
    pred = base.copy()
    pred["Year"] = 2028
    pred["TotalEvents"] = base["TotalEvents"].values  # scenario: keep same as 2024
    pred["is_host"] = 0
    # If you want host USA: set after you define NOC for USA, here placeholder:
    # pred.loc[pred["NOC"]=="USA","is_host"]=1

    train = df[df["Year"]<=last_year].copy()

    # Total model
    model_total = fit_nb_glm(train, target="Total")
    pred_total = bootstrap_prediction_interval(fit_nb_glm, train, pred, target="Total", B=200, seed=42)

    # Gold model
    model_gold = fit_nb_glm(train, target="Gold")
    pred_gold = bootstrap_prediction_interval(fit_nb_glm, train, pred, target="Gold", B=200, seed=43)

    out = pred_total.merge(pred_gold[["NOC","Year","pred_Gold","pi_low_Gold","pi_high_Gold"]], on=["NOC","Year"], how="left")
    ts = stamp()
    out_path = TAB_DIR / f"a1_pred_2028_baseline_{ts}.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["clean","features","eda","a1"])
    args = parser.parse_args()

    if args.stage == "clean":
        stage_clean()
    elif args.stage == "features":
        stage_features()
    elif args.stage == "eda":
        stage_eda()
    elif args.stage == "a1":
        stage_a1()

if __name__ == "__main__":
    main()