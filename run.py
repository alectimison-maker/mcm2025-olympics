import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import RAW_DIR, PROCESSED_DIR, AUDIT_DIR, FIG_DIR, TAB_DIR, stamp
from src.io_utils import read_csv_flex
from src.clean_data import clean_hosts, clean_medals, clean_athletes, extract_events, write_audit_report
from src.build_features import build_country_year_features, build_country_sport_year_medals, build_sport_share
from src.eda_plots import plot_events_vs_gold, plot_host_boost, plot_medal_hist_2024
from src.model_a1_nb import bootstrap_prediction_interval


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
    sport_share = build_sport_share(country_sport_year, window_years=5)

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


# -------------------- NEW HELPERS FOR A1 (KEY FIX) --------------------
def _clean_host_country(x: str) -> str:
    if x is None:
        return ""
    s = str(x).replace("\u00a0", " ").strip()
    s = s.rstrip(",").strip()
    # remove parenthesis tail: "Japan (postponed ...)" -> "Japan"
    s = s.split("(")[0].strip().rstrip(",").strip()
    return s

def _host_year_to_country_from_hosts_df(hosts_df: pd.DataFrame) -> dict:
    """
    Accept either:
      - cleaned hosts with column Host_country, or
      - raw hosts with column Host (city,country,...)
    Return: {Year: HostCountryString}
    """
    if "Host_country" in hosts_df.columns:
        m = hosts_df[["Year","Host_country"]].copy()
        m["Host_country"] = m["Host_country"].apply(_clean_host_country)
    elif "Host" in hosts_df.columns:
        m = hosts_df[["Year","Host"]].copy()
        def parse_country(host):
            s = str(host).replace("\u00a0"," ").strip()
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= 2:
                country = parts[-1]
            else:
                country = s
            return _clean_host_country(country)
        m["Host_country"] = m["Host"].apply(parse_country)
        m = m[["Year","Host_country"]]
    else:
        raise ValueError("Hosts dataframe missing Host_country/Host columns.")

    m["Year"] = pd.to_numeric(m["Year"], errors="coerce")
    m = m.dropna(subset=["Year"])
    m["Year"] = m["Year"].astype(int)

    out = {}
    for y, c in m.itertuples(index=False):
        if c and c.lower() != "cancelled":
            out[int(y)] = c
    return out

def _map_host_country_to_noc_label(host_country: str, noc_set: set) -> str | None:
    """
    Map host country string into the exact label used in df['NOC'].
    """
    if not host_country:
        return None

    # direct match
    if host_country in noc_set:
        return host_country

    # common aliases in this dataset
    alias = {
        "United Kingdom": "Great Britain",
        "UK": "Great Britain",
        "U.K.": "Great Britain",
        "Korea": "South Korea",
    }
    if host_country in alias and alias[host_country] in noc_set:
        return alias[host_country]

    # case-insensitive exact
    hc = host_country.strip().lower()
    for n in noc_set:
        if isinstance(n, str) and n.strip().lower() == hc:
            return n

    # substring fallback (safe only if unique)
    cands = [n for n in noc_set if isinstance(n, str) and (hc in n.lower() or n.lower() in hc)]
    if len(cands) == 1:
        return cands[0]

    return None

def _rebuild_is_host_for_train(df: pd.DataFrame, hosts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Overwrite df['is_host'] based on hosts list and df's own NOC labels.
    This is the core fix to make host effect learnable.
    """
    d = df.copy()
    noc_set = set(d["NOC"].dropna().unique().tolist())
    year_to_country = _host_year_to_country_from_hosts_df(hosts_df)

    d["is_host"] = 0
    mapped = 0
    set1 = 0

    for year, host_country in year_to_country.items():
        host_noc = _map_host_country_to_noc_label(host_country, noc_set)
        if host_noc is None:
            continue
        mapped += 1
        mask = (d["Year"] == year) & (d["NOC"] == host_noc)
        if mask.any():
            d.loc[mask, "is_host"] = 1
            set1 += int(mask.sum())

    print(f"[A1] Rebuilt is_host in training: mapped_years={mapped}, is_host_ones={int(d['is_host'].sum())}", flush=True)
    return d

def _guess_usa_label(noc_series: pd.Series) -> str:
    nocs = [x for x in noc_series.dropna().unique() if isinstance(x, str)]
    s = set([x.strip() for x in nocs])

    candidates = ["USA", "United States", "United States of America"]
    for c in candidates:
        if c in s:
            return c

    # fuzzy
    for x in nocs:
        xl = x.lower()
        if ("united" in xl) and ("states" in xl):
            return x

    raise ValueError("Cannot find USA label in NOC column.")


def stage_a1():
    import pandas as pd
    import numpy as np

    from src.config import PROCESSED_DIR, TAB_DIR, stamp
    from src.model_a1_nb import bootstrap_prediction_interval, fit_nb_glm

    TAB_DIR.mkdir(parents=True, exist_ok=True)

    def latest(prefix):
        files = sorted([p for p in PROCESSED_DIR.glob(f"{prefix}_*.csv")])
        if not files:
            raise FileNotFoundError(f"No processed files for prefix {prefix}. Run: python run.py features")
        return files[-1]

    def _norm_country(x: str) -> str:
        if x is None:
            return ""
        s = str(x).replace("\u00a0", " ").strip()
        # remove anything in parentheses, e.g., "Japan (postponed to 2021 ...)"
        s = __import__("re").sub(r"\(.*?\)", "", s).strip()
        # collapse multiple spaces
        s = " ".join(s.split())
        return s.lower()

    def rebuild_is_host(df: pd.DataFrame, hosts_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Rebuild df['is_host'] from hosts_clean.
        hosts_clean expected columns: ['Year','Host_country'] (or similar).
        """
        d = df.copy()
        d["is_host"] = 0

        # detect columns
        year_col = "Year"
        host_col = None
        for c in hosts_clean.columns:
            if c.lower() in ["host_country", "hostcountry", "host_noc", "hostnoc"]:
                host_col = c
                break
        if host_col is None:
            # fallback: try parse from 'Host'
            if "Host" in hosts_clean.columns:
                host_col = "Host"
            else:
                raise ValueError("hosts_clean missing Host_country/Host column")

        h = hosts_clean.copy()
        h[year_col] = pd.to_numeric(h[year_col], errors="coerce")
        h = h.dropna(subset=[year_col])
        h[year_col] = h[year_col].astype(int)

        # parse host country if needed
        if host_col == "Host":
            tmp = h["Host"].astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
            tmp = tmp.str.split(",").str[-1].str.strip()
            tmp = tmp.str.replace(r"\(.*?\)", "", regex=True).str.strip()
            h["Host_country_parsed"] = tmp
            host_col_use = "Host_country_parsed"
        else:
            host_col_use = host_col
            h[host_col_use] = h[host_col_use].astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
            h[host_col_use] = h[host_col_use].str.replace(r"\(.*?\)", "", regex=True).str.strip()

        # build mapping from normalized NOC name -> original label
        noc_map = {}
        for v in d["NOC"].astype(str).tolist():
            noc_map[_norm_country(v)] = v

        mapped_years = 0
        set_ones = 0

        for _, r in h.iterrows():
            y = int(r[year_col])
            hc = str(r[host_col_use])
            key = _norm_country(hc)

            # some common aliases
            alias = {
                "united kingdom": "great britain",
                "uk": "great britain",
            }
            key = alias.get(key, key)

            if key in noc_map:
                host_label = noc_map[key]
            else:
                # fuzzy fallback: contains all words
                host_label = None
                for k, vv in noc_map.items():
                    if (len(key) >= 4) and (key in k):
                        host_label = vv
                        break
                if host_label is None:
                    continue

            # set host flag
            m = (d["Year"] == y) & (d["NOC"] == host_label)
            if m.any():
                mapped_years += 1
                before = int(d.loc[m, "is_host"].sum())
                d.loc[m, "is_host"] = 1
                after = int(d.loc[m, "is_host"].sum())
                set_ones += (after - before)

        print(f"[A1] Rebuilt is_host: mapped_years={mapped_years}, is_host_ones={int(d['is_host'].sum())}", flush=True)
        return d

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
        raise ValueError("Cannot identify USA NOC label from predictions (check df['NOC'].unique()).")

    # ---------- load features + rebuild is_host ----------
    df = pd.read_csv(latest("country_year_features"))
    hosts_clean = pd.read_csv(latest("hosts_clean"))
    df = rebuild_is_host(df, hosts_clean)

    # ---------- training set ----------
    last_year = 2024
    train = df[(df["Year"] <= last_year) & (df["Year"] >= 1952)].copy()

    # ---------- 2028 prediction frame (baseline from 2024) ----------
    base = train[train["Year"] == last_year].copy()
    pred_base = base.copy()
    pred_base["Year"] = 2028
    pred_base["TotalEvents"] = base["TotalEvents"].values  # keep same as 2024
    pred_base["is_host"] = 0  # host0 baseline

    # ---------- bootstrap settings ----------
    B = 200
    alpha = 1.0
    l2 = 1e-4
    ts = stamp()

    # ========== Scenario host0 ==========
    pred0 = pred_base.copy()
    pred_total_0 = bootstrap_prediction_interval(
        train_df=train, pred_df=pred0, target="Total", B=B, seed=42, alpha=alpha, l2=l2
    )
    pred_gold_0 = bootstrap_prediction_interval(
        train_df=train, pred_df=pred0, target="Gold", B=B, seed=43, alpha=alpha, l2=l2
    )
    out0 = pred_total_0.merge(
        pred_gold_0[["NOC","Year","pred_Gold","mu_low_Gold","mu_high_Gold","pi_low_Gold","pi_high_Gold"]],
        on=["NOC","Year"], how="left"
    )
    out0_path = TAB_DIR / f"a1_pred_2028_host0_B{B}_{ts}.csv"
    out0.to_csv(out0_path, index=False)
    print("Saved scenario host=0:", out0_path, flush=True)

    # ========== Scenario host1 (USA host) with SHRUNK multiplier ==========
    # Fit once on full train to extract host coefficient
    m_total = fit_nb_glm(train, target="Total", alpha=alpha, l2=l2)
    m_gold  = fit_nb_glm(train, target="Gold",  alpha=alpha, l2=l2)

    def get_beta_is_host(m):
        params = m["res"].params
        if hasattr(params, "index") and ("is_host" in list(params.index)):
            return float(params["is_host"])
        # fallback: by column index
        cols = m["columns"]
        return float(params[cols.index("is_host")])

    beta_host_total = get_beta_is_host(m_total)
    beta_host_gold  = get_beta_is_host(m_gold)

    mult_total_raw = float(np.exp(beta_host_total))
    mult_gold_raw  = float(np.exp(beta_host_gold))

    # --- key: cap (shrink) to avoid overfit explosion ---
    CAP_TOTAL = 1.25   # max +25% total medals
    CAP_GOLD  = 1.35   # max +35% gold medals
    mult_total = min(mult_total_raw, CAP_TOTAL)
    mult_gold  = min(mult_gold_raw,  CAP_GOLD)

    print(
        f"[A1] Host multiplier raw: Total={mult_total_raw:.3f}, Gold={mult_gold_raw:.3f} | "
        f"capped: Total={mult_total:.3f}, Gold={mult_gold:.3f}",
        flush=True
    )

    out1 = out0.copy()
    usa_label = guess_usa_noc(out1["NOC"])
    mask_usa = (out1["NOC"] == usa_label)

    # apply multiplier to USA rows only (pred + mean-band CI; pi columns if present are scaled as approximation)
    scale_cols_total = ["pred_Total","mu_low_Total","mu_high_Total","pi_low_Total","pi_high_Total"]
    scale_cols_gold  = ["pred_Gold","mu_low_Gold","mu_high_Gold","pi_low_Gold","pi_high_Gold"]

    for c in scale_cols_total:
        if c in out1.columns:
            out1.loc[mask_usa, c] = out1.loc[mask_usa, c].astype(float) * mult_total
    for c in scale_cols_gold:
        if c in out1.columns:
            out1.loc[mask_usa, c] = out1.loc[mask_usa, c].astype(float) * mult_gold

    out1_path = TAB_DIR / f"a1_pred_2028_host1USA_B{B}_{ts}.csv"
    out1.to_csv(out1_path, index=False)
    print(f"Saved scenario host=1 (USA='{usa_label}', capped):", out1_path, flush=True)

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