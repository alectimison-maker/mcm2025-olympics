import re
import numpy as np
import pandas as pd
from .io_utils import read_csv_flex

NBSP = "\xa0"

def _clean_text(x):
    if pd.isna(x):
        return x
    s = str(x).replace(NBSP, " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def clean_hosts(df_hosts):
    df = df_hosts.copy()
    df["Host"] = df["Host"].map(_clean_text)
    # remove parentheses notes
    df["Host_clean"] = df["Host"].str.replace(r"\(.*?\)", "", regex=True).str.strip()
    df["Host_clean"] = df["Host_clean"].str.replace(r",\s*$", "", regex=True)
    # split city/country by last comma
    def split_city_country(s):
        if not isinstance(s, str) or "," not in s:
            return (np.nan, _clean_text(s))
        parts = [p.strip() for p in s.split(",")]
        country = parts[-1]
        city = ", ".join(parts[:-1]).strip()
        return (city, country)

    tmp = df["Host_clean"].apply(split_city_country)
    df["Host_city"] = tmp.apply(lambda z: z[0])
    df["Host_country"] = tmp.apply(lambda z: z[1])

    # common inconsistency in datasets: UK vs Great Britain
    df["Host_country"] = df["Host_country"].replace({"United Kingdom": "Great Britain"})
    return df

def clean_medals(df_medals):
    df = df_medals.copy()
    for c in ["NOC", "Team"]:
        if c in df.columns:
            df[c] = df[c].map(_clean_text)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    for c in ["Gold", "Silver", "Bronze", "Total", "Rank"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # ensure integer counts (keep NaN if any)
    for c in ["Gold", "Silver", "Bronze", "Total"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    # consistency check
    df["Total_check"] = df["Gold"] + df["Silver"] + df["Bronze"]
    df["Total_mismatch"] = (df["Total"] != df["Total_check"]).astype(int)
    return df

def normalize_team(team, valid_country_set):
    """
    Implements problem note: Team may contain more detail (e.g., Germany-1).
    If base name exists in valid_country_set, map to base; otherwise keep original.
    """
    if not isinstance(team, str):
        return team
    s = _clean_text(team)
    m = re.match(r"^(.*?)-(\d+)$", s)
    if m:
        base = m.group(1).strip()
        if base in valid_country_set:
            return base
    return s

def clean_athletes(df_ath, valid_country_set):
    df = df_ath.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(_clean_text)

    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)

    # keep NOC as primary country key; normalize Team for descriptive checks
    if "Team" in df.columns:
        df["Team_norm"] = df["Team"].apply(lambda x: normalize_team(x, valid_country_set))
    return df

def extract_events(programs_df):
    df = programs_df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(_clean_text)

    year_cols = [c for c in df.columns if re.match(r"^\d{4}", str(c))]
    # Total events row
    tot = df[df["Discipline"].str.lower() == "total events"].copy()
    total_events = tot.melt(
        id_vars=["Sport","Discipline","Code","Sports Governing Body"],
        value_vars=year_cols,
        var_name="Year",
        value_name="TotalEvents"
    )
    total_events["Year"] = total_events["Year"].str.replace("*", "", regex=False).astype(int)
    total_events["TotalEvents"] = pd.to_numeric(total_events["TotalEvents"], errors="coerce").fillna(0).astype(int)
    total_events = total_events[["Year","TotalEvents"]].sort_values("Year")

    # events by sport-year (using Sport rows where Discipline not "Total events")
    bysport = df[df["Discipline"].str.lower() != "total events"].copy()
    bysport_long = bysport.melt(
        id_vars=["Sport","Discipline"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Events"
    )
    bysport_long["Year"] = bysport_long["Year"].str.replace("*", "", regex=False).astype(int)
    bysport_long["Events"] = pd.to_numeric(bysport_long["Events"], errors="coerce").fillna(0).astype(int)
    # aggregate to Sport level
    bysport_long = bysport_long.groupby(["Sport","Year"], as_index=False)["Events"].sum()
    return total_events, bysport_long

def write_audit_report(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")