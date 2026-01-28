import numpy as np
import pandas as pd

def build_country_year_features(medals_df, hosts_df, total_events_df):
    """
    Output columns:
      Year, NOC, Gold, Total, is_host, TotalEvents, lag_total_1, lag_total_2, lag_gold_1, lag_gold_2
    """
    df = medals_df.copy()
    # host mapping by year -> host country name
    host_map = hosts_df[["Year","Host_country"]].copy()
    host_map = host_map[host_map["Host_country"] != "Cancelled"]
    host_map = host_map.rename(columns={"Host_country":"HostCountry"})

    # merge events
    df = df.merge(total_events_df, on="Year", how="left")
    df = df.merge(host_map, on="Year", how="left")

    # host indicator: compare NOC to host country name is not reliable across name mapping,
    # so we use the dataset's country name alignment strategy:
    # if medal_counts has "Team" (country name), compare to HostCountry; else keep only NOC-level and host NOC can be manually set later.
    if "Team" in df.columns:
        df["is_host"] = (df["Team"] == df["HostCountry"]).astype(int)
    else:
        df["is_host"] = 0

    df = df.sort_values(["NOC","Year"])
    # lags
    df["lag_total_1"] = df.groupby("NOC")["Total"].shift(1)
    df["lag_total_2"] = df.groupby("NOC")["Total"].shift(2)
    df["lag_gold_1"]  = df.groupby("NOC")["Gold"].shift(1)
    df["lag_gold_2"]  = df.groupby("NOC")["Gold"].shift(2)

    # fill missing lags with 0 for early years (conservative)
    for c in ["lag_total_1","lag_total_2","lag_gold_1","lag_gold_2"]:
        df[c] = df[c].fillna(0).astype(int)

    keep = ["Year","NOC","Gold","Total","is_host","TotalEvents","lag_total_1","lag_total_2","lag_gold_1","lag_gold_2"]
    return df[keep]

def build_country_sport_year_medals(athletes_df):
    """
    Build country-sport-year medal counts from athletes records.
    Requires columns: Year, NOC, Sport, Medal (Gold/Silver/Bronze/NA).
    Output: Year, NOC, Sport, Medals (count), Gold (count), TotalPoints (3-2-1)
    """
    df = athletes_df.copy()
    required = ["Year","NOC","Sport","Medal"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"athletes_df missing column: {c}")

    df = df[required].copy()
    df = df[df["Medal"].notna()]
    df["Medal"] = df["Medal"].astype(str)

    df["is_gold"] = (df["Medal"].str.lower() == "gold").astype(int)
    df["points"] = df["Medal"].str.lower().map({"gold":3, "silver":2, "bronze":1}).fillna(0).astype(int)

    agg = df.groupby(["Year","NOC","Sport"], as_index=False).agg(
        Medals=("Medal","count"),
        Gold=("is_gold","sum"),
        TotalPoints=("points","sum"),
    )
    return agg

def build_sport_share(country_sport_year_df, window_years=None):
    """
    D1 share method: pi_{i,s} = medals_{i,s} / medals_i
    If window_years is not None, use only last N years (by olympic editions) to compute.
    """
    df = country_sport_year_df.copy()
    if window_years is not None:
        years = sorted(df["Year"].unique())
        keep_years = years[-window_years:]
        df = df[df["Year"].isin(keep_years)]

    denom = df.groupby(["NOC"], as_index=False)["Medals"].sum().rename(columns={"Medals":"Medals_all"})
    df2 = df.groupby(["NOC","Sport"], as_index=False)["Medals"].sum()
    df2 = df2.merge(denom, on="NOC", how="left")
    df2["pi"] = df2["Medals"] / df2["Medals_all"].replace(0, np.nan)
    df2["pi"] = df2["pi"].fillna(0.0)
    return df2[["NOC","Sport","pi","Medals","Medals_all"]]