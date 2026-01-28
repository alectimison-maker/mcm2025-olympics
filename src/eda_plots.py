import matplotlib.pyplot as plt

def plot_events_vs_gold(year_stats_df, save_path=None):
    plt.figure()
    plt.plot(year_stats_df["Year"], year_stats_df["TotalEvents"], label="Total events")
    plt.plot(year_stats_df["Year"], year_stats_df["GoldMedals"], label="Total gold medals")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.title("Events vs Gold Medals Over Time")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_host_boost(host_boost_df, save_path=None):
    plt.figure()
    plt.plot(host_boost_df["Year"], host_boost_df["HostBoostTotal"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Host boost (Total medals vs prev 2 Games mean)")
    plt.title("Host Advantage Proxy")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_medal_hist_2024(medals_2024_df, save_path=None):
    plt.figure()
    plt.hist(medals_2024_df["Total"], bins=30)
    plt.xlabel("Total medals (2024)")
    plt.ylabel("Number of countries")
    plt.title("Distribution of 2024 Total Medals")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()