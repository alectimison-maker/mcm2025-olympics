import numpy as np
import pandas as pd

def fit_nb_glm(country_year_df, target="Total"):
    """
    Baseline A1: Negative Binomial regression with country fixed effects.
    Features: log(TotalEvents), is_host, lag1, lag2, trend
    Country fixed effects via one-hot.
    Returns: fitted model object dict with coefficients.
    """
    # lazy import (so project can run even if statsmodels not installed at import time)
    import statsmodels.api as sm

    df = country_year_df.copy()
    df = df.sort_values(["Year","NOC"])
    df["log_events"] = np.log(df["TotalEvents"].replace(0, np.nan)).fillna(0.0)
    df["trend"] = (df["Year"] - df["Year"].min()) / 4.0

    # choose lags based on target
    if target.lower() == "gold":
        lag1, lag2 = "lag_gold_1", "lag_gold_2"
    else:
        lag1, lag2 = "lag_total_1", "lag_total_2"

    y = df[target].astype(float).values

    # design matrix
    X_base = df[["log_events","is_host", lag1, lag2, "trend"]].astype(float)
    X_country = pd.get_dummies(df["NOC"], prefix="NOC", drop_first=True).astype(float)
    X = pd.concat([X_base, X_country], axis=1)

    X = sm.add_constant(X, has_constant="add")

    # NB family in GLM requires alpha; estimate alpha via discrete NB first, then refit GLM
    try:
        from statsmodels.discrete.discrete_model import NegativeBinomial
        nb2 = NegativeBinomial(y, X).fit(disp=False)
        alpha = float(nb2.params.get("alpha", 1.0)) if hasattr(nb2, "params") else 1.0
    except Exception:
        alpha = 1.0

    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha))
    res = model.fit()

    return {"res": res, "alpha": alpha, "columns": X.columns.tolist(), "target": target}

def predict_mean(model, country_year_df):
    import statsmodels.api as sm
    df = country_year_df.copy()
    df["log_events"] = np.log(df["TotalEvents"].replace(0, np.nan)).fillna(0.0)
    df["trend"] = (df["Year"] - df["Year"].min()) / 4.0

    target = model["target"]
    if target.lower() == "gold":
        lag1, lag2 = "lag_gold_1", "lag_gold_2"
    else:
        lag1, lag2 = "lag_total_1", "lag_total_2"

    X_base = df[["log_events","is_host", lag1, lag2, "trend"]].astype(float)
    # rebuild country dummies with training columns
    X_country = pd.get_dummies(df["NOC"], prefix="NOC", drop_first=True).astype(float)
    X = pd.concat([X_base, X_country], axis=1)
    X = sm.add_constant(X, has_constant="add")

    # align columns
    for c in model["columns"]:
        if c not in X.columns:
            X[c] = 0.0
    X = X[model["columns"]]

    mu = model["res"].predict(X)
    return mu

def bootstrap_prediction_interval(model_func, train_df, pred_df, target="Total", B=200, seed=0):
    """
    Simple bootstrap: resample rows of train_df with replacement, refit, predict mean for pred_df.
    Then sample predictive counts using NB approximation.
    """
    rng = np.random.default_rng(seed)
    mus = []
    for b in range(B):
        if (b == 0) or ((b + 1) % 10 == 0):
            print(f"[{target}] bootstrap {b+1}/{B} (success={len(mus)})", flush=True)

        idx = rng.integers(0, len(train_df), len(train_df))
        boot = train_df.iloc[idx].copy()
        try:
            m = model_func(boot, target=target)
            mu = predict_mean(m, pred_df)
            mus.append(mu)
        except Exception as e:
            # 可选：想看失败原因就打开下一行
            # print(f"[{target}] bootstrap {b+1} failed: {e}", flush=True)
            continue
    if len(mus) == 0:
        raise RuntimeError("Bootstrap failed: no successful fits.")
    mus = np.vstack(mus)  # shape (B, n)

    mu_med = np.median(mus, axis=0)
    lo_mu = np.quantile(mus, 0.025, axis=0)
    hi_mu = np.quantile(mus, 0.975, axis=0)

    # predictive interval (rough): use Poisson sampling around mu to get count PI
    # (NB sampling can be added later once we lock parameterization)
    n = mus.shape[1]
    draws = []
    for b in range(mus.shape[0]):
        lam = np.clip(mus[b], 0, None)
        draws.append(rng.poisson(lam=lam))
    draws = np.vstack(draws)
    lo_y = np.quantile(draws, 0.025, axis=0)
    hi_y = np.quantile(draws, 0.975, axis=0)

    out = pred_df.copy()
    out[f"pred_{target}"] = mu_med
    out[f"pi_low_{target}"] = lo_y
    out[f"pi_high_{target}"] = hi_y
    out[f"mu_low_{target}"] = lo_mu
    out[f"mu_high_{target}"] = hi_mu
    return out
