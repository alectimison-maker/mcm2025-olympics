import numpy as np
import pandas as pd

def _build_design_matrix(df: pd.DataFrame, target: str):
    """
    Build design matrix X and target y for A1.
    Features:
      - log_events = log(max(TotalEvents,1))
      - is_host
      - log1p(lag1), log1p(lag2)
      - trend (scaled)
      - country fixed effects via one-hot NOC (drop_first=True)
    """
    import statsmodels.api as sm

    d = df.copy()

    # basic numeric safety
    d["TotalEvents"] = pd.to_numeric(d["TotalEvents"], errors="coerce").fillna(0).astype(float)
    d["log_events"] = np.log(np.maximum(d["TotalEvents"], 1.0))

    d["is_host"] = pd.to_numeric(d["is_host"], errors="coerce").fillna(0).astype(int)

    # trend scaled (per olympic cycle)
    YEAR0 = 1952
    d["trend"] = (d["Year"] - YEAR0) / 4.0

    if target.lower() == "gold":
        lag1, lag2 = "lag_gold_1", "lag_gold_2"
        cap_add = 20.0
    else:
        lag1, lag2 = "lag_total_1", "lag_total_2"
        cap_add = 50.0

    # lag safety + scale down
    for c in [lag1, lag2]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).clip(lower=0)
        d[c] = np.log1p(d[c])  # prevent explosive coefficients

    y = pd.to_numeric(d[target], errors="coerce").fillna(0).astype(float).values

    X_base = d[["log_events", "is_host", lag1, lag2, "trend"]].astype(float)
    X_country = pd.get_dummies(d["NOC"], prefix="NOC", drop_first=True).astype(float)
    X = pd.concat([X_base, X_country], axis=1)
    X = sm.add_constant(X, has_constant="add")

    # hard check: must be finite
    if not np.isfinite(X.values).all():
        raise ValueError("Design matrix contains NaN/inf.")
    if not np.isfinite(y).all():
        raise ValueError("Target contains NaN/inf.")

    # cap for numerical guard in bootstrap/sampling
    # robust cap from training distribution (99.5% quantile)
    y_q = float(np.quantile(y, 0.995)) if len(y) else 0.0
    y_max = float(np.max(y)) if len(y) else 0.0
    cap = max(30.0, 10.0 * y_q + cap_add, 3.0 * y_max + cap_add)

    return X, y, X.columns.tolist(), cap


def fit_nb_glm(df: pd.DataFrame, target: str = "Total", alpha: float = 1.0, l2: float = 1e-4):
    """
    Stable NB GLM fit with ridge regularization (fit_regularized) to avoid divergence.
    alpha: NB2 dispersion parameter (variance = mu + alpha*mu^2). Fixed for stability.
    l2: ridge penalty strength.
    """
    import statsmodels.api as sm

    X, y, cols, cap = _build_design_matrix(df, target)

    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha))

    # regularized fit is much more stable for high-dim one-hot + sparse counts
    try:
        res = model.fit_regularized(alpha=l2, L1_wt=0.0, maxiter=200)
    except Exception:
        # fallback to unregularized if regularized fails for any reason
        res = model.fit(maxiter=200, tol=1e-8)

    return {
        "res": res,
        "alpha": float(alpha),
        "l2": float(l2),
        "columns": cols,
        "target": target,
        "cap": float(cap),
    }


def predict_mu(model, df_pred: pd.DataFrame):
    """
    Predict expected mean mu for pred_df using stored columns alignment.
    Applies a safe cap to avoid explosive exp.
    """
    import statsmodels.api as sm

    d = df_pred.copy()
    d["TotalEvents"] = pd.to_numeric(d["TotalEvents"], errors="coerce").fillna(0).astype(float)
    d["log_events"] = np.log(np.maximum(d["TotalEvents"], 1.0))
    d["is_host"] = pd.to_numeric(d["is_host"], errors="coerce").fillna(0).astype(int)
    YEAR0 = 1952
    d["trend"] = (d["Year"] - YEAR0) / 4.0

    target = model["target"]
    if target.lower() == "gold":
        lag1, lag2 = "lag_gold_1", "lag_gold_2"
    else:
        lag1, lag2 = "lag_total_1", "lag_total_2"

    for c in [lag1, lag2]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).clip(lower=0)
        d[c] = np.log1p(d[c])

    X_base = d[["log_events", "is_host", lag1, lag2, "trend"]].astype(float)
    X_country = pd.get_dummies(d["NOC"], prefix="NOC", drop_first=True).astype(float)
    X = pd.concat([X_base, X_country], axis=1)
    X = sm.add_constant(X, has_constant="add")

    # align columns
    for c in model["columns"]:
        if c not in X.columns:
            X[c] = 0.0
    X = X[model["columns"]]

    mu = model["res"].predict(X)
    mu = np.asarray(mu, dtype=float)

    # final safety clamp
    cap = float(model.get("cap", 1e4))
    mu = np.clip(mu, 0.0, cap)
    return mu


def _nb_sample(rng: np.random.Generator, mu: np.ndarray, alpha: float):
    """
    Sample NB2 with variance = mu + alpha*mu^2.
    NB parameterization for numpy: n (size), p
      size = 1/alpha
      p = size / (size + mu)
    """
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, 0.0, None)

    # avoid alpha=0
    alpha = max(float(alpha), 1e-10)
    size = 1.0 / alpha
    p = size / (size + mu)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)

    return rng.negative_binomial(size, p)


def bootstrap_prediction_interval(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    target: str = "Total",
    B: int = 200,
    seed: int = 42,
    alpha: float = 1.0,
    l2: float = 1e-4,
    max_attempts_factor: int = 8,
):
    """
    Robust bootstrap:
      - resample train rows
      - fit stable regularized NB
      - predict mu on pred_df
      - drop explosive/non-finite fits
      - sample 1 NB draw per bootstrap replicate for predictive PI

    Returns pred_df with:
      pred_target, pi_low_target, pi_high_target, mu_low_target, mu_high_target
    """
    rng = np.random.default_rng(seed)

    # cap reference from full training (robust)
    _, y_full, _, cap_ref = _build_design_matrix(train_df, target)
    cap_ref = float(cap_ref)

    mus = []
    draws = []
    dropped = 0
    attempts = 0
    max_attempts = max(B * max_attempts_factor, B + 10)

    while (len(mus) < B) and (attempts < max_attempts):
        attempts += 1
        if (len(mus) == 0) or ((len(mus) + dropped) % 10 == 0):
            print(f"[{target}] bootstrap success={len(mus)}/{B}, dropped={dropped}, attempts={attempts}", flush=True)

        idx = rng.integers(0, len(train_df), len(train_df))
        boot = train_df.iloc[idx].copy()

        try:
            m = fit_nb_glm(boot, target=target, alpha=alpha, l2=l2)
            mu = predict_mu(m, pred_df)

            # extra guard: if mu hits cap_ref too often, treat as unstable
            if (not np.isfinite(mu).all()) or (mu.max() > cap_ref):
                dropped += 1
                continue

            mus.append(mu)

            # predictive draw (NB) with global alpha; mu already clipped
            y_draw = _nb_sample(rng, mu, alpha=alpha)
            draws.append(y_draw)

        except Exception:
            dropped += 1
            continue

    if len(mus) < max(10, int(0.5 * B)):
        raise RuntimeError(
            f"[{target}] Too few successful bootstrap fits ({len(mus)}/{B}). "
            f"Consider increasing l2 (e.g., 1e-3), removing NOC dummies for Gold, or reducing B."
        )

    mus = np.vstack(mus)            # (B_success, n_pred)
    draws = np.vstack(draws)        # (B_success, n_pred)

    mu_med = np.median(mus, axis=0)
    mu_lo = np.quantile(mus, 0.025, axis=0)
    mu_hi = np.quantile(mus, 0.975, axis=0)

    y_lo = np.quantile(draws, 0.025, axis=0)
    y_hi = np.quantile(draws, 0.975, axis=0)

    out = pred_df.copy()
    out[f"pred_{target}"] = mu_med
    out[f"mu_low_{target}"] = mu_lo
    out[f"mu_high_{target}"] = mu_hi
    out[f"pi_low_{target}"] = y_lo
    out[f"pi_high_{target}"] = y_hi

    print(f"[{target}] DONE. success={len(mus)}, dropped={dropped}, cap_ref={cap_ref:.1f}", flush=True)
    return out