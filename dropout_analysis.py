"""
Explaining state-wise secondary dropout from UDISE+ indicators
Regression pipeline: merge -> correlation/VIF -> OLS -> ridge/lasso/poly2 -> outliers -> LOOCV

HOW TO USE
1. Place merged_dataset.csv in the same directory as this script.
2. Run:  python dropout_analysis.py
Everything is saved in OUT_DIR (tables as CSV, figures as PNG).
"""
import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
SEED = 42
DATA_DIR = "."
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- CONFIG
MERGED_FILE = "merged_dataset.csv"

# Small UTs to flag / drop in the sensitivity run
SMALL_UTS = ["Lakshadweep", "Ladakh", "Andaman and Nicobar Islands", "Chandigarh",
             "Puducherry", "Dadra and Nagar Haveli and Daman and Diu"]


# ---------------------------------------------------------------- HELPERS
def norm(s):
    """Normalise state names so files merge."""
    s = str(s).strip().lower().replace("&", "and")
    s = re.sub(r"\s+", " ", s)
    s = s.replace("jammu and kashmir", "jammu and kashmir")
    return s


# ---------------------------------------------------------------- 1. LOAD MERGED DATA
df = pd.read_csv(os.path.join(DATA_DIR, MERGED_FILE))
df.columns = [c.strip() for c in df.columns]

FEATURES = ["PTR_sec", "GER_sec", "pct_electricity", "pct_computer", "pct_internet"]
df = df.dropna(subset=FEATURES).reset_index(drop=True)
n, p = df.shape[0], len(FEATURES)
print(f"\nFinal analysis table: n={n}, predictors={p}")
print(f"Features: {FEATURES}")

# ---------------------------------------------------------------- 2. DESCRIPTIVES + CORRELATION
df[["dropout_sec"] + FEATURES].describe().T.round(2).to_csv(f"{OUT_DIR}/descriptives.csv")
corr = df[["dropout_sec"] + FEATURES].corr()
corr.round(3).to_csv(f"{OUT_DIR}/correlation_matrix.csv")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im); ax.set_title("Pearson correlation matrix")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_correlation.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 3. VIF
Xc = sm.add_constant(df[FEATURES])
vif = pd.DataFrame({"feature": FEATURES,
                    "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(p)]})
vif.round(2).to_csv(f"{OUT_DIR}/vif.csv", index=False)
print("\nVIF:\n", vif.round(2))

# ---------------------------------------------------------------- 4. OLS (+ outliers)
def run_ols(data, label):
    X = sm.add_constant(data[FEATURES]); yv = data["dropout_sec"]
    m = sm.OLS(yv, X).fit()
    tab = pd.DataFrame({"coef": m.params, "std_err": m.bse, "t": m.tvalues, "p_value": m.pvalues}).round(4)
    tab.to_csv(f"{OUT_DIR}/ols_{label}.csv")
    return m, tab

ols_all, tab_all = run_ols(df, "all")
print("\nOLS (all states):\n", tab_all)
print(f"R2={ols_all.rsquared:.3f}  adjR2={ols_all.rsquared_adj:.3f}  F p={ols_all.f_pvalue:.4f}")

infl = ols_all.get_influence()
cooks = infl.cooks_distance[0]
df["cooks_d"] = cooks
df["std_resid"] = infl.resid_studentized_external
thr = 4 / n
flag = df.loc[(df.cooks_d > thr) | (df.std_resid.abs() > 3), ["state", "dropout_sec", "cooks_d", "std_resid"]]
flag.round(3).to_csv(f"{OUT_DIR}/outliers_flagged.csv", index=False)
print(f"\nInfluential states (Cook's D > 4/n = {thr:.3f} or |t-resid| > 3):\n", flag.round(3))

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(n), cooks, color="#3b6ea5"); ax.axhline(thr, color="red", ls="--", label="4/n")
ax.set_xticks(range(n)); ax.set_xticklabels(df["state"], rotation=90, fontsize=7)
ax.set_ylabel("Cook's distance"); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_cooks.png", dpi=200); plt.close(fig)

mask_small = df["state"].map(norm).isin([norm(s) for s in SMALL_UTS])
df_nosmall = df[~mask_small].reset_index(drop=True)
ols_ns, tab_ns = run_ols(df_nosmall, "no_small_UTs")
print(f"\nOLS without small UTs (n={len(df_nosmall)}): R2={ols_ns.rsquared:.3f}, adjR2={ols_ns.rsquared_adj:.3f}")

# ---------------------------------------------------------------- 5. LOOCV model comparison
def loocv_table(data, label):
    X = data[FEATURES].values; yv = data["dropout_sec"].values
    top = data[FEATURES].corrwith(data["dropout_sec"]).abs().sort_values(ascending=False).index[:min(3, p)]
    models = {
        "Mean baseline": None,
        "OLS": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Lasso": make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-3, 1, 30), cv=5, random_state=SEED, max_iter=50000)),
        "Poly-2 + Ridge (top 3)": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                                 StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))),
    }
    rows = []
    for name, mdl in models.items():
        if mdl is None:
            pred = np.array([np.delete(yv, i).mean() for i in range(len(yv))])
        else:
            Xm = data[list(top)].values if name.startswith("Poly") else X
            pred = cross_val_predict(mdl, Xm, yv, cv=LeaveOneOut())
        rmse = np.sqrt(mean_squared_error(yv, pred))
        r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        mae = np.abs(yv - pred).mean()
        rows.append([name, round(mae, 2), round(rmse, 2), round(r2, 3)])
    t = pd.DataFrame(rows, columns=["Model", "MAE", "LOOCV RMSE", "LOOCV R2"])
    t.to_csv(f"{OUT_DIR}/loocv_{label}.csv", index=False)
    return t

res_all = loocv_table(df, "all")
res_ns = loocv_table(df_nosmall, "no_small_UTs")
print("\nLOOCV, all states:\n", res_all)
print("\nLOOCV, without small UTs:\n", res_ns)

# ---------------------------------------------------------------- 6. Standardised coefficients
sc = StandardScaler().fit(df[FEATURES]); Xs = sc.transform(df[FEATURES]); yv = df["dropout_sec"].values
ridge = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(Xs, yv)
lasso = LassoCV(alphas=np.logspace(-3, 1, 30), cv=5, random_state=SEED, max_iter=50000).fit(Xs, yv)
coefs = pd.DataFrame({"feature": FEATURES, "ridge_std_coef": ridge.coef_, "lasso_std_coef": lasso.coef_}).round(3)
coefs["ridge_alpha"] = round(ridge.alpha_, 3); coefs["lasso_alpha"] = round(lasso.alpha_, 3)
coefs.to_csv(f"{OUT_DIR}/standardised_coefficients.csv", index=False)
print("\nStandardised coefficients:\n", coefs)

fig, ax = plt.subplots(figsize=(6, 4))
w = 0.38; xs = np.arange(p)
ax.bar(xs - w / 2, coefs.ridge_std_coef, w, label="Ridge")
ax.bar(xs + w / 2, coefs.lasso_std_coef, w, label="Lasso")
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(xs); ax.set_xticklabels(FEATURES, rotation=30, ha="right")
ax.set_ylabel("Std. coefficient (pp per 1 SD)"); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_coefficients.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 7. Observed vs predicted (LOOCV ridge)
pred = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
                         df[FEATURES].values, yv, cv=LeaveOneOut())
fig, ax = plt.subplots(figsize=(4.8, 4.5))
ax.scatter(yv, pred, s=22, color="#3b6ea5")
lim = [0, max(yv.max(), pred.max()) + 2]; ax.plot(lim, lim, color="orange")
ax.set_xlabel("Observed secondary dropout (%)"); ax.set_ylabel("LOOCV predicted (%)")
ax.set_title("Ridge, leave-one-out predictions"); fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_obs_vs_pred.png", dpi=200); plt.close(fig)

print(f"\nDone. All outputs in {OUT_DIR}")
