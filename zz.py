# -*- coding: utf-8 -*-
# ============================================================
# Зам тээврийн осол — Auto ML & Hotspot Dashboard (Streamlit)
# Хувилбар: 2025-08-17r3.1 (refined++) — leakage-гүй, TSCV + Poisson,
#            САР ба ӨДӨР аль алинд нь таамаглал (динамик улирлын экзоген)
#            + бодит нэгжийн метрик, integer прогноз, өдөр гөлгөршүүлэх
# ============================================================

from __future__ import annotations
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
from datetime import timedelta

# Sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit  # rolling backtest

# 3rd-party regressors (optional)
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:
    CatBoostRegressor = None

from scipy.stats import chi2_contingency

# (map/cluster-related imports are kept for your other tabs)
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# -------------------------- UI setup --------------------------
st.set_page_config(page_title="Осол — Auto ML & Hotspot (auto-binary)", layout="wide")
st.title("С.Цолмон, А.Тамир нарын хар цэгийн судалгаа 2025-08-18")

# -------------------------- Туслах функцууд --------------------------
def _canon(s: str) -> str:
    return "".join(str(s).lower().split()) if isinstance(s, str) else str(s)

def resolve_col(df: pd.DataFrame, candidates) -> str | None:
    """Нэршлийг robust байдлаар олно (whitespace/том-жижиг үл тооно)."""
    lut = {_canon(c): c for c in df.columns}
    for name in candidates:
        key = _canon(name)
        if key in lut:
            return lut[key]
    return None

def is_binary_series(s: pd.Series) -> bool:
    vals = pd.to_numeric(s.dropna(), errors="coerce").dropna().unique()
    if len(vals) == 0:
        return False
    return set(np.unique(vals)).issubset({0, 1})

def plot_correlation_matrix(df, title, columns):
    n_unique = df[columns].nunique()
    if all((n_unique == 2) | (n_unique == 1)):
        st.warning("Сонгосон баганууд one-hot (0/1) тул корреляци туйлшрах мэт харагдаж болно.")
    df_encoded = df[columns].copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
    corr_matrix = df_encoded.corr()
    corr_matrix = corr_matrix.iloc[::-1]
    fig, ax = plt.subplots(figsize=(max(8, 1.5*len(columns)), max(6, 1.2*len(columns))))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax, fmt=".3f")
    ax.set_title(title)
    plt.tight_layout()
    return fig

# -------------------------- Өгөгдөл ачаалалт --------------------------
uploaded_file = st.sidebar.file_uploader("Excel файл оруулах (.xlsx)", type=["xlsx"])

@st.cache_data(show_spinner=True)
def load_data(file=None, default_path: str = "кодлогдсон.xlsx"):
    """Excel дата уншина. Энд UI/stop хийхгүй, алдааг raise хийнэ."""
    if file is not None:
        df = pd.read_excel(file)
    else:
        local = Path(default_path)
        if not local.exists():
            raise FileNotFoundError(f"Excel файл олдсонгүй: {default_path}")
        df = pd.read_excel(local)

    # Нэршил цэвэрлэгээ
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Огноо багана хайх
    recv_col = resolve_col(df, ["Хүлээн авсан", "Хүлээн авсан ", "Огноо", "Зөрчил огноо",
                                "Осол огноо", "Ослын огноо", "Date"])
    if recv_col is None:
        raise ValueError("Огнооны багана олдсонгүй. Жишээ: 'Хүлээн авсан'.")

    # 'Зөрчил огноо' үүсгэх
    df["Зөрчил огноо"] = pd.to_datetime(df[recv_col], errors="coerce")
    if df["Зөрчил огноо"].isna().all():
        raise ValueError("Огноог parse хийж чадсангүй. Огнооны формат шалгана уу.")

    df["Year"]  = df["Зөрчил огноо"].dt.year
    df["Month"] = df["Зөрчил огноо"].dt.month
    df["Day"]   = df["Зөрчил огноо"].dt.day_name()

    # Он жилүүдийн one-hot
    years = sorted(df["Year"].dropna().unique().tolist())
    for y in years:
        df[f"Зөрчил огноо жил {int(y)}"] = (df["Year"] == int(y)).astype(int)
    if len(years) > 0:
        df["Зөрчил огноо жил (min-max)"] = df["Year"].between(min(years), max(years)).astype(int)

    # Координат нэршил
    lat_col = resolve_col(df, ["Өргөрөг", "lat", "latitude"])
    lon_col = resolve_col(df, ["Уртраг", "lon", "longitude"])

    # Автоматаар binary багануудыг илрүүлэх
    exclude = {"Зөрчил огноо", "Year", "Month", "Day", "д/д", "Хороо-Сум", "Аймаг-Дүүрэг"}
    if lat_col: exclude.add(lat_col)
    if lon_col: exclude.add(lon_col)
    binary_cols = [c for c in df.columns if c not in exclude and is_binary_series(df[c])]

    # Нэмэлт тоон candidate-үүд
    numeric_candidates = []
    if "Авто зам - Зорчих хэсгийн өргөн" in df.columns:
        numeric_candidates.append("Авто зам - Зорчих хэсгийн өргөн")

    # Дүүрэг/Аймаг fallback
    if "Дүүрэг" not in df.columns:
        df["Дүүрэг"] = 0
    if "Аймаг" not in df.columns:
        df["Аймаг"] = 0

    meta = {
        "lat_col": lat_col,
        "lon_col": lon_col,
        "binary_cols": binary_cols,
        "numeric_candidates": numeric_candidates,
        "years": years,
    }
    return df, meta

# === load ===
try:
    df, meta = load_data(uploaded_file)
except Exception as e:
    st.error(f"Өгөгдөл ачаалахад алдаа: {e}")
    st.stop()

lat_col, lon_col = meta["lat_col"], meta["lon_col"]
binary_cols = meta["binary_cols"]
num_additional = meta["numeric_candidates"]
years = meta["years"]

# -------------------------- Target тохиргоо --------------------------
st.sidebar.markdown("### 🎯 Зорилтот тодорхойлолт (Осол)")
target_mode = st.sidebar.radio(
    "Осол гэж тооцох ангиллыг сонгоно уу:",
    ("Хоёуланг 1 гэж тооц", "Зөвхөн Гэмт хэрэг", "Зөвхөн Зөрчлийн хэрэг"),
)

torol_col = resolve_col(df, ["Төрөл"])
if torol_col is None:
    st.error("`Төрөл` багана олдсонгүй. Target үүсгэх боломжгүй.")
    st.stop()

if target_mode == "Хоёуланг 1 гэж тооц":
    df["Осол"] = df[torol_col].isin(["Гэмт хэрэг", "Зөрчлийн хэрэг"]).astype(int)
elif target_mode == "Зөвхөн Гэмт хэрэг":
    df["Осол"] = (df[torol_col] == "Гэмт хэрэг").astype(int)
else:
    df["Осол"] = (df[torol_col] == "Зөрчлийн хэрэг").astype(int)

# -------------------------- Прогнозын тохиргоо --------------------------
st.header("5. Ирээдүйн ослын таамаглал (Олон ML/DL загвар)")
agg_mode = st.sidebar.selectbox("Прогнозын агрегат", ["Сар", "Өдөр"], index=0)

st.caption(
    "Binary (0/1) шинжүүд автоматаар илрүүлэгдэнэ. "
    "Прогнозын агрегатыг **Сар / Өдөр**-өөр сонгож болно. "
    "Өдөр горимд 7 хоног/365 өдрийн Фурье улирлыг динамикаар шинэчилнэ."
)

# Feature pool (leakage-гүй)
def nonleaky(col: str) -> bool:
    s = str(col)
    if s == "Осол":
        return False
    if s.startswith("Зөрчил огноо жил "):
        return False
    if "Төрөл" in s:
        return False
    if s in {"Year", "Month", "Day"}:
        return False
    return True

feature_pool = [c for c in (binary_cols + num_additional) if nonleaky(c)]
if len(feature_pool) == 0:
    st.error("Leakage-гүй шинж олдсонгүй. Metadata/one-hot үүсгэх дүрмээ шалгана уу.")
    st.stop()

# -------------------------- SERIES BUILD (Сар/Өдөр) --------------------------

def build_monthly(df_in: pd.DataFrame):
    # Target
    monthly_target = (
        df_in[df_in["Осол"] == 1]
        .groupby(["Year", "Month"])
        .agg(osol_count=("Осол", "sum"))
        .reset_index()
    )
    monthly_target["date"] = pd.to_datetime(monthly_target[["Year", "Month"]].assign(DAY=1))

    # Exog (sum per month) and shift by 1 month to avoid leakage
    monthly_features = (
        df_in.groupby(["Year", "Month"])[feature_pool]
            .sum()
            .reset_index()
            .sort_values(["Year", "Month"])
    )
    for c in feature_pool:
        monthly_features[c] = monthly_features[c].shift(1)

    grouped = (
        pd.merge(monthly_target, monthly_features, on=["Year", "Month"], how="left")
        .sort_values(["Year", "Month"]).reset_index(drop=True)
    )

    # Fourier (12)
    grouped["m"] = grouped["Month"].astype(int)
    K = [1, 2, 3]
    for k in K:
        grouped[f"m_sin_{k}"] = np.sin(2*np.pi*k*grouped["m"]/12)
        grouped[f"m_cos_{k}"] = np.cos(2*np.pi*k*grouped["m"]/12)
    fourier_cols = [f"m_sin_{k}" for k in K] + [f"m_cos_{k}" for k in K]

    # Lags (months)
    n_lag = st.sidebar.slider("Сарын лаг цонх (n_lag)", min_value=6, max_value=18, value=12, step=1, key="lag_m")
    lag_cols = [f"osol_lag_{i}" for i in range(1, n_lag + 1)]
    for i in range(1, n_lag + 1):
        grouped[f"osol_lag_{i}"] = grouped["osol_count"].shift(i)

    exog_cols = feature_pool + fourier_cols
    return grouped, lag_cols, fourier_cols, exog_cols, n_lag, "MS"  # month start freq


def build_daily(df_in: pd.DataFrame):
    # Build continuous daily index
    df_in = df_in.copy()
    df_in["date"] = df_in["Зөрчил огноо"].dt.floor("D")
    start, end = df_in["date"].min().normalize(), df_in["date"].max().normalize()

    # 👉 Index-ээ НЭРТЭЙ үүсгэнэ, ингэснээр reset_index() үед 'date' багана тогтвортой гарна
    all_days = pd.date_range(start, end, freq="D", name="date")

    # Target = sum of accidents per day
    daily_target = (
        df_in.groupby("date")["Осол"].sum()
             .reindex(all_days, fill_value=0)
             .rename("osol_count")
             .rename_axis("date")      # хамгаалалт
             .reset_index()
    )

    # Exog = sum of binary features per day (shift by 1 day)
    if len(feature_pool) > 0:
        daily_features = (
            df_in.groupby("date")[feature_pool].sum()
                 .reindex(all_days, fill_value=0)
                 .rename_axis("date")  # хамгаалалт
                 .reset_index()
        )
        for c in feature_pool:
            daily_features[c] = daily_features[c].shift(1)
    else:
        # feature_pool хоосон бол зөвхөн 'date' баганатай dataframe үүсгэнэ
        daily_features = pd.DataFrame({"date": all_days}).reset_index(drop=True)

    grouped = (
        pd.merge(daily_target, daily_features, on="date", how="left")
          .sort_values("date")
          .reset_index(drop=True)
    )

    # Fourier seasonal: weekly (7) + yearly (365)
    Kw, Ky = [1, 2, 3], [1, 2, 3]
    dow = grouped["date"].dt.dayofweek  # 0..6
    doy = grouped["date"].dt.dayofyear  # 1..365/366
    for k in Kw:
        grouped[f"w_sin_{k}"] = np.sin(2*np.pi*k*dow/7)
        grouped[f"w_cos_{k}"] = np.cos(2*np.pi*k*dow/7)
    for k in Ky:
        grouped[f"y_sin_{k}"] = np.sin(2*np.pi*k*doy/365)
        grouped[f"y_cos_{k}"] = np.cos(2*np.pi*k*doy/365)
    fourier_cols = (
        [f"w_sin_{k}" for k in Kw] + [f"w_cos_{k}" for k in Kw] +
        [f"y_sin_{k}" for k in Ky] + [f"y_cos_{k}" for k in Ky]
    )

    # Lags (days)
    n_lag = st.sidebar.slider("Өдрийн лаг цонх (n_lag)", min_value=7, max_value=120, value=30, step=1, key="lag_d")
    lag_cols = [f"osol_lag_{i}" for i in range(1, n_lag + 1)]
    for i in range(1, n_lag + 1):
        grouped[f"osol_lag_{i}"] = grouped["osol_count"].shift(i)

    exog_cols = feature_pool + fourier_cols

    # downstream нийцтэй байдлаар Year/Month гаргаж өгье
    grouped["Year"] = grouped["date"].dt.year
    grouped["Month"] = grouped["date"].dt.month

    return grouped, lag_cols, fourier_cols, exog_cols, n_lag, "D"


# Build chosen series
if agg_mode == "Сар":
    grouped, lag_cols, seasonal_cols, exog_cols, n_lag, freq_code = build_monthly(df)
else:
    grouped, lag_cols, seasonal_cols, exog_cols, n_lag, freq_code = build_daily(df)

# Drop NA after lags/shift
grouped = grouped.dropna().reset_index(drop=True)
if grouped.empty or len(grouped) < max(10, n_lag + 5):
    st.warning(f"Сургалт хийхэд хангалттай өгөгдөл алга (lag={n_lag}, mode={agg_mode}). Он/сар/өдрийн хүрээг шалгана уу.")
    st.stop()

# ---- Train/Test split хувь ----
split_ratio = st.sidebar.slider("Train ratio", 0.5, 0.9, 0.8, 0.05)

# ---- Feature selection (TRAIN-only) ----
X_all_fs = grouped[lag_cols + exog_cols].fillna(0.0).values
y_all = grouped["osol_count"].values.reshape(-1, 1)
train_size = int(len(X_all_fs) * split_ratio)
X_train_fs, y_train_fs = X_all_fs[:train_size], y_all[:train_size].ravel()

try:
    rf_fs = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_fs.fit(X_train_fs, y_train_fs)
    imp_series = pd.Series(rf_fs.feature_importances_, index=lag_cols + exog_cols)
    k = min(14, len(exog_cols))
    exog_top = imp_series.loc[exog_cols].sort_values(ascending=False).head(k).index.tolist()
    st.caption("Train дээр тодорсон top exogenous features (leakage-гүй):")
    st.write(exog_top)
except Exception as e:
    st.warning(f"Feature selection train дээр ажиллахад алдаа: {e}")
    exog_top = exog_cols[:min(14, len(exog_cols))]

# ---- Эцсийн шинжүүд ----
feature_cols = lag_cols + exog_top

# ---- X/y, split, scale ----
X = grouped[feature_cols].fillna(0.0).values
y = grouped["osol_count"].values.reshape(-1, 1)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train)
X_test  = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train).ravel()
y_test  = scaler_y.transform(y_test).ravel()

# ---- Leakage safety ----
identicals = [c for c in feature_cols if np.allclose(grouped[c].values, grouped["osol_count"].values, equal_nan=False)]
if identicals:
    st.error(f"IDENTICAL leakage үлдлээ: {identicals}")
    st.stop()
corrs = grouped[feature_cols].corrwith(grouped["osol_count"]).sort_values(ascending=False)
st.write("Target-тэй корреляци (дээд 10):", corrs.head(10))

# -------------------------- Models --------------------------
estimators = [
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ("ridge", Ridge()),
    ("dt", DecisionTreeRegressor(random_state=42)),
]

MODEL_LIST = [
    ("LinearRegression", LinearRegression()),
    ("Ridge", Ridge()),
    ("Lasso", Lasso()),
    ("ElasticNet", ElasticNet()),
    ("DecisionTree", DecisionTreeRegressor(random_state=42)),
    ("RandomForest", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ("ExtraTrees", ExtraTreesRegressor(random_state=42, n_jobs=-1)),
    ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
    ("HistGB-Poisson", HistGradientBoostingRegressor(loss="poisson", learning_rate=0.06, max_depth=None, random_state=42)),
    ("AdaBoost", AdaBoostRegressor(random_state=42)),
    ("KNeighbors", KNeighborsRegressor()),
    ("SVR", SVR()),
    ("MLPRegressor", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42)),
    ("Stacking", StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=5)),
]

# Poisson objective
if XGBRegressor is not None:
    MODEL_LIST.append(("XGBRegressor", XGBRegressor(tree_method="hist", predictor="cpu_predictor", random_state=42, n_estimators=400)))
    MODEL_LIST.append(("XGB-Poisson", XGBRegressor(
        objective="count:poisson", tree_method="hist", predictor="cpu_predictor",
        n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        random_state=42)))
if CatBoostRegressor is not None:
    MODEL_LIST.append(("CatBoostRegressor", CatBoostRegressor(task_type="CPU", random_state=42, verbose=0)))
if LGBMRegressor is not None:
    MODEL_LIST.append(("LGBMRegressor", LGBMRegressor(device="cpu", random_state=42)))
    MODEL_LIST.append(("LGBM-Poisson", LGBMRegressor(
        objective="poisson", metric="rmse", n_estimators=1200, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, device="cpu", random_state=42)))

# Voting/Stacking ensemble
voting_estimators = []
if XGBRegressor is not None:
    voting_estimators.append(("xgb", XGBRegressor(tree_method="hist", predictor="cpu_predictor", random_state=42)))
if LGBMRegressor is not None:
    voting_estimators.append(("lgbm", LGBMRegressor(device="cpu", random_state=42)))
if CatBoostRegressor is not None:
    voting_estimators.append(("cat", CatBoostRegressor(task_type="CPU", random_state=42, verbose=0)))
voting_estimators += [
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ("gb", GradientBoostingRegressor(random_state=42)),
]
if len(voting_estimators) > 1:
    MODEL_LIST.append(("VotingRegressor", VotingRegressor(estimators=voting_estimators)))

stacking_estimators = [("rf", RandomForestRegressor(random_state=42, n_jobs=-1))]
if XGBRegressor is not None:
    stacking_estimators.append(("xgb", XGBRegressor(tree_method="hist", predictor="cpu_predictor", random_state=42)))
if LGBMRegressor is not None:
    stacking_estimators.append(("lgbm", LGBMRegressor(device="cpu", random_state=42)))
if CatBoostRegressor is not None:
    stacking_estimators.append(("cat", CatBoostRegressor(task_type="CPU", verbose=0, random_state=42)))
MODEL_LIST.append(("StackingEnsemble", StackingRegressor(estimators=stacking_estimators, final_estimator=LinearRegression(), cv=5)))

# -------------------------- Rolling Backtest (optional) --------------------------

def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom))


def mase(y_insample, y_true, y_pred, m=12, eps=1e-8):
    y_insample = np.asarray(y_insample, float).ravel()
    if len(y_insample) <= m:
        return float(np.nan)
    denom = np.mean(np.abs(y_insample[m:] - y_insample[:-m]))
    return float(np.mean(np.abs(y_pred - y_true) / (denom + eps)))


def rolling_backtest(X_raw, y_raw, model_list, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for name, base_model in model_list:
        maes, rmses, r2s, smapes, mases = [], [], [], [], []
        for tr, te in tscv.split(X_raw):
            X_tr, X_te = X_raw[tr], X_raw[te]
            y_tr, y_te = y_raw[tr], y_raw[te]
            sx, sy = MinMaxScaler(), MinMaxScaler()
            X_tr_s, X_te_s = sx.fit_transform(X_tr), sx.transform(X_te)
            y_tr_s = sy.fit_transform(y_tr.reshape(-1,1)).ravel()
            model = base_model
            model.fit(X_tr_s, y_tr_s)
            yhat_s = np.asarray(model.predict(X_te_s)).ravel()
            yhat   = sy.inverse_transform(yhat_s.reshape(-1,1)).ravel()
            maes.append(mean_absolute_error(y_te, yhat))
            rmses.append(np.sqrt(mean_squared_error(y_te, yhat)))
            r2s.append(r2_score(y_te, yhat))
            smapes.append(smape(y_te, yhat))
            # m: 12 for monthly, 7 for daily seasonal naive
            m_period = 12 if agg_mode == "Сар" else 7
            mases.append(mase(y_raw[:te[0]].ravel(), y_te.ravel(), yhat.ravel(), m=m_period))
        rows.append({
            "Model": name,
            "MAE (cv)": np.mean(maes),
            "RMSE (cv)": np.mean(rmses),
            "R2 (cv)": np.mean(r2s),
            "sMAPE (cv)": np.mean(smapes),
            "MASE (cv)": np.mean(mases),
        })
    return pd.DataFrame(rows).sort_values("RMSE (cv)", na_position="last")

if st.sidebar.checkbox("⏱ Rolling backtest (5-fold TSCV)", value=False):
    cv_df = rolling_backtest(X, y.reshape(-1), MODEL_LIST, n_splits=5)
    st.subheader("Rolling backtest — дундаж метрик")
    st.dataframe(cv_df, use_container_width=True)

# -------------------------- Train all --------------------------
progress_bar = st.progress(0, text="ML моделийг сургаж байна...")
results, y_preds, fitted_models = [], {}, {}
for i, (name, model) in enumerate(MODEL_LIST):
    try:
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # scaled prediction for R2, and keep for later inverse-transform
        y_pred_s = np.asarray(model.predict(X_test)).reshape(-1)
        y_preds[name] = y_pred_s
        r2_s = r2_score(y_test, y_pred_s)

        # inverse-scale metrics in REAL units
        y_true = scaler_y.inverse_transform(y_test.reshape(-1,1)).ravel()
        y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).ravel()
        y_pred = np.clip(y_pred, 0, None)

        results.append({
            "Model": name,
            "MAE (real)": mean_absolute_error(y_true, y_pred),
            "RMSE (real)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2 (scaled)": float(r2_s)
        })
    except Exception as e:
        results.append({"Model": name, "MAE (real)": np.nan, "RMSE (real)": np.nan, "R2 (scaled)": np.nan, "Error": str(e)})
    progress = min(int((i + 1) / len(MODEL_LIST) * 100), 100)
    progress_bar.progress(progress, text=f"{name} дууслаа")
progress_bar.empty()
st.success("Бүх ML модел сургагдлаа!")

results_df = pd.DataFrame(results).sort_values("RMSE (real)", na_position="last")
st.dataframe(
    results_df.style.format({"MAE (real)": "{:.3f}", "RMSE (real)": "{:.3f}", "R2 (scaled)": "{:.4f}"}),
    use_container_width=True,
)

# Excel татах (метрик)
with pd.ExcelWriter("model_metrics.xlsx", engine="xlsxwriter") as writer:
    results_df.to_excel(writer, index=False)
with open("model_metrics.xlsx", "rb") as f:
    st.download_button(
        "Моделийн метрик Excel татах",
        data=f,
        file_name="model_metrics.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# -------------------------- Прогноз helper --------------------------
lag_count = len(lag_cols)

def seasonal_values_for_date(dt: pd.Timestamp, mode: str):
    """Дараагийн алхмын улирлын Фурье шинжүүдийг тухайн өдрийн/сарын огнооноос тооцно."""
    vals = {}
    if mode == "Сар":
        m = int(dt.month)
        K = [1, 2, 3]
        for k in K:
            vals[f"m_sin_{k}"] = np.sin(2*np.pi*k*m/12)
            vals[f"m_cos_{k}"] = np.cos(2*np.pi*k*m/12)
    else:
        dow = int(dt.dayofweek)
        doy = int(dt.dayofyear)
        Kw, Ky = [1, 2, 3], [1, 2, 3]
        for k in Kw:
            vals[f"w_sin_{k}"] = np.sin(2*np.pi*k*dow/7)
            vals[f"w_cos_{k}"] = np.cos(2*np.pi*k*dow/7)
        for k in Ky:
            vals[f"y_sin_{k}"] = np.sin(2*np.pi*k*doy/365)
            vals[f"y_cos_{k}"] = np.cos(2*np.pi*k*doy/365)
    return vals

def step_next_date(d: pd.Timestamp, mode: str):
    if mode == "Сар":
        # next month start
        return (d + pd.offsets.MonthBegin(1)).normalize()
    else:
        return (d + pd.Timedelta(days=1)).normalize()

def forecast_next(model, last_raw_row, steps, last_date, mode, seasonal_cols, feature_cols):
    """
    last_raw_row: grouped[feature_cols]-ийн СҮҮЛИЙН мөр (анскейлд)
    steps: хэдэн алхам
    last_date: grouped['date']-ийн сүүлийн огноо
    mode: "Сар" | "Өдөр"
    seasonal_cols: тухайн горимын улирлын багануудын нэрс
    feature_cols: lag + exog_top дарааллаар
    """
    preds = []
    # lag болон "бусад экзоген"-ийг ялгана
    lag_vals = last_raw_row[:lag_count].astype(float).copy()
    other_exog_names = [c for c in feature_cols[lag_count:] if c not in seasonal_cols]
    # бусад экзогенүүд ирээдүйд мэдэгдэхгүй тул status quo (сүүлийн мөрийн утга)-гаар явна
    other_exog_vals = {n: float(last_raw_row[lag_count + feature_cols[lag_count:].index(n)]) for n in other_exog_names}
    cur_date = pd.to_datetime(last_date)

    for _ in range(steps):
        cur_date = step_next_date(cur_date, mode)
        seas = seasonal_values_for_date(cur_date, mode)
        # экзогенүүдийг feature_cols-ийн дарааллаар угсарч байрлуулна
        exog_vector = []
        for name in feature_cols[lag_count:]:
            if name in seasonal_cols:
                exog_vector.append(seas.get(name, 0.0))
            else:
                exog_vector.append(other_exog_vals.get(name, 0.0))
        seq_raw = np.concatenate([lag_vals, np.array(exog_vector, float)]).reshape(1, -1)
        seq_scaled = scaler_X.transform(seq_raw)
        p_scaled = float(np.asarray(model.predict(seq_scaled)).ravel()[0])
        p = float(scaler_y.inverse_transform(np.array([[p_scaled]])).ravel()[0])
        p = max(p, 0.0)
        preds.append(p)
        # lag-уудаа шинэчилнэ
        lag_vals = np.roll(lag_vals, 1)
        lag_vals[0] = p
    return np.array(preds)

# Horizon options
if agg_mode == "Сар":
    h_map = {"7 хоног (≈1 сар)": 1, "14 хоног (≈1 сар)": 1, "30 хоног (≈1 сар)": 1, "90 хоног (≈3 сар)": 3,
             "180 хоног (≈6 сар)": 6, "365 хоног (≈12 сар)": 12}
else:
    h_map = {"7 хоног": 7, "14 хоног": 14, "30 хоног": 30, "90 хоног": 90, "180 хоног": 180, "365 хоног": 365}

# Forecasts by model
model_forecasts = {}
last_row_raw = grouped[feature_cols].iloc[-1].values  # анскейлд
last_known_date = grouped["date"].iloc[-1]

for name, model in fitted_models.items():
    if name not in y_preds:
        continue
    preds_dict = {}
    for k, steps in h_map.items():
        preds_dict[k] = forecast_next(
            model, last_row_raw, steps=steps, last_date=last_known_date,
            mode=agg_mode, seasonal_cols=seasonal_cols, feature_cols=feature_cols
        )
    model_forecasts[name] = preds_dict

# Test дээрх бодит/таамаг (анскейлд)
test_dates = grouped["date"].iloc[-len(X_test):].values
test_true  = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
test_preds_df = pd.DataFrame({"date": test_dates, "real": test_true})
for name in model_forecasts.keys():
    ypi = scaler_y.inverse_transform(np.array(y_preds[name]).reshape(-1, 1)).flatten()
    ypi = np.clip(ypi, 0, None)
    test_preds_df[name] = ypi
# integer хувилбарууд
for col in list(test_preds_df.columns)[2:]:
    test_preds_df[col + "_int"] = np.rint(test_preds_df[col]).astype(int)

# Ирээдүйн 12 нэгжийн таамаг: сар горимд 12 САР, өдөр горимд 365 ӨДӨР (дохио)
future_steps = 12 if agg_mode == "Сар" else 365
future_dates = (pd.date_range(start=step_next_date(last_known_date, agg_mode), periods=future_steps,
                              freq=("MS" if agg_mode == "Сар" else "D")))
future_preds_df = pd.DataFrame({"date": future_dates})
for name, model in fitted_models.items():
    if name not in y_preds:
        continue
    future_preds_df[name] = forecast_next(
        model, last_row_raw, steps=future_steps, last_date=last_known_date,
        mode=agg_mode, seasonal_cols=seasonal_cols, feature_cols=feature_cols
    )
# integer columns + clip to 0
for col in list(future_preds_df.columns)[1:]:
    future_preds_df[col] = np.clip(future_preds_df[col], 0, None)
    future_preds_df[col + "_int"] = np.rint(future_preds_df[col]).astype(int)

with pd.ExcelWriter("model_predictions.xlsx", engine="xlsxwriter") as writer:
    test_preds_df.to_excel(writer, index=False, sheet_name="Test_Predictions")
    future_preds_df.to_excel(writer, index=False, sheet_name="Future_Predictions")
with open("model_predictions.xlsx", "rb") as f:
    st.download_button(
        "Test/Forecast бүх моделийн таамаглалуудыг Excel-р татах",
        data=f,
        file_name="model_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.subheader("Test датан дээрх бодит/таамаг (толгой 10 мөр)")
st.dataframe(test_preds_df.head(10), use_container_width=True)

st.subheader(("Ирээдүйн 12 САР" if agg_mode == "Сар" else "Ирээдүйн 365 ӨДӨР") + " — прогноз (модел бүрээр)")
st.dataframe(future_preds_df, use_container_width=True)

# -------------------------- График UI --------------------------
model_options = list(model_forecasts.keys())
if len(model_options) == 0:
    st.warning("Прогноз харах модел олдсонгүй.")
else:
    # Автоматаар best model (RMSE real) default болгох
    try:
        best_model_name = results_df.loc[results_df["RMSE (real)"].idxmin(), "Model"]
    except Exception:
        best_model_name = None
    default_index = model_options.index(best_model_name) if best_model_name in model_options else 0

    selected_model = st.selectbox("Модель сонгох:", model_options, index=default_index)
    selected_h = st.selectbox("Хоризонт:", list(h_map.keys()), index=2)
    steps = h_map[selected_h]
    start_future = step_next_date(last_known_date, agg_mode)
    dates_future = pd.date_range(start=start_future, periods=steps, freq=("MS" if agg_mode == "Сар" else "D"))
    future_df = pd.DataFrame({"date": dates_future, "forecast": model_forecasts[selected_model][selected_h]})

    # Өдөр горимд гөлгөршүүлэх сонголт
    y_col = "forecast"
    if agg_mode == "Өдөр":
        smooth_win = st.sidebar.slider("Гөлгөршүүлэх цонх (өдөр)", 1, 21, 7, 1)
        if smooth_win and smooth_win > 1:
            future_df["forecast_smooth"] = future_df["forecast"].rolling(smooth_win, min_periods=1).mean()
            y_col = "forecast_smooth"

    fig = px.line(
        future_df, x="date", y=y_col, markers=True,
        title=f"{selected_model} — {selected_h} ({'сар' if agg_mode=='Сар' else 'өдөр'}ийн шагналт)"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------- Horizon metrics for selected model --------------------------

def horizon_scores(model, X_full, y_full, scaler_y, horizons):
    out = {}
    y_true_full = scaler_y.inverse_transform(y_full.reshape(-1,1)).ravel()
    for h in horizons:
        if len(X_full) < h + 1: 
            out[h] = np.nan; continue
        X_te = X_full[-h:]
        yhat_s = np.asarray(model.predict(X_te)).ravel()
        yhat   = scaler_y.inverse_transform(yhat_s.reshape(-1,1)).ravel()
        yhat = np.clip(yhat, 0, None)
        out[h] = {
            "MAE": mean_absolute_error(y_true_full[-h:], yhat),
            "RMSE": np.sqrt(mean_squared_error(y_true_full[-h:], yhat))
        }
    return out

if len(model_options) > 0:
    horizons = [1, 3, 6, 12] if agg_mode == "Сар" else [7, 14, 30, 90]
    hs = horizon_scores(fitted_models[selected_model], X_test, y_test, scaler_y, horizons)
    st.write("Horizon metrics:", hs)

# -------------------------- Naive baselines --------------------------

def naive_forecast(y):
    return np.roll(y, 1)[1:]  # t-1

def snaive_forecast(y, m):
    y_hat = np.roll(y, m)[m:]
    return y_hat

y_true_full = scaler_y.inverse_transform(y_test.reshape(-1,1)).ravel()
y_hist = scaler_y.inverse_transform(y_train.reshape(-1,1)).ravel()
y_naive = naive_forecast(np.concatenate([y_hist, y_true_full]))[-len(y_true_full):]
m_period = 12 if agg_mode == "Сар" else 7
y_snaive = snaive_forecast(np.concatenate([y_hist, y_true_full]), m=m_period)[-len(y_true_full):]

baseline_df = pd.DataFrame({
    "MAE_naive": [mean_absolute_error(y_true_full, y_naive)],
    f"MAE_snaive(m={m_period})": [mean_absolute_error(y_true_full, y_snaive)]
})
st.write("Baseline metrics:", baseline_df)

# -------------------------- 1. Корреляцийн шинжилгээ --------------------------
st.header("1. Осолд нөлөөлөх хүчин зүйлсийн тархалт/корреляцийн шинжилгээ")
st.write("Доорх multiselect-оос ихдээ 15 хувьсагч сонгож корреляцийн матрицыг үзнэ үү.")

vars_for_corr = ["Year"]
vars_for_corr += [c for c in df.columns if c.startswith("Зөрчил огноо жил ")][:10]
vars_for_corr += [c for c in (binary_cols + num_additional) if c in df.columns]
vars_for_corr = list(dict.fromkeys(vars_for_corr))

if len(vars_for_corr) > 1:
    Xx = df[vars_for_corr].fillna(0.0).values
    yy = pd.to_numeric(df["Осол"], errors="coerce").fillna(0).values
    try:
        rf_cor = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_cor.fit(Xx, yy)
        importances_cor = rf_cor.feature_importances_
        indices_cor = np.argsort(importances_cor)[::-1]
        top_k_cor = min(15, len(vars_for_corr))
        default_cols = [vars_for_corr[i] for i in indices_cor[:top_k_cor]]
    except Exception:
        default_cols = vars_for_corr[: min(15, len(vars_for_corr))]
else:
    default_cols = vars_for_corr

selected_cols = st.multiselect(
    "Корреляцийн матрицад оруулах хувьсагчид:", vars_for_corr, default=default_cols, max_selections=15
)
if selected_cols:
    st.pyplot(plot_correlation_matrix(df, "Correlation Matrix", selected_cols))
else:
    st.info("Хувьсагч сонгоно уу.")

# -------------------------- 2. Ослын өсөлтийн тренд --------------------------
st.header("2. Ослын өсөлтийн тренд")
st.subheader("Жил, сар бүрийн ослын тоо")
trend_data = (
    df[df["Осол"] == 1]
    .groupby(["Year", "Month"]).agg(osol_count=("Осол", "sum"))
    .reset_index()
)
trend_data["YearMonth"] = trend_data.apply(lambda x: f"{int(x['Year'])}-{int(x['Month']):02d}", axis=1)
available_years = sorted(trend_data["Year"].unique())
year_options = ["Бүгд"] + [str(y) for y in available_years]
selected_year = st.selectbox("Жил сонгох:", year_options)
plot_df = trend_data if selected_year == "Бүгд" else trend_data[trend_data["Year"] == int(selected_year)].copy()
fig = px.line(plot_df, x="YearMonth", y="osol_count", markers=True,
              labels={"YearMonth": "Он-Сар", "osol_count": "Ослын тоо"}, title="")
fig.update_layout(xaxis_tickangle=45, hovermode="x unified", plot_bgcolor="white",
                  yaxis=dict(title="Ослын тоо", rangemode="tozero"), xaxis=dict(title="Он-Сар"))
fig.update_traces(line=dict(width=3))
st.plotly_chart(fig, use_container_width=True)

# -------------------------- 4. Категори хамаарал (Cramér’s V + Chi-square) --------------------------
st.header("4. Категори хувьсагчдын хоорондын хамаарал (Cramér’s V болон Chi-square)")
low_card_cols = []
for c in df.columns:
    if c in ["Осол", "Зөрчил огноо", "Year", "Month", "Day"]:
        continue
    u = df[c].dropna().unique()
    if 2 <= len(u) <= 15:
        low_card_cols.append(c)

categorical_cols = sorted(list(set(binary_cols + low_card_cols)))
if len(categorical_cols) < 2:
    st.info("Категори багана (2–15 түвшин) олдсонгүй.")
else:
    var1 = st.selectbox("1-р категори хувьсагч:", categorical_cols)
    var2 = st.selectbox("2-р категори хувьсагч:", [c for c in categorical_cols if c != var1])

    table = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    cramers_v = np.sqrt(chi2 / (n * (min(k, r) - 1))) if min(k, r) > 1 else np.nan

    st.subheader("1. Chi-square тест")
    st.write(f"**Chi-square statistic:** {chi2:.3f}")
    st.write(f"**p-value:** {p:.4f}")
    st.info("Тайлбар: p-value < 0.05 бол статистикийн хувьд хамааралтай гэж үзнэ.")

    st.subheader("2. Cramér’s V")
    st.write(f"**Cramér’s V:** {cramers_v:.3f} (0=хамааралгүй, 1=хүчтэй хамаарал)")

    st.write("**Crosstab:**")
    st.dataframe(table, use_container_width=True)

# -------------------------- Улирлын ялгаа --------------------------

def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Өвөл"
    elif month in [3, 4, 5]:
        return "Хавар"
    elif month in [6, 7, 8]:
        return "Зун"
    elif month in [9, 10, 11]:
        return "Намар"
    return "Тодорхойгүй"

df["Season"] = df["Зөрчил огноо"].dt.month.apply(get_season)
table = pd.crosstab(df["Season"], df[torol_col])
chi2, p, dof, exp = chi2_contingency(table)
n = table.values.sum()
r, k = table.shape
cramers_v = np.sqrt(chi2 / (n*(min(k,r)-1)))
st.subheader("Улирлын ялгаа (χ² ба Cramér’s V)")
st.write("**Chi-square statistic:**", round(chi2, 3))
st.write("**p-value:**", round(p, 4))
st.write("**Cramér’s V:**", round(cramers_v, 3))
st.dataframe(table, use_container_width=True)

# -------------------------- 6. Empirical Bayes (сар бүр) --------------------------
st.header("6. Empirical Bayes before–after шинжилгээ (сар бүр)")

def empirical_bayes(obs, exp, prior_mean, prior_var):
    """EB жигнэлт"""
    weight = prior_var / (prior_var + exp) if (prior_var + exp) > 0 else 0.0
    return weight * obs + (1 - weight) * prior_mean

monthly = (
    df[df["Осол"] == 1]
    .groupby(["Year", "Month"])
    .agg(osol_count=("Осол", "sum"))
    .reset_index()
)
monthly["date"] = pd.to_datetime(monthly[["Year", "Month"]].assign(DAY=1))
monthly["period"] = np.where(monthly["Year"] <= 2023, "before", "after")

expected = monthly[monthly["period"]=="before"]["osol_count"].mean()
prior_mean = float(expected) if pd.notna(expected) else 0.0
prior_var  = prior_mean / 2 if prior_mean > 0 else 1.0

monthly["EB"] = monthly.apply(
    lambda row: empirical_bayes(row["osol_count"], prior_mean, prior_mean, prior_var)
    if row["period"]=="after" else row["osol_count"],
    axis=1
)
fig = px.line(monthly, x="date", y=["osol_count","EB"], color="period", markers=True,
              labels={"value":"Осол (тоо)", "date":"Он-Сар"},
              title="Ослын сар бүрийн тоо (EB жигнэлт)")
st.plotly_chart(fig, use_container_width=True)
