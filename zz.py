# -*- coding: utf-8 -*-
# ============================================================
# Зам тээврийн осол — Auto ML & Hotspot Dashboard (Streamlit)
# Хувилбар: 2025-08-17r3.5 (cached+TA+fix+leap+debias+LY365)
# ============================================================

from __future__ import annotations
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path

# Sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, StackingRegressor, HistGradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance

# 3rd-party (optional)
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

# (map packages kept for other tabs)
import folium  # noqa
from folium.plugins import MarkerCluster  # noqa
from streamlit_folium import st_folium  # noqa
import matplotlib.cm as cm  # noqa
import matplotlib.colors as mcolors  # noqa

RANDOM_STATE = 42

# -------------------------- UI setup --------------------------
st.set_page_config(page_title="Осол — Auto ML & Hotspot (auto-binary)", layout="wide")
st.title("С.Цолмон, А.Тамир нарын хар цэгийн судалгаа 2025-08-18")

# -------------------------- Helpers --------------------------
def _canon(s: str) -> str:
    return "".join(str(s).lower().split()) if isinstance(s, str) else str(s)

def resolve_col(df: pd.DataFrame, candidates) -> str | None:
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
    corr_matrix = df_encoded.corr().iloc[::-1]
    fig, ax = plt.subplots(figsize=(max(8, 1.5*len(columns)), max(6, 1.2*len(columns))))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax, fmt=".3f")
    ax.set_title(title)
    plt.tight_layout()
    return fig

# ---------- NEW: safe integer rounding (avoids IntCastingNaNError) ----------
# ---------- REPLACE your to_int_safe with this ----------
def to_int_safe(series: pd.Series) -> pd.Series:
    # 1) Тоон руу хүчээр
    s = pd.to_numeric(series, errors="coerce")
    # 2) ±inf -> NaN
    s = s.replace([np.inf, -np.inf], np.nan)

    # 3) Хөвөгчийг тойруулж бүхэл рүү ойртуулах
    rounded = np.rint(s.values.astype("float64"))

    # 4) NaN-уудыг түр 0 болгоод NumPy int64 рүү хөрвүүлэх (энд safe-каст шаардлагагүй)
    tmp = np.where(np.isfinite(rounded), rounded, 0.0)
    ints = tmp.astype(np.int64)

    # 5) Series болгож nullable Int64 болгоод, NaN масктай мөрүүдийг <NA> болгон сэргээх
    out = pd.Series(ints, index=series.index, copy=False).astype("Int64")
    mask = ~np.isfinite(s.values)  # NaN/inf байсан байршлууд
    if mask.any():
        out[mask] = pd.NA
    return out


# -------------------------- Data load (cached) --------------------------
uploaded_file = st.sidebar.file_uploader("Excel файл оруулах (.xlsx)", type=["xlsx"])

@st.cache_data(show_spinner=True)
def load_data(file_bytes: bytes | None, default_path: str = "кодлогдсон.xlsx"):
    if file_bytes is not None:
        df = pd.read_excel(file_bytes)
    else:
        local = Path(default_path)
        if not local.exists():
            raise FileNotFoundError(f"Excel файл олдсонгүй: {default_path}")
        df = pd.read_excel(local)

    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    recv_col = resolve_col(df, ["Хүлээн авсан", "Хүлээн авсан ", "Огноо", "Зөрчил огноо",
                                "Осол огноо", "Ослын огноо", "Date"])
    if recv_col is None:
        raise ValueError("Огнооны багана олдсонгүй. Жишээ: 'Хүлээн авсан'.")
    df["Зөрчил огноо"] = pd.to_datetime(df[recv_col], errors="coerce")
    if df["Зөрчил огноо"].isna().all():
        raise ValueError("Огноог parse хийж чадсангүй. Огнооны формат шалгана уу.")

    df["Year"]  = df["Зөрчил огноо"].dt.year
    df["Month"] = df["Зөрчил огноо"].dt.month
    df["Day"]   = df["Зөрчил огноо"].dt.day_name()

    years = sorted(df["Year"].dropna().unique().tolist())
    for y in years:
        df[f"Зөрчил огноо жил {int(y)}"] = (df["Year"] == int(y)).astype(int)
    if len(years) > 0:
        df["Зөрчил огноо жил (min-max)"] = df["Year"].between(min(years), max(years)).astype(int)

    lat_col = resolve_col(df, ["Өргөрөг", "lat", "latitude"])
    lon_col = resolve_col(df, ["Уртраг", "lon", "longitude"])

    exclude = {"Зөрчил огноо", "Year", "Month", "Day", "д/д", "Хороо-Сум", "Аймаг-Дүүрэг"}
    if lat_col: exclude.add(lat_col)
    if lon_col: exclude.add(lon_col)
    binary_cols = [c for c in df.columns if c not in exclude and is_binary_series(df[c])]

    numeric_candidates = []
    if "Авто зам - Зорчих хэсгийн өргөн" in df.columns:
        numeric_candidates.append("Авто зам - Зорчих хэсгийн өргөн")

    if "Дүүрэг" not in df.columns:
        df["Дүүрэг"] = 0
    if "Аймаг" not in df.columns:
        df["Аймаг"] = 0

    meta = {
        "lat_col": lat_col, "lon_col": lon_col,
        "binary_cols": binary_cols, "numeric_candidates": numeric_candidates, "years": years,
    }
    return df, meta

try:
    df, meta = load_data(uploaded_file if uploaded_file is None else uploaded_file.getvalue())
except Exception as e:
    st.error(f"Өгөгдөл ачаалахад алдаа: {e}")
    st.stop()

lat_col, lon_col = meta["lat_col"], meta["lon_col"]
binary_cols = meta["binary_cols"]
num_additional = meta["numeric_candidates"]
years = meta["years"]

# -------------------------- Target config --------------------------
st.sidebar.markdown("### 🎯 Зорилтот тодорхойлолт (Осол)")
target_mode = st.sidebar.radio("Осол гэж тооцох ангиллыг сонгоно уу:",
    ("Хоёуланг 1 гэж тооц", "Зөвхөн Гэмт хэрэг", "Зөвхөн Зөрчлийн хэрэг"))

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

# -------------------------- Forecast settings --------------------------
st.header("5. Ирээдүйн ослын таамаглал (Олон ML/DL загвар)")
agg_mode = st.sidebar.selectbox("Прогнозын агрегат", ["Сар", "Өдөр"], index=0)
st.caption("Binary (0/1) шинжүүд автоматаар илрүүлэгдэнэ. **Сар / Өдөр** хоёр горим. "
           "Өдөр горимд 7 хоног/365–366 өдрийн Фурье улирал динамикаар шинэчлэгдэнэ.")

def nonleaky(col: str) -> bool:
    s = str(col)
    if s == "Осол": return False
    if s.startswith("Зөрчил огноо жил "): return False
    if "Төрөл" in s: return False
    if s in {"Year", "Month", "Day"}: return False
    return True

feature_pool = [c for c in (binary_cols + num_additional) if nonleaky(c)]
if len(feature_pool) == 0:
    st.error("Leakage-гүй шинж олдсонгүй. Metadata/one-hot үүсгэх дүрмээ шалгана уу.")
    st.stop()

# -------------------------- TA settings (өдөр горимд) --------------------------
ta_use: bool = False
ta_params: tuple | None = None
ta_show_chart: bool = False
top_k_exog = st.sidebar.slider("Exogenous шинжийн дээд k", 8, 40, 20, 1)
ta_force_min = st.sidebar.number_input("TA-г дор хаяж N-г багтаах", 0, 15, 4, 1)
if agg_mode == "Өдөр":
    with st.sidebar.expander("📈 Техник шинжилгээ (өдөр)", expanded=True):
        ta_use = st.checkbox("Индикаторуудыг ML шинжид ашиглах", value=True)
        sma_short = st.number_input("SMA (богино, өдөр)", min_value=3, max_value=60, value=7, step=1)
        sma_long  = st.number_input("SMA (урт, өдөр)",    min_value=5, max_value=200, value=30, step=1)
        ema_short = st.number_input("EMA (богино, өдөр)", min_value=3, max_value=60, value=12, step=1)
        ema_long  = st.number_input("EMA (урт, өдөр)",    min_value=5, max_value=200, value=26, step=1)
        macd_sig  = st.number_input("MACD signal",        min_value=3, max_value=20, value=9, step=1)
        rsi_win   = st.number_input("RSI цонх",           min_value=5, max_value=60, value=14, step=1)
        bb_win    = st.number_input("Bollinger цонх",     min_value=5, max_value=60, value=20, step=1)
        bb_k      = st.number_input("Bollinger σ (k)",    min_value=1.0, max_value=4.0, value=2.0, step=0.5)
        roc_win   = st.number_input("ROC/MOM цонх",       min_value=2, max_value=60, value=7, step=1)
        ta_show_chart = st.checkbox("TA график харуулах", value=True)
        ta_dynamic_forecast = st.checkbox("Прогнозын үед TA-г динамикаар дахин тооцоолох", value=True)

    ta_params = (int(sma_short), int(sma_long), int(ema_short), int(ema_long),
                 int(rsi_win), int(macd_sig), int(bb_win), float(bb_k), int(roc_win))

# -------------------------- SERIES BUILD (cached) --------------------------
@st.cache_data(show_spinner=True)
def build_monthly_cached(df_in: pd.DataFrame, feature_pool: list, n_lag: int):
    monthly_target = (
        df_in[df_in["Осол"] == 1]
        .groupby(["Year", "Month"]).agg(osol_count=("Осол", "sum")).reset_index()
    )
    monthly_target["date"] = pd.to_datetime(monthly_target[["Year","Month"]].assign(DAY=1))

    monthly_features = (
        df_in.groupby(["Year","Month"])[feature_pool].sum().reset_index().sort_values(["Year","Month"])
    )
    for c in feature_pool:
        monthly_features[c] = monthly_features[c].shift(1)

    grouped = (pd.merge(monthly_target, monthly_features, on=["Year","Month"], how="left")
               .sort_values(["Year","Month"]).reset_index(drop=True))
    grouped["m"] = grouped["Month"].astype(int)
    K = [1,2,3]
    for k in K:
        grouped[f"m_sin_{k}"] = np.sin(2*np.pi*k*grouped["m"]/12)
        grouped[f"m_cos_{k}"] = np.cos(2*np.pi*k*grouped["m"]/12)
    fourier_cols = [f"m_sin_{k}" for k in K] + [f"m_cos_{k}" for k in K]
    lag_cols = [f"osol_lag_{i}" for i in range(1, n_lag+1)]
    for i in range(1, n_lag+1):
        grouped[f"osol_lag_{i}"] = grouped["osol_count"].shift(i)

    # ---- Year-ago & trailing features for MONTHLY mode (no leakage)
    s = grouped["osol_count"].astype(float)
    grouped["LY_12"]      = s.shift(12)
    grouped["ROLL12_SUM"] = s.rolling(12, min_periods=3).sum().shift(1)

    exog_cols = feature_pool + fourier_cols + ["LY_12", "ROLL12_SUM"]
    return grouped, lag_cols, fourier_cols, exog_cols

# ---- TA helpers (cached) ----
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(s: pd.Series, window: int) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(show_spinner=False)
def compute_ta_cached(y: pd.Series, params: tuple) -> pd.DataFrame:
    sma_s, sma_l, ema_s, ema_l, rsi_w, macd_sig, bb_w, bb_k, roc_w = params
    s = y.astype(float).copy()

    sma_short = s.rolling(sma_s).mean()
    sma_long  = s.rolling(sma_l).mean()

    ema_short = _ema(s, ema_s)
    ema_long  = _ema(s, ema_l)
    macd = ema_short - ema_long
    macd_signal = _ema(macd, macd_sig)
    macd_hist = macd - macd_signal

    rsi = _rsi(s, rsi_w)

    bb_mid = s.rolling(bb_w).mean()
    bb_std = s.rolling(bb_w).std()
    bb_up  = bb_mid + bb_k * bb_std
    bb_lo  = bb_mid - bb_k * bb_std

    roc = s.pct_change(roc_w)
    mom = s.diff(roc_w)

    vol_7  = s.rolling(7).std()
    vol_30 = s.rolling(30).std()

    df = pd.DataFrame({
        "TA_SMA_S":     sma_short.shift(1),
        "TA_SMA_L":     sma_long.shift(1),
        "TA_EMA_S":     ema_short.shift(1),
        "TA_EMA_L":     ema_long.shift(1),
        "TA_MACD":      macd.shift(1),
        "TA_MACD_SIG":  macd_signal.shift(1),
        "TA_MACD_HIST": macd_hist.shift(1),
        "TA_RSI":       rsi.shift(1),
        "TA_BB_UP":     bb_up.shift(1),
        "TA_BB_MID":    bb_mid.shift(1),
        "TA_BB_LO":     bb_lo.shift(1),
        "TA_ROC":       roc.shift(1),
        "TA_MOM":       mom.shift(1),
        "TA_VOL_7":     vol_7.shift(1),
        "TA_VOL_30":    vol_30.shift(1),
    })
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

@st.cache_data(show_spinner=True)
def build_daily_cached(df_in: pd.DataFrame, feature_pool: list, n_lag: int,
                       ta_use: bool, ta_params: tuple | None):
    df_temp = df_in.copy()
    df_temp["date"] = df_temp["Зөрчил огноо"].dt.floor("D")
    start, end = df_temp["date"].min().normalize(), df_temp["date"].max().normalize()
    all_days = pd.date_range(start, end, freq="D", name="date")

    daily_target = (
        df_temp.groupby("date")["Осол"].sum().reindex(all_days, fill_value=0)
        .rename("osol_count").rename_axis("date").reset_index()
    )
    if feature_pool:
        daily_features = (
            df_temp.groupby("date")[feature_pool].sum().reindex(all_days, fill_value=0)
            .rename_axis("date").reset_index()
        )
        for c in feature_pool:
            daily_features[c] = daily_features[c].shift(1)
    else:
        daily_features = pd.DataFrame({"date": all_days}).reset_index(drop=True)

    grouped = (pd.merge(daily_target, daily_features, on="date", how="left")
               .sort_values("date").reset_index(drop=True))

    # Weekly + Yearly Fourier (year length aware)
    Kw, Ky = [1,2,3], [1]
    dow = grouped["date"].dt.dayofweek
    doy = grouped["date"].dt.dayofyear
    year_len = np.where(grouped["date"].dt.is_leap_year, 366, 365).astype(float)

    for k in Kw:
        grouped[f"w_sin_{k}"] = np.sin(2*np.pi*k*dow/7)
        grouped[f"w_cos_{k}"] = np.cos(2*np.pi*k*dow/7)
    for k in Ky:
        grouped[f"y_sin_{k}"] = np.sin(2*np.pi*k*doy/year_len)
        grouped[f"y_cos_{k}"] = np.cos(2*np.pi*k*doy/year_len)

    fourier_cols = (
        [f"w_sin_{k}" for k in Kw] + [f"w_cos_{k}" for k in Kw] +
        [f"y_sin_{k}" for k in Ky] + [f"y_cos_{k}" for k in Ky]
    )

    lag_cols = [f"osol_lag_{i}" for i in range(1, n_lag+1)]
    for i in range(1, n_lag+1):
        grouped[f"osol_lag_{i}"] = grouped["osol_count"].shift(i)

    grouped["Year"] = grouped["date"].dt.year
    grouped["Month"] = grouped["date"].dt.month

    # ---- Year-ago & trailing features for DAILY mode (no leakage)
    s = grouped["osol_count"].astype(float)
    grouped["LY_365"]  = s.shift(365)
    grouped["LY_366"]  = s.shift(366)
    grouped["LY_MEAN"] = grouped[["LY_365", "LY_366"]].mean(axis=1)

    grouped["ROLL365_SUM"]  = s.rolling(365, min_periods=30).sum().shift(1)
    grouped["ROLL365_MEAN"] = s.rolling(365, min_periods=30).mean().shift(1)
    grouped["ROLL30_SUM"]   = s.rolling(30,  min_periods=7).sum().shift(1)

    # DOY-based previous-year alignment (handles leap years)
    doy_all = grouped["date"].dt.dayofyear
    doy_norm = np.where(doy_all == 366, 365, doy_all)
    grouped["DOY_NORM"] = doy_norm
    grouped = grouped.sort_values(["DOY_NORM", "Year"])
    grouped["LY_DOY_1"] = grouped.groupby("DOY_NORM")["osol_count"].shift(1)
    grouped["LY_DOY_2"] = grouped.groupby("DOY_NORM")["osol_count"].shift(2)
    grouped["LY_DOY_MEAN_2"] = grouped[["LY_DOY_1","LY_DOY_2"]].mean(axis=1)
    grouped = grouped.sort_values("date")  # restore chronological order

    exog_cols = feature_pool + fourier_cols + [
        "LY_365","LY_366","LY_MEAN",
        "ROLL365_SUM","ROLL365_MEAN","ROLL30_SUM",
        "LY_DOY_1","LY_DOY_2","LY_DOY_MEAN_2"
    ]

    # TA индикаторууд
    if ta_params is not None:
        ta_df = compute_ta_cached(grouped["osol_count"], ta_params)
        grouped = pd.concat([grouped, ta_df], axis=1)
        if ta_use:
            exog_cols = exog_cols + list(ta_df.columns)

    return grouped, lag_cols, fourier_cols, exog_cols

# sliders (these trigger rebuild when changed)
if agg_mode == "Сар":
    n_lag = st.sidebar.slider("Сарын лаг цонх (n_lag)", 6, 18, 12, 1, key="lag_m")
    grouped, lag_cols, seasonal_cols, exog_cols = build_monthly_cached(df, feature_pool, n_lag)
    freq_code = "MS"
else:
    n_lag = st.sidebar.slider("Өдрийн лаг цонх (n_lag)", 7, 365, 30, 1, key="lag_d")
    grouped, lag_cols, seasonal_cols, exog_cols = build_daily_cached(df, feature_pool, n_lag, ta_use, ta_params)
    freq_code = "D"

grouped = grouped.dropna().reset_index(drop=True)
if grouped.empty or len(grouped) < max(10, n_lag + 5):
    st.warning(f"Сургалт хийхэд хангалттай өгөгдөл алга (lag={n_lag}, mode={agg_mode}). Он/сар/өдрийн хүрээг шалгана уу.")
    st.stop()

# >>> ADD: цагийн трендийг заавал оруулах
grouped["T"]  = np.arange(len(grouped))
grouped["T2"] = grouped["T"] ** 2
exog_cols = list(dict.fromkeys(exog_cols + ["T", "T2"]))

split_ratio = st.sidebar.slider("Train ratio", 0.5, 0.9, 0.8, 0.05)

# -------------------------- Feature selection (cached) --------------------------
@st.cache_data(show_spinner=True)
def select_top_exog_cached(grouped: pd.DataFrame, lag_cols: list, exog_cols: list, split_ratio: float, top_k: int = 14):
    X_all_fs = grouped[lag_cols + exog_cols].fillna(0.0).values
    y_all = grouped["osol_count"].values.reshape(-1, 1)
    train_size = int(len(X_all_fs) * split_ratio)
    X_train_fs, y_train_fs = X_all_fs[:train_size], y_all[:train_size].ravel()
    try:
        rf_fs = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        rf_fs.fit(X_train_fs, y_train_fs)
        imp_series = pd.Series(rf_fs.feature_importances_, index=lag_cols + exog_cols)
        k = min(top_k, len(exog_cols))
        exog_top = imp_series.loc[exog_cols].sort_values(ascending=False).head(k).index.tolist()
    except Exception:
        exog_top = exog_cols[:min(top_k, len(exog_cols))]
    return exog_top

# TA-г дор хаяж N багтаах логик
exog_top = select_top_exog_cached(grouped, lag_cols, exog_cols, split_ratio, top_k=top_k_exog)
st.caption("Train дээр тодорсон top exogenous features (leakage-гүй):")
st.write(exog_top)

ta_cols_all = [c for c in exog_cols if c.startswith("TA_")]
if ta_use and ta_force_min > 0:
    must_have_ta = [c for c in ta_cols_all if c not in exog_top][:ta_force_min]
    exog_top = list(dict.fromkeys(exog_top + must_have_ta))  # давхардалгүй union

# Year-ago & rolling features-ийг дор хаяж үндсэн хэсгийг хүчээр багтаая
force_ly = [c for c in [
    "LY_365","LY_366","LY_MEAN","ROLL365_SUM","ROLL365_MEAN",  # өдөр горимын гол
    "LY_DOY_1","LY_DOY_2","LY_DOY_MEAN_2",
    "LY_12","ROLL12_SUM"  # сар горимын гол
] if c in exog_cols and c not in exog_top]
exog_top = list(dict.fromkeys(exog_top + force_ly))

# >>> FORCE-IN: Fourier улирал + цагийн тренд + LY/ROLL
exog_top = list(dict.fromkeys(seasonal_cols + ["T","T2"] + exog_top))

feature_cols = lag_cols + exog_top
if ta_use and not any(c.startswith("TA_") for c in exog_top):
    st.warning("TA асаалттай боловч feature selection TA-г хассан байна (exog_top-д TA_* алга). "
               "k-г өсгөх эсвэл N-г нэмнэ үү.")

# -------------------------- Make matrices --------------------------
X = grouped[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

y = grouped["osol_count"].values.reshape(-1, 1)
train_size = int(len(X) * split_ratio)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

scaler_X = MinMaxScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s  = scaler_X.transform(X_test)

# log1p/expm1 scale (count data-д илүү зөөлөн)
scaler_y = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=False)
y_train_s = scaler_y.fit_transform(y_train).ravel()
y_test_s  = scaler_y.transform(y_test).ravel()

identicals = [c for c in feature_cols if np.allclose(grouped[c].values, grouped["osol_count"].values, equal_nan=False)]
if identicals:
    st.error(f"IDENTICAL leakage үлдлээ: {identicals}")
    st.stop()
corrs = grouped[feature_cols].corrwith(grouped["osol_count"])
st.write("Target-тэй корреляци (дээд 10):", corrs.head(10))

# -------------------------- Model zoo --------------------------
estimators = [
    ("rf", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
    ("ridge", Ridge()),
    ("dt", DecisionTreeRegressor(random_state=RANDOM_STATE)),
]
MODEL_LIST = [
    ("LinearRegression", LinearRegression()),
    ("Ridge", Ridge()),
    ("Lasso", Lasso()),
    ("ElasticNet", ElasticNet()),
    ("DecisionTree", DecisionTreeRegressor(random_state=RANDOM_STATE)),
    ("RandomForest", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ("ExtraTrees", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ("GradientBoosting", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ("HistGB-Poisson", HistGradientBoostingRegressor(loss="poisson", learning_rate=0.06, max_depth=None, random_state=RANDOM_STATE)),
    ("AdaBoost", AdaBoostRegressor(random_state=RANDOM_STATE)),
    ("KNeighbors", KNeighborsRegressor()),
    ("SVR", SVR()),
    ("MLPRegressor", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=RANDOM_STATE)),
    ("Stacking", StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=5)),
]
if XGBRegressor is not None:
    MODEL_LIST.append(("XGBRegressor", XGBRegressor(tree_method="hist", predictor="cpu_predictor", random_state=RANDOM_STATE, n_estimators=400)))
    MODEL_LIST.append(("XGB-Poisson", XGBRegressor(objective="count:poisson", tree_method="hist",
        predictor="cpu_predictor", n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE)))
if CatBoostRegressor is not None:
    MODEL_LIST.append(("CatBoostRegressor", CatBoostRegressor(task_type="CPU", random_state=RANDOM_STATE, verbose=0)))
if LGBMRegressor is not None:
    MODEL_LIST.append(("LGBMRegressor", LGBMRegressor(device="cpu", random_state=RANDOM_STATE)))
    MODEL_LIST.append(("LGBM-Poisson", LGBMRegressor(objective="poisson", metric="rmse", n_estimators=1200,
        learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, device="cpu", random_state=RANDOM_STATE)))

voting_estimators = []
if XGBRegressor is not None:
    voting_estimators.append(("xgb", XGBRegressor(tree_method="hist", predictor="cpu_predictor", random_state=RANDOM_STATE)))
if LGBMRegressor is not None:
    voting_estimators.append(("lgbm", LGBMRegressor(device="cpu", random_state=RANDOM_STATE)))
if CatBoostRegressor is not None:
    voting_estimators.append(("cat", CatBoostRegressor(task_type="CPU", random_state=RANDOM_STATE, verbose=0)))
voting_estimators += [
    ("rf", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
    ("gb", GradientBoostingRegressor(random_state=RANDOM_STATE)),
]
if len(voting_estimators) > 1:
    MODEL_LIST.append(("VotingRegressor", VotingRegressor(estimators=voting_estimators)))

stacking_estimators = [("rf", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))]
if XGBRegressor is not None:
    stacking_estimators.append(("xgb", XGBRegressor(tree_method="hist", predictor="cpu_predictor", random_state=RANDOM_STATE)))
if LGBMRegressor is not None:
    stacking_estimators.append(("lgbm", LGBMRegressor(device="cpu", random_state=RANDOM_STATE)))
if CatBoostRegressor is not None:
    stacking_estimators.append(("cat", CatBoostRegressor(task_type="CPU", verbose=0, random_state=RANDOM_STATE)))
MODEL_LIST.append(("StackingEnsemble", StackingRegressor(estimators=stacking_estimators, final_estimator=LinearRegression(), cv=5)))

# -------------------------- Train all (CACHED) --------------------------
@st.cache_resource(show_spinner=True)
def train_all_models_cached(
    X_train_s, y_train_s, X_test_s, y_test_s,
    _scaler_y, _model_list,   # ⬅️ unhashable аргументуудыг _-тай нэрлэв
    cache_signature: str
):
    results, y_preds_s, fitted_models = [], {}, {}

    for name, model in _model_list:
        try:
            model.fit(X_train_s, y_train_s)
            fitted_models[name] = model

            y_pred_s = np.asarray(model.predict(X_test_s)).reshape(-1)
            y_preds_s[name] = y_pred_s
            r2_s = r2_score(y_test_s, y_pred_s)

            y_true = _scaler_y.inverse_transform(y_test_s.reshape(-1, 1)).ravel()
            y_pred = _scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
            y_pred = np.clip(y_pred, 0, None)

            results.append({
                "Model": name,
                "MAE (real)": mean_absolute_error(y_true, y_pred),
                "RMSE (real)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "R2 (scaled)": float(r2_s),
            })
        except Exception as e:
            results.append({
                "Model": name,
                "MAE (real)": np.nan,
                "RMSE (real)": np.nan,
                "R2 (scaled)": np.nan,
                "Error": str(e),
            })

    results_df = pd.DataFrame(results).sort_values("RMSE (real)", na_position="last")
    return fitted_models, results_df, y_preds_s

sig_parts = {
    "mode": agg_mode, "n_lag": n_lag, "split": round(float(split_ratio), 4),
    "cols": tuple(feature_cols), "ver": "r3.5-ly",
    "nrows": int(len(grouped))
}
cache_signature = json.dumps(sig_parts, sort_keys=True)

with st.spinner("ML моделийг сургаж байна (эхний удаад л)…"):
    fitted_models, results_df, y_preds_s = train_all_models_cached(
        X_train_s, y_train_s, X_test_s, y_test_s,
        scaler_y, MODEL_LIST, cache_signature
    )

st.success("Моделүүд бэлэн! (дараагийн сонголтууд дээр дахин сургахгүй)")
st.dataframe(
    results_df.style.format({"MAE (real)": "{:.3f}", "RMSE (real)": "{:.3f}", "R2 (scaled)": "{:.4f}"}),
    use_container_width=True,
)

try:
    best_model_name = results_df.loc[results_df["RMSE (real)"].idxmin(), "Model"]
except Exception:
    best_model_name = list(fitted_models.keys())[0]

# -------------------------- Permutation Importance --------------------------
with st.expander("🔍 Permutation importance (test)", expanded=False):
    pi_model_name = st.selectbox("PI тооцох модель",
                                 list(fitted_models.keys()),
                                 index=list(fitted_models.keys()).index(best_model_name))
    scoring_opt = st.selectbox("Scoring (test дээр)", 
                               ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"], index=1)
    n_repeats = st.slider("n_repeats", 5, 50, 20, 1)
    est = fitted_models[pi_model_name]

    X_te_safe = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)
    y_te_safe = y_test_s

    pi = permutation_importance(est, X_te_safe, y_te_safe,
                                n_repeats=n_repeats, random_state=RANDOM_STATE,
                                scoring=scoring_opt, n_jobs=-1)

    imp_df = (pd.DataFrame({
                "feature": feature_cols,
                "importance": pi.importances_mean,
                "std": pi.importances_std,
             })
             .sort_values("importance", ascending=False)
             .reset_index(drop=True))

    st.subheader("Permutation importance (test)")
    st.dataframe(imp_df.head(50), use_container_width=True)

    imp_ta = imp_df[imp_df["feature"].str.startswith("TA_")]
    if not imp_ta.empty:
        st.caption("TA_* шинжүүдийн ач холбогдлоос дээд хэсэг")
        st.dataframe(imp_ta.head(30), use_container_width=True)

    fig_pi = px.bar(imp_df.head(30), x="importance", y="feature", error_x="std",
                    orientation="h", title="Top permutation importance (test)")
    fig_pi.update_layout(yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig_pi, use_container_width=True)

# -------------------------- Forecast helpers --------------------------
lag_count = len(lag_cols)

def seasonal_values_for_date(dt: pd.Timestamp, mode: str):
    vals = {}
    if mode == "Сар":
        m = int(dt.month); K = [1,2,3]
        for k in K:
            vals[f"m_sin_{k}"] = np.sin(2*np.pi*k*m/12)
            vals[f"m_cos_{k}"] = np.cos(2*np.pi*k*m/12)
    else:
        dow = int(dt.dayofweek)
        year_len = 366 if pd.Timestamp(dt).is_leap_year else 365
        doy = int(dt.dayofyear)
        Kw, Ky = [1,2,3], [1]
        for k in Kw:
            vals[f"w_sin_{k}"] = np.sin(2*np.pi*k*dow/7)
            vals[f"w_cos_{k}"] = np.cos(2*np.pi*k*dow/7)
        for k in Ky:
            vals[f"y_sin_{k}"] = np.sin(2*np.pi*k*doy/year_len)
            vals[f"y_cos_{k}"] = np.cos(2*np.pi*k*doy/year_len)
    return vals

def step_next_date(d: pd.Timestamp, mode: str):
    return (d + (pd.offsets.MonthBegin(1) if mode == "Сар" else pd.Timedelta(days=1))).normalize()

def forecast_next(model, last_raw_row, steps, last_date, mode,
                  seasonal_cols, feature_cols, scaler_X, scaler_y,
                  ta_dynamic=False, ta_params=None, hist_series=None,
                  resid_bias=0.0, bias_decay=0.85, alpha_snv=0.25, m_snaive=7,
                  soft_clip=True):

    preds = []
    lag_vals = last_raw_row[:lag_count].astype(float).copy()

    # Бусад exog эхний утгууд
    other_idx = {name: lag_count + i for i, name in enumerate(feature_cols[lag_count:])}
    other_exog_names = [c for c in feature_cols[lag_count:] if c not in seasonal_cols]
    other_exog_vals = {n: float(last_raw_row[other_idx[n]]) for n in other_exog_names if n in other_idx}

    # Trend суурь
    base_T = float(other_exog_vals.get("T", 0.0))

    # TA / series
    ta_cols_present = [c for c in other_exog_names if c.startswith("TA_")]
    series = list(map(float, hist_series)) if hist_series is not None else []
    cur_date = pd.to_datetime(last_date)

    for t in range(1, steps+1):
        cur_date = step_next_date(cur_date, mode)
        seas = seasonal_values_for_date(cur_date, mode)

        exog_vector = []
        for name in feature_cols[lag_count:]:
            if name in seasonal_cols:
                exog_vector.append(seas.get(name, 0.0))

            elif name == "T":
                exog_vector.append(base_T + t)
            elif name == "T2":
                tt = base_T + t
                exog_vector.append(tt * tt)

            elif name.startswith("TA_") and ta_dynamic and ta_params is not None:
                s_tmp = pd.Series(series, dtype=float)
                ta_last = compute_ta_cached(s_tmp, ta_params).iloc[-1]
                exog_vector.append(float(ta_last.get(name, 0.0)))

            elif name in {"LY_365","LY_366","LY_MEAN","ROLL365_SUM","ROLL365_MEAN","ROLL30_SUM",
                          "LY_DOY_1","LY_DOY_2","LY_DOY_MEAN_2","LY_12","ROLL12_SUM"}:
                def safe_tail(arr, k): return float(arr[-k]) if len(arr) >= k else float("nan")
                if mode == "Өдөр":
                    if name == "LY_365":      val = safe_tail(series, 365)
                    elif name == "LY_366":    val = safe_tail(series, 366)
                    elif name == "LY_MEAN":   val = np.nanmean([safe_tail(series,365), safe_tail(series,366)])
                    elif name == "ROLL365_SUM":  val = float(np.nansum(series[-365:])) if len(series) else float("nan")
                    elif name == "ROLL365_MEAN": val = float(np.nanmean(series[-365:])) if len(series) else float("nan")
                    elif name == "ROLL30_SUM":   val = float(np.nansum(series[-30:]))  if len(series) else float("nan")
                    elif name in {"LY_DOY_1","LY_DOY_2","LY_DOY_MEAN_2"}:
                        ly1 = np.nanmean([safe_tail(series,365), safe_tail(series,366)])
                        ly2 = np.nanmean([safe_tail(series,730), safe_tail(series,731), safe_tail(series,732)])
                        val = ly1 if name=="LY_DOY_1" else ly2 if name=="LY_DOY_2" else np.nanmean([ly1, ly2])
                else:
                    if name == "LY_12":        val = safe_tail(series, 12)
                    elif name == "ROLL12_SUM": val = float(np.nansum(series[-12:])) if len(series) else float("nan")
                exog_vector.append(val)
            else:
                exog_vector.append(other_exog_vals.get(name, 0.0))

        seq_raw = np.concatenate([lag_vals, np.array(exog_vector, float)]).reshape(1, -1)
        seq_raw = np.nan_to_num(seq_raw, nan=0.0, posinf=0.0, neginf=0.0)
        seq_scaled = scaler_X.transform(seq_raw)

        p_scaled = float(np.asarray(model.predict(seq_scaled)).ravel()[0])
        p = float(scaler_y.inverse_transform(np.array([[p_scaled]])).ravel()[0])
        if not np.isfinite(p): p = 0.0
        p = max(p, 0.0)

        # Seasonal naive
        p_snaive = float(series[-m_snaive]) if len(series) >= m_snaive else (series[-1] if len(series) else p)

        # Residual debias + blend
        bias_t = resid_bias * (bias_decay ** (t-1))
        p = (1 - alpha_snv) * p + alpha_snv * p_snaive + bias_t

        # Soft-clip (optional)
        if soft_clip and len(series) >= 20:
            lo = float(np.nanquantile(series, 0.01))
            hi = float(np.nanquantile(series, 0.99) * 1.2)
            p = float(np.clip(p, max(0.0, lo*0.5), hi))

        preds.append(p)

        # update lags & series
        lag_vals = np.roll(lag_vals, 1); lag_vals[0] = p
        series.append(p)

    return np.array(preds)

# -------------------------- Build test/future tables --------------------------
test_dates = grouped["date"].iloc[-len(X_test):].values
test_true  = scaler_y.inverse_transform(y_test_s.reshape(-1, 1)).flatten()
test_preds_df = pd.DataFrame({"date": test_dates, "real": test_true})
for name, yps in y_preds_s.items():
    ypi = scaler_y.inverse_transform(np.array(yps).reshape(-1, 1)).flatten()
    test_preds_df[name] = np.clip(ypi, 0, None)

# SAFE int columns (nullable)
for col in list(test_preds_df.columns)[2:]:
    test_preds_df[col + "_int"] = to_int_safe(test_preds_df[col])

# ------------------ Калибровк/холимог UI ------------------
unit_label = "сар" if agg_mode == "Сар" else "өдөр"
with st.sidebar.expander("Прогнозын калибровк/холимог", expanded=True):
    resid_win  = st.slider(f"Калибровкын цонх W ({unit_label})",
                           1 if agg_mode=="Сар" else 7,
                           12 if agg_mode=="Сар" else 60,
                           2 if agg_mode=="Сар" else 14, 1)
    bias_decay = st.slider("Bias decay (0..1)", 0.0, 1.0, 0.85, 0.05)
    alpha_snv  = st.slider("sNaive холих жин α", 0.0, 1.0, 0.10, 0.05)
    use_soft_clip = st.checkbox("Soft-clip идэвхжүүлэх", value=False)

W = int(min(resid_win, len(test_preds_df)))
bias_by_model = {m: float((test_preds_df["real"].tail(W) - test_preds_df[m].tail(W)).mean())
                 for m in fitted_models.keys()}

last_row_raw = grouped[feature_cols].iloc[-1].values
last_known_date = grouped["date"].iloc[-1]

future_steps = 12 if agg_mode == "Сар" else 365
future_dates = pd.date_range(start=step_next_date(last_known_date, agg_mode), periods=future_steps, freq=freq_code)
future_preds_df = pd.DataFrame({"date": future_dates})
for name, model in fitted_models.items():
    m_snaive = 12 if agg_mode=="Сар" else 7
    future_preds_df[name] = forecast_next(
        model, last_row_raw, future_steps, last_known_date, agg_mode,
        seasonal_cols, feature_cols, scaler_X, scaler_y,
        ta_dynamic=(agg_mode=="Өдөр" and ta_dynamic_forecast),
        ta_params=ta_params,
        hist_series=grouped["osol_count"].values,
        resid_bias=bias_by_model.get(name, 0.0),
        bias_decay=bias_decay, alpha_snv=alpha_snv, m_snaive=m_snaive,
        soft_clip=use_soft_clip
    )

for col in list(future_preds_df.columns)[1:]:
    future_preds_df[col] = np.clip(future_preds_df[col], 0, None)
    future_preds_df[col + "_int"] = to_int_safe(future_preds_df[col])

with pd.ExcelWriter("model_predictions.xlsx", engine="xlsxwriter") as writer:
    test_preds_df.to_excel(writer, index=False, sheet_name="Test_Predictions")
    future_preds_df.to_excel(writer, index=False, sheet_name="Future_Predictions")
with open("model_predictions.xlsx", "rb") as f:
    st.download_button("Test/Forecast бүх моделийн таамаглалуудыг Excel-р татах",
        data=f, file_name="model_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.subheader("Test датан дээрх бодит/таамаг (толгой 10000 мөр)")
st.dataframe(test_preds_df.head(10000), use_container_width=True)

st.subheader(("Ирээдүйн 12 САР" if agg_mode == "Сар" else "Ирээдүйн 365 ӨДӨР") + " — прогноз (модел бүрээр)")
st.dataframe(future_preds_df, use_container_width=True)

# -------------------------- Graph UI (slice only; no recompute) --------------------------
model_options = list(fitted_models.keys())
if len(model_options) == 0:
    st.warning("Прогноз харах модел олдсонгүй.")
else:
    default_index = model_options.index(best_model_name) if best_model_name in model_options else 0
    selected_model = st.selectbox("Модель сонгох:", model_options, index=default_index)
    if agg_mode == "Сар":
        h_map = {"7 хоног (≈1 сар)": 1, "14 хоног (≈1 сар)": 1, "30 хоног (≈1 сар)": 1,
                 "90 хоног (≈3 сар)": 3, "180 хоног (≈6 сар)": 6, "365 хоног (≈12 сар)": 12}
    else:
        h_map = {"7 хоног": 7, "14 хоног": 14, "30 хоног": 30, "90 хоног": 90, "180 хоног": 180, "365 хоног": 365}
    selected_h = st.selectbox("Хоризонт:", list(h_map.keys()), index=2)
    steps = h_map[selected_h]

    view_df = future_preds_df[["date", selected_model]].iloc[:steps].rename(columns={selected_model: "forecast"})

    y_col = "forecast"
    if agg_mode == "Өдөр":
        smooth_win = st.sidebar.slider("Гөлгөршүүлэх цонх (өдөр)", 1, 21, 7, 1)
        if smooth_win > 1:
            view_df["forecast_smooth"] = view_df["forecast"].rolling(smooth_win, min_periods=1).mean()
            y_col = "forecast_smooth"

    fig = px.line(view_df, x="date", y=y_col, markers=True,
                  title=f"{selected_model} — {selected_h} ({'сар' if agg_mode=='Сар' else 'өдөр'}ийн шагналт)")
    st.plotly_chart(fig, use_container_width=True)

# Олон модель график
with st.expander("Ирээдүйн 365 өдөр — график (олон моделиор)", expanded=False):
    models_to_plot = st.multiselect("Графикт оруулах моделүүд", model_options, default=[best_model_name])
    if models_to_plot:
        fig_all = px.line(future_preds_df, x="date", y=models_to_plot, markers=True,
                          title="Ирээдүйн 365 өдөр — олон моделиор")
        st.plotly_chart(fig_all, use_container_width=True)

# -------------------------- Horizon metrics (no retrain) --------------------------
def horizon_scores(model, X_full_s, y_full_s, scaler_y, horizons):
    out = {}
    y_true_full = scaler_y.inverse_transform(y_full_s.reshape(-1,1)).ravel()
    for h in horizons:
        if len(X_full_s) < h + 1:
            out[h] = np.nan; continue
        X_te = X_full_s[-h:]
        yhat_s = np.asarray(model.predict(X_te)).ravel()
        yhat   = scaler_y.inverse_transform(yhat_s.reshape(-1,1)).ravel()
        yhat = np.clip(yhat, 0, None)
        out[h] = {"MAE": mean_absolute_error(y_true_full[-h:], yhat),
                  "RMSE": float(np.sqrt(mean_squared_error(y_true_full[-h:], yhat)))}
    return out

if len(model_options) > 0:
    horizons = [1, 3, 6, 12] if agg_mode == "Сар" else [7, 14, 30, 90]
    hs = horizon_scores(fitted_models[best_model_name], X_test_s, y_test_s, scaler_y, horizons)
    st.write("Horizon metrics:", hs)

# -------------------------- Baselines (no retrain) --------------------------
def naive_forecast(y): return np.roll(y, 1)[1:]
def snaive_forecast(y, m): return np.roll(y, m)[m:]

y_true_full = scaler_y.inverse_transform(y_test_s.reshape(-1,1)).ravel()
y_hist = scaler_y.inverse_transform(y_train_s.reshape(-1,1)).ravel()
y_naive = naive_forecast(np.concatenate([y_hist, y_true_full]))[-len(y_true_full):]
m_period = 12 if agg_mode == "Сар" else 7
y_snaive = snaive_forecast(np.concatenate([y_hist, y_true_full]), m=m_period)[-len(y_true_full):]
label_unit = "сар" if agg_mode=="Сар" else "өдөр"
baseline_df = pd.DataFrame({
    "MAE_naive": [mean_absolute_error(y_true_full, y_naive)],
    f"MAE_snaive(m={m_period}, {label_unit})": [mean_absolute_error(y_true_full, y_snaive)]
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
        rf_cor = RandomForestRegressor(n_estimATORS=200, random_state=RANDOM_STATE, n_jobs=-1)
    except TypeError:
        rf_cor = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf_cor.fit(Xx, yy)
    importances_cor = rf_cor.feature_importances_
    indices_cor = np.argsort(importances_cor)[::-1]
    top_k_cor = min(15, len(vars_for_corr))
    default_cols = [vars_for_corr[i] for i in indices_cor[:top_k_cor]]
else:
    default_cols = vars_for_corr

selected_cols = st.multiselect("Корреляцийн матрицад оруулах хувьсагчид:",
                               vars_for_corr, default=default_cols, max_selections=15)
if selected_cols:
    st.pyplot(plot_correlation_matrix(df, "Correlation Matrix", selected_cols))
else:
    st.info("Хувьсагч сонгоно уу.")

# -------------------------- 2. Ослын өсөлтийн тренд --------------------------
st.header("2. Ослын өсөлтийн тренд")
st.subheader("Жил, сар бүрийн ослын тоо")
trend_data = (
    df[df["Осол"] == 1].groupby(["Year", "Month"]).agg(osol_count=("Осол", "sum")).reset_index()
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
    if month in [12, 1, 2]: return "Өвөл"
    elif month in [3, 4, 5]: return "Хавар"
    elif month in [6, 7, 8]: return "Зун"
    elif month in [9, 10, 11]: return "Намар"
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
    weight = prior_var / (prior_var + exp) if (prior_var + exp) > 0 else 0.0
    return weight * obs + (1 - weight) * prior_mean
monthly = (
    df[df["Осол"] == 1].groupby(["Year","Month"]).agg(osol_count=("Осол", "sum")).reset_index()
)
monthly["date"] = pd.to_datetime(monthly[["Year","Month"]].assign(DAY=1))
monthly["period"] = np.where(monthly["Year"] <= 2023, "before", "after")
expected = monthly[monthly["period"]=="before"]["osol_count"].mean()
prior_mean = float(expected) if pd.notna(expected) else 0.0
prior_var  = prior_mean / 2 if prior_mean > 0 else 1.0

mu_m = (monthly[monthly["period"]=="before"]
        .groupby("Month")["osol_count"].mean())

def eb_row(row):
    if row["period"] == "after":
        exp_t = float(mu_m.get(row["Month"], prior_mean))
        return empirical_bayes(row["osol_count"], exp_t, prior_mean, prior_var)
    return row["osol_count"]

monthly["EB"] = monthly.apply(eb_row, axis=1)
fig = px.line(monthly, x="date", y=["osol_count","EB"], color="period", markers=True,
              labels={"value":"Осол (тоо)", "date":"Он-Сар"},
              title="Ослын сар бүрийн тоо (EB жигнэлт)")
st.plotly_chart(fig, use_container_width=True)

def neg_mae_real(y_true_s, y_pred_s):
    y_t = scaler_y.inverse_transform(y_true_s.reshape(-1,1)).ravel()
    y_p = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).ravel()
    return -mean_absolute_error(y_t, y_p)

mae_real_scorer = make_scorer(neg_mae_real, greater_is_better=False)

@st.cache_data(show_spinner=True)
def select_top_exog_by_pi_cached(estimator, X_te, y_te, feature_cols, lag_cols, exog_cols,
                                 top_k: int = 20, scoring="r2"):
    exog_idx = [feature_cols.index(c) for c in exog_cols if c in feature_cols]
    if len(exog_idx) == 0:
        return pd.DataFrame(columns=["feature","importance","std"]), []
    X_te_exog = X_te[:, exog_idx]
    pi = permutation_importance(estimator, X_te_exog, y_te, n_repeats=20,
                                random_state=RANDOM_STATE, scoring=scoring, n_jobs=-1)
    df_pi = (pd.DataFrame({"feature": [exog_cols[i] for i in range(len(exog_idx))],
                           "importance": pi.importances_mean,
                           "std": pi.importances_std})
                .sort_values("importance", ascending=False))
    return df_pi, df_pi["feature"].head(min(top_k, len(exog_cols))).tolist()

use_pi_for_fs = st.sidebar.checkbox("Feature selection-д PI ашиглах", value=False)
if use_pi_for_fs:
    df_pi, exog_top_pi = select_top_exog_by_pi_cached(
        fitted_models[best_model_name], X_test_s, y_test_s, feature_cols, lag_cols, exog_cols,
        top_k=top_k_exog, scoring="r2"
    )
    st.write("PI-д суурилсан top exogenous:")
    st.dataframe(df_pi.head(30), use_container_width=True)
