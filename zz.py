# -*- coding: utf-8 -*-
# ============================================================
# Зам тээврийн осол — Auto ML & Hotspot Dashboard (Streamlit)
# Хувилбар: 2025-08-21r4.2-RF+LSTM (anti-leak FS + RF Direct + seasonal bias fix)
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
from typing import List, Dict, Tuple

# Sklearn
from sklearn.base import clone
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from scipy.stats import chi2_contingency

# (map packages kept for other tabs)
import folium  # noqa
from folium.plugins import MarkerCluster  # noqa
from streamlit_folium import st_folium  # noqa
import matplotlib.cm as cm  # noqa
import matplotlib.colors as mcolors  # noqa

# Deep Learning (LSTM)
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM as KLSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------- UI setup --------------------------
st.set_page_config(page_title="Осол — Auto ML & Hotspot (RF+LSTM)", layout="wide")
st.title("С.Цолмон, А.Тамир нарын хар цэгийн судалгаа — RF + LSTM r4.2")

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

# ---------- Safe integer rounding ----------
def to_int_safe(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    rounded = np.rint(s.values.astype("float64"))
    tmp = np.where(np.isfinite(rounded), rounded, 0.0)
    ints = tmp.astype(np.int64)
    out = pd.Series(ints, index=series.index, copy=False).astype("Int64")
    mask = ~np.isfinite(s.values)
    if mask.any():
        out[mask] = pd.NA
    return out

# -------------------------- LSTMRegressor (sklearn-like) --------------------------
class LSTMRegressor:
    """
    Time-ordered X (2D) ба y(1D)-г авч, seq_len урттай цонхоор LSTM сургана.
    predict(X_test) үед fit үеийн train төгсгөлийн "сүүл" цонхоор гүүрлэж,
    тестийн мөр бүрт таамаг буцаана (output урт = len(X_test)).
    """
    def __init__(self,
                 seq_len=12, units=64, units2=32, dropout=0.1,
                 epochs=80, batch_size=32, loss="poisson",
                 patience=8, learning_rate=1e-3, verbose=0, random_state=42):
        self.seq_len = int(seq_len)
        self.units = int(units)
        self.units2 = int(units2)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.loss = str(loss)
        self.patience = int(patience)
        self.learning_rate = float(learning_rate)
        self.verbose = int(verbose)
        self.random_state = int(random_state)
        self.is_poisson_like = (self.loss.lower() == "poisson")

        self.model_ = None
        self.n_features_ = None
        self._last_window = None  # train X төгсгөлийн seq-1 мөр

    def get_params(self, deep=True):
        return {
            "seq_len": self.seq_len, "units": self.units, "units2": self.units2,
            "dropout": self.dropout, "epochs": self.epochs, "batch_size": self.batch_size,
            "loss": self.loss, "patience": self.patience,
            "learning_rate": self.learning_rate, "verbose": self.verbose,
            "random_state": self.random_state
        }
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        self.is_poisson_like = (self.loss.lower() == "poisson")
        return self

    def _make_sequences(self, X2d, y1d=None):
        S = self.seq_len
        n = len(X2d)
        if n < S:
            return np.empty((0, S, X2d.shape[1])), (np.empty((0,)) if y1d is not None else None)
        Xs, ys = [], []
        for i in range(S - 1, n):
            Xs.append(X2d[i - S + 1:i + 1, :])
            if y1d is not None:
                ys.append(y1d[i])
        Xs = np.asarray(Xs, dtype=np.float32)
        ys = (np.asarray(ys, dtype=np.float32) if y1d is not None else None)
        return Xs, ys

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.n_features_ = X.shape[1]
        tf.keras.utils.set_random_seed(self.random_state)

        Xs, ys = self._make_sequences(X, y)
        if len(Xs) == 0:
            self.model_ = Sequential([
                Dense(32, activation="relu", input_shape=(self.n_features_,)),
                Dense(1)
            ])
            self.model_.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                                loss=self.loss if self.loss in ["mse", "poisson"] else "mse")
            self.model_.fit(X, y, epochs=max(5, self.epochs//4), batch_size=self.batch_size,
                            verbose=self.verbose, shuffle=False)
            self._last_window = X[-(self.seq_len - 1):, :] if self.seq_len > 1 else None
            return self

        self.model_ = Sequential([
            KLSTM(self.units, input_shape=(self.seq_len, self.n_features_), return_sequences=False),
            Dropout(self.dropout),
            Dense(self.units2, activation="relu"),
            Dense(1)
        ])
        self.model_.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                            loss=(self.loss if self.loss in ["poisson", "mse"] else "poisson"))
        es = EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        self.model_.fit(Xs, ys, validation_split=0.1, shuffle=False,
                        epochs=self.epochs, batch_size=self.batch_size,
                        callbacks=[es], verbose=self.verbose)
        self._last_window = X[-(self.seq_len - 1):, :] if self.seq_len > 1 else None
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("LSTMRegressor is not fitted yet.")
        X = np.asarray(X, dtype=np.float32)
        if self.n_features_ is not None and X.shape[1] != self.n_features_:
            raise ValueError(f"LSTMRegressor: X has {X.shape[1]} features, but model was trained with {self.n_features_}.")
        n = len(X)
        preds = []

        if self._last_window is not None and self._last_window.shape[0] == (self.seq_len - 1):
            if self._last_window.shape[1] != X.shape[1]:
                raise ValueError(f"LSTMRegressor bridge window mismatch: last_window has {self._last_window.shape[1]} features, X has {X.shape[1]}.")
            ctx = np.vstack([self._last_window, X])
        else:
            ctx = X

        for t in range(n):
            start = t
            end = t + self.seq_len
            window = ctx[start:end, :]
            if window.shape[0] < self.seq_len:
                pad = np.zeros((self.seq_len - window.shape[0], X.shape[1]), dtype=np.float32)
                window = np.vstack([pad, window])
            yhat = float(self.model_.predict(window[None, :, :], verbose=0)[0, 0])
            preds.append(max(0.0, yhat))
        return np.asarray(preds, dtype=np.float32)

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
st.header("5. Ирээдүйн ослын таамаглал (RF + LSTM)")
agg_mode = st.sidebar.selectbox("Прогнозын агрегат", ["Сар", "Өдөр"], index=0)
st.caption("Leakage-гүй шинжүүд: feature engineering бүх сэргөлтийг өмнөх хугацаанаас shift/align хийсэн.")

def nonleaky(col: str) -> bool:
    s = str(col)
    if s == "Осол": return False
    if s.startswith("Зөрчил огноо жил "): return False
    if "Төрөл" in s: return False
    if s in {"Year", "Month", "Day"}: return False
    return True

pure_exog_pool = [c for c in (binary_cols + num_additional) if nonleaky(c)]
if len(pure_exog_pool) == 0:
    st.error("Leakage-гүй экзоген шинж олдсонгүй. One-hot/meta багануудыг шалгана уу.")
    st.stop()

# -------------------------- TA settings (өдөр горимд) --------------------------
ta_use: bool = False
ta_params: tuple | None = None
ta_show_chart: bool = False

top_k_exog = st.sidebar.slider("Экзоген квот (сонгох k)", 5, 40, 20, 1)
ta_force_min = st.sidebar.number_input("TA-г дор хаяж N багтаах (өдөр горимд)", 0, 15, 0, 1)

strategy = st.sidebar.radio("Прогнозын арга", ["Direct multi-step (RF, зөвлөмж)", "Recursive baseline"], index=0)
loss_pref = st.sidebar.radio("Loss/scaling", ["Poisson/Tweedie (зөвлөмж)", "log1p scale (optional)"], index=0)

with st.sidebar.expander("🧠 LSTM тохиргоо", expanded=False):
    default_seq = 12 if agg_mode == "Сар" else 30
    lstm_seq_len = st.number_input("Seq length (цонх)", 5, 180, default_seq, 1)
    lstm_units   = st.number_input("Hidden units", 16, 256, 64, 8)
    lstm_units2  = st.number_input("Dense units", 8, 256, 32, 8)
    lstm_epochs  = st.number_input("Epochs", 10, 300, 80, 10)
    lstm_bs      = st.number_input("Batch size", 8, 256, 32, 8)
    lstm_dropout = st.slider("Dropout", 0.0, 0.8, 0.1, 0.05)
    lstm_loss    = st.selectbox("Loss", ["poisson", "mse"], index=(0 if loss_pref.startswith("Poisson") else 1))

if agg_mode == "Өдөр":
    with st.sidebar.expander("📈 Техник шинжилгээ (өдөр)", expanded=False):
        ta_use = st.checkbox("TA индикаторуудыг шинжинд ашиглах (зөвхөн training/ex-post)", value=False)
        sma_short = st.number_input("SMA (богино, өдөр)", min_value=3, max_value=60, value=7, step=1)
        sma_long  = st.number_input("SMA (урт, өдөр)",    min_value=5, max_value=200, value=30, step=1)
        ema_short = st.number_input("EMA (богино, өдөр)", min_value=3, max_value=60, value=12, step=1)
        ema_long  = st.number_input("EMA (урт, өдөр)",    min_value=5, max_value=200, value=26, step=1)
        macd_sig  = st.number_input("MACD signal",        min_value=3, max_value=20, value=9, step=1)
        rsi_win   = st.number_input("RSI цонх",           min_value=5, max_value=60, value=14, step=1)
        bb_win    = st.number_input("Bollinger цонх",     min_value=5, max_value=60, value=20, step=1)
        bb_k      = st.number_input("Bollinger σ (k)",    min_value=1.0, max_value=4.0, value=2.0, step=0.5)
        roc_win   = st.number_input("ROC/MOM цонх",       min_value=2, max_value=60, value=7, step=1)
        ta_show_chart = st.checkbox("TA график харуулах", value=False)
        st.info("⚠️ Прогноз дээр TA-г forward simulate хийхгүй. (зөвхөн сургалт/тайлбар дээр ашиглана)")

    ta_params = (int(sma_short), int(sma_long), int(ema_short), int(ema_long),
                 int(rsi_win), int(macd_sig), int(bb_win), float(bb_k), int(roc_win))

# -------------------------- SERIES BUILD (cached) --------------------------
@st.cache_data(show_spinner=True)
def build_monthly_cached(df_in: pd.DataFrame, pure_exog_pool: list, n_lag: int):
    monthly_target = (
        df_in[df_in["Осол"] == 1]
        .groupby(["Year", "Month"]).agg(osol_count=("Осол", "sum")).reset_index()
    )
    monthly_target["date"] = pd.to_datetime(monthly_target[["Year","Month"]].assign(DAY=1))

    monthly_features = (
        df_in.groupby(["Year","Month"])[pure_exog_pool].sum().reset_index().sort_values(["Year","Month"])
    )
    for c in pure_exog_pool:
        monthly_features[c] = monthly_features[c].shift(1)

    grouped = (pd.merge(monthly_target, monthly_features, on=["Year","Month"], how="left")
               .sort_values(["Year","Month"]).reset_index(drop=True))
    grouped["m"] = grouped["Month"].astype(int)
    K = [1,2,3]
    for k in K:
        grouped[f"m_sin_{k}"] = np.sin(2*np.pi*k*grouped["m"]/12)
        grouped[f"m_cos_{k}"] = np.cos(2*np.pi*k*grouped["m"]/12)
    seasonal_cols = [f"m_sin_{k}" for k in K] + [f"m_cos_{k}" for k in K]

    lag_cols = [f"osol_lag_{i}" for i in range(1, n_lag+1)]
    for i in range(1, n_lag+1):
        grouped[f"osol_lag_{i}"] = grouped["osol_count"].shift(i)

    s = grouped["osol_count"].astype(float)
    grouped["LY_12"]      = s.shift(12)
    grouped["ROLL12_SUM"] = s.rolling(12, min_periods=3).sum().shift(1)

    ta_cols = []
    return grouped, lag_cols, seasonal_cols, ta_cols, ["LY_12", "ROLL12_SUM"]

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
    df_ta = pd.DataFrame({
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
    df_ta.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df_ta

@st.cache_data(show_spinner=True)
def build_daily_cached(df_in: pd.DataFrame, pure_exog_pool: list, n_lag: int,
                       ta_use: bool, ta_params: tuple | None):
    df_temp = df_in.copy()
    date_col = "Зөрчлийн огноо" if "Зөрчлийн огноо" in df_in.columns else "Зөрчил огноо"
    df_temp["date"] = df_temp[date_col].dt.floor("D")
    start, end = df_temp["date"].min().normalize(), df_temp["date"].max().normalize()
    all_days = pd.date_range(start, end, freq="D", name="date")

    daily_target = (
        df_temp.groupby("date")["Осол"].sum().reindex(all_days, fill_value=0)
        .rename("osол_count" if "осол_count" in df_temp.columns else "osol_count").rename_axis("date").reset_index()
    )
    daily_target.rename(columns={"осол_count":"osol_count"}, inplace=True)

    if pure_exog_pool:
        daily_features = (
            df_temp.groupby("date")[pure_exog_pool].sum().reindex(all_days, fill_value=0)
            .rename_axis("date").reset_index()
        )
        for c in pure_exog_pool:
            daily_features[c] = daily_features[c].shift(1)
    else:
        daily_features = pd.DataFrame({"date": all_days}).reset_index(drop=True)

    grouped = (pd.merge(daily_target, daily_features, on="date", how="left")
               .sort_values("date").reset_index(drop=True))

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

    seasonal_cols = (
        [f"w_sin_{k}" for k in Kw] + [f"w_cos_{k}" for k in Kw] +
        [f"y_sin_{k}" for k in Ky] + [f"y_cos_{k}"] for k in Ky
    )
    # fix list creation
    seasonal_cols = [f"w_sin_{k}" for k in Kw] + [f"w_cos_{k}" for k in Kw] + [f"y_sin_{k}" for k in Ky] + [f"y_cos_{k}" for k in Ky]

    lag_cols = [f"osol_lag_{i}" for i in range(1, n_lag+1)]
    for i in range(1, n_lag+1):
        grouped[f"osol_lag_{i}"] = grouped["osol_count"].shift(i)

    grouped["Year"] = grouped["date"].dt.year
    grouped["Month"] = grouped["date"].dt.month
    grouped["DOW"] = grouped["date"].dt.dayofweek

    s = grouped["osol_count"].astype(float)
    grouped["LY_365"]  = s.shift(365)
    grouped["LY_366"]  = s.shift(366)
    grouped["LY_MEAN"] = grouped[["LY_365", "LY_366"]].mean(axis=1)

    grouped["ROLL365_SUM"]  = s.rolling(365, min_periods=30).sum().shift(1)
    grouped["ROLL365_MEAN"] = s.rolling(365, min_periods=30).mean().shift(1)
    grouped["ROLL30_SUM"]   = s.rolling(30,  min_periods=7).sum().shift(1)

    doy_all = grouped["date"].dt.dayofyear
    doy_norm = np.where(doy_all == 366, 365, doy_all)
    grouped["DOY_NORM"] = doy_norm
    grouped = grouped.sort_values(["DOY_NORM", "Year"])
    grouped["LY_DOY_1"] = grouped.groupby("DOY_NORM")["osol_count"].shift(1)
    grouped["LY_DOY_2"] = grouped.groupby("DOY_NORM")["osol_count"].shift(2)
    grouped["LY_DOY_MEAN_2"] = grouped[["LY_DOY_1","LY_DOY_2"]].mean(axis=1)
    grouped = grouped.sort_values("date")

    ta_cols = []
    if ta_params is not None:
        ta_df = compute_ta_cached(grouped["osol_count"], ta_params)
        grouped = pd.concat([grouped, ta_df], axis=1)
        ta_cols = list(ta_df.columns) if ta_use else []

    ar_cols_all = ["LY_365","LY_366","LY_MEAN","LY_DOY_1","LY_DOY_2","LY_DOY_MEAN_2",
                   "ROLL365_SUM","ROLL365_MEAN","ROLL30_SUM"]

    return grouped, lag_cols, seasonal_cols, ta_cols, ar_cols_all

# sliders (rebuild triggers)
if agg_mode == "Сар":
    n_lag = st.sidebar.slider("Сарын лаг цонх (n_lag) — зөвхөн recursive baseline-д", 6, 18, 12, 1, key="lag_m")
    grouped, lag_cols, seasonal_cols, ta_cols_all, ar_cols_all = build_monthly_cached(df, pure_exog_pool, n_lag)
    freq_code = "MS"
    seg_key = "Month"
else:
    n_lag = st.sidebar.slider("Өдрийн лаг цонх (n_lag) — зөвхөн recursive baseline-д", 7, 120, 30, 1, key="lag_d")
    grouped, lag_cols, seasonal_cols, ta_cols_all, ar_cols_all = build_daily_cached(df, pure_exog_pool, n_lag, ta_use, ta_params)
    freq_code = "D"
    seg_key = "DOW"

grouped = grouped.dropna(subset=["osol_count"]).reset_index(drop=True)
if grouped.empty or len(grouped) < max(10, n_lag + 5):
    st.warning(f"Сургалт хийхэд хангалттай өгөгдөл алга (lag={n_lag}, mode={agg_mode}).")
    st.stop()

grouped["T"]  = np.arange(len(grouped))
grouped["T2"] = grouped["T"] ** 2

# -------------------------- Feature pools --------------------------
seasonal_pool = list(dict.fromkeys(seasonal_cols + ["T","T2"]))
ta_pool       = ta_cols_all
ar_pool_safe_for_direct = [c for c in ar_cols_all if c.startswith("LY_")]
ar_pool_risky = [c for c in ar_cols_all if c.startswith("ROLL")]
lag_pool      = lag_cols

split_ratio = st.sidebar.slider("Train ratio", 0.5, 0.9, 0.8, 0.05)
train_size = int(len(grouped) * split_ratio)

# -------------------------- Feature selection (экзоген квот) --------------------------
@st.cache_data(show_spinner=True)
def select_top_pure_exog(grouped: pd.DataFrame, pure_exog_pool: List[str], seasonal_pool: List[str],
                         split_ratio: float, top_k: int = 14):
    if top_k <= 0 or not pure_exog_pool:
        return []
    X_all = grouped[pure_exog_pool + seasonal_pool].fillna(0.0).values
    y_all = grouped["osol_count"].values
    cut = int(len(X_all)*split_ratio)
    X_tr, y_tr = X_all[:cut], y_all[:cut]
    try:
        rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        imp = pd.Series(rf.feature_importances_, index=pure_exog_pool + seasonal_pool)
        exog_top = imp.loc[pure_exog_pool].sort_values(ascending=False).head(min(top_k, len(pure_exog_pool))).index.tolist()
    except Exception:
        exog_top = pure_exog_pool[:min(top_k, len(pure_exog_pool))]
    return exog_top

exog_top_init = select_top_pure_exog(grouped, pure_exog_pool, seasonal_pool, split_ratio, top_k=top_k_exog)
st.caption("Train дээр тодорсон топ экзоген (квотоор):")
st.write(exog_top_init)

# TA квот
if ta_use and ta_force_min > 0:
    must_have_ta = [c for c in ta_pool if c not in exog_top_init][:ta_force_min]
else:
    must_have_ta = []

# FINAL feature sets by strategy
if strategy.startswith("Direct"):
    feature_cols_direct = seasonal_pool + ar_pool_safe_for_direct + exog_top_init + must_have_ta
    feature_cols_direct = list(dict.fromkeys(feature_cols_direct))
    feature_cols_recursive = lag_pool + seasonal_pool + ar_pool_safe_for_direct + exog_top_init + must_have_ta + ar_pool_risky
else:
    feature_cols_recursive = lag_pool + seasonal_pool + ar_pool_safe_for_direct + exog_top_init + must_have_ta + ar_pool_risky
    feature_cols_recursive = list(dict.fromkeys(feature_cols_recursive))
    feature_cols_direct = seasonal_pool + ar_pool_safe_for_direct + exog_top_init + must_have_ta
    feature_cols_direct = list(dict.fromkeys(feature_cols_direct))

identicals = [c for c in (set(feature_cols_recursive) | set(feature_cols_direct))
              if np.allclose(grouped[c].values, grouped["osol_count"].values, equal_nan=False)]
if identicals:
    st.error(f"IDENTICAL leakage илэрлээ: {identicals}")
    st.stop()

# -------------------------- Train/test split & scaling --------------------------
def make_xy(cols: List[str]):
    X = grouped[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = grouped["osol_count"].values.astype(float)
    return X, y

X_rec, y_orig = make_xy(feature_cols_recursive)
X_dir, _      = make_xy(feature_cols_direct)

cut = train_size
X_train_rec, X_test_rec = X_rec[:cut], X_rec[cut:]
X_train_dir, X_test_dir = X_dir[:cut], X_dir[cut:]
y_train_orig, y_test_orig = y_orig[:cut], y_orig[cut:]

scaler_X_rec = MinMaxScaler().fit(X_train_rec)
X_train_rec_s = scaler_X_rec.transform(X_train_rec)
X_test_rec_s  = scaler_X_rec.transform(X_test_rec)

scaler_X_dir = MinMaxScaler().fit(X_train_dir)
X_train_dir_s = scaler_X_dir.transform(X_train_dir)
X_test_dir_s  = scaler_X_dir.transform(X_test_dir)

use_log_scale = (loss_pref == "log1p scale (optional)")
if use_log_scale:
    scaler_y_log = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=False)
    y_train_log = scaler_y_log.fit_transform(y_train_orig.reshape(-1,1)).ravel()
    y_test_log  = scaler_y_log.transform(y_test_orig.reshape(-1,1)).ravel()

# >>> base contexts saved BEFORE any refit
feature_cols_recursive_base = list(feature_cols_recursive)
scaler_X_rec_base = scaler_X_rec
X_test_rec_s_base = X_test_rec_s

# -------------------------- Poisson-like шалгах --------------------------
def is_poisson_like(name: str, model) -> bool:
    if hasattr(model, "is_poisson_like") and bool(getattr(model, "is_poisson_like")):
        return True
    name_l = name.lower()
    return ("poisson" in name_l) or (
        isinstance(model, HistGradientBoostingRegressor) and getattr(model, "loss", "") == "poisson"
    )

# -------------------------- Model zoo: зөвхөн RF + LSTM --------------------------
MODEL_LIST_BASE = [
    ("RandomForest", RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)),
    ("LSTM", LSTMRegressor(
        seq_len=int(lstm_seq_len), units=int(lstm_units), units2=int(lstm_units2),
        dropout=float(lstm_dropout), epochs=int(lstm_epochs), batch_size=int(lstm_bs),
        loss=str(lstm_loss), patience=8, learning_rate=1e-3, verbose=0, random_state=RANDOM_STATE
    )),
]

# -------------------------- Train all (cached) --------------------------
@st.cache_resource(show_spinner=True)
def train_all_models_cached(
    X_train_s_rec, y_train_orig, y_train_log, X_test_s_rec, y_test_orig, y_test_log,
    _model_list, use_log_scale: bool, cache_signature: str
):
    results, y_preds_test_real, fitted = [], {}, {}
    for name, model in _model_list:
        try:
            y_tr = y_train_orig
            y_te_for_r2 = y_test_orig
            use_log_for_this = False

            if use_log_scale and not is_poisson_like(name, model):
                y_tr = y_train_log
                y_te_for_r2 = y_test_log
                use_log_for_this = True

            model_fit = clone(model)
            model_fit.fit(X_train_s_rec, y_tr)
            fitted[name] = (model_fit, use_log_for_this)

            y_pred_scale = np.asarray(model_fit.predict(X_test_s_rec)).reshape(-1)
            if use_log_for_this:
                y_pred = np.expm1(y_pred_scale)
                y_pred = np.clip(y_pred, 0, None)
                r2_s = r2_score(y_te_for_r2, y_pred_scale)
            else:
                y_pred = np.clip(y_pred_scale, 0, None)
                r2_s = r2_score(y_te_for_r2, y_pred_scale)

            results.append({
                "Model": name,
                "MAE (real)": mean_absolute_error(y_test_orig, y_pred),
                "RMSE (real)": float(np.sqrt(mean_squared_error(y_test_orig, y_pred))),
                "R2 (internal)": float(r2_s),
            })
            y_preds_test_real[name] = y_pred
        except Exception as e:
            results.append({"Model": name, "MAE (real)": np.nan, "RMSE (real)": np.nan,
                            "R2 (internal)": np.nan, "Error": str(e)})
    results_df = pd.DataFrame(results).sort_values("RMSE (real)", na_position="last")
    return fitted, results_df, y_preds_test_real

sig_parts = {
    "mode": agg_mode, "n_lag": n_lag, "split": round(float(split_ratio), 4),
    "cols_rec": tuple(feature_cols_recursive), "loss_pref": loss_pref,
    "ver": "r4.2-RF+LSTM", "nrows": int(len(grouped))
}
cache_signature = json.dumps(sig_parts, sort_keys=True)

with st.spinner("ML моделийг сургаж байна…"):
    fitted_models, results_df, test_preds_map = train_all_models_cached(
        X_train_rec_s, y_train_orig, (y_train_log if use_log_scale else None),
        X_test_rec_s,  y_test_orig, (y_test_log if use_log_scale else None),
        MODEL_LIST_BASE, use_log_scale, cache_signature
    )

st.success("Моделүүд бэлэн!")
st.dataframe(results_df.style.format({"MAE (real)": "{:.3f}", "RMSE (real)": "{:.3f}", "R2 (internal)": "{:.4f}"}),
             use_container_width=True)

try:
    best_model_name = results_df.loc[results_df["RMSE (real)"].idxmin(), "Model"]
except Exception:
    best_model_name = next(iter(fitted_models.keys()))
best_model, best_uses_log = fitted_models[best_model_name]

# -------------------------- Permutation Importance + optional refit --------------------------
with st.expander("🔍 Permutation importance (test) + refit", expanded=False):
    scoring_opt = st.selectbox("Scoring (test дээр)", 
                               ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"], index=1)
    n_repeats = st.slider("n_repeats", 5, 50, 20, 1)
    est = best_model

    # --- Pick the right test matrix for THIS estimator ---
    candidates = [(X_test_rec_s_base, feature_cols_recursive_base)]
    if "refit_ctx" in st.session_state:
        rc = st.session_state["refit_ctx"]
        candidates.append((rc["X_test_s"], rc["feature_cols"]))

    need_nfeat = getattr(est, "n_features_in_", getattr(est, "n_features_", None))
    X_te_raw, feat_names = None, None
    for Xcand, names in candidates:
        if need_nfeat is None or Xcand.shape[1] == need_nfeat:
            X_te_raw, feat_names = Xcand, names
            break

    if X_te_raw is None:
        st.error(
            f"Permutation importance: модель {need_nfeat} шинж хүлээж байна, "
            f"боловсруулсан тест матрицуудын хэмжээс {[X.shape[1] for X,_ in candidates]}."
        )
        st.stop()

    X_te_safe = np.nan_to_num(X_te_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_te_safe = (y_test_log if (best_uses_log and use_log_scale) else y_test_orig)

    pi = permutation_importance(
        est, X_te_safe, y_te_safe,
        n_repeats=n_repeats, random_state=RANDOM_STATE,
        scoring=scoring_opt, n_jobs=-1
    )

    imp_df = (pd.DataFrame({
                "feature": feat_names,
                "importance": pi.importances_mean,
                "std": pi.importances_std,
             })
             .sort_values("importance", ascending=False)
             .reset_index(drop=True))

    st.subheader(f"PI — {best_model_name}")
    st.dataframe(imp_df.head(50), use_container_width=True)

    k_refit = st.slider("PI-д суурилж дахин сургахдаа авах шинжийн тоо (экзоген квотоос гадна)",
                        10, min(80, len(feat_names)), min(30, len(feat_names)), 1)
    do_refit = st.checkbox("Top-PI шинжүүдээр best моделийг дахин сургах", value=False)
    if do_refit:
        top_feats = imp_df["feature"].head(k_refit).tolist()
        for f in exog_top_init:
            if f not in top_feats:
                top_feats.append(f)
        for f in seasonal_pool + ["T","T2"]:
            if f not in top_feats:
                top_feats.append(f)
        for f in ar_pool_safe_for_direct:
            if f not in top_feats:
                top_feats.append(f)
        top_feats = list(dict.fromkeys(top_feats))

        Xr = grouped[top_feats].fillna(0).values
        Xr_tr, Xr_te = Xr[:cut], Xr[cut:]
        scaler_X_r = MinMaxScaler().fit(Xr_tr)
        Xr_tr_s, Xr_te_s = scaler_X_r.transform(Xr_tr), scaler_X_r.transform(Xr_te)

        # ✅ refit target: train-set, log if needed
        y_refit = (y_train_log if (best_uses_log and use_log_scale) else y_train_orig)
        est_refit = clone(best_model).fit(Xr_tr_s, y_refit)

        refit_name = f"{best_model_name}_RefitPI"
        fitted_models[refit_name] = (est_refit, best_uses_log)

        y_pred_scale = np.asarray(est_refit.predict(Xr_te_s)).reshape(-1)
        y_pred_real = np.expm1(y_pred_scale) if (best_uses_log and use_log_scale) else y_pred_scale
        y_pred_real = np.clip(y_pred_real, 0, None)
        st.info(f"Refit MAE={mean_absolute_error(y_test_orig, y_pred_real):.3f} / RMSE={np.sqrt(mean_squared_error(y_test_orig, y_pred_real)):.3f}")

        st.session_state["refit_ctx"] = {
            "name": refit_name,
            "scaler": scaler_X_r,
            "X_test_s": Xr_te_s,
            "feature_cols": top_feats
        }

# -------------------------- Seasonal bias correction (residuals by segment) --------------------------
unit_label = "сар" if agg_mode == "Сар" else "өдөр"
with st.sidebar.expander("Прогнозын калибровк/холимог", expanded=True):
    W_tail  = st.slider(f"Калибровкын test tail W ({unit_label})", 2 if agg_mode=="Сар" else 14,
                        12 if agg_mode=="Сар" else 60, 2 if agg_mode=="Сар" else 14, 1)
    bias_decay = st.slider("Bias decay (0..1)", 0.0, 1.0, 0.85, 0.05)
    alpha_cap  = st.slider("sNaive холих дээд жин α_max (adaptive)", 0.0, 0.8, 0.4, 0.05)
    use_soft_clip = st.checkbox("Soft-clip идэвхжүүлэх", value=False)

test_dates = grouped["date"].iloc[cut:].values
y_test_real = y_test_orig
y_pred_best = test_preds_map[best_model_name]
resid = y_test_real - y_pred_best
tail = min(len(resid), int(W_tail))
resid_tail = resid[-tail:] if tail > 0 else resid
dates_tail = grouped.iloc[cut:].iloc[-tail:][["date", seg_key]]

if tail > 0:
    df_res = pd.DataFrame({"seg": dates_tail[seg_key].values, "resid": resid_tail})
    bias_by_seg = df_res.groupby("seg")["resid"].mean().to_dict()
else:
    bias_by_seg = {}

resid_std = float(np.std(resid_tail)) if tail > 1 else 0.0
alpha_adapt = min(alpha_cap, (0.1 + 0.9 * (resid_std / (np.mean(y_test_real[-tail:]) + 1e-6))) if tail > 0 else 0.1)
alpha_snv = round(alpha_adapt, 3)

# -------------------------- Forecast helpers --------------------------
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

def seasonal_segment_key(dt: pd.Timestamp) -> int:
    return dt.month if agg_mode=="Сар" else dt.dayofweek

# Recursive baseline forecast
def forecast_recursive(model, feature_cols, scaler_X, use_log_for_model,
                       steps, last_date, mode, hist_series,
                       bias_decay: float, alpha_snv: float, m_snaive: int,
                       bias_by_seg: Dict, use_soft_clip: bool):
    preds = []
    last_raw_row = grouped[feature_cols].iloc[cut-1 if cut>0 else -1].values
    lag_count = len([c for c in feature_cols if c.startswith("osol_lag_")])
    lag_vals = last_raw_row[:lag_count].astype(float).copy() if lag_count>0 else np.array([], float)

    other_idx = {name: (lag_count + i) for i, name in enumerate(feature_cols[lag_count:])}
    other_vals0 = {n: float(last_raw_row[other_idx[n]]) for n in feature_cols[lag_count:] if n in other_idx}

    base_T = float(other_vals0.get("T", grouped["T"].iloc[cut-1] if cut>0 else grouped["T"].iloc[-1]))

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
                exog_vector.append(tt*tt)
            elif name.startswith("LY_"):
                def safe_tail(arr, k): return float(arr[-k]) if len(arr) >= k else np.nan
                if agg_mode=="Өдөр":
                    if name == "LY_365": val = safe_tail(series,365)
                    elif name == "LY_366": val = safe_tail(series,366)
                    elif name == "LY_MEAN": val = np.nanmean([safe_tail(series,365), safe_tail(series,366)])
                    elif name == "LY_DOY_1": val = np.nanmean([safe_tail(series,365), safe_tail(series,366)])
                    elif name == "LY_DOY_2": val = np.nanmean([safe_tail(series,730), safe_tail(series,731), safe_tail(series,732)])
                    elif name == "LY_DOY_MEAN_2": val = np.nanmean([
                        np.nanmean([safe_tail(series,365), safe_tail(series,366)]),
                        np.nanmean([safe_tail(series,730), safe_tail(series,731), safe_tail(series,732)])
                    ])
                    else: val = np.nan
                else:
                    if name == "LY_12": val = float(series[-12]) if len(series)>=12 else np.nan
                    else: val = np.nan
                exog_vector.append(val if np.isfinite(val) else 0.0)
            elif name.startswith("ROLL"):
                if name == "ROLL365_SUM":  val = float(np.nansum(series[-365:])) if len(series) else 0.0
                elif name == "ROLL365_MEAN": val = float(np.nanmean(series[-365:])) if len(series) else 0.0
                elif name == "ROLL30_SUM":   val = float(np.nansum(series[-30:]))  if len(series) else 0.0
                else: val = 0.0
                exog_vector.append(val)
            else:
                exog_vector.append(other_vals0.get(name, 0.0))

        seq_raw = np.concatenate([lag_vals, np.array(exog_vector, float)]).reshape(1, -1) if lag_count>0 else np.array([exog_vector], float)
        seq_raw = np.nan_to_num(seq_raw, nan=0.0, posinf=0.0, neginf=0.0)
        seq_scaled = scaler_X.transform(seq_raw)

        pred_scale = float(np.asarray(model.predict(seq_scaled)).ravel()[0])
        p = float(np.expm1(pred_scale) if use_log_for_model else pred_scale)
        p = max(p, 0.0)

        p_snaive = float(series[-m_snaive]) if len(series) >= m_snaive else (series[-1] if len(series) else p)

        seg = seasonal_segment_key(cur_date)
        bias_seg = float(bias_by_seg.get(seg, 0.0))
        bias_t = (resid_tail.mean() if len(resid_tail)>0 else 0.0) * (bias_decay ** (t-1))
        p = (1 - alpha_snv) * p + alpha_snv * p_snaive + bias_t + bias_seg

        if use_soft_clip and len(series) >= 20:
            lo = float(np.nanquantile(series, 0.01)); hi = float(np.nanquantile(series, 0.99) * 1.2)
            p = float(np.clip(p, max(0.0, lo*0.5), hi))

        preds.append(p)
        if lag_count>0:
            lag_vals = np.roll(lag_vals, 1); lag_vals[0] = p
        series.append(p)
    return np.array(preds)

# Direct multi-step (RF)
@st.cache_resource(show_spinner=False)
def train_direct_models(
    X_full_s: np.ndarray,
    y_full: np.ndarray,
    dates: pd.Series,
    feature_cols_direct: List[str],
    horizons: List[int],
    _base_model_name: str,
    _base_model
):
    models = {}
    cut = int(len(X_full_s) * 0.8)
    for h in horizons:
        if len(X_full_s) <= h:
            continue
        X_h = X_full_s[h:]
        y_h = y_full[h:]
        est = clone(_base_model)
        est.fit(X_h[:cut - h], y_h[:cut - h])
        models[h] = est
    return models

# -------------------------- Build test and future tables --------------------------
test_dates = grouped["date"].iloc[cut:].values
test_preds_df = pd.DataFrame({"date": test_dates, "real": y_test_orig})

def _expected_n_features(mdl):
    return getattr(mdl, "n_features_in_", getattr(mdl, "n_features_", None))

X_test_map = {X_test_rec_s_base.shape[1]: X_test_rec_s_base}
if "refit_ctx" in st.session_state:
    X_test_map[ st.session_state["refit_ctx"]["X_test_s"].shape[1] ] = st.session_state["refit_ctx"]["X_test_s"]

for name, (mdl, uses_log) in fitted_models.items():
    nfeat = _expected_n_features(mdl)
    X_te = X_test_map.get(nfeat, None)
    if X_te is None:
        st.warning(f"{name}: тестийн шинжийн хэмжээ таарахгүй байна (expect {nfeat}). Энэ моделийг алгаслаа.")
        continue
    y_hat_scale = np.asarray(mdl.predict(X_te)).reshape(-1)
    y_hat = np.expm1(y_hat_scale) if (uses_log and use_log_scale) else y_hat_scale
    y_hat = np.clip(y_hat, 0, None)
    test_preds_df[name] = y_hat

for col in list(test_preds_df.columns)[2:]:
    test_preds_df[col + "_int"] = to_int_safe(test_preds_df[col])

last_known_date = grouped["date"].iloc[-1]
future_steps = 12 if agg_mode == "Сар" else 365
future_dates = pd.date_range(start=step_next_date(last_known_date, agg_mode), periods=future_steps, freq=freq_code)

horizons = ([1,3,6,12] if agg_mode=="Сар" else [7,14,30,90,180,365])
base_direct_model = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
X_full_dir_s = scaler_X_dir.transform(X_dir)
direct_models = train_direct_models(X_full_dir_s, y_orig, grouped["date"], feature_cols_direct, horizons,
                                    "RF-Direct", base_direct_model)

def forecast_direct(direct_models: Dict[int,object], scaler_X_dir, feature_cols_direct, steps, start_date):
    preds = []
    hist_series = grouped["osol_count"].values.tolist()
    last_T = grouped["T"].iloc[-1]
    rows = []
    for i, d in enumerate(future_dates, start=1):
        row = {}
        row.update(seasonal_values_for_date(d, agg_mode))
        tt = last_T + i
        row["T"] = tt; row["T2"] = tt*tt
        def safe_tail(arr, k): return float(arr[-k]) if len(arr) >= k else 0.0
        if agg_mode=="Өдөр":
            row["LY_365"] = safe_tail(hist_series,365); row["LY_366"] = safe_tail(hist_series,366)
            row["LY_MEAN"] = float(np.nanmean([row["LY_365"], row["LY_366"]]))
            row["LY_DOY_1"] = row["LY_MEAN"]
            row["LY_DOY_2"] = float(np.nanmean([safe_tail(hist_series,730), safe_tail(hist_series,731), safe_tail(hist_series,732)]))
            row["LY_DOY_MEAN_2"] = float(np.nanmean([row["LY_DOY_1"], row["LY_DOY_2"]]))
        else:
            row["LY_12"] = safe_tail(hist_series,12)
        for c in exog_top_init + [c for c in feature_cols_direct if c.startswith("TA_")]:
            row[c] = 0.0
        rows.append(row)
    F = pd.DataFrame(rows)
    for c in feature_cols_direct:
        if c not in F.columns:
            F[c] = 0.0
    F = F[feature_cols_direct].fillna(0.0)
    Xf = scaler_X_dir.transform(F.values)

    for i, d in enumerate(future_dates, start=1):
        h = min(horizons, key=lambda hh: abs(hh - i)) if len(direct_models)>0 else None
        if h in direct_models:
            mdl = direct_models[h]
            yhat = float(np.clip(mdl.predict(Xf[i-1:i])[0], 0, None))
        else:
            yhat = float(np.clip(best_model.predict(scaler_X_rec.transform(grouped[feature_cols_recursive].iloc[[-1]].values))[0], 0, None))
        seg = seasonal_segment_key(d)
        bias_seg = float(bias_by_seg.get(seg, 0.0))
        m_snaive = 12 if agg_mode=="Сар" else 7
        p_snaive = grouped["osol_count"].iloc[-m_snaive] if len(grouped)>=m_snaive else grouped["osol_count"].iloc[-1]
        p = (1 - alpha_snv) * yhat + alpha_snv * p_snaive + bias_seg
        preds.append(max(0.0, p))
    return np.array(preds)

future_preds_df = pd.DataFrame({"date": future_dates})
hist_series = grouped["osol_count"].values

for name, (model, uses_log) in fitted_models.items():
    feats = feature_cols_recursive_base
    scaler_for_model = scaler_X_rec_base
    if "refit_ctx" in st.session_state and name == st.session_state["refit_ctx"]["name"]:
        feats = st.session_state["refit_ctx"]["feature_cols"]
        scaler_for_model = st.session_state["refit_ctx"]["scaler"]

    m_snaive = 12 if agg_mode=="Сар" else 7
    y_future = forecast_recursive(
        model, feats, scaler_for_model, (uses_log and use_log_scale),
        future_steps, grouped["date"].iloc[-1], agg_mode, hist_series,
        bias_decay=bias_decay, alpha_snv=alpha_snv, m_snaive=m_snaive,
        bias_by_seg=bias_by_seg, use_soft_clip=use_soft_clip
    )
    future_preds_df[f"{name}__rec"] = y_future

future_preds_df["Direct__RF"] = forecast_direct(direct_models, scaler_X_dir, feature_cols_direct,
                                                future_steps, grouped["date"].iloc[-1])

for col in list(future_preds_df.columns)[1:]:
    future_preds_df[col] = np.clip(future_preds_df[col], 0, None)
    future_preds_df[col + "_int"] = to_int_safe(future_preds_df[col])

with pd.ExcelWriter("model_predictions.xlsx", engine="xlsxwriter") as writer:
    test_preds_df.to_excel(writer, index=False, sheet_name="Test_Predictions")
    future_preds_df.to_excel(writer, index=False, sheet_name="Future_Predictions")
with open("model_predictions.xlsx", "rb") as f:
    st.download_button("Test/Forecast бүх моделийн таамаг (Excel)", data=f,
                       file_name="model_predictions.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.subheader("Test датан дээрх бодит/таамаг (толгой 10000 мөр)")
st.dataframe(test_preds_df.head(10000), use_container_width=True)

st.subheader(("Ирээдүйн 12 САР" if agg_mode == "Сар" else "Ирээдүйн 365 ӨДӨР") + " — прогноз (Direct + Recursive)")
st.dataframe(future_preds_df, use_container_width=True)

# -------------------------- Graph UI --------------------------
model_options = [best_model_name] + [m for m in fitted_models.keys() if m != best_model_name]
selected_kind = st.selectbox("Графикт харуулах:", ["Direct__RF"] + [m+"__rec" for m in model_options], index=0)

if agg_mode == "Сар":
    h_map = {"≈1 сар": 1, "≈3 сар": 3, "≈6 сар": 6, "≈12 сар": 12}
else:
    h_map = {"7 хоног": 7, "14 хоног": 14, "30 хоног": 30, "90 хоног": 90, "180 хоног": 180, "365 хоног": 365}
selected_h = st.selectbox("Хоризонт:", list(h_map.keys()), index=2)
steps = h_map[selected_h]
view_df = future_preds_df[["date", selected_kind]].iloc[:steps].rename(columns={selected_kind: "forecast"})

if agg_mode == "Өдөр":
    smooth_win = st.sidebar.slider("Гөлгөршүүлэх цонх (өдөр)", 1, 21, 7, 1)
    if smooth_win > 1:
        view_df["forecast_smooth"] = view_df["forecast"].rolling(smooth_win, min_periods=1).mean()
        y_col = "forecast_smooth"
    else:
        y_col = "forecast"
else:
    y_col = "forecast"

fig = px.line(view_df, x="date", y=y_col, markers=True,
              title=f"{selected_kind} — {selected_h} ({'сар' if agg_mode=='Сар' else 'өдөр'}ийн алхам)")
st.plotly_chart(fig, use_container_width=True)

# -------------------------- Horizon metrics --------------------------
def horizon_scores(model, X_full_s, y_full, uses_log, scaler_y_log, horizons):
    out = {}
    for h in horizons:
        if len(X_full_s) < h + 1:
            out[h] = np.nan; continue
        X_te = X_full_s[-h:]
        yhat_s = np.asarray(model.predict(X_te)).ravel()
        yhat   = np.expm1(yhat_s) if (uses_log and scaler_y_log is not None) else yhat_s
        yhat = np.clip(yhat, 0, None)
        out[h] = {"MAE": mean_absolute_error(y_full[-h:], yhat),
                  "RMSE": float(np.sqrt(mean_squared_error(y_full[-h:], yhat)))}
    return out

if len(fitted_models) > 0:
    horizons_eval = [1, 3, 6, 12] if agg_mode == "Сар" else [7, 14, 30, 90]
    hs = horizon_scores(best_model, scaler_X_rec.transform(X_rec), y_orig, best_uses_log,
                        (scaler_y_log if use_log_scale else None), horizons_eval)
    st.write("Horizon metrics (recursive best):", hs)

# -------------------------- Baselines --------------------------
def naive_forecast(y): return np.roll(y, 1)[1:]
def snaive_forecast(y, m): return np.roll(y, m)[m:]

y_true_full = y_test_orig
y_hist = y_train_orig
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

vars_for_corr = ["Year"]
vars_for_corr += [c for c in df.columns if c.startswith("Зөрчил огноо жил ")][:10]
vars_for_corr += [c for c in (binary_cols + num_additional) if c in df.columns]
vars_for_corr = list(dict.fromkeys(vars_for_corr))

if len(vars_for_corr) > 1:
    Xx = df[vars_for_corr].fillna(0.0).values
    yy = pd.to_numeric(df["Осол"], errors="coerce").fillna(0).values
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

# -------------------------- 4. Категори хамаарал --------------------------
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

mu_m = (monthly[monthly["period"]=="before"].groupby("Month")["osol_count"].mean())
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
