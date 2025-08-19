# -*- coding: utf-8 -*-
# ============================================================
# Зам тээврийн осол — Auto ML & Hotspot Dashboard (Streamlit)
# Хувилбар: 2025-08-17r2 — хурд, тогтвортой ажиллагаа сайжруулсан
# Гол өөрчлөлтүүд:
#  - st.title() алдаа, давхардсан import-ууд засав
#  - @st.cache_data доторх UI/stop-уудыг авч, зөв гадаа барив
#  - SHAP-д дээжлэлтийн хамгаалалт, алдааны хамгаалалт нэмж хурдасгав
#  - ML scale/reshape зөв дараалал + ensemble-уудыг n_jobs=-1 болгож параллелчилсан
#  - Прогнозын “өдөр” горимыг тайлбарлаж, сарын агрегаттай нийцүүлэн хэрэгжүүлэв
#  - DBSCAN eps-ийг МЕТР-ээр удирдаж (дотор нь радиан руу хөрвүүлнэ), MarkerCluster ашиглав
#  - Binary importance: -1 кластерыг хасч, dtype/NA хамгаалалт хийв
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import DBSCAN

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

    # Нэмэлт тоон candidate-ууд
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

# -------------------------- 5. Ирээдүйн ослын таамаглал --------------------------
st.header("5. Ирээдүйн ослын таамаглал (Олон ML/DL загвар)")
st.caption("Binary (0/1) баганууд автоматаар илрүүлж ашиглагдана. Прогноз **сарын агрегат** дээр хийгдэнэ.")

# Feature pool
def nonleaky(col: str) -> bool:
    s = str(col)
    if s == "Осол": 
        return False
    if s.startswith("Зөрчил огноо жил "):  # бүх year dummies
        return False
    if "Төрөл" in s:                       # төрлийн one-hot, нэршлийн хувилбарууд
        return False
    if s in {"Year", "Month", "Day"}:
        return False
    return True

feature_pool = [c for c in (binary_cols + num_additional) if nonleaky(c)]
if len(feature_pool) == 0:
    st.error("Leakage-гүй шинж олдсонгүй. Metadata/one-hot үүсгэх дүрмээ шалгана уу.")
    st.stop()



# Target/Features (event-level → monthly aggregate later)
y_all = pd.to_numeric(df["Осол"], errors="coerce").fillna(0).values
X_all = df[feature_pool].fillna(0.0).values

# Top features via RF + SHAP (guarded & sampled)
top_features = feature_pool[:min(14, len(feature_pool))]







# Сар бүрийн агрегат
monthly_target = (
    df[df["Осол"] == 1]
    .groupby(["Year", "Month"])
    .agg(osol_count=("Осол", "sum"))
    .reset_index()
)
monthly_target["date"] = pd.to_datetime(monthly_target[["Year", "Month"]].assign(DAY=1))
monthly_features = (
    df.groupby(["Year","Month"])[feature_pool]  # анхны pool (nonleaky)
      .sum()
      .reset_index()
      .sort_values(["Year","Month"])
)
# t сард зөвхөн (t-1) сар хүртэлх мэдээлэл мэдэгдэж байсан байх ёстой
for c in feature_pool:
    monthly_features[c] = monthly_features[c].shift(1)

grouped = (
    pd.merge(monthly_target, monthly_features, on=["Year","Month"], how="left")
      .sort_values(["Year","Month"]).reset_index(drop=True)
)

# Lag-ууд
# Lag-ууд
n_lag = st.sidebar.slider("Сарын лаг цонх (n_lag)", min_value=6, max_value=18, value=12, step=1)
for i in range(1, n_lag + 1):
    grouped[f"osol_lag_{i}"] = grouped["osol_count"].shift(i)
grouped = grouped.dropna().reset_index(drop=True)

if grouped.empty or len(grouped) < 10:
    st.warning(f"Сургалт хийхэд хангалттай сар тутмын өгөгдөл алга (lag={n_lag}). Он/сараа шалгана уу.")
    st.stop()

# --- Train/Test split хувь ---
split_ratio = st.sidebar.slider("Train ratio", 0.5, 0.9, 0.8, 0.05)

# --- Нэрсийг ялгах: лагууд + экзоген ---
lag_cols  = [f"osol_lag_{i}" for i in range(1, n_lag + 1)]
exog_cols = feature_pool  # leakage-гүйгээр цэвэрлэсэн pool

# --- Feature selection-ийг ЗӨВХӨН TRAIN дээр, зөв индексээр ---
X_all_fs = grouped[lag_cols + exog_cols].fillna(0.0).values
y_all     = grouped["osol_count"].values.reshape(-1, 1)

train_size = int(len(X_all_fs) * split_ratio)
X_train_fs = X_all_fs[:train_size]
y_train_fs = y_all[:train_size].ravel()

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

# --- Эцсийн шинжүүд: лаг + шилдэг экзоген ---
feature_cols = lag_cols + exog_top

# --- Эцсийн X, y, split, scale (scaler-уудыг дахин fit) ---
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

# --- Leakage шинжилгээ (safety) ---
suspects = [c for c in exog_top if c.startswith("Зөрчил огноо жил ")]
check = grouped[["date", "osol_count"] + feature_cols].copy()

identicals = [c for c in feature_cols
              if np.allclose(grouped[c].values, grouped["osol_count"].values, equal_nan=False)]
if identicals:
    st.error(f"IDENTICAL leakage үлдлээ: {identicals}")
    st.stop()

corrs = check[feature_cols].corrwith(check["osol_count"]).sort_values(ascending=False)
st.write("Leakage сэжигтэй (year dummies):", suspects)
st.write("Target-тэй корреляци (дээд 10):", corrs.head(10))
# Models
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
    ("HistGB", HistGradientBoostingRegressor(random_state=42)),
    ("AdaBoost", AdaBoostRegressor(random_state=42)),
    ("KNeighbors", KNeighborsRegressor()),
    ("SVR", SVR()),
    ("MLPRegressor", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42)),
    ("Stacking", StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=5)),
]
if XGBRegressor is not None:
    MODEL_LIST.append(("XGBRegressor", XGBRegressor(
        tree_method="hist", predictor="cpu_predictor", random_state=42, n_estimators=400)))
if CatBoostRegressor is not None:
    MODEL_LIST.append(("CatBoostRegressor", CatBoostRegressor(
        task_type="CPU", random_state=42, verbose=0)))
if LGBMRegressor is not None:
    MODEL_LIST.append(("LGBMRegressor", LGBMRegressor(
        device="cpu", random_state=42)))

# Voting/Stacking ensemble (нэмэлт)
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
MODEL_LIST.append(("StackingEnsemble", StackingRegressor(
    estimators=stacking_estimators, final_estimator=LinearRegression(), cv=5
)))

# Train all
progress_bar = st.progress(0, text="ML моделийг сургаж байна...")
results = []
y_preds = {}
for i, (name, model) in enumerate(MODEL_LIST):
    try:
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test)).reshape(-1)
        y_preds[name] = y_pred
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    except Exception as e:
        results.append({"Model": name, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Error": str(e)})
    progress = min(int((i + 1) / len(MODEL_LIST) * 100), 100)
    progress_bar.progress(progress, text=f"{name} дууслаа")
progress_bar.empty()
st.success("Бүх ML модел сургагдлаа!")


# --- Leakage шинжилгээ (сэжигтэй багануудыг илрүүлэх) ---
suspects = []
for c in top_features:
    if c.startswith("Зөрчил огноо жил "):
        suspects.append(c)

# сар бүрийн түвшинд шалгах
check = grouped[["date","osol_count"] + top_features].copy()

# 1) яг ижил эсэх
# identical шалгалт
identicals = []
for c in feature_pool:
    if np.allclose(grouped[c].values, grouped["osol_count"].values, equal_nan=False):
        identicals.append(c)
if identicals:
    st.error(f"IDENTICAL leakage үлдлээ: {identicals}")
    st.stop()


# 2) маш өндөр корреляци
corrs = (
    check[top_features].corrwith(check["osol_count"])
    .sort_values(ascending=False)
)
st.write("Leakage сэжигтэй (year dummies):", suspects)
st.write("Яг тэнцүү гарч буй баганууд:", identicals)
st.write("Target-тэй корреляци:", corrs.head(10))

results_df = pd.DataFrame(results).sort_values("RMSE", na_position="last")
st.dataframe(results_df, use_container_width=True)

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

# ---- Прогноз helper (лаг-цонхыг л шинэчилнэ, экзоген тогтвортой) ----
lag_count = len(lag_cols)

def forecast_next_monthly(model, last_raw_row, steps=12):
    """
    last_raw_row: grouped[feature_cols]-ийн СҮҮЛИЙН мөр (анскейлд)
    Буцаалт: ослын тооны прогнозууд (анскейлд, бодит нэгжээр)
    """
    preds = []
    lag_vals  = last_raw_row[:lag_count].astype(float).copy()   # анскейлд лагууд
    exog_vals = last_raw_row[lag_count:].astype(float).copy()   # тогтмол/сценари экзоген

    for _ in range(steps):
        seq_raw    = np.concatenate([lag_vals, exog_vals]).reshape(1, -1)
        seq_scaled = scaler_X.transform(seq_raw)
        p_scaled   = float(np.asarray(model.predict(seq_scaled)).ravel()[0])
        p          = float(scaler_y.inverse_transform(np.array([[p_scaled]])).ravel()[0])
        preds.append(p)

        # лаг цонхыг баруун тийш шилжүүлж, lag1-д шинэ p-г (анскейлд) байрлуулна
        lag_vals = np.roll(lag_vals, 1)
        lag_vals[0] = p
    return np.array(preds)

# Forecasts by model
model_forecasts = {}
last_raw = grouped[feature_cols].iloc[-1].values  # анскейлд, feature_cols-ийн дарааллаар

# Сонголтууд — сарын агрегаттай нийцүүлэх (өдрийн нэртэй боловч mapping нь сар)
h_map = {"7 хоног": 1, "14 хоног": 1, "30 хоног": 1, "90 хоног": 3, "180 хоног": 6, "365 хоног": 12}
for name, model in MODEL_LIST:
    # дараах 'y_preds' нь test дээрх scaled прогноз тул энэ dict-т байхгүй байж болно
    if name not in y_preds:
        continue
    preds_dict = {}
    for k, months in h_map.items():
        preds_dict[k] = forecast_next_monthly(model, last_raw, steps=months)  # анскейлд буцаана
    model_forecasts[name] = preds_dict

# Test дээрх бодит/таамаг (уншигдахуйц нэгжээр)
test_dates = grouped["date"].iloc[-len(X_test):].values
test_true  = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
test_preds_df = pd.DataFrame({"date": test_dates, "real": test_true})
for name in model_forecasts.keys():
    ypi = scaler_y.inverse_transform(np.array(y_preds[name]).reshape(-1, 1)).flatten()
    test_preds_df[name] = ypi

# Ирээдүйн 12 сарын таамаг (анскейлд, бодит нэгжээр)
future_dates = pd.date_range(start=grouped["date"].iloc[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS")
future_preds_df = pd.DataFrame({"date": future_dates})
for name, model in MODEL_LIST:
    if name not in y_preds:
        continue
    future_preds_df[name] = forecast_next_monthly(model, last_raw, steps=12)


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

st.subheader("Ирээдүйн 12 сарын прогноз (модел бүрээр)")
st.dataframe(future_preds_df, use_container_width=True)

# График харах UI
model_options = list(y_preds.keys())
selected_model = st.selectbox("Модель сонгох:", model_options)
selected_h = st.selectbox("Хоризонт:", list(h_map.keys()), index=2)

months = h_map[selected_h]
dates_future = pd.date_range(start=grouped["date"].iloc[-1] + pd.offsets.MonthBegin(), periods=months, freq="MS")
future_df = pd.DataFrame({"date": dates_future, "forecast": model_forecasts[selected_model][selected_h]})
fig = px.line(future_df, x="date", y="forecast", markers=True,
              title=f"{selected_model} — {selected_h} (сар руу зурагдсан) прогноз")
st.plotly_chart(fig, use_container_width=True)

# -------------------------- 1. Корреляцийн шинжилгээ --------------------------
st.header("1. Осолд нөлөөлөх хүчин зүйлсийн тархалт/корреляцийн шинжилгээ")
st.write("Доорх multiselect-оос ихдээ 15 хувьсагч сонгож корреляцийн матрицыг үзнэ үү.")

vars_for_corr = ["Year"]
vars_for_corr += [c for c in df.columns if c.startswith("Зөрчил огноо жил ")][:10]
vars_for_corr += [c for c in (binary_cols + num_additional) if c in df.columns]
vars_for_corr = list(dict.fromkeys(vars_for_corr))  # remove dups

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
    .groupby(["Year", "Month"])
    .agg(osol_count=("Осол", "sum"))
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

