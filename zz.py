# -*- coding: utf-8 -*-
# ============================================================
# Зам тээврийн осол — Auto ML & Hotspot Dashboard (Streamlit)
# Хувилбар: 2025-08-17b — Leakage-free scaling, 200-row quick sample, bias-corrected Cramér's V
# Тайлбар:
#  - Хавсаргасан Excel ("кодлогдсон - Copy.xlsx")-тай шууд зохицно.
#  - Binary (0/1) бүх баганыг автоматаар илрүүлж, модел/корреляц/хотспотод ашиглана.
#  - Координат баганууд (Өргөрөг/Уртраг эсвэл lat/lon) байвал газрын зураг зурна.
#  - Олон ML модел сургалт, метрик/таамаглалыг Excel болгон татах боломжтой.
#  - "Осол" багана байхгүй тохиолдолд "Төрөл"-өөс (Гэмт хэрэг/Зөрчлийн хэрэг) зорилтыг үүсгэнэ.
#  - Шинэчлэлтүүд: (i) scaler-ийг train-д л fit хийж data leakage арилгав, (ii) 200 мөрийн хурдан ажиллуулах сонголт, (iii) Cramér's V-ийн bias-corrected хувилбар нэмэв.
# Гүйцэтгэх: streamlit run osol_auto_streamlit.py
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

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import DBSCAN

# Гуравдагч gradient boosting сангууд (optional)
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None  # pragma: no cover
try:
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:
    LGBMRegressor = None  # pragma: no cover
try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:
    CatBoostRegressor = None  # pragma: no cover

from scipy.stats import chi2_contingency
import folium
from streamlit_folium import st_folium

# -------------------------- UI setup --------------------------
st.set_page_config(page_title="Осол — Auto ML & Hotspot (auto-binary)", layout="wide")
st.title("Зам тээврийн ослын анализ — Auto (Binary autodetect)")

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

def clean_road_width(width):
    """'Авто зам - Зорчих хэсгийн өргөн' талбарын текстэн утгуудыг тоон болгох."""
    if pd.isna(width):
        return np.nan
    if isinstance(width, (int, float)):
        return float(width)
    if isinstance(width, str):
        w = (
            width.replace("м", "")
            .replace("-ээс дээш", "")
            .replace("хүртэл", "")
            .replace(",", ".")
            .strip()
        )
        if "-" in w:
            try:
                low, high = map(float, w.split("-"))
                return (low + high) / 2
            except Exception:
                return np.nan
        try:
            return float(w)
        except Exception:
            return np.nan
    return np.nan

def plot_correlation_matrix(df, title, columns):
    n_unique = df[columns].nunique()
    if all((n_unique == 2) | (n_unique == 1)):
        st.warning("Сонгосон баганууд нь one-hot (0/1) хэлбэртэй байна. Ийм тохиолдолд корреляци -1~1 туйлруугаа хэлбийж харагдаж болно.")
    df_encoded = df[columns].copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
    corr_matrix = df_encoded.corr(numeric_only=True)
    corr_matrix = corr_matrix.iloc[::-1]
    fig, ax = plt.subplots(figsize=(max(8, 1.5*len(columns)), max(6, 1.2*len(columns))))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax, fmt=".3f")
    plt.title(title)
    plt.tight_layout()
    return fig

# Bias-corrected Cramér's V (Bergsma, 2013)

def cramers_v_bias_corrected(table: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(table)
    n = table.values.sum()
    phi2 = chi2 / max(n, 1)
    r, k = table.shape
    phi2_corr = max(0, phi2 - (k-1)*(r-1)/(max(n-1, 1)))
    r_corr = r - (r-1)**2 / max(n-1, 1)
    k_corr = k - (k-1)**2 / max(n-1, 1)
    denom = max(min(k_corr-1, r_corr-1), 1e-12)
    return float(np.sqrt(phi2_corr / denom))

# -------------------------- Өгөгдөл ачаалалт --------------------------

@st.cache_data(show_spinner=True)
def load_data(default_path: str = "кодлогдсон.xlsx"):
    """
    - Sidebar дээрээс .xlsx файлаар upload хийж болно.
    - Хэрэв оруулаагүй бол default_path-ыг уншина.
    - Огнооны баганыг robust байдлаар олно, 'Зөрчил огноо' үүсгэнэ.
    - Координат ба binary багануудыг автоматаар илрүүлнэ.
    """
    up = st.sidebar.file_uploader("Excel файл оруулах (.xlsx)", type=["xlsx"])
    if up is not None:
        df = pd.read_excel(up)
    else:
        local = Path("/mnt/data/кодлогдсон - Copy.xlsx")
        if local.exists():
            df = pd.read_excel(local)
        else:
            df = pd.read_excel(default_path)

    # Нэршил цэвэрлэгээ
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Огноо баганыг robust байдлаар олох
    recv_col = resolve_col(df, ["Хүлээн авсан", "Хүлээн авсан ", "Огноо", "Зөрчил огноо", "Осол огноо", "Ослын огноо", "Date"]) 
    if recv_col is None:
        st.error("Огнооны багана олдсонгүй. Жишээ нь: 'Хүлээн авсан'.")
        st.stop()

    # 'Зөрчил огноо' үүсгэх
    df["Зөрчил огноо"] = pd.to_datetime(df[recv_col], errors="coerce")
    df["Year"]  = df["Зөрчил огноо"].dt.year
    df["Month"] = df["Зөрчил огноо"].dt.month
    df["Day"]   = df["Зөрчил огноо"].dt.day_name()

    # Он жилүүдийн one-hot
    years = sorted(df["Year"].dropna().unique().tolist())
    for y in years:
        df[f"Зөрчил огноо жил {int(y)}"] = (df["Year"] == int(y)).astype(int)
    if len(years) > 0:
        df["Зөрчил огноо жил (min-max)"] = df["Year"].between(min(years), max(years)).astype(int)

    # Замын өргөн цэвэрлэх (байвал)
    if "Авто зам - Зорчих хэсгийн өргөн" in df.columns:
        df["Авто зам - Зорчих хэсгийн өргөн"] = df["Авто зам - Зорчих хэсгийн өргөн"].apply(clean_road_width)

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

    meta = {"lat_col": lat_col, "lon_col": lon_col, "binary_cols": binary_cols, "numeric_candidates": numeric_candidates, "years": years}
    return df, meta

# -------------------------- Ачаалж эхлэх --------------------------

df, meta = load_data()
lat_col, lon_col = meta["lat_col"], meta["lon_col"]
binary_cols = meta["binary_cols"]
num_additional = meta["numeric_candidates"]
years = meta["years"]

# -------------------------- Sidebar: Sampling & Seed --------------------------

st.sidebar.markdown("### ⚙️ Ашиглалтын тохиргоо")
seed = int(st.sidebar.number_input("Random seed", value=42, step=1))
quick_sample = st.sidebar.checkbox("⚡ 200 мөрөөр хурдан ажиллуулах (санамсаргүй)", value=False)
if quick_sample and len(df) > 200:
    df = df.sample(n=200, random_state=seed).sort_values("Зөрчил огноо")

# -------------------------- Target тохиргоо --------------------------

st.sidebar.markdown("### 🎯 Зорилтот тодорхойлолт (Осол)")
target_mode = st.sidebar.radio(
    "Осол гэж тооцох ангиллыг сонгоно уу:",
    ("Хоёуланг 1 гэж тооц", "Зөвхөн Гэмт хэрэг", "Зөвхөн Зөрчлийн хэрэг"),
)

# 'Осол' target үүсгэх ('Төрөл' баганаас)
torol_col = resolve_col(df, ["Төрөл"])  # таны датанд бий
if torol_col is None:
    st.error("`Төрөл` багана олдсонгүй. Target үүсгэх боломжгүй байна.")
    st.stop()

if target_mode == "Хоёуланг 1 гэж тооц":
    df["Осол"] = df[torol_col].isin(["Гэмт хэрэг", "Зөрчлийн хэрэг"]).astype(int)
elif target_mode == "Зөвхөн Гэмт хэрэг":
    df["Осол"] = (df[torol_col] == "Гэмт хэрэг").astype(int)
else:  # Зөвхөн Зөрчлийн хэрэг
    df["Осол"] = (df[torol_col] == "Зөрчлийн хэрэг").astype(int)

# -------------------------- 5. Ирээдүйн ослын таамаглал --------------------------

st.header("5. Ирээдүйн ослын таамаглал (Олон ML/DL загвар)")
st.caption("Binary (0/1) багануудыг автоматаар илрүүлж, загварт ашигласан. Leakage-free scaling хэрэглэсэн.")

# Feature pool: 'Осол'-оос бусад binary + нэмэлт тоон
feature_pool = [c for c in (binary_cols + num_additional) if c != "Осол"]
if len(feature_pool) == 0:
    st.error("Binary (0/1) хэлбэрийн багана олдсонгүй. Excel-ээ шалгана уу.")
    st.stop()

# Target/Features
y_all = pd.to_numeric(df["Осол"], errors="coerce").fillna(0).values
X_all = df[feature_pool].fillna(0.0).values

# Top features via RandomForest
try:
    rf_global = RandomForestRegressor(n_estimators=300, random_state=seed)
    rf_global.fit(X_all, y_all)
    importances = rf_global.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_k = min(14, len(feature_pool))
    top_features = [feature_pool[i] for i in indices[:top_k]]
    st.caption("RandomForest-аар сонгосон нөлөө ихтэй шинжүүд:")
    st.write(top_features)
except Exception as e:
    st.warning(f"Top features тооцоход алдаа гарлаа: {e}")
    top_features = feature_pool[:min(14, len(feature_pool))]

# Сар бүрийн агрегат (target=Осол==1 давтамж)
monthly_target = (
    df[df["Осол"] == 1]
    .groupby(["Year", "Month"])
    .agg(osol_count=("Осол", "sum"))
    .reset_index()
)
monthly_target["date"] = pd.to_datetime(monthly_target[["Year", "Month"]].assign(DAY=1))
monthly_features = df.groupby(["Year", "Month"])[top_features].sum().reset_index()

grouped = pd.merge(monthly_target, monthly_features, on=["Year", "Month"], how="left").sort_values("date").reset_index(drop=True)

# Lag үүсгэх
n_lag = st.sidebar.slider("Сарны лаг цонх (n_lag)", min_value=6, max_value=18, value=12, step=1)
for i in range(1, n_lag + 1):
    grouped[f"osol_lag_{i}"] = grouped["osol_count"].shift(i)

grouped = grouped.dropna().reset_index(drop=True)

if grouped.empty or len(grouped) < 10:
    st.warning(f"Сургалт хийхэд хангалттай сар тутмын өгөгдөл алга (lag={n_lag}). Он/сараа шалгана уу.")
else:
    feature_cols = [f"osol_lag_{i}" for i in range(1, n_lag + 1)] + top_features
    X = grouped[feature_cols].fillna(0.0).values
    y = grouped["osol_count"].astype(float).values.reshape(-1, 1)

    # ---------------- Leakage-free scaling ----------------
    split_ratio = st.sidebar.slider("Train ratio", 0.5, 0.9, 0.8, 0.05)
    train_size = int(len(X) * split_ratio)

    X_train_raw, X_test_raw = X[:train_size], X[train_size:]
    y_train_raw, y_test_raw = y[:train_size], y[train_size:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train_raw)
    X_test  = scaler_X.transform(X_test_raw)
    y_train = scaler_y.fit_transform(y_train_raw).flatten()
    y_test  = scaler_y.transform(y_test_raw).flatten()

    estimators = [
        ("rf", RandomForestRegressor(n_estimators=120, random_state=seed)),
        ("ridge", Ridge()),
        ("dt", DecisionTreeRegressor(random_state=seed)),
    ]

    MODEL_LIST = [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge()),
        ("Lasso", Lasso()),
        ("DecisionTree", DecisionTreeRegressor(random_state=seed)),
        ("RandomForest", RandomForestRegressor(random_state=seed)),
        ("ExtraTrees", ExtraTreesRegressor(random_state=seed)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=seed)),
        ("HistGB", HistGradientBoostingRegressor(random_state=seed)),
        ("AdaBoost", AdaBoostRegressor(random_state=seed)),
        ("KNeighbors", KNeighborsRegressor()),
        ("SVR", SVR()),
        ("MLPRegressor", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=seed)),
        ("ElasticNet", ElasticNet()),
        ("Stacking", StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=5)),
    ]
    if XGBRegressor is not None:
        MODEL_LIST.append(("XGBRegressor", XGBRegressor(verbosity=0, random_state=seed)))
    if CatBoostRegressor is not None:
        MODEL_LIST.append(("CatBoostRegressor", CatBoostRegressor(verbose=0, random_state=seed)))
    if LGBMRegressor is not None:
        MODEL_LIST.append(("LGBMRegressor", LGBMRegressor(random_state=seed)))

    progress_bar = st.progress(0, text="ML моделийг сургаж байна...")
    results = []
    y_preds = {}

    for i, (name, model) in enumerate(MODEL_LIST):
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
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

    # Ирээдүйн прогноз helper
    def forecast_next(model, last_values, steps=12):
        preds = []
        seq = last_values.copy()
        for _ in range(steps):
            pred = model.predict([seq])[0]
            preds.append(pred)
            seq = np.roll(seq, -1)
            seq[-1] = pred
        return np.array(preds)

    forecast_steps = {"30 хоног": 1, "90 хоног": 3, "180 хоног": 6, "365 хоног": 12}
    model_forecasts = {}
    last_seq = scaler_X.transform(X[-1].reshape(1, -1)).flatten()

    for name, model in MODEL_LIST:
        if name not in y_preds:
            continue
        preds_dict = {}
        for k, s in forecast_steps.items():
            scaled_preds = forecast_next(model, last_seq, steps=s)
            inv_preds = scaler_y.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
            preds_dict[k] = inv_preds
        model_forecasts[name] = preds_dict

    # Test дээрх бодит/таамаг
    test_dates = grouped["date"].iloc[-len(X_test):].values
    test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_preds_df = pd.DataFrame({"date": test_dates, "real": test_true})
    for name in model_forecasts.keys():
        ypi = scaler_y.inverse_transform(np.array(y_preds[name]).reshape(-1, 1)).flatten()
        test_preds_df[name] = ypi

    # Ирээдүйн 12 сарын таамаг (модел бүрээр)
    future_dates = pd.date_range(start=grouped["date"].iloc[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS")
    future_preds_df = pd.DataFrame({"date": future_dates})
    for name, model in MODEL_LIST:
        if name not in y_preds:
            continue
        scaled_preds = forecast_next(model, last_seq, steps=12)
        inv_preds = scaler_y.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
        future_preds_df[name] = inv_preds

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

    st.subheader("Test датан дээрх модел бүрийн бодит болон таамагласан утгууд (толгой 10 мөр):")
    st.dataframe(test_preds_df.head(10), use_container_width=True)

    st.subheader("Ирээдүйн 12 сарын прогноз (модел бүрээр):")
    st.dataframe(future_preds_df, use_container_width=True)

    st.subheader("1 жилийн прогноз график (модел сонгоод харна):")
    selected_model = st.selectbox("Модель сонгох:", list(model_forecasts.keys()))
    future = model_forecasts[selected_model]["365 хоног"]
    dates_future = pd.date_range(start=grouped["date"].iloc[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS")
    future_df = pd.DataFrame({"date": dates_future, "forecast": future})
    fig = px.line(future_df, x="date", y="forecast", markers=True, title=f"{selected_model}-ийн ирэх 12 сарын прогноз")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------- Hotspot (DBSCAN) --------------------------

st.subheader("Анхаарах газрын байршил (DBSCAN кластерчилсан hotspot)")
if lat_col and lon_col:
    # Сүүлийн 12 сарын ОЛОНТОЙ (Осол==1) мөрүүдээр кластерчилна
    recent_df = df[(df["Зөрчил огноо"] >= (df["Зөрчил огноо"].max() - pd.DateOffset(months=12))) & (df["Осол"] == 1)].copy()
    recent_df = recent_df.dropna(subset=[lat_col, lon_col]).copy()
    coords = recent_df[[lat_col, lon_col]].to_numpy()
    if len(coords) >= 3:
        kms_per_radian = 6371.0088
        epsilon = 0.1 / kms_per_radian  # ≈100м
        try:
            db = DBSCAN(eps=epsilon, min_samples=3, algorithm="ball_tree", metric="haversine").fit(np.radians(coords))
            recent_df["cluster"] = db.labels_
        except Exception:
            # Зарим sklearn хувилбарт metric="haversine" асуудал гарвал euclidean-д шилжинэ (өргөрөг/уртрагийн хэмжээнд ойролцоолол)
            db = DBSCAN(eps=0.001, min_samples=3, metric="euclidean").fit(coords)
            recent_df["cluster"] = db.labels_
    else:
        recent_df["cluster"] = -1

    hotspots = (
        recent_df[recent_df["cluster"] != -1]
        .groupby("cluster")
        .agg(
            n_osol=("Осол", "sum"),
            lat=(lat_col, "mean"),
            lon=(lon_col, "mean"),
        )
        .sort_values("n_osol", ascending=False)
        .reset_index()
    )

    map_center_lat = hotspots["lat"].mean() if not hotspots.empty else (recent_df[lat_col].mean() if len(recent_df) else 47)
    map_center_lon = hotspots["lon"].mean() if not hotspots.empty else (recent_df[lon_col].mean() if len(recent_df) else 106)
    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=12 if not hotspots.empty else 6)

    for _, row in hotspots.iterrows():
        folium.Circle(
            location=[row["lat"], row["lon"]],
            radius=120 + row["n_osol"] * 1,
            color="orange",
            fill=True,
            fill_opacity=0.55,
            popup=folium.Popup(f"<b>Hotspot (кластер)</b><br>Ослын тоо: <b>{int(row['n_osol'])}</b>", max_width=350),
        ).add_to(m)

    st_folium(m, width=1920, height=700)
else:
    st.info("Координатын баганууд (Өргөрөг/Уртраг эсхүл lat/lon) байхгүй тул газрын зургийг алгаслаа.")

# -------------------------- 1. Корреляцийн шинжилгээ --------------------------

st.header("1. Осолд нөлөөлөх хүчин зүйлсийн тархалт/корреляцийн шинжилгээ")
st.write("Доорх multiselect дээрээс ихдээ 15 хувьсагч сонгож корреляцийн матриц харна.")

vars_for_corr = ["Year"]
vars_for_corr += [c for c in df.columns if c.startswith("Зөрчил огноо жил ")][:10]
vars_for_corr += [c for c in (binary_cols + num_additional) if c in df.columns]
# Давхардал арилгах
vars_for_corr = list(dict.fromkeys(vars_for_corr))

if len(vars_for_corr) > 1:
    Xx = df[vars_for_corr].fillna(0.0).values
    yy = pd.to_numeric(df["Осол"], errors="coerce").fillna(0).values
    try:
        rf_cor = RandomForestRegressor(n_estimators=200, random_state=seed)
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
    st.warning("Сонгох хувьсагчидыг оруулна уу!")

# -------------------------- 2. Ослын өсөлтийн тренд --------------------------

st.header("2. Ослын өсөлтийн тренд")
st.subheader("Жил, сар бүрээр ослын тооны тренд")

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
fig = px.line(plot_df, x="YearMonth", y="osol_count", markers=True, labels={"YearMonth": "Он-Сар", "osol_count": "Ослын тоо"}, title="")
fig.update_layout(
    xaxis_tickangle=45,
    hovermode="x unified",
    plot_bgcolor="white",
    yaxis=dict(title="Ослын тоо", rangemode="tozero"),
    xaxis=dict(title="Он-Сар"),
)
fig.update_traces(line=dict(width=3))
st.write("Доорх графикт ослын тооны өөрчлөлтийг харуулав.")
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
    st.info("Категори багана (2-оос 15 ялгаатай утгатай) олдсонгүй.")
else:
    var1 = st.selectbox("1-р категори хувьсагч:", categorical_cols)
    var2 = st.selectbox("2-р категори хувьсагч:", [c for c in categorical_cols if c != var1])

    table = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, expected = chi2_contingency(table)

    # Хоёр хувилбарын V
    n = table.values.sum()
    r, k = table.shape
    cramers_v_naive = np.sqrt(chi2 / (n * (min(k, r) - 1))) if min(k, r) > 1 else np.nan
    cramers_v_bc = cramers_v_bias_corrected(table)

    st.subheader("1. Chi-square тест")
    st.write("p-value < 0.05 бол статистикийн хувьд хамааралтай гэж үзнэ.")
    st.write(f"**Chi-square statistic:** {chi2:.3f}")
    st.write(f"**p-value:** {p:.4f}")
    if p < 0.05:
        st.success("p < 0.05 → Статистикийн хувьд хамааралтай!")
    else:
        st.info("p ≥ 0.05 → Статистикийн хувьд хамааралгүй.")

    use_bc = st.checkbox("Bias-corrected Cramér’s V (санал болгож байна)", value=True)
    v_to_show = cramers_v_bc if use_bc else cramers_v_naive

    st.subheader("2. Cramér’s V")
    st.write("0-д ойрхон бол бараг хамааралгүй, 1-д ойр бол хүчтэй хамааралтай.")
    st.write(f"**Cramér’s V:** {v_to_show:.3f} (0=хамааралгүй, 1=хүчтэй хамаарал)")

    st.write("**Crosstab:**")
    st.dataframe(table, use_container_width=True)

# -------------------------- Төслийн төгсгөл --------------------------

st.markdown(
    """
    ---
    **Тайлбар**  
    • Зорилтот хувьсагчийг Sidebar дээрээс сонгох боломжтой (Гэмт хэрэг/Зөрчлийн хэрэг/Хосолсон).  
    • "Leakage-free" scaling: scaler-уудыг зөвхөн train дээр fit хийдэг тул үнэлгээ илүү найдвартай.  
    • ⚡ Хурдан ажиллуулахад 200 мөрийн санамсаргүй дэд дээж авч туршина.  
    • Хэрэв XGBoost/LightGBM/CatBoost суулгагдаагүй бол суулгалгүйгээр бусад моделүүд ажиллана.  
    """
)