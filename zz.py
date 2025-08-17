# -*- coding: utf-8 -*-
# ============================================================
# Зам тээврийн осол — Auto ML & Hotspot Dashboard (Streamlit)
# Хувилбар: 2025-08-17 — Binary autodetect + robust column resolver + target UI
# Тайлбар:
#  - Хавсаргасан Excel ("кодлогдсон - Copy.xlsx")-тай шууд зохицно.
#  - Binary (0/1) бүх баганыг автоматаар илрүүлж, модел/корреляц/хотспотод ашиглана.
#  - Координат баганууд (Өргөрөг/Уртраг эсвэл lat/lon) байвал газрын зураг зурна.
#  - Олон ML модел сургалт, метрик/таамаглалыг Excel болгон татах боломжтой.
#  - "Осол" багана байхгүй тохиолдолд "Төрөл"-өөс (Гэмт хэрэг/Зөрчлийн хэрэг) зорилтыг үүсгэнэ.
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm_model(input_shape, units=64):
    model = Sequential()
    model.add(LSTM(units, activation="tanh", input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model
# -------------------------- UI setup --------------------------
st.set_page_config(page_title="Осол — Auto ML & Hotspot (auto-binary)", layout="wide")

st.title=("С.Цолмон, А.Тамир нарын хар цэгийн судалгаа 2025-08-18")

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
        st.warning("Сонгосон баганууд нь one-hot (0/1) хэлбэртэй байна. Ийм тохиолдолд корреляци -1~1 туйлруугаа хэлбийж харагдаж болно.")
    df_encoded = df[columns].copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
    corr_matrix = df_encoded.corr()
    corr_matrix = corr_matrix.iloc[::-1]
    fig, ax = plt.subplots(figsize=(max(8, 1.5*len(columns)), max(6, 1.2*len(columns))))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax, fmt=".3f")
    plt.title(title)
    plt.tight_layout()
    return fig

# -------------------------- Өгөгдөл ачаалалт --------------------------
from pathlib import Path
import pandas as pd
import streamlit as st

# 1. Энд widget-ээ гадна гаргаж өгнө
uploaded_file = st.sidebar.file_uploader("Excel файл оруулах (.xlsx)", type=["xlsx"])

# 2. Кэштэй зөвхөн дата унших функц
@st.cache_data(show_spinner=True)
def load_data(file=None, default_path: str = "кодлогдсон.xlsx"):
    """
    Excel дата унших функц (widget дотор биш).
    """
    if file is not None:
        df = pd.read_excel(file)
    else:
        local = Path("кодлогдсон.xlsx")
        if local.exists():
            df = pd.read_excel(local)
        else:
            df = pd.read_excel(default_path)

    # Нэршил цэвэрлэгээ
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Огноо багана хайх
    recv_col = resolve_col(df, ["Хүлээн авсан", "Хүлээн авсан ", "Огноо", "Зөрчил огноо", 
                                "Осол огноо", "Ослын огноо", "Date"]) 
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
        "years": years
    }
    return df, meta

# -------------------------- Ачаалж эхлэх --------------------------

df, meta = load_data(uploaded_file)
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
st.caption("Binary (0/1) багануудыг автоматаар илрүүлж, загварт ашигласан.")

# Feature pool: 'Осол'-оос бусад binary + нэмэлт тоон
feature_pool = [c for c in (binary_cols + num_additional) if c != "Осол"]
if len(feature_pool) == 0:
    st.error("Binary (0/1) хэлбэрийн багана олдсонгүй. Excel-ээ шалгана уу.")
    st.stop()

# Target/Features
y_all = pd.to_numeric(df["Осол"], errors="coerce").fillna(0).values
X_all = df[feature_pool].fillna(0.0).values

# Top features via RandomForest
# Top features via RandomForest + SHAP
# Top features via RandomForest + SHAP
try:
    import shap
    rf_global = RandomForestRegressor(n_estimators=300, random_state=42)
    rf_global.fit(X_all, y_all)

    importances = rf_global.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_k = min(14, len(feature_pool))
    top_features = [feature_pool[i] for i in indices[:top_k]]

    st.caption("RandomForest-аар сонгосон нөлөө ихтэй шинжүүд (top importance):")
    st.write(top_features)

    # SHAP plot
    explainer = shap.TreeExplainer(rf_global)
    shap_values = explainer.shap_values(X_all)
    st.subheader("🔎 SHAP value шинжилгээ (global importance)")
    shap.summary_plot(shap_values, X_all, feature_names=feature_pool, plot_type="bar", show=False)
    st.pyplot(plt.gcf())  # ← жинхэнэ SHAP графикийг харуулна


    # Rare feature filter
    rare_threshold = 0.01  # <1% мөрөнд л 1 гэсэн утгатай бол 'rare'
    rare_features = []
    for col in feature_pool:
        freq = df[col].mean() if col in df else 0
        if freq < rare_threshold:
            rare_features.append(col)

    if rare_features:
        st.warning(f"⚠️ Доорх баганууд маш цөөн тохиолдолтой тул importance хэт өндөр гарч магадгүй: {rare_features}")

except Exception as e:
    st.warning(f"Top features/SHAP тооцоход алдаа гарлаа: {e}")
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
    y = grouped["osol_count"].values.reshape(-1, 1)

    # Scale
    split_ratio = st.sidebar.slider("Train ratio", 0.5, 0.9, 0.8, 0.05)
    train_size = int(len(X) * split_ratio)

    X_train, y_train = X[:train_size], y[:train_size].reshape(-1, 1)
    X_test, y_test = X[train_size:], y[train_size:].reshape(-1, 1)

    # Scale зөв дарааллаар
    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train).flatten()
    y_test = scaler_y.transform(y_test).flatten()

    estimators = [
        ("rf", RandomForestRegressor(n_estimators=120, random_state=42)),
        ("ridge", Ridge()),
        ("dt", DecisionTreeRegressor(random_state=42)),
    ]

    MODEL_LIST = [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge()),
        ("Lasso", Lasso()),
        ("DecisionTree", DecisionTreeRegressor(random_state=42)),
        ("RandomForest", RandomForestRegressor(random_state=42)),
        ("ExtraTrees", ExtraTreesRegressor(random_state=42)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
        ("HistGB", HistGradientBoostingRegressor(random_state=42)),
        ("AdaBoost", AdaBoostRegressor(random_state=42)),
        ("KNeighbors", KNeighborsRegressor()),
        ("SVR", SVR()),
        ("MLPRegressor", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42)),
        ("ElasticNet", ElasticNet()),
        ("Stacking", StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=5)),
        ("LSTM", build_lstm_model)
    ]
    if XGBRegressor is not None:
        MODEL_LIST.append(("XGBRegressor",
            XGBRegressor(tree_method="hist", predictor="cpu_predictor", random_state=42)))
    if CatBoostRegressor is not None:
        MODEL_LIST.append(("CatBoostRegressor",
            CatBoostRegressor(task_type="CPU", random_state=42, verbose=0)))
    if LGBMRegressor is not None:
        MODEL_LIST.append(("LGBMRegressor",
            LGBMRegressor(device="cpu", random_state=42)))


    # ====================================================
    # 🆕 VotingRegressor + StackingEnsemble нэмсэн хэсэг
    # ====================================================
    from sklearn.ensemble import VotingRegressor, StackingRegressor

    voting_estimators = []
    if XGBRegressor is not None:
        voting_estimators.append(("xgb", XGBRegressor(tree_method="hist", predictor="cpu_predictor", random_state=42)))
    if LGBMRegressor is not None:
        voting_estimators.append(("lgbm", LGBMRegressor(device="cpu", random_state=42)))
    if CatBoostRegressor is not None:
        voting_estimators.append(("cat", CatBoostRegressor(task_type="CPU", random_state=42, verbose=0)))
    voting_estimators.append(("rf", RandomForestRegressor(n_estimators=200, random_state=42)))
    voting_estimators.append(("gb", GradientBoostingRegressor(random_state=42)))

    if len(voting_estimators) > 1:
        MODEL_LIST.append(("VotingRegressor", VotingRegressor(estimators=voting_estimators)))

    stacking_estimators = [("rf", RandomForestRegressor(random_state=42))]
    if XGBRegressor is not None:
        stacking_estimators.append(("xgb", XGBRegressor(tree_method="hist", predictor="cpu_predictor", random_state=42)))
    if LGBMRegressor is not None:
        stacking_estimators.append(("lgbm", LGBMRegressor(device="cpu", random_state=42)))
    if CatBoostRegressor is not None:
        stacking_estimators.append(("cat", CatBoostRegressor(task_type="CPU", verbose=0, random_state=42)))


    MODEL_LIST.append((
        "StackingEnsemble",
        StackingRegressor(estimators=stacking_estimators, final_estimator=LinearRegression(), cv=5)
    ))


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
    
    def forecast_next_daily(model, last_values, steps=30):
        preds = []
        seq = np.array(last_values).reshape(1, -1)  # ML model оролт зөв хэлбэртэй болгоно
        for _ in range(steps):
            pred = model.predict(seq)[0]
            preds.append(pred)
            seq = np.roll(seq, -1, axis=1)
            seq[0, -1] = pred
        return np.array(preds)
    def forecast_next_daily_lstm(model, last_values, steps=30, window=12):
        preds = []
        seq = np.array(last_values[-window:]).reshape(1, window, 1)
        for _ in range(steps):
            pred = model.predict(seq, verbose=0)[0][0]
            preds.append(pred)
            # дараагийн алхамд input-г update хийнэ
            seq = np.roll(seq, -1, axis=1)
            seq[0, -1, 0] = pred
        return np.array(preds)

    model_forecasts = {}
    # 🛠 Алдааг зассан: X_scaled биш X_test ашиглав
    last_seq = X_test[-1]
    forecast_steps = {"7 хоног": 7, "14 хоног": 14, "30 хоног": 30, "90 хоног": 90, "180 хоног": 180, "365 хоног": 365}
    for name, model in MODEL_LIST:
        if name not in y_preds:
            continue
        preds_dict = {}
        for k, s in forecast_steps.items():
            if name == "LSTM":
                scaled_preds = forecast_next_daily_lstm(model, last_seq, steps=s, window=n_lag)
            else:
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

    st.subheader("Хоризонт сонгож графикаар харах:")


# Модель сонголт (өдөрийн pipeline ажиллаж байгаа эсэхээс шалтгаалан)
model_options = list(y_preds.keys()) if 'y_preds' in locals() and len(y_preds) > 0 else list(model_forecasts.keys())
selected_model = st.selectbox("Модель сонгох:", model_options)
selected_h = st.selectbox("Хоризонт:", list(forecast_steps.keys()), index=2)

# Огнооны нягтрал шилжлүүлт
gran = st.radio("Дэлгэцлэх огнооны нягтрал №1:", ["Өдөр", "Сар"], index=0, horizontal=True)
last_date = grouped["date"].iloc[-1]
last_seq = X_test[-1]

last_lags_raw = grouped[feature_cols].iloc[-1].values

steps = forecast_steps[selected_h]
if 'forecast_next_daily' in globals():
    # Өдөр тутмын таамаглалтай горим
    plot_future = forecast_next_daily(dict(MODEL_LIST)[selected_model], last_seq, steps)
    plot_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq="D")
    future_df = pd.DataFrame({"date": plot_dates, "forecast": plot_future})
    if gran == "Сар":
        future_df = future_df.set_index("date").resample("MS").sum().reset_index()
    title = f"{selected_model} — ирэх {steps} хоногийн прогноз ({'өдөр' if gran=='Өдөр' else 'сар'})"
else:
    # Сарын fallback горим
    preds = model_forecasts[selected_model].get(selected_h)
    months = len(preds) if preds is not None else 0
    dates_future = pd.date_range(start=grouped["date"].iloc[-1] + pd.offsets.MonthBegin(), periods=months, freq="MS")
    future_df = pd.DataFrame({"date": dates_future, "forecast": preds})
    gran = "Сар"
    title = f"{selected_model} — {selected_h} прогноз (сар)"


fig = px.line(future_df, x="date", y="forecast", markers=True, title=title)
st.plotly_chart(fig, use_container_width=True)

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
        rf_cor = RandomForestRegressor(n_estimators=200, random_state=42)
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
# -------------------------- 2. Ослын өсөлтийн тренд --------------------------
st.header("2. Ослын өсөлтийн тренд") 
st.subheader("Жил, сар бүрээр ослын тооны тренд") 
trend_data = ( df[df["Осол"] == 1] .groupby(["Year", "Month"]) .agg(osol_count=("Осол", "sum")) .reset_index() ) 
trend_data["YearMonth"] = trend_data.apply(lambda x: f"{int(x['Year'])}-{int(x['Month']):02d}", axis=1) 
available_years = sorted(trend_data["Year"].unique()) 
year_options = ["Бүгд"] + [str(y) for y in available_years] 
selected_year = st.selectbox("Жил сонгох:", year_options) 
plot_df = trend_data if selected_year == "Бүгд" else trend_data[trend_data["Year"] == int(selected_year)].copy() 
fig = px.line(plot_df, x="YearMonth", y="osol_count", markers=True, labels={"YearMonth": "Он-Сар", "osol_count": "Ослын тоо"}, title="") 
fig.update_layout( xaxis_tickangle=45, hovermode="x unified", plot_bgcolor="white", yaxis=dict(title="Ослын тоо", rangemode="tozero"), xaxis=dict(title="Он-Сар"), ) 
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
    n = table.values.sum()
    r, k = table.shape
    cramers_v = np.sqrt(chi2 / (n * (min(k, r) - 1))) if min(k, r) > 1 else np.nan

    st.subheader("1. Chi-square тест")
    st.write("p-value < 0.05 бол статистикийн хувьд хамааралтай гэж үзнэ.")
    st.write(f"**Chi-square statistic:** {chi2:.3f}")
    st.write(f"**p-value:** {p:.4f}")
    if p < 0.05:
        st.success("p < 0.05 → Статистикийн хувьд хамааралтай!")
    else:
        st.info("p ≥ 0.05 → Статистикийн хувьд хамааралгүй.")

    st.subheader("2. Cramér’s V")
    st.write("0-д ойрхон бол бараг хамааралгүй, 1-д ойр бол хүчтэй хамааралтай.")
    st.write(f"**Cramér’s V:** {cramers_v:.3f} (0=хамааралгүй, 1=хүчтэй хамаарал)")

    st.write("**Crosstab:**")
    st.dataframe(table, use_container_width=True)

# -------------------------- Төслийн төгсгөл --------------------------
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

# Crosstab + χ² + Cramér’s V
table = pd.crosstab(df["Season"], df["Төрөл"])
chi2, p, dof, exp = chi2_contingency(table)
n = table.values.sum()
r, k = table.shape
cramers_v = np.sqrt(chi2 / (n*(min(k,r)-1)))

st.subheader("Улирлын ялгаа (χ² ба Cramér’s V)")
st.write("**Chi-square statistic:**", round(chi2, 3))
st.write("**p-value:**", round(p, 4))
st.write("**Cramér’s V:**", round(cramers_v, 3))
st.dataframe(table, use_container_width=True)






# -------------------------- 6. Empirical Bayes шинжилгээ --------------------------

st.header("6. Empirical Bayes before–after шинжилгээ (сар бүр)")


def empirical_bayes(obs, exp, prior_mean, prior_var):
    """EB хүлээгдэж буй vs ажиглагдсан тооцоолол"""
    weight = prior_var / (prior_var + exp)
    return weight * obs + (1 - weight) * prior_mean

# Сар бүрийн агрегат (аль хэдийн trend_data дээр байгаа)
monthly = (
    df[df["Осол"] == 1]
    .groupby(["Year", "Month"])
    .agg(osol_count=("Осол", "sum"))
    .reset_index()
)
monthly["date"] = pd.to_datetime(monthly[["Year", "Month"]].assign(DAY=1))

# Before/After хуваалт: 2020–2022 before, 2023–2024 after
monthly["period"] = np.where(monthly["Year"] <= 2023, "before", "after")

# Хүлээгдэж буй утга = before үеийн дундаж
expected = monthly[monthly["period"]=="before"]["osol_count"].mean()
prior_mean = expected
prior_var = expected / 2

# EB-г зөвхөн after дээр тооцно
monthly["EB"] = monthly.apply(
    lambda row: empirical_bayes(
        row["osol_count"], expected, prior_mean, prior_var
    ) if row["period"]=="after" else row["osol_count"],
    axis=1
)

# st.write("EB үр дүн (сар бүрийн түвшинд):")
# st.dataframe(monthly.head(24))

# Графикаар харуулах
fig = px.line(
    monthly, x="date", y=["osol_count","EB"], 
    color="period", markers=True,
    labels={"value":"Осол (тоо)", "date":"Он-Сар"},
    title="Ослын сар бүрийн тоо (EB жинлэлийн коэффициент)"
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------- 7. Empirical Bayes шинжилгээ (байршлаар, сар бүр) --------------------------
st.header("7. Empirical Bayes before–after шинжилгээ (байршлаар, сар бүр)")

st.markdown("""
EB жинлэлийн коэффициент гэж юу вэ?
Empirical Bayes (EB) арга нь бодит ажиглагдсан (Observed) болон хүлээгдэж буй (Expected) утгыг 
жинлэж нэгтгэдэг статистик аргачлал юм. Энэ нь санамсаргүй хэлбэлзлээс шалтгаалсан гажуудлыг 
багасгах зорилготой.

Математик загварчлал:

EB = w * Observed + (1 - w) * Expected  
w = PriorVar / (PriorVar + Expected)

- **Observed** = тухайн сар/байршлын ослын тоо  
- **Expected** = өмнөх хугацааны дундаж ослын тоо  
- **PriorVar** = суурь хэлбэлзэл (дундажийн талыг ашигласан)

""")


def empirical_bayes(obs, exp, prior_mean, prior_var):
    """EB жигнэлт: ажиглагдсан ба хүлээгдэж буйг нэгтгэх"""
    weight = prior_var / (prior_var + exp)
    return weight * obs + (1 - weight) * prior_mean

# -------------------------- Байршлын баганыг таних --------------------------
loc_col = resolve_col(df, ["Замын байршил ", "Байршил", "location", "Road Location"])
if loc_col is None:
    st.error("⚠️ Замын байршлын багана олдсонгүй. Excel дээр 'Замын байршил' гэх мэт багана байгаа эсэхийг шалгана уу.")
    st.stop()

# -------------------------- Сар бүрийн агрегат --------------------------
monthly_loc = (
    df[df["Осол"] == 1]
    .groupby([loc_col, "Year", "Month"])
    .agg(osol_count=("Осол", "sum"))
    .reset_index()
)
monthly_loc["date"] = pd.to_datetime(monthly_loc[["Year", "Month"]].assign(DAY=1))

# -------------------------- Before/After хугацаа сонгох (жилээр) --------------------------
years = sorted(df["Year"].unique())
col1, col2, col3, col4 = st.columns(4)
with col1:
    before_start = st.selectbox("Before эхлэх жил", years, index=0)
with col2:
    before_end = st.selectbox("Before дуусах жил", years, index=len(years)//2 - 1)
with col3:
    after_start = st.selectbox("After эхлэх жил", years, index=len(years)//2)
with col4:
    after_end = st.selectbox("After дуусах жил", years, index=len(years)-1)

before_range = (pd.to_datetime(f"{before_start}-01-01"), pd.to_datetime(f"{before_end}-12-31"))
after_range = (pd.to_datetime(f"{after_start}-01-01"), pd.to_datetime(f"{after_end}-12-31"))

# -------------------------- Period оноох --------------------------
monthly_loc["period"] = np.where(
    (monthly_loc["date"] >= before_range[0]) & (monthly_loc["date"] <= before_range[1]),
    "before",
    np.where(
        (monthly_loc["date"] >= after_range[0]) & (monthly_loc["date"] <= after_range[1]),
        "after",
        "outside"
    )
)

# зөвхөн before/after үлдээнэ
monthly_loc = monthly_loc[monthly_loc["period"].isin(["before","after"])]

# -------------------------- EB тооцоолол --------------------------
results = []
for loc, grp in monthly_loc.groupby(loc_col):
    expected = grp[grp["period"] == "before"]["osol_count"].mean()
    if pd.isna(expected):  # before хоосон бол алгасна
        continue
    prior_mean = expected
    prior_var = expected / 2

    grp["EB"] = grp.apply(
        lambda row: empirical_bayes(
            row["osol_count"], expected, prior_mean, prior_var
        ) if row["period"] == "after" else row["osol_count"],
        axis=1
    )
    results.append(grp)

if results:
    monthly_loc = pd.concat(results)
else:
    st.warning("⚠️ Сонгосон хугацаанд EB тооцоолох өгөгдөл олдсонгүй.")

# -------------------------- OUTPUT --------------------------
st.write("EB үр дүн (байршлаар, сар бүр):")
st.dataframe(monthly_loc.head(500))

# График
if not monthly_loc.empty:
    fig = px.line(
        monthly_loc, x="date", y="EB",
        color=loc_col, line_dash="period", markers=True,
        labels={"EB":"EB-жигнэсэн ослын тоо", "date":"Он-Сар", loc_col:"Байршил"},
        title="Ослын EB-жигнэлттэй тоо (байршлаар, сар бүр)"
    )
    st.plotly_chart(fig, use_container_width=True)
# -------------------------- 8. Байршлын өөрчлөлтөөр эрэмбэлэх --------------------------
st.header("8. Хамгийн сайн / муу 100 байршлын жагсаалт")

# Before / After дундаж EB-г тооцоолох
summary = (
    monthly_loc.groupby([loc_col, "period"])["EB"]
    .mean()
    .reset_index()
    .pivot(index=loc_col, columns="period", values="EB")
    .reset_index()
)

# Before ба After ялгавар
summary["Δ"] = summary["after"] - summary["before"]

# Хамгийн сайн 100 (Δ хамгийн бага)
best_100 = summary.nsmallest(100, "Δ")

# Хамгийн муу 100 (Δ хамгийн их)
worst_100 = summary.nlargest(100, "Δ")

# ==================== OUTPUT ====================
st.subheader("✅ Хамгийн сайн 100 байршил (EB буурсан)")
st.dataframe(best_100)

st.subheader("❌ Хамгийн муу 100 байршил (EB нэмэгдсэн)")
st.dataframe(worst_100)

# Хүсвэл Excel болгож татаж авах
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "📥 Best 100 (CSV)", convert_df(best_100), "best_100.csv", "text/csv"
)

st.download_button(
    "📥 Worst 100 (CSV)", convert_df(worst_100), "worst_100.csv", "text/csv"
)


st.markdown(
    """
    ---
    **Тайлбар**  
    • Зорилтот хувьсагчийг Sidebar дээрээс сонгох боломжтой (Гэмт хэрэг/Зөрчлийн хэрэг/Хосолсон).  
    • Том хэмжээтэй файлуудад `@st.cache_data` ачааллааг бууруулна.  
    • Хэрэв XGBoost/LightGBM/CatBoost суулгагдаагүй бол суулгалгүйгээр бусад моделүүд ажиллана.  
    """
)


# ============================================================
# 9. Координат удамшуулах (2024 → 2020–2023) + DBSCAN шинжилгээ + Газрын зураг
# ============================================================

st.header("9. Удамшсан координат дээр DBSCAN кластерчилал")

# --- шаардлагатай баганууд ---
req_cols = ["Замын код", "Аймаг-Дүүрэг", "Хороо-Сум", "Зөрчил гарсан газрын хаяг", "Замын байршил "]
missing_cols = [c for c in req_cols if c not in df.columns]



if not (lat_col and lon_col):
    st.info("Координатын баганууд олдсонгүй (Өргөрөг/Уртраг эсвэл lat/lon). Энэ хэсгийг алгаслаа.")
else:
    # --- 9.1 Reference (2024 он) бэлтгэх ---
    df_2024 = df[df["Year"] == 2024].copy()
    df_2024["ref_key"] = (
        df_2024["Замын код"].astype(str) + "_" +
        df_2024["Аймаг-Дүүрэг"].astype(str) + "_" +
        df_2024["Хороо-Сум"].astype(str) + "_" +
        df_2024["Зөрчил гарсан газрын хаяг"].astype(str) + "_" +
        df_2024["Замын байршил "].astype(str)
    )
    df_2024_grouped = (
        df_2024.groupby("ref_key")[[lat_col, lon_col]]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )
    ref_dict = df_2024_grouped.set_index("ref_key")[[lat_col, lon_col]].to_dict("index")

    # --- 9.2 Координат удамшуулах функц ---
    try:
        from haversine import haversine
    except Exception:
        haversine = None

    def inherit_coords(row, threshold_m=500):
        if row["Year"] == 2024:
            return row[lat_col], row[lon_col]
        key = (
            str(row["Замын код"]) + "_" +
            str(row["Аймаг-Дүүрэг"]) + "_" +
            str(row["Хороо-Сум"]) + "_" +
            str(row["Зөрчил гарсан газрын хаяг"]) + "_" +
            str(row["Замын байршил "])
        )


        if key in ref_dict:
            ref_lat = ref_dict[key][lat_col]
            ref_lon = ref_dict[key][lon_col]
            if haversine is not None and pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
                try:
                    dist = haversine((ref_lat, ref_lon), (row[lat_col], row[lon_col])) * 1000
                except:
                    dist = np.inf
            else:
                dist = np.inf
            if (41 <= row[lat_col] <= 52) and (87 <= row[lon_col] <= 120):
                return (row[lat_col], row[lon_col]) if dist <= threshold_m else (ref_lat, ref_lon)
            else:
                return ref_lat, ref_lon
        else:
            return np.nan, np.nan

    df[[lat_col, lon_col]] = df.apply(lambda row: inherit_coords(row), axis=1, result_type="expand")
    df = df.dropna(subset=[lat_col, lon_col])

    # --- 9.3 DBSCAN кластерчилал ---
    coords = df[[lat_col, lon_col]].to_numpy()
    if len(coords) > 5:
        eps_val = st.sidebar.slider("DBSCAN eps (радиан)", 0.001, 0.02, 0.005, step=0.001)
        db = DBSCAN(eps=eps_val, min_samples=5, metric="haversine")
        df["cluster_inherited"] = db.fit_predict(np.radians(coords))
    else:
        df["cluster_inherited"] = -1

    # --- 9.4 Кластер тренд тооцох ---
    trend_list = []
    for cl in df["cluster_inherited"].unique():
        if cl == -1:
            continue
        subset = df[df["cluster_inherited"] == cl]
        counts = subset.groupby("Year").size()
        if counts.shape[0] < 2:
            trend = "тогтвортой"
        else:
            diff = counts.diff().dropna()
            if all(diff > 0):
                trend = "өсөлт"
            elif all(diff < 0):
                trend = "бууралт"
            elif diff.max() > 2 * abs(diff.mean()): 
                trend = "огцом өсөлт"
            elif diff.min() < -2 * abs(diff.mean()):
                trend = "огцом бууралт"
            else:
                trend = "тогтвортой"
        trend_list.append({"cluster": cl, "trend": trend, "тоо": counts.sum()})

    if trend_list:
        df_trend = pd.DataFrame(trend_list).sort_values("тоо", ascending=False)
        st.subheader("Кластерийн трендүүд (2020–2024)")
        st.dataframe(df_trend, use_container_width=True)
    else:
        st.warning("⚠️ DBSCAN-аар кластер тодорхойлогдсонгүй (бүгд -1 болсон байж магадгүй). eps/min_samples тохиргоог шалгана уу.")
        df_trend = pd.DataFrame()


    st.subheader("Кластерийн трендүүд (2020–2024)")
    st.dataframe(df_trend, use_container_width=True)

    # --- 9.5 Binary хувьсагчийн ач холбогдол ---
    from sklearn.ensemble import RandomForestClassifier

    if binary_cols and len(df) > 0 and df["cluster_inherited"].nunique() > 1:
        X = df[binary_cols]
        y = df["cluster_inherited"].astype(str)

        if len(X) > 0:
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X, y)
            feature_imp = pd.DataFrame({
                "feature": binary_cols,
                "importance": rf.feature_importances_
            }).sort_values("importance", ascending=False)
            st.subheader("Binary хувьсагчийн ач холбогдол (RandomForest importance)")
            st.dataframe(feature_imp, use_container_width=True)
        else:
            st.info("⚠️ Кластер болон binary хувьсагчид огтлолцох мөр олдсонгүй.")
    else:
        st.info("⚠️ Binary (0/1) багана эсвэл кластерийн өгөгдөл байхгүй тул importance тооцоогүй.")


    # --- 9.6 Газрын зураг дээр дүрслэх ---
    import folium
    from streamlit_folium import st_folium
    st.subheader("DBSCAN кластеруудын газрын зураг (2020–2024)")

    if len(df) > 0:
        m = folium.Map(location=[47.92, 106.92], zoom_start=5, tiles="OpenStreetMap")

        import matplotlib.cm as cm
        import matplotlib.colors as colors
        clusters = df["cluster_inherited"].unique()
        norm = colors.Normalize(vmin=min(clusters), vmax=max(clusters))
        colormap = cm.ScalarMappable(norm=norm, cmap="tab20")

        trend_dict = dict(zip(df_trend["cluster"], df_trend["trend"]))

        for _, row in df.iterrows():
            cl = row["cluster_inherited"]
            popup_txt = (
                f"Он: {row['Year']}<br>"
                f"Замын код: {row['Замын код']}<br>"
                f"Аймаг-Дүүрэг: {str(row['Аймаг-Дүүрэг'])}<br>"
                f"Хороо-Сум: {str(row['Хороо-Сум'])}<br>"
                f"Байршил: {str(row['Замын байршил '])}<br>"
                f"Кластер: {cl}<br>"
                f"Тренд: {trend_dict.get(cl, 'N/A')}"
            )
            color = "gray" if cl == -1 else colors.to_hex(colormap.to_rgba(cl))
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=popup_txt
            ).add_to(m)

        st_folium(m, width=900, height=600)
    else:
        st.info("Газрын зурагт харуулах дата алга.")


