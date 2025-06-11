import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import chi2_contingency, pearsonr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import folium
from folium.plugins import MarkerCluster
import warnings

warnings.filterwarnings('ignore')


def load_data(file_path):
    df = pd.read_excel(file_path)
    print("Columns in the DataFrame:", df.columns.tolist())

    df.columns = df.columns.str.strip()
    df['Уртраг'] = df.get('Уртраг', pd.Series()).fillna(0)
    df['Өргөрөг'] = df.get('Өргөрөг', pd.Series()).fillna(0)

    if 'Авто зам - Зорчих хэсгийн өргөн' in df.columns:
        df['Зорчих хэсгийн өргөн'] = df['Авто зам - Зорчих хэсгийн өргөн'].apply(convert_width)
        df['Зорчих хэсгийн өргөн'].fillna(df['Зорчих хэсгийн өргөн'].median(), inplace=True)

    if 'Зөрчил огноо' in df.columns:
        df['Зөрчил огноо'] = pd.to_datetime(df['Зөрчил огноо'])
        df['Сар'] = df['Зөрчил огноо'].dt.month
        df['Жил'] = df['Зөрчил огноо'].dt.year
        df['Өдрийн цаг'] = df['Зөрчил огноо'].dt.hour
        df['Шөнө'] = df['Өдрийн цаг'].apply(lambda x: 1 if x >= 20 or x < 6 else 0)

    road_surfaces = ['Авто зам - Замын хучилт Тодорхойгүй',
                     'Авто зам - Замын хучилт Асфальт',
                     'Авто зам - Замын хучилт Бетон',
                     'Авто зам - Замын хучилт Сайжруулсан',
                     'Авто зам - Замын хучилт Хайрган',
                     'Авто зам - Замын хучилт Хөрсөн',
                     'Авто зам - Замын хучилт Цементэн']

    df['Замын хучилт'] = df[road_surfaces].idxmax(axis=1).str.replace('Авто зам - Замын хучилт ', '')

    return df


def convert_width(value):
    if isinstance(value, str):
        if 'ээс дээш' in value:
            return np.nan
        parts = value.replace('м', '').split('-')
        if len(parts) == 2:
            try:
                return (float(parts[0].replace(',', '.')) + float(parts[1].replace(',', '.'))) / 2
            except ValueError:
                return np.nan
    return np.nan


def descriptive_analysis(df):
    results = {}
    results['Дүүрэг'] = df['Дүүргийн нэр'].value_counts()
    results['Замын хучилт'] = df['Замын хучилт'].value_counts()

    violation_types = ['хурд хэтрүүлсэн', 'согтуугаар тээврийн хэрэгсэл жолоодсон']
    if all(col in df.columns for col in violation_types):
        results['Зөрчлийн төрөл'] = df[violation_types].sum()

    if 'Ослын ноцтой байдал' in df:
        results['Ослын ноцтой байдал'] = df['Ослын ноцтой байдал'].value_counts()

    weather_cols = [col for col in df.columns if 'Цаг агаар' in col]
    for col in weather_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    results['Цаг агаар'] = df[weather_cols].sum() if weather_cols else pd.Series()

    if 'Зорчих хэсгийн өргөн' in df:
        df['Замын өргөн бүлэг'] = pd.cut(df['Зорчих хэсгийн өргөн'], bins=5)
        results['Замын өргөн'] = df['Замын өргөн бүлэг'].value_counts()

    results['Өдөр/Шөнө'] = df['Шөнө'].value_counts() if 'Шөнө' in df else pd.Series()

    if 'Авто зам - Замын ангилал' in df:
        results['Замын ангилал'] = df['Авто зам - Замын ангилал'].value_counts()

    season_map = {12: 'Өвөл', 1: 'Өвөл', 2: 'Өвөл',
                  3: 'Хаврын', 4: 'Хаврын', 5: 'Хаврын',
                  6: 'Зун', 7: 'Зун', 8: 'Зун',
                  9: 'Намар', 10: 'Намар', 11: 'Намар'}
    df['Улирал'] = df['Сар'].map(season_map)
    results['Улирал'] = df['Улирал'].value_counts()
    results['Жилийн чиг хандлага'] = df['Жил'].value_counts().sort_index()

    return results


def filter_mongolia_points(df):
    # Filter out points outside Mongolia (bounding box roughly for Mongolia)
    df_filtered = df[(df['Өргөрөг'] >= 41.6) & (df['Өргөрөг'] <= 52.1) &
                     (df['Уртраг'] >= 87.7) & (df['Уртраг'] <= 119.9)].copy()

    # Add specified columns or default placeholders
    df_filtered['Газар /Хэлтэс/'] = df_filtered.get('Газар /Хэлтэс/', 'Тодорхойгүй')
    df_filtered['Зөрчил гарсан газрын хаяг'] = df_filtered.get('Зөрчил гарсан газрын хаяг', 'Тодорхойгүй')
    df_filtered['Замын байршил /тайлбар/'] = df_filtered.get('Замын байршил /тайлбар/', 'Тодорхойгүй')
    df_filtered['Зөрчил огноо'] = df_filtered.get('Зөрчил огноо', pd.NaT)
    df_filtered['Газар'] = df_filtered.get('Газар', 'Тодорхойгүй')

    return df_filtered


def identify_black_spots(df):
    geometry = [Point(xy) for xy in zip(df['Уртраг'], df['Өргөрөг'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Buffer distances in meters converted to degrees approximately (1 degree latitude ~ 111 km)
    district_buffer = 25 / 111000
    province_buffer = 2500 / 111000

    black_spots_records = []
    sindex = gdf.sindex

    for idx, row in gdf.iterrows():
        point = row.geometry
        if row.get('Дүүргийн нэр', None) == 1:
            buffer_dist = district_buffer
            threshold = 3
        elif row.get('Аймгийн нэр', None) == 1:
            buffer_dist = province_buffer
            threshold = 2
        else:
            continue

        buffer = point.buffer(buffer_dist)
        possible_matches_index = list(sindex.intersection(buffer.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(buffer)]

        accident_count = len(precise_matches)
        if accident_count >= threshold:
            spot_info = {
                "index": idx,
                "geometry": point,
                "accident_count": accident_count,
                "area_type": "Дүүрэг" if buffer_dist == district_buffer else "Аймаг",
                "black_spot_id": len(black_spots_records) + 1,
                "related_records": precise_matches.index.tolist()
            }
            black_spots_records.append(spot_info)

    black_spots_gdf = gpd.GeoDataFrame(black_spots_records)
    if not black_spots_gdf.empty:
        black_spots_gdf.set_geometry('geometry', inplace=True)
        black_spots_gdf.crs = "EPSG:4326"
    return black_spots_gdf


def create_black_spots_map(df, black_spots_gdf):
    m = folium.Map(location=[46.8, 103.8], zoom_start=6, tiles='CartoDB positron')

    # Add accident points
    marker_cluster = MarkerCluster(name='Accidents').add_to(m)

    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Өргөрөг'], row['Уртраг']],
            radius=4,
            color='blue',
            fill=True,
            fill_opacity=0.6,
            popup=folium.Popup(f"Дүүрэг: {row.get('Дүүргийн нэр', 'Тодорхойгүй')}<br>"
                               f"Зөрчил огноо: {row.get('Зөрчил огноо', 'Тодорхойгүй')}<br>"
                               f"Газар /Хэлтэс/: {row.get('Газар /Хэлтэс/', 'Тодорхойгүй')}", max_width=300)
        ).add_to(marker_cluster)

    # Add black spots on map
    for _, spot in black_spots_gdf.iterrows():
        folium.CircleMarker(
            location=[spot.geometry.y, spot.geometry.x],
            radius=10,
            color='black',
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(f"Хар цэг ID: {spot.black_spot_id}<br>"
                               f"Ослын тоо: {spot.accident_count}<br>"
                               f"Оролцсон тохиолдлууд: {spot.related_records}", max_width=300)
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save('black_spots_map.html')
    print("Black spots interactive map saved as 'black_spots_map.html'")
    return m


def future_black_spots(df):
    # С зөв coordinates (Монголд тааруулах)
    df = df[(df['Өргөрөг'] >= 41.5) & (df['Өргөрөг'] <= 52.2) & (df['Уртраг'] >= 87.5) & (df['Уртраг'] <= 120.5)].copy()

    geometry = [Point(xy) for xy in zip(df['Уртраг'], df['Өргөрөг'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Cluster for possible future risk spots
    n_clusters = min(15, len(gdf))  # avoid n_clusters > n_samples
    coords = np.array([[row['Өргөрөг'], row['Уртраг']] for _, row in gdf.iterrows()])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    gdf['risk_cluster'] = kmeans.fit_predict(coords)

    # Label clusters with few points as "potential future black spots"
    cluster_counts = gdf['risk_cluster'].value_counts()
    risk_clusters = cluster_counts[cluster_counts < 3].index.tolist()

    future_spots = gdf[gdf['risk_cluster'].isin(risk_clusters)].copy()
    future_spots['Хар цэгийн дугаар'] = future_spots['risk_cluster']
    future_spots['Шалгаанууд'] = "Ирээдүйд эрсдэлтэй"

    # Folium Map
    m = folium.Map(location=[46.8, 103.8], zoom_start=6, tiles='CartoDB positron')
    marker_cluster = MarkerCluster(name='Ireedui Black Spots').add_to(m)

    for idx, row in future_spots.iterrows():
        # folium expects [latitude, longitude]
        folium.CircleMarker(
            location=[row['Өргөрөг'], row['Уртраг']],
            radius=6,
            color='orange',
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(f"Хар цэгийн дугаар: {row['Хар цэгийн дугаар']}<br>Шалгаанууд: {row['Шалгаанууд']}", max_width=300)
        ).add_to(marker_cluster)

    folium.LayerControl().add_to(m)
    m.save('future_black_spots.html')
    print("Future risk black spots map saved as 'future_black_spots.html'")
    return m


def predict_future_accidents(df):
    results = {}
    # Prepare features and target
    features = ['Замын хучилт', 'Зорчих хэсгийн өргөн', 'Сар', 'Өдрийн цаг', 'Шөнө']
    X = pd.get_dummies(df[features], drop_first=True)
    if 'хурд хэтрүүлсэн' not in df.columns:
        print("Target column 'хурд хэтрүүлсэн' missing for prediction.")
        return results
    y = df['хурд хэтрүүлсэн']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost Regressor': XGBRegressor(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = mse

    # Placeholder for forecasting per region (province, district)
    # Real implementation would need time-series or panel data structure

    return results


def spatial_analysis_html(df):
    df_filtered = filter_mongolia_points(df)
    geometry = [Point(xy) for xy in zip(df_filtered['Уртраг'], df_filtered['Өргөрөг'])]
    gdf = gpd.GeoDataFrame(df_filtered, geometry=geometry, crs="EPSG:4326")

    coords = np.array([[geom.y, geom.x] for geom in gdf.geometry])
    kmeans = KMeans(n_clusters=10, random_state=42)
    gdf['hotspot_cluster'] = kmeans.fit_predict(coords)

    m = folium.Map(location=[46.8, 103.8], zoom_start=6, tiles='CartoDB positron')
    marker_cluster = MarkerCluster(name='Hotspot Clusters').add_to(m)

    color_palette = sns.color_palette("hls", 10).as_hex()

    for idx, row in gdf.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color=color_palette[row['hotspot_cluster']],
            fill=True,
            fill_color=color_palette[row['hotspot_cluster']],
            fill_opacity=0.7,
            popup=folium.Popup(f"Дүүрэг: {row.get('Дүүргийн нэр', 'Тодорхойгүй')}<br>"
                               f"Замын хучилт: {row.get('Замын хучилт', 'Тодорхойгүй')}<br>"
                               f"Зөрчил тоо: {row.get('хурд хэтрүүлсэн', 'Мэдээлэлгүй')}", max_width=300)
        ).add_to(marker_cluster)

    folium.LayerControl().add_to(m)
    m.save('hotspots.html')
    print("Interactive hotspot map saved as 'hotspots.html'")
    return m


def temporal_analysis(df):
    results = {}
    monthly = df.resample('M', on='Зөрчил огноо').size()

    arima_model = ARIMA(monthly, order=(1, 1, 1))
    arima_results = arima_model.fit()
    results['arima_forecast'] = arima_results.forecast(steps=12)

    sarima_model = SARIMAX(monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_results = sarima_model.fit()
    results['sarima_forecast'] = sarima_results.forecast(steps=12)

    results['Цагийн хуваарилалт'] = df.groupby('Өдрийн цаг').size()
    df['Ажлын өдөр'] = df['Зөрчил огноо'].dt.dayofweek < 5
    results['Ажлын өдөр'] = df['Ажлын өдөр'].value_counts()

    annual = df.groupby('Жил').size()
    results['Жилийн өсөлт'] = annual.pct_change().mean()

    seq_length = 12
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(monthly.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length])

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.array(X), np.array(y), epochs=50, verbose=0)
    results['lstm_model'] = model

    return results


def correlation_analysis(df):
    results = {}

    if 'Ослын ноцтой байдал' in df:
        contingency = pd.crosstab(df['Замын хучилт'], df['Ослын ноцтой байдал'])
        chi2, p, dof, ex = chi2_contingency(contingency)
        results['chi2_road_severity'] = {'chi2': chi2, 'p-value': p}

    weather_col = [col for col in df.columns if 'Цаг агаар' in col]
    violation_col = 'хурд хэтрүүлсэн'
    # Continuing correlation_analysis function

    if weather_col and violation_col in df:
        valid_data = df[[weather_col[0], violation_col]].dropna()
        if len(valid_data[weather_col[0]]) >= 2 and len(valid_data[violation_col]) >= 2:
            corr, p_val = pearsonr(valid_data[weather_col[0]], valid_data[violation_col])
            results['weather_violation_corr'] = {'correlation': corr, 'p-value': p_val}
        else:
            results['weather_violation_corr'] = {'error': 'Insufficient data for correlation analysis'}

    if 'Зорчих хэсгийн өргөн' in df and 'хурд хэтрүүлсэн' in df:
        valid_data = df[['Зорчих хэсгийн өргөн', 'хурд хэтрүүлсэн']].dropna()
        if len(valid_data['Зорчих хэсгийн өргөн']) >= 2 and len(valid_data['хурд хэтрүүлсэн']) >= 2:
            corr, p_val = pearsonr(valid_data['Зорчих хэсгийн өргөн'], valid_data['хурд хэтрүүлсэн'])
            results['road_width_violation'] = {'correlation': corr, 'p-value': p_val}
        else:
            results['road_width_violation'] = {'error': 'Insufficient data for correlation analysis'}

    if 'замын гэрэлтүүлэг хангалтгүй' in df:
        night_violations = df[df['Шөнө'] == 1]['замын гэрэлтүүлэг хангалтгүй']
        results['lighting_night_corr'] = night_violations.mean()

    return results
def temporal_analysis(df):
    """
    Conduct temporal analysis on traffic violation data.
    Returns a dictionary with:
    - ARIMA and SARIMA forecasts for 12 months,
    - Hourly distribution of violations,
    - Workday vs holiday distribution,
    - Annual growth rate,
    - Trained LSTM model for sequence prediction.
    """
    results = {}

    # Monthly counts time series for forecasting
    monthly = df.resample('M', on='Зөрчил огноо').size()

    # ARIMA model for trend forecasting
    arima_model = ARIMA(monthly, order=(1, 1, 1))
    arima_results = arima_model.fit()
    results['arima_forecast'] = arima_results.forecast(steps=12)

    # SARIMA model capturing seasonality
    sarima_model = SARIMAX(monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_results = sarima_model.fit()
    results['sarima_forecast'] = sarima_results.forecast(steps=12)

    # Hourly distribution of violations
    results['Цагийн хуваарилалт'] = df.groupby('Өдрийн цаг').size()

    # Workday (Mon-Fri) vs weekend
    df['Ажлын өдөр'] = df['Зөрчил огноо'].dt.dayofweek < 5
    results['Ажлын өдөр'] = df['Ажлын өдөр'].value_counts()

    # Annual growth rate in violations
    annual = df.groupby('Жил').size()
    results['Жилийн өсөлт'] = annual.pct_change().mean()

    # Prepare data for LSTM sequence model
    seq_length = 12
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(monthly.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length])

    X_arr = np.array(X)
    y_arr = np.array(y)

    # LSTM neural network model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_arr, y_arr, epochs=50, verbose=0)
    results['lstm_model'] = model

    return results


def spatial_analysis_html(df):
    """
    Generates an interactive HTML map of traffic violation hotspots within Mongolia using clustering.
    Points outside Mongolia are filtered out.
    Points are clustered using K-Means (10 clusters).
    Returns the Folium Map object and saves to 'hotspots.html'.
    """
    # Filter points to Mongolia bounding box
    df_filtered = df[(df['Өргөрөг'] >= 41.6) & (df['Өргөрөг'] <= 52.1) &
                     (df['Уртраг'] >= 87.7) & (df['Уртраг'] <= 119.9)].copy()

    geometry = [Point(xy) for xy in zip(df_filtered['Уртраг'], df_filtered['Өргөрөг'])]
    gdf = gpd.GeoDataFrame(df_filtered, geometry=geometry, crs="EPSG:4326")

    coords = np.array([[geom.y, geom.x] for geom in gdf.geometry])
    kmeans = KMeans(n_clusters=10, random_state=42)
    gdf['hotspot_cluster'] = kmeans.fit_predict(coords)

    # Create interactive folium map centered on Mongolia
    m = folium.Map(location=[46.8, 103.8], zoom_start=6, tiles='CartoDB positron')

    marker_cluster = MarkerCluster(name='Зөрчил халуун цэгүүд').add_to(m)

    # Color palette for clusters (10 colors)
    color_palette = sns.color_palette("hls", 10).as_hex()

    for idx, row in gdf.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color=color_palette[row['hotspot_cluster']],
            fill=True,
            fill_color=color_palette[row['hotspot_cluster']],
            fill_opacity=0.7,
            popup=folium.Popup(
                html=f"""
                <b>Дүүрэг:</b> {row.get('Дүүргийн нэр', 'Тодорхойгүй')}<br/>
                <b>Замын хучилт:</b> {row.get('Замын хучилт', 'Тодорхойгүй')}<br/>
                <b>Зөрчил тоо:</b> {row.get('хурд хэтрүүлсэн', 'Мэдээлэлгүй')}
                """,
                max_width=300)
        ).add_to(marker_cluster)

    folium.LayerControl().add_to(m)
    m.save('hotspots.html')
    print("Interactive hotspot map saved as 'hotspots.html'")
    return m


def run_analysis(file_path):
    print("Ачаалж байна...")
    df = load_data(file_path)

    print("Тайлбар статистик шинжилгээ хийж байна...")
    desc_results = descriptive_analysis(df)

    print("Газар зүйн шинжилгээ хийж байна...")
    spatial_results = spatial_analysis_html(df)

    print("Хар цэгүүдийг тодорхойлж байна...")
    black_spots_gdf = identify_black_spots(df)
    if black_spots_gdf is not None and not black_spots_gdf.empty:
        create_black_spots_map(df, black_spots_gdf)

    print("Ирээдүйд болох хар цэгүүдийг тодорхойлж байна...")
    future_black_spots(df)

    print("Цаг хугацааны шинжилгээ хийж байна...")
    temporal_results = temporal_analysis(df)

    print("Хамаарлын шинжилгээ хийж байна...")
    corr_results = correlation_analysis(df)

    print("Урьдчилан таамаглах загварчлал хийж байна...")
    model_results = predict_future_accidents(df)

    with open("analysis_results.txt", "w", encoding="utf-8") as f:
        f.write("===== ТАЙЛБАР СТАТИСТИК ШИНЖИЛГЭЭ =====\n")
        for key, value in desc_results.items():
            f.write(f"\n{key}:\n{value}\n")

        f.write("\n===== ХАМААРЛЫН ШИНЖИЛГЭЭ =====\n")
        for key, value in corr_results.items():
            f.write(f"\n{key}:\n{value}\n")

        f.write("\n===== ЗАГВАРЧЛАЛЫН ҮР ДҮН =====\n")
        for key, value in model_results.items():
            f.write(f"\n{key}:\n{value}\n")

    print(
        "Шинжилгээ амжилттай дууслаа! Үр дүн 'analysis_results.txt' файлд хадгалагдлаа."
    )
    print(
        "Халуун цэгүүдийн интерактив газрын зураг 'hotspots.html' файлд хадгалагдлаа."
    )


if __name__ == "__main__":
    file_path = "ЗТО_2020-2024_ашиглах_final.xlsx"
    run_analysis(file_path)
