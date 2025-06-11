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
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
def load_data(file_path):
    df = pd.read_excel(file_path)
    
    # Print column names for debugging
    print("Columns in the DataFrame:", df.columns.tolist())
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Handle missing values for coordinates
    if 'Уртраг' in df.columns and 'Өргөрөг' in df.columns:
        df['Уртраг'] = df['Уртраг'].fillna(0)
        df['Өргөрөг'] = df['Өргөрөг'].fillna(0)
    else:
        print("Columns 'Уртраг' or 'Өргөрөг' not found in the DataFrame.")
    
    # Handle missing values for road width
    if 'Авто зам - Зорчих хэсгийн өргөн' in df.columns:
        # Convert the road width to numeric by extracting the average of the range
        df['Зорчих хэсгийн өргөн'] = df['Авто зам - Зорчих хэсгийн өргөн'].apply(convert_width)
        df['Зорчих хэсгийн өргөн'] = df['Зорчих хэсгийн өргөн'].fillna(df['Зорчих хэсгийн өргөн'].median())
    else:
        print("Column 'Авто зам - Зорчих хэсгийн өргөн' not found in the DataFrame.")
    
    # Convert date
    if 'Зөрчил огноо' in df.columns:
        df['Зөрчил огноо'] = pd.to_datetime(df['Зөрчил огноо'])
        df['Сар'] = df['Зөрчил огноо'].dt.month
        df['Жил'] = df['Зөрчил огноо'].dt.year
        df['Өдрийн цаг'] = df['Зөрчил огноо'].dt.hour
        df['Шөнө'] = df['Өдрийн цаг'].apply(lambda x: 1 if x >= 20 or x < 6 else 0)
    else:
        print("Column 'Зөрчил огноо' not found in the DataFrame.")
    
    # Road surface consolidation
    road_surfaces = ['Авто зам - Замын хучилт Тодорхойгүй', 
                     'Авто зам - Замын хучилт Асфальт', 
                     'Авто зам - Замын хучилт Бетон', 
                     'Авто зам - Замын хучилт Сайжруулсан', 
                     'Авто зам - Замын хучилт Хайрган', 
                     'Авто зам - Замын хучилт Хөрсөн', 
                     'Авто зам - Замын хучилт Цементэн']
    
    # Create a new column for road surface type
    df['Замын хучилт'] = df[road_surfaces].idxmax(axis=1).str.replace('Авто зам - Замын хучилт ', '')
    
    return df

def convert_width(value):
    if isinstance(value, str):
        # Check for specific known non-numeric values
        if 'ээс дээш' in value:
            return np.nan  # Return NaN for non-convertible values
        
        # Split the range and calculate the average
        parts = value.replace('м', '').split('-')
        if len(parts) == 2:
            try:
                return (float(parts[0].replace(',', '.')) + float(parts[1].replace(',', '.'))) / 2
            except ValueError:
                return np.nan  # Return NaN if conversion fails
    return np.nan  # Return NaN for non-string or unrecognized formats


# Descriptive statistics
def descriptive_analysis(df):
    results = {}
    
    # 1. Violations by district
    results['Дүүрэг'] = df['Дүүргийн нэр'].value_counts()
    
    # 2. Violations by road surface
    results['Замын хучилт'] = df['Замын хучилт'].value_counts()
    
    # 3. Violation type distribution
    violation_types = ['хурд хэтрүүлсэн', 'согтуугаар тээврийн хэрэгсэл жолоодсон']
    results['Зөрчлийн төрөл'] = df[violation_types].sum()
    
    # 4. Accident severity
    if 'Ослын ноцтой байдал' in df:
        results['Ослын ноцтой байдал'] = df['Ослын ноцтой байдал'].value_counts()
    
    # 5. Weather impact
    weather_cols = [col for col in df.columns if 'Цаг агаар' in col]
    
    # Convert weather columns to numeric, forcing errors to NaN
    for col in weather_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    results['Цаг агаар'] = df[weather_cols].sum()
    
    # 6. Road width analysis
    df['Замын өргөн бүлэг'] = pd.cut(df['Зорчих хэсгийн өргөн'], bins=5)
    results['Замын өргөн'] = df['Замын өргөн бүлэг'].value_counts()
    
    # 7. Day vs night violations
    results['Өдөр/Шөнө'] = df['Шөнө'].value_counts()
    
    # 8. Road classification
    if 'Авто зам - Замын ангилал' in df:
        results['Замын ангилал'] = df['Авто зам - Замын ангилал'].value_counts()
    
    # 9. Seasonal distribution
    season_map = {12: 'Өвөл', 1: 'Өвөл', 2: 'Өвөл', 
                  3: 'Хаврын', 4: 'Хаврын', 5: 'Хаврын',
                  6: 'Зун', 7: 'Зун', 8: 'Зун',
                  9: 'Намар', 10: 'Намар', 11: 'Намар'}
    df['Улирал'] = df['Сар'].map(season_map)
    results['Улирал'] = df['Улирал'].value_counts()
    
    # 10. Annual trends
    results['Жилийн чиг хандлага'] = df['Жил'].value_counts().sort_index()
    
    return results




# Spatial analysis
def spatial_analysis(df):
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['Уртраг'], df['Өргөрөг'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Hotspot analysis
    coords = np.array([[geom.x, geom.y] for geom in gdf.geometry])
    kmeans = KMeans(n_clusters=10, random_state=42)
    gdf['hotspot_cluster'] = kmeans.fit_predict(coords)
    
    # Plot hotspots
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111)
    
    # Load the world shapefile from local path
    world = gpd.read_file("C:\\Users\\Tamir-Laptop\\Desktop\\Autozam\\ne_110m_admin_0_countries\\ne_110m_admin_0_countries.shp")  # Update this path
    
    # Check the columns in the world GeoDataFrame
    print("Columns in the world GeoDataFrame:", world.columns.tolist())
    
    # Use the correct column name for country names
    if 'ADMIN' in world.columns:  # Commonly used column for country names
        world[world['ADMIN'] == 'Mongolia'].plot(ax=ax, color='lightgrey')
    else:
        print("Column 'ADMIN' not found in the world GeoDataFrame.")
    
    gdf.plot(ax=ax, column='hotspot_cluster', cmap='viridis', markersize=5, alpha=0.7)
    plt.title('Зөрчлийн халуун цэгүүд')
    plt.savefig('hotspots.png')
    plt.close()
    
    # District-level mapping
    district_counts = gdf.groupby('Дүүргийн нэр').size().reset_index(name='counts')
    # (Would require district shapefiles for full implementation)
    
    return gdf



# Temporal analysis
def temporal_analysis(df):
    results = {}
    
    # Monthly trends
    monthly = df.resample('M', on='Зөрчил огноо').size()
    
    # ARIMA model
    arima_model = ARIMA(monthly, order=(1,1,1))
    arima_results = arima_model.fit()
    results['arima_forecast'] = arima_results.forecast(steps=12)
    
    # SARIMA model
    sarima_model = SARIMAX(monthly, order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_results = sarima_model.fit()
    results['sarima_forecast'] = sarima_results.forecast(steps=12)
    
    # Hourly distribution
    results['Цагийн хуваарилалт'] = df.groupby('Өдрийн цаг').size()
    
    # Workday vs holiday
    df['Ажлын өдөр'] = df['Зөрчил огноо'].dt.dayofweek < 5
    results['Ажлын өдөр'] = df['Ажлын өдөр'].value_counts()
    
    # Annual growth rate
    annual = df.groupby('Жил').size()
    results['Жилийн өсөлт'] = annual.pct_change().mean()
    
    # LSTM model
    seq_length = 12
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(monthly.values.reshape(-1,1))
    
    X, y = [], []
    for i in range(len(scaled_data)-seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.array(X), np.array(y), epochs=50, verbose=0)
    results['lstm_model'] = model
    
    return results

# Correlation analysis
def correlation_analysis(df):
    results = {}
    
    # Road surface vs accident severity
    if 'Ослын ноцтой байдал' in df:
        contingency = pd.crosstab(df['Замын хучилт'], df['Ослын ноцтой байдал'])
        chi2, p, dof, ex = chi2_contingency(contingency)
        results['chi2_road_severity'] = {'chi2': chi2, 'p-value': p}
    
    # Weather vs violation types
    weather_col = [col for col in df.columns if 'Цаг агаар' in col]
    violation_col = 'хурд хэтрүүлсэн'
    if weather_col and violation_col in df:
        # Drop NaN values for correlation calculation
        valid_data = df[[weather_col[0], violation_col]].dropna()
        
        # Check the length of the arrays
        if len(valid_data[weather_col[0]]) >= 2 and len(valid_data[violation_col]) >= 2:
            corr, p_val = pearsonr(valid_data[weather_col[0]], valid_data[violation_col])
            results['weather_violation_corr'] = {'correlation': corr, 'p-value': p_val}
        else:
            results['weather_violation_corr'] = {'error': 'Insufficient data for correlation analysis'}
    
    # Road width vs violations
    if 'Зорчих хэсгийн өргөн' in df and 'хурд хэтрүүлсэн' in df:
        valid_data = df[['Зорчих хэсгийн өргөн', 'хурд хэтрүүлсэн']].dropna()
        
        # Check the length of the arrays
        if len(valid_data['Зорчих хэсгийн өргөн']) >= 2 and len(valid_data['хурд хэтрүүлсэн']) >= 2:
            corr, p_val = pearsonr(valid_data['Зорчих хэсгийн өргөн'], valid_data['хурд хэтрүүлсэн'])
            results['road_width_violation'] = {'correlation': corr, 'p-value': p_val}
        else:
            results['road_width_violation'] = {'error': 'Insufficient data for correlation analysis'}
    
    # Lighting vs night violations
    if 'замын гэрэлтүүлэг хангалтгүй' in df:
        night_violations = df[df['Шөнө'] == 1]['замын гэрэлтүүлэг хангалтгүй']
        results['lighting_night_corr'] = night_violations.mean()
    
    return results

# Predictive modeling
def predictive_modeling(df):
    results = {}
    
    # Feature engineering
    features = [
        'Замын хучилт', 
        'Зорчих хэсгийн өргөн',
        'Сар',
        'Өдрийн цаг',
        'Шөнө'
    ]
    
    # Encode categorical features
    X = pd.get_dummies(df[features], drop_first=True)
    
    # 1. Violation count prediction (Regression)
    if 'хурд хэтрүүлсэн' in df:
        y_count = df['хурд хэтрүүлсэн']
        X_train, X_test, y_train, y_test = train_test_split(X, y_count, test_size=0.2, random_state=42)
        
        # Random Forest
        rf_reg = RandomForestRegressor(n_estimators=100)
        rf_reg.fit(X_train, y_train)
        y_pred = rf_reg.predict(X_test)
        results['rf_mse'] = mean_squared_error(y_test, y_pred)
        
        # XGBoost
        xgb_reg = XGBRegressor()
        xgb_reg.fit(X_train, y_train)
        y_pred = xgb_reg.predict(X_test)
        results['xgb_mse'] = mean_squared_error(y_test, y_pred)
    
    # 2. Accident severity prediction (Classification)
    if 'Ослын ноцтой байдал' in df:
        y_severity = df['Ослын ноцтой байдал']
        X_train, X_test, y_train, y_test = train_test_split(X, y_severity, test_size=0.2, random_state=42)
        
        # Random Forest
        rf_clf = RandomForestClassifier(n_estimators=100)
        rf_clf.fit(X_train, y_train)
        y_pred = rf_clf.predict(X_test)
        results['rf_class_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        # XGBoost
        xgb_clf = XGBClassifier()
        xgb_clf.fit(X_train, y_train)
        y_pred = xgb_clf.predict(X_test)
        results['xgb_class_report'] = classification_report(y_test, y_pred, output_dict=True)
    
    # 3. Clustering analysis
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['violation_cluster'] = kmeans.fit_predict(X)
    results['clusters'] = df['violation_cluster'].value_counts()
    
    return results

# Main analysis function
def run_analysis(file_path):
    print("Ачаалж байна...")
    df = load_data(file_path)
    
    print("Тайлбар статистик шинжилгээ хийж байна...")
    desc_results = descriptive_analysis(df)
    
    print("Газар зүйн шинжилгээ хийж байна...")
    spatial_results = spatial_analysis(df)
    
    print("Цаг хугацааны шинжилгээ хийж байна...")
    temporal_results = temporal_analysis(df)
    
    print("Хамаарлын шинжилгээ хийж байна...")
    corr_results = correlation_analysis(df)
    

          
    print("Урьдчилан таамаглах загварчлал хийж байна...")
    model_results = predictive_modeling(df)
    
    # Save results to files with UTF-8 encoding
    with open('analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("===== ТАЙЛБАР СТАТИСТИК ШИНЖИЛГЭЭ =====\n")
        for key, value in desc_results.items():
            f.write(f"\n{key}:\n{value}\n")
        
        f.write("\n===== ХАМААРЛЫН ШИНЖИЛГЭЭ =====\n")
        for key, value in corr_results.items():
            f.write(f"\n{key}:\n{value}\n")
        
        f.write("\n===== ЗАГВАРЧЛАЛЫН ҮР ДҮН =====\n")
        for key, value in model_results.items():
            f.write(f"\n{key}:\n{value}\n")
    
    print("Шинжилгээ амжилттай дууслаа! Үр дүн 'analysis_results.txt' файлд хадгалагдлаа.")
    print("Халуун цэгүүдийн газрын зураг 'hotspots.png' файлд хадгалагдлаа.")

    # Additional Analyses
    # 1. Monthly Trends Visualization
    plt.figure(figsize=(12, 6))
    monthly_counts = df.resample('M', on='Зөрчил огноо').size()
    monthly_counts.plot()
    plt.title('Traffic Violations Monthly Trends')
    plt.xlabel('Month')
    plt.ylabel('Number of Violations')
    plt.grid()
    plt.savefig('monthly_trends.png')
    plt.close()

    # 2. Yearly Comparison
    plt.figure(figsize=(12, 6))
    yearly_counts = df['Жил'].value_counts().sort_index()
    yearly_counts.plot(kind='bar')
    plt.title('Yearly Traffic Violations Comparison')
    plt.xlabel('Year')
    plt.ylabel('Number of Violations')
    plt.grid()
    plt.savefig('yearly_comparison.png')
    plt.close()

    # 3. Heatmap of Violations by District
    plt.figure(figsize=(12, 10))
    district_counts = df['Дүүргийн нэр'].value_counts()
    sns.heatmap(district_counts.values.reshape(-1, 1), annot=True, fmt='d', cmap='YlGnBu', 
                yticklabels=district_counts.index)
    plt.title('Heatmap of Violations by District')
    plt.xlabel('Count of Violations')
    plt.ylabel('Districts')
    plt.savefig('district_heatmap.png')
    plt.close()

    # 4. Clustering Analysis Visualization
    plt.figure(figsize=(12, 10))
    sns.countplot(data=df, x='violation_cluster')
    plt.title('Clustering of Violations')
    plt.xlabel('Cluster')
    plt.ylabel('Count of Violations')
    plt.savefig('clustering_analysis.png')
    plt.close()

if __name__ == "__main__":
    file_path = "ЗТО_2020-2024_ашиглах_final.xlsx"
    run_analysis(file_path)
