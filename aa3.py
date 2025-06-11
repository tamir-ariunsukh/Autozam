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
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score, f1_score
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
    """Load and preprocess traffic violation data"""
    df = pd.read_excel(file_path)
    print("Columns in the DataFrame:", df.columns.tolist())
    
    # Clean column names and handle missing geodata
    df.columns = df.columns.str.strip()
    df['Уртраг'] = df.get('Уртраг', pd.Series()).fillna(0)
    df['Өргөрөг'] = df.get('Өргөрөг', pd.Series()).fillna(0)

    # Process road width data
    if 'Авто зам - Зорчих хэсгийн өргөн' in df.columns:
        df['Зорчих хэсгийн өргөн'] = df['Авто зам - Зорчих хэсгийн өргөн'].apply(convert_width)
        # Use median imputation only if valid values exist
        if not df['Зорчих хэсгийн өргөн'].empty:
            median_val = df['Зорчих хэсгийн өргөн'].median(skipna=True)
            df['Зорчих хэсгийн өргөн'].fillna(median_val, inplace=True)

    # Process datetime features
    if 'Зөрчил огноо' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Зөрчил огноо']):
        df['Зөрчил огноо'] = pd.to_datetime(df['Зөрчил огноо'])
        df['Сар'] = df['Зөрчил огноо'].dt.month
        df['Жил'] = df['Зөрчил огноо'].dt.year
        df['Өдрийн цаг'] = df['Зөрчил огноо'].dt.hour
        df['Шөнө'] = df['Өдрийн цаг'].apply(lambda x: 1 if x >= 20 or x < 6 else 0)
    else:
        print("Warning: 'Зөрчил огноо' column missing or invalid")

    # Process road surface types
    road_surfaces = ['Авто зам - Замын хучилт Тодорхойгүй',
                     'Авто зам - Замын хучилт Асфальт',
                     'Авто зам - Замын хучилт Бетон',
                     'Авто зам - Замын хучилт Сайжруулсан',
                     'Авто зам - Замын хучилт Хайрган',
                     'Авто зам - Замын хучилт Хөрсөн',
                     'Авто зам - Замын хучилт Цементэн']
    
    # Only filter if columns exist
    existing_surfaces = [col for col in road_surfaces if col in df.columns]
    if existing_surfaces:
        df['Замын хучилт'] = df[existing_surfaces].idxmax(axis=1).str.replace('Авто зам - Замын хучилт ', '')
    else:
        df['Замын хучилт'] = 'Тодорхойгүй'
    
    return df


def convert_width(value):
    """Convert road width string to numeric value"""
    if isinstance(value, str):
        if 'ээс дээш' in value:  # Handle "15-ээс дээш" format
            return np.nan
        parts = value.replace('м', '').split('-')
        if len(parts) == 2:
            try:
                # Handle both comma and dot decimal formats
                start = float(parts[0].replace(',', '.').strip())
                end = float(parts[1].replace(',', '.').strip())
                return (start + end) / 2
            except ValueError:
                return np.nan
    return np.nan


def descriptive_analysis(df):
    """Perform descriptive statistical analysis"""
    results = {}
    
    # Safely count values only if columns exist
    if 'Дүүргийн нэр' in df:
        results['Дүүрэг'] = df['Дүүргийн нэр'].value_counts()
    
    if 'Замын хучилт' in df:
        results['Замын хучилт'] = df['Замын хучилт'].value_counts()

    # Handle violation types
    violation_types = ['хурд хэтрүүлсэн', 'согтуугаар тээврийн хэрэгсэл жолоодсон']
    existing_violations = [v for v in violation_types if v in df.columns]
    if existing_violations:
        results['Зөрчлийн төрөл'] = df[existing_violations].sum()

    if 'Ослын ноцтой байдал' in df:
        results['Ослын ноцтой байдал'] = df['Ослын ноцтой байдал'].value_counts()

    # Process weather data
    weather_cols = [col for col in df.columns if 'Цаг агаар' in col]
    if weather_cols:
        for col in weather_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        results['Цаг агаар'] = df[weather_cols].sum()
    else:
        results['Цаг агаар'] = pd.Series()

    # Process road width categories
    if 'Зорчих хэсгийн өргөн' in df:
        # Handle NaN before binning
        road_width = df['Зорчих хэсгийн өргөн'].dropna()
        if not road_width.empty:
            df['Замын өргөн бүлэг'] = pd.cut(road_width, bins=5)
            results['Замын өргөн'] = df['Замын өргөн бүлэг'].value_counts()

    if 'Шөнө' in df:
        results['Өдөр/Шөнө'] = df['Шөнө'].value_counts()

    if 'Авто зам - Замын ангилал' in df:
        results['Замын ангилал'] = df['Авто зам - Замын ангилал'].value_counts()

    # Map months to seasons
    season_map = {
        12: 'Өвөл', 1: 'Өвөл', 2: 'Өвөл',
        3: 'Хаврын', 4: 'Хаврын', 5: 'Хаврын',
        6: 'Зун', 7: 'Зун', 8: 'Зун',
        9: 'Намар', 10: 'Намар', 11: 'Намар'
    }
    if 'Сар' in df:
        df['Улирал'] = df['Сар'].map(season_map)
        results['Улирал'] = df['Улирал'].value_counts()
    
    if 'Жил' in df:
        results['Жилийн чиг хандлага'] = df['Жил'].value_counts().sort_index()

    return results


def filter_mongolia_points(df):
    """Filter and format geographic points within Mongolia"""
    # Filter points within Mongolia's bounding box
    df_filtered = df[
        (df['Өргөрөг'] >= 41.6) & (df['Өргөрөг'] <= 52.1) &
        (df['Уртраг'] >= 87.7) & (df['Уртраг'] <= 119.9)
    ].copy()

    # Safely fill missing values with safer defaults
    df_filtered['Газар /Хэлтэс/'] = df_filtered.get('Газар /Хэлтэс/', 'Тодорхойгүй').fillna('Тодорхойгүй')
    df_filtered['Зөрчил гарсан газрын хаяг'] = df_filtered.get('Зөрчил гарсан газрын хаяг', 'Тодорхойгүй').fillna('Тодорхойгүй')
    df_filtered['Замын байршил /тайлбар/'] = df_filtered.get('Замын байршил /тайлбар/', 'Тодорхойгүй').fillna('Тодорхойгүй')
    df_filtered['Газар'] = df_filtered.get('Газар', 'Тодорхойгүй').fillna('Тодорхойгүй')

    # Handle datetime conversion safely
    if 'Зөрчил огноо' in df_filtered:
        df_filtered['Зөрчил огноо'] = pd.to_datetime(df_filtered['Зөрчил огноо'], errors='coerce')
    
    return df_filtered


def identify_black_spots(df):
    """Identify high-risk accident locations based on spatial clusters"""
    # First filter to Mongolia
    df = filter_mongolia_points(df)
    geometry = [Point(xy) for xy in zip(df['Уртраг'], df['Өргөрөг'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Buffer distances (approx: 25m ~ 0.000225°, 2500m ~ 0.0225°)
    district_buffer = 0.000225
    province_buffer = 0.0225
    
    black_spots_records = []
    sindex = gdf.sindex

    for idx, row in gdf.iterrows():
        point = row.geometry
        
        # Determine buffer based on administrative level
        # FIXED: Using proper admin name checks instead of ==1
        if 'Дүүргийн нэр' in row and pd.notna(row['Дүүргийн нэр']):
            buffer_dist = district_buffer
            threshold = 3
            area_type = "Дүүрэг"
        elif 'Аймгийн нэр' in row and pd.notna(row['Аймгийн нэр']):
            buffer_dist = province_buffer
            threshold = 2
            area_type = "Аймаг"
        else:
            continue  # Skip if no admin level identified
            
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
                "area_type": area_type,
                "black_spot_id": len(black_spots_records) + 1,
                "related_records": precise_matches.index.tolist()
            }
            black_spots_records.append(spot_info)

    if black_spots_records:
        black_spots_gdf = gpd.GeoDataFrame(black_spots_records)
        black_spots_gdf.set_geometry('geometry', inplace=True)
        black_spots_gdf.crs = "EPSG:4326"
        return black_spots_gdf
    return gpd.GeoDataFrame()  # Return empty GeoDataFrame


# Enhanced map visualization functions
def create_black_spots_map(df, black_spots_gdf):
    """Generate professional interactive folium map"""
    # Create base map with proper styling
    m = folium.Map(location=[47.92, 106.9], zoom_start=12, tiles='cartodbpositron', 
                  control_scale=True, prefer_canvas=True)
    
    # Add tile layer options
    folium.TileLayer('openstreetmap').add_to(m)
    folium.TileLayer('stamenterrain').add_to(m)
    
    # Add accident points with clustering
    if not df.empty:
        accident_group = folium.FeatureGroup(name='Ослын цэгүүд', show=True)
        
        for idx, row in df.iterrows():
            popup_content = f"""
            <b>Дүүрэг:</b> {row.get('Дүүргийн нэр', 'Тодорхойгүй')}<br>
            <b>Огноо:</b> {row.get('Зөрчил огноо', 'Тодорхойгүй')}<br>
            <b>Газар:</b> {row.get('Газар /Хэлтэс/', 'Тодорхойгүй')}
            """
            
            folium.CircleMarker(
                location=[row['Өргөрөг'], row['Уртраг']],
                radius=3,
                color='#1f77b4',
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(accident_group)
        
        accident_group.add_to(m)

    # Add black spots with proper styling
    if not black_spots_gdf.empty:
        blackspot_group = folium.FeatureGroup(name='Хар цэгүүд', show=True)
        
        for _, spot in black_spots_gdf.iterrows():
            folium.CircleMarker(
                location=[spot.geometry.y, spot.geometry.x],
                radius=6 + min(spot['accident_count']/2, 15),
                color='#d62728',
                fill=True,
                fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>Хар цэг ID:</b> {spot.black_spot_id}<br>"
                    f"<b>Ослын тоо:</b> {spot.accident_count}<br>"
                    f"<b>Төрөл:</b> {spot.area_type}",
                    max_width=300)
            ).add_to(blackspot_group)
        
        blackspot_group.add_to(m)

    # Add layer control and minimap
    folium.LayerControl(collapsed=False).add_to(m)
    folium.plugins.MiniMap(toggle_display=True).add_to(m)
    
    # Add fullscreen button
    folium.plugins.Fullscreen(
        position='topright',
        title='Бүтэн дэлгэц',
        title_cancel='Гарах',
        force_separate_button=True
    ).add_to(m)
    
    # Add measure control
    folium.plugins.MeasureControl(
        position='bottomleft',
        primary_length_unit='meters',
        secondary_length_unit='kilometers'
    ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background: white; padding: 10px; border: 1px solid grey;
                border-radius: 5px; font-size: 14px;">
        <h4 style="margin-top:0; margin-bottom:5px;">Тэмдэглэгээ</h4>
        <div><i style="background: #1f77b4; width: 12px; height: 12px; 
                      border-radius: 50%; display: inline-block;"></i> Ослын цэг</div>
        <div><i style="background: #d62728; width: 12px; height: 12px; 
                      border-radius: 50%; display: inline-block;"></i> Хар цэг</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save and return map
    m.save('black_spots_map.html')
    print("Хар цэгүүдийн газрын зураг 'black_spots_map.html' файлд хадгалагдлаа")
    return m

def future_black_spots(df):
    """Create professional future risk spots map"""
    # Filter to Mongolia
    df = df[
        (df['Өргөрөг'] >= 41.5) & (df['Өргөрөг'] <= 52.2) &
        (df['Уртраг'] >= 87.5) & (df['Уртраг'] <= 120.5)
    ].copy()
    
    geometry = [Point(xy) for xy in zip(df['Уртраг'], df['Өргөрөг'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Cluster analysis
    n_clusters = min(20, max(5, len(df)//100))
    if n_clusters > 1:
        coords = np.array([[row['Өргөрөг'], row['Уртраг']] for _, row in gdf.iterrows()])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        gdf['risk_cluster'] = kmeans.fit_predict(coords)
        
        cluster_counts = gdf['risk_cluster'].value_counts()
        mean_count = cluster_counts.mean()
        risk_clusters = cluster_counts[
            (cluster_counts > mean_count * 0.8) & 
            (cluster_counts < mean_count * 1.5)
        ].index.tolist()
        
        future_spots = gdf[gdf['risk_cluster'].isin(risk_clusters)].copy()
        
        # Create professional map
        m = folium.Map(location=[47.92, 106.9], zoom_start=12, 
                      tiles='cartodbpositron', control_scale=True)
        
        # Add tile layers
        folium.TileLayer('openstreetmap').add_to(m)
        folium.TileLayer('stamenterrain').add_to(m)
        
        if not future_spots.empty:
            # Add cluster markers with tooltips
            for _, row in future_spots.iterrows():
                folium.CircleMarker(
                    location=[row['Өргөрөг'], row['Уртраг']],
                    radius=6,
                    color='#ff7f0e',
                    fill=True,
                    fill_opacity=0.7,
                    tooltip=f"Ирээдүйн эрсдэлтэй цэг",
                    popup=folium.Popup(
                        f"<b>Кластер ID:</b> {row['risk_cluster']}<br>"
                        f"<b>Дүүрэг:</b> {row.get('Дүүргийн нэр', 'Тодорхойгүй')}<br>"
                        f"<b>Огноо:</b> {row.get('Зөрчил огноо', 'Тодорхойгүй')}",
                        max_width=300
                    )
                ).add_to(m)
            
            # Add heatmap for density visualization
            heat_data = [[row['Өргөрөг'], row['Уртраг']] for _, row in future_spots.iterrows()]
            folium.plugins.HeatMap(
                heat_data, 
                name='Ослын нягтрал',
                radius=15,
                blur=10,
                min_opacity=0.3
            ).add_to(m)
            
            # Add cluster centroids
            centroids = kmeans.cluster_centers_
            for i, centroid in enumerate(centroids):
                if i in risk_clusters:
                    folium.Marker(
                        location=[centroid[0], centroid[1]],
                        icon=folium.Icon(icon='exclamation-triangle', color='red', prefix='fa'),
                        tooltip=f"Кластер {i} төв",
                        popup=f"<b>Ирээдүйн эрсдэлтэй төв цэг</b><br>Кластер ID: {i}"
                    ).add_to(m)
        
        # Add controls
        folium.LayerControl(collapsed=False).add_to(m)
        folium.plugins.MiniMap().add_to(m)
        folium.plugins.Fullscreen().add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                    background: white; padding: 10px; border: 1px solid grey;
                    border-radius: 5px; font-size: 14px;">
            <h4 style="margin-top:0; margin-bottom:5px;">Тэмдэглэгээ</h4>
            <div><i style="background: #ff7f0e; width: 12px; height: 12px; 
                          border-radius: 50%; display: inline-block;"></i> Ирээдүйн эрсдэлтэй цэг</div>
            <div><i class="fa fa-exclamation-triangle" style="color: red;"></i> Кластерын төв</div>
            <div><i style="background: rgba(255,0,0,0.3); width: 12px; height: 12px; 
                          border-radius: 50%; display: inline-block;"></i> Ослын нягтрал</div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        m.save('future_black_spots.html')
        print("Ирээдүйн эрсдэлтэй цэгүүдийн газрын зураг 'future_black_spots.html' файлд хадгалагдлаа")
        return m
    return None




def spatial_analysis_html(df):
    """Create a visually appealing spatial analysis visualization"""
    df_filtered = filter_mongolia_points(df)
    if df_filtered.empty:
        print("No valid geolocated data for spatial analysis")
        return None
    
    geometry = [Point(xy) for xy in zip(df_filtered['Уртраг'], df_filtered['Өргөрөг'])]
    gdf = gpd.GeoDataFrame(df_filtered, geometry=geometry, crs="EPSG:4326")
    
    # Create spatial clusters
    n_clusters = 12
    coords = np.array([[geom.y, geom.x] for geom in gdf.geometry])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    gdf['hotspot_cluster'] = kmeans.fit_predict(coords)
    
    # Create interactive map with professional design
    m = folium.Map(location=[47.92, 106.9], zoom_start=12, 
                  tiles='cartodbpositron', 
                  control_scale=True, 
                  prefer_canvas=True)
    
    # Add tile layer options
    folium.TileLayer('openstreetmap').add_to(m)
    folium.TileLayer('stamenterrain').add_to(m)
    
    # Create cluster analysis dashboard
    dashboard_html = """
    <div style="position: fixed; top: 50px; left: 20px; z-index: 1000; 
                background: #fff; padding: 20px; border-radius: 8px; 
                box-shadow: 0 4px 18px rgba(0,0,0,0.1); width: 300px;
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <h2 style="color: #1a1a1a; font-size: 22px; margin-top: 0; 
                  border-bottom: 1px solid #f0f0f0; padding-bottom: 12px;
                  font-weight: 600;">
            Ослын нягтралын шинжилгээ
        </h2>
        
        <div style="margin-bottom: 16px; color: #6b7280; font-size: 16px;">
            Кластер бүр дараах цэгүүдийг агуулна:
        </div>
        
        <div style="display: grid; grid-template-columns: auto 1fr; 
                   gap: 10px; margin-bottom: 20px;">
            <div style="color: #4e79a7;">▉</div>
            <div>1-10 цэг</div>
            <div style="color: #f28e2c;">▉</div>
            <div>11-20 цэг</div>
            <div style="color: #e15759;">▉</div>
            <div>21-30 цэг</div>
            <div style="color: #59a14f;">▉</div>
            <div>31+ цэг</div>
        </div>
        
        <div style="color: #555; font-size: 14px; margin-top: 15px;
                  border-top: 1px solid #f0f0f0; padding-top: 15px;">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="background: rgba(0,0,0,0.1); border-radius: 50%; 
                          width: 32px; height: 32px; display: flex; align-items: center;
                          justify-content: center; margin-right: 12px;">!</div>
                <div>Цэг дээр дарж дэлгэрэнгүй мэдээлэл авах</div>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="background: rgba(0,0,0,0.1); border-radius: 50%; 
                          width: 32px; height: 32px; display: flex; align-items: center;
                          justify-content: center; margin-right: 12px;">⊕</div>
                <div>Баруун дээд буланд зохимжтой байдлаар нуман</div>
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(dashboard_html))
    
    # Add clustering heatmap for density visualization (simplified without gradient)
    heat_data = [[row['Өргөрөг'], row['Уртраг']] for _, row in gdf.iterrows()]
    folium.plugins.HeatMap(
        heat_data, 
        name='Ослын нягтрал',
        radius=15,
        blur=18,
        min_opacity=0.4
    ).add_to(m)
    
    # Create color palette for clusters
    palette = ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f', 
              '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab']
    
    # Add cluster centers
    centroids = kmeans.cluster_centers_
    for i, centroid in enumerate(centroids):
        cluster_points = gdf[gdf['hotspot_cluster'] == i]
        cluster_size = len(cluster_points)
        color_idx = min(cluster_size // 10, len(palette) - 1)
        
        # Show only clusters with more than 1 point
        if cluster_size > 1:
            folium.Circle(
                location=[centroid[0], centroid[1]],
                radius=cluster_size * 30,  # Scale circle size by cluster size
                color=palette[color_idx],
                fill=True,
                fill_opacity=0.2,
                popup=folium.Popup(
                    f"<b>Кластер #{i+1}</b><br>"
                    f"Ослын тоо: {cluster_size}<br>"
                    f"Нягтрал: {cluster_size / gdf.shape[0] * 100:.1f}%",
                    max_width=300
                )
            ).add_to(m)
    
    # Add interactive layer control
    folium.LayerControl(collapsed=False, position='bottomright').add_to(m)
    
    # Add minimap
    folium.plugins.MiniMap(
        tile_layer='cartodbpositron',
        position='bottomleft',
        toggle_display=True
    ).add_to(m)
    
    # Add fullscreen control
    folium.plugins.Fullscreen(
        position='topright',
        title='Бүтэн дэлгэц',
        title_cancel='Гарах',
        force_separate_button=True
    ).add_to(m)
    
    # Add reset view button
    folium.plugins.LocateControl(
        position='topright',
        strings={"title": "Одоохун газар зүй дээрээ харах"},
        show_accuracy=True
    ).add_to(m)
    
    # Add mouse position display
    fmtr = "function(num) {return L.Util.formatNum(num, 5) + '° ';};"
    folium.plugins.MousePosition(
        position='bottomright',
        separator=' | ',
        empty_string='Хоосон',
        lng_first=True,
        num_digits=5,
        prefix='Байршил:',
        lat_formatter=fmtr,
        lng_formatter=fmtr,
    ).add_to(m)
    
    # Save interactive visualization
    m.save('spatial_analysis.html')
    print("Газарзүйн шинжилгээний интерактив дэлгэц 'spatial_analysis.html' файлд хадгалагдлаа")
    
    # Simplify cluster data for dashboard visualization
    insights = []
    for i in range(n_clusters):
        cluster_subset = gdf[gdf['hotspot_cluster'] == i]
        if len(cluster_subset) > 0:
            if 'Дүүргийн нэр' in cluster_subset.columns and not cluster_subset['Дүүргийн нэр'].dropna().empty:
                top_district = cluster_subset['Дүүргийн нэр'].value_counts().index[0]
            else:
                top_district = 'Тодорхойгүй'

            avg_hour = cluster_subset['Өдрийн цаг'].mean() if 'Өдрийн цаг' in cluster_subset else 0
            common_cause = 'хурд хэтрүүлсэн' if 'хурд хэтрүүлсэн' in df.columns else 'Тодорхойгүй'
            
            insights.append({
                'cluster_id': i,
                'count': len(cluster_subset),
                'top_district': top_district,
                'avg_time': f"{(avg_hour//60):02}:{(avg_hour%60):02}" if avg_hour > 0 else "Тодорхойгүй",
                'common_cause': common_cause
            })
    
    # Sort clusters by count
    insights_sorted = sorted(insights, key=lambda x: x['count'], reverse=True)
    
    print("\nӨндөр эрсдэлтэй бүсүүдийн дүн шинжилгээ:")
    print("========================================")
    for i, insight in enumerate(insights_sorted[:5]):
        print(f"{i+1}. Кластер #{insight['cluster_id']}:")
        print(f"   - Ослын тоо: {insight['count']}")
        print(f"   - Төв дүүрэг: {insight['top_district']}")
        print(f"   - Дундаж цаг: {insight['avg_time']}")
        print(f"   - Түгээмэл шалтгаан: {insight['common_cause']}")
        print("----------------------------------------")
    
    return m


def temporal_analysis(df):
    """
    Analyze temporal patterns of accidents using statistical models
    FIXED: Single enhanced implementation
    """
    results = {}
    
    # Handle datetime preprocessing
    if 'Зөрчил огноо' not in df or not pd.api.types.is_datetime64_any_dtype(df['Зөрчил огноо']):
        return results
        
    # Create time series at monthly level
    monthly = df.resample('M', on='Зөрчил огноо').size()
    
    # Time series forecasting
    try:
        arima_model = ARIMA(monthly, order=(1, 1, 1))
        arima_results = arima_model.fit()
        results['arima_forecast'] = arima_results.forecast(steps=12).tolist()
    except Exception as e:
        print(f"ARIMA modeling failed: {e}")
    
    try:
        sarima_model = SARIMAX(monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_results = sarima_model.fit()
        results['sarima_forecast'] = sarima_results.forecast(steps=12).tolist()
    except Exception as e:
        print(f"SARIMAX modeling failed: {e}")
    
    # Hourly distribution analysis
    if 'Өдрийн цаг' in df:
        results['Цагийн хуваарилалт'] = df['Өдрийн цаг'].value_counts().sort_index().to_dict()
    
    # Weekday/weekend analysis
    df['Ажлын өдөр'] = df['Зөрчил огноо'].dt.dayofweek < 5
    results['Ажлын өдөр'] = df['Ажлын өдөр'].value_counts().to_dict()
    
    # Annual trends
    if 'Жил' in df:
        annual = df['Жил'].value_counts().sort_index()
        results['Жилийн өсөлт'] = annual.pct_change().mean()
    
    # LSTM forecasting
    seq_length = 12
    if len(monthly) > seq_length + 1:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(monthly.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:i+seq_length])
            y.append(scaled_data[i+seq_length])
        
        X_arr = np.array(X)
        y_arr = np.array(y)
        
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_length, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_arr, y_arr, epochs=50, verbose=0)
        results['lstm_forecast'] = model.predict(X_arr[-1:]).flatten().tolist()
    
    return results


def correlation_analysis(df):
    """Calculate statistical correlations between accident factors"""
    results = {}
    
    # Quick calculations between features and severity
    if 'Ослын ноцтой байдал' in df:
        severity_columns = [col for col in df.columns if 'Ослын ноцтой байдал' != col]
        for col in severity_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    corr, p_val = pearsonr(df[col].fillna(0), df['Ослын ноцтой байдал'])
                    results[f"{col}_severity_corr"] = {'pearson_r': corr, 'p_value': float(p_val)}
                
                except Exception:
                    # Handle small samples or constant data
                    results[f"{col}_severity_corr"] = {'error': 'Cannot compute correlation'}
    
    # More specific analysis cases
    if 'Замын хучилт' in df and 'Ослын ноцтой байдал' in df:
        try:
            contingency = pd.crosstab(df['Замын хучилт'], df['Ослын ноцтой байдал'])
            chi2, p, dof, ex = chi2_contingency(contingency)
            results['road_severity_chi2'] = {
                'chi2': chi2,
                'p_value': p,
                'df': dof
            }
        except Exception as e:
            results['road_severity_chi2'] = {'error': str(e)}
    
    # Analyze correlation between violations and road width
    if 'Зорчих хэсгийн өргөн' in df and 'хурд хэтрүүлсэн' in df:
        try:
            corr, p_val = pearsonr(
                df['Зорчих хэсгийн өргөн'].fillna(0), 
                df['хурд хэтрүүлсэн'].fillna(0)
            )
            results['road_width_violation'] = {'pearson_r': corr, 'p_value': float(p_val)}
        except Exception:
            pass
    
    # Analysis of night incidents and lighting
    if 'замын гэрэлтүүлэг хангалтгүй' in df and 'Шөнө' in df:
        try:
            night_data = df[df['Шөнө'] == 1]['замын гэрэлтүүлэг хангалтгүй']
            results['night_lighting_mean'] = float(night_data.mean())
        except Exception:
            pass
    
    return results

# --- Predict Future Accidents ---

def predict_future_accidents(df):
    """
    Train RandomForest and XGBoost classifiers to predict severe accidents.
    Returns accuracy and F1-score for each model.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from xgboost import XGBClassifier

    results = {}

    if 'Ослын ноцтой байдал' not in df:
        print("Ослын ноцтой байдал багана байхгүй байна.")
        return results

    target = 'Ослын ноцтой байдал'
    features = [
        'Шөнө', 'Өдрийн цаг', 'Сар', 'Жил', 'хурд хэтрүүлсэн',
        'согтуугаар тээврийн хэрэгсэл жолоодсон', 'Авто зам - Зорчих хэсгийн өргөн'
    ]
    features = [f for f in features if f in df.columns]

    df_model = df[features + [target]].dropna()
    if df_model.empty:
        print("Загварын сургалтанд ашиглах өгөгдөл байхгүй.")
        return results

    X = df_model[features]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['random_forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'f1_score': f1_score(y_test, y_pred_rf, average='macro')
    }

    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results['xgboost'] = {
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'f1_score': f1_score(y_test, y_pred_xgb, average='macro')
    }

    return results

def run_analysis(file_path):
    """Main function to execute all analysis steps"""
    print("Ачаалж байна...")
    df = load_data(file_path)
    
    # Results container for all analyses
    all_results = {
        'descriptive': {},
        'spatial': None,
        'black_spots': None,
        'temporal': {},
        'correlation': {},
        'predictive': {}
    }

    print("Тайлбар статистик шинжилгээ хийж байна...")
    all_results['descriptive'] = descriptive_analysis(df)
    
    print("Газар зүйн шинжилгээ хийж байна...")
    all_results['spatial'] = spatial_analysis_html(df)
    
    print("Хар цэгүүдийг тодорхойлж байна...")
    black_spots_gdf = identify_black_spots(df)
    all_results['black_spots'] = black_spots_gdf
    
    if not black_spots_gdf.empty:
        create_black_spots_map(df, black_spots_gdf)
    else:
        print("No black spots identified")
    
    print("Ирээдүйн боломжит хар цэгүүдийг тодорхойлж байна...")
    future_black_spots(df)
    
    print("Цаг хугацааны шинжилгээ хийж байна...")
    all_results['temporal'] = temporal_analysis(df)
    
    print("Хамаарлын шинжилгээ хийж байна...")
    all_results['correlation'] = correlation_analysis(df)
    
    print("Урьдчилан таамаглах загварчлал хийж байна...")
    all_results['predictive'] = predict_future_accidents(df)
    
    # Save results in JSON format for better interoperability
    import json
    from datetime import datetime
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open("analysis_results.json", "w", encoding="utf-8") as f:
        serializable_results = json.loads(json.dumps(all_results, default=convert_to_serializable))
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': serializable_results
        }, f, ensure_ascii=False, indent=2)
    
    print("Шинжилгээ амжилттай дууслаа! Үр дүн 'analysis_results.json' файлд хадгалагдлаа")
    print("Халуун цэгүүдийн интерактив газрын зураг 'hotspots.html' файлд хадгалагдлаа")


if __name__ == "__main__":
    file_path = "ЗТО_2020-2024_ашиглах_final.xlsx"
    run_analysis(file_path)
