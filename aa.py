# --- ПРОФЕССИОНАЛ ML + ГАЗРЫН ЗУРАГТАЙ ХАР ЦЭГИЙН ШИНЖИЛГЭЭ ---
# Tamir + ChatGPT 2025-06-08
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import folium
from folium.plugins import MarkerCluster, HeatMap, MiniMap, Fullscreen, MeasureControl, LocateControl, MousePosition
import warnings
warnings.filterwarnings('ignore')

# 1. Өгөгдөл ачаалж, цэвэрлэх

def load_and_prep(file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    df['Уртраг'] = df.get('Уртраг', pd.Series()).fillna(0)
    df['Өргөрөг'] = df.get('Өргөрөг', pd.Series()).fillna(0)
    # Зөрчил огноо, цаг, гариг
    if 'Зөрчил огноо' in df:
        df['Зөрчил огноо'] = pd.to_datetime(df['Зөрчил огноо'], errors='coerce')
        df['Сар'] = df['Зөрчил огноо'].dt.month
        df['Жил'] = df['Зөрчил огноо'].dt.year
  
        df['Гариг'] = df['Зөрчил огноо'].dt.dayofweek
    # Замын хучилтын төрлүүдийн нэг багана
    surf_cols = [c for c in df.columns if 'хучилт' in c]
    if surf_cols:
        df['Хучилт'] = df[surf_cols].idxmax(axis=1).str.replace('Авто зам - Замын хучилт ', '')
    else:
        df['Хучилт'] = 'Тодорхойгүй'
    # Монгол координат filter
    df = df[(df['Өргөрөг'] > 41) & (df['Өргөрөг'] < 52.2) & (df['Уртраг'] > 87) & (df['Уртраг'] < 120)]
    df = df.copy().reset_index(drop=True)
    return df

# 2. Хар цэг (spatial clustering)
def blackspot_ml_cluster(df, n_clusters=12):
    gdf = gpd.GeoDataFrame(df, geometry=[Point(x, y) for y, x in zip(df['Өргөрөг'], df['Уртраг'])], crs='EPSG:4326')
    coords = np.array([[y, x] for y, x in zip(df['Өргөрөг'], df['Уртраг'])])
    km = KMeans(n_clusters=n_clusters, random_state=1, n_init=12)
    gdf['cluster'] = km.fit_predict(coords)
    centers = km.cluster_centers_
    # Хар цэгийг том, дунд нягтралаар тодорхойлох
    cluster_sizes = gdf['cluster'].value_counts()
    top = cluster_sizes.nlargest(min(10, n_clusters)).index
    gdf['is_blackspot'] = gdf['cluster'].isin(top)
    return gdf, centers, cluster_sizes

# 3. Машин сургалт (зуршлийн урьдчилсан таамаг)
def train_rf_predict(df):
    features = ['Сар', 'Жил', 'Гариг']
    if 'хурд хэтрүүлсэн' in df.columns:
        features.append('хурд хэтрүүлсэн')
    if 'согтуугаар тээврийн хэрэгсэл жолоодсон' in df.columns:
        features.append('согтуугаар тээврийн хэрэгсэл жолоодсон')
    if 'Авто зам - Ослын ноцтой байдал' not in df:
        print('Ослын ноцтой байдал багана алга')
        return None
    sub = df[features + ['Авто зам - Ослын ноцтой байдал']].dropna()
    if sub.empty: return None
    X = sub[features]
    y = sub['Авто зам - Ослын ноцтой байдал']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    clf = RandomForestClassifier(n_estimators=80, random_state=1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    print(f'RF таамаглал (acc={acc:.3f}, f1={f1:.3f})')
    return clf, acc, f1

# 4. Интерактив газрын зураг

def make_blackspot_map(gdf, centers, cluster_sizes, file_name='aa3_blackspot_ml_map.html'):
    m = folium.Map(location=[47.92, 106.9], zoom_start=12, tiles='cartodbpositron', control_scale=True)
    # Хар цэгүүд болон энгийн цэгүүд
    for _, row in gdf.iterrows():
        color = '#d62728' if row['is_blackspot'] else '#1f77b4'
        radius = 8 if row['is_blackspot'] else 3
        popup = folium.Popup(f"""
        <b>Кластер:</b> {row['cluster']}<br>
        <b>Нягтрал:</b> {cluster_sizes[row['cluster']]}<br>
        <b>Байршил:</b> {row.get('Газар /Хэлтэс/','NA')}<br>
        <b>Хучилт:</b> {row.get('Хучилт','NA')}<br>
        <b>Огноо:</b> {row.get('Зөрчил огноо','NA')}<br>
        <b>Хар цэг үү:</b> {'Тийм' if row['is_blackspot'] else 'Үгүй'}<br>
        """, max_width=360)
        folium.CircleMarker(location=[row['Өргөрөг'], row['Уртраг']], radius=radius,
            color=color, fill=True, fill_opacity=0.8, popup=popup).add_to(m)
    # Кластер төвүүд
    for idx, c in enumerate(centers):
        folium.Marker([c[0], c[1]], icon=folium.Icon(color='red', icon='flag', prefix='fa'),
                      popup=f"<b>Кластер төв #{idx+1}</b>").add_to(m)
    # Heatmap
    HeatMap([[row['Өргөрөг'], row['Уртраг']] for _, row in gdf.iterrows()], radius=20, blur=13, min_opacity=0.32,
            name='Ослын нягтрал').add_to(m)
    # Легенд, нэмэлт тохиргоо
    legend = '''<div style="position: fixed; bottom: 44px; left: 25px; z-index: 9999; background: white; padding: 10px 18px; border: 1px solid #888; border-radius: 8px; font-size: 15px;"><b>Тэмдэглэгээ:</b><br><span style="color:#d62728;font-size:1.2em;">●</span> Хар цэг<br><span style="color:#1f77b4;font-size:1.2em;">●</span> Бусад осол<br><span style="color:#faad14;font-size:1.2em;">■</span> Heatmap</div>'''
    m.get_root().html.add_child(folium.Element(legend))
    # Контролууд
    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    Fullscreen(title="Бүтэн дэлгэц", title_cancel="Гарах").add_to(m)
    MeasureControl(primary_length_unit='meters').add_to(m)
    LocateControl().add_to(m)
    MousePosition().add_to(m)
    m.save(file_name)
    print(f"Газрын зураг: {file_name}")
    return m

# 5. PIPELINE

def main():
    file_path = 'ЗТО_2020-2024_ашиглах_final.xlsx'
    print('1. Өгөгдөл ачаалж...')
    df = load_and_prep(file_path)
    print('2. Хар цэг (кластер) ML ...')
    gdf, centers, cluster_sizes = blackspot_ml_cluster(df, n_clusters=14)
    print('3. Машин сургалтын таамаглал ...')
    train_rf_predict(df)
    print('4. Газрын зураг ...')
    make_blackspot_map(gdf, centers, cluster_sizes)
    print('--- ML + SPATIAL VISUAL DONE ---')

if __name__ == '__main__':
    main()
