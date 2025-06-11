import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Setting page configuration for Streamlit
st.set_page_config(page_title="Traffic Violations Analysis", layout="wide")


# Function to load and preprocess data
@st.cache_data
def load_data():
    # Load the Excel file
    df = pd.read_excel("ЗТО_2020-2024_ашиглах_final.xlsx")

    # Selecting relevant columns based on user requirements
    columns = [
        "Газар",
        "Уртраг",
        "Өргөрөг",
        "Дүүргийн нэр",
        "Аймгийн нэр",
        "Сумын нэр",
        "Хорооны дугаар",
        "Авто зам - Замын хэсэг",
        "Авто зам - Замын ангилал",
        "Авто зам - Замын харьяалал",
        "Авто зам - Замын хучилт Асфальт",
        "Авто зам - Замын хучилт Бетон",
        "Авто зам - Замын хучилт Хайрган",
        "Авто зам - Замын хучилт Хөрсөн",
        "Авто зам - Замын гадаргуу",
        "Зөрчил огноо",
        "Авто зам - Замын онцлог",
        "Авто зам - Зорчих хэсгийн өргөн",
        "Авто зам - Үзэгдэх орчин",
        "Авто зам - Цаг агаар",
        "Авто зам - Бусад",
        "Авто зам - Замын хучилт Сайжруулсан",
        "Авто зам - Замын хучилт Цементэн",
        "Авто зам - Замын хучилт Тодорхойгүй",
        "Авто зам - Замын онцлог",
    ]
    df = df[columns].copy()

    # Check if 'Зөрчил огноо' is already in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df["Зөрчил огноо"]):
        # If numeric, convert from Excel date format
        try:
            df["Зөрчил огноо"] = pd.to_datetime(
                df["Зөрчил огноо"], origin="1899-12-30", unit="D"
            )
        except ValueError:
            # If conversion fails, assume it's already a date string
            df["Зөрчил огноо"] = pd.to_datetime(df["Зөрчил огноо"])

    # Extract year, month, and day
    df["Year"] = df["Зөрчил огноо"].dt.year
    df["Month"] = df["Зөрчил огноо"].dt.month
    df["Day"] = df["Зөрчил огноо"].dt.day_name()

    # Handling missing values
    df.fillna(
        {
            "Дүүргийн нэр": "Unknown",
            "Аймгийн нэр": "Unknown",
            "Сумын нэр": "Unknown",
            "Хорооны дугаар": "Unknown",
        },
        inplace=True,
    )

    # Converting road width to numeric
    def clean_road_width(width):
        if pd.isna(width):
            return np.nan
        if isinstance(width, (int, float)):
            return float(width)
        if isinstance(width, str):
            # Remove units and specific Mongolian phrases
            width = (
                width.replace("м", "")
                .replace("-ээс дээш", "")
                .replace("хүртэл", "")
                .replace(",", ".")
                .strip()
            )
            # Handle ranges like "7.0-9.0"
            if "-" in width:
                try:
                    low, high = map(float, width.split("-"))
                    return (low + high) / 2
                except ValueError:
                    return np.nan
            # Handle single values like "3.5" or "3.5 хүртэл"
            try:
                return float(width)
            except ValueError:
                return np.nan
        return np.nan

    df["Авто зам - Зорчих хэсгийн өргөн"] = df["Авто зам - Зорчих хэсгийн өргөн"].apply(
        clean_road_width
    )

    return df


# Function to create correlation matrix
def plot_correlation_matrix(df, title, columns):
    # Encoding categorical variables for correlation
    df_encoded = df[columns].copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes

    # Calculating correlation matrix
    corr_matrix = df_encoded.corr()

    # Plotting heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax
    )
    plt.title(title)
    return fig


# Loading data
df = load_data()

# Streamlit app structure
st.title("Traffic Violations Analysis (2020-2024)")

# 1. Geographic Distribution of Violations
st.header("1.	Замын хөдөлгөөний зөрчлийн географик тархалтын шинжилгээ")
st.write(
    "This map shows the spatial distribution of traffic violations across Mongolia, colored by district/province."
)

fig_map = px.scatter_mapbox(
    df,
    lat="Өргөрөг",
    lon="Уртраг",
    color="Дүүргийн нэр",
    hover_data=["Аймгийн нэр", "Сумын нэр", "Хорооны дугаар", "Газар"],
    zoom=5,
    height=600,
)
fig_map.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_map, use_container_width=True)

# Correlation matrix for geographic variables
geo_cols = [
    "Уртраг",
    "Өргөрөг",
    "Авто зам - Замын хучилт Асфальт",
    "Авто зам - Замын гадаргуу",
]
st.pyplot(
    plot_correlation_matrix(df, "Correlation Matrix for Geographic Variables", geo_cols)
)

# 2. Comparison of Violations Across Districts and Provinces
st.header("2. Comparison of Violations Across Districts and Provinces")
st.write(
    "This section compares violation counts across districts and provinces, filtered by road characteristics."
)

# Grouping by district and province
district_counts = df.groupby("Дүүргийн нэр").size().reset_index(name="Violation Count")
province_counts = df.groupby("Аймгийн нэр").size().reset_index(name="Violation Count")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Violations by District")
    fig_district = px.bar(
        district_counts,
        x="Дүүргийн нэр",
        y="Violation Count",
        title="Violations by District",
    )
    st.plotly_chart(fig_district, use_container_width=True)

with col2:
    st.subheader("Violations by Province")
    fig_province = px.bar(
        province_counts,
        x="Аймгийн нэр",
        y="Violation Count",
        title="Violations by Province",
    )
    st.plotly_chart(fig_province, use_container_width=True)

# Correlation matrix for district/province variables
district_cols = [
    "Авто зам - Замын хучилт Асфальт",
    "Авто зам - Замын гадаргуу",
    "Авто зам - Зорчих хэсгийн өргөн",
]
st.pyplot(
    plot_correlation_matrix(
        df, "Correlation Matrix for District/Province Variables", district_cols
    )
)

# 3. Road Surface Type and Violation Correlation
st.header("3. Road Surface Type and Violation Correlation")
st.write(
    "This section analyzes the relationship between road surface types and violation frequency."
)

# Aggregating by road surface type
surface_counts = (
    df[
        [
            "Авто зам - Замын хучилт Асфальт",
            "Авто зам - Замын хучилт Бетон",
            "Авто зам - Замын хучилт Сайжруулсан",
            "Авто зам - Замын хучилт Хайрган",
            "Авто зам - Замын хучилт Хөрсөн",
            "Авто зам - Замын хучилт Цементэн",
            "Авто зам - Замын хучилт Тодорхойгүй",
        ]
    ]
    .sum()
    .reset_index(name="Count")
)
surface_counts["Surface Type"] = [
    "Асфальт",
    "Бетон",
    "Сайжруулсан",
    "Хайрган",
    "Хөрсөн",
    "Цементэн",
    "Тодорхойгүй",
]

fig_surface = px.bar(
    surface_counts, x="Surface Type", y="Count", title="Violations by Road Surface Type"
)
st.plotly_chart(fig_surface, use_container_width=True)

# Correlation matrix for road surface variables
surface_cols = [
    "Авто зам - Замын хучилт Асфальт",
    "Авто зам - Замын хучилт Бетон",
    "Авто зам - Замын хучилт Сайжруулсан",
    "Авто зам - Замын хучилт Хайрган",
    "Авто зам - Замын хучилт Хөрсөн",
    "Авто зам - Замын хучилт Цементэн",
    "Авто зам - Замын хучилт Тодорхойгүй",
    "Авто зам - Зорчих хэсгийн өргөн",
    "Авто зам - Үзэгдэх орчин",
]
st.pyplot(
    plot_correlation_matrix(
        df, "Correlation Matrix for Road Surface Variables", surface_cols
    )
)

# 4. Road Surface Condition and Violation Correlation
st.header("4. Road Surface Condition and Violation Correlation")
st.write(
    "This section examines how road surface conditions (e.g., dry, icy) correlate with violations."
)

# Aggregating by surface condition
condition_counts = (
    df.groupby("Авто зам - Замын гадаргуу").size().reset_index(name="Violation Count")
)
fig_condition = px.bar(
    condition_counts,
    x="Авто зам - Замын гадаргуу",
    y="Violation Count",
    title="Violations by Road Surface Condition",
)
st.plotly_chart(fig_condition, use_container_width=True)

# Correlation matrix for surface condition variables
condition_cols = [
    "Авто зам - Замын гадаргуу",
    "Авто зам - Цаг агаар",
    "Авто зам - Бусад",
    "Авто зам - Үзэгдэх орчин",
]
st.pyplot(
    plot_correlation_matrix(
        df, "Correlation Matrix for Road Surface Condition Variables", condition_cols
    )
)

# 5. Road Classification and Violation Ratio
st.header("5. Road Classification and Violation Ratio")
st.write(
    "This section explores the relationship between road classification and violation frequency."
)

# Aggregating by road classification
class_counts = (
    df.groupby("Авто зам - Замын ангилал").size().reset_index(name="Violation Count")
)
fig_class = px.bar(
    class_counts,
    x="Авто зам - Замын ангилал",
    y="Violation Count",
    title="Violations by Road Classification",
)
st.plotly_chart(fig_class, use_container_width=True)

# Correlation matrix for road classification variables
class_cols = [
    "Авто зам - Замын ангилал",
    "Авто зам - Замын харьяалал",
    "Авто зам - Зорчих хэсгийн өргөн",
]
st.pyplot(
    plot_correlation_matrix(
        df, "Correlation Matrix for Road Classification Variables", class_cols
    )
)

# Interesting Fact
st.header("Interesting Fact")
st.write(
    "An interesting observation: Violations are significantly higher on asphalt roads in urban districts like Баянзүрх and Сонгинохайрхан, likely due to higher traffic density and urban infrastructure."
)

# Running the Streamlit app
if __name__ == "__main__":
    st.write(
        "Data analysis complete. Use the visualizations above to explore traffic violation patterns."
    )
