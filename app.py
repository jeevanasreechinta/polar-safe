import os 
import json 
import numpy as np 
import pandas as pd 
import streamlit as st 
import plotly.graph_objects as go 
import plotly.express as px 
 
# ============================================================ 
# POLAR-SAFE - Streamlit Dashboard 
# SIH26059: AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, 
# and Navigation Decision Support System 
# ============================================================ 
 
st.set_page_config( 
    page_title="POLAR-SAFE | Antarctic Navigation Intelligence", 
    page_icon="🧊", 
    layout="wide", 
    initial_sidebar_state="expanded", 
) 
 
# ============================================================ 
# STYLE 
# ============================================================ 
 
st.markdown( 
    """ 
    <style> 
    .main { background-color: #0b1020; } 
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; } 
 
    .hero { 
        padding: 1.5rem 2rem; 
        border-radius: 18px; 
        background: linear-gradient(135deg,#101a35 0%,#16284a 50%,#10243c 100%); 
        border: 1px solid #29456b; 
        margin-bottom: 1rem; 
    } 
 
    .hero-title { 
        font-size: 2.4rem; 
        font-weight: 800; 
        margin-bottom: 0.2rem; 
    } 
 
    .hero-subtitle { 
        font-size: 1.05rem; 
        opacity: 0.85; 
    } 
 
    .warning-box { 
        padding: 0.9rem 1.2rem; 
        border-radius: 12px; 
        background-color: #332b12; 
        border: 1px solid #806d22; 
        margin-bottom: 1rem; 
    } 
 
    .section-title { 
        font-size: 1.35rem; 
        font-weight: 700; 
        margin-top: 1rem; 
        margin-bottom: 0.6rem; 
    } 
 
    .route-card { 
        padding: 1.2rem; 
        border-radius: 14px; 
        border: 1px solid #29456b; 
        background-color: #111a2c; 
        margin-bottom: 0.8rem; 
    } 
    </style> 
    """, 
    unsafe_allow_html=True, 
) 
 
# ============================================================ 
# FILE PATHS 
# ============================================================ 
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
 
PATHS = { 
    "sea_ice": os.path.join( 
        BASE_DIR, "sea_ice", "antarctic_sea_ice_risk_2025-04-16.csv" 
    ), 
    "iceberg_risk": os.path.join( 
        BASE_DIR, "iceberg", "iceberg_risk_layer.csv" 
    ), 
    "iceberg_trajectory": os.path.join( 
        BASE_DIR, "iceberg", "iceberg_trajectory_predictions.csv" 
    ), 
    "combined": os.path.join( 
        BASE_DIR, "combined", 
        "antarctic_combined_navigation_risk_2025-04-16.csv" 
    ), 
    "shortest": os.path.join(BASE_DIR, "routes", "shortest_route.csv"), 
    "balanced": os.path.join(BASE_DIR, "routes", "balanced_route.csv"), 
    "risk_aware": os.path.join(BASE_DIR, "routes", "risk_aware_route.csv"), 
    "comparison": os.path.join(BASE_DIR, "routes", "route_comparison.csv"), 
    "dashboard": os.path.join( 
        BASE_DIR, "dashboard", "dashboard_data.json" 
    ), 
"bathymetry": os.path.join(BASE_DIR, "bathymetry", "bathymetry_depth_256.npy"), 
"land_mask": os.path.join(BASE_DIR, "bathymetry", "land_mask_256.npy"), 
} 
 
# ============================================================ 
# DATA LOADING 
# ============================================================ 
 
@st.cache_data 
def load_csv(path): 
    if not os.path.exists(path): 
        return pd.DataFrame() 
    try: 
        return pd.read_csv(path) 
    except Exception as exc: 
        st.error(f"Could not read {os.path.basename(path)}: {exc}") 
        return pd.DataFrame() 
 
 
@st.cache_data 
def load_json(path): 
    if not os.path.exists(path): 
        return {} 
    try: 
        with open(path, "r", encoding="utf-8") as f: 
            return json.load(f) 
    except Exception: 
        return {} 
 
 
sea_ice = load_csv(PATHS["sea_ice"]) 
iceberg_risk = load_csv(PATHS["iceberg_risk"]) 
iceberg_trajectory = load_csv(PATHS["iceberg_trajectory"]) 
combined = load_csv(PATHS["combined"]) 
try: 
    bathymetry = np.load(PATHS["bathymetry"], allow_pickle=False) 
except Exception: 
    bathymetry = None 
 
try: 
    land_mask = np.load(PATHS["land_mask"], allow_pickle=False) 
except Exception: 
    land_mask = np.zeros((100, 100), dtype=bool) 
shortest_route = load_csv(PATHS["shortest"]) 
balanced_route = load_csv(PATHS["balanced"]) 
risk_aware_route = load_csv(PATHS["risk_aware"]) 
route_comparison = load_csv(PATHS["comparison"]) 
 
dashboard_data = load_json(PATHS["dashboard"]) 
 
# ============================================================ 
# CLEANING 
# ============================================================ 
 
def clean_coordinates(df): 
    if df.empty: 
        return df 
 
    out = df.copy() 
 
    for col in ("latitude", "longitude"): 
        if col in out.columns: 
            out[col] = pd.to_numeric(out[col], errors="coerce") 
 
    if "longitude" in out.columns: 
        out["longitude"] = ((out["longitude"] + 180) % 360) - 180 
 
    keep = [c for c in ("latitude", "longitude") if c in out.columns] 
    if keep: 
        out = out.dropna(subset=keep) 
 
    return out 
 
 
sea_ice = clean_coordinates(sea_ice) 
iceberg_risk = clean_coordinates(iceberg_risk) 
iceberg_trajectory = clean_coordinates(iceberg_trajectory) 
combined = clean_coordinates(combined) 
shortest_route = clean_coordinates(shortest_route) 
balanced_route = clean_coordinates(balanced_route) 
risk_aware_route = clean_coordinates(risk_aware_route) 
 
# ============================================================ 
# COLUMN HELPERS 
# ============================================================ 
 
def first_existing_column(df, names): 
    for name in names: 
        if name in df.columns: 
            return name 
    return None 
 
 
def risk_column(df): 
    return first_existing_column( 
        df, 
        [ 
            "combined_risk_score", 
            "sea_ice_risk_score", 
            "iceberg_risk_score", 
            "risk_score", 
        ], 
    ) 
 
 
def route_distance(df): 
    col = first_existing_column( 
        df, 
        [ 
            "distance_km", 
            "route_distance_km", 
            "cumulative_distance_km", 
            "total_distance_km", 
        ], 
    ) 
    if col is None or df.empty: 
        return np.nan 
    values = pd.to_numeric(df[col], errors="coerce").dropna() 
    return float(values.max()) if not values.empty else np.nan 
 
 
# ============================================================ 
# MAP HELPERS 
# IMPORTANT: 
# We intentionally use go.Scattergeo only. 
# No px.scatter_geo() is used, avoiding the Plotly projection 
# compatibility problem encountered with the installed version. 
# ============================================================ 
 
def polar_geo_settings(): 
    return dict( 
        projection=dict(type="stereographic"), 
        projection_rotation=dict(lon=0, lat=-90, roll=0), 
        showland=True, 
        landcolor="#d9e0e7", 
        showocean=True, 
        oceancolor="#081a2c", 
        showcountries=True, 
        countrycolor="#536b82", 
        showcoastlines=True, 
        coastlinecolor="#ffffff", 
        coastlinewidth=0.8, 
        lataxis=dict(showgrid=True, gridcolor="#40556d"), 
        lonaxis=dict(showgrid=True, gridcolor="#40556d"), 
        bgcolor="#07111f", 
    ) 
 
 
def create_risk_map(df, risk_col, title, marker_size=6): 
    fig = go.Figure() 
 
    if df.empty: 
        fig.add_annotation( 
            text="No data available for this layer.", 
            x=0.5, 
            y=0.5, 
            xref="paper", 
            yref="paper", 
            showarrow=False, 
        ) 
    elif risk_col and risk_col in df.columns: 
        values = pd.to_numeric(df[risk_col], errors="coerce").fillna(0) 
 
        hover_text = [ 
            f"Latitude: {lat:.2f}<br>" 
            f"Longitude: {lon:.2f}<br>" 
            f"Risk: {risk:.1f}/100" 
            for lat, lon, risk in zip( 
                df["latitude"], df["longitude"], values 
            ) 
        ] 
 
        fig.add_trace( 
            go.Scattergeo( 
                lon=df["longitude"], 
                lat=df["latitude"], 
                mode="markers", 
                marker=dict( 
                    size=marker_size, 
                    color=values, 
                    colorscale="Turbo", 
                    cmin=0, 
                    cmax=100, 
                    opacity=0.78, 
                    colorbar=dict(title="Risk<br>0–100"), 
                ), 
                text=hover_text, 
                hovertemplate="%{text}<extra></extra>", 
                name="Risk", 
            ) 
        ) 
    else: 
        fig.add_trace( 
            go.Scattergeo( 
                lon=df["longitude"], 
                lat=df["latitude"], 
                mode="markers", 
                marker=dict(size=marker_size, opacity=0.7), 
                hovertemplate=( 
                    "Latitude: %{lat:.2f}<br>" 
                    "Longitude: %{lon:.2f}<extra></extra>" 
                ), 
                name="Observations", 
            ) 
        ) 
 
    fig.update_geos(**polar_geo_settings()) 
 
    fig.update_layout( 
        title=title, 
        height=600, 
        margin=dict(l=0, r=0, t=55, b=0), 
        paper_bgcolor="#07111f", 
        plot_bgcolor="#07111f", 
        font=dict(color="white"), 
        legend=dict(bgcolor="rgba(0,0,0,0.3)"), 
    ) 
 
    return fig 
 
def nearest_grid_cell(lat, lon, lat_min, lat_max, lon_min, lon_max, shape): 
    rows, cols = shape 
 
    row = int(round( 
        (lat_max - lat) / (lat_max - lat_min) * (rows - 1) 
    )) 
 
    col = int(round( 
        (lon - lon_min) / (lon_max - lon_min) * (cols - 1) 
    )) 
 
    row = max(0, min(rows - 1, row)) 
    col = max(0, min(cols - 1, col)) 
 
    return row, col 
 
 
def astar_navigation(cost_grid, start, goal): 
    import heapq 
    import math 
 
    rows, cols = cost_grid.shape 
 
    if not np.isfinite(cost_grid[start]): 
        return [] 
 
    if not np.isfinite(cost_grid[goal]): 
        return [] 
 
    neighbors = [ 
        (-1, -1), (-1, 0), (-1, 1), 
        (0, -1),           (0, 1), 
        (1, -1),  (1, 0),  (1, 1), 
    ] 
 
    def heuristic(a, b): 
        return math.hypot( 
            a[0] - b[0], 
            a[1] - b[1], 
        ) 
 
    open_set = [] 
 
    heapq.heappush( 
        open_set, 
        (heuristic(start, goal), start) 
    ) 
 
    came_from = {} 
    g_score = {start: 0.0} 
 
    while open_set: 
        _, current = heapq.heappop(open_set) 
 
        if current == goal: 
            path = [current] 
 
            while current in came_from: 
                current = came_from[current] 
                path.append(current) 
 
            path.reverse() 
            return path 
 
        for dr, dc in neighbors: 
            nr = current[0] + dr 
            nc = current[1] + dc 
 
            if nr < 0 or nr >= rows: 
                continue 
 
            if nc < 0 or nc >= cols: 
                continue 
 
            if not np.isfinite(cost_grid[nr, nc]): 
                continue 
 
            step_distance = math.sqrt( 
                2 if dr != 0 and dc != 0 else 1 
            ) 
 
            tentative_g = ( 
                g_score[current] 
                + step_distance * cost_grid[nr, nc] 
            ) 
 
            neighbor = (nr, nc) 
 
            if tentative_g < g_score.get(neighbor, float("inf")): 
                came_from[neighbor] = current 
                g_score[neighbor] = tentative_g 
 
                f_score = ( 
                    tentative_g 
                    + heuristic(neighbor, goal) 
                ) 
 
                heapq.heappush( 
                    open_set, 
                    (f_score, neighbor) 
                ) 
 
    return [] 
def build_navigation_cost_grid(combined_df, bathymetry_array, safety_preference): 
    """ 
    Build an A* cost grid using environmental risk and bathymetry. 
 
    Higher environmental risk and shallow water produce higher travel cost. 
    Cells blocked by the land/shallow mask are treated as impassable. 
    """ 
 
    rows, cols = bathymetry_array.shape 
 
    # Resize environmental risk to the routing grid 
    risk_grid = np.zeros((rows, cols), dtype=np.float32) 
 
    if not combined_df.empty: 
        risk_col = first_existing_column( 
            combined_df, 
            [ 
                "combined_risk_score", 
                "combined_navigation_risk_score", 
                "combined_risk", 
            ], 
        ) 
 
        if risk_col and { 
            "latitude", 
            "longitude", 
        }.issubset(combined_df.columns): 
 
            lat_values = pd.to_numeric( 
                combined_df["latitude"], 
                errors="coerce", 
            ) 
 
            lon_values = pd.to_numeric( 
                combined_df["longitude"], 
                errors="coerce", 
            ) 
 
            risk_values = pd.to_numeric( 
                combined_df[risk_col], 
                errors="coerce", 
            ) 
 
            valid = ( 
                lat_values.notna() 
                & lon_values.notna() 
                & risk_values.notna() 
            ) 
 
            lat_values = lat_values[valid].to_numpy() 
            lon_values = lon_values[valid].to_numpy() 
            risk_values = risk_values[valid].to_numpy() 
 
            if len(risk_values) > 0: 
                lat_min = lat_values.min() 
                lat_max = lat_values.max() 
                lon_min = lon_values.min() 
                lon_max = lon_values.max() 
 
                lat_idx = ( 
                    (lat_max - lat_values) 
                    / max(lat_max - lat_min, 1e-9) 
                    * (rows - 1) 
                ).astype(int) 
 
                lon_idx = ( 
                    (lon_values - lon_min) 
                    / max(lon_max - lon_min, 1e-9) 
                    * (cols - 1) 
                ).astype(int) 
 
                lat_idx = np.clip(lat_idx, 0, rows - 1) 
                lon_idx = np.clip(lon_idx, 0, cols - 1) 
 
                risk_grid[lat_idx, lon_idx] = risk_values 
 
    # Normalize environmental risk 
    risk_grid = np.clip(risk_grid, 0, 100) 
 
    # Convert safety preference into a routing penalty. 
    # 0% = distance-focused 
    # 100% = strongly risk-focused 
    risk_weight = safety_preference / 100.0 
 
    environmental_cost = ( 
        1.0 
        + risk_weight * (risk_grid / 100.0) * 10.0 
    ) 
 
    # Bathymetry penalty. 
    # Deeper water has lower penalty. 
    depth = np.asarray(bathymetry_array, dtype=np.float32) 
 
    depth_penalty = np.clip( 
        (depth + 1000.0) / 1000.0, 
        0.0, 
        5.0, 
    ) 
 
    cost_grid = environmental_cost + depth_penalty 
 
    # Land / shallow cells are impassable. 
    cost_grid[land_mask == 1.0] = np.inf 
 
    return cost_grid 
def calculate_dynamic_route( 
    origin_coords, 
    destination_coords, 
    safety_preference, 
): 
    """ 
    Calculate a dynamic A* route using the available 
    Weddell Sea bathymetry grid. 
    """ 
 
    LAT_MIN = -75.0 
    LAT_MAX = -60.0 
    LON_MIN = -60.0 
    LON_MAX = -20.0 
 
    origin_lat, origin_lon = origin_coords 
    destination_lat, destination_lon = destination_coords 
 
    # Check whether the selected locations are inside 
    # the geographic coverage of the bathymetry grid. 
    for lat, lon, label in [ 
        (origin_lat, origin_lon, "Origin"), 
        (destination_lat, destination_lon, "Destination"), 
    ]: 
        if not ( 
            LAT_MIN <= lat <= LAT_MAX 
            and LON_MIN <= lon <= LON_MAX 
        ): 
            return [], ( 
                f"{label} is outside the available " 
                "bathymetry coverage " 
                f"({LAT_MIN} to {LAT_MAX} latitude, " 
                f"{LON_MIN} to {LON_MAX} longitude)." 
            ) 
 
    cost_grid = build_navigation_cost_grid( 
        combined, 
        bathymetry, 
        safety_preference, 
    ) 
 
    start = nearest_grid_cell( 
        origin_lat, 
        origin_lon, 
        LAT_MIN, 
        LAT_MAX, 
        LON_MIN, 
        LON_MAX, 
        cost_grid.shape, 
    ) 
 
    goal = nearest_grid_cell( 
        destination_lat, 
        destination_lon, 
        LAT_MIN, 
        LAT_MAX, 
        LON_MIN, 
        LON_MAX, 
        cost_grid.shape, 
    ) 
 
    # Find the nearest navigable cells if the selected 
    # coordinates fall directly on blocked cells. 
    def nearest_navigable(cell): 
        r0, c0 = cell 
 
        if np.isfinite(cost_grid[r0, c0]): 
            return cell 
 
        best = None 
        best_distance = float("inf") 
 
        for radius in range(1, 20): 
            for dr in range(-radius, radius + 1): 
                for dc in range(-radius, radius + 1): 
 
                    r = r0 + dr 
                    c = c0 + dc 
 
                    if r < 0 or r >= cost_grid.shape[0]: 
                        continue 
 
                    if c < 0 or c >= cost_grid.shape[1]: 
                        continue 
 
                    if not np.isfinite(cost_grid[r, c]): 
                        continue 
 
                    distance = dr * dr + dc * dc 
 
                    if distance < best_distance: 
                        best_distance = distance 
                        best = (r, c) 
 
            if best is not None: 
                return best 
 
        return None 
 
    start = nearest_navigable(start) 
    goal = nearest_navigable(goal) 
 
    if start is None or goal is None: 
        return [], "Could not find navigable start/end cells." 
 
    path = astar_navigation( 
        cost_grid, 
        start, 
        goal, 
    ) 
 
    if not path: 
        return [], "No navigable A* route was found." 
 
    # Convert grid cells back to geographic coordinates. 
    route = [] 
 
    rows, cols = cost_grid.shape 
 
    for row, col in path: 
 
        lat = LAT_MAX - ( 
            row / max(rows - 1, 1) 
            * (LAT_MAX - LAT_MIN) 
        ) 
 
        lon = LON_MIN + ( 
            col / max(cols - 1, 1) 
            * (LON_MAX - LON_MIN) 
        ) 
 
        route.append( 
            { 
                "latitude": lat, 
                "longitude": lon, 
            } 
        ) 
 
    return pd.DataFrame(route), None 
 
def add_route_trace(fig, df, name, width): 
    if df.empty: 
        return 
 
    if not {"latitude", "longitude"}.issubset(df.columns): 
        return 
 
    fig.add_trace( 
        go.Scattergeo( 
            lon=df["longitude"], 
            lat=df["latitude"], 
            mode="lines+markers", 
            line=dict(width=width), 
            marker=dict(size=4), 
            name=name, 
            hovertemplate=( 
                "Latitude: %{lat:.2f}<br>" 
                "Longitude: %{lon:.2f}" 
                f"<extra>{name}</extra>" 
            ), 
        ) 
    ) 
 
 
def create_route_map(origin_coords, destination_coords, origin_name, destination_name): 
    fig = go.Figure() 
 
    add_route_trace(fig, shortest_route, "Shortest Route", 3) 
    add_route_trace(fig, balanced_route, "Balanced Route", 4) 
    add_route_trace(fig, risk_aware_route, "Risk-Aware Route", 5) 
 
    # Dynamic origin marker 
    fig.add_trace( 
        go.Scattergeo( 
            lon=[origin_coords[1]], 
            lat=[origin_coords[0]], 
            mode="markers+text", 
            marker=dict(size=12, symbol="star"), 
            text=[origin_name], 
            textposition="top center", 
            name="Origin", 
        ) 
    ) 
 
    # Dynamic destination marker 
    fig.add_trace( 
        go.Scattergeo( 
            lon=[destination_coords[1]], 
            lat=[destination_coords[0]], 
            mode="markers+text", 
            marker=dict(size=12, symbol="diamond"), 
            text=[destination_name], 
            textposition="top center", 
            name="Destination", 
        ) 
    ) 
 
    fig.update_geos(**polar_geo_settings()) 
 
    fig.update_layout( 
        title=f"{origin_name} → {destination_name} Route Comparison", 
        height=650, 
        margin=dict(l=0, r=0, t=55, b=0), 
        paper_bgcolor="#07111f", 
        font=dict(color="white"), 
        legend=dict(bgcolor="rgba(0,0,0,0.35)"), 
    ) 
 
    return fig 
 
# ============================================================ 
# HEADER 
# ============================================================ 
 
st.markdown( 
    """ 
    <div class="hero"> 
        <div class="hero-title">🧊 POLAR-SAFE</div> 
        <div class="hero-subtitle"> 
            AI-Powered Antarctic Navigation & Risk Intelligence System 
        </div> 
        <div class="hero-subtitle"> 
            Predict the Ice • Track the Iceberg • Navigate Safely 
        </div> 
    </div> 
    """, 
    unsafe_allow_html=True, 
) 
 
st.markdown( 
    """ 
    <div class="warning-box"> 
        ⚠️ <b>Prototype Decision-Support System:</b> 
        POLAR-SAFE demonstrates AI-based environmental risk prediction 
        and route optimization. It supports professional navigation 
        decisions and does not replace qualified navigators. 
    </div> 
    """, 
    unsafe_allow_html=True, 
) 
 
# ============================================================ 
# SIDEBAR 
# ============================================================ 
 
st.sidebar.title("🧭 Navigation Controls") 
 
origin_options = { 
    "Bharati Station": (-69.4068, 76.19525), 
    "Maitri Station": (-70.76444, 11.73333), 
    "Custom Location": None, 
} 
 
origin = st.sidebar.selectbox( 
    "Origin", 
    list(origin_options.keys()), 
) 
 
destination_options = { 
    "Maitri Station": (-70.76444, 11.73333), 
    "Bharati Station": (-69.4068, 76.19525), 
    "Custom Location": None, 
} 
 
destination = st.sidebar.selectbox( 
    "Destination", 
    list(destination_options.keys()), 
) 
 
# Custom origin coordinates 
if origin == "Custom Location": 
    st.sidebar.markdown("### Custom Origin") 
 
    origin_lat = st.sidebar.number_input( 
        "Origin Latitude", 
        min_value=-90.0, 
        max_value=90.0, 
        value=-70.0, 
        step=0.1, 
    ) 
 
    origin_lon = st.sidebar.number_input( 
        "Origin Longitude", 
        min_value=-180.0, 
        max_value=180.0, 
        value=30.0, 
        step=0.1, 
    ) 
 
    origin_coords = (origin_lat, origin_lon) 
else: 
    origin_coords = origin_options[origin] 
 
 
# Custom destination coordinates 
if destination == "Custom Location": 
    st.sidebar.markdown("### Custom Destination") 
 
    destination_lat = st.sidebar.number_input( 
        "Destination Latitude", 
        min_value=-90.0, 
        max_value=90.0, 
        value=-70.0, 
        step=0.1, 
    ) 
 
    destination_lon = st.sidebar.number_input( 
        "Destination Longitude", 
        min_value=-180.0, 
        max_value=180.0, 
        value=20.0, 
        step=0.1, 
    ) 
 
    destination_coords = (destination_lat, destination_lon) 
else: 
    destination_coords = destination_options[destination] 
 
layer = st.sidebar.radio( 
    "Intelligence Layer", 
    [ 
        "Combined Navigation Risk", 
        "Sea-Ice Risk", 
        "Iceberg Risk", 
        "Routes", 
    ], 
) 
 
st.sidebar.markdown("---") 
 
safety_preference = st.sidebar.slider( 
    "Safety Preference", 
    0, 
    100, 
    80, 
    help="Higher values indicate a stronger preference for lower-risk routing.", 
) 
 
st.sidebar.caption( 
    f"Safety Preference: {safety_preference}%" 
) 
st.sidebar.caption("POLAR-SAFE • SIH26059") 
 
# ============================================================ 
# KPIs 
# ============================================================ 
 
combined_col = risk_column(combined) 
sea_ice_col = risk_column(sea_ice) 
iceberg_col = risk_column(iceberg_risk) 
 
if combined_col and not combined.empty: 
    average_risk = float( 
        pd.to_numeric(combined[combined_col], errors="coerce").mean() 
    ) 
    maximum_risk = float( 
        pd.to_numeric(combined[combined_col], errors="coerce").max() 
    ) 
else: 
    average_risk = 0.0 
    maximum_risk = 0.0 
 
if "combined_risk_level" in combined.columns: 
    high_risk_cells = int( 
        combined["combined_risk_level"] 
        .isin(["High", "Very High"]) 
        .sum() 
    ) 
elif combined_col: 
    high_risk_cells = int( 
        (pd.to_numeric(combined[combined_col], errors="coerce") >= 40).sum() 
    ) 
else: 
    high_risk_cells = 0 
 
if "iceberg_id" in iceberg_risk.columns: 
    tracked_icebergs = int(iceberg_risk["iceberg_id"].nunique()) 
else: 
    tracked_icebergs = 0 
 
k1, k2, k3, k4 = st.columns(4) 
 
k1.metric("Average Risk", f"{average_risk:.1f}/100") 
k2.metric("Maximum Risk", f"{maximum_risk:.1f}/100") 
k3.metric("Tracked Icebergs", f"{tracked_icebergs}") 
k4.metric("High-Risk Cells", f"{high_risk_cells:,}") 
 
st.markdown("---") 
 
# ============================================================ 
# COMBINED RISK 
# ============================================================ 
 
if layer == "Combined Navigation Risk": 
 
    st.markdown( 
        '<div class="section-title">🌐 Combined Navigation Risk Intelligence</div>', 
        unsafe_allow_html=True, 
    ) 
 
    st.write( 
        "Sea-ice conditions and nearby iceberg trajectory risk are " 
        "combined into one navigation-risk surface." 
    ) 
 
    if combined.empty: 
        st.error( 
            "Combined risk file is missing or could not be loaded." 
        ) 
    else: 
        fig = create_risk_map( 
            combined, 
            combined_col, 
            "Antarctic Combined Navigation Risk", 
            6, 
        ) 
        st.plotly_chart(fig, width="stretch") 
 
        a, b, c = st.columns(3) 
        a.metric("Risk Surface Points", f"{len(combined):,}") 
        b.metric("Mean Risk", f"{average_risk:.2f}") 
        c.metric("Peak Risk", f"{maximum_risk:.2f}") 
 
        if "combined_risk_level" in combined.columns: 
            distribution = ( 
                combined["combined_risk_level"] 
                .value_counts() 
                .rename_axis("Risk Level") 
                .reset_index(name="Cells") 
            ) 
 
            fig_dist = go.Figure( 
                go.Bar( 
                    x=distribution["Risk Level"], 
                    y=distribution["Cells"], 
                    text=distribution["Cells"], 
                    textposition="auto", 
                ) 
            ) 
            fig_dist.update_layout( 
                title="Navigation Risk Categories", 
                height=400, 
                paper_bgcolor="#07111f", 
                plot_bgcolor="#07111f", 
                font=dict(color="white"), 
            ) 
            st.plotly_chart(fig_dist, width="stretch") 
 
# ============================================================ 
# SEA-ICE 
# ============================================================ 
 
elif layer == "Sea-Ice Risk": 
 
    st.markdown( 
        '<div class="section-title">🧊 AI Sea-Ice Risk Forecast</div>', 
        unsafe_allow_html=True, 
    ) 
 
    st.write( 
        "AI-predicted sea-ice concentration is converted into a " 
        "navigation-risk layer." 
    ) 
 
    if sea_ice.empty: 
        st.error( 
            "Sea-ice risk file is missing or could not be loaded." 
        ) 
    else: 
        fig = create_risk_map( 
            sea_ice, 
            sea_ice_col, 
            "Antarctic Sea-Ice Risk", 
            6, 
        ) 
        st.plotly_chart(fig, width="stretch") 
 
        predicted_col = first_existing_column( 
            sea_ice, 
            [ 
                "predicted_sea_ice_concentration", 
                "sea_ice_concentration", 
            ], 
        ) 
 
        a, b, c = st.columns(3) 
        a.metric("Grid Cells", f"{len(sea_ice):,}") 
 
        if predicted_col: 
            mean_sic = pd.to_numeric( 
                sea_ice[predicted_col], errors="coerce" 
            ).mean() 
            b.metric( 
                "Mean Sea-Ice Concentration", 
                f"{mean_sic:.1f}%", 
            ) 
        else: 
            b.metric("Mean Sea-Ice Concentration", "N/A") 
 
        if sea_ice_col: 
            max_sic_risk = pd.to_numeric( 
                sea_ice[sea_ice_col], errors="coerce" 
            ).max() 
            c.metric("Maximum Ice Risk", f"{max_sic_risk:.1f}") 
        else: 
            c.metric("Maximum Ice Risk", "N/A") 
 
        preview_cols = [ 
            col 
            for col in [ 
                "latitude", 
                "longitude", 
                "sea_ice_concentration", 
                "predicted_sea_ice_concentration", 
                "sea_ice_risk_score", 
                "sea_ice_risk_level", 
            ] 
            if col in sea_ice.columns 
        ] 
 
        if preview_cols: 
            st.markdown("### Sea-Ice Data Preview") 
            st.dataframe( 
                sea_ice[preview_cols].head(100), 
                width="stretch", 
            ) 
 
# ============================================================ 
# ICEBERG 
# ============================================================ 
 
elif layer == "Iceberg Risk": 
 
    st.markdown( 
        '<div class="section-title">🧊 Iceberg Trajectory Intelligence</div>', 
        unsafe_allow_html=True, 
    ) 
 
    st.write( 
        "AI-predicted iceberg positions are combined with trajectory " 
        "uncertainty and movement to estimate iceberg hazard." 
    ) 
 
    if iceberg_risk.empty: 
        st.error( 
            "Iceberg risk file is missing or could not be loaded." 
        ) 
    else: 
        fig = create_risk_map( 
            iceberg_risk, 
            iceberg_col, 
            "Antarctic Iceberg Risk", 
            9, 
        ) 
        st.plotly_chart(fig, width="stretch") 
 
        a, b, c = st.columns(3) 
 
        a.metric( 
            "Tracked Icebergs", 
            f"{iceberg_risk['iceberg_id'].nunique():,}" 
            if "iceberg_id" in iceberg_risk.columns 
            else "N/A", 
        ) 
 
        if "prediction_error_km" in iceberg_risk.columns: 
            error = pd.to_numeric( 
                iceberg_risk["prediction_error_km"], 
                errors="coerce", 
            ).median() 
            b.metric( 
                "Median Prediction Error", 
                f"{error:.2f} km", 
            ) 
        else: 
            b.metric("Median Prediction Error", "N/A") 
 
        if iceberg_col: 
            max_iceberg_risk = pd.to_numeric( 
                iceberg_risk[iceberg_col], 
                errors="coerce", 
            ).max() 
            c.metric( 
                "Maximum Iceberg Risk", 
                f"{max_iceberg_risk:.1f}", 
            ) 
        else: 
            c.metric("Maximum Iceberg Risk", "N/A") 
 
        if "iceberg_risk_level" in iceberg_risk.columns: 
            distribution = ( 
                iceberg_risk["iceberg_risk_level"] 
                .value_counts() 
                .rename_axis("Risk Level") 
                .reset_index(name="Observations") 
            ) 
 
            fig_dist = go.Figure( 
                go.Bar( 
                    x=distribution["Risk Level"], 
                    y=distribution["Observations"], 
                    text=distribution["Observations"], 
                    textposition="auto", 
                ) 
            ) 
            fig_dist.update_layout( 
                title="Iceberg Risk Categories", 
                height=400, 
                paper_bgcolor="#07111f", 
                plot_bgcolor="#07111f", 
                font=dict(color="white"), 
            ) 
            st.plotly_chart(fig_dist, width="stretch") 
 
# ============================================================ 
# ROUTES 
# ============================================================ 
 
elif layer == "Routes": 
 
    st.markdown( 
        '<div class="section-title">🚢 AI Route Optimization</div>', 
        unsafe_allow_html=True, 
    ) 
 
    st.write( 
        f"Prototype voyage scenario: **{origin} → {destination}**" 
    ) 
 
    # Calculate dynamic route 
dynamic_route, route_error = calculate_dynamic_route( 
    origin_coords, 
    destination_coords, 
    safety_preference, 
) 
 
if route_error: 
    st.warning(route_error) 
 
    # Fall back to the stored routes for locations 
    # outside the current bathymetry coverage. 
    st.plotly_chart( 
        create_route_map( 
            origin_coords, 
            destination_coords, 
            origin, 
            destination, 
        ), 
        width="stretch", 
    ) 
 
else: 
    st.success( 
        f"Dynamic A* route generated for " 
        f"{origin} → {destination}" 
    ) 
 
    fig = go.Figure() 
 
    add_route_trace( 
        fig, 
        dynamic_route, 
        "Dynamic A* Route", 
        5, 
    ) 
 
    fig.add_trace( 
        go.Scattergeo( 
            lon=[origin_coords[1]], 
            lat=[origin_coords[0]], 
            mode="markers+text", 
            marker=dict(size=12, symbol="star"), 
            text=[origin], 
            textposition="top center", 
            name="Origin", 
        ) 
    ) 
 
    fig.add_trace( 
        go.Scattergeo( 
            lon=[destination_coords[1]], 
            lat=[destination_coords[0]], 
            mode="markers+text", 
            marker=dict(size=12, symbol="diamond"), 
            text=[destination], 
            textposition="top center", 
            name="Destination", 
        ) 
    ) 
 
    fig.update_geos(**polar_geo_settings()) 
 
    fig.update_layout( 
        title=f"Dynamic A* Route: {origin} → {destination}", 
        height=650, 
        margin=dict(l=0, r=0, t=55, b=0), 
        paper_bgcolor="#07111f", 
        font=dict(color="white"), 
        legend=dict(bgcolor="rgba(0,0,0,0.35)"), 
    ) 
 
    st.plotly_chart( 
        fig, 
        width="stretch", 
    ) 
 
    route_rows = [] 
 
    for name, route_df in [ 
        ("Shortest", shortest_route), 
        ("Balanced", balanced_route), 
        ("Risk-Aware", risk_aware_route), 
    ]: 
        if route_df.empty: 
            continue 
 
        rcol = first_existing_column( 
            route_df, 
            [ 
                "combined_risk_score", 
                "combined_risk", 
                "risk_score", 
            ], 
        ) 
 
        if rcol: 
            risks = pd.to_numeric( 
                route_df[rcol], errors="coerce" 
            ).dropna() 
            mean_risk = risks.mean() if not risks.empty else np.nan 
            max_risk = risks.max() if not risks.empty else np.nan 
        else: 
            mean_risk = np.nan 
            max_risk = np.nan 
 
        route_rows.append( 
            { 
                "Route": name, 
                "Distance (km)": route_distance(route_df), 
                "Mean Risk": mean_risk, 
                "Max Risk": max_risk, 
                "Waypoints": len(route_df), 
            } 
        ) 
 
    if route_rows: 
        st.markdown("### Route Comparison") 
        st.dataframe( 
            pd.DataFrame(route_rows), 
            width="stretch", 
            hide_index=True, 
        ) 
 
    if not route_comparison.empty: 
        st.markdown("### 📊 Stored Optimization Results") 
        st.dataframe( 
            route_comparison, 
            width="stretch", 
            hide_index=True, 
        ) 
 
    st.markdown( 
        """ 
        <div class="route-card"> 
            <h3>🧠 POLAR-SAFE Recommendation</h3> 
            <p> 
                <b>Risk-Aware Route</b> is recommended when vessel safety 
                is prioritized over minimum travel distance. It trades 
                additional distance for lower accumulated environmental risk. 
            </p> 
        </div> 
        """, 
        unsafe_allow_html=True, 
    ) 
 
# ============================================================ 
# ICEBERG TRAJECTORY EXPLORER 
# ============================================================ 
 
st.markdown("---") 
st.markdown( 
    '<div class="section-title">📍 Iceberg Trajectory Explorer</div>', 
    unsafe_allow_html=True, 
) 
 
if not iceberg_trajectory.empty and "iceberg_id" in iceberg_trajectory.columns: 
 
    iceberg_ids = sorted( 
        iceberg_trajectory["iceberg_id"] 
        .dropna() 
        .unique() 
        .tolist() 
    ) 
 
    if iceberg_ids: 
 
        selected_iceberg = st.selectbox( 
            "Select Iceberg", 
            iceberg_ids, 
        ) 
 
        selected = iceberg_trajectory[ 
            iceberg_trajectory["iceberg_id"] == selected_iceberg 
        ].copy() 
 
        selected = selected.sort_values( 
            "datetime" 
        ) if "datetime" in selected.columns else selected 
 
        fig = go.Figure() 
 
        fig.add_trace( 
            go.Scattergeo( 
                lon=selected["longitude"], 
                lat=selected["latitude"], 
                mode="lines+markers", 
                name="Observed", 
                line=dict(width=3), 
                marker=dict(size=5), 
                hovertemplate=( 
                    "Latitude: %{lat:.3f}<br>" 
                    "Longitude: %{lon:.3f}" 
                    "<extra>Observed</extra>" 
                ), 
            ) 
        ) 
 
        if { 
            "predicted_latitude", 
            "predicted_longitude", 
        }.issubset(selected.columns): 
 
            fig.add_trace( 
                go.Scattergeo( 
                    lon=selected["predicted_longitude"], 
                    lat=selected["predicted_latitude"], 
                    mode="lines+markers", 
                    name="AI Prediction", 
                    line=dict(width=3, dash="dash"), 
                    marker=dict(size=5), 
                    hovertemplate=( 
                        "Latitude: %{lat:.3f}<br>" 
                        "Longitude: %{lon:.3f}" 
                        "<extra>AI Prediction</extra>" 
                    ), 
                ) 
            ) 
 
        fig.update_geos(**polar_geo_settings()) 
 
        fig.update_layout( 
            title=f"Iceberg {selected_iceberg} — Observed vs AI Predicted", 
            height=550, 
            margin=dict(l=0, r=0, t=55, b=0), 
            paper_bgcolor="#07111f", 
            font=dict(color="white"), 
        ) 
 
        st.plotly_chart(fig, width="stretch") 
 
        if "prediction_error_km" in selected.columns: 
            mean_error = pd.to_numeric( 
                selected["prediction_error_km"], 
                errors="coerce", 
            ).mean() 
            st.metric( 
                "Mean Prediction Error", 
                f"{mean_error:.2f} km", 
            ) 
 
# ============================================================ 
# DECISION SUMMARY 
# ============================================================ 
 
st.markdown("---") 
st.markdown( 
    '<div class="section-title">🎯 Navigation Decision Summary</div>', 
    unsafe_allow_html=True, 
) 
 
a, b, c = st.columns(3) 
 
a.markdown( 
    """ 
    ### 1️⃣ Predict 
    **Sea-Ice AI** 
 
    Forecast sea-ice concentration and convert it into a 
    navigation-risk surface. 
    """ 
) 
 
b.markdown( 
    """ 
    ### 2️⃣ Track 
    **Iceberg AI** 
 
    Predict iceberg movement and estimate trajectory uncertainty. 
    """ 
) 
 
c.markdown( 
    """ 
    ### 3️⃣ Navigate 
    **Route Optimization** 
 
    Compare shortest, balanced and risk-aware routes using 
    environmental risk. 
    """ 
) 
 
st.markdown("---") 
 
st.markdown( 
    """ 
    ### 🔄 POLAR-SAFE Intelligence Pipeline 
 
    **Satellite / Ocean / Weather Data** 
    ↓ 
    **Data Fusion & Preprocessing** 
    ↓ 
    **AI Sea-Ice + Iceberg Prediction** 
    ↓ 
    **Environmental Risk Engine** 
    ↓ 
    **A* Route Optimization** 
    ↓ 
    **Navigation Decision Support** 
    """ 
) 
 
# ============================================================ 
# FOOTER 
# ============================================================ 
 
st.markdown("---") 
 
st.caption( 
    "POLAR-SAFE • SIH26059 • AI-Enabled Antarctic Sea-Ice, " 
    "Iceberg Trajectory, and Navigation Decision Support System" 
) 
 
st.caption( 
    "Prototype for demonstration and research decision support. " 
    "Final navigation decisions remain with qualified operators." 
) give a proper app.py file where all will display right values adn take it from the datasets
