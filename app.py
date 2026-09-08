import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Antarctica Environmental Risk Explorer",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

/* Main background */

.stApp {
    background: linear-gradient(
        135deg,
        #eef7ff 0%,
        #dceeff 45%,
        #c9e4ff 100%
    );
}

/* Main content */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headers */

h1 {
    color: #082f49 !important;
    font-weight: 800 !important;
}

h2 {
    color: #0c4a6e !important;
    font-weight: 750 !important;
}

h3 {
    color: #075985 !important;
    font-weight: 700 !important;
}

p, span, label, div {
    color: #102a43;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0b3c5d 0%,
        #075985 50%,
        #0c4a6e 100%
    );
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #ffffff !important;
}

/* Sidebar inputs */

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #e0f2fe !important;
}

/* Cards */

.metric-card {
    background: rgba(255, 255, 255, 0.92);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #bfdbfe;
    box-shadow: 0 5px 20px rgba(7, 89, 133, 0.10);
}

/* Risk cards */

.risk-high {
    background: linear-gradient(
        135deg,
        #fee2e2,
        #fecaca
    );
    color: #7f1d1d !important;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #fca5a5;
}

.risk-high h2,
.risk-high h3 {
    color: #7f1d1d !important;
}

.risk-medium {
    background: linear-gradient(
        135deg,
        #fef3c7,
        #fde68a
    );
    color: #78350f !important;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #fbbf24;
}

.risk-medium h2,
.risk-medium h3 {
    color: #78350f !important;
}

.risk-low {
    background: linear-gradient(
        135deg,
        #dcfce7,
        #bbf7d0
    );
    color: #14532d !important;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #86efac;
}

.risk-low h2,
.risk-low h3 {
    color: #14532d !important;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        135deg,
        #075985,
        #0369a1
    );
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.65rem 1rem;
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #0369a1,
        #0284c7
    );
    color: white !important;
}

/* Info boxes */

.stAlert {
    border-radius: 12px;
}

/* Select boxes */

div[data-baseweb="select"] > div {
    background-color: white;
    color: #102a43;
}

/* Number inputs */

input {
    color: #102a43 !important;
}

/* Captions */

.stCaption {
    color: #365f7d !important;
}

/* Divider */

hr {
    border-color: #93c5fd;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.title("🧊 Antarctica Environmental Risk Explorer")

st.caption(
    "Explore Antarctic marine, ice and seabed conditions "
    "to identify environmentally sensitive areas."
)

st.divider()

# =========================================================
# REAL ANTARCTIC LOCATIONS
# =========================================================

antarctic_locations = {
    "Antarctic Peninsula": (-63.5, -57.0),
    "Ross Sea": (-72.0, -175.0),
    "Weddell Sea": (-72.0, -45.0),
    "Amundsen Sea": (-73.0, -110.0),
    "Bellingshausen Sea": (-70.0, -80.0),
    "Ross Ice Shelf": (-82.0, -175.0),
    "Filchner-Ronne Ice Shelf": (-80.0, -50.0),
    "South Pole": (-90.0, 0.0)
}

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🧊 Antarctica Explorer")

st.sidebar.markdown(
    "### Map Layers"
)

show_risk = st.sidebar.checkbox(
    "🌡️ Environmental Risk",
    True
)

show_sea_ice = st.sidebar.checkbox(
    "❄️ Sea Ice Concentration",
    False
)

show_icebergs = st.sidebar.checkbox(
    "🧊 Iceberg Monitoring",
    False
)

show_currents = st.sidebar.checkbox(
    "🌊 Ocean Currents",
    False
)

show_bathymetry = st.sidebar.checkbox(
    "🌎 Bathymetry",
    False
)

st.sidebar.divider()

st.sidebar.markdown(
    "### 📍 Antarctic Region"
)

selected_region = st.sidebar.selectbox(
    "Select region",
    list(antarctic_locations.keys())
)

region_lat, region_lon = antarctic_locations[selected_region]

st.sidebar.caption(
    f"Latitude: {region_lat:.2f}°"
)

st.sidebar.caption(
    f"Longitude: {region_lon:.2f}°"
)

st.sidebar.divider()

st.sidebar.markdown(
    "### ⚙️ Risk Settings"
)

risk_threshold = st.sidebar.slider(
    "Risk threshold",
    0,
    100,
    60
)

st.sidebar.divider()

st.sidebar.info(
    "Risk assessment combines environmental "
    "conditions from the available Antarctic datasets."
)

# =========================================================
# TOP STATUS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Region",
        "Antarctica"
    )

with c2:
    st.metric(
        "Overall Risk",
        "HIGH"
    )

with c3:
    st.metric(
        "Risk Score",
        "72 / 100"
    )

with c4:
    st.metric(
        "Active Layers",
        sum([
            show_risk,
            show_sea_ice,
            show_icebergs,
            show_currents,
            show_bathymetry
        ])
    )

st.divider()

# =========================================================
# MAP
# =========================================================

st.subheader("🗺️ Environmental Risk Map")

# Demo environmental points
# These will later be replaced by integrated dataset values.

lat = np.array([
    -65, -66, -67, -68, -69,
    -70, -71, -72, -73, -74
])

lon = np.array([
    -60, -40, -20, 0, 20,
    40, 60, 80, 100, 120
])

risk = np.array([
    35, 48, 61, 72, 82,
    67, 45, 76, 88, 52
])

fig = go.Figure()

# ---------------------------------------------------------
# RISK
# ---------------------------------------------------------

if show_risk:

    fig.add_trace(
        go.Scattergeo(
            lat=lat,
            lon=lon,
            mode="markers",
            marker=dict(
                size=12,
                color=risk,
                colorscale="RdYlGn_r",
                cmin=0,
                cmax=100,
                colorbar=dict(
                    title="Risk"
                ),
                line=dict(
                    width=0.5,
                    color="#ffffff"
                )
            ),
            text=[
                f"Risk Score: {r}"
                for r in risk
            ],
            hovertemplate=
                "<b>%{text}</b><br>" +
                "Latitude: %{lat:.2f}<br>" +
                "Longitude: %{lon:.2f}" +
                "<extra></extra>",
            name="Environmental Risk"
        )
    )

# ---------------------------------------------------------
# SEA ICE DEMO LAYER
# ---------------------------------------------------------

if show_sea_ice:

    sea_ice = np.array([
        70, 75, 82, 91, 85,
        76, 88, 93, 80, 72
    ])

    fig.add_trace(
        go.Scattergeo(
            lat=lat,
            lon=lon,
            mode="markers",
            marker=dict(
                size=18,
                color=sea_ice,
                colorscale="Blues",
                cmin=0,
                cmax=100,
                opacity=0.65,
                colorbar=dict(
                    title="Sea Ice %"
                )
            ),
            text=[
                f"Sea Ice: {v}%"
                for v in sea_ice
            ],
            hovertemplate=
                "<b>%{text}</b><br>" +
                "Latitude: %{lat:.2f}<br>" +
                "Longitude: %{lon:.2f}" +
                "<extra></extra>",
            name="Sea Ice"
        )
    )

# ---------------------------------------------------------
# OCEAN CURRENT DEMO LAYER
# ---------------------------------------------------------

if show_currents:

    current_speed = np.array([
        0.35, 0.42, 0.55, 0.68, 0.74,
        0.63, 0.49, 0.81, 0.72, 0.58
    ])

    fig.add_trace(
        go.Scattergeo(
            lat=lat,
            lon=lon,
            mode="markers",
            marker=dict(
                size=10,
                color=current_speed,
                colorscale="Viridis",
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title="Current m/s"
                )
            ),
            text=[
                f"Current Speed: {v:.2f} m/s"
                for v in current_speed
            ],
            hovertemplate=
                "<b>%{text}</b><br>" +
                "Latitude: %{lat:.2f}<br>" +
                "Longitude: %{lon:.2f}" +
                "<extra></extra>",
            name="Ocean Current"
        )
    )

# ---------------------------------------------------------
# BATHYMETRY DEMO LAYER
# ---------------------------------------------------------

if show_bathymetry:

    bed_depth = np.array([
        -1200, -1800, -2500, -3200, -4100,
        -2900, -3500, -4300, -5100, -2200
    ])

    fig.add_trace(
        go.Scattergeo(
            lat=lat,
            lon=lon,
            mode="markers",
            marker=dict(
                size=14,
                color=bed_depth,
                colorscale="Cividis",
                colorbar=dict(
                    title="Bed Depth (m)"
                )
            ),
            text=[
                f"Bed Depth: {v} m"
                for v in bed_depth
            ],
            hovertemplate=
                "<b>%{text}</b><br>" +
                "Latitude: %{lat:.2f}<br>" +
                "Longitude: %{lon:.2f}" +
                "<extra></extra>",
            name="Bathymetry"
        )
    )

# ---------------------------------------------------------
# ICEBERG DEMO LAYER
# ---------------------------------------------------------

if show_icebergs:

    iceberg_lat = np.array([
        -62.5,
        -66.0,
        -69.5,
        -72.5,
        -75.0
    ])

    iceberg_lon = np.array([
        -55,
        -35,
        20,
        70,
        120
    ])

    iceberg_size = np.array([
        18,
        25,
        12,
        30,
        20
    ])

    fig.add_trace(
        go.Scattergeo(
            lat=iceberg_lat,
            lon=iceberg_lon,
            mode="markers",
            marker=dict(
                size=iceberg_size,
                symbol="diamond",
                color="#ffffff",
                line=dict(
                    width=2,
                    color="#075985"
                )
            ),
            text=[
                "Iceberg observation"
                for _ in iceberg_lat
            ],
            hovertemplate=
                "<b>%{text}</b><br>" +
                "Latitude: %{lat:.2f}<br>" +
                "Longitude: %{lon:.2f}" +
                "<extra></extra>",
            name="Icebergs"
        )
    )

# ---------------------------------------------------------
# ANTARCTIC MAP STYLE
# ---------------------------------------------------------

fig.update_geos(
    projection_type="stereographic",
    center=dict(
        lat=-90,
        lon=0
    ),
    projection_scale=2.8,
    showland=True,
    showocean=True,
    showcoastlines=True,
    showcountries=True,
    showframe=False,
    landcolor="#dbeafe",
    oceancolor="#eff6ff",
    coastlinecolor="#075985"
)

fig.update_layout(
    height=600,
    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="#102a43"
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="#bfdbfe",
        borderwidth=1,
        font=dict(
            color="#102a43"
        )
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="environmental_risk_map"
)

# =========================================================
# MAP LEGEND
# =========================================================

st.markdown(
    """
    **Map layers:**  
    🔴 High environmental risk  
    🟠 Moderate environmental risk  
    🟢 Lower environmental risk  
    ❄️ Sea ice concentration  
    🌊 Ocean currents  
    🧊 Iceberg observations  
    🌎 Bathymetry
    """
)

# =========================================================
# ENVIRONMENTAL CONDITIONS
# =========================================================

st.divider()

st.subheader("🌍 Environmental Conditions")

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Sea Ice Concentration",
        "82%"
    )

with c2:

    st.metric(
        "Ocean Current",
        "0.74 m/s"
    )

with c3:

    st.metric(
        "Iceberg Activity",
        "HIGH"
    )

with c4:

    st.metric(
        "Bed Depth",
        "-2,431 m"
    )

# =========================================================
# SEA ICE ANALYSIS
# =========================================================

st.divider()

st.subheader("❄️ Sea Ice Analysis")

ice_c1, ice_c2, ice_c3 = st.columns(3)

with ice_c1:

    st.metric(
        "Current Concentration",
        "82%"
    )

with ice_c2:

    st.metric(
        "Ice Condition",
        "Dense"
    )

with ice_c3:

    st.metric(
        "Navigation Impact",
        "High"
    )

st.info(
    "High sea-ice concentration can increase navigation difficulty "
    "and should be considered when evaluating environmentally "
    "sensitive routes."
)

# =========================================================
# ICEBERG MONITORING
# =========================================================

st.divider()

st.subheader("🧊 Iceberg Monitoring")

ib_c1, ib_c2, ib_c3 = st.columns(3)

with ib_c1:

    st.metric(
        "Observed Icebergs",
        "24"
    )

with ib_c2:

    st.metric(
        "High-Risk Proximity",
        "7"
    )

with ib_c3:

    st.metric(
        "Iceberg Activity",
        "HIGH"
    )

st.info(
    "Iceberg locations should be checked against the planned "
    "route before navigation or field operations."
)

# =========================================================
# LOCATION ASSESSMENT
# =========================================================

st.divider()

st.subheader("📍 Location Assessment")

st.caption(
    "Select a real Antarctic location or enter coordinates "
    "for environmental assessment."
)

c1, c2, c3 = st.columns([1, 1, 1])

with c1:

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=-48.0,
        value=float(region_lat),
        step=0.1
    )

with c2:

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=float(region_lon),
        step=0.1
    )

with c3:

    st.write("")
    st.write("")

    assess = st.button(
        "🔍 Assess Location",
        use_container_width=True
    )

if assess:

    # Demo calculation
    # Replace with actual integrated environmental
    # dataset values later.

    location_risk = 72

    st.divider()

    st.markdown(
        f"### 📍 {latitude:.2f}°, {longitude:.2f}°"
    )

    if location_risk >= 70:

        st.markdown(
            f"""
            <div class="risk-high">
                <h2>🔴 HIGH RISK</h2>
                <h3>{location_risk} / 100</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif location_risk >= 40:

        st.markdown(
            f"""
            <div class="risk-medium">
                <h2>🟠 MODERATE RISK</h2>
                <h3>{location_risk} / 100</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="risk-low">
                <h2>🟢 LOW RISK</h2>
                <h3>{location_risk} / 100</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Why?")

    st.write(
        "• High sea-ice concentration"
    )

    st.write(
        "• Elevated iceberg activity"
    )

    st.write(
        "• Strong ocean current"
    )

# =========================================================
# ROUTE PLANNER
# =========================================================

st.divider()

st.subheader("🧭 Antarctic Route Planner")

st.caption(
    "Plan a route between real Antarctic geographic locations "
    "using environmental risk information."
)

# ---------------------------------------------------------
# SOURCE / DESTINATION
# ---------------------------------------------------------

c1, c2 = st.columns(2)

with c1:

    st.markdown("### 📍 Source")

    source_name = st.selectbox(
        "Select source location",
        list(antarctic_locations.keys()),
        index=0,
        key="route_source"
    )

    source_lat, source_lon = (
        antarctic_locations[source_name]
    )

    st.caption(
        f"Latitude: {source_lat:.2f}° | "
        f"Longitude: {source_lon:.2f}°"
    )

with c2:

    st.markdown("### 🎯 Destination")

    destination_options = [
        x
        for x in antarctic_locations.keys()
        if x != source_name
    ]

    destination_name = st.selectbox(
        "Select destination location",
        destination_options,
        index=0,
        key="route_destination"
    )

    destination_lat, destination_lon = (
        antarctic_locations[destination_name]
    )

    st.caption(
        f"Latitude: {destination_lat:.2f}° | "
        f"Longitude: {destination_lon:.2f}°"
    )

route_button = st.button(
    "🧭 Calculate Route",
    use_container_width=True
)

if route_button:

    # -----------------------------------------------------
    # GENERATE ROUTE
    # -----------------------------------------------------

    route_lats = np.linspace(
        source_lat,
        destination_lat,
        100
    )

    route_lons = np.linspace(
        source_lon,
        destination_lon,
        100
    )

    # -----------------------------------------------------
    # HAVERSINE DISTANCE
    # -----------------------------------------------------

    lat1 = np.radians(source_lat)
    lat2 = np.radians(destination_lat)

    dlat = np.radians(
        destination_lat - source_lat
    )

    dlon = np.radians(
        destination_lon - source_lon
    )

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )

    earth_radius = 6371

    distance_km = (
        2
        *
        earth_radius
        *
        np.arcsin(
            np.sqrt(a)
        )
    )

    # -----------------------------------------------------
    # ROUTE MAP
    # -----------------------------------------------------

    route_fig = go.Figure()

    route_fig.add_trace(
        go.Scattergeo(
            lat=route_lats,
            lon=route_lons,
            mode="lines",
            line=dict(
                width=4
            ),
            name="Planned Route"
        )
    )

    route_fig.add_trace(
        go.Scattergeo(
            lat=[source_lat],
            lon=[source_lon],
            mode="markers+text",
            marker=dict(
                size=15,
                symbol="circle"
            ),
            text=["SOURCE"],
            textposition="top center",
            name=source_name
        )
    )

    route_fig.add_trace(
        go.Scattergeo(
            lat=[destination_lat],
            lon=[destination_lon],
            mode="markers+text",
            marker=dict(
                size=15,
                symbol="diamond"
            ),
            text=["DESTINATION"],
            textposition="top center",
            name=destination_name
        )
    )

    route_fig.update_geos(
        projection_type="stereographic",
        center=dict(
            lat=-90,
            lon=0
        ),
        projection_scale=2.8,
        showland=True,
        showocean=True,
        showcoastlines=True,
        showcountries=True,
        showframe=False,
        landcolor="#dbeafe",
        oceancolor="#eff6ff",
        coastlinecolor="#075985"
    )

    route_fig.update_layout(
        height=550,
        margin=dict(
            l=0,
            r=0,
            t=20,
            b=0
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#102a43"
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="#bfdbfe",
            borderwidth=1,
            font=dict(
                color="#102a43"
            )
        )
    )

    st.plotly_chart(
        route_fig,
        use_container_width=True,
        key="route_map"
    )

    # -----------------------------------------------------
    # ROUTE INFORMATION
    # -----------------------------------------------------

    st.markdown("### 📊 Route Information")

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Source",
            source_name
        )

    with r2:

        st.metric(
            "Destination",
            destination_name
        )

    with r3:

        st.metric(
            "Approx. Distance",
            f"{distance_km:,.0f} km"
        )

    st.info(
        "⚠️ The current route is a geographic visualization. "
        "The integrated version will evaluate sea ice, iceberg "
        "activity, ocean currents and bathymetry along the route."
    )

# =========================================================
# DATASET STATUS
# =========================================================

st.divider()

st.subheader("📡 Environmental Data Status")

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.success("❄️ Sea Ice\nAvailable")

with d2:
    st.success("🌊 Ocean Current\nAvailable")

with d3:
    st.success("🌎 Geography & Bathymetry\nAvailable")

with d4:
    st.warning("🧊 Iceberg\nMonitoring")

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Antarctica Environmental Risk Explorer • "
    "Integrated marine, cryosphere and seabed analysis"
)
