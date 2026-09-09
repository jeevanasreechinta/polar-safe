from pathlib import Path

app_code = r'''import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# POLAR-SAFE
# AI-Enabled Antarctic Navigation Decision Support System
# ============================================================

st.set_page_config(
    page_title="POLAR-SAFE | Antarctic Navigation Intelligence",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0b1020; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .hero {
        padding: 1.4rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg,#101a35 0%,#16284a 50%,#10243c 100%);
        border: 1px solid #29456b;
        margin-bottom: 1rem;
    }
    .hero-title { font-size: 2.35rem; font-weight: 800; }
    .hero-subtitle { font-size: 1rem; opacity: .86; }
    .warning-box {
        padding: .85rem 1.1rem;
        border-radius: 12px;
        background: #332b12;
        border: 1px solid #806d22;
        margin-bottom: 1rem;
    }
    .route-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid #29456b;
        background: #111a2c;
        margin-bottom: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# DATA DISCOVERY
# Works with either the expected project folders OR the files
# placed beside app.py.
# ============================================================

def find_file(*relative_paths):
    for rel in relative_paths:
        p = os.path.join(BASE_DIR, rel)
        if os.path.isfile(p):
            return p

    # Last-resort recursive lookup by filename.
    wanted = {os.path.basename(x).lower() for x in relative_paths}
    for root, _, files in os.walk(BASE_DIR):
        for f in files:
            if f.lower() in wanted:
                return os.path.join(root, f)
    return None


FILES = {
    "sea_ice": find_file(
        "sea_ice/antarctic_sea_ice_risk_2025-04-16.csv",
        "antarctic_sea_ice_risk_2025-04-16.csv",
    ),
    "iceberg_risk": find_file(
        "iceberg/iceberg_risk_layer.csv",
        "iceberg_risk_layer.csv",
    ),
    "iceberg_trajectory": find_file(
        "iceberg/iceberg_trajectory_predictions.csv",
        "iceberg_trajectory_predictions.csv",
    ),
    "combined": find_file(
        "combined/antarctic_combined_navigation_risk_2025-04-16.csv",
        "antarctic_combined_navigation_risk_2025-04-16.csv",
    ),
    "shortest": find_file("routes/shortest_route.csv", "shortest_route.csv"),
    "balanced": find_file("routes/balanced_route.csv", "balanced_route.csv"),
    "risk_aware": find_file("routes/risk_aware_route.csv", "risk_aware_route.csv"),
    "comparison": find_file("routes/route_comparison.csv", "route_comparison.csv"),
    "dashboard": find_file("dashboard/dashboard_data.json", "dashboard_data.json"),
    "bathymetry": find_file(
        "bathymetry/bathymetry_depth_256.npy",
        "bathymetry_depth_256.npy",
    ),
    "land_mask": find_file(
        "bathymetry/land_mask_256.npy",
        "land_mask_256.npy",
    ),
}


@st.cache_data
def load_csv(path):
    if not path:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {os.path.basename(path)}: {exc}")
        return pd.DataFrame()


@st.cache_data
def load_json(path):
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.warning(f"Could not read {os.path.basename(path)}: {exc}")
        return {}


@st.cache_data
def load_npy(path):
    if not path:
        return None
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:
        st.warning(f"Could not read {os.path.basename(path)}: {exc}")
        return None


sea_ice = load_csv(FILES["sea_ice"])
iceberg_risk = load_csv(FILES["iceberg_risk"])
iceberg_trajectory = load_csv(FILES["iceberg_trajectory"])
combined = load_csv(FILES["combined"])

shortest_route = load_csv(FILES["shortest"])
balanced_route = load_csv(FILES["balanced"])
risk_aware_route = load_csv(FILES["risk_aware"])
route_comparison = load_csv(FILES["comparison"])

dashboard_data = load_json(FILES["dashboard"])
bathymetry = load_npy(FILES["bathymetry"])
land_mask = load_npy(FILES["land_mask"])


# ============================================================
# Helpers
# ============================================================

def numeric(df, col):
    if col not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def first_existing_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def clean_coordinates(df):
    if df.empty:
        return df

    out = df.copy()
    for col in ("latitude", "longitude"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "longitude" in out.columns:
        out["longitude"] = ((out["longitude"] + 180) % 360) - 180

    required = [c for c in ("latitude", "longitude") if c in out.columns]
    if required:
        out = out.dropna(subset=required)

    return out


def risk_column(df):
    return first_existing_column(
        df,
        [
            "combined_risk_score",
            "combined_navigation_risk_score",
            "combined_risk",
            "sea_ice_risk_score",
            "iceberg_risk_score",
            "risk_score",
        ],
    )


def route_distance(df):
    """Distance from the route file itself, never from the number of rows."""
    if df.empty:
        return np.nan

    col = first_existing_column(
        df,
        [
            "distance_km",
            "route_distance_km",
            "cumulative_distance_km",
            "total_distance_km",
        ],
    )
    if col:
        values = numeric(df, col).dropna()
        if not values.empty:
            return float(values.max())

    # Route waypoint files do not contain distance, so calculate the
    # great-circle distance from consecutive waypoints.
    if {"latitude", "longitude"}.issubset(df.columns):
        d = df.sort_values("route_order") if "route_order" in df.columns else df
        lat = np.radians(d["latitude"].to_numpy())
        lon = np.radians(d["longitude"].to_numpy())
        if len(lat) > 1:
            dlat = lat[1:] - lat[:-1]
            dlon = lon[1:] - lon[:-1]
            a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
            return float(np.sum(6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))))
    return np.nan


def route_stats(name, df, comparison_df=None):
    # Prefer the official precomputed comparison dataset when available.
    if comparison_df is not None and not comparison_df.empty and "route" in comparison_df.columns:
        hit = comparison_df[
            comparison_df["route"].astype(str).str.strip().str.lower() == name.lower()
        ]
        if not hit.empty:
            r = hit.iloc[0]
            return {
                "Route": name,
                "Points": int(r["points"]) if pd.notna(r.get("points")) else len(df),
                "Distance (km)": float(r["distance_km"]),
                "Mean Risk": float(r["mean_risk"]),
                "Max Risk": float(r["max_risk"]),
                "High Risk %": float(r["high_risk_percent"]),
                "Very High %": float(r["very_high_risk_percent"]),
            }

    rcol = risk_column(df)
    risks = numeric(df, rcol).dropna() if rcol else pd.Series(dtype=float)
    levels = (
        df["combined_risk_level"].astype(str)
        if "combined_risk_level" in df.columns
        else pd.Series(dtype=str)
    )

    return {
        "Route": name,
        "Points": len(df),
        "Distance (km)": route_distance(df),
        "Mean Risk": float(risks.mean()) if not risks.empty else np.nan,
        "Max Risk": float(risks.max()) if not risks.empty else np.nan,
        "High Risk %": float(levels.str.lower().eq("high").mean() * 100) if len(levels) else np.nan,
        "Very High %": float(levels.str.lower().eq("very high").mean() * 100) if len(levels) else np.nan,
    }


def risk_distribution(df):
    if df.empty:
        return pd.DataFrame(columns=["Risk Level", "Cells"])

    level_col = first_existing_column(
        df, ["combined_risk_level", "sea_ice_risk_level", "iceberg_risk_level"]
    )
    if level_col:
        return (
            df[level_col]
            .astype(str)
            .value_counts()
            .rename_axis("Risk Level")
            .reset_index(name="Cells")
        )

    rcol = risk_column(df)
    if not rcol:
        return pd.DataFrame(columns=["Risk Level", "Cells"])

    values = numeric(df, rcol)
    labels = pd.cut(
        values,
        bins=[-np.inf, 25, 50, 75, np.inf],
        labels=["Low", "Moderate", "High", "Very High"],
    )
    return labels.value_counts().rename_axis("Risk Level").reset_index(name="Cells")


# Clean after loading.
sea_ice = clean_coordinates(sea_ice)
iceberg_risk = clean_coordinates(iceberg_risk)
iceberg_trajectory = clean_coordinates(iceberg_trajectory)
combined = clean_coordinates(combined)
shortest_route = clean_coordinates(shortest_route)
balanced_route = clean_coordinates(balanced_route)
risk_aware_route = clean_coordinates(risk_aware_route)


# ============================================================
# Build a combined layer if the combined CSV is not present.
# This fuses sea-ice and iceberg risk by nearest spatial cell.
# If the official combined CSV exists, it is always used.
# ============================================================

def build_combined_layer(sea_ice_df, iceberg_df):
    if not sea_ice_df.empty:
        s = sea_ice_df.copy()
        s_col = first_existing_column(
            s, ["sea_ice_risk_score", "risk_score", "sea_ice_risk"]
        )
        if s_col:
            s["__risk"] = numeric(s, s_col)
        else:
            sic_col = first_existing_column(
                s, ["predicted_sea_ice_concentration", "sea_ice_concentration"]
            )
            if sic_col:
                sic = numeric(s, sic_col).clip(0, 100)
                s["__risk"] = sic
            else:
                s["__risk"] = 0.0

        # Add iceberg risk to each sea-ice grid point using nearest
        # observed/predicted iceberg location.
        if not iceberg_df.empty and {"latitude", "longitude"}.issubset(iceberg_df.columns):
            i = iceberg_df.copy()
            icol = first_existing_column(i, ["iceberg_risk_score", "risk_score"])
            if icol:
                i = i[["latitude", "longitude", icol]].dropna()
                if not i.empty:
                    # Efficient nearest-neighbour approximation on lat/lon.
                    s_xy = s[["latitude", "longitude"]].to_numpy(float)
                    i_xy = i[["latitude", "longitude"]].to_numpy(float)
                    ir = numeric(i, icol).to_numpy(float)
                    result = np.zeros(len(s_xy), dtype=float)
                    chunk = 2000
                    for start in range(0, len(s_xy), chunk):
                        q = s_xy[start:start + chunk]
                        dist2 = (q[:, None, 0] - i_xy[None, :, 0]) ** 2 + (
                            (q[:, None, 1] - i_xy[None, :, 1]) * np.cos(np.radians(q[:, None, 0]))
                        ) ** 2
                        result[start:start + len(q)] = ir[np.argmin(dist2, axis=1)]
                    s["nearby_iceberg_risk"] = result
                else:
                    s["nearby_iceberg_risk"] = 0.0
            else:
                s["nearby_iceberg_risk"] = 0.0
        else:
            s["nearby_iceberg_risk"] = 0.0

        # Use a conservative weighted fusion.
        s["combined_risk_score"] = (
            0.70 * s["__risk"].clip(0, 100)
            + 0.30 * s["nearby_iceberg_risk"].clip(0, 100)
        )
        s["combined_risk_score"] = s["combined_risk_score"].clip(0, 100)
        s["combined_risk_level"] = pd.cut(
            s["combined_risk_score"],
            bins=[-np.inf, 25, 50, 75, np.inf],
            labels=["Low", "Moderate", "High", "Very High"],
        ).astype(str)
        return s.drop(columns=["__risk"], errors="ignore")

    return pd.DataFrame()


if combined.empty:
    combined = build_combined_layer(sea_ice, iceberg_risk)


# ============================================================
# Polar map
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

    if df.empty or not {"latitude", "longitude"}.issubset(df.columns):
        fig.add_annotation(
            text="No data available for this layer.",
            x=.5, y=.5, xref="paper", yref="paper", showarrow=False
        )
    else:
        values = numeric(df, risk_col).fillna(0) if risk_col else pd.Series(0.0, index=df.index)

        hover_cols = []
        for c in [
            "iceberg_id", "datetime", "sea_ice_concentration",
            "predicted_sea_ice_concentration", "nearby_iceberg_risk",
            "trajectory_uncertainty", "prediction_error_km",
        ]:
            if c in df.columns:
                hover_cols.append(c)

        custom = df[hover_cols].to_numpy() if hover_cols else None
        template = "Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>Risk: %{marker.color:.1f}/100"
        if hover_cols:
            for idx, c in enumerate(hover_cols):
                template += f"<br>{c}: %{{customdata[{idx}]}}"
        template += "<extra></extra>"

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
                    opacity=.78,
                    colorbar=dict(title="Risk<br>0–100"),
                ),
                customdata=custom,
                hovertemplate=template,
                name="Risk",
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
        legend=dict(bgcolor="rgba(0,0,0,.3)"),
    )
    return fig


def add_route_trace(fig, df, name, width):
    if df.empty or not {"latitude", "longitude"}.issubset(df.columns):
        return

    d = df.sort_values("route_order") if "route_order" in df.columns else df
    fig.add_trace(
        go.Scattergeo(
            lon=d["longitude"],
            lat=d["latitude"],
            mode="lines+markers",
            line=dict(width=width),
            marker=dict(size=4),
            name=name,
            customdata=d["combined_risk_score"].to_numpy() if "combined_risk_score" in d else None,
            hovertemplate=(
                "Latitude: %{lat:.2f}<br>"
                "Longitude: %{lon:.2f}<br>"
                "Risk: %{customdata:.1f}/100"
                f"<extra>{name}</extra>"
            ) if "combined_risk_score" in d else (
                "Latitude: %{lat:.2f}<br>Longitude: %{lon:.2f}"
                f"<extra>{name}</extra>"
            ),
        )
    )


def create_route_map(origin_coords, destination_coords, origin_name, destination_name):
    fig = go.Figure()
    add_route_trace(fig, shortest_route, "Shortest Route", 3)
    add_route_trace(fig, balanced_route, "Balanced Route", 4)
    add_route_trace(fig, risk_aware_route, "Risk-Aware Route", 5)

    fig.add_trace(
        go.Scattergeo(
            lon=[origin_coords[1]], lat=[origin_coords[0]],
            mode="markers+text", marker=dict(size=12, symbol="star"),
            text=[origin_name], textposition="top center", name="Origin",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=[destination_coords[1]], lat=[destination_coords[0]],
            mode="markers+text", marker=dict(size=12, symbol="diamond"),
            text=[destination_name], textposition="top center", name="Destination",
        )
    )

    fig.update_geos(**polar_geo_settings())
    fig.update_layout(
        title=f"{origin_name} → {destination_name} Route Comparison",
        height=650,
        margin=dict(l=0, r=0, t=55, b=0),
        paper_bgcolor="#07111f",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,.35)"),
    )
    return fig


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🧊 POLAR-SAFE</div>
        <div class="hero-subtitle">AI-Powered Antarctic Navigation & Risk Intelligence System</div>
        <div class="hero-subtitle">Predict the Ice • Track the Iceberg • Navigate Safely</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="warning-box">
        ⚠️ <b>Prototype Decision-Support System:</b>
        POLAR-SAFE demonstrates AI-based environmental risk prediction and route
        optimization. It supports professional navigation decisions and does not
        replace qualified navigators.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("🧭 Navigation Controls")

origin_options = {
    "Bharati Station": (-69.4068, 76.19525),
    "Maitri Station": (-70.76444, 11.73333),
    "Custom Location": None,
}
destination_options = {
    "Maitri Station": (-70.76444, 11.73333),
    "Bharati Station": (-69.4068, 76.19525),
    "Custom Location": None,
}

origin = st.sidebar.selectbox("Origin", list(origin_options.keys()))
destination = st.sidebar.selectbox("Destination", list(destination_options.keys()))

if origin == "Custom Location":
    origin_coords = (
        st.sidebar.number_input("Origin Latitude", -90.0, 90.0, -70.0, .1),
        st.sidebar.number_input("Origin Longitude", -180.0, 180.0, 30.0, .1),
    )
else:
    origin_coords = origin_options[origin]

if destination == "Custom Location":
    destination_coords = (
        st.sidebar.number_input("Destination Latitude", -90.0, 90.0, -70.0, .1),
        st.sidebar.number_input("Destination Longitude", -180.0, 180.0, 20.0, .1),
    )
else:
    destination_coords = destination_options[destination]

layer = st.sidebar.radio(
    "Intelligence Layer",
    [
        "Combined Navigation Risk",
        "Sea-Ice Risk",
        "Iceberg Risk",
        "Routes",
        "Bathymetry",
    ],
)

st.sidebar.markdown("---")
safety_preference = st.sidebar.slider(
    "Safety Preference", 0, 100, 80,
    help="Used to choose the stored route that best matches the selected safety priority."
)

st.sidebar.caption(f"Safety Preference: {safety_preference}%")
st.sidebar.caption("POLAR-SAFE • SIH26059")


# ============================================================
# KPI values
# Prefer dashboard_data.json because it is the project's
# authoritative precomputed summary; otherwise derive values
# directly from the loaded datasets.
# ============================================================

risk_summary = dashboard_data.get("risk_summary", {})
iceberg_summary = dashboard_data.get("iceberg_summary", {})

if risk_summary:
    average_risk = float(risk_summary.get("mean_combined_risk", np.nan))
    maximum_risk = float(risk_summary.get("maximum_combined_risk", np.nan))
else:
    ccol = risk_column(combined)
    vals = numeric(combined, ccol).dropna() if ccol else pd.Series(dtype=float)
    average_risk = float(vals.mean()) if len(vals) else 0.0
    maximum_risk = float(vals.max()) if len(vals) else 0.0

if not iceberg_risk.empty and "iceberg_id" in iceberg_risk.columns:
    tracked_icebergs = int(iceberg_risk["iceberg_id"].nunique())
else:
    tracked_icebergs = int(iceberg_summary.get("unique_icebergs", 0))

if risk_summary.get("risk_distribution"):
    high_risk_cells = int(
        risk_summary["risk_distribution"].get("High", 0)
        + risk_summary["risk_distribution"].get("Very High", 0)
    )
else:
    dist = risk_distribution(combined)
    high_risk_cells = int(
        dist.loc[dist["Risk Level"].isin(["High", "Very High"]), "Cells"].sum()
    ) if not dist.empty else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Average Combined Risk", f"{average_risk:.2f}/100")
k2.metric("Maximum Combined Risk", f"{maximum_risk:.2f}/100")
k3.metric("Tracked Icebergs", f"{tracked_icebergs:,}")
k4.metric("High + Very High Cells", f"{high_risk_cells:,}")


# ============================================================
# COMBINED NAVIGATION RISK
# ============================================================

if layer == "Combined Navigation Risk":
    st.markdown("### 🌐 Combined Navigation Risk Intelligence")
    st.write(
        "The navigation layer combines sea-ice risk and nearby iceberg hazard. "
        "The official combined dataset is used when available; otherwise the app "
        "builds a fused layer from the available sea-ice and iceberg datasets."
    )

    if combined.empty:
        st.error(
            "No combined/sea-ice risk CSV was found. Add the combined dataset "
            "or sea-ice dataset to the same project folder as app.py."
        )
    else:
        ccol = risk_column(combined)
        st.plotly_chart(
            create_risk_map(
                combined, ccol, "Antarctic Combined Navigation Risk", 6
            ),
            use_container_width=True,
        )

        a, b, c = st.columns(3)
        a.metric("Risk Surface Points", f"{len(combined):,}")
        cvals = numeric(combined, ccol).dropna() if ccol else pd.Series(dtype=float)
        b.metric("Dataset Mean Risk", f"{cvals.mean():.2f}" if len(cvals) else "N/A")
        c.metric("Dataset Peak Risk", f"{cvals.max():.2f}" if len(cvals) else "N/A")

        dist = risk_distribution(combined)
        if not dist.empty:
            st.markdown("### Risk Distribution")
            fig = px.bar(
                dist, x="Risk Level", y="Cells", text="Cells",
                title="Navigation Risk Categories"
            )
            fig.update_layout(
                height=400,
                paper_bgcolor="#07111f",
                plot_bgcolor="#07111f",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Combined dataset preview"):
            st.dataframe(combined.head(100), use_container_width=True, hide_index=True)


# ============================================================
# SEA ICE
# ============================================================

elif layer == "Sea-Ice Risk":
    st.markdown("### 🧊 AI Sea-Ice Risk Forecast")
    st.write(
        "Sea-ice concentration and sea-ice risk are displayed directly from "
        "the sea-ice dataset."
    )

    if sea_ice.empty:
        st.error(
            "Sea-ice dataset not found. Expected "
            "`sea_ice/antarctic_sea_ice_risk_2025-04-16.csv`."
        )
    else:
        scol = first_existing_column(
            sea_ice,
            ["sea_ice_risk_score", "risk_score", "sea_ice_risk"]
        )
        st.plotly_chart(
            create_risk_map(sea_ice, scol, "Antarctic Sea-Ice Risk", 6),
            use_container_width=True,
        )

        predicted_col = first_existing_column(
            sea_ice,
            ["predicted_sea_ice_concentration", "sea_ice_concentration"]
        )

        a, b, c = st.columns(3)
        a.metric("Grid Cells", f"{len(sea_ice):,}")

        if predicted_col:
            mean_sic = numeric(sea_ice, predicted_col).mean()
            b.metric("Mean Sea-Ice Concentration", f"{mean_sic:.2f}%")
        else:
            b.metric("Mean Sea-Ice Concentration", "N/A")

        if scol:
            max_sic_risk = numeric(sea_ice, scol).max()
            c.metric("Maximum Ice Risk", f"{max_sic_risk:.2f}")
        else:
            c.metric("Maximum Ice Risk", "N/A")

        with st.expander("Sea-ice dataset preview"):
            st.dataframe(sea_ice.head(100), use_container_width=True, hide_index=True)


# ============================================================
# ICEBERG RISK
# ============================================================

elif layer == "Iceberg Risk":
    st.markdown("### 🧊 Iceberg Trajectory Intelligence")
    st.write(
        "Observed iceberg positions, AI-predicted positions, trajectory uncertainty, "
        "movement risk and final iceberg risk are shown from the supplied datasets."
    )

    if iceberg_risk.empty:
        st.error("iceberg_risk_layer.csv was not found.")
    else:
        icol = risk_column(iceberg_risk)
        st.plotly_chart(
            create_risk_map(iceberg_risk, icol, "Antarctic Iceberg Risk", 8),
            use_container_width=True,
        )

        a, b, c, d = st.columns(4)
        a.metric(
            "Tracked Icebergs",
            f"{iceberg_risk['iceberg_id'].nunique():,}"
            if "iceberg_id" in iceberg_risk else "N/A",
        )
        if "prediction_error_km" in iceberg_risk:
            d_err = numeric(iceberg_risk, "prediction_error_km").dropna()
            b.metric("Mean Prediction Error", f"{d_err.mean():.2f} km")
        else:
            b.metric("Mean Prediction Error", "N/A")

        if "trajectory_uncertainty" in iceberg_risk:
            unc = numeric(iceberg_risk, "trajectory_uncertainty").dropna()
            c.metric("Mean Trajectory Uncertainty", f"{unc.mean():.3f}")
        else:
            c.metric("Mean Trajectory Uncertainty", "N/A")

        if icol:
            rv = numeric(iceberg_risk, icol).dropna()
            d.metric("Mean Iceberg Risk", f"{rv.mean():.2f}/100")
        else:
            d.metric("Mean Iceberg Risk", "N/A")

        if "iceberg_risk_level" in iceberg_risk:
            dist = (
                iceberg_risk["iceberg_risk_level"]
                .astype(str)
                .value_counts()
                .rename_axis("Risk Level")
                .reset_index(name="Observations")
            )
            fig = px.bar(
                dist, x="Risk Level", y="Observations",
                text="Observations", title="Iceberg Risk Categories"
            )
            fig.update_layout(
                height=400,
                paper_bgcolor="#07111f",
                plot_bgcolor="#07111f",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Iceberg risk dataset preview"):
            st.dataframe(iceberg_risk.head(100), use_container_width=True, hide_index=True)


# ============================================================
# ROUTES
# ============================================================

elif layer == "Routes":
    st.markdown("### 🚢 AI Route Optimization")
    st.write(f"Voyage scenario: **{origin} → {destination}**")

    routes = {
        "Shortest": shortest_route,
        "Balanced": balanced_route,
        "Risk-Aware": risk_aware_route,
    }

    # Use the actual route_comparison.csv values if supplied.
    stats = pd.DataFrame(
        [route_stats(name, df, route_comparison) for name, df in routes.items()]
    )

    # Safety preference chooses among the three already-generated routes.
    # This avoids pretending that a 256x256 bathymetry grid covers the
    # Bharati/Maitri longitudes when the supplied route files are the
    # authoritative voyage outputs.
    if safety_preference < 35:
        recommended = "Shortest"
    elif safety_preference < 70:
        recommended = "Balanced"
    else:
        recommended = "Risk-Aware"

    rec = stats[stats["Route"] == recommended].iloc[0]

    st.success(
        f"Recommended route for safety preference {safety_preference}%: "
        f"**{recommended} Route**"
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Recommended Distance", f"{rec['Distance (km)']:,.2f} km")
    r2.metric("Mean Route Risk", f"{rec['Mean Risk']:.2f}/100")
    r3.metric("Maximum Route Risk", f"{rec['Max Risk']:.2f}/100")
    r4.metric("Waypoints", f"{int(rec['Points']):,}")

    st.plotly_chart(
        create_route_map(
            origin_coords, destination_coords, origin, destination
        ),
        use_container_width=True,
    )

    st.markdown("### 📊 Route Comparison")
    display_stats = stats.copy()
    for col in ["Distance (km)", "Mean Risk", "Max Risk", "High Risk %", "Very High %"]:
        display_stats[col] = display_stats[col].round(2)
    st.dataframe(display_stats, use_container_width=True, hide_index=True)

    # Direct values from dashboard_data.json, if present.
    decision = dashboard_data.get("decision", {})
    if decision:
        st.markdown("### 🎯 Project Decision")
        d1, d2, d3 = st.columns(3)
        d1.metric("Official Recommendation", decision.get("recommended_route", "N/A"))
        d2.metric(
            "Risk Reduction vs Shortest",
            f"{float(decision.get('risk_reduction_vs_shortest_percent', np.nan)):.2f}%"
        )
        d3.metric(
            "Distance Increase",
            f"{float(decision.get('distance_increase_percent', np.nan)):.2f}%"
        )

    st.markdown(
        """
        <div class="route-card">
            <h3>🧠 Why Risk-Aware?</h3>
            <p>
            The Risk-Aware route trades additional travel distance for a much
            lower accumulated environmental risk. For the supplied project
            results, it is the recommended route when safety is prioritized.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not route_comparison.empty:
        with st.expander("Raw route_comparison.csv"):
            st.dataframe(route_comparison, use_container_width=True, hide_index=True)


# ============================================================
# BATHYMETRY
# ============================================================

elif layer == "Bathymetry":
    st.markdown("### 🌊 Antarctic Bathymetry & Land Mask")
    st.write(
        "The supplied 256×256 bathymetry grid and land mask are displayed "
        "as a geographic-depth support layer."
    )

    if bathymetry is None:
        st.error("bathymetry_depth_256.npy was not found.")
    else:
        depth = np.asarray(bathymetry, dtype=float)
        masked_depth = depth.copy()

        if land_mask is not None and np.shape(land_mask) == depth.shape:
            masked_depth[np.asarray(land_mask) == 1] = np.nan

        fig = go.Figure(
            go.Heatmap(
                z=masked_depth,
                colorscale="Viridis",
                colorbar=dict(title="Depth / Elevation"),
                hovertemplate="Grid row: %{y}<br>Grid column: %{x}<br>Value: %{z:.1f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Bathymetry Depth Grid",
            height=600,
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            font=dict(color="white"),
            xaxis_title="Grid Longitude Index",
            yaxis_title="Grid Latitude Index",
        )
        st.plotly_chart(fig, use_container_width=True)

        a, b, c = st.columns(3)
        a.metric("Grid Size", f"{depth.shape[0]} × {depth.shape[1]}")
        b.metric("Minimum Value", f"{np.nanmin(depth):.2f}")
        c.metric("Maximum Value", f"{np.nanmax(depth):.2f}")

        if land_mask is not None and np.shape(land_mask) == depth.shape:
            land_pct = float(np.mean(np.asarray(land_mask) == 1) * 100)
            st.info(f"Land-mask coverage: {land_pct:.2f}% of grid cells are marked as land.")


# ============================================================
# ICEBERG TRAJECTORY EXPLORER
# Always available below the selected layer.
# ============================================================

st.markdown("---")
st.markdown("### 📍 Iceberg Trajectory Explorer")

if iceberg_trajectory.empty or "iceberg_id" not in iceberg_trajectory.columns:
    st.info("Iceberg trajectory prediction dataset is not available.")
else:
    iceberg_ids = sorted(iceberg_trajectory["iceberg_id"].dropna().unique().tolist())

    if iceberg_ids:
        selected_iceberg = st.selectbox("Select Iceberg", iceberg_ids)
        selected = iceberg_trajectory[
            iceberg_trajectory["iceberg_id"] == selected_iceberg
        ].copy()

        if "datetime" in selected.columns:
            selected["datetime"] = pd.to_datetime(selected["datetime"], errors="coerce")
            selected = selected.sort_values("datetime")

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
                    "Longitude: %{lon:.3f}<br>"
                    "<extra>Observed</extra>"
                ),
            )
        )

        if {"predicted_latitude", "predicted_longitude"}.issubset(selected.columns):
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
                        "Longitude: %{lon:.3f}<br>"
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
        st.plotly_chart(fig, use_container_width=True)

        a, b, c = st.columns(3)
        a.metric("Trajectory Records", f"{len(selected):,}")

        if "prediction_error_km" in selected:
            err = numeric(selected, "prediction_error_km").dropna()
            b.metric("Mean Prediction Error", f"{err.mean():.2f} km")
            c.metric("Maximum Prediction Error", f"{err.max():.2f} km")
        else:
            b.metric("Mean Prediction Error", "N/A")
            c.metric("Maximum Prediction Error", "N/A")


# ============================================================
# Decision Summary
# ============================================================

st.markdown("---")
st.markdown("### 🎯 Navigation Decision Summary")

a, b, c = st.columns(3)
a.markdown(
    """
    ### 1️⃣ Predict
    **Sea-Ice AI**

    Forecast sea-ice concentration and convert it into a navigation-risk surface.
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

    Compare shortest, balanced and risk-aware routes using environmental risk.
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
    **A* / Route Optimization**
    ↓
    **Navigation Decision Support**
    """
)

# -----------------------------
# Dataset status
# -----------------------------
with st.expander("📦 Dataset Status"):
    status_rows = []
    for key, path in FILES.items():
        status_rows.append({
            "Dataset": key,
            "Status": "Loaded" if path else "Missing",
            "File": os.path.basename(path) if path else "Not found",
        })
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "POLAR-SAFE • SIH26059 • AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, "
    "and Navigation Decision Support System"
)
st.caption(
    "Prototype for demonstration and research decision support. "
    "Final navigation decisions remain with qualified operators."
)
'''

out = Path("/mnt/data/app.py")
out.write_text(app_code, encoding="utf-8")

# Syntax check and quick sanity checks.
import py_compile
py_compile.compile(str(out), doraise=True)
print(f"Created and syntax-checked: {out}")
print(f"Lines: {len(app_code.splitlines())}")
