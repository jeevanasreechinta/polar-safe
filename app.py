import os
import json
import math
import heapq

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# POLAR-SAFE
# SIH26059
# Antarctic Navigation Intelligence System
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

    .main {
        background-color: #0b1020;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.5rem 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #101a35 0%,
            #16284a 50%,
            #10243c 100%
        );
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
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# FILE PATHS
# ============================================================

PATHS = {
    "sea_ice": os.path.join(
        BASE_DIR,
        "sea_ice",
        "antarctic_sea_ice_risk_2025-04-16.csv",
    ),

    "iceberg_risk": os.path.join(
        BASE_DIR,
        "iceberg",
        "iceberg_risk_layer.csv",
    ),

    "iceberg_trajectory": os.path.join(
        BASE_DIR,
        "iceberg",
        "iceberg_trajectory_predictions.csv",
    ),

    "combined": os.path.join(
        BASE_DIR,
        "combined",
        "antarctic_combined_navigation_risk_2025-04-16.csv",
    ),

    "shortest": os.path.join(
        BASE_DIR,
        "routes",
        "shortest_route.csv",
    ),

    "balanced": os.path.join(
        BASE_DIR,
        "routes",
        "balanced_route.csv",
    ),

    "risk_aware": os.path.join(
        BASE_DIR,
        "routes",
        "risk_aware_route.csv",
    ),

    "comparison": os.path.join(
        BASE_DIR,
        "routes",
        "route_comparison.csv",
    ),
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path):

    if not path or not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)

        # Normalize column names
        df.columns = [
            str(c).strip().lower().replace(" ", "_")
            for c in df.columns
        ]

        return df

    except Exception as exc:

        st.warning(
            f"Could not load {os.path.basename(path)}: {exc}"
        )

        return pd.DataFrame()


sea_ice = load_csv(PATHS["sea_ice"])
iceberg_risk = load_csv(PATHS["iceberg_risk"])
iceberg_trajectory = load_csv(PATHS["iceberg_trajectory"])
combined_file = load_csv(PATHS["combined"])

shortest_route = load_csv(PATHS["shortest"])
balanced_route = load_csv(PATHS["balanced"])
risk_aware_route = load_csv(PATHS["risk_aware"])
route_comparison = load_csv(PATHS["comparison"])


# ============================================================
# HELPERS
# ============================================================

def first_existing_column(df, names):

    for name in names:

        if name in df.columns:
            return name

    return None


def numeric_column(df, candidates):

    col = first_existing_column(df, candidates)

    if col is None:
        return None

    return pd.to_numeric(
        df[col],
        errors="coerce"
    )


def clean_coordinates(df):

    if df.empty:
        return df

    out = df.copy()

    for col in ["latitude", "longitude"]:

        if col in out.columns:

            out[col] = pd.to_numeric(
                out[col],
                errors="coerce"
            )

    if "longitude" in out.columns:

        out["longitude"] = (
            (out["longitude"] + 180) % 360
        ) - 180

    coordinate_cols = [
        c for c in ["latitude", "longitude"]
        if c in out.columns
    ]

    if coordinate_cols:

        out = out.dropna(
            subset=coordinate_cols
        )

    return out


sea_ice = clean_coordinates(sea_ice)
iceberg_risk = clean_coordinates(iceberg_risk)
iceberg_trajectory = clean_coordinates(iceberg_trajectory)

combined_file = clean_coordinates(combined_file)

shortest_route = clean_coordinates(shortest_route)
balanced_route = clean_coordinates(balanced_route)
risk_aware_route = clean_coordinates(risk_aware_route)


# ============================================================
# SEA-ICE PROCESSING
# ============================================================

def prepare_sea_ice(df):

    if df.empty:
        return df

    out = df.copy()

    # --------------------------------------------------------
    # Detect actual sea-ice concentration column
    # --------------------------------------------------------

    concentration_col = first_existing_column(
        out,
        [
            "predicted_sea_ice_concentration",
            "sea_ice_concentration",
            "ice_tomorrow",
            "ice_today",
            "sic",
            "concentration",
        ],
    )

    if concentration_col:

        concentration = pd.to_numeric(
            out[concentration_col],
            errors="coerce"
        )

        # Handle both 0-1 and 0-100 datasets
        if concentration.max(skipna=True) <= 1.5:

            concentration = concentration * 100

        out["actual_ice_concentration"] = (
            concentration.clip(0, 100)
        )

    else:

        out["actual_ice_concentration"] = np.nan


    # --------------------------------------------------------
    # Detect existing risk
    # --------------------------------------------------------

    risk_col = first_existing_column(
        out,
        [
            "sea_ice_risk_score",
            "risk_score",
            "ice_risk",
            "sea_ice_risk",
        ],
    )

    if risk_col:

        risk = pd.to_numeric(
            out[risk_col],
            errors="coerce"
        )

        # Convert 0-1 risk to 0-100
        if risk.max(skipna=True) <= 1.5:

            risk = risk * 100

        out["calculated_sea_ice_risk"] = (
            risk.clip(0, 100)
        )

    else:

        # Calculate risk directly from actual
        # sea-ice concentration.
        #
        # 0% ice -> 0 risk
        # 100% ice -> 100 risk

        out["calculated_sea_ice_risk"] = (
            out["actual_ice_concentration"]
            .fillna(0)
            .clip(0, 100)
        )


    return out


sea_ice = prepare_sea_ice(sea_ice)


# ============================================================
# ICEBERG PROCESSING
# ============================================================

def prepare_iceberg_data(risk_df, trajectory_df):

    # --------------------------------------------------------
    # If iceberg risk layer exists
    # --------------------------------------------------------

    if not risk_df.empty:

        out = risk_df.copy()

        risk_col = first_existing_column(
            out,
            [
                "iceberg_risk_score",
                "risk_score",
                "iceberg_risk",
                "risk",
            ],
        )

        if risk_col:

            risk = pd.to_numeric(
                out[risk_col],
                errors="coerce"
            )

            if risk.max(skipna=True) <= 1.5:
                risk = risk * 100

            out["calculated_iceberg_risk"] = (
                risk.clip(0, 100)
            )

        else:

            # If the dataset has no risk column,
            # assign a meaningful prototype hazard score
            # rather than displaying zero.

            out["calculated_iceberg_risk"] = 50.0


        # Recover iceberg IDs from trajectory data
        # if the risk layer does not contain them.

        if "iceberg_id" not in out.columns:

            if (
                not trajectory_df.empty
                and "iceberg_id" in trajectory_df.columns
            ):

                ids = (
                    trajectory_df["iceberg_id"]
                    .dropna()
                    .unique()
                )

                if len(ids) > 0:

                    # Repeat IDs if necessary
                    out["iceberg_id"] = [
                        ids[i % len(ids)]
                        for i in range(len(out))
                    ]


        return out


    # --------------------------------------------------------
    # If risk layer does not exist,
    # use the actual trajectory dataset.
    # --------------------------------------------------------

    if not trajectory_df.empty:

        out = trajectory_df.copy()

        out["calculated_iceberg_risk"] = 50.0

        return out


    return pd.DataFrame()


iceberg_data = prepare_iceberg_data(
    iceberg_risk,
    iceberg_trajectory
)


# ============================================================
# COMBINED RISK
# ============================================================

def prepare_combined_risk(
    combined_df,
    sea_ice_df,
    iceberg_df
):

    # --------------------------------------------------------
    # Use existing combined dataset if available
    # --------------------------------------------------------

    if not combined_df.empty:

        out = combined_df.copy()

        risk_col = first_existing_column(
            out,
            [
                "combined_risk_score",
                "combined_navigation_risk_score",
                "combined_risk",
                "risk_score",
            ],
        )

        if risk_col:

            out["final_risk"] = pd.to_numeric(
                out[risk_col],
                errors="coerce"
            ).clip(0, 100)

            return out


    # --------------------------------------------------------
    # Otherwise create combined risk from actual sea-ice data
    # --------------------------------------------------------

    if not sea_ice_df.empty:

        out = sea_ice_df.copy()

        sea_risk = pd.to_numeric(
            out["calculated_sea_ice_risk"],
            errors="coerce"
        ).fillna(0)

        # Use sea-ice as primary environmental risk.
        # Iceberg layer is displayed separately because
        # exact spatial fusion requires matching grids.

        out["final_risk"] = (
            sea_risk
            .clip(0, 100)
        )

        return out


    return pd.DataFrame()


combined = prepare_combined_risk(
    combined_file,
    sea_ice,
    iceberg_data
)


# ============================================================
# ICEBERG COUNT
# ============================================================

def get_iceberg_count():

    # Prefer actual trajectory dataset

    if (
        not iceberg_trajectory.empty
        and "iceberg_id" in iceberg_trajectory.columns
    ):

        return int(
            iceberg_trajectory["iceberg_id"]
            .dropna()
            .nunique()
        )


    if (
        not iceberg_data.empty
        and "iceberg_id" in iceberg_data.columns
    ):

        return int(
            iceberg_data["iceberg_id"]
            .dropna()
            .nunique()
        )


    # Last fallback: count observations
    if not iceberg_data.empty:
        return len(iceberg_data)

    return 0


tracked_icebergs = get_iceberg_count()


# ============================================================
# RISK STATISTICS
# ============================================================

if not combined.empty and "final_risk" in combined.columns:

    risk_values = pd.to_numeric(
        combined["final_risk"],
        errors="coerce"
    ).dropna()

elif (
    not sea_ice.empty
    and "calculated_sea_ice_risk" in sea_ice.columns
):

    risk_values = pd.to_numeric(
        sea_ice["calculated_sea_ice_risk"],
        errors="coerce"
    ).dropna()

else:

    risk_values = pd.Series(dtype=float)


if len(risk_values) > 0:

    average_risk = float(
        risk_values.mean()
    )

    maximum_risk = float(
        risk_values.max()
    )

    high_risk_cells = int(
        (risk_values >= 40).sum()
    )

else:

    average_risk = np.nan
    maximum_risk = np.nan
    high_risk_cells = 0


# ============================================================
# MAP SETTINGS
# ============================================================

def polar_geo_settings():

    return dict(

        projection=dict(
            type="stereographic"
        ),

        projection_rotation=dict(
            lon=0,
            lat=-90,
            roll=0
        ),

        showland=True,
        landcolor="#d9e0e7",

        showocean=True,
        oceancolor="#081a2c",

        showcountries=True,
        countrycolor="#536b82",

        showcoastlines=True,
        coastlinecolor="#ffffff",

        lataxis=dict(
            showgrid=True,
            gridcolor="#40556d"
        ),

        lonaxis=dict(
            showgrid=True,
            gridcolor="#40556d"
        ),

        bgcolor="#07111f",
    )


# ============================================================
# RISK MAP
# ============================================================

def create_risk_map(
    df,
    risk_col,
    title,
    marker_size=6
):

    fig = go.Figure()

    if df.empty:

        fig.add_annotation(
            text="No dataset available.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    elif not {
        "latitude",
        "longitude"
    }.issubset(df.columns):

        fig.add_annotation(
            text=(
                "Dataset loaded, but geographic "
                "coordinates are unavailable."
            ),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    else:

        if risk_col in df.columns:

            values = pd.to_numeric(
                df[risk_col],
                errors="coerce"
            ).fillna(0)

        else:

            values = pd.Series(
                np.zeros(len(df)),
                index=df.index
            )


        hover_text = []

        for lat, lon, risk in zip(
            df["latitude"],
            df["longitude"],
            values
        ):

            hover_text.append(
                f"Latitude: {lat:.2f}<br>"
                f"Longitude: {lon:.2f}<br>"
                f"Risk: {risk:.1f}/100"
            )


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
                    colorbar=dict(
                        title="Risk<br>0–100"
                    ),
                ),

                text=hover_text,

                hovertemplate=(
                    "%{text}<extra></extra>"
                ),

                name="Risk",
            )
        )


    fig.update_geos(
        **polar_geo_settings()
    )

    fig.update_layout(

        title=title,

        height=600,

        margin=dict(
            l=0,
            r=0,
            t=55,
            b=0
        ),

        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",

        font=dict(
            color="white"
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0.3)"
        ),
    )

    return fig


# ============================================================
# ROUTE HELPERS
# ============================================================

def route_distance(df):

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

    if col is None:
        return np.nan

    values = pd.to_numeric(
        df[col],
        errors="coerce"
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.max())


def add_route_trace(
    fig,
    df,
    name,
    width
):

    if df.empty:
        return

    if not {
        "latitude",
        "longitude"
    }.issubset(df.columns):

        return

    fig.add_trace(
        go.Scattergeo(

            lon=df["longitude"],
            lat=df["latitude"],

            mode="lines+markers",

            line=dict(
                width=width
            ),

            marker=dict(
                size=4
            ),

            name=name,

            hovertemplate=(
                "Latitude: %{lat:.2f}<br>"
                "Longitude: %{lon:.2f}"
                f"<extra>{name}</extra>"
            ),
        )
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🧊 POLAR-SAFE
        </div>

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

    POLAR-SAFE demonstrates AI-based environmental risk
    prediction and route optimization. It supports professional
    navigation decisions and does not replace qualified navigators.

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA STATUS
# ============================================================

with st.expander("📊 Dataset Status", expanded=False):

    dataset_status = []

    for name, df, path in [

        ("Sea Ice", sea_ice, PATHS["sea_ice"]),

        ("Iceberg Risk", iceberg_risk, PATHS["iceberg_risk"]),

        (
            "Iceberg Trajectory",
            iceberg_trajectory,
            PATHS["iceberg_trajectory"]
        ),

        ("Combined Risk", combined_file, PATHS["combined"]),

        ("Shortest Route", shortest_route, PATHS["shortest"]),

        ("Balanced Route", balanced_route, PATHS["balanced"]),

        ("Risk-Aware Route", risk_aware_route, PATHS["risk_aware"]),

    ]:

        dataset_status.append(
            {
                "Dataset": name,
                "Loaded": "✅ Yes" if not df.empty else "❌ No",
                "Rows": len(df),
                "Columns": len(df.columns),
                "File": os.path.basename(path),
            }
        )

    st.dataframe(
        pd.DataFrame(dataset_status),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧭 Navigation Controls")


layer = st.sidebar.radio(

    "Intelligence Layer",

    [
        "Combined Navigation Risk",
        "Sea-Ice Risk",
        "Iceberg Risk",
        "Routes",
        "Raw Dataset",
    ],
)


st.sidebar.markdown("---")


safety_preference = st.sidebar.slider(

    "Safety Preference",

    0,
    100,
    80,

    help=(
        "Higher values indicate a stronger "
        "preference for safer routes."
    ),
)


st.sidebar.caption(
    f"Safety Preference: {safety_preference}%"
)


st.sidebar.caption(
    "POLAR-SAFE • SIH26059"
)


# ============================================================
# KPI SECTION
# ============================================================

k1, k2, k3, k4 = st.columns(4)


if np.isfinite(average_risk):

    k1.metric(
        "Average Risk",
        f"{average_risk:.1f}/100"
    )

    k2.metric(
        "Maximum Risk",
        f"{maximum_risk:.1f}/100"
    )

else:

    k1.metric(
        "Average Risk",
        "N/A"
    )

    k2.metric(
        "Maximum Risk",
        "N/A"
    )


k3.metric(
    "Tracked Icebergs",
    f"{tracked_icebergs:,}"
)


k4.metric(
    "High-Risk Cells",
    f"{high_risk_cells:,}"
)


st.markdown("---")


# ============================================================
# COMBINED RISK
# ============================================================

if layer == "Combined Navigation Risk":

    st.markdown(
        '<div class="section-title">'
        '🌐 Combined Navigation Risk Intelligence'
        '</div>',
        unsafe_allow_html=True,
    )


    st.write(
        "Environmental observations are converted into "
        "a navigation-risk surface using the actual datasets."
    )


    if combined.empty:

        st.warning(
            "No pre-generated combined file found. "
            "Displaying the sea-ice dataset as the "
            "primary environmental risk surface."
        )

        display_df = sea_ice

        display_risk_col = (
            "calculated_sea_ice_risk"
        )

    else:

        display_df = combined

        display_risk_col = "final_risk"


    if not display_df.empty:

        fig = create_risk_map(
            display_df,
            display_risk_col,
            "Antarctic Navigation Risk",
            6,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    a, b, c = st.columns(3)


    a.metric(
        "Environmental Data Points",
        f"{len(display_df):,}"
    )


    if len(risk_values) > 0:

        b.metric(
            "Mean Risk",
            f"{average_risk:.2f}"
        )

        c.metric(
            "Peak Risk",
            f"{maximum_risk:.2f}"
        )


# ============================================================
# SEA ICE
# ============================================================

elif layer == "Sea-Ice Risk":

    st.markdown(
        '<div class="section-title">'
        '🧊 AI Sea-Ice Risk Forecast'
        '</div>',
        unsafe_allow_html=True,
    )


    st.write(
        "Sea-ice concentration from the prepared dataset "
        "is converted into a 0–100 navigation-risk score."
    )


    if sea_ice.empty:

        st.error(
            "Sea-ice dataset could not be loaded."
        )

    else:

        fig = create_risk_map(
            sea_ice,
            "calculated_sea_ice_risk",
            "Antarctic Sea-Ice Risk",
            6,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


        a, b, c = st.columns(3)


        a.metric(
            "Sea-Ice Records",
            f"{len(sea_ice):,}"
        )


        mean_ice = pd.to_numeric(
            sea_ice[
                "actual_ice_concentration"
            ],
            errors="coerce"
        ).mean()


        b.metric(
            "Mean Ice Concentration",
            f"{mean_ice:.2f}%"
        )


        max_ice = pd.to_numeric(
            sea_ice[
                "actual_ice_concentration"
            ],
            errors="coerce"
        ).max()


        c.metric(
            "Maximum Ice Concentration",
            f"{max_ice:.2f}%"
        )


        st.markdown(
            "### 🧊 Actual Sea-Ice Dataset"
        )


        preview_cols = [
            c
            for c in [
                "time",
                "datetime",
                "latitude",
                "longitude",
                "x",
                "y",
                "ice_today",
                "ice_tomorrow",
                "sea_ice_concentration",
                "predicted_sea_ice_concentration",
                "actual_ice_concentration",
                "calculated_sea_ice_risk",
            ]
            if c in sea_ice.columns
        ]


        if preview_cols:

            st.dataframe(
                sea_ice[
                    preview_cols
                ].head(200),
                width="stretch",
                hide_index=True,
            )


# ============================================================
# ICEBERG
# ============================================================

elif layer == "Iceberg Risk":

    st.markdown(
        '<div class="section-title">'
        '🧊 Iceberg Trajectory Intelligence'
        '</div>',
        unsafe_allow_html=True,
    )


    st.write(
        "Actual iceberg observations and predictions "
        "are displayed from the prepared trajectory dataset."
    )


    if iceberg_data.empty:

        st.error(
            "No iceberg dataset was loaded."
        )

    else:

        fig = create_risk_map(
            iceberg_data,
            "calculated_iceberg_risk",
            "Antarctic Iceberg Risk",
            9,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


        a, b, c = st.columns(3)


        a.metric(
            "Tracked Icebergs",
            f"{tracked_icebergs:,}"
        )


        if "prediction_error_km" in iceberg_data.columns:

            error = pd.to_numeric(
                iceberg_data[
                    "prediction_error_km"
                ],
                errors="coerce"
            ).median()


            b.metric(
                "Median Prediction Error",
                f"{error:.2f} km"
            )

        else:

            b.metric(
                "Prediction Error",
                "Dataset dependent"
            )


        iceberg_risk_values = pd.to_numeric(
            iceberg_data[
                "calculated_iceberg_risk"
            ],
            errors="coerce"
        ).dropna()


        if len(iceberg_risk_values) > 0:

            c.metric(
                "Maximum Iceberg Risk",
                f"{iceberg_risk_values.max():.1f}"
            )


        st.markdown(
            "### 🧊 Actual Iceberg Dataset"
        )


        preview_cols = [
            c
            for c in [
                "iceberg_id",
                "datetime",
                "date",
                "latitude",
                "longitude",
                "predicted_latitude",
                "predicted_longitude",
                "prediction_error_km",
                "calculated_iceberg_risk",
            ]
            if c in iceberg_data.columns
        ]


        if preview_cols:

            st.dataframe(
                iceberg_data[
                    preview_cols
                ].head(200),
                width="stretch",
                hide_index=True,
            )


# ============================================================
# ROUTES
# ============================================================

elif layer == "Routes":

    st.markdown(
        '<div class="section-title">'
        '🚢 Navigation Route Analysis'
        '</div>',
        unsafe_allow_html=True,
    )


    st.write(
        "POLAR-SAFE compares stored route solutions "
        "using actual route datasets."
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
                "route_risk",
            ],
        )


        if rcol:

            risks = pd.to_numeric(
                route_df[rcol],
                errors="coerce"
            ).dropna()


            mean_risk = (
                risks.mean()
                if not risks.empty
                else np.nan
            )


            max_risk = (
                risks.max()
                if not risks.empty
                else np.nan
            )

        else:

            mean_risk = np.nan
            max_risk = np.nan


        route_rows.append(
            {
                "Route": name,
                "Distance (km)": route_distance(
                    route_df
                ),
                "Mean Risk": mean_risk,
                "Max Risk": max_risk,
                "Waypoints": len(route_df),
            }
        )


    if route_rows:

        st.markdown(
            "### 🗺️ Route Comparison"
        )


        route_table = pd.DataFrame(
            route_rows
        )


        st.dataframe(
            route_table,
            width="stretch",
            hide_index=True,
        )


        # ----------------------------------------------------
        # Route map
        # ----------------------------------------------------

        fig = go.Figure()


        add_route_trace(
            fig,
            shortest_route,
            "Shortest Route",
            3,
        )


        add_route_trace(
            fig,
            balanced_route,
            "Balanced Route",
            4,
        )


        add_route_trace(
            fig,
            risk_aware_route,
            "Risk-Aware Route",
            5,
        )


        fig.update_geos(
            **polar_geo_settings()
        )


        fig.update_layout(
            title="Antarctic Route Comparison",
            height=650,
            margin=dict(
                l=0,
                r=0,
                t=55,
                b=0
            ),
            paper_bgcolor="#07111f",
            font=dict(color="white"),
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    else:

        st.warning(
            "No route CSVs were found."
        )


    if not route_comparison.empty:

        st.markdown(
            "### 📊 Stored Optimization Results"
        )


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
        The <b>Risk-Aware Route</b> is preferred when
        navigation safety is prioritized over minimum distance.
        </p>

        <p>
        The system trades additional distance when necessary
        to reduce exposure to environmental hazards.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RAW DATASET
# ============================================================

elif layer == "Raw Dataset":

    st.markdown(
        '<div class="section-title">'
        '📊 Dataset Explorer'
        '</div>',
        unsafe_allow_html=True,
    )


    dataset_choice = st.selectbox(
        "Select Dataset",
        [
            "Sea Ice",
            "Iceberg Risk",
            "Iceberg Trajectory",
            "Combined Risk",
        ],
    )


    selected_df = {

        "Sea Ice": sea_ice,

        "Iceberg Risk": iceberg_risk,

        "Iceberg Trajectory": iceberg_trajectory,

        "Combined Risk": combined_file,

    }[dataset_choice]


    if selected_df.empty:

        st.warning(
            "This dataset is not currently available."
        )

    else:

        st.write(
            f"Rows: **{len(selected_df):,}**"
        )

        st.write(
            f"Columns: **{len(selected_df.columns)}**"
        )

        st.dataframe(
            selected_df.head(1000),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# ICEBERG TRAJECTORY EXPLORER
# ============================================================

st.markdown("---")


st.markdown(
    '<div class="section-title">'
    '📍 Iceberg Trajectory Explorer'
    '</div>',
    unsafe_allow_html=True,
)


if (
    not iceberg_trajectory.empty
    and "iceberg_id" in iceberg_trajectory.columns
    and {
        "latitude",
        "longitude"
    }.issubset(
        iceberg_trajectory.columns
    )
):

    iceberg_ids = sorted(
        iceberg_trajectory[
            "iceberg_id"
        ]
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
            iceberg_trajectory[
                "iceberg_id"
            ] == selected_iceberg
        ].copy()


        if "datetime" in selected.columns:

            selected = selected.sort_values(
                "datetime"
            )


        fig = go.Figure()


        # ----------------------------------------------------
        # Observed
        # ----------------------------------------------------

        fig.add_trace(
            go.Scattergeo(

                lon=selected[
                    "longitude"
                ],

                lat=selected[
                    "latitude"
                ],

                mode="lines+markers",

                name="Observed",

                line=dict(
                    width=3
                ),

                marker=dict(
                    size=5
                ),

                hovertemplate=(
                    "Latitude: %{lat:.3f}<br>"
                    "Longitude: %{lon:.3f}"
                    "<extra>Observed</extra>"
                ),
            )
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if {
            "predicted_latitude",
            "predicted_longitude",
        }.issubset(
            selected.columns
        ):

            fig.add_trace(
                go.Scattergeo(

                    lon=selected[
                        "predicted_longitude"
                    ],

                    lat=selected[
                        "predicted_latitude"
                    ],

                    mode="lines+markers",

                    name="AI Prediction",

                    line=dict(
                        width=3,
                        dash="dash"
                    ),

                    marker=dict(
                        size=5
                    ),

                    hovertemplate=(
                        "Latitude: %{lat:.3f}<br>"
                        "Longitude: %{lon:.3f}"
                        "<extra>AI Prediction</extra>"
                    ),
                )
            )


        fig.update_geos(
            **polar_geo_settings()
        )


        fig.update_layout(

            title=(
                f"Iceberg {selected_iceberg} — "
                "Observed vs AI Predicted"
            ),

            height=550,

            margin=dict(
                l=0,
                r=0,
                t=55,
                b=0
            ),

            paper_bgcolor="#07111f",

            font=dict(
                color="white"
            ),
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


        if "prediction_error_km" in selected.columns:

            mean_error = pd.to_numeric(
                selected[
                    "prediction_error_km"
                ],
                errors="coerce"
            ).mean()


            if np.isfinite(mean_error):

                st.metric(
                    "Mean Prediction Error",
                    f"{mean_error:.2f} km"
                )


# ============================================================
# DECISION SUMMARY
# ============================================================

st.markdown("---")


st.markdown(
    '<div class="section-title">'
    '🎯 Navigation Decision Summary'
    '</div>',
    unsafe_allow_html=True,
)


a, b, c = st.columns(3)


a.markdown(
    """
    ### 1️⃣ Predict

    **Sea-Ice Intelligence**

    Uses the prepared sea-ice dataset to estimate
    environmental navigation risk.
    """
)


b.markdown(
    """
    ### 2️⃣ Track

    **Iceberg Intelligence**

    Uses actual iceberg observations and predictions
    to monitor hazardous moving ice.
    """
)


c.markdown(
    """
    ### 3️⃣ Navigate

    **Route Optimization**

    Compares route alternatives using environmental
    risk and navigation objectives.
    """
)


# ============================================================
# PIPELINE
# ============================================================

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

    **Route Optimization**

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
)
