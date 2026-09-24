# ============================================================
# JAPAN EARTHQUAKE FORECASTING - STREAMLIT APP
# ============================================================

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Japan Earthquake Forecasting",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

MODEL_FILE = MODELS_DIR / "forecast_models.joblib"
METRICS_FILE = MODELS_DIR / "test_metrics.csv"

FEATURE_FILE = PROCESSED_DIR / "model_features.csv"
EARTHQUAKE_FILE = PROCESSED_DIR / "japan_clean_events.csv"


# ============================================================
# GLOBALS REQUIRED BY PICKLED BASELINE
# ============================================================

# This gets populated after loading the bundle.
grid_labels_by_col = {}


# ============================================================
# CUSTOM CLASSES
#
# IMPORTANT:
# These classes MUST be defined before joblib.load().
# The saved .joblib file was created when these classes existed
# under __main__ in Notebook 04.
# ============================================================

class GridProbabilityBaseline:

    def __init__(
        self,
        grid_probabilities,
        grid_columns,
        probability_from_rate=False
    ):
        self.grid_probabilities = grid_probabilities
        self.grid_columns = grid_columns
        self.probability_from_rate = probability_from_rate

    def fit(self, X, y):

        self.classes_ = np.array([0, 1])

        self.default_probability_ = float(
            np.mean(y)
        )

        return self

    def predict_proba(self, X):

        X = pd.DataFrame(X).copy()

        # Identify active grid column
        active_col = X[
            self.grid_columns
        ].idxmax(axis=1)

        # Convert grid column name to grid label
        labels = active_col.map(
            grid_labels_by_col
        )

        values = (
            labels
            .map(self.grid_probabilities)
            .fillna(
                self.default_probability_
            )
            .astype(float)
            .to_numpy()
        )

        # For Poisson-rate style baseline
        if self.probability_from_rate:

            values = (
                1.0
                - np.exp(
                    -values * 14.0
                )
            )

        values = np.clip(
            values,
            0.0,
            1.0
        )

        return np.column_stack(
            [
                1.0 - values,
                values
            ]
        )

    def predict(self, X):

        return (
            self.predict_proba(X)[:, 1] >= 0.5
        ).astype(int)


class PersistenceBaseline:

    def __init__(self):
        pass

    def fit(self, X, y):

        self.classes_ = np.array([0, 1])

        # Probability when recent M4.5+ activity exists
        active_mask = (
            X["recent_m45_30d"].to_numpy() == 1
        )

        if active_mask.any():

            self.active_risk_ = float(
                np.mean(
                    np.asarray(y)[active_mask]
                )
            )

        else:

            self.active_risk_ = float(
                np.mean(y)
            )

        # Probability when there is no recent M4.5+
        quiet_mask = (
            X["recent_m45_30d"].to_numpy() == 0
        )

        if quiet_mask.any():

            self.quiet_risk_ = float(
                np.mean(
                    np.asarray(y)[quiet_mask]
                )
            )

        else:

            self.quiet_risk_ = float(
                np.mean(y)
            )

        return self

    def predict_proba(self, X):

        X = pd.DataFrame(X).copy()

        p = np.where(
            X[
                "recent_m45_30d"
            ].to_numpy() == 1,

            getattr(
                self,
                "active_risk_",
                0.5
            ),

            getattr(
                self,
                "quiet_risk_",
                0.5
            )
        )

        p = np.clip(
            p,
            0.0,
            1.0
        )

        return np.column_stack(
            [
                1.0 - p,
                p
            ]
        )

    def predict(self, X):

        return (
            self.predict_proba(X)[:, 1] >= 0.5
        ).astype(int)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model_bundle():

    if not MODEL_FILE.exists():

        st.error(
            f"Model file not found:\n\n"
            f"{MODEL_FILE}"
        )

        st.stop()

    return joblib.load(
        MODEL_FILE
    )


# ============================================================
# LOAD MODEL
#
# This happens AFTER the classes above have been defined.
# ============================================================

bundle = load_model_bundle()


# ============================================================
# EXTRACT MODEL INFORMATION
# ============================================================

models = bundle.get(
    "models",
    {}
)

baselines = bundle.get(
    "baselines",
    {}
)

scaler = bundle.get(
    "scaler",
    None
)

continuous_cols = bundle.get(
    "continuous_cols",
    []
)

feature_set = bundle.get(
    "feature_set",
    []
)

grid_cols = bundle.get(
    "grid_cols",
    []
)


# ============================================================
# GRID LABEL MAPPING
# ============================================================

grid_labels_by_col = {
    column: str(column).replace(
        "grid_",
        ""
    )
    for column in grid_cols
}


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_metrics():

    if not METRICS_FILE.exists():

        return pd.DataFrame()

    df = pd.read_csv(
        METRICS_FILE
    )

    df.columns = [
        str(column)
        .replace("*", "")
        .strip()
        for column in df.columns
    ]

    return df


@st.cache_data
def load_features():

    if not FEATURE_FILE.exists():

        return pd.DataFrame()

    df = pd.read_csv(
        FEATURE_FILE
    )

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

    return df


@st.cache_data
def load_earthquakes():

    if not EARTHQUAKE_FILE.exists():

        return pd.DataFrame()

    df = pd.read_csv(
        EARTHQUAKE_FILE
    )

    if "time" in df.columns:

        df["time"] = pd.to_datetime(
            df["time"],
            errors="coerce"
        )

    return df


metrics = load_metrics()
features_df = load_features()
earthquakes_df = load_earthquakes()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if np.isnan(value) or np.isinf(value):

            return default

        return value

    except Exception:

        return default


def display_model_name(name):

    name = str(name)

    replacements = {
        "LogisticRegression": "Logistic Regression",
        "RandomForestClassifier": "Random Forest",
        "HistGradientBoostingClassifier":
            "Gradient Boosted Trees",
        "MLPClassifier":
            "Multi-Layer Perceptron",
    }

    return replacements.get(
        name,
        name
    )


def get_model_probability(model, X):

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            X
        )

        return float(
            probabilities[0, 1]
        )

    prediction = model.predict(
        X
    )

    return float(
        prediction[0]
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2.2rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #111827,
            #1f2937,
            #334155
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #d1d5db;
        font-size: 1.05rem;
        max-width: 900px;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 15px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        text-align: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
    }

    .forecast-box {
        padding: 2rem;
        border-radius: 18px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        text-align: center;
        margin: 1.5rem 0;
    }

    .forecast-probability {
        font-size: 4rem;
        font-weight: 800;
        color: #111827;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Japan Earthquake Forecasting"
)

st.sidebar.caption(
    "Machine Learning Research Dashboard"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Earthquake Explorer",
        "14-Day Forecast",
        "Model Performance",
        "Feature Insights",
        "Limitations",
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "30-day seismic activity → "
    "14-day M5+ occurrence"
)


# ============================================================
# HEADER
# ============================================================

# FIX: no blank lines / indentation inside the HTML, otherwise
# Streamlit's markdown parser renders the inner tags as a code block.
st.markdown(
    """
<div class="hero">
<h1>Japan Earthquake Forecasting</h1>
<p>An interactive machine-learning dashboard exploring whether recent seismic activity can provide useful signals for short-horizon M5+ earthquake occurrence.</p>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":

    st.header(
        "Project Overview"
    )

    st.write(
        """
        The project frames earthquake occurrence as a binary
        classification problem. The model uses seismic activity
        observed during the previous 30 days to estimate whether
        an M5.0 or greater earthquake occurs during the following
        14-day period within a Japan spatial grid cell.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # PROJECT METRICS
    # --------------------------------------------------------

    total_events = len(
        earthquakes_df
    )

    total_windows = len(
        features_df
    )

    if total_events == 0:

        total_events = 21954

    if total_windows == 0:

        total_windows = 1794

    number_grids = len(
        grid_cols
    )

    if number_grids == 0:

        number_grids = 6

    columns = st.columns(4)

    with columns[0]:

        st.markdown(
            f"""
<div class="metric-card">
<div class="metric-value">{total_events:,}</div>
<div class="metric-label">Japan earthquake events</div>
</div>
""",
            unsafe_allow_html=True
        )

    with columns[1]:

        st.markdown(
            f"""
<div class="metric-card">
<div class="metric-value">{total_windows:,}</div>
<div class="metric-label">Forecasting windows</div>
</div>
""",
            unsafe_allow_html=True
        )

    with columns[2]:

        st.markdown(
            """
<div class="metric-card">
<div class="metric-value">30 days</div>
<div class="metric-label">Historical lookback</div>
</div>
""",
            unsafe_allow_html=True
        )

    with columns[3]:

        st.markdown(
            """
<div class="metric-card">
<div class="metric-value">14 days</div>
<div class="metric-label">Forecast horizon</div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.subheader(
        "Forecasting Pipeline"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            "### 01 · Recent seismic activity"
        )

        st.write(
            """
            Earthquake events from the preceding 30-day
            window are summarized into numerical features.
            """
        )

    with c2:

        st.markdown(
            "### 02 · Feature engineering"
        )

        st.write(
            """
            Event count, cumulative seismic energy,
            maximum magnitude, depth, recent M4.5+
            activity and spatial grid are represented
            as model features.
            """
        )

    with c3:

        st.markdown(
            "### 03 · 14-day occurrence"
        )

        st.write(
            """
            Classification models estimate the probability
            associated with an M5+ occurrence during the
            subsequent 14-day window.
            """
        )

    st.markdown("---")

    st.subheader(
        "Research Questions"
    )

    questions = [
        "Can recent 30-day seismicity provide useful information for 14-day M5+ occurrence?",
        "Which recent-activity features contain the strongest signal?",
        "Where do false alarms and missed events occur?"
    ]

    for number, question in enumerate(
        questions,
        start=1
    ):

        st.write(
            f"**{number}.** {question}"
        )

    st.markdown("---")

    st.info(
        """
        This dashboard demonstrates a historical machine-learning
        experiment. It is not an operational earthquake warning
        system and does not predict the exact time, location or
        magnitude of an earthquake.
        """
    )


# ============================================================
# PAGE 2 — EARTHQUAKE EXPLORER
# ============================================================

elif page == "Earthquake Explorer":

    st.header(
        "Earthquake Explorer"
    )

    if earthquakes_df.empty:

        st.warning(
            "japan_clean_events.csv could not be loaded."
        )

        st.stop()

    df = earthquakes_df.copy()

    # --------------------------------------------------------
    # FIND COLUMNS
    # --------------------------------------------------------

    magnitude_col = next(
        (
            column
            for column in [
                "mag",
                "magnitude"
            ]
            if column in df.columns
        ),
        None
    )

    latitude_col = next(
        (
            column
            for column in [
                "latitude",
                "lat"
            ]
            if column in df.columns
        ),
        None
    )

    longitude_col = next(
        (
            column
            for column in [
                "longitude",
                "lon",
                "lng"
            ]
            if column in df.columns
        ),
        None
    )

    depth_col = next(
        (
            column
            for column in [
                "depth",
                "depth_km"
            ]
            if column in df.columns
        ),
        None
    )

    time_col = next(
        (
            column
            for column in [
                "time",
                "timestamp"
            ]
            if column in df.columns
        ),
        None
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        if magnitude_col:

            min_magnitude = st.slider(
                "Minimum magnitude",
                2.5,
                8.0,
                4.0,
                0.1
            )

        else:

            min_magnitude = 0.0

    with c2:

        selected_dates = None

        if time_col:

            dates = pd.to_datetime(
                df[time_col],
                errors="coerce"
            ).dropna()

            if not dates.empty:

                min_date = dates.min().date()
                max_date = dates.max().date()

                selected_dates = st.date_input(
                    "Date range",
                    value=(
                        min_date,
                        max_date
                    ),
                    min_value=min_date,
                    max_value=max_date
                )

    filtered = df.copy()

    if magnitude_col:

        filtered = filtered[
            pd.to_numeric(
                filtered[magnitude_col],
                errors="coerce"
            ) >= min_magnitude
        ]

    if (
        time_col
        and selected_dates
        and isinstance(
            selected_dates,
            tuple
        )
        and len(selected_dates) == 2
    ):

        start_date, end_date = selected_dates

        timestamps = pd.to_datetime(
            filtered[time_col],
            errors="coerce"
        )

        filtered = filtered[
            (
                timestamps.dt.date
                >= start_date
            )
            &
            (
                timestamps.dt.date
                <= end_date
            )
        ]

    st.markdown("---")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Events",
            f"{len(filtered):,}"
        )

    with c2:

        if (
            magnitude_col
            and len(filtered) > 0
        ):

            max_mag = pd.to_numeric(
                filtered[magnitude_col],
                errors="coerce"
            ).max()

            st.metric(
                "Maximum magnitude",
                f"{max_mag:.2f}"
            )

        else:

            st.metric(
                "Maximum magnitude",
                "N/A"
            )

    with c3:

        if (
            depth_col
            and len(filtered) > 0
        ):

            avg_depth = pd.to_numeric(
                filtered[depth_col],
                errors="coerce"
            ).mean()

            st.metric(
                "Average depth",
                f"{avg_depth:.1f} km"
            )

        else:

            st.metric(
                "Average depth",
                "N/A"
            )

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    if (
        latitude_col
        and longitude_col
        and len(filtered) > 0
    ):

        st.subheader(
            "Spatial distribution"
        )

        map_df = filtered[
            [
                latitude_col,
                longitude_col
            ]
        ].copy()

        map_df.columns = [
            "lat",
            "lon"
        ]

        map_df["lat"] = pd.to_numeric(
            map_df["lat"],
            errors="coerce"
        )

        map_df["lon"] = pd.to_numeric(
            map_df["lon"],
            errors="coerce"
        )

        map_df = map_df.dropna()

        st.map(
            map_df
        )

    # --------------------------------------------------------
    # MAGNITUDE DISTRIBUTION
    # --------------------------------------------------------

    if (
        magnitude_col
        and len(filtered) > 0
    ):

        st.subheader(
            "Magnitude distribution"
        )

        magnitudes = pd.to_numeric(
            filtered[magnitude_col],
            errors="coerce"
        ).dropna()

        bins = [
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.5
        ]

        counts = (
            pd.cut(
                magnitudes,
                bins=bins,
                include_lowest=True
            )
            .value_counts()
            .sort_index()
        )

        counts.index = (
            counts.index.astype(str)
        )

        st.bar_chart(
            counts
        )

    # --------------------------------------------------------
    # TIMELINE
    # --------------------------------------------------------

    if (
        time_col
        and len(filtered) > 0
    ):

        st.subheader(
            "Earthquake activity over time"
        )

        timeline = pd.to_datetime(
            filtered[time_col],
            errors="coerce"
        )

        monthly = (
            timeline
            .dropna()
            .dt.to_period("M")
            .astype(str)
            .value_counts()
            .sort_index()
        )

        st.line_chart(
            monthly
        )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    st.subheader(
        "Earthquake records"
    )

    st.dataframe(
        filtered.head(500),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3 — FORECAST
# ============================================================

elif page == "14-Day Forecast":

    st.header(
        "14-Day M5+ Occurrence Forecast"
    )

    st.write(
        """
        Enter values representing the seismic activity observed
        during a previous 30-day window. The selected trained
        model will produce an estimated probability for an M5+
        occurrence during the following 14 days.
        """
    )

    st.warning(
        """
        Research/demo use only. This output is a model probability
        estimate, not an operational earthquake warning.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # AVAILABLE MODELS
    # --------------------------------------------------------

    available_models = list(
        models.keys()
    )

    if not available_models:

        st.error(
            "No trained models were found."
        )

        st.stop()

    model_display_map = {
        display_model_name(name): name
        for name in available_models
    }

    selected_display_name = st.selectbox(
        "Select model",
        list(
            model_display_map.keys()
        )
    )

    selected_model_name = (
        model_display_map[
            selected_display_name
        ]
    )

    selected_model = models[
        selected_model_name
    ]

    st.markdown("---")

    # --------------------------------------------------------
    # USER INPUTS
    # --------------------------------------------------------

    st.subheader(
        "Previous 30-day activity"
    )

    c1, c2 = st.columns(2)

    with c1:

        event_count = st.number_input(
            "Number of earthquakes",
            min_value=0,
            max_value=10000,
            value=20,
            step=1
        )

        lag_event_count = st.number_input(
            "Previous-period earthquake count",
            min_value=0,
            max_value=10000,
            value=20,
            step=1
        )

        max_magnitude = st.number_input(
            "Maximum magnitude",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.1
        )

    with c2:

        mean_depth = st.number_input(
            "Mean depth (km)",
            min_value=0.0,
            max_value=700.0,
            value=50.0,
            step=1.0
        )

        recent_m45 = st.selectbox(
            "M4.5+ event during previous 30 days?",
            [
                "No",
                "Yes"
            ]
        )

        if grid_cols:

            grid_labels = [
                grid_labels_by_col.get(
                    column,
                    column.replace(
                        "grid_",
                        ""
                    )
                )
                for column in grid_cols
            ]

            selected_grid = st.selectbox(
                "Spatial grid",
                grid_labels
            )

        else:

            selected_grid = st.selectbox(
                "Spatial grid",
                [
                    "south_west",
                    "south_east",
                    "central_west",
                    "central_east",
                    "north_west",
                    "north_east"
                ]
            )

    st.subheader(
        "Cumulative seismic energy"
    )

    log_energy = st.number_input(
        "log10 cumulative seismic energy",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        step=0.1
    )

    st.markdown("---")

    # --------------------------------------------------------
    # RUN FORECAST
    # --------------------------------------------------------

    if st.button(
        "Run 14-Day Forecast",
        type="primary",
        use_container_width=True
    ):

        # --------------------------------------------
        # Create complete feature row
        # --------------------------------------------

        input_data = {
            feature: 0.0
            for feature in feature_set
        }

        # --------------------------------------------
        # Continuous features
        # --------------------------------------------

        if "log_event_count_30d" in input_data:

            input_data[
                "log_event_count_30d"
            ] = np.log1p(
                safe_float(
                    event_count
                )
            )

        if "log_event_count_lag1" in input_data:

            input_data[
                "log_event_count_lag1"
            ] = np.log1p(
                safe_float(
                    lag_event_count
                )
            )

        if "max_mag_30d" in input_data:

            input_data[
                "max_mag_30d"
            ] = safe_float(
                max_magnitude
            )

        if "mean_depth_imputed" in input_data:

            input_data[
                "mean_depth_imputed"
            ] = safe_float(
                mean_depth
            )

        if "log_cum_energy_30d" in input_data:

            input_data[
                "log_cum_energy_30d"
            ] = safe_float(
                log_energy
            )

        if "recent_m45_30d" in input_data:

            input_data[
                "recent_m45_30d"
            ] = (
                1.0
                if recent_m45 == "Yes"
                else 0.0
            )

        # --------------------------------------------
        # Spatial one-hot feature
        # --------------------------------------------

        for column in grid_cols:

            label = grid_labels_by_col.get(
                column,
                column.replace(
                    "grid_",
                    ""
                )
            )

            if label == selected_grid:

                input_data[
                    column
                ] = 1.0

        # --------------------------------------------
        # DataFrame
        # --------------------------------------------

        X_input = pd.DataFrame(
            [input_data]
        )

        # Ensure exact feature order
        if feature_set:

            X_input = X_input.reindex(
                columns=feature_set,
                fill_value=0.0
            )

        X_input = X_input.astype(
            float
        )

        # --------------------------------------------
        # Apply saved scaler
        # --------------------------------------------

        if (
            scaler is not None
            and continuous_cols
        ):

            missing_continuous = [
                column
                for column in continuous_cols
                if column not in X_input.columns
            ]

            if missing_continuous:

                st.error(
                    "Missing model features: "
                    + ", ".join(
                        missing_continuous
                    )
                )

                st.stop()

            try:

                X_input.loc[
                    :,
                    continuous_cols
                ] = scaler.transform(
                    X_input[
                        continuous_cols
                    ]
                )

            except Exception as error:

                st.error(
                    f"Scaling failed: {error}"
                )

                st.stop()

        # --------------------------------------------
        # Prediction
        # --------------------------------------------

        try:

            probability = get_model_probability(
                selected_model,
                X_input
            )

        except Exception as error:

            st.error(
                f"Prediction failed:\n\n{error}"
            )

            st.stop()

        probability = float(
            np.clip(
                probability,
                0.0,
                1.0
            )
        )

        percentage = (
            probability * 100
        )

        # --------------------------------------------
        # Result
        # --------------------------------------------

        model_label = display_model_name(
            selected_model_name
        )

        st.markdown(
            f"""
<div class="forecast-box">
<div>Estimated M5+ occurrence probability</div>
<div class="forecast-probability">{percentage:.1f}%</div>
<div>Model: {model_label}</div>
</div>
""",
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Probability",
                f"{percentage:.1f}%"
            )

        with c2:

            st.metric(
                "Forecast horizon",
                "14 days"
            )

        with c3:

            st.metric(
                "Lookback window",
                "30 days"
            )

        st.info(
            """
            The displayed probability is generated by the selected
            trained model from the supplied feature values. It does
            not represent certainty and should not be interpreted
            as a real-world earthquake warning.
            """
        )

        with st.expander(
            "View model input"
        ):

            st.dataframe(
                X_input,
                use_container_width=True
            )


# ============================================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header(
        "Model Performance"
    )

    if metrics.empty:

        st.warning(
            "test_metrics.csv could not be loaded."
        )

        st.stop()

    # --------------------------------------------------------
    # Convert metrics to numeric
    # --------------------------------------------------------

    metric_columns = [
        "PR-AUC",
        "ROC-AUC",
        "Brier Score",
        "F1-Score",
        "Precision",
        "Recall",
        "Accuracy",
        "Balanced Acc"
    ]

    for column in metric_columns:

        if column in metrics.columns:

            metrics[column] = pd.to_numeric(
                metrics[column],
                errors="coerce"
            )

    st.write(
        """
        The following results are from the held-out chronological
        test partition. The test period covers 2020–2024 and
        contains 270 observations with a 23.3% positive base rate.
        """
    )

    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    if "PR-AUC" in metrics.columns:

        st.subheader(
            "PR-AUC"
        )

        pr_chart = (
            metrics[
                [
                    "Model",
                    "PR-AUC"
                ]
            ]
            .set_index(
                "Model"
            )
            .sort_values(
                "PR-AUC",
                ascending=False
            )
        )

        st.bar_chart(
            pr_chart
        )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    if "ROC-AUC" in metrics.columns:

        st.subheader(
            "ROC-AUC"
        )

        roc_chart = (
            metrics[
                [
                    "Model",
                    "ROC-AUC"
                ]
            ]
            .set_index(
                "Model"
            )
            .sort_values(
                "ROC-AUC",
                ascending=False
            )
        )

        st.bar_chart(
            roc_chart
        )

    # --------------------------------------------------------
    # FULL TABLE
    # --------------------------------------------------------

    st.subheader(
        "Complete test results"
    )

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 5 — FEATURE INSIGHTS
# ============================================================

elif page == "Feature Insights":

    st.header(
        "Feature Insights"
    )

    st.write(
        """
        Instead of using individual earthquake records directly,
        the forecasting dataset summarizes seismic activity over
        the preceding 30-day period.
        """
    )

    # --------------------------------------------------------
    # FEATURE TABLE
    # --------------------------------------------------------

    feature_info = pd.DataFrame(
        [
            [
                "log_cum_energy_30d",
                "Log-transformed cumulative seismic energy during the previous 30 days."
            ],
            [
                "max_mag_30d",
                "Maximum earthquake magnitude during the previous 30 days."
            ],
            [
                "log_event_count_30d",
                "Log-transformed number of events during the previous 30 days."
            ],
            [
                "log_event_count_lag1",
                "Lagged 30-day earthquake count."
            ],
            [
                "mean_depth_imputed",
                "Mean focal depth with training-based imputation."
            ],
            [
                "recent_m45_30d",
                "Indicator for recent M4.5+ earthquake activity."
            ],
            [
                "grid_*",
                "One-hot encoded spatial grid indicator."
            ]
        ],
        columns=[
            "Feature",
            "Meaning"
        ]
    )

    st.dataframe(
        feature_info,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # FEATURE DISTRIBUTION
    # --------------------------------------------------------

    if not features_df.empty:

        st.markdown("---")

        st.subheader(
            "Feature Distribution"
        )

        available_features = [
            feature
            for feature in continuous_cols
            if feature in features_df.columns
        ]

        if available_features:

            selected_feature = st.selectbox(
                "Select feature",
                available_features
            )

            values = pd.to_numeric(
                features_df[
                    selected_feature
                ],
                errors="coerce"
            ).dropna()

            if len(values) > 0:

                distribution = (
                    pd.cut(
                        values,
                        bins=20
                    )
                    .value_counts()
                    .sort_index()
                )

                distribution.index = (
                    distribution.index.astype(
                        str
                    )
                )

                st.bar_chart(
                    distribution
                )

    # --------------------------------------------------------
    # REPORTED FEATURE RANKING
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Mutual Information Feature Ranking"
    )

    ranking = pd.DataFrame(
        {
            "Feature": [
                "Cumulative seismic energy",
                "Maximum magnitude",
                "30-day event count",
                "Spatial grid indicators"
            ],

            "Mutual Information": [
                0.092,
                0.088,
                0.076,
                0.052
            ]
        }
    )

    st.bar_chart(
        ranking.set_index(
            "Feature"
        )
    )

    st.caption(
        "Values shown are from the project's feature-analysis results."
    )


# ============================================================
# PAGE 6 — LIMITATIONS
# ============================================================

elif page == "Limitations":

    st.header(
        "Limitations & Responsible Use"
    )

    st.write(
        """
        This application is a research-oriented machine-learning
        interface. It should not be treated as an operational
        earthquake prediction or warning system.
        """
    )

    limitations = {
        "Historical catalogue":
            """
            The model learns patterns from historical earthquake
            catalogue data. Catalogue completeness and reporting
            quality can influence the learned relationships.
            """,

        "Coarse spatial resolution":
            """
            Japan is represented using six spatial grid cells,
            so individual faults and local geological differences
            are not represented at fine resolution.
            """,

        "14-day target":
            """
            The model predicts occurrence within a 14-day window.
            It does not predict the exact time of an earthquake.
            """,

        "False positives":
            """
            Strong recent seismic activity can result in elevated
            model probability even when an M5+ event does not occur
            inside the exact forecast window.
            """,

        "False negatives":
            """
            An M5+ earthquake can occur after a relatively quiet
            preceding period, meaning recent catalogue activity
            alone cannot capture every event.
            """,

        "No exact magnitude prediction":
            """
            The target is binary M5+ occurrence. The model does not
            estimate the exact magnitude of a future earthquake.
            """,

        "No exact location prediction":
            """
            The model operates on predefined spatial grid cells and
            does not identify the exact future epicentre.
            """
    }

    for title, description in limitations.items():

        with st.expander(title):

            st.write(
                description
            )

    st.markdown("---")

    st.subheader(
        "What the model actually estimates"
    )

    st.info(
        """
        INPUT
        Previous 30 days of seismic activity

        ↓

        MODEL

        Machine-learning classification model

        ↓

        OUTPUT

        Estimated probability of an M5+ earthquake
        occurring during the following 14 days
        within the selected spatial grid.

        It does NOT provide:
        • exact earthquake time
        • exact earthquake magnitude
        • exact epicentre
        • operational warning
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Japan Earthquake Forecasting | "
    "Machine Learning Research Dashboard"
)