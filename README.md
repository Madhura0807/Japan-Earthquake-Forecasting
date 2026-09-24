# Japan Earthquake Forecasting

A machine-learning research dashboard, built with Streamlit, that examines whether recent seismic activity provides a useful signal for short-horizon earthquake occurrence in Japan.

> **Disclaimer.** This project is a historical machine-learning experiment intended for research and demonstration purposes only. It is not an operational earthquake prediction or warning system. It does not predict the exact time, magnitude, or epicentre of any earthquake.

---

## 1. Overview

The project formulates earthquake occurrence as a binary classification problem:

- **Input:** Summary statistics of seismic activity observed during the preceding **30 days**.
- **Output:** The estimated probability that at least one earthquake of magnitude **M5.0 or greater** occurs during the following **14 days** within a given spatial grid cell.

Japan is divided into six coarse spatial grid cells (south-west, south-east, central-west, central-east, north-west, north-east). Several trained classifiers and two baseline models are compared using a chronological train/test split.

## 2. Research Questions

1. Can recent 30-day seismicity provide useful information for 14-day M5+ occurrence?
2. Which recent-activity features contain the strongest signal?
3. Where do false alarms and missed events occur?

## 3. Features

Each forecasting window is represented by the following features:

| Feature | Description |
|---|---|
| `log_cum_energy_30d` | Log-transformed cumulative seismic energy during the previous 30 days |
| `max_mag_30d` | Maximum earthquake magnitude during the previous 30 days |
| `log_event_count_30d` | Log-transformed number of events during the previous 30 days |
| `log_event_count_lag1` | Log-transformed event count for the preceding 30-day period (lag 1) |
| `mean_depth_imputed` | Mean focal depth, with training-based imputation |
| `recent_m45_30d` | Binary indicator of an M4.5+ event during the previous 30 days |
| `grid_*` | One-hot encoded spatial grid indicator |

Continuous features are standardised using a scaler fitted on the training partition and saved with the model bundle.

## 4. Models

The saved bundle (`forecast_models.joblib`) contains:

- **Trained models:** Logistic Regression, Random Forest, Gradient Boosted Trees (`HistGradientBoostingClassifier`), and Multi-Layer Perceptron.
- **Baselines:** `GridProbabilityBaseline` (per-grid historical rate) and `PersistenceBaseline` (probability conditioned on recent M4.5+ activity).
- **Preprocessing artefacts:** the fitted scaler, the list of continuous columns, the ordered feature set, and the grid column names.

## 5. Evaluation

Models are evaluated on a held-out **chronological** test partition covering **2020–2024**, comprising 270 observations with a positive base rate of 23.3%. Reported metrics include PR-AUC, ROC-AUC, Brier score, F1-score, precision, recall, accuracy, and balanced accuracy. Full results are stored in `models/test_metrics.csv` and displayed on the *Model Performance* page.

## 6. Application Pages

| Page | Purpose |
|---|---|
| **Overview** | Project description, summary statistics, forecasting pipeline, and research questions |
| **Earthquake Explorer** | Interactive filtering by magnitude and date; map, magnitude distribution, monthly activity, and record table |
| **14-Day Forecast** | Select a trained model, enter previous 30-day activity values, and obtain an estimated M5+ probability |
| **Model Performance** | Test-set metrics, PR-AUC and ROC-AUC charts, and the complete results table |
| **Feature Insights** | Feature definitions, feature distributions, and mutual-information ranking |
| **Limitations** | Discussion of constraints and responsible use |

## 7. Project Structure

```
.
├── app.py
├── data/
│   └── processed/
│       ├── model_features.csv
│       └── japan_clean_events.csv
         RAW

└── models/
    ├── forecast_models.joblib
    └── test_metrics.csv
```

## 8. Requirements

- Python 3.9 or later
- streamlit
- pandas
- numpy
- joblib
- scikit-learn (required to load the saved models; use the same version that was used for training)

Install the dependencies with:

```bash
pip install streamlit pandas numpy joblib scikit-learn
```

## 9. Usage

From the project root directory, run:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` by default.

If a data file is missing, the corresponding page displays a warning. If the model file is missing, the application reports the expected path and stops.

## 10. Implementation Notes

- The classes `GridProbabilityBaseline` and `PersistenceBaseline` are defined in `app.py` **before** the model bundle is loaded. The `.joblib` file was created when these classes resided in `__main__`, so they must be available under the same names at load time. Do not rename or relocate them without re-saving the bundle.
- The `grid_labels_by_col` global is populated after the bundle is loaded and is required by `GridProbabilityBaseline.predict_proba`.
- Input features on the forecast page are assembled in the exact order stored in the bundle's `feature_set`, and the saved scaler is applied to the continuous columns before prediction.

## 11. Limitations

- **Historical catalogue:** Learned relationships depend on catalogue completeness and reporting quality.
- **Coarse spatial resolution:** Six grid cells cannot represent individual faults or local geological variation.
- **Binary, windowed target:** The model estimates occurrence within a 14-day window, not the precise timing.
- **False positives:** Strong recent activity can raise the predicted probability even when no M5+ event follows.
- **False negatives:** M5+ events can follow quiet periods, so recent catalogue activity alone cannot capture every event.
- **No magnitude or location prediction:** The target is binary M5+ occurrence within a predefined grid cell.

## 12. Responsible Use

Outputs of this application are model probability estimates derived from historical data. They must not be used for safety-critical decisions, public warnings, or emergency planning. For authoritative earthquake information, consult official agencies such as the Japan Meteorological Agency (JMA).
