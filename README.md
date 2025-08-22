# 🏁 F1 Podium Predictor – Phase 1

Predicting **top 3 drivers (P1, P2, P3)** for Formula 1 races using comprehensive race data analysis.

## 🎯 **Project Goal**

Build a machine learning system that predicts the podium finishers for any upcoming F1 race by analyzing:
- Practice and qualifying session data
- Circuit characteristics and metadata
- Driver and team performance statistics
- Weather conditions and forecasts
- Pit stop performance metrics
- Driver rain adaptability indices

## 📁 **Project Structure**

```
F1_Podium_Predictor/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── manual/                    # Manually collected/curated data
│   │   ├── 2025_Hungarian_GP_driver_session_data.csv
│   │   ├── rain_driver_index.csv
│   │   ├── circuit_info.csv       # Circuit metadata
│   │   ├── driver_metadata.csv
│   │   ├── pitstop_data.csv       # Aggregate pitstop data across races
│   │   ├── race_results.csv       # Aggregate race results (podium only)
│   │   └── previous_season_results.csv# Historical race data (last year, all drivers)
│   │
│   ├── raw/                       # Future: unprocessed scraped data
│   ├── processed/                 # Final engineered feature sets
│   │   └── 2025_Hungarian_GP_features.csv
│   └── weather/                   # Weather API data
│       └── 2025_Hungarian_GP_forecast.json
│
├── models/
│   └── podium_model.pkl           # Trained XGBoost model
│
├── notebooks/
│   ├── 01_data_collection.ipynb   # Main race prediction pipeline
│   └── rain_processing.py         # Generate rain driver indices
│
├── outputs/
│   ├── metrics/
│   │   └── model_eval_2025_Hungarian_GP.json
│   └── visualization/
│       └── podium_probs_plot.png
│
└── src/
    ├── data_loader.py             # Load all input data sources
    ├── feature_engineer.py        # Build complete feature set
    └── model.py                   # Train and predict podium
```

## 🧱 **Implementation Modules**

### 🔹 **data_loader.py**
**Purpose**: Centralized data loading for all input sources

| Function | Input File | Status | Description |
|----------|------------|--------|-------------|
| `load_driver_session_data()` | `*_driver_session_data.csv` | ✅ Done | Practice/quali times, positions |
| `load_rain_driver_index()` | `rain_driver_index.csv` | ✅ Done | Driver wet weather performance |
| `load_circuit_info()` | `circuit_info.csv` | ✅ Done | Track characteristics, layout |
| `load_weather_data()` | `*_forecast.json` | ⬜ To implement | Rain probability, conditions |
| `load_pitstop_data()` | `pitstop_data.csv` | ⬜ To implement | Aggregate pit stop performance |
| `load_driver_team_metadata()` | `driver_metadata.csv` | ✅ Done | Driver/team background info |
| `load_race_results()` | `race_results.csv` | ⬜ To implement | Actual results for evaluation |
| `load_previous_season_results()` | `previous_season_results.csv` | ⬜ To implement | Driver/Team perfomance last season |


### 🔹 **feature_engineer.py**
**Purpose**: Merge, clean, and create comprehensive feature set

| Feature Component | Dependencies | Status | Description |
|------------------|--------------|--------|-------------|
| Session data merging | Driver session + rain index | ⬜ To implement | Combine practice/quali with rain data |
| Circuit integration | Circuit info + session data | ⬜ To implement | Add track-specific features |
| Pit stop metrics | Pit stop data | ⬜ To implement | Team pit stop performance |
| Weather features | Rain index + weather forecast | ⬜ To implement | Rain-adjusted performance |
| Derived statistics | Calculated from base data | ⬜ To implement | Qualifying deltas, consistency metrics |
| Previous year performance | Historical race results + driver data | ⬜ To implement | Add prior year’s performance at same circuit |
| Final feature set | All above components | ⬜ To implement | Output to `data/processed/` |

### 🔹 **model.py**
**Purpose**: Train XGBoost model and predict podium

| Task | Implementation | Status | Description |
|------|----------------|--------|-------------|
| Model training | XGBClassifier | ⬜ To implement | Train on historical race data |
| Probability prediction | predict_proba() | ⬜ To implement | Get podium probabilities for each driver |
| Top 3 selection | Sort by probability | ⬜ To implement | Return P1, P2, P3 with confidence |
| Model persistence | Pickle save | ⬜ Optional | Save trained model to `/models/` |

### 🔹 **rain_processing.py**
**Purpose**: Generate driver rain performance indices

| Task | Status | Description |
|------|--------|-------------|
| Calculate wet deltas | ✅ Done | Empirical wet vs dry performance |
| Merge F1 game scores | ✅ Done | Combine with simulation data |
| Output rain index | ✅ Done | Generate `rain_driver_index.csv` |

### 🔹 **01_data_collection.ipynb**
**Purpose**: End-to-end race prediction pipeline

| Section | Status | Description |
|---------|--------|-------------|
| Data loading | ⬜ To implement | Import and load all race data |
| Feature engineering | ⬜ To implement | Build complete feature set |
| Model training | ⬜ To implement | Train on available data |
| Podium prediction | ⬜ To implement | Predict top 3 for current race |
| Results visualization | ⬜ Optional | Generate plots and metrics |

## 📊 **Data Flow**

1. **Input Data Collection**
   - Manual data entry for session times, positions
   - Weather API data collection
   - Circuit metadata compilation
   - Pit stop performance aggregation
   - Driver/team metadata
   - Previous season results (global reference file)

2. **Feature Engineering**
   - Merge all data sources on driver/session/circuit
   - Join with previous season’s race result for the same GP (e.g., Hungarian GP 2025 uses Hungarian GP 2024 row)
   - Create derived features (deltas, consistency metrics)
   - Handle missing data and edge cases
   - Output processed feature set

3. **Model Training & Prediction**
   - Train XGBoost on historical data (current and previous race features)
   - Predict podium probabilities for current race
   - Return top 3 drivers with confidence scores

4. **Evaluation & Validation**
   - Compare predictions with actual race results
   - Track model performance over time
   - Generate performance metrics and visualizations

## 🎯 **Current Focus Areas**

### **Immediate Next Steps**
1. ✅ Complete `data_loader.py` functions for all data sources
2. ✅ Implement `feature_engineer.py` to merge and process data
3. ✅ Build `model.py` for training and prediction
4. ✅ Create end-to-end pipeline in notebook

### **Data Requirements**
- **Circuit Info**: Track characteristics, layout, DRS zones
- **Pit Stop Data**: Aggregate performance across all races
- **Race Results**: Podium positions for model evaluation
- **Weather Data**: Rain probability and conditions
- **Driver Metadata**: Background information and stats
- **Previous Season Data**: Last year data for the race

## 🚀 **Usage**

### **For a New Race**
1. Collect session data (practice, qualifying)
2. Update weather forecast data
3. Run the prediction pipeline:
   ```python
   # In 01_data_collection.ipynb
   from src.data_loader import *
   from src.feature_engineer import *
   from src.model import *
   
   # Load data
   session_data = load_driver_session_data("2025_Hungarian_GP")
   # ... load other data sources
   
   # Build features
   features = engineer_features(session_data, ...)
   
   # Predict podium
   podium_prediction = predict_podium(features)
   ```

### **Model Evaluation**
- Compare predictions with actual race results
- Track accuracy over multiple races
- Generate performance metrics and visualizations

## 📈 **Future Enhancements (Phase 2+)**
- Automated data collection from F1 APIs
- Real-time weather integration
- Advanced feature engineering (tire degradation, fuel loads)
- Ensemble models and deep learning approaches
- Web interface for predictions
- Historical performance analysis

## 🔧 **Technical Stack**
- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **XGBoost**: Machine learning model
- **Scikit-learn**: Model evaluation and preprocessing
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive development and analysis

---

*This project aims to combine traditional F1 analysis with modern machine learning techniques to predict race outcomes with high accuracy.*
