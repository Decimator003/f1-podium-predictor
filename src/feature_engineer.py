"""
F1 Podium Predictor - Feature Engineering Module

This module merges, cleans, and creates comprehensive feature sets for F1 podium prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import warnings
# warnings.filterwarnings('ignore')

# Robust imports: support running as package or script
try:
    # When executed as part of the src package
    from .data_loader import load_all_data_sources  # type: ignore
except Exception:  # noqa: BLE001
    try:
        # When executed with project root on sys.path
        from src.data_loader import load_all_data_sources  # type: ignore
    except Exception:
        # As a final fallback, inject project root to sys.path and import
        import sys
        CURRENT_DIR = Path(__file__).resolve().parent
        PROJECT_ROOT = CURRENT_DIR.parent
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.data_loader import load_all_data_sources  # type: ignore


def merge_session_data_with_rain_index(session_data: pd.DataFrame, rain_index: pd.DataFrame, driver_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Merge driver session data with rain performance indices and driver metadata (dry ratings).
    
    Args:
        session_data (pd.DataFrame): Driver session data
        rain_index (pd.DataFrame): Rain performance indices
        driver_metadata (pd.DataFrame): Driver metadata containing normal/dry ratings
        
    Returns:
        pd.DataFrame: Merged session data with rain performance and dry ratings
    """
    # Merge session with rain index on DriverCode
    merged_data = session_data.merge(
        rain_index[['DriverCode', 'RainRating', 'DeltaRating', 'GameRating']],
        on='DriverCode',
        how='left'
    )

    # Merge driver metadata to get DryRating (normal rating)
    metadata_cols = ['DriverCode']
    if 'Rating' in driver_metadata.columns:
        driver_metadata = driver_metadata.rename(columns={'Rating': 'DryRating'})
        metadata_cols.append('DryRating')
    # Optionally pull DriverPoints from metadata if not present
    if 'DriverPoints' in driver_metadata.columns and 'DriverPoints' not in merged_data.columns:
        metadata_cols.append('DriverPoints')
    
    merged_data = merged_data.merge(
        driver_metadata[metadata_cols],
        on='DriverCode',
        how='left'
    )
    
    print(f"✅ Merged session data with rain index and driver metadata: {len(merged_data)} drivers")
    return merged_data


def integrate_circuit_features(session_data: pd.DataFrame, circuit_info: pd.DataFrame, race_code: str) -> pd.DataFrame:
    """
    Integrate circuit-specific features with session data.
    
    Args:
        session_data (pd.DataFrame): Driver session data
        circuit_info (pd.DataFrame): Circuit characteristics
        race_code (str): Race identifier
        
    Returns:
        pd.DataFrame: Session data with circuit features
    """
    # Filter circuit info for the specific race
    circuit_data = circuit_info[circuit_info['race_code'] == race_code]
    
    if circuit_data.empty:
        print(f"⚠️  No circuit info found for {race_code}")
        return session_data
    
    # Add circuit features to all drivers
    for col in ['street_circuit', 'overtake_score', 'avg_speed_kph', 'circuit_length_km', 'elevation_change_m', 'turn_count']:
        if col in circuit_data.columns:
            session_data[col] = circuit_data[col].iloc[0]
    
    print(f"✅ Integrated circuit features for {race_code}")
    return session_data


def create_weather_features(session_data: pd.DataFrame, weather_data: Dict[str, Any], rain_index: pd.DataFrame) -> pd.DataFrame:
    """
    Create weather-adjusted performance features.
    
    Args:
        session_data (pd.DataFrame): Driver session data
        weather_data (Dict[str, Any]): Weather forecast data
        rain_index (pd.DataFrame): Rain performance indices
        
    Returns:
        pd.DataFrame: Session data with weather-adjusted features
    """
    # Extract rain probability (given in percentage, e.g., 64 means 64%)
    rain_probability_pct = weather_data.get('rain_probability', 0)
    try:
        rain_probability_pct = float(rain_probability_pct)
    except Exception:
        rain_probability_pct = 0.0
    rain_probability = max(0.0, min(1.0, rain_probability_pct / 100.0))

    session_data['rain_probability_pct'] = rain_probability_pct
    session_data['rain_probability'] = rain_probability

    # Threshold: use rain ratings when probability >= 75%
    use_rain = rain_probability >= 0.75
    session_data['use_rain_ratings'] = int(use_rain)

    # Effective driver rating: choose between RainRating and DryRating based on threshold
    session_data['effective_driver_rating'] = np.where(
        use_rain,
        session_data.get('RainRating', np.nan),
        session_data.get('DryRating', np.nan)
    )

    # Weather-adjusted rating can mirror the effective selection
    session_data['weather_adjusted_rain_rating'] = session_data['effective_driver_rating']

    # Rain performance multiplier only boosts in heavy-rain scenarios
    session_data['rain_performance_multiplier'] = np.where(
        use_rain,
        1 + (rain_probability * 0.2),
        1.0
    )
    
    print(f"✅ Created weather features with rain probability: {rain_probability_pct:.1f}% (threshold 75%)")
    return session_data


def calculate_derived_statistics(session_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived statistics and consistency metrics.
    
    Args:
        session_data (pd.DataFrame): Driver session data
        
    Returns:
        pd.DataFrame: Session data with derived statistics
    """
    # Practice session consistency (lower std = more consistent)
    practice_times = ['P1Time', 'P2Time', 'P3Time']
    valid_practice_times = session_data[practice_times].replace([np.inf, -np.inf], np.nan)
    
    session_data['practice_consistency'] = valid_practice_times.std(axis=1, skipna=True)
    session_data['avg_practice_time'] = valid_practice_times.mean(axis=1, skipna=True)
    
    # Best practice time
    session_data['best_practice_time'] = valid_practice_times.min(axis=1, skipna=True)
    
    # Qualifying performance
    quali_times = ['Q1Time', 'Q2Time', 'Q3Time']
    valid_quali_times = session_data[quali_times].replace([np.inf, -np.inf], np.nan)
    
    session_data['best_quali_time'] = valid_quali_times.min(axis=1, skipna=True)
    session_data['quali_consistency'] = valid_quali_times.std(axis=1, skipna=True)
    
    # Practice to qualifying improvement
    session_data['practice_to_quali_improvement'] = (
        session_data['avg_practice_time'] - session_data['best_quali_time']
    )
    
    # Lap count features (more laps = more practice)
    lap_counts = ['P1Laps', 'P2Laps', 'P3Laps']
    session_data['total_practice_laps'] = session_data[lap_counts].sum(axis=1, skipna=True)
    session_data['avg_laps_per_session'] = session_data[lap_counts].mean(axis=1, skipna=True)
    
    print(f"✅ Calculated derived statistics for {len(session_data)} drivers")
    return session_data


def add_previous_year_performance(session_data: pd.DataFrame, previous_results: pd.DataFrame, race_code: str) -> pd.DataFrame:
    """
    Add previous year's performance at the same circuit, with DNF/DNP handling.
    
    Args:
        session_data (pd.DataFrame): Driver session data
        previous_results (pd.DataFrame): Previous season results
        race_code (str): Current race identifier
        
    Returns:
        pd.DataFrame: Session data with historical performance
    """
    # Extract circuit name from race code (e.g., "2025_Hungarian_GP" -> "Hungarian_GP")
    circuit_name = race_code.split('_', 1)[1] if '_' in race_code else race_code
    
    # Find previous year's race for the same circuit
    previous_race = previous_results[previous_results['Race_Name'].str.contains(circuit_name, na=False)]
    
    if not previous_race.empty:
        # Use status to derive experience/finish flags
        previous_performance = previous_race[['Driver_Code', 'Finish_Position', 'Total_Time_Seconds', 'Status']].copy()
        previous_performance.columns = ['DriverCode', 'prev_finish_position', 'prev_total_time', 'prev_status']
        
        # Participation and finish flags
        previous_performance['prev_participated_same_circuit'] = (previous_performance['prev_status'] != 'DNP').astype(int)
        previous_performance['prev_finished_same_circuit'] = (previous_performance['prev_status'] == 'Finished').astype(int)
        previous_performance['prev_dnf'] = (previous_performance['prev_status'] == 'DNF').astype(int)
        
        # Merge with current session data
        session_data = session_data.merge(previous_performance, on='DriverCode', how='left')
        
        # Create top-k indicators only when a position is available (participated)
        session_data['prev_top_3'] = ((session_data['prev_finish_position'] <= 3)).astype(float)
        session_data['prev_top_5'] = ((session_data['prev_finish_position'] <= 5)).astype(float)
        session_data['prev_top_10'] = ((session_data['prev_finish_position'] <= 10)).astype(float)
        
        # Where no participation, set top-k to 0
        session_data.loc[session_data['prev_participated_same_circuit'] != 1, ['prev_top_3', 'prev_top_5', 'prev_top_10']] = 0.0
        
        print(f"✅ Added previous year performance (with DNF/DNP flags) for {len(previous_race)} drivers")
    else:
        print(f"⚠️  No previous year data found for {circuit_name}")
        # Add empty columns
        session_data['prev_finish_position'] = np.nan
        session_data['prev_total_time'] = np.nan
        session_data['prev_status'] = 'Unknown'
        session_data['prev_participated_same_circuit'] = 0
        session_data['prev_finished_same_circuit'] = 0
        session_data['prev_dnf'] = 0
        session_data['prev_top_3'] = 0.0
        session_data['prev_top_5'] = 0.0
        session_data['prev_top_10'] = 0.0
    
    return session_data


def create_final_feature_set(session_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create the final feature set for model training.
    
    Args:
        session_data (pd.DataFrame): Processed session data
        
    Returns:
        pd.DataFrame: Final feature set
    """
    # Select and order features for the model
    feature_columns = [
        # Driver identification
        'DriverCode', 'TeamCode',
        
        # Current performance
        'DriverPoints', 'TeamPoints',
        
        # Ratings
        'DryRating', 'RainRating', 'DeltaRating', 'GameRating', 'effective_driver_rating',
        
        # Practice performance
        'best_practice_time', 'avg_practice_time', 'practice_consistency',
        'total_practice_laps', 'avg_laps_per_session',
        
        # Qualifying performance
        'best_quali_time', 'quali_consistency',
        'practice_to_quali_improvement',
        
        # Weather
        'rain_probability_pct', 'rain_probability', 'use_rain_ratings', 'rain_performance_multiplier',
        
        # Circuit characteristics
        'street_circuit', 'overtake_score', 'avg_speed_kph', 
        'circuit_length_km', 'elevation_change_m', 'turn_count',
        
        # Historical performance
        'prev_finish_position', 'prev_total_time', 'prev_status',
        'prev_participated_same_circuit', 'prev_finished_same_circuit', 'prev_dnf',
        'prev_top_3', 'prev_top_5', 'prev_top_10'
    ]
    
    # Filter to available columns
    available_features = [col for col in feature_columns if col in session_data.columns]
    
    final_features = session_data[available_features].copy()
    
    # Fill missing values
    numeric_columns = final_features.select_dtypes(include=[np.number]).columns
    final_features[numeric_columns] = final_features[numeric_columns].fillna(final_features[numeric_columns].median())
    
    # Fill categorical columns
    categorical_columns = final_features.select_dtypes(include=['object']).columns
    final_features[categorical_columns] = final_features[categorical_columns].fillna('Unknown')
    
    print(f"✅ Created final feature set with {len(final_features)} drivers and {len(available_features)} features")
    return final_features


def engineer_features(race_code: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Main function to engineer features for a specific race.
    
    Args:
        race_code (str): Race identifier (e.g., '2025_Hungarian_GP')
        output_path (Optional[str]): Path to save processed features
        
    Returns:
        pd.DataFrame: Engineered feature set
    """
    print(f"🔄 Starting feature engineering for {race_code}...")
    
    try:
        # Load all data sources
        data_sources = load_all_data_sources(race_code)
        
        # Start with session data
        session_data = data_sources['session_data'].copy()
        
        # Step 1: Merge session data with rain index and driver metadata
        session_data = merge_session_data_with_rain_index(session_data, data_sources['rain_index'], data_sources['driver_metadata'])
        
        # Step 2: Integrate circuit features
        session_data = integrate_circuit_features(session_data, data_sources['circuit_info'], race_code)
        
        # Step 3: Create weather features (percentage + threshold handling)
        session_data = create_weather_features(session_data, data_sources['weather_data'], data_sources['rain_index'])
        
        # Step 4: Calculate derived statistics
        session_data = calculate_derived_statistics(session_data)
        
        # Step 5: Add previous year performance (DNF/DNP flags)
        session_data = add_previous_year_performance(session_data, data_sources['previous_results'], race_code)
        
        # Step 6: Create final feature set
        final_features = create_final_feature_set(session_data)
        
        # Save to processed directory if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            final_features.to_csv(output_file, index=False)
            print(f"✅ Saved engineered features to {output_file}")
        
        print(f"✅ Feature engineering completed for {race_code}")
        return final_features
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        raise


def get_feature_summary(features: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of the engineered features.
    
    Args:
        features (pd.DataFrame): Engineered feature set
        
    Returns:
        Dict[str, Any]: Feature summary statistics
    """
    summary = {
        'total_drivers': len(features),
        'total_features': len(features.columns),
        'numeric_features': len(features.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(features.select_dtypes(include=['object']).columns),
        'missing_values': features.isnull().sum().sum(),
        'feature_list': list(features.columns)
    }
    
    print(f"📊 Feature Summary:")
    print(f"   - Total drivers: {summary['total_drivers']}")
    print(f"   - Total features: {summary['total_features']}")
    print(f"   - Numeric features: {summary['numeric_features']}")
    print(f"   - Categorical features: {summary['categorical_features']}")
    print(f"   - Missing values: {summary['missing_values']}")
    
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Engineer features for a given F1 race")
    parser.add_argument("race_code", nargs="?", default="2025_Hungarian_GP", help="Race code, e.g., 2025_Hungarian_GP")
    parser.add_argument("--out", dest="output_path", default=str(Path("data/processed") / "features.csv"), help="Output CSV path")
    args = parser.parse_args()

    print(f"🔧 Running feature engineering for {args.race_code} ...")
    try:
        features = engineer_features(args.race_code, args.output_path)
        summary = get_feature_summary(features)
        print("\n✅ Done. Saved to:", args.output_path)
        print("Columns:", ", ".join(features.columns.tolist()))
        print("Records:", len(features))
    except Exception as e:
        print("❌ Failed:", e)
        raise
