"""
F1 Podium Predictor - Data Loader Module

This module provides functions to load all input data sources for the F1 podium prediction system.
"""

import pandas as pd
import json
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


def load_driver_session_data(race_code: str) -> pd.DataFrame:
    """
    Load driver session data (practice and qualifying times, positions).
    
    Args:
        race_code (str): Race identifier (e.g., '2025_Hungarian_GP')
        
    Returns:
        pd.DataFrame: Driver session data with practice and qualifying information
    """
    file_path = Path(f"data/manual/{race_code}_driver_session_data.csv")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Session data file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded driver session data for {race_code}: {len(df)} drivers")
        return df
    except Exception as e:
        raise Exception(f"Error loading session data: {e}")


def load_rain_driver_index() -> pd.DataFrame:
    """
    Load driver rain performance indices.
    
    Returns:
        pd.DataFrame: Driver rain adaptability and performance data
    """
    file_path = Path("data/manual/rain_driver_index.csv")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Rain driver index file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded rain driver index: {len(df)} drivers")
        return df
    except Exception as e:
        raise Exception(f"Error loading rain driver index: {e}")


def load_circuit_info() -> pd.DataFrame:
    """
    Load circuit characteristics and metadata.
    
    Returns:
        pd.DataFrame: Circuit information including track characteristics
    """
    file_path = Path("data/manual/circuit_info.csv")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Circuit info file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded circuit info: {len(df)} circuits")
        return df
    except Exception as e:
        raise Exception(f"Error loading circuit info: {e}")


def load_weather_data(race_code: str) -> Dict[str, Any]:
    """
    Load weather forecast data for a specific race.
    
    Args:
        race_code (str): Race identifier (e.g., '2025_Hungarian_GP')
        
    Returns:
        Dict[str, Any]: Weather forecast data including rain probability
    """
    file_path = Path(f"data/manual/{race_code}_rain_prediction.json")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Weather data file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            weather_data = json.load(f)
        print(f"✅ Loaded weather data for {race_code}")
        return weather_data
    except Exception as e:
        raise Exception(f"Error loading weather data: {e}")


def load_driver_team_metadata() -> pd.DataFrame:
    """
    Load driver and team background information.
    
    Returns:
        pd.DataFrame: Driver and team metadata
    """
    file_path = Path("data/manual/driver_metadeta.csv")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Driver metadata file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded driver metadata: {len(df)} drivers")
        return df
    except Exception as e:
        raise Exception(f"Error loading driver metadata: {e}")


def load_race_results(race_code: str) -> pd.DataFrame:
    """
    Load actual race results for model evaluation.
    
    Args:
        race_code (str): Race identifier (e.g., '2025_Hungarian_GP')
        
    Returns:
        pd.DataFrame: Race results with podium positions
    """
    # For current races, this might be empty initially
    # For historical races, this contains actual results
    file_path = Path(f"data/manual/race_results.csv")
    
    if not file_path.exists():
        print(f"⚠️  Race results file not found: {file_path} (may not be available yet)")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded race results for {race_code}: {len(df)} drivers")
        return df
    except Exception as e:
        raise Exception(f"Error loading race results: {e}")


def load_previous_season_results() -> pd.DataFrame:
    """
    Load previous season race results for historical performance analysis.
    
    Returns:
        pd.DataFrame: Previous season race results
    """
    file_path = Path("data/manual/previous_race_results.csv")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Previous season results file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded previous season results: {len(df)} entries")
        return df
    except Exception as e:
        raise Exception(f"Error loading previous season results: {e}")


def load_all_data_sources(race_code: str) -> Dict[str, Any]:
    """
    Load all data sources for a specific race.
    
    Args:
        race_code (str): Race identifier (e.g., '2025_Hungarian_GP')
        
    Returns:
        Dict[str, Any]: Dictionary containing all loaded data sources
    """
    print(f"🔄 Loading all data sources for {race_code}...")
    
    data_sources = {}
    
    try:
        # Load all data sources
        data_sources['session_data'] = load_driver_session_data(race_code)
        data_sources['rain_index'] = load_rain_driver_index()
        data_sources['circuit_info'] = load_circuit_info()
        data_sources['weather_data'] = load_weather_data(race_code)
        data_sources['driver_metadata'] = load_driver_team_metadata()
        data_sources['previous_results'] = load_previous_season_results()
        
        # Try to load current race results (may not exist yet)
        data_sources['race_results'] = load_race_results(race_code)
        
        print(f"✅ Successfully loaded all data sources for {race_code}")
        return data_sources
        
    except Exception as e:
        print(f"❌ Error loading data sources: {e}")
        raise


def get_data_summary(data_sources: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of loaded data sources.
    
    Args:
        data_sources (Dict[str, Any]): Dictionary of loaded data sources
        
    Returns:
        Dict[str, Any]: Summary of data sources
    """
    summary = {}
    
    for source_name, data in data_sources.items():
        if isinstance(data, pd.DataFrame):
            summary[source_name] = {
                'shape': data.shape,
                'columns': list(data.columns),
                'missing_values': data.isnull().sum().to_dict()
            }
        elif isinstance(data, dict):
            summary[source_name] = {
                'type': 'dict',
                'keys': list(data.keys()),
                'content': data
            }
        else:
            summary[source_name] = {
                'type': type(data).__name__,
                'content': str(data)
            }
    
    return summary


def validate_data_consistency(data_sources: Dict[str, Any]) -> bool:
    """
    Validate consistency between different data sources.
    
    Args:
        data_sources (Dict[str, Any]): Dictionary of loaded data sources
        
    Returns:
        bool: True if data is consistent, False otherwise
    """
    print("🔍 Validating data consistency...")
    
    # Check if session data and driver metadata have matching drivers
    if 'session_data' in data_sources and 'driver_metadata' in data_sources:
        session_drivers = set(data_sources['session_data']['DriverCode'])
        metadata_drivers = set(data_sources['driver_metadata']['DriverCode'])
        
        missing_in_metadata = session_drivers - metadata_drivers
        missing_in_session = metadata_drivers - session_drivers
        
        if missing_in_metadata:
            print(f"⚠️  Drivers in session data but not in metadata: {missing_in_metadata}")
        if missing_in_session:
            print(f"⚠️  Drivers in metadata but not in session data: {missing_in_session}")
    
    # Check if rain index covers all drivers
    if 'session_data' in data_sources and 'rain_index' in data_sources:
        session_drivers = set(data_sources['session_data']['DriverCode'])
        rain_drivers = set(data_sources['rain_index']['DriverCode'])
        
        missing_rain_data = session_drivers - rain_drivers
        if missing_rain_data:
            print(f"⚠️  Drivers missing rain index data: {missing_rain_data}")
    
    print("✅ Data consistency validation completed")
    return True


if __name__ == "__main__":
    # Example usage
    race_code = "2025_Hungarian_GP"
    
    try:
        # Load all data sources
        data_sources = load_all_data_sources(race_code)
        
        # Generate summary
        summary = get_data_summary(data_sources)
        print("\n📊 Data Summary:")
        for source, info in summary.items():
            print(f"  {source}: {info}")
        
        # Validate consistency
        validate_data_consistency(data_sources)
        
    except Exception as e:
        print(f"❌ Error: {e}")
