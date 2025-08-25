"""
Simple Rain Probability Checker
Returns only rain probability in JSON format
"""

import requests
import json
from urllib.parse import urlencode
from datetime import datetime


def get_rain_probability(latitude, longitude, date, hour):
    """
    Gets rain probability for a specific location, date and time
    
    Args:
        latitude (float): Latitude coordinate (-90 to 90)
        longitude (float): Longitude coordinate (-180 to 180)
        date (str): Date in YYYY-MM-DD format
        hour (int): Hour in 24-hour format (0-23)
    
    Returns:
        dict: JSON with rain probability only
    """
    # Validate inputs
    if not -90 <= latitude <= 90:
        raise ValueError('Latitude must be between -90 and 90')
    if not -180 <= longitude <= 180:
        raise ValueError('Longitude must be between -180 and 180')
    if not 0 <= hour <= 23:
        raise ValueError('Hour must be between 0 and 23')
    
    # Validate date format
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise ValueError('Date must be in YYYY-MM-DD format')
    
    # Build API URL for Open-Meteo
    base_url = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'precipitation_probability',
        'start_date': date,
        'end_date': date,
        'timezone': 'UTC'
    }
    
    try:
        response = requests.get(f"{base_url}?{urlencode(params)}", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract rain probability for the specific hour
        hourly_data = data.get('hourly', {})
        precipitation_prob = hourly_data.get('precipitation_probability', [])
        
        if hour >= len(precipitation_prob):
            raise ValueError('No data available for the specified hour')
        
        rain_probability = precipitation_prob[hour] if precipitation_prob[hour] is not None else 0
        
        return {"rain_probability": rain_probability}
        
    except requests.RequestException as e:
        raise requests.RequestException(f"API request failed: {e}")


if __name__ == '__main__':
    # Race configuration - modify these for different races
    race_config = {
        'year': '2025',
        'race_name': 'Hungarian_GP',
        'latitude': 47.5819,      # Hungaroring latitude
        'longitude': 19.2508,     # Hungaroring longitude
        'race_date': '2025-07-27', # Hungarian GP 2025 date
        'race_hour': 13,          # Race start hour in UTC
        'circuit_name': 'Hungaroring',
        'location': 'Mogyoród, Hungary'
    }
    
    try:
        # Get rain probability
        result = get_rain_probability(
            race_config['latitude'], 
            race_config['longitude'], 
            race_config['race_date'], 
            race_config['race_hour']
        )
        
        # Just save the rain probability result
        race_data = result
        
        # Create filename with race code
        filename = f"{race_config['year']}_{race_config['race_name']}_rain_prediction.json"
        filepath = f"data/manual/{filename}"
        
        # Save to manual folder
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(race_data, f, indent=4, ensure_ascii=False)
        
        print(f"Rain prediction data saved to: {filepath}")
        print(json.dumps(race_data, indent=2))
        
    except Exception as e:
        error_result = {"error": str(e)}
        print(json.dumps(error_result))