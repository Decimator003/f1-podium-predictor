import fastf1
import pandas as pd
import numpy as np
from datetime import timedelta

# Driver to team mapping
driver_to_team = {
    'PIA': 'MCL', 'NOR': 'MCL', 'VER': 'RBR', 'RUS': 'MERC', 'LEC': 'FER',
    'HAM': 'FER', 'ANT': 'MERC', 'ALB': 'WIL', 'HUL': 'SAU', 'OCO': 'HAA',
    'HAD': 'VCARB', 'GAS': 'ALP', 'STR': 'AST', 'LAW': 'VCARB', 'ALO': 'AST',
    'SAI': 'WIL', 'TSU': 'RBR', 'BEA': 'HAA', 'BOR': 'SAU', 'COL': 'ALP'
}

def get_2024_hungarian_gp_data(output_path="data/manual/2024_Hungarian_GP_race_results.csv"):
    """
    Extract 2024 Hungarian GP race data for all drivers in the mapping.
    
    Args:
        output_path (str): Path to save the CSV file
    
    Returns:
        pd.DataFrame: Race data with columns [Race_Name, Driver_Code, Team_Code, 
                     Quali_Position, Start_Position, Finish_Position, Total_Time_Seconds, 
                     Best_Lap_Time_Seconds, Average_Lap_Time_Seconds]
    """
    
    # Load 2024 Hungarian GP race session
    print("Loading 2024 Hungarian GP race session...")
    race_session = fastf1.get_session(2024, 'Hungary', 'R')
    race_session.load()
    
    # Load qualifying session for quali positions
    print("Loading qualifying session...")
    quali_session = fastf1.get_session(2024, 'Hungary', 'Q')
    quali_session.load()
    
    # Get race results
    race_results = race_session.results
    quali_results = quali_session.results
    
    # Get lap data for timing analysis
    laps = race_session.laps
    
    # Initialize results list
    results_list = []
    
    for driver_code, team_code in driver_to_team.items():
        print(f"Processing {driver_code} ({team_code})...")
        
        # Initialize row with basic info
        row = {
            'Race_Name': '2024_Hungarian_GP',
            'Driver_Code': driver_code,
            'Team_Code': team_code,
            'Quali_Position': None,
            'Start_Position': None,
            'Finish_Position': None,
            'Total_Time_Seconds': None,
            'Best_Lap_Time_Seconds': None,
            'Average_Lap_Time_Seconds': None,
            'Status': 'DNF'  # Default to DNF, will update if finished
        }
        
        # Find driver in qualifying results
        quali_driver_data = quali_results[quali_results['Abbreviation'] == driver_code]
        if not quali_driver_data.empty:
            row['Quali_Position'] = int(quali_driver_data.iloc[0]['Position']) if pd.notna(quali_driver_data.iloc[0]['Position']) else None
        
        # Find driver in race results
        race_driver_data = race_results[race_results['Abbreviation'] == driver_code]
        if not race_driver_data.empty:
            driver_result = race_driver_data.iloc[0]
            
            # Get positions
            row['Start_Position'] = int(driver_result['GridPosition']) if pd.notna(driver_result['GridPosition']) else None
            row['Finish_Position'] = int(driver_result['Position']) if pd.notna(driver_result['Position']) else None
            
            # Get total race time (convert to seconds)
            if pd.notna(driver_result['Time']):
                if isinstance(driver_result['Time'], timedelta):
                    row['Total_Time_Seconds'] = driver_result['Time'].total_seconds()
                    row['Status'] = 'Finished'
                else:
                    # Handle string format like "1:21:478"
                    time_str = str(driver_result['Time'])
                    if ':' in time_str:
                        try:
                            parts = time_str.split(':')
                            if len(parts) == 3:  # H:M:S format
                                hours = int(parts[0])
                                minutes = int(parts[1])
                                seconds = float(parts[2])
                                row['Total_Time_Seconds'] = hours * 3600 + minutes * 60 + seconds
                                row['Status'] = 'Finished'
                        except:
                            pass
            
            # Get driver's lap data
            driver_laps = laps[laps['Driver'] == driver_code]
            if not driver_laps.empty:
                # Best lap time
                valid_laps = driver_laps.dropna(subset=['LapTime'])
                if not valid_laps.empty:
                    best_lap = valid_laps['LapTime'].min()
                    if pd.notna(best_lap):
                        row['Best_Lap_Time_Seconds'] = best_lap.total_seconds()
                    
                    # Average lap time (excluding outliers)
                    lap_times_seconds = valid_laps['LapTime'].dt.total_seconds()
                    # Remove outliers (laps > 2 minutes, likely slow laps or pit stops)
                    normal_laps = lap_times_seconds[lap_times_seconds < 120]
                    if len(normal_laps) > 0:
                        row['Average_Lap_Time_Seconds'] = normal_laps.mean()
        
        results_list.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results_list)
    
    # Sort by finish position (nulls last)
    df = df.sort_values('Finish_Position', na_position='last')
    
    # Save to CSV
    print(f"Saving results to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # Display summary
    print("\n=== 2024 Hungarian GP Results Summary ===")
    print(f"Total drivers processed: {len(df)}")
    print(f"Drivers who finished: {len(df[df['Status'] == 'Finished'])}")
    print(f"DNF/DNS drivers: {len(df[df['Status'] == 'DNF'])}")
    
    # Show top 10 finishers
    finishers = df[df['Finish_Position'].notna()].head(10)
    if not finishers.empty:
        print("\nTop 10 Finishers:")
        for _, row in finishers.iterrows():
            total_time = row['Total_Time_Seconds']
            if pd.notna(total_time):
                minutes = int(total_time // 60)
                seconds = total_time % 60
                time_str = f"{minutes}:{seconds:.3f}"
            else:
                time_str = "No time"
            print(f"P{int(row['Finish_Position'])}: {row['Driver_Code']} ({row['Team_Code']}) - {time_str}")
    
    return df

# Alternative function for more detailed lap analysis
def get_detailed_lap_analysis(output_path="data/manual/2024_Hungarian_GP_detailed_analysis.csv"):
    """
    Get detailed lap-by-lap analysis similar to your sector times example.
    """
    print("Loading 2024 Hungarian GP session for detailed analysis...")
    session_2024 = fastf1.get_session(2024, 'Hungary', "R")
    session_2024.load()
    
    laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    laps_2024.dropna(inplace=True)
    
    # Convert lap and sector times to seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps_2024[f"{col}_seconds"] = laps_2024[col].dt.total_seconds()
    
    # Filter for drivers in our mapping
    laps_2024 = laps_2024[laps_2024['Driver'].isin(driver_to_team.keys())]
    
    # Aggregate sector times by driver
    sector_times_2024 = laps_2024.groupby("Driver").agg({
        "LapTime_seconds": ["mean", "min", "std"],
        "Sector1Time_seconds": "mean",
        "Sector2Time_seconds": "mean",
        "Sector3Time_seconds": "mean"
    }).reset_index()
    
    # Flatten column names
    sector_times_2024.columns = [
        'Driver_Code', 'Average_Lap_Time_Seconds', 'Best_Lap_Time_Seconds', 
        'Lap_Time_Std', 'Average_Sector1_Seconds', 'Average_Sector2_Seconds', 
        'Average_Sector3_Seconds'
    ]
    
    # Add team mapping
    sector_times_2024['Team_Code'] = sector_times_2024['Driver_Code'].map(driver_to_team)
    
    # Calculate total sector time
    sector_times_2024["Total_Sector_Time_Seconds"] = (
        sector_times_2024["Average_Sector1_Seconds"] +
        sector_times_2024["Average_Sector2_Seconds"] +
        sector_times_2024["Average_Sector3_Seconds"]
    )
    
    # Save detailed analysis
    sector_times_2024.to_csv(output_path, index=False)
    
    return sector_times_2024

if __name__ == "__main__":
    # Run the main extraction
    race_data = get_2024_hungarian_gp_data()
    
    # Run detailed analysis
    detailed_data = get_detailed_lap_analysis()
    
    print("\nExtraction complete! Check the generated CSV files.")
    print("\nFirst few rows of race data:")
    print(race_data.head())