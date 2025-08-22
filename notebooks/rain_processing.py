import pandas as pd
import numpy as np

def process_ratings_csv(input_file, output_file=None):
    """
    Process CSV file to calculate GameRating, DeltaRating, and RainRating columns.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file (optional, defaults to input_file if not provided)
    """
    
    # Read the CSV file with encoding detection
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    df = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            print(f"Successfully read file with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError("Could not read the file with any of the tried encodings")
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    print("Available columns:", df.columns.tolist())
    
    # Convert all numeric columns to float, handling both positive and negative values
    numeric_columns = ['Experience', 'Racecraft', 'Awareness', 'Pace', 'GameRating', 
                      'M22', 'S22', 'J22', 'D23', 'B23', 'M23', 'Br24', 'C24', 'A25', 'B25', 'Br25']
    
    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 1) Calculate GameRating
    df['GameRating'] = (
        df['Awareness'] * 0.35 + 
        df['Experience'] * 0.30 + 
        df['Racecraft'] * 0.20 + 
        df['Pace'] * 0.15
    )
    
    # 2) Calculate DeltaRating
    # Get columns after GameRating (race result columns)
    race_columns = ['M22', 'S22', 'J22', 'D23', 'B23', 'M23', 'Br24', 'C24', 'A25', 'B25', 'Br25']
    
    # Calculate DeltaRating for each row
    delta_ratings = []
    
    for index, row in df.iterrows():
        # Get values from race columns, excluding nulls
        race_values = []
        for col in race_columns:
            if col in df.columns and pd.notna(row[col]):
                # More negative is better, so we use the actual value (not absolute)
                race_values.append(row[col])
        
        # Calculate average if there are non-null values
        if race_values:
            avg_delta = np.mean(race_values)
        else:
            avg_delta = 0
        
        delta_ratings.append(avg_delta)
    
    df['DeltaRating_raw'] = delta_ratings
    
    # Normalize DeltaRating to score out of 100 (more negative = better = closer to 100)
    min_delta = df['DeltaRating_raw'].min()  # Most negative (best performance)
    max_delta = df['DeltaRating_raw'].max()  # Most positive (worst performance)
    
    if min_delta != max_delta:
        # Scale so that most negative gets 100, most positive gets 0
        df['DeltaRating'] = ((max_delta - df['DeltaRating_raw']) / (max_delta - min_delta)) * 100
    else:
        df['DeltaRating'] = 50  # If all values are the same, give middle score
    
    # Remove the raw DeltaRating column
    df = df.drop('DeltaRating_raw', axis=1)
    
    # 3) Calculate RainRating
    df['RainRating'] = (df['GameRating'] * 0.6 + df['DeltaRating'] * 0.4)
    
    # Save the updated CSV
    if output_file is None:
        output_file = input_file
    
    df.to_csv(output_file, index=False)
    
    print(f"CSV processing completed successfully!")
    print(f"Updated file saved as: {output_file}")
    print(f"Total rows processed: {len(df)}")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"GameRating - Min: {df['GameRating'].min():.2f}, Max: {df['GameRating'].max():.2f}, Avg: {df['GameRating'].mean():.2f}")
    print(f"DeltaRating - Min: {df['DeltaRating'].min():.2f}, Max: {df['DeltaRating'].max():.2f}, Avg: {df['DeltaRating'].mean():.2f}")
    print(f"RainRating - Min: {df['RainRating'].min():.2f}, Max: {df['RainRating'].max():.2f}, Avg: {df['RainRating'].mean():.2f}")
    
    return df

# Usage example:
if __name__ == "__main__":
    # Replace 'your_file.csv' with your actual CSV file path
    process_ratings_csv('data/manual/rain_driver_index.csv')
    
    # Or to save to a different file:
    # process_ratings_csv('input_file.csv', 'output_file.csv')