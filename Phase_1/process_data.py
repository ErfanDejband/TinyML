import argparse
import json
import os
from typing import Union
import numpy as np
import pandas as pd

def process_data(row_data_dir: str, mode: str) -> Union[pd.DataFrame, np.ndarray, None]:
    """
    Processes raw JSON data from a directory.

    Based on the mode, it can:
    - 'save': Save the processed data as a CSV file.
    - 'csv': Return a pandas DataFrame.
    - 'numpy': Return a NumPy array.

    Args:
        row_data_dir (str): Directory containing the raw JSON data.
        mode (str): Output mode. One of ['save', 'csv', 'numpy'].

    Returns:
        Union[pd.DataFrame, np.ndarray, None]: A DataFrame, NumPy array, or None,
        depending on the mode.
    """
    all_data = []
    labels = []
    output_dir = os.path.join(os.path.dirname(row_data_dir), 'RowCSVData')

    if mode == 'save':
        os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(row_data_dir):
        if filename.endswith('.json'):
            label = os.path.splitext(filename)[0]
            filepath = os.path.join(row_data_dir, filename)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                interval_ms = data['payload']['interval_ms']
                time_increment_s = interval_ms / 1000.0
                
                values = data['payload']['values']
                # float with 2 decimal places
                values = [[round(float(v), 2) for v in value_set] for value_set in values]
                
                current_time = 0
                for value_set in values:
                    all_data.append([round(current_time, 2)] + value_set)
                    labels.append(label)
                    current_time += time_increment_s

    df = pd.DataFrame(all_data, columns=['time', 'x', 'y', 'z'])
    df['label'] = labels

    if mode == 'save':
        output_path = os.path.join(output_dir, 'processed_data.csv')
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        return None
    elif mode == 'csv':
        return df
    elif mode == 'numpy':
        return df.to_numpy()
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose from 'save', 'csv', or 'numpy'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process TinyML raw data.')
    parser.add_argument(
        '--datadir', 
        type=str, 
        required=True, 
        help='Directory of the raw JSON data.'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['save', 'csv', 'numpy'], 
        help='Output mode: save the CSV, return a DataFrame, or return a NumPy array.'
    )
    
    args = parser.parse_args()
    
    result = process_data(args.datadir, args.mode)
    
    if result is not None:
        print(f"Returning data as {args.mode}:")
        print(result)
