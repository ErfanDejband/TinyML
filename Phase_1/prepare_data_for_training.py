from typing import Tuple
import numpy as np
import pandas as pd
from process_data import process_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def prepare_data(
    data_directory: str,
    window_size_s: float = 1.0,
    split_ratio: int = 80,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads data, creates windows, encodes labels, and splits into training/validation sets.

    Args:
        data_directory (str): Directory containing the raw JSON data.
        window_size_s (float): The desired window size in seconds.
        split_ratio (int): The percentage of data for training.
        shuffle (bool): Whether to shuffle the data before splitting.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing
        X_train, X_test, y_train, y_test (with labels encoded).
    """
    # Get the DataFrame
    df = process_data(row_data_dir=data_directory, mode='csv')

    # Determine the number of timesteps per window
    if len(df['time']) < 2:
        raise ValueError("Not enough data to determine sampling interval.")
    
    # Calculate interval, handling potential floating point inaccuracies
    interval = df['time'].diff().median()
    if pd.isna(interval) or interval <= 0:
        raise ValueError("Could not determine a valid sampling interval.")

    timesteps_per_window = int(window_size_s / interval)
    actual_window_s = timesteps_per_window * interval
    print(f"Requested window: {window_size_s}s. Using {timesteps_per_window} timesteps, creating windows of ~{actual_window_s:.2f}s and =actual_window_s.")

    windows = []
    window_labels = []

    # Create windows for each gesture recording
    for label, group_df in df.groupby('label'):
        features = group_df[['x', 'y', 'z']].values
        
        # Using a sliding window with 50% overlap for data augmentation
        step = timesteps_per_window // 2 
        for i in range(0, len(features) - timesteps_per_window + 1, step):
            window = features[i : i + timesteps_per_window]
            windows.append(window)
            window_labels.append(label)

    if not windows:
        raise ValueError(f"Could not create any windows of size {timesteps_per_window}. "
                         f"The longest continuous recording might be too short.")

    X_windows = np.array(windows)
    y_windows = np.array(window_labels)

    # Encode string labels to integers
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_windows)
    print("Labels encoded:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))


    # Calculate the test size from the split ratio
    test_size = 1.0 - (split_ratio / 100.0)

    # Split the data, ensuring label distribution is similar in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_windows, y_encoded, test_size=test_size, shuffle=shuffle, random_state=42, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Define the path to your data directory
    data_dir = 'RowData'
    
    # Get the prepared data with 1-second windows
    X_train, X_test, y_train, y_test = prepare_data(data_dir, window_size_s=1.0, split_ratio=80)

    # Print the shapes of the resulting datasets
    print("\nData prepared for training with windowing.")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)