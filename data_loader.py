import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


class DataLoader:
    def __init__(self, df, fs):  # Example: 30 Hz sampling rate
        self.df = df
        self.fs = fs
        self.n_samples = df.shape[1]

    def get_trace(self, roi_id):
        return self.df.loc[roi_id].values

    def get_times(self):
        # Return an array from 0 to (n_samples - 1) / sampling_rate
        times = np.arange(self.n_samples) / self.fs
        return times

def load_and_preprocess(file_path, file_type, transpose, truncate_samples, smoothing_window_length, poly_order):
    # Load data
    if file_type.upper() == "CSV":
        df_raw = pd.read_csv(file_path, index_col=0)
        df_raw = df_raw.reset_index(drop=True)
        if 'Err' in df_raw.columns:
            df_raw = df_raw.drop(columns=['Err'])
        data = df_raw
    else:
        npy_data = np.load(file_path)
        data = pd.DataFrame(npy_data)
        data.index = [f"ROI_{i}" for i in range(data.shape[0])]
    
    if transpose:
        data = data.T

    # Truncate data
    total_samples = data.shape[1]
    if truncate_samples >= total_samples:
        raise ValueError(f"Cannot truncate {truncate_samples} samples: data only has {total_samples} samples.")
    elif truncate_samples > 0:
        data = data.iloc[:, truncate_samples:]

    # After truncation, ensure data is valid
    if data.shape[1] < 5:
        raise ValueError(f"Too few samples ({data.shape[1]}) after truncation.")

    dff_array = data.values
    baseline = np.nanmean(dff_array, axis=1, keepdims=True)
    baseline[baseline == 0] = np.nan  # prevent division by zero
    dff_array = (dff_array - baseline) / baseline
    dff_array = np.nan_to_num(dff_array)

    # Smoothing safeguard
    if data.shape[1] < smoothing_window_length:
        print(f"[INFO] Reducing smoothing window from {smoothing_window_length} to {data.shape[1] | 1}")
        smoothing_window_length = max(3, data.shape[1] | 1)  # Make odd and safe

    if smoothing_window_length < poly_order + 2:
        raise ValueError(f"Smoothing window ({smoothing_window_length}) must be >= poly_order + 2 ({poly_order + 2})")

    smoothed_dff = savgol_filter(dff_array, smoothing_window_length, poly_order, axis=1)
    smoothed_dff_df = pd.DataFrame(smoothed_dff, index=data.index, columns=data.columns)
    return data, smoothed_dff_df
