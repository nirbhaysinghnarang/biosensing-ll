import argparse
import collections
import logging
import os
import pickle
import shutil
import traceback as tb
from glob import iglob
import msgpack
import numpy as np
import pandas as pd
import json

class Serialized_Dict(object):
    __slots__ = ['_ser_data', '_data']
    cache_len = 100
    _cache_ref = [None] * cache_len
    MSGPACK_EXT_CODE = 13

    def __init__(self, python_dict=None, msgpack_bytes=None):
        if type(python_dict) is dict:
            self._ser_data = msgpack.packb(python_dict, use_bin_type=True)
        elif type(msgpack_bytes) is bytes:
            self._ser_data = msgpack_bytes
        else:
            raise ValueError("Neither dict nor payload is supplied.")
        self._data = None

    def _deser(self):
        if not self._data:
            self._data = msgpack.unpackb(self._ser_data, raw=False, use_list=False)
            self._cache_ref.pop(0)
            self._cache_ref.append(self)

    def __getitem__(self, key):
        self._deser()
        return self._data[key]

    def items(self):
        self._deser()
        return self._data.items()

def read_gaze_data(folder_location):
    pupil_data = os.path.join(folder_location, 'gaze.pldata')
    timestamps_path = os.path.join(folder_location, 'gaze_timestamps.npy')
    metadata_path = os.path.join(folder_location, "info.player.json")
    
    # Load metadata to calculate offset
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    SYSTEM_START_TIME = metadata["start_time_system_s"]
    PUPIL_LABS_START_TIME = metadata["start_time_synced_s"]
    OFFSET = SYSTEM_START_TIME - PUPIL_LABS_START_TIME
    print(f"Offset calculated: {OFFSET}")

    # Load gaze data
    gaze_data = []
    with open(pupil_data, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False, use_list=False)
        for _, payload in unpacker:
            unpacked = Serialized_Dict(msgpack_bytes=payload)
            gaze_data.append(unpacked)
    
    # Load timestamps and adjust with OFFSET
    gaze_timestamps = np.load(timestamps_path) + OFFSET
    print(f"First few timestamps after offset: {gaze_timestamps[:5]}")
    
    return gaze_data, gaze_timestamps

def flatten_dict_column(df, col_name):
    # Flatten the dictionary in a specific column, prefixing the column name to the keys
    col_as_df = pd.json_normalize(df[col_name])
    col_as_df.columns = [f"{col_name}_{subcol}" for subcol in col_as_df.columns]
    
    # Drop the original column and join the new flattened columns
    df = df.drop(col_name, axis=1).join(col_as_df)
    return df

def gaze_data_to_dataframe(gaze_data, gaze_timestamps):
    gaze_list = []
    for gaze_dict, timestamp in zip(gaze_data, gaze_timestamps):
        gaze_dict_items = dict(gaze_dict.items())
        gaze_dict_items['timestamp'] = timestamp
        gaze_list.append(gaze_dict_items)

    # Create pandas DataFrame
    gaze_df = pd.DataFrame(gaze_list)
    
    # Flatten the columns that contain dictionaries (like eye_centers_3d and gaze_normals_3d)
    if 'eye_centers_3d' in gaze_df.columns:
        gaze_df = flatten_dict_column(gaze_df, 'eye_centers_3d')
    if 'gaze_normals_3d' in gaze_df.columns:
        gaze_df = flatten_dict_column(gaze_df, 'gaze_normals_3d')
    
    # Move the 'timestamp' column to be the first column
    cols = ['timestamp'] + [col for col in gaze_df if col != 'timestamp']
    gaze_df = gaze_df[cols]

    return gaze_df

def main(gaze_path, output_csv_path):
    # Read and process the gaze data
    gaze_data, gaze_timestamps = read_gaze_data(gaze_path)
    gaze_df = gaze_data_to_dataframe(gaze_data, gaze_timestamps)
    print(gaze_df.columns)
    print(gaze_df[['timestamp', 'norm_pos', 'confidence']])
    
    # Save the DataFrame to CSV with corrected timestamps as the first column
    gaze_df.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process gaze data and output to CSV.")
    parser.add_argument('--gaze', type=str, required=True, help='Path to the gaze data folder.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the output CSV file.')
    
    args = parser.parse_args()
    main(args.gaze, args.output_csv)

