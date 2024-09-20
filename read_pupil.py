import pandas as pd
import os
import json


recording_location = "/Users/nirbhaysinghnarang/recordings/2024_09_18/002"


def load_pupil_data(file_path=recording_location):
    pupil_data = os.path.join(file_path, 'pupil.pldata')
    metadata_path = os.path.join(file_path, "info.player.json")

    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    SYSTEM_START_TIME = metadata["start_time_system_s"]
    PUPIL_LABS_START_TIME = metadata["start_time_synced_s"]
    OFFSET = SYSTEM_START_TIME - PUPIL_LABS_START_TIME