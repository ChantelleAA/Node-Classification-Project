import os
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import shutil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tabulate_events(dpath):
    """
    Extracts and tabulates scalar events from TensorBoard log files located within a directory.
    
    Args:
        dpath (str): Path to the directory containing event files.
    
    Returns:
        dict: A dictionary where each key is a scalar tag and each value is a list of these scalar values.
        list: A list of step indices corresponding to the scalar values.
    """
    # Load all event files from the directory
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() 
                         for dname in os.listdir(dpath) 
                         if dname.startswith('events')]
    assert len(summary_iterators) == 1, "Directory should contain exactly one events file."
    
    # Aggregate all scalar tags
    tags = set(*[si.Tags()['scalars'] for si in summary_iterators])
    
    # Prepare a dictionary to hold the scalar events
    out = defaultdict(list)
    steps = []

    # Iterate over each tag and accumulate their scalar values
    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1, "Mismatch in steps across event files."
            out[tag].append([e.value for e in events])
            
    return out, steps

def to_csv(dpath):
    """
    Converts TensorBoard scalar data to a CSV file.
    
    Args:
        dpath (str): Path to the directory containing the TensorBoard logs.
    """
    dirs = os.listdir(dpath)

    # Extract events and their corresponding steps
    d, steps = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values = np.array(values)
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame({f"{tag}": np_values[i][:, 0] for i, tag in enumerate(tags)}, 
                      index=steps, 
                      columns=tags)
    df.to_csv(os.path.join(dpath, "logger.csv"))

def read_event(path):
    """
    Reads TensorBoard events from a directory and returns them as a pandas DataFrame.
    
    Args:
        path (str): Path to the directory containing the TensorBoard logs.
    
    Returns:
        pd.DataFrame: DataFrame containing the logged scalar data.
    """
    to_csv(path)
    return pd.read_csv(os.path.join(path, "logger.csv"), index_col=0)

def empty_dir(folder):
    """
    Empties all contents of a specified directory.
    
    Args:
        folder (str): Path to the directory to be emptied.
    """
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def load_mapping(filename='mapping.json'):
    """
    Loads a JSON mapping file.
    
    Args:
        filename (str): Filename of the JSON file to load.
    
    Returns:
        dict: A dictionary loaded from the JSON file.
    """
    with open(filename, 'r') as file:
        return json.load(file)
