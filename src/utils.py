import os
import sys
import pickle
import numpy as np
import pandas as pd

def save_object(file_path, obj):
    """
    Saves a Python object as a pickle file.
    Args:
        file_path (str): Path where the pickle file will be saved.
        obj: Python object to save.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise Exception(f"Error saving object: {e}")
    
