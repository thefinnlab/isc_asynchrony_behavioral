import os, glob
import pandas as pd
import pickle
import numpy as np

def save_dict(fn, d):
    '''
    Save a pkl file.
    '''
    with open(fn, 'wb') as f:
        pickle.dump(d, f)
        
        
def load_dict(fn):
    '''
    Load a pkl file.
    '''
    
    with open(fn, 'rb') as f:
        d = pickle.load(f)
    
    return d

def combine_dicts(fns, delete=False):
    '''
    Create a stack of dictionaries from a list of dictionary 
    files. Optionally the files once loaded.
    
    Inputs:
        - fns: List of strings to pickle files.
        - delete: Bool. Default is False. Delete files
            once loaded. Typically used for temporary
            files.
    '''
    
    ds = {}
    
    for fn in fns:

        d = load_dict(fn)
        
        if delete:
            os.remove(fn)
           
        if not ds:
            ds = {key: d[key][np.newaxis,:] for key in d.keys()}
        else:
            ds = {key: np.concatenate((ds[key], d[key][np.newaxis,:])) for key in d.keys()}
            
    return ds

def read_text(filename): 
    
    '''
    
    '''
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line for line in f.readlines()]
    
    return lines

def load_csv(path, filename, columns=None):
    '''
    Load a CSV file as a dataframe from a given path. 
    
    Parameters
    ----------
    path : str
        Path at to search for the given file identifier to load.
    fname : str
        The filename of partial identifier of the file to load.

    Returns
    -------
    df : pandas DataFrame
        The CSV file loaded as a pandas DataFrame.
    '''
    
    #find the first file with fname and .csv extension
    file = sorted(glob.glob(os.path.join(path, f'{filename}*.csv')))[0]
    
    #load as dataframe
    return pd.read_csv(file, names=columns)