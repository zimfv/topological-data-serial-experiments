import numpy as np
import pandas as pd


def increment_dimension(df, dim=None):
    # Add giant cycle with 1 higher dimension, but not higher given dim
    if dim is not None:
        if df['Dimension'].max() >= dim:
            return df
    df_incremented = df[df['Death'] == np.inf]
    df_incremented = df_incremented.groupby('Space', as_index=False).max()
    df_incremented['Dimension'] += 1
    return pd.concat([df, df_incremented])


def calculate_mean_betti(df: pd.DataFrame, array, dim=None):
    """
    Calculate mean betti numbers
    
    Parameters:
    -----------
    df : DataFrame with columns ['Birth', 'Death', 'Dimension', 'Space']
    
    array : list or np.array
        Array of values
        
    dim : int or None
        Dimension of topological space
        If that's None< that will be maximal dimension from dfs['Dimension']
    
    Returns:
    --------
    res : np.array shape (len(array), dim)
        Array of 
    """
    spaces = len(np.unique(df['Space'].values))
    if dim is None:
        dim = df['Dimension'].max()
    res = np.zeros([len(array), dim+1])
    for i in range(len(array)):
        val = array[i]
        dfi = df[(df['Birth'] <= val)&(df['Death'] > val)]
        dfi = dfi.groupby(['Space', 'Dimension'], as_index=False).count()[['Space', 'Dimension', 'Birth']]
        dfi = dfi.groupby('Dimension', as_index=False).sum()
        index = dfi['Dimension'].values
        vals = dfi['Birth'].values / spaces
        res[i, index] = vals
    return res


def calculate_mean_EC(df: pd.DataFrame, array):
    """
    Calculate mean betty numbers
    
    Parameters:
    -----------
    df : DataFrame with columns ['Birth', 'Death', 'Dimension', 'Space']
    
    array : list or np.array
        Array of values
        
    
    Returns:
    --------
    res : np.array length len(array)
        Array of 
    """
    spaces = len(np.unique(df['Space'].values))
    res = np.zeros(len(array))
    for i in range(len(array)):
        val = array[i]
        dfi = df[(df['Birth'] <= val)&(df['Death'] > val)][['Space', 'Dimension', 'Birth']]
        dfi = dfi.groupby(['Space', 'Dimension'], as_index=False).count()
        dfi.columns = ['Space', 'Dimension', 'Count']
        dfi['Summand EC'] = (-1)**dfi['Dimension'] * dfi['Count']
        dfi = dfi.groupby('Space', as_index=False).sum()['Summand EC']
        res[i] = dfi.sum() / spaces
    res[np.isnan(res)] = 0
    return res


def catch_zeros(x, y):
    # returns all zero arguments for graph y(x)
    ids = np.where(y[1:]*y[:-1] <= 0)[0]
    zeros = []
    for i in ids:
        x0, x1 = x[i], x[i+1]
        y0, y1 = abs(y[i]), abs(y[i+1])
        xz = x0 + (x1 - x0)*y0/(y0 + y1)
        zeros.append(xz)
    return np.array(zeros)