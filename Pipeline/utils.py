#=================================================================
# Import Libraries
#=================================================================

import pandas as pd

#=================================================================
# Transform Utilities
#=================================================================

def mapOverRows(df, f, col):
    "Applies function f over every row of dataframe df."
    return df.apply(lambda row : f(row[col]), axis = 1) 

def genTransform(f, new_colname, from_colname):
    '''
    Returns a function that will add a
    column to a dataframe by applying function
    f over every row in column from_colname.
    '''
    def transformReplace(df):
        df[new_colname] = mapOverRows(df, f, from_colname)
        return df
    
    def transformAppendCounts(df):
        counts = mapOverRows(df, f, from_colname) 
        counts_df = pd.DataFrame(counts).fillna(0)
        return pd.concat([df, counts_df], axis= 1)
    
    def transformAppend(df):
        new_df = mapOverRows(df, f, from_colname) 
        return pd.concat([df, new_df], axis= 1)
    
    if new_colname == "Append":
        return transformAppend
    elif new_colname == "AppendCounts":
        return transformAppendCounts
    else:
        return transformReplace


def applyTransforms(df, configs):    
    '''
    Takes a list of dataframe transformations
    and appends them one at a time to the input
    dataframe, which it then returns.
    '''
    for config in configs:
        f, new_name, old_name = config
        transform = genTransform(f, new_name, old_name)
        df = transform(df)
    return df
