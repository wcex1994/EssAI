#=================================================================
# Load Libraries
#=================================================================

import pandas as pd
import numpy as np

#==================================================================================
# Helpers
#==================================================================================

# Credit to https://stackoverflow.com/questions/19894939/calculate-arbitrary-percentile-on-pandas-groupby
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

#==================================================================================
# Constants
#==================================================================================

SUMMARY_STATS = ["mean", "min", "max", percentile(5), percentile(95)]
METRICS = ["sentenceLengthByChar", "shannonEntropy"]

#==================================================================================
# User Input -> D3 Compatible Format
#==================================================================================

def user_metrics_to_longform(df, metrics):
    
    return  pd.melt(df, 
                 id_vars=['Author'], 
                 value_vars= metrics,
                 var_name='Metric', 
                 value_name='value')

def trim_columns(col):
    colname = ''.join(col)
    return colname.replace("value", "", )

def format_columns(df):
    df.columns = list(map(trim_columns, df.columns.values))
    return df

def statistics_to_longform(df):
    return pd.melt(df, 
                   id_vars=['Author', "Metric"],
                   var_name = "Statistic",
                   value_name = "Value")


def input_to_stats(df, metrics):

    return df \
        .pipe(user_metrics_to_longform, metrics = metrics) \
        .groupby(["Author", "Metric"]) \
        .agg({"value" : SUMMARY_STATS}) \
        .reset_index() \
        .pipe(format_columns) \
        .pipe(statistics_to_longform) \
        .reset_index(drop = True)



