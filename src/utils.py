# copyright 2023
# author: Christian Leininger
# Email: leiningc@tf.uni-freiburg.de
# date: 06.07.2023

import pandas as pd
import logging


def clean_data(df: pd.DataFrame, policy="drop") -> pd.DataFrame:
    """ Clean the data
    Args:
        df: data frame to clean
    Returns:
        cleaned data frame
    """
    if policy == "drop":
        logging.debug(f"Drop rows with NaN values size before {df.shape}")
        df = df.dropna()
        logging.debug(f"Drop rows with NaN values size after {df.shape}")
        
    elif policy == "fill":
        logging.debug(f"Fill rows with NaN values size before {df.shape}")
        df=  df.fillna(0)
        logging.debug(f"Fill rows with NaN values size after {df.shape}")
    elif policy == "mean":
        logging.debug(f"Drop rows with NaN values size before {df.shape}")
        df['WO'] = df['WO'].replace({'W': 1, 'O': 0})
        # import pdb; pdb.set_trace()
        df = df.fillna(df.mean(numeric_only=True))
    else:
        raise NotImplementedError("Policy {} not implemented".format(policy))
    
    assert df.isnull().sum().sum() == 0
    return df
    