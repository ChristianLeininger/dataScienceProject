# copyright 2023
# author: Christian Leininger
# Email: leiningc@tf.uni-freiburg.de
# date: 06.07.2023

import os
import pandas as pd
import numpy as np
import logging
import plotly.figure_factory as ff
import plotly.io as pio
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



IMAGES_PATH = os.path.join(os.getcwd())

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """ Save the figure to
    Args:
        fig_id: name of the figure
        tight_layout: bool
        fig_extension: str
        resolution: int
    """
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)





def clean_data(df: pd.DataFrame, policy="drop") -> pd.DataFrame:
    """ Clean the data
    Args:
        df: data frame to clean
    Returns:
        cleaned data frame
    """
    # replace string with int
    # import pdb; pdb.set_trace()
    
    df['WO'] = df['WO'].replace({'W': 1, 'O': 0, 'F': 2})
    
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
        df = df.fillna(df.mean(numeric_only=True))
    else:
        raise NotImplementedError("Policy {} not implemented".format(policy))
    
    assert df.isnull().sum().sum() == 0
    return df
    


def check_balance(labels: np.ndarray) -> bool:
    """" test if the data is balanced 
    Args:
        labels: np.ndarray
    Returns: if the data is balanced return True else False
    """
    unique, counts = np.unique(labels, return_counts=True)
    average = (counts[0] + counts[1]) / 2
    logging.debug(f"unique {unique} counts {counts}")
    import pdb; pdb.set_trace()
    

    return True


def analize_data(df: pd.DataFrame, label_class: str, data_name: str) -> None:
    """ Analize the data
    Args:
        df: data frame to analize
    """
    logging.debug(f" Analize the data {df.shape} with label {label_class}")
    negative_classes = (df[label_class] == 0).sum()
    positiv_classes = (df[label_class] == 1).sum()
    # 
    assert negative_classes + positiv_classes == df.shape[0]
    neg_per = negative_classes  / (df.shape[0] / 100)
    pos_per =  positiv_classes / (df.shape[0] / 100)
    logging.debug(f"Negative classes {negative_classes} : {neg_per:.2f} %  positive classes {positiv_classes} : {pos_per:.2f} ")
    
    corr_matrix = df.corr()
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.round(2).values,
        showscale=True)
    pio.write_image(fig, f'{data_name}_correlation_matrix.png', width=1600, height=1200)
    # create histogram to see the distribution of the data
    # find possible outliers 
    df.hist(bins=50, figsize=(20, 15))
    save_fig(f"data_{data_name}_attribute_histogram_plots")
    # plt.show()
    # import pdb; pdb.set_trace()


def split_data(df: pd.DataFrame, label_class: str, test_size: float = 0.2, valid_size: float = 0.2, seed: int = 42):
    """ Split the data in train, valid and test
    Args:
        df: data frame to split
    Returns:
        train and test data frame
    """
    data = df.drop(label_class, axis=1).values
    labels = df[label_class].values
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size + valid_size, random_state=seed)
    logging.debug(f"train_data {train_data.shape} test_data {test_data.shape} train_labels {train_labels.shape} test_labels {test_labels.shape}")
    first_point = [f'{num:.0f}' if num.is_integer() else f'{num:.2f}'   for num in train_data[0]]
    logging.debug(f"first data {first_point} first label {train_labels[0]}")
    valid_data, test_data, valid_labels, test_labels = train_test_split(test_data, test_labels, test_size=valid_size / (test_size + valid_size), random_state=seed)
    #import pdb; pdb.set_trace()
    check_balance(train_labels)
    check_balance(valid_labels)
    check_balance(test_labels)
    return train_data, valid_data, test_data, train_labels, valid_labels, test_labels

def seed_expriement(seed: int = 42):
    """ Seed the expriement
    Args:
        seed: int
    """
    np.random.seed(seed)


def train_model(model, train_data, valid_data):
    """ Train the model with the data in the train data frame
        evaluate the model with the valid data frame

    Args:
    """