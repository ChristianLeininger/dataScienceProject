# copyright 2023
# author: Christian Leininger
# Email: leiningc@tf.uni-freiburg.de
# date: 06.07.2023

import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import clean_data, analize_data, split_data, seed_expriement, create_scatter_plot
from utils import baseline, bayesianModel

def main():
    """ Main entry point of the app """
    # Specify the file path
    seed=0
    seed_expriement(seed=seed)
    logging.debug(f"np.random.randint(0, 100) {np.random.randint(0, 100)}")
   
    file_path = "../data/dmc2001_class.txt"
    file_path_learn = "../data/dmc2001_learn.txt"
    # Read the text file using pandas
    # df_classes = pd.read_csv(file_path, delimiter=";")
    df_learn = pd.read_csv(file_path_learn, delimiter=";")
    
    # df_learn_drp =  clean_data(df_learn, policy="drop")
    # df_learn_fill =  clean_data(df_learn, policy="fill")
    df_learn_mean =  clean_data(df_learn, policy="mean")
    logging.debug(f"df_learn {df_learn.shape}")
    logging.debug(f"df_learn {df_learn.columns}")
    
    bayesianModel(df_learn_mean, df_learn_mean, df_learn_mean)
    corr_matrix = df_learn_mean.corr()
    logging.debug(f"corr_matrix {corr_matrix}")
    corr_with_target = corr_matrix['AKTIV'].sort_values(ascending=False)
    # Features auswählen, die eine hohe Korrelation mit der Zielvariable haben
    threathold = 0.05
    relevant_features = corr_with_target[corr_with_target.abs() > threathold].index.tolist()
    relevant_features.remove('AKTIV')
    logging.debug(f"relevant_features {relevant_features}")
    baseline(df_learn_mean, relevant_features=relevant_features)
    # Entfernen Sie die Zielvariable selbst aus der Liste
    corr_matrix_relevant = df_learn_mean[relevant_features].corr()
    sns.heatmap(corr_matrix_relevant, annot=True)
    plt.show()
    import pdb; pdb.set_trace()
    # logging.debug(f"df_learn_drp {df_learn_drp.shape}")
    # logging.debug(f"df_learn_fill {df_learn_fill.shape}")
    logging.debug(f"df_learn_mean {df_learn_mean.shape}")
    # analize_data(df_learn_drp, "AKTIV")
    # analize_data(df = df_learn_fill, label_class="AKTIV", data_name= "flilledWitZero")
    create_scatter_plot(df_learn_mean, "AKTIV", "flilledWitMean")
    analize_data(df = df_learn_mean, label_class="AKTIV", data_name= "flilledWitMean")
    # analize_data(df = df_learn_drp, label_class="AKTIV", data_name= "drop")
    
    train_data, valid_data, test_data, train_labels, valid_labels, test_labels = split_data(df_learn_mean, label_class="AKTIV", test_size=0.2, valid_size=0.2, seed=seed)
    logging.debug(f"train_data {train_data.shape}")    




if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)s: %(message)s")
    main()    