# copyright 2023
# author: Christian Leininger
# Email: leiningc@tf.uni-freiburg.de
# date: 06.07.2023

import pandas as pd
import logging


from utils import clean_data

def main():
    """ Main entry point of the app """
    # Specify the file path
    file_path = "../data/dmc2001_class.txt"
    file_path_learn = "../data/dmc2001_learn.txt"
    # Read the text file using pandas
    df_classes = pd.read_csv(file_path, delimiter=";")
    df_learn = pd.read_csv(file_path_learn, delimiter=";")


    
    df_learn_drp =  clean_data(df_learn, policy="drop")
    df_learn_fill =  clean_data(df_learn, policy="fill")
    df_learn_mean =  clean_data(df_learn, policy="mean")
    logging.debug(f"df_learn_drp {df_learn_drp.shape}")
    logging.debug(f"df_learn_fill {df_learn_fill.shape}")
    logging.debug(f"df_learn_mean {df_learn_mean.shape}")
    import pdb; pdb.set_trace()




if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)s: %(message)s")
    main()    