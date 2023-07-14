import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

def create_shared_data(data):
    import theano
    shared_data = {}
    for column in data.columns:
        shared_data[column] = theano.shared(data[column].values)
    return shared_data


def cost_matrix(prediction, labels):
    true_positive = 1.100
    false_positive = -265
    true_negative = -25
    false_negative = 662
    total_cost = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for pred, label in zip(prediction, labels):
        if pred == 1 and label == 1:
            total_cost += true_positive
            true_positive += 1
        elif pred == 1 and label == 0:
            total_cost += false_positive
            false_positive += 1
        elif pred == 0 and label == 0:
            total_cost += true_negative
            true_negative += 1
        elif pred == 0 and label == 1:
            total_cost += false_negative
            false_negative += 1
        else:
            raise ValueError(f"pred {pred} label {label}")
    
    print(f"true_positive {true_positive} false_positive {false_positive} true_negative {true_negative} false_negative {false_negative}")
    return total_cost, true_positive, false_positive, true_negative, false_negative



def create_data(path="../data/dmc2001_learn.txt", features=['jahrstart', 'Altersgr', 'Abogr', 'AKTIV'], test=False,  valid_size=0.2, debug=False):
    df = pd.read_csv(path, sep=";")
    
    if test:
        df = pd.read_csv(path + "dmc2001_class.txt", sep=";")
        df2 = pd.read_csv(path + "dmc2001_class.txt", sep=";")
    if debug:
        df = df.head(100)
    train = df.fillna(df.mean(numeric_only=True))
    train['WO'] = train['WO'].replace({'W': 1, 'O': 0, 'F': 2})
    import pdb; pdb.set_trace()
    # features = ['jahrstart', 'WO', 'Regiotyp', 'Kaufkraft', 'Bebautyp', 'Altersgr', 'Abogr', 'AKTIV']
    train = train[features]
    train, valid = train_test_split(train, test_size=valid_size)
    train_y = train['AKTIV']
    train = train.drop('AKTIV', axis=1)
    valid_y = valid['AKTIV']
    valid = valid.drop('AKTIV', axis=1)
    return train, train_y, valid, valid_y


def eval_model(ppc, labels, name):
    prediction =  np.mean(ppc["AKTIV"], axis=0)
    pred = [1 if x > 0.5 else 0 for x in prediction] 
    correct_predictions = sum(p == l for p, l in zip(pred, labels))
    accuracy = correct_predictions / len(labels) * 100
    logging.info(f"{name} accuracy {accuracy}")
    total_cost, true_positive, false_positive, true_negative, false_negative = cost_matrix(pred, labels)
    logging.info(f"{name} total_cost {total_cost} true_positive {true_positive} false_positive {false_positive} true_negative {true_negative} false_negative {false_negative}")
    return accuracy, total_cost, true_positive, false_positive, true_negative, false_negative