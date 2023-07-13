import theano


def create_shared_data(data):
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