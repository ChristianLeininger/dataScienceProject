import pymc3 as pm
import pandas as pd
import numpy as np
import theano.tensor as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from theano import shared
import theano
from utils import cost_matrix, create_shared_data

df = pd.read_csv("../data/dmc2001_learn.txt", sep=";")
df = df.fillna(df.mean(numeric_only=True))
# Preprocessing: let's assume all your columns are numerical for simplicity
df['WO'] = df['WO'].replace({'W': 1, 'O': 0, 'F': 2})
train, valid = train_test_split(df, test_size=0.2)

df_test_data = pd.read_csv("../data/dmc2001_class.txt", sep=";")
df_test_data = df_test_data.fillna(df_test_data.mean(numeric_only=True))
# Preprocessing: let's assume all your columns are numerical for simplicity
df_test_data['WO'] = df_test_data['WO'].replace({'W': 1, 'O': 0, 'F': 2})
df_test_labels = pd.read_csv("../data/dmc2001_resp.txt", sep=";")


train, valid = train_test_split(df, test_size=0.2)
print(f"train {train.shape}, valid {valid.shape}, test {df_test_data.shape} ")

shared_train_data = create_shared_data(train.drop('AKTIV', axis=1))
shared_valid_data = create_shared_data(valid.drop('AKTIV', axis=1))
shared_test_data = create_shared_data(df_test_data)

with pm.Model() as model:
    # Priors for each feature
    priors = {col: pm.Normal(col, mu=0, sd=1) for col in train.columns if col != "AKTIV"}

    # Expected value using logistic function
    mu = pm.math.sigmoid(sum([priors[col] * shared_train_data[col] for col in train.columns if col != "AKTIV"]))

    # Likelihood
    AKTIV = pm.Bernoulli('AKTIV', p=mu, observed=train['AKTIV'])

    # Sample
    num = 2000
    trace = pm.sample(num, tune=int(num/2))

# Switch to test data and generate posterior predictive samples
for column in valid.columns:
    if column in shared_valid_data:
        shared_train_data[column].set_value(valid[column].values)

with model:
    ppc_valid = pm.sample_posterior_predictive(trace, samples=500)


prediction =  np.mean(ppc_valid["AKTIV"], axis=0)
pred = [1 if x > 0.5 else 0 for x in prediction] 
lables_valid = valid["AKTIV"].to_list()
correct_predictions = sum(p == l for p, l in zip(pred, lables_valid))
accuracy = correct_predictions / len(lables_valid) * 100
print(f"Validation accuracy {accuracy}")
total_cost, true_positive, false_positive, true_negative, false_negative = cost_matrix(pred, lables_valid)
print(f"total_cost {total_cost} true_positive {true_positive} false_positive {false_positive} true_negative {true_negative} false_negative {false_negative}")

for column in df_test_data.columns:
    if column in shared_test_data:
        shared_train_data[column].set_value(df_test_data[column].values)

with model:
    ppc_test = pm.sample_posterior_predictive(trace, samples=500)



prediction =  np.mean(ppc_test["AKTIV"], axis=0)
pred = [1 if x > 0.5 else 0 for x in prediction]
import pdb; pdb.set_trace()
lables_test = df_test_labels["AKTIV"].to_list()
correct_predictions = sum(p == l for p, l in zip(pred, lables_test))
accuracy = correct_predictions / len(lables_test) * 100
print(f"Test accuracy {accuracy}")

total_cost, true_positive, false_positive, true_negative, false_negative = cost_matrix(pred, lables_test)
print(f"total_cost {total_cost} true_positive {true_positive} false_positive {false_positive} true_negative {true_negative} false_negative {false_negative}")