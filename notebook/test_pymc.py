import pymc3 as pm
import numpy as np
import logging

import pymc3 as pm
import pandas as pd
import numpy as np
import theano.tensor as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from theano import shared
from utils import cost_matrix, create_shared_data

import matplotlib.pyplot as plt



logging.basicConfig(level=logging.INFO)
df = pd.read_csv("../data/dmc2001_learn.txt", sep=";")
# import pdb; pdb.set_trace()
df = df.head(10)
train = df.fillna(df.mean(numeric_only=True))
# Preprocessing: let's assume all your columns are numerical for simplicity
train['WO'] = train['WO'].replace({'W': 1, 'O': 0, 'F': 2})
features = ['jahrstart', 'WO', 'Regiotyp', 'Kaufkraft', 'Bebautyp', 'Altersgr', 'Abogr', 'AKTIV']
features = ['jahrstart', 'Altersgr', 'Abogr', 'AKTIV']
train = train[features]
train, valid = train_test_split(train, test_size=0.2)
train_y = train['AKTIV']
train = train.drop('AKTIV', axis=1)
valid_y = valid['AKTIV']
valid = valid.drop('AKTIV', axis=1)
# One-hot encode the categorical variables
# train = pd.get_dummies(df, columns=['Altersgr', 'Abogr', 'jahrstart'])
#valid = pd.get_dummies(valid, columns=['Altersgr', 'abogrup', 'jahrstart'])
# df_test_data = pd.get_dummies(df_test_data, columns=['Altersgr', 'abogrup', 'jahrstart'])
# Update shared variables
shared_train_data = create_shared_data(train)
shared_valid_data = create_shared_data(valid)
# shared_test_data = create_shared_data(df_test_data)
logging.info(f" train {train.shape}, y_train {train_y.shape}")
logging.info(f"head {train.head()}")    
# train.columns
with pm.Model() as model:
    # Priors for each feature
    priors = {col: pm.Normal(col, mu=0, sd=1) for col in train.columns if col != "AKTIV"}

    # Hidden node 'treue'
    treue_coef = {col: pm.Normal(col+"_coef", mu=0, sd=1) for col in ['Altersgr', 'Abogr', 'jahrstart']}
    treue = pm.math.sigmoid(sum([treue_coef[col] * shared_train_data[col] for col in ['Altersgr', 'Abogr', 'jahrstart']]))

    treue2 = pm.Deterministic('treue2', pm.math.sigmoid(sum([treue_coef[col] * shared_train_data[col] for col in train.columns if col.startswith(('Altersgr', 'abogrup', 'jahrstart'))])))
    # Expected value using logistic function
    mu = pm.math.sigmoid(sum([priors[col] * shared_train_data[col] for col in train.columns if col != "AKTIV"]) + treue)

    # Likelihood
    AKTIV = pm.Bernoulli('AKTIV', p=mu, observed=train_y)

    # Sample
    num = 20
    logging.info(f"start sampling {num}")
    trace = pm.sample(num, tune=int(num/2))

# Switch to validation data
for column in valid.columns:
    if column in shared_valid_data:
        shared_train_data[column].set_value(valid[column].values)
with model:
    
    ppc_valid = pm.sample_posterior_predictive(trace, samples=500)
graph = pm.model_to_graphviz(model)
graph.render('graph.png', view=True)
hv = pm.plot_posterior(trace['treue2'])
plt.savefig('treue2.png')
import pdb; pdb.set_trace()