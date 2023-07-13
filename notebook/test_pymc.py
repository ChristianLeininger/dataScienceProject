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

df = pd.read_csv("../data/dmc2001_learn.txt", sep=";")
df = df.fillna(df.mean(numeric_only=True))
# Preprocessing: let's assume all your columns are numerical for simplicity
df['WO'] = df['WO'].replace({'W': 1, 'O': 0, 'F': 2})
df.head()
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
train, test = train_test_split(df, test_size=0.2)


modelx = shared(np.array(train))
bglm = pm.Model()
with bglm:
    # Priors for each feature
    priors = {
        col: pm.Normal(col, mu=0, sd=1) for col in train.columns if col != "AKTIV"
    }

    # Expected value using logistic function
    mu = pm.math.sigmoid(sum([priors[col] * train[col] for col in train.columns if col != "AKTIV"]))

    # Likelihood
    AKTIV = pm.Bernoulli('AKTIV', p=mu, observed=train['AKTIV'])

    # Sample
    trace = pm.sample(2000, tune=1000)
modelx.set_value(np.array(test))
samples = 5
ppc = pm.sample_posterior_predictive(trace, model=bglm, samples=samples,random_seed=6)
print(ppc["AKTIV"].shape)

import pdb; pdb.set_trace()
