import pymc3 as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import cost_matrix, create_shared_data, eval_model, create_data
from sklearn.model_selection import train_test_split
from scipy import stats
import logging


logging.basicConfig(level=logging.INFO)
data = pd.read_csv('../data/dmc2001_learn.txt', sep=';', na_values=['?', '#NULL!'])
data['WO'] = data['WO'].replace({'W': 1, 'O': 0, 'F': 2})
data_filled = data.fillna(data.mean())
features = ['Kaufkraft', 'Bonitaet']

scaler = StandardScaler()
n = 200
size = 1000
# import pdb; pdb.set_trace()
X1 = scaler.fit_transform(data_filled['Kaufkraft'].values.reshape(-1, 1))[:size]
X2 = scaler.fit_transform(data_filled['Bonitaet'].values.reshape(-1, 1))[:size]
y = data_filled['AKTIV'].values[:size]
X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

with pm.Model() as model:
    # Priors for unknown model parameters (standard normal priors)
    alpha = pm.Normal('alpha', mu=0, sd=1)
    beta1 = pm.Normal('beta1', mu=0, sd=1)
    beta2 = pm.Normal('beta2', mu=0, sd=1)

    # Latent variable 'Buying Power'
    buying_power = alpha + beta1*X1_train + beta2*X2_train

    # Logistic function to transform 'buying power' to probability
    p = pm.Deterministic('p', pm.math.sigmoid(buying_power))

    # Likelihood
    y_obs = pm.Bernoulli('y_obs', p=p, observed=y_train)

    # Inference!
    trace = pm.sample(n, tune=int(n/2), cores=8)

# Posterior predictive checks
with model:
    ppc = pm.sample_posterior_predictive(trace, var_names=['alpha', 'beta1', 'beta2'])
# import pdb; pdb.set_trace()
# Compute 'Buying Power' for validation set
buying_power_val = ppc['alpha'] + ppc['beta1']*X1_val + ppc['beta2']*X2_val

# Compute predictions for validation set
prob_samples = 1 / (1 + np.exp(-buying_power_val))

# Compute the average probability for each instance
avg_prob = prob_samples.mean(axis=1)

# Make binary predictions for each instance
all_cost = {}
start = 0.1
end = 0.9
increment = 0.1
for i in range(int((end - start) / increment) + 1):
    threshold = float(i) * increment + start
    predictions_val = (avg_prob > threshold).astype(int)
    cost, tp, fp, tn, fn = cost_matrix(predictions_val, y_val)
    all_cost[threshold] = [cost, tp, fp, tn, fn]
    logging.info(f"threshold {threshold}  cost {cost} tp {tp} fp {fp} tn {tn} fn {fn}")
import pdb; pdb.set_trace()
# sort by cost
sorted_cost = sorted(all_cost.items(), key=lambda x: x[1])
logging.info(f"best threshold {sorted_cost[0][0]} with cost {sorted_cost[0][1]} and tp {sorted_cost[0][2]} fp {sorted_cost[0][3]} tn {sorted_cost[0][4]} fn {sorted_cost[0][5]}")
logging.info(f"second best threshold {sorted_cost[1][0]} with cost {sorted_cost[1][1]} and tp {sorted_cost[1][2]} fp {sorted_cost[1][3]} tn {sorted_cost[1][4]} fn {sorted_cost[1][5]}")
logging.info(f"third best threshold {sorted_cost[2][0]} with cost {sorted_cost[2][1]} and tp {sorted_cost[2][2]} fp {sorted_cost[2][3]} tn {sorted_cost[2][4]} fn {sorted_cost[2][5]}")
logging.info(f"worst threshold {sorted_cost[-1][0]} with cost {sorted_cost[-1][1]} and tp {sorted_cost[-1][2]} fp {sorted_cost[-1][3]} tn {sorted_cost[-1][4]} fn {sorted_cost[-1][5]}")



