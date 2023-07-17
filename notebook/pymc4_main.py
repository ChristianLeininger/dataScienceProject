
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import cost_matrix, create_shared_data, eval_model, create_data
from sklearn.model_selection import train_test_split
from scipy import stats
import logging
import wandb
import datetime 
import time
import pymc as pm
import theano 
import pymc as pm4
import xarray as xr

import argparse



def main(args):
    logging.basicConfig(level=logging.INFO)
    # track time
    start_time = time.time()
    n = args.number
    size = args.size
    cores = 1
    track = False
    logging.info(f"size {size} draws {n} cores {cores} tracking {track}")
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%A, %B %d, %Y %I:%M %p")
    logging.info(f"current time {formatted_datetime}")
    formatted_datetime = current_datetime.strftime("%m%d%Y_%H%M%S")
    logging.info(f"current time {formatted_datetime}")
    # import pdb; pdb.set_trace()
    if track:
        wandb.init(project="baysian network", 
                config={"dataset_size": size,
                        "draws": n,
                        "features": ['Kaufkraft', 'Bonitaet'],
                },
                name=f"{formatted_datetime}_size{size}_draws{n}",
                )
    data = pd.read_csv('../data/dmc2001_learn.txt', sep=';', na_values=['?', '#NULL!'])
    data['WO'] = data['WO'].replace({'W': 1, 'O': 0, 'F': 2})
    data_filled = data.fillna(data.mean())
    features = ['Kaufkraft', 'Bonitaet']

    scaler = StandardScaler()
    
    # 
    scale = False
    X1 = data_filled['Kaufkraft'].values.reshape(-1, 1)[:size]
    X2 = data_filled['Bonitaet'].values.reshape(-1, 1)[:size]
    if scale:
        X1 = scaler.fit_transform(data_filled['Kaufkraft'].values.reshape(-1, 1))[:size]
        X2 = scaler.fit_transform(data_filled['Bonitaet'].values.reshape(-1, 1))[:size]
    
    y = data_filled['AKTIV'].values[:size]
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1, X2, y, test_size=0.2, random_state=42)
    
    with pm.Model() as model:
        # Priors for unknown model parameters (standard normal priors)
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta1 = pm.Normal('beta1', mu=0, sigma=1)
        beta2 = pm.Normal('beta2', mu=0, sigma=1)

        # Latent variable 'Buying Power'
        buying_power = alpha + beta1*X1_train + beta2*X2_train

        # Logistic function to transform 'buying power' to probability
        bp = pm.Deterministic('bp', pm.math.sigmoid(buying_power))

        # Likelihood
        bp_flat = bp.flatten()
        
        y_obs = pm.Bernoulli('y_obs', p=bp_flat, observed=y_train)

        # Inference!
        trace = pm.sample(n, cores=cores)
    
    
    with model:
        # import pdb; pdb.set_trace() 
        ppc = pm.sample_posterior_predictive(trace, var_names=['alpha', 'beta1', 'beta2'])
    # import pdb; pdb.set_trace()
    # Compute 'Buying Power' for validation set
     
    alpha = xr.concat([ppc['posterior_predictive']['alpha'][i] for i in range(len(ppc['posterior_predictive']['alpha']))], dim='draw')
    beta1 = xr.concat([ppc['posterior_predictive']['beta1'][i] for i in range(len(ppc['posterior_predictive']['beta1']))], dim='draw')
    beta2 = xr.concat([ppc['posterior_predictive']['beta2'][i] for i in range(len(ppc['posterior_predictive']['beta2']))], dim='draw')

    # Now you can use these variables in your computation
    
    X1_val_transposed = X1_val.transpose()
    X2_val_transposed = X2_val.transpose()
    alpha = alpha.to_numpy()
    beta1 = beta1.to_numpy()
    beta2 = beta2.to_numpy()
    alpha_reshaped = alpha.reshape(alpha.shape[0], 1)
    beta1_reshaped = beta1.reshape(beta1.shape[0], 1)
    beta2_reshaped = beta2.reshape(beta2.shape[0], 1)
    buying_power_val = alpha_reshaped + beta1_reshaped * X1_val_transposed + beta2_reshaped * X2_val_transposed
    # Now these arrays have shape (1, 200), which can be broadcasted with the shape (8000,)
    buying_power_val_mean = np.mean(buying_power_val, axis=0)   # thismaybe not the best way to do it 
    # we lose the information about the draws, but we can still compute the mean and std
    # Compute predictions for validation set
    
    prob_samples = 1 / (1 + np.exp(-buying_power_val_mean))

    end_time = time.time()
    # import pdb; pdb.set_trace()
    elapsed_time = end_time - start_time

    logging.info(f"Elapsed time: {elapsed_time :.2f} seconds")
    # import pdb; pdb.set_trace()
    predictions_val = (prob_samples > 0.5).astype(int)
    cost, tp, fp, tn, fn = cost_matrix(predictions_val, y_val)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    logging.info(f"cost: {cost} tp: {tp} fp: {fp} tn: {tn} fn: {fn} accuracy: {accuracy:.2f}")
    
    if track:
        all_cost = {}
        start = 0.1
        end = 0.9
        increment = 0.1
        table = wandb.Table(columns=["Threshold", "Cost", "TP", "FP", "TN", "FN"])
        for i in range(int((end - start) / increment) + 1):
            threshold = float(i) * increment + start
            predictions_val = (prob_samples > threshold).astype(int)
            cost, tp, fp, tn, fn = cost_matrix(predictions_val, y_val)
            all_cost[threshold] = [cost, tp, fp, tn, fn]
            logging.info(f"threshold {threshold}  cost {cost} tp {tp} fp {fp} tn {tn} fn {fn}")
            table.add_data(threshold, cost, tp, fp, tn, fn)
        # sort by cost
        wandb.log({"experiment_results": table})
        sorted_cost = sorted(all_cost.items(), key=lambda x: x[1])
        # import pdb; pdb.set_trace()
        logging.info(f"best {sorted_cost[0]} tp {sorted_cost[0][1][1]} fp {sorted_cost[0][1][2]} tn {sorted_cost[0][1][3]} fn {sorted_cost[0][1][4]}")
        logging.info(f"second best {sorted_cost[1]} tp {sorted_cost[1][1][1]} fp {sorted_cost[1][1][2]} tn {sorted_cost[1][1][3]} fn {sorted_cost[1][1][4]}")
        logging.info(f"third best {sorted_cost[2]} tp {sorted_cost[2][1][1]} fp {sorted_cost[2][1][2]} tn {sorted_cost[2][1][3]} fn {sorted_cost[2][1][4]}")
        # import pdb; pdb.set_trace()
        table2 = wandb.Table(columns=["Threshold", "Cost", "TP", "FP", "TN", "FN"]) 
        table2.add_data(sorted_cost[0][0], sorted_cost[0][1][0], sorted_cost[0][1][1], sorted_cost[0][1][2], sorted_cost[0][1][3], sorted_cost[0][1][4])
        table2.add_data(sorted_cost[1][0], sorted_cost[1][1][0], sorted_cost[1][1][1], sorted_cost[1][1][2], sorted_cost[1][1][3], sorted_cost[1][1][4])
        table2.add_data(sorted_cost[2][0], sorted_cost[2][1][0], sorted_cost[2][1][1], sorted_cost[2][1][2], sorted_cost[2][1][3], sorted_cost[2][1][4])
        wandb.log({"best thresholds": table2})
        wandb.finish()






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program.')
    # Add command line arguments
    parser.add_argument('-s', '--size', type=int, default=1000, help='amount of data to use')
    parser.add_argument('-n', '--number', type=int, default=2000, help='number of draws')
    # Parse the command line arguments
    args = parser.parse_args()
    main(args)