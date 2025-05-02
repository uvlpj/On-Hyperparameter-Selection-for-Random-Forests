#%%
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from itertools import product
import pickle
import os
from scipy.stats import norm
cwd = os.getcwd()

SEED = 7531
np.random.seed(SEED)

plt.rcParams.update({
    "text.usetex": True,
    "font.size" : 12,
    #"pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": True,
    "lines.antialiased": True,
    "patch.antialiased": True,
    'axes.linewidth': 0.1
})
%matplotlib inline
%config InlineBackend.figure_format='retina'

from plotting_helpers import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from RF import RandomForestWeight
from data_preprocessor import load_energy, prep_energy

def simple_dm(l1, l2):
    d = l1 - l2
    mod = sm.OLS(d, np.ones(len(d)))
    res = mod.fit().get_robustcov_results(cov_type='HAC',maxlags=1)
    return res

def calc_r2(y_true, y_pred, y_train=None):

    res_mean = ((y_true - y_pred)**2).mean()

    if y_train is None:
        y_true_var = y_true.var()
    else:
        y_train_mean = np.mean(y_train)
        y_true_var = ((y_true - y_train_mean)**2).mean()

    r2 = 1 - (res_mean / y_true_var)
    
    return r2

def quantile_score(y_true, y_pred, alpha):
    diff = y_true - y_pred
    indicator = (diff >= 0).astype(diff.dtype)
    loss = indicator * alpha * diff + (1 - indicator) * (1 - alpha) * (-diff)
    
    return 2*loss

def se(y_true, y_pred):
    return (y_true - y_pred)**2

def ae(y_true, y_pred):
    return np.abs(y_true - y_pred)

def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

def crps_sample(y, dat, w, return_mean=True):
    '''
    CRPS Werte für Random Forest
    '''
    
    y = y.astype(np.float32)
    dat = dat.astype(np.float32)
   

    order = np.argsort(dat)

    x = dat[order]


    score_arr = np.zeros((len(y)))

    for i in range(w.shape[0]):
        wi = w[i][order]
        yi = y[i]
        p = np.cumsum(wi)
        P = p[-1]
        a = (p - 0.5 * wi) / P

        # score = 2 / P * np.sum(wi * (np.where(yi < x, 1. , 0.) - a) * (x - yi))
        indicator = (yi < x).astype(x.dtype)
        score = 2 / P * (wi * (indicator - a) * (x - yi)).sum()

        score_arr[i] = score

    if return_mean:
        return score_arr.mean()

    return score_arr


def calculate_individual_crps_linear(y_true, y_pred, sigma):
    """
    Berechnet individuelle CRPS-Werte für ein lineares Modell.
    :param y_true: Wahre Werte (Testdaten)
    :param y_pred: Vorhersagen des Modells
    :param sigma: Standardabweichung (Modellunsicherheit)
    :return: Array der individuellen CRPS-Werte
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)
    
    z = (y_true - y_pred) / sigma
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    return crps


def calculate_individual_crps(y_test, dat_test, weights_test):

    individual_crps = crps_sample(y_test, dat_test, weights_test, return_mean=False)
    print('shape crps_values ', individual_crps.shape)
    
    
    return individual_crps

        

def calculate_individual_se(y_test, y_pred):

    individual_se = se(y_test, y_pred)
    print('shape se_values ', individual_se.shape)
    
    return individual_se



#%%
# Set seed for reproducibility
np.random.seed(2024)

# Simulation parameters
n_samples = 500
n_features = 9
correlation = 0.2
n_trees = 100

# Create covariance matrix
cov_matrix = np.full((n_features + 1, n_features + 1), correlation)
np.fill_diagonal(cov_matrix, 1)

# Generate multivariate normal data
mean_vector = np.zeros(n_features + 1)
data = np.random.multivariate_normal(mean=mean_vector, cov=cov_matrix, size=n_samples)

# Create a DataFrame
columns = [f"X{i}" for i in range(1, n_features + 1)] + ["y"]
df = pd.DataFrame(data, columns=columns)
df["Index"] = np.arange(1, n_samples + 1)

# Split data into training and test sets
train_size = 300
dat_train = df.iloc[:train_size]
dat_test = df.iloc[train_size:]
y_train = dat_train["y"].values
y_test = dat_test["y"].values

#%%
base_fml = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']

fmls = []
fmls.append(base_fml) # without timetrend

X_train = dat_train[base_fml]
X_test = dat_test[base_fml]
#%%
# Initialize results
results = {}


crps_arr = []
se_arr = []
mse_arr = []
mae_arr = []
individual_crps_arr = []
individual_se_arr = []

N_TREES = 100

mtry_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Loop over mtry values
for m_try in range(1, n_features + 1):
    print(f"Training with mtry = {m_try}")

    hyperparams = {
                'n_estimators': N_TREES,
                'random_state': SEED,
                'n_jobs': -1,
                'max_features': m_try,
                'min_samples_split': 2,
            }

    hyperparams['random_state'] = hyperparams['random_state'] 
    rf = RandomForestWeight(hyperparams=hyperparams)
    rf.fit(X_train, y_train)
    
    _, w_hat = rf.weight_predict(X_test)
    y_pred = rf.predict(X_test)
    

    crps = crps_sample(y_test, y_train, w_hat)
    crps_arr.append({
            'm_try': m_try,
            'CRPS': crps
        })
    
    individual_crps = calculate_individual_crps(y_test, y_train, w_hat)
    individual_crps_arr.append({
            'm_try': m_try,
            'CRPS': individual_crps})
    

    se_val = se(y_test, y_pred)
    se_arr.append({
    'm_try': m_try,
    'SE': se_val})

    # Individual SE ---
    individual_se = calculate_individual_se(y_test, y_pred)
    individual_se_arr.append({
    'm_try': m_try,
    'SE': individual_se
        })
    
# %%
# CRPS ---
df_crps = pd.DataFrame(crps_arr)
df_crps
#%%
# Individual CRPS ---
df_individual_crps = pd.DataFrame(individual_crps_arr)
df_individual_crps
#%%
# Individual SE --
df_se = pd.DataFrame(se_arr)
df_se
#%%
df_individual_se_arr = pd.DataFrame(individual_se_arr)
df_individual_se_arr
#%%
# MSE ---
df_mse = pd.DataFrame(mse_arr)
df_mse
#%%
# MAE ---
df_mae = pd.DataFrame(mae_arr)
df_mae
#%%
output_folder = r"/home/siefert/projects/Masterarbeit/Data/Results/Python_Results/python_res_simulated_data_different_mtry"
os.makedirs(output_folder, exist_ok=True)
#%%
def save_results_different_mtry(individual_crps_arr, individual_se_arr, dat_test, prefix="python_res_simulated_data_different_mtry/"):
    '''
    Funktion um die Ergebnisse von sklearn zu speichern
    
    Output: CSV-Dateien, die die CRPS- sowie die SE-Werte für jeden Datenpunkt im Testdatensatz enthalten.
    '''
    os.makedirs(prefix, exist_ok=True)  # Erstelle das Verzeichnis, falls es nicht existiert
    
    for i, result in enumerate(individual_crps_arr):
        mtry_value = f"mtry{result['m_try']}"
        
        # Speichern für Random Forest
        save_name = f"{prefix}sklearn_{mtry_value}.csv"
    
        # Erstelle einen DataFrame für Random Forest
        df = pd.DataFrame({
            'crps': result['CRPS'],  # Individuelle CRPS-Werte für RF
            'se': se_arr[i]['SE'],    # Individuelle SE-Werte für RF
        })
        
        # Speichern der Ergebnisse als CSV-Datei mit Index
        df.to_csv(save_name, index=False)  # Index wird mitgespeichert
      
        
        print(f"Saved: {save_name}")
# %%
save_results_different_mtry(individual_crps_arr, individual_se_arr, dat_test ,prefix="python_res_simulated_data_different_mtry/")
# %%
