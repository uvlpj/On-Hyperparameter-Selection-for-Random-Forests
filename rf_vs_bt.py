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


# SE ---
def se(y_true, y_pred):
    return (y_true - y_pred)**2

def se_1(y_true, y_pred):
    return (y_true - np.mean(y_pred, axis = 1))**2

# AE ---
def ae(y_true, y_pred):
    return np.abs(y_true - y_pred)

def ae_1(y_true, y_pred):
    return np.abs(y_true - np.median(y_pred, axis = 1))

# MSE --
def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

# MAE ---
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))



def crps_sample(y, dat, w, return_mean=True):

    y = y.astype(np.float32)
    #print("y: ", y.shape)
    dat = dat.astype(np.float32)
    #print("dat: ", dat.shape)

    order = np.argsort(dat)
    x = dat[order]

    score_arr = np.zeros((len(y)))

    for i in range(w.shape[0]):
        wi = w[i][order]    #
        #print("wi: ", wi)
        yi = y[i]    
        #print("yi: ", yi)       
        p = np.cumsum(wi)  
        #print("p: ", p) 
        P = p[-1]   
        #print(P)        
        a = (p - 0.5 * wi) / P

        # Berechnung des CRPS 
        # score = 2 / P * np.sum(wi * (np.where(yi < x, 1. , 0.) - a) * (x - yi))
        indicator = (yi < x).astype(x.dtype)
        score = 2 / P * (wi * (indicator - a) * (x - yi)).sum()

        score_arr[i] = score

    if return_mean:
        return score_arr.mean()

    return score_arr



def calculate_individual_crps(y_test, dat_test, weights_test):

    individual_crps = crps_sample(y_test, dat_test, weights_test, return_mean=False)
    print('shape crps_values ', individual_crps.shape)
    return individual_crps
        

def calculate_individual_se(y_test, y_pred):

    individual_se = se(y_test, y_pred)
    print('shape se_values ', individual_se.shape)
    return individual_se

def calculate_individual_se_1(y_test, y_pred):

    individual_se_1 = se_1(y_test, y_pred)
    print('shape se_values ', individual_se_1.shape)
    return individual_se_1

def calculate_individual_ae(y_test, y_pred):

    individual_ae = ae(y_test, y_pred)
    print('shape ae_values ', individual_ae.shape)
    return individual_se

def calculate_individual_ae_1(y_test, y_pred):

    individual_ae_1 = ae_1(y_test, y_pred)
    print('shape ae_values ', individual_ae_1.shape)
    return individual_ae_1


#%%

def save_results(individual_crps_arr, individual_se_arr, individual_ae_arr ,dat_test, prefix="python_res_rf_bt/"):

    '''
    Funktion um die Ergebnisse von sklearn zu speichern
    
    Output: CSV-Dateien, die die CRPS- sowie die SE-Werte für jeden Datenpunkt im Testdatensatz enthalten.
    '''
    os.makedirs(prefix, exist_ok=True)  # Erstelle das Verzeichnis
    
    for i, result in enumerate(individual_crps_arr):
        # Dynamischer Dateiname basierend auf den Parametern (time_trend, day_of_year)
        time_trend_part = 'tt' if result['time_trend'] == 'yes' else 'nott'
        day_part = 'day' if result['day_of_year'] == 'yes' else 'month'
        lag_part = 'lagged' if result['load_lag1'] == 'yes' else 'notlagged'
        
        # Speichern für Random Forest
        save_name_rf = f"{prefix}sklearn_{time_trend_part}_{day_part}_{lag_part}_rf.csv"
        # Speichern für Bagged Trees
        save_name_bt = f"{prefix}sklearn_{time_trend_part}_{day_part}_{lag_part}_bt.csv"

        # RF
        df_rf = pd.DataFrame({
            'date_time': dat_test.index, 
            'crps': result['CRPS_RF'],  # Individuelle CRPS-Werte für RF
            'se': individual_se_arr.iloc[i]['SE_RF'],    # Individuelle SE-Werte für RF
            'ae': individual_ae_arr.iloc[i]['AE_RF'],
        })
        
       # BT
        df_bt = pd.DataFrame({
            'date_time': dat_test.index, 
            'crps': result['CRPS_BT'],  # Individuelle CRPS-Werte für BT
            'se': individual_se_arr.iloc[i]['SE_BT'],    # Individuelle SE-Werte für BT
            'ae': individual_ae_arr.iloc[i]['AE_BT'],
        })
        
        # Speichern der Ergebnisse als CSV-Datei mit Index
        df_rf.to_csv(save_name_rf, index=False)  # Index wird mitgespeichert
        df_bt.to_csv(save_name_bt, index=False)  # Index wird mitgespeichert
        
        print(f"Saved: {save_name_rf}")
        print(f"Saved: {save_name_bt}")

#%%

#%%
#output_folder = r"/home/siefert/projects/Masterarbeit/sophia_code/python_res"
#os.makedirs(output_folder, exist_ok=True)
#%%
#output_folder = r"/home/siefert/projects/Masterarbeit/Data/Results/Python_Results/python_res_different_mtry3"
#os.makedirs(output_folder, exist_ok=True)
#%%
def save_results_different_mtry(individual_crps_arr, individual_se_arr, individual_ae_arr, dat_test, prefix="python_res_different_mtry/"):

    '''
    Funktion um die Ergebnisse von sklearn zu speichern
    
    Output: CSV-Dateien, die die CRPS- sowie die SE-Werte für jeden Datenpunkt im Testdatensatz enthalten.
    '''
    os.makedirs(prefix, exist_ok=True)  # Erstelle das Verzeichnis
    
    for i, result in enumerate(individual_crps_arr):
        # Dynamischer Dateiname basierend auf den Parametern (time_trend, day_of_year)
        time_trend_part = 'tt' if result['time_trend'] == 'yes' else 'nott'
        day_part = 'day' if result['day_of_year'] == 'yes' else 'month'
        lag_part = 'lagged' if result['load_lag1'] == 'yes' else 'notlagged'
        mtry_value = f"mtry{result['m_try']}"
        
        # Speichern für Random Forest
        save_name = f"{prefix}sklearn_{time_trend_part}_{day_part}_{lag_part}_{mtry_value}.csv"
    
        # Erstelle einen DataFrame für Random Forest
        df = pd.DataFrame({
            'date_time': dat_test.index, 
            'crps': result['CRPS'],  # Individuelle CRPS-Werte für RF
            'se': se_arr[i]['SE'],    # Individuelle SE-Werte für RF
            'ae': individual_ae_arr[i]['AE']
        })
        
        # Speichern der Ergebnisse als CSV-Datei mit Index
        df.to_csv(save_name, index=False)  # Index wird mitgespeichert
      
        
        print(f"Saved: {save_name}")
   

#%%
winter_time = ['2018-10-28 02:00:00',
               '2019-10-27 02:00:00',
               '2020-10-25 02:00:00',
               '2021-10-31 02:00:00',
               '2022-10-30 02:00:00',
               '2023-10-29 02:00:00']

df_orig = load_energy('/home/siefert/projects/Masterarbeit/dat_energy/rf_data_1823_clean.csv')

# remove samples with 0 load. these arise due to daylight saving
df_orig = df_orig[df_orig.load > 0]
df_orig = df_orig[df_orig.load < 82000]
df_orig = df_orig[~df_orig.date_time.isin(winter_time)]

df_orig['load_lag1'] = df_orig['load'].shift(1)

data_encoding = dict(time_trend=False,
                     time_trend_sq=False,
                     cat_features=False,
                     fine_resolution=False,
                     sin_cos_features=False,
                     last_obs=False)
    
base_fml = ['hour_int', 'weekday_int', 'holiday']
#base_fml = ['hour_int', 'weekday_int', 'holiday', 'load_lag1']


fmls = []
fmls.append(base_fml + ['month_int']) # without timetrend
fmls.append(base_fml + ['yearday']) # without timetrend
fmls.append(base_fml + ['time_trend', 'month_int'])
fmls.append(base_fml + ['time_trend', 'yearday'])
fmls.append(base_fml + ['month_int', 'load_lag1']) 
fmls.append(base_fml + ['yearday', 'load_lag1']) 
fmls.append(base_fml + ['time_trend', 'month_int', 'load_lag1'])
fmls.append(base_fml + ['time_trend', 'yearday', 'load_lag1'])

#%% 
# ============
# Model with BT und RF and mtry = p and mtry = p/3 ---
'''
Es gibt zwei Schritte
Schritt 1 => das ist die Ausganglage, da sind keine Hyperparameter angepasst und der Intercept ist noch in quantregForest
    wenn dieser Schritt simuliert werden muss 
    => N_TREES = 1000, n_estimators=N_TREES, random_state=SEED, max_features=MTRY_RF, min_samples_split = 2,

Schritt 2 => Hyperparameter wurden auf die defaultwerte von sklearn gesetzt
    wenn dieser Schritt simuliert werden soll
    => N_TREES = 100, n_estimators=N_TREES, random_state=SEED, max_features=MTRY_RF, min_samples_split = 2, min_samples_leaf = 1, max_depth = None
'''
SEED = 7531
#N_TREES = 1000
N_TREES = 100 # Default value of sklearn

crps_arr = []
se_arr = []
mse_arr = []
mae_arr = []
individual_crps_arr = []

individual_se_arr = []
individual_ae_arr = []

individual_se_1_arr = []
individual_ae_1_arr = []

quant_results_bt = []
quant_results_rf = []


df_orig.dropna(inplace=True)

for fml in fmls:
    print(fml)
    print(f"Verwendete Formel: {fml}")

    df, _, _ = prep_energy(df=df_orig, **data_encoding)
    
    COMBINED_TEST_PERIOD = True

    if COMBINED_TEST_PERIOD:
        tp_start = "2022-01-01 00:00:00"
        
        dat_train = df.set_index("date_time")[:tp_start]
        dat_test = df.set_index("date_time")[tp_start:]
        
        y_train = dat_train['load'].values
        y_test = dat_test['load'].values
        
        dat_train.drop(columns=['load'], inplace=True)
        dat_test.drop(columns=['load'], inplace=True)
        
        X_train = dat_train[fml]
        X_test = dat_test[fml]
    else:
        tp_start1 = "2022-01-01 00:00:00"
        tp_start2 = "2023-01-01 00:00:00"

        # test_period = 2022
        dat_train = df.set_index("date_time")[:tp_start1]
        dat_test1 = df.set_index("date_time")[tp_start1:tp_start2][:-1]
        dat_test2 = df.set_index("date_time")[tp_start2:]

        y_train = dat_train['load'].values
        y_test = dat_test1['load'].values
        y_test2 = dat_test2['load'].values

        dat_train.drop(columns=['load'], inplace=True)
        dat_test1.drop(columns=['load'], inplace=True)
        dat_test2.drop(columns=['load'], inplace=True)

        X_train = dat_train[fml]
        X_test = dat_test1[fml]
        X_test2 = dat_test2[fml]
    
    MTRY_RF = int(len(fml) / 3)
    print('Number of features p: ', len(fml))
    print('mtry BT = p: ', len(fml))
    print('mtry RF = p/3 :',  MTRY_RF)

    # keep this in the loop so each config gets same seed
    hyperparams = dict(n_estimators=N_TREES,
                    random_state=SEED,
                    n_jobs=-1,
                    max_features=MTRY_RF,
                    min_samples_split=2,
                    min_samples_leaf = 1,
                    max_depth = None
                    )

    hyperparams['random_state'] = hyperparams['random_state'] 
    rf = RandomForestWeight(hyperparams=hyperparams)
    rf.fit(X_train, y_train)

    hyperparams_bt = hyperparams.copy()
    hyperparams_bt["max_features"] = len(fml)

    bt = RandomForestWeight(hyperparams=hyperparams_bt)
    bt.fit(X_train, y_train)

    _, w_hat_rf = rf.weight_predict(X_test)
    _, w_hat_bt = bt.weight_predict(X_test)
    
    #y_pred_bt, w_hat_rf = rf.weight_predict(X_test)
    #y_pred_rf, w_hat_bt = bt.weight_predict(X_test)
    print("w_hat_rf: ", w_hat_rf)
    print("Shape von w_hat_rf: ", w_hat_rf.shape)
    print("w_hat_bf: ", w_hat_bt)
    print("Shape von w_hat_bt: ", w_hat_bt.shape)


    y_pred_rf = rf.predict(X_test)
    y_pred_bt = bt.predict(X_test)
    print(y_pred_bt)
    print(y_pred_rf)

   #n_quantiles = int(1e2)
    #grid_quantiles = (2 * np.arange(1, n_quantiles + 1) - 1) / (2 * n_quantiles)

    #quant_pred_rf = rf.quantile_predict(grid_quantiles, X_test = X_test)
    #quant_pred_bt = bt.quantile_predict(grid_quantiles, X_test = X_test)


    # CRPS ---
    crps_rf = crps_sample(y_test, y_train, w_hat_rf)
    crps_bt = crps_sample(y_test, y_train, w_hat_bt)
    
    crps_arr.append({'time_trend': 'yes' if 'time_trend' in fml else 'no',
                     'day_of_year': 'yes' if 'yearday' in fml else 'no',
                     'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
                     'CRPS_RF': crps_rf, 'CRPS_BT': crps_bt})
    

    # Individual CRPS ---
    individual_crps_rf = calculate_individual_crps(y_test, y_train, w_hat_rf)
    individual_crps_bt = calculate_individual_crps(y_test, y_train, w_hat_bt)
    print('Shape individual CRPS RF', individual_crps_rf.shape)
    print('Shape individual CRPS BT', individual_crps_bt.shape)
    individual_crps_arr.append({
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
            'CRPS_RF': individual_crps_rf,#[i],  
            'CRPS_BT': individual_crps_bt,#[i], 
            'date_time': dat_test.index,#[i]    
        })

    
    # SE ---
    se_rf = se(y_test, y_pred_rf)
    se_bt = se(y_test, y_pred_bt)
    se_arr.append({'time_trend': 'yes' if 'time_trend' in fml else 'no',
                     'day_of_year': 'yes' if 'yearday' in fml else 'no',
                     'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
                     'SE_RF': se_rf, 'SE_BT': se_bt})
    
    
    #individual SE ---
    individual_se_rf = calculate_individual_se(y_test, y_pred_rf)
    individual_se_bt = calculate_individual_se(y_test, y_pred_bt)

    print('Shape individual SE RF', individual_se_rf.shape)
    print('Shape individual SE BT', individual_se_bt.shape)

    individual_se_arr.append({
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
            'SE_RF': individual_se_rf,
            'SE_BT': individual_se_bt,
            'date_time': dat_test.index,
            })
    

    # SE 1 nach version von Fabian ---
    #individual_se_1_rf = calculate_individual_se_1(y_test, quant_pred_rf)
    #individual_se_1_bt = calculate_individual_se_1(y_test, quant_pred_bt)
    #individual_se_1_arr.append({
    #        'time_trend': 'yes' if 'time_trend' in fml else 'no',
    #        'day_of_year': 'yes' if 'yearday' in fml else 'no',
    #        'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
    #        'SE_1_RF': individual_se_1_rf,
    #        'SE_1_BT': individual_se_1_bt,
    #        'date_time': dat_test.index,
    #        })


    #MSE ---
    mse_rf = mse(y_test, y_pred_rf)
    mse_bt = mse(y_test, y_pred_bt)
    print('MSE RF', mse_rf)
    print('MSE BT', mse_bt)
    mse_arr.append({'time_trend': 'yes' if 'time_trend' in fml else 'no',
                     'day_of_year': 'yes' if 'yearday' in fml else 'no',
                     'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
                     'MSE_RF': mse_rf, 'MSE_BT': mse_bt})



    # MAE ---
    #mae_rf = mae(y_test, y_pred_rf)
    #mae_bt = mae(y_test, y_pred_bt)
    mae_rf = mae(y_test, rf.quantile_predict(q=.5, X_test=X_test))
    mae_bt = mae(y_test, bt.quantile_predict(q=.5, X_test=X_test))



    # AE ---
    individual_ae_rf = ae(y_test, rf.quantile_predict(q=.5, X_test=X_test))
    individual_ae_bt = ae(y_test, bt.quantile_predict(q=.5, X_test=X_test))


    #individual_ae_rf = ae(y_test, y_pred_rf)
    #individual_ae_bt = ae(y_test, y_pred_bt)

    individual_ae_arr.append({
    'time_trend': 'yes' if 'time_trend' in fml else 'no',
    'day_of_year': 'yes' if 'yearday' in fml else 'no',
    'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
    'AE_RF': individual_ae_rf,
    'AE_BT': individual_ae_bt,
    'date_time': dat_test.index
    })

    mae_arr.append({'time_trend': 'yes' if 'time_trend' in fml else 'no',
                     'day_of_year': 'yes' if 'yearday' in fml else 'no',
                     'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
                     'MAE_RF': mae_rf, 'MAE_BT': mae_bt})
    

    # AE 1 nach version von Fabian ---
    #individual_ae_1_rf = calculate_individual_ae_1(y_test, quant_pred_rf)
    #individual_ae_1_bt = calculate_individual_ae_1(y_test, quant_pred_bt)
    #print(" individual_ae_1_rf",  individual_ae_1_rf)

    #individual_ae_1_arr.append({
    #'time_trend': 'yes' if 'time_trend' in fml else 'no',
    #'day_of_year': 'yes' if 'yearday' in fml else 'no',
    #'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
    #'AE_1_RF': individual_ae_1_rf,
    #'AE_1_BT': individual_ae_1_bt,
    #'date_time': dat_test.index,  
    #})


    #median_quant_rf = np.median(quant_pred_rf, axis=1)
    #mean_quant_rf = np.mean(quant_pred_rf, axis=1)
 
    #median_quant_bt = np.median(quant_pred_bt, axis=1)
    #mean_quant_bt = np.mean(quant_pred_bt, axis=1)

    #quant_results_rf = {
    #'time_trend': 'yes' if 'time_trend' in fml else 'no',
    #'day_of_year': 'yes' if 'yearday' in fml else 'no',
    #'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
    #'Median_RF': median_quant_rf,
    #'Mean_RF': mean_quant_rf
#}

#quant_results_bt = {
#    'time_trend': 'yes' if 'time_trend' in fml else 'no',
#    'day_of_year': 'yes' if 'yearday' in fml else 'no',
#    'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
#    'Median_BT': median_quant_bt,
#    'Mean_BT': mean_quant_bt
#}
#%%
# CRPS ---
df_crps = pd.DataFrame(crps_arr)
df_crps
#%%
# Individual CRPS ---
df_individual_crps = pd.DataFrame(individual_crps_arr)
df_individual_crps
#%%
# Individual SE ---
individual_se_arr = pd.DataFrame(individual_se_arr)
individual_se_arr
#%%
# Individual SE1 ---
individual_se_1_arr = pd.DataFrame(individual_se_1_arr)
individual_se_1_arr
#%%
# Individual AE ---
individual_ae_arr = pd.DataFrame(individual_ae_arr)
individual_ae_arr
#%%
# Individual AE1 ---
individual_ae_1_arr = pd.DataFrame(individual_ae_1_arr)
individual_ae_1_arr
#%%
# MSE ---
df_mse = pd.DataFrame(mse_arr)
df_mse
#%%
# MAE ---
df_mae = pd.DataFrame(mae_arr)
df_mae
#%%
# Quantile predictions BT ---
df_quant_results_bt = pd.DataFrame(quant_results_bt)

#%%
# Quantile predictions RF ---
df_quant_results_rf = pd.DataFrame(quant_results_rf)

#%%
#save_results(individual_crps_arr, individual_se_arr, individual_ae_arr, individual_se_1_arr, individual_ae_1_arr,dat_test ,prefix="python_res_rf_bt/")

save_results(individual_crps_arr, individual_se_arr, individual_ae_arr, dat_test ,prefix="python_res_rf_bt/")


#%%

#%%

#%%
#%%
# ==========================================================================================
# Model without the specification of BT and RF => instead one model and mtry = 1,....p ---

N_TREES = 100 # Default value of sklearn
#N_TREES = 500 # Default value of sklearn

SEED = 7531


crps_arr = []
se_arr = []
mse_arr = []
mae_arr = []
individual_crps_arr = []
individual_se_arr = []
individual_ae_arr = []

# Gewichte
w_hat_arr = []
q_hat_arr = []
quant_results = []


rf_models = []

df_orig.dropna(inplace=True)

for fml in fmls:
    print(fml)
    print(f"Verwendete Formel: {fml}")

    df, _, _ = prep_energy(df=df_orig, **data_encoding)
    
    COMBINED_TEST_PERIOD = True

    if COMBINED_TEST_PERIOD:
        tp_start = "2022-01-01 00:00:00"
        
        dat_train = df.set_index("date_time")[:tp_start]
        dat_test = df.set_index("date_time")[tp_start:]
        
        y_train = dat_train['load'].values
        y_test = dat_test['load'].values
        
        dat_train.drop(columns=['load'], inplace=True)
        dat_test.drop(columns=['load'], inplace=True)
        
        X_train = dat_train[fml]
        X_test = dat_test[fml]
    else:
        tp_start1 = "2022-01-01 00:00:00"
        tp_start2 = "2023-01-01 00:00:00"

        # test_period = 2022
        dat_train = df.set_index("date_time")[:tp_start1]
        dat_test1 = df.set_index("date_time")[tp_start1:tp_start2][:-1]
        dat_test2 = df.set_index("date_time")[tp_start2:]

        y_train = dat_train['load'].values
        y_test = dat_test1['load'].values
        y_test2 = dat_test2['load'].values

        dat_train.drop(columns=['load'], inplace=True)
        dat_test1.drop(columns=['load'], inplace=True)
        dat_test2.drop(columns=['load'], inplace=True)

        X_train = dat_train[fml]
        X_test = dat_test1[fml]
        X_test2 = dat_test2[fml]
    
    if dat_train.iloc[-1].name == dat_test.iloc[0].name:
        dat_test = dat_test.iloc[1:]
    
    if y_train[-1] == y_test[0]:
        y_test = y_test[1:]  # Entferne den ersten Wert aus y_test
        X_test = X_test[1:] 

       
    print('Number of features p: ', len(fml))

    first_y_train = y_train[0]  # Erster Wert von y_train
    last_y_train = y_train[-1]  # Letzter Wert von y_train
    first_y_test = y_test[0]    # Erster Wert von y_test
    last_y_test = y_test[-1]    # Letzter Wert von y_test


    print(f"First value of y_train: {first_y_train}")
    print(f"Last value of y_train: {last_y_train}")
    print(f"First value of y_test: {first_y_test}")
    print(f"Last value of y_test: {last_y_test}")


    

    for m_try in range(1, len(fml) + 1):
        print(f"Training RandomForest with mtry = {m_try}")
        print("Checking the variable type of m_try: ", type(m_try) )

        # Setze Hyperparameter für Random Forest
        hyperparams = {
            'n_estimators': N_TREES,
            'random_state': SEED,
            'n_jobs': -1,
            'max_features': m_try,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_depth': None
        }

        print(X_test.shape)

        hyperparams['random_state'] = hyperparams['random_state'] 
        rf = RandomForestWeight(hyperparams=hyperparams)
        rf.fit(X_train, y_train)

        rf_models.append(rf.rf)

        _, w_hat = rf.weight_predict(X_test)
        #y_pred, w_hat = rf.weight_predict(X_test)
        print("w_hat", w_hat)
        print("w_hat.shape", w_hat.shape)
    
        y_pred = rf.predict(X_test)


        # W_hat
        #w_hat_filename = f"w_hat_mtry_{m_try}_{fml}.npz"
        #np.savez_compressed(w_hat_filename, w_hat=w_hat)

        # Füge nur den Pfad zur Liste hinzu, nicht die gesamte Matrix
        #w_hat_arr.append({
        #    'm_try': m_try,
        #    'time_trend': 'yes' if 'time_trend' in fml else 'no',
        #    'day_of_year': 'yes' if 'yearday' in fml else 'no',
        #    'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
        #    'w_hat_filepath': w_hat_filename  # Speichern des Dateipfads zur Matrix
        #})

        # Quantile Predictions ---

        #n_quantiles = int(1e2)
        #grid_quantiles = (2 * (np.arange(1, n_quantiles + 1)) - 1) / (2 * n_quantiles)

        #q_hat = rf.quantile_predict(grid_quantiles, X_test= X_test)

        #quantile_column_names = [f'quantile_{q:.4f}' for q in grid_quantiles]
        #q_hat_df = pd.DataFrame(q_hat, columns=quantile_column_names)

        #q_hat_filename = f"q_hat_mtry_{m_try}_{fml}.npz"
        #np.savez_compressed(q_hat_filename, q_hat=q_hat_df)

        # Füge nur den Pfad zur Liste hinzu, nicht die gesamte Matrix
        #q_hat_arr.append({
        #    'm_try': m_try,
        #    'time_trend': 'yes' if 'time_trend' in fml else 'no',
        #    'day_of_year': 'yes' if 'yearday' in fml else 'no',
        #    'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
        #    'w_hat_filepath': q_hat_filename  # Speichern des Dateipfads zur Matrix
        #})
        


   

        # CRPS ---
        crps = crps_sample(y_test, y_train, w_hat)

        crps_arr.append({
            'm_try': m_try,
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
            'CRPS': crps
        })

        # Individual CRPS ---
        individual_crps = calculate_individual_crps(y_test, y_train, w_hat)
        
        individual_crps_arr.append({
            'm_try': m_try,
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
            'CRPS': individual_crps,
            'date_time': dat_test.index
        })

         # SE ---
        se_val = se(y_test, y_pred)

        se_arr.append({
            'm_try': m_try,
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
            'SE': se_val
        })


        # Individual SE ---
        individual_se = calculate_individual_se(y_test, y_pred)


        individual_se_arr.append({
            'm_try': m_try,
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
            'SE': individual_se,
            'date_time': dat_test.index
        })

        # Berechne den Median (Quantil bei q=0.5) für die Vorhersagen
        y_pred_median = rf.quantile_predict(q=0.5, X_test=X_test)
        individual_ae = ae(y_test, y_pred_median)
        print(individual_ae)

        #individual_ae = ae(y_test, rf.quantile_predict(q=.5, X_test=X_test))
        #print(individual_ae)


        quant_results = {
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
            'q_hat': y_pred_median,
            'date_time': dat_test.index
        }

        # Individual AE ---
        individual_ae_arr.append({
            'm_try': m_try,
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
            'AE': individual_ae,
            'date_time': dat_test.index
        })

        # MSE ---
        mse_val = mse(y_test, y_pred)

        mse_arr.append({
            'm_try': m_try,
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
            'MSE': mse_val
        })

        # MAE ---
        mae_val = mae(y_test, rf.quantile_predict(q=.5, X_test=X_test))

        mae_arr.append({
            'm_try': m_try,
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'load_lag1': 'yes' if 'load_lag1' in fml else 'no',
            'MAE': mae_val
        })

#%%
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
# Individual SE ---
df_individual_se_arr = pd.DataFrame(individual_se_arr)
df_individual_se_arr
#%%
# Individual AE ---
df_individual_ae_arr = pd.DataFrame(individual_ae_arr)
df_individual_ae_arr
#%%
# MSE ---
df_mse = pd.DataFrame(mse_arr)
df_mse
#%%
# MAE ---
df_mae = pd.DataFrame(mae_arr)
df_mae
#%%
# Quantile ---
df_quant = pd.DataFrame(quant_results)
df_quant
#%%
save_results_different_mtry(individual_crps_arr, individual_se_arr, individual_ae_arr, dat_test ,prefix="python_res_different_mtry/")

#%%
# Combined Forecasts ----
# Combined CRPS
'''
for mtry = 1
and "nott_day_lagged", "nott_month_lagged", "tt_day_lagged", "tt_month_lagged"
    w_hat = 0.25*w_hat1 + 0.25*w_hat2 + 0.25*w_hat3 + 0.25*w_hat4

for mtry = 5
and "nott_day_lagged", "nott_month_lagged", "tt_day_lagged", "tt_month_lagged"
    w_hat = 0.25*w_hat1 + 0.25*w_hat2 + 0.25*w_hat3 + 0.25*w_hat4
'''

# nott_month ---
w_hat_filepath_1 = "/home/siefert/projects/Masterarbeit/sophia_code/w_hat_mtry_1_['hour_int', 'weekday_int', 'holiday', 'month_int', 'load_lag1'].npz"
# tt_day ---
w_hat_filepath_2 = "/home/siefert/projects/Masterarbeit/sophia_code/w_hat_mtry_1_['hour_int', 'weekday_int', 'holiday', 'time_trend', 'yearday', 'load_lag1'].npz"
# tt_month ---
w_hat_filepath_3 = "/home/siefert/projects/Masterarbeit/sophia_code/w_hat_mtry_1_['hour_int', 'weekday_int', 'holiday', 'time_trend', 'month_int', 'load_lag1'].npz"
# nott_day ---
w_hat_filepath_4 = "/home/siefert/projects/Masterarbeit/sophia_code/w_hat_mtry_1_['hour_int', 'weekday_int', 'holiday', 'yearday', 'load_lag1'].npz"


# Datei laden
loaded_data_1 = np.load(w_hat_filepath_1)
loaded_data_2 = np.load(w_hat_filepath_2)
loaded_data_3 = np.load(w_hat_filepath_3)
loaded_data_4 = np.load(w_hat_filepath_4)

# `w_hat` extrahieren
w_hat_1 = loaded_data_1["w_hat"]
w_hat_2 = loaded_data_2["w_hat"]
w_hat_3 = loaded_data_3["w_hat"]
w_hat_4 = loaded_data_4["w_hat"]

w_hat_combined = 0.25*w_hat_1 + 0.25*w_hat_2 + 0.25*w_hat_3 + 0.25*w_hat_4

individual_crps = calculate_individual_crps(y_test, y_train, w_hat_combined)

df_individual_crps = pd.DataFrame({
    "date_time": dat_test.index,  
    "crps_combined": individual_crps
})



df_individual_crps.to_csv("sklearn_mtry1.csv", index=False)

print("Individuelle CRPS-Werte wurden erfolgreich als CSV gespeichert!")
#%%


#%%

#%%
# Plot CRPS ---
plt.style.use('seaborn-whitegrid')
plt.rcParams['text.usetex'] = False
plt.figure(figsize=(12, 8))

for result in cumulative_crps_arr:
    label_rf = f"RF - Time Trend: {result['time_trend']}, day_of_year: {result['day_of_year']}"
    label_bt = f"BT - Time Trend: {result['time_trend']}, day_of_year: {result['day_of_year']}"

    result['date_time'] = pd.to_datetime(result['date_time'])
    # Plot für Random Forest
    plt.plot(result['date_time'], result['CRPS_RF'], label=label_rf)
    
    # Plot für Bagging Trees
    plt.plot(result['date_time'], result['CRPS_BT'], label=label_bt, linestyle='--')

# Plot-Details
plt.title('Cumulative CRPS for sklearn', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative CRPS', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()

# Plot anzeigen
plt.show()

#%%

# Plot SE ---
plt.style.use('seaborn-whitegrid')
plt.rcParams['text.usetex'] = False
plt.figure(figsize=(12, 8))

for result in cumulative_se_arr:
    label_rf = f"RF - Time Trend: {result['time_trend']}, day_of_year: {result['day_of_year']}"
    label_bt = f"BT - Time Trend: {result['time_trend']}, day_of_year: {result['day_of_year']}"
    print(f"Date Time shape: {result['date_time'].shape}")
    print(f"SE_RF shape: {result['SE_RF'].shape}")
    print(f"SE_BT shape: {result['SE_BT'].shape}")

    result['date_time'] = pd.to_datetime(result['date_time'])


    # Plot für Random Forest
    plt.plot(result['date_time'], result['SE_RF'], label=label_rf)
    
    # Plot für Bagging Trees
    plt.plot(result['date_time'], result['SE_BT'], label=label_bt, linestyle='--')

# Plot-Details
plt.title('Cumulative SE for sklearn', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative SE', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()

# Plot anzeigen
plt.show()

#%%
# Plot ------
fml = ['hour_int', 'weekday_int', 'holiday']
PLOT = True
df = df_orig.copy()

if PLOT:
    fig, (ax0, ax1) = plt.subplots(nrows=1, 
                        ncols=2, 
                        figsize=set_size(fraction=1.9,
                                            subplots=(1,2)))

    xticks = [df[df.year == 2018].values[0][0],
            df[df.year == 2019].values[0][0],
            df[df.year == 2020].values[0][0],
            df[df.year == 2021].values[0][0],
            df[df.year == 2022].values[0][0],
            df[df.year == 2023].values[0][0]]

    xlabels = ['2018', '2019', '2020', '2021', '2022', '2023', '2024']


    ax0.plot(df['date'], df['load'], color=blue)
    ax0.set_xlabel('Date')
    ax0.set_ylabel('Energy Demand')
    ax0.set_ylim(29000,86050)
    ax0.set_title("Full Data Set")

    fill_yl = 30000
    fill_yu = 90000

    ax0.axvline(df['date'][0], 
            ls='--', 
            color='black', 
            alpha=.7, 
            zorder=-1)
    ax0.text(df['date'][0], 
            82050, 
            s='Training')
    ax0.axvline(df[(df.date_time >= "2022")&(df.date_time < "2023")]['date'].values[0], 
            ls='--', 
            color='black', 
            alpha=.7,
            zorder=-1)
    ax0.text(df[(df.date_time >= "2022")&(df.date_time < "2023")]['date'].values[0], 
            82050, 
            s='Test 1')
    ax0.axvline(df[df.date_time >= "2023"]['date'].values[0], 
            ls='--', 
            color='black', 
            alpha=.7,
            zorder=-1)
    ax0.text(df[df.date_time >= "2023"]['date'].values[0], 
            82000, 
            s='Test 2')

    df[fml+['load', 'hour_week_int']].groupby(['hour_week_int']).mean()['load'].plot(ax=ax1, color=blue)
    ax1.set_xlabel("Weekly Hour")
    ax1.set_ylabel("Energy Demand")
    ax1.set_title("Average Week")
    ax1.set_ylim(29000,86000)

#%%
#=================================
# PLOT ---- 
# Average Week plot ---
# Berechnen des Durchschnittlichen Energieverbrauchs pro Stunde der Woche
df_avg_weekly_hour = df_orig[['load', 'hour_week_int']].groupby(['hour_week_int']).mean()['load']

# Erstellen des Plots
fig, ax1 = plt.subplots(figsize=(12, 6), dpi = 500)
df_avg_weekly_hour.plot(ax=ax1, color='blue')

# Achsentitel und Plot-Eigenschaften
ax1.set_xlabel("Weekly Hour", fontsize=14)
ax1.set_ylabel("Energy Load", fontsize=14)
ax1.set_title("Average Week", fontsize=16)
ax1.set_ylim(29000, 86000)  
ax1.set_xlim(-8, 175)  
ax1.tick_params(axis='x', colors='black', length=0, width=0)  # Ticks für x-Achse entfernen
ax1.tick_params(axis='y', colors='black', length=0, width=0)  # Ticks für y-Achse entfernen

for spine in ax1.spines.values():
    spine.set_edgecolor('gray')


weekend_start = 120  # Freitag, 18:00 Uhr (Stunde 120)
weekend_end = 168    # Sonntag, 23:00 Uhr (Stunde 168)

# Hinzufügen der vertikalen Linien
ax1.axvline(x=weekend_start, color='green', linestyle='--', label='Weekend starts')
ax1.axvline(x=weekend_end, color='red', linestyle='--', label='Weekend ends')
ax1.grid(True)
ax1.legend()

# Layout anpassen und den Plot anzeigen
plt.tight_layout()
plt.show()


#%%
#=================================
# Plot ---
# Plot Average Week ----

# Berechnen des Durchschnittlichen Energieverbrauchs pro Stunde der Woche
df_avg_weekly_hour = df_orig[['load', 'hour_week_int']].groupby(['hour_week_int']).mean()['load']

# Erstellen des Plots
#fig, ax1 = plt.subplots(figsize=(10, 6), dpi=500)
fig, ax1 = plt.subplots(figsize=(12, 6), dpi=500)


# Plot der Durchschnittlichen Last pro Stunde der Woche
df_avg_weekly_hour.plot(ax=ax1, color='blue', label = "Load")

# Achsentitel und Plot-Eigenschaften
ax1.set_xlabel("Weekly Hour", fontsize=14)
ax1.set_ylabel("Energy Load", fontsize=14)
ax1.set_title("Average Week", fontsize=16, y=1.1)
ax1.set_ylim(29000, 86000)
ax1.set_xlim(-8, 175)
ax1.tick_params(axis='x', colors='black', length=0, width=0)  # Ticks für x-Achse entfernen
ax1.tick_params(axis='y', colors='black', length=0, width=0)  # Ticks für y-Achse entfernen

# Den Rand um den Plot herum in Grau setzen
for spine in ax1.spines.values():
    spine.set_edgecolor('gray')

# Vertikale Linien für "Weekend Start" und "Weekend End" hinzufügen
weekend_start = 120  # Freitag, 18:00 Uhr (Stunde 120)
weekend_end = 168    # Sonntag, 23:00 Uhr (Stunde 168)

# Hinzufügen der vertikalen Linien
ax1.axvline(x=weekend_start, color='green', linestyle='--', label='Weekend starts')
ax1.axvline(x=weekend_end, color='red', linestyle='--', label='Weekend ends')
ax1.grid(True)
ax1.legend()

# Manuelle Anpassung der x-Achse
ticks = [0, 24, 48, 72, 96, 120, 144]  # Stunden, bei denen die Wochentage beginnen
labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']  # Wochentage

# Setze die x-Achse Ticks und Labels
ax1.set_xticks(ticks)

# Füge eine zweite x-Achse hinzu, die unten liegt
ax2 = ax1.twiny()

# Den Rand für ax2 auch auf Grau setzen
for spine in ax2.spines.values():
    spine.set_edgecolor('lightgray')

# Setze die x-Achse der zweiten Achse (ax2) so, dass sie den Stundenbereich von 0 bis 168 umfasst
ax2.set_xlim(ax1.get_xlim())  # Die x-Achse bleibt gleich
ax2.set_xticks([12, 36, 60, 84, 108, 132, 156])  # Die Mitte jeder 24-Stunden-Periode
ax2.set_xticklabels(labels, fontsize=12, color='black')

# Layout anpassen und den Plot anzeigen
plt.tight_layout()

plt.show()

#%%


# %%
#=================================
# PLOT ----
# Plots the Timeseries data
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(plt.rcParamsDefault)
df_orig['date_time'] = pd.to_datetime(df_orig['date_time'])

# Zeitpunkt des Fixen Train-Test-Splits
tp_start = pd.Timestamp("2022-01-01")

# Trainings- und Testdaten filtern
train_data = df_orig[df_orig['date_time'] < tp_start]
test_data = df_orig[df_orig['date_time'] >= tp_start]

# Exakte Farben und Transparenz
train_color = '#1f77b4'  # Oder #87CEFA (falls Hex benötigt)
test_color = '#1f77b4'     # Oder #90EE90 (falls Hex benötigt)
alpha_train = 0.7  # GLEICHE Transparenz wie im Expanding Window Plot
alpha_test = 0.7  # GLEICHE Transparenz wie im Expanding Window Plot
linewidth = 2  # GLEICHE Linienstärke für Konsistenz

# Einzelnen Plot erstellen
plt.figure(figsize=(14, 6), dpi = 500)
plt.plot(df_orig['date_time'], df_orig['load'], color='black', alpha=0.4)
plt.plot(train_data['date_time'], train_data['load'], color=train_color, alpha=alpha_train, linewidth=linewidth)
plt.plot(test_data['date_time'], test_data['load'], color=test_color, alpha=alpha_test, linewidth=linewidth)
plt.title('Energy data', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Energy Load', fontsize=16)
#plt.legend(fontsize=14, loc='upper left')
plt.xticks(fontsize=12)  # Increase the font size of the x-axis ticks
plt.yticks(fontsize=12)  

plt.grid(True)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('lightgrey')
ax.spines['bottom'].set_color('lightgrey')


plt.tight_layout()
plt.show()




# %%
#=================================
# Plot ----
# Durchschnittlicher Load pro Stunde im Janura (2023) ---

# Sicherstellen, dass 'date_time' ein Timestamp ist
df_orig['date_time'] = pd.to_datetime(df_orig['date_time'])

# Nur Daten aus Januar 2023 auswählen
df_jan = df_orig[df_orig['date_time'].dt.month == 1]
df_jan = df_jan[df_jan['date_time'].dt.year == 2023]  

# Mittelwert der Last pro Stunde NUR für Januar 2023 berechnen
hour_avg_jan = df_jan.groupby('hour_int')['load'].mean()

plt.figure(figsize=(8, 4))
sns.lineplot(x=hour_avg_jan.index, y=hour_avg_jan.values, marker='o', color="darkblue")

plt.xlabel("Stunde des Tages")
plt.ylabel("Durchschnittliche Last")
plt.title("Durchschnittliche Last pro Stunde (Januar 2023)")
plt.xticks(range(24))
plt.grid(alpha=0.7)
plt.show()

#%%
#=================================
# Plot ----
# Zeigt den Fixed-Train-Test Split
# wie im Expanding-Window-Ansatz:

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(plt.rcParamsDefault)
df_orig['date_time'] = pd.to_datetime(df_orig['date_time'])

# Zeitpunkt des Fixen Train-Test-Splits
tp_start = pd.Timestamp("2022-01-01")

# Trainings- und Testdaten filtern
train_data = df_orig[df_orig['date_time'] < tp_start]
test_data = df_orig[df_orig['date_time'] >= tp_start]

# Exakte Farben und Transparenz
train_color = 'lightskyblue'  # Oder #87CEFA (falls Hex benötigt)
test_color = 'lightgreen'     # Oder #90EE90 (falls Hex benötigt)
alpha_train = 0.6  # GLEICHE Transparenz wie im Expanding Window Plot
alpha_test = 0.9  # GLEICHE Transparenz wie im Expanding Window Plot
linewidth = 2  # GLEICHE Linienstärke für Konsistenz

# Einzelnen Plot erstellen
plt.figure(figsize=(14, 6), dpi = 500)
plt.plot(df_orig['date_time'], df_orig['load'], color='black', alpha=0.4)
plt.plot(train_data['date_time'], train_data['load'], color=train_color, alpha=alpha_train, linewidth=linewidth, label='train data')
plt.plot(test_data['date_time'], test_data['load'], color=test_color, alpha=alpha_test, linewidth=linewidth, label='test data')
plt.axvline(tp_start, color='red', linestyle='-', linewidth=2, label='train-test split', alpha = 1.0)
plt.title('Fixed Train-Test Split', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Energy Load', fontsize=16)
plt.legend(fontsize=14, loc='upper left')
plt.xticks(fontsize=12)  # Increase the font size of the x-axis ticks
plt.yticks(fontsize=12)  

plt.grid(True)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('lightgrey')
ax.spines['bottom'].set_color('lightgrey')


plt.tight_layout()
plt.show()

#%%



#%%
#=================================
# Plot ----
# Durchschnittlier Load für jeden Januar für Jahr 2018, 2019, 2020, ..., 2023 ----

# Sicherstellen, dass 'date_time' als Timestamp erkannt wird
df_orig['date_time'] = pd.to_datetime(df_orig['date_time'])

# Farben für die verschiedenen Jahre
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

plt.figure(figsize=(10, 6))

# Loop über die Jahre 2018–2023
for i, year in enumerate(range(2018, 2024)):
    df_year = df_orig[df_orig['date_time'].dt.year == year]  # Daten für das Jahr filtern
    hour_avg = df_year.groupby('hour_int')['load'].mean()    # Mittelwert pro Stunde berechnen

    # Plotten mit individueller Farbe und Label
    sns.lineplot(x=hour_avg.index, y=hour_avg.values, marker='o', label=str(year), color=colors[i])

# Achsentitel & Design
plt.xlabel("Stunde des Tages")
plt.ylabel("Durchschnittliche Last")
plt.title("Durchschnittliche Last pro Stunde (Vergleich 2018–2023)")
plt.xticks(range(24))
plt.grid(alpha=0.5)
plt.legend(title="Jahr")  # Legende für die Farben
plt.show()




#%%
# PLOTS -----
# Sicherstellen, dass 'date_time' als Timestamp erkannt wird
plt.rcParams['text.usetex'] = False  
df_orig['date_time'] = pd.to_datetime(df_orig['date_time'])

# Liste der spezifischen Tage, die wir vergleichen wollen
specific_dates = ["2020-01-06", "2020-04-06", "2020-06-22", "2020-10-19"]
colors = ['blue', 'red', 'green', 'purple']  # Farben für die Tage

plt.figure(figsize=(10, 6))

# Loop über die spezifischen Tage
for i, date in enumerate(specific_dates):
    df_day = df_orig[df_orig['date_time'].dt.date == pd.to_datetime(date).date()]  # Daten filtern
    hour_avg = df_day.groupby('hour_int')['load'].mean()  # Mittelwert pro Stunde berechnen
    sns.lineplot(x=hour_avg.index, y=hour_avg.values, marker='o', label=date, color=colors[i])

# Achsentitel & Design
plt.xlabel("Hour")
plt.ylabel("Load")
plt.title("Averag Hourly Load for specific days (2020)")
plt.xticks(range(24))
plt.grid(alpha=0.5)
plt.legend(title="Date")  # Legende für die Farben
plt.show()




# %%
#=================================
# PLOTS ----
# Durchschnittliche stündliche Last für verschiedene Monate
# Sicherstellen, dass 'date_time' als Timestamp erkannt wird
df_orig['date_time'] = pd.to_datetime(df_orig['date_time'])

# Monate definieren, die wir analysieren wollen
selected_months = [1, 4, 7, 10]  # Januar, April, Juli, Oktober

# Farben für die Wochentage
weekday_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
weekday_labels = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']

# Loop über die Monate (ein separater Plot pro Monat)
for month in selected_months:
    plt.figure(figsize=(10, 6))

    # Daten für den aktuellen Monat auswählen
    df_month = df_orig[df_orig['date_time'].dt.month == month]

    # Gruppieren nach Stunde und Wochentag
    hourly_avg = df_month.groupby(['weekday_int', 'hour_int'])['load'].mean().reset_index()

    # Plot für jeden Wochentag
    for i in range(7):  # 0=Montag, 6=Sonntag
        df_weekday = hourly_avg[hourly_avg['weekday_int'] == i]
        sns.lineplot(x=df_weekday['hour_int'], y=df_weekday['load'], marker='o', label=weekday_labels[i], color=weekday_colors[i])

    # Plot-Details
    plt.xlabel("Stunde des Tages")
    plt.ylabel("Durchschnittliche Last")
    plt.title(f"Durchschnittliche stündliche Last im Monat {pd.to_datetime(f'2020-{month}-01').strftime('%B')}")
    plt.xticks(range(24))
    plt.grid(alpha=0.5)
    plt.legend(title="Wochentag")
    plt.show()





#%%
#=================================
# Plot ----
# Showing the Correlation between the Target and the Features

plt.rcParams['text.usetex'] = False
plt.rcdefaults() 
# Relevante Spalten für die Korrelation
features = ['holiday' ,'hour_int', 'weekday_int', 'month_int' , 'yearday', 'time_trend','load_lag1', ]
target = 'load'

# Datensatz laden (angenommen, df_orig ist bereits vorhanden)
df = df_orig[features + [target]]

# Korrelationsmatrix berechnen
corr_matrix = df.corr()

# Maske für die obere Dreiecksmatrix erstellen (nur obere Hälfte ohne Diagonale)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Heatmap plotten
plt.figure(figsize=(8, 6), dpi = 500)
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
#plt.title("Correlationmatrix")
plt.show()



# %%
# ======================================
# Alle Datenpunkte vom Testsample 
# vergleiche predictions
# Kein Bootstrapping
# mtry = p oder mtry = 1
from sklearn.tree import plot_tree
from collections import Counter

plt.rcParams['text.usetex'] = False

N_TREES = 1

SEED = 7531


crps_arr = []
se_arr = []
mse_arr = []
mae_arr = []
individual_crps_arr = []
individual_se_arr = []
individual_ae_arr = []

# Gewichte
w_hat_arr = []
q_hat_arr = []
quant_results = []


rf_models = []

df_orig.dropna(inplace=True)

base_fml = ['hour_int', 'weekday_int', 'holiday']
fmls = []
fmls.append(base_fml + ['yearday', 'load_lag1']) 


for fml in fmls:
    print(fml)
    print(f"Verwendete Formel: {fml}")

    df, _, _ = prep_energy(df=df_orig, **data_encoding)
    
    COMBINED_TEST_PERIOD = True

    if COMBINED_TEST_PERIOD:
        tp_start = "2022-01-01 00:00:00"
        
        dat_train = df.set_index("date_time")[:tp_start]
        dat_test = df.set_index("date_time")[tp_start:]
        
        y_train = dat_train['load'].values
        y_test = dat_test['load'].values
        
        dat_train.drop(columns=['load'], inplace=True)
        dat_test.drop(columns=['load'], inplace=True)
        
        X_train = dat_train[fml]
        X_test = dat_test[fml]
    else:
        tp_start1 = "2022-01-01 00:00:00"
        tp_start2 = "2023-01-01 00:00:00"

        # test_period = 2022
        dat_train = df.set_index("date_time")[:tp_start1]
        dat_test1 = df.set_index("date_time")[tp_start1:tp_start2][:-1]
        dat_test2 = df.set_index("date_time")[tp_start2:]

        y_train = dat_train['load'].values
        y_test = dat_test1['load'].values
        y_test2 = dat_test2['load'].values

        dat_train.drop(columns=['load'], inplace=True)
        dat_test1.drop(columns=['load'], inplace=True)
        dat_test2.drop(columns=['load'], inplace=True)

        X_train = dat_train[fml]
        X_test = dat_test1[fml]
        X_test2 = dat_test2[fml]
    
    if dat_train.iloc[-1].name == dat_test.iloc[0].name:
        dat_test = dat_test.iloc[1:]
    
    if y_train[-1] == y_test[0]:
        y_test = y_test[1:]  # Entferne den ersten Wert aus y_test
        X_test = X_test[1:] 

       
    print('Number of features p: ', len(fml))

    first_y_train = y_train[0]  # Erster Wert von y_train
    last_y_train = y_train[-1]  # Letzter Wert von y_train
    first_y_test = y_test[0]    # Erster Wert von y_test
    last_y_test = y_test[-1]    # Letzter Wert von y_test


    print(f"First value of y_train: {first_y_train}")
    print(f"Last value of y_train: {last_y_train}")
    print(f"First value of y_test: {first_y_test}")
    print(f"Last value of y_test: {last_y_test}")

    index_to_predict = 0  # Wählen einen Index aus, um eine Vorhersage zu machen (z.B. der erste Punkt)
    single_point_X_test = X_test.iloc[[index_to_predict]] 

    p = len(fml)
    print("p ", p)
        # Setze Hyperparameter für Random Forest
    hyperparams = {
            'n_estimators': N_TREES,
            'random_state': SEED,
            'n_jobs': -1,
            'max_features': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_depth': None,
            'bootstrap': False
        }

   

    hyperparams['random_state'] = hyperparams['random_state'] 
    rf = RandomForestWeight(hyperparams = hyperparams)
    rf.fit(X_train, y_train)
    print("X_train: ", X_train)
    print("Shape X_train: ", X_train.shape)
    print("y_train: ", y_train)
    print("Shape y_train: ", y_train.shape)
    print("Datentypen von Features in X_train: ", X_train.dtypes)
 


    rf_models.append(rf.rf)

        #_, w_hat = rf.weight_predict(X_test)
        #y_pred, w_hat = rf.weight_predict(X_test)
    #print("w_hat", w_hat)
    #print("w_hat.shape", w_hat.shape)
    #print(single_point_X_test)

    #y_single_pred = rf.predict(single_point_X_test)
    #print("Vorhersage für den einzelnen Punkt:", y_single_pred)

    y_pred = rf.predict(X_test)
    print("Vorhersage für den einzelnen Punkt:", y_pred)

    #predictions_df = pd.DataFrame(y_pred, columns=["predictions"])
    #predictions_df.to_csv("predictions.csv", index=False)
    #print("Vorhersagen wurden in 'predictions.csv' gespeichert.")

    tree = rf.rf.estimators_[0]  # Nimm den ersten Baum im Random Forest

    # Plotten des Baumes mit einer maximalen Tiefe von 3
    fig = plt.figure(figsize=(23, 14), facecolor="white", dpi=500)
    ax = fig.add_subplot(111)
    plot_tree(tree, max_depth=2, filled=True, feature_names=X_train.columns, class_names=["Load"], rounded=False, fontsize=12, ax=ax, impurity = False)
    plt.title("Visualisierung der ersten drei Ebenen des Entscheidungsbaums")
    plt.show()

    # Jetzt die maximale Tiefe der Bäume im Random Forest ausgeben
    tree_depths = [tree.get_depth() for tree in rf.rf.estimators_]
    max_depth = max(tree_depths)
    print(f"Maximale Tiefe der Bäume im Random Forest: {max_depth}")

    # Optional: Tiefe aller Bäume im Random Forest ausgeben
    print("Tiefen der einzelnen Bäume im Random Forest:", tree_depths)

    # Anzahl der Blätter und internen Knoten pro Baum
    leaf_counts = [tree.tree_.n_leaves for tree in rf.rf.estimators_]
    internal_node_counts = [tree.tree_.node_count - tree.tree_.n_leaves for tree in rf.rf.estimators_]


    # Gesamte Anzahl der Blätter und internen Knoten
    total_leaves = sum(leaf_counts)
    total_internal_nodes = sum(internal_node_counts)

    print(f"Gesamte Anzahl der Blätter im Random Forest: {total_leaves}")
    print(f"Gesamte Anzahl der internen Knoten im Random Forest: {total_internal_nodes}")

    # Optional: Anzahl der Blätter und internen Knoten der einzelnen Bäume ausgeben
    print("Anzahl der Blätter in den einzelnen Bäumen:", leaf_counts)
    print("Anzahl der internen Knoten in den einzelnen Bäumen:", internal_node_counts)

    # Feature-Split-Häufigkeiten für den gesamten Random Forest
    feature_split_counts = Counter()

    # Iteriere über die Bäume im Random Forest
    for tree in rf.rf.estimators_:
        # Extrahiere die Feature-Indizes für die Splits des Entscheidungsbaums
        feature_indices = tree.tree_.feature
        
        # Filtere die Indizes der Blätter heraus (Blätter haben den Wert -2)
        feature_indices = feature_indices[feature_indices != -2]
        
        # Zähle die Vorkommen der Features, die für Splits verwendet wurden
        feature_split_counts.update(feature_indices)

    # Zeige die Häufigkeit der Splits pro Feature an
    print("Feature-Split-Häufigkeiten:")
    for feature_idx, count in feature_split_counts.items():
        feature_name = X_train.columns[feature_idx]
        print(f"Feature '{feature_name}' wurde {count} mal gesplittet.")
    
        #y_pred = rf

#%%
# Predictions in ranger
pred_r = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/predictions_r_mtryp.csv")
# predictions in quantregForest
pred_s = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/predictions_s_mtryp.csv")
#%%
pred_r
#%%
pred_s
#%%
diff = pred_r['x'] - pred_s['predictions']

# Zeige die Differenz an
print(f"Die Differenz zwischen den Vorhersagen:\n{diff}")
diff_count = (diff != 0).sum()

# Ausgeben, wie oft sich die Vorhersagen unterscheiden
print(f"Die Vorhersagen unterscheiden sich an {diff_count} Stellen.")
# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_r = mean_absolute_error(y_test, pred_r['x'])
mae_s = mean_absolute_error(y_test, pred_s['predictions'])

# MSE berechnen
mse_r = mean_squared_error(y_test, pred_r['x'])
mse_s = mean_squared_error(y_test, pred_s['predictions'])
#%%
mae_r
#%%
mae_s
#%%
mse_r
#%%
mse_s
#%%
#=================================
# Plot ----
# Matching vs. Different Predictions
# Plotting the points were the predictions are similar and where there are different --

plt.rcParams['text.usetex'] = False
plt.style.use('seaborn-whitegrid')

# Erstelle eine Maske für unterschiedliche Vorhersagen
mask_diff = diff != 0

# Erstelle die x-Achse basierend auf der Länge von y_test
x_values = np.arange(len(y_test))

# Erstelle den Plot
plt.figure(figsize=(14, 6), dpi = 500)
plt.scatter(x_values[~mask_diff], y_test[~mask_diff], color='green', label='similar predictions', alpha=0.6)
plt.scatter(x_values[mask_diff], y_test[mask_diff], color='blue', label='different predictions', alpha=0.6)
plt.title('Matching vs Different Predictions', fontsize = 16, y = 1.1)
plt.xlabel('Time Step', fontsize = 14)
plt.ylabel('Energy Load', fontsize = 14)
plt.xticks(fontsize=12)  # Increase the font size of the x-axis ticks
plt.yticks(fontsize=12) 
plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)
plt.show()

#%%
#=================================
# Plot ----
# Absolute Difference in Predictions between ranger and sklearn
# Difference in Predictions (ranger vs sklearn)
# Berechnung der absoluten Differenzen

# Berechnung der absoluten Differenz
abs_diff = abs(pred_r.iloc[:, 0] - pred_s.iloc[:, 0])

# Erstelle die x-Achse basierend auf der Länge der Daten
x_values = np.arange(len(abs_diff))

# Plot der absoluten Differenzen mit Punkten
plt.figure(figsize=(14, 6), dpi=500)
plt.scatter(x_values, abs_diff, color='red', label='Absolute Prediction Difference', alpha=0.6, s=10)
plt.title('Absolute Difference in Predictions between ranger and sklearn', fontsize=16)
plt.xlabel('Time Step', fontsize=14)
plt.ylabel('Absolute Difference', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Berechnung von Mittelwert und Maximum der absoluten Differenzen
mean_abs_diff = abs_diff.mean()
max_abs_diff = abs_diff.max()

# Ausgabe der Werte
print(f"Mean Absolute Difference: {mean_abs_diff:.4f}")
print(f"Max Absolute Diff: {max_abs_diff:.4f}")

#%%
diff = pd.DataFrame(diff, columns=['diff'])

# X_test und diff zusammenführen
X_diff = pd.concat([X_test.reset_index(drop=True), diff], axis=1)
X_diff['diff_present'] = (X_diff['diff'] != 0).astype(int)

# Überprüfen, ob die Spalte korrekt hinzugefügt wurde
print(X_diff.head())
#%%


#%%
#=================================
# Plot ---
# Barplot der Häufigkeit von Differenz (ja/nein) nach Stunde
import seaborn as sns

count_diff = X_diff.groupby(['hour_int', 'diff_present']).size().unstack(fill_value=0)

# Berechne den Prozentsatz für jede Ausprägung von hour_int
for hour in count_diff.index:
    no_diff_count = count_diff.loc[hour, 0]  # diff_present = 0 (No Differences)
    diff_count = count_diff.loc[hour, 1]     # diff_present = 1 (Differences)
    total_count = no_diff_count + diff_count
    
    if total_count > 0:
        no_diff_percentage = (no_diff_count / total_count) * 100
        diff_percentage = (diff_count / total_count) * 100
    else:
        no_diff_percentage = 0
        diff_percentage = 0
    
    # Ausgabe der Prozentsätze in der Konsole
    print(f"Hour {hour}: no differences = {no_diff_percentage:.1f}%, differences = {diff_percentage:.1f}%")


plt.figure(figsize=(14, 6), dpi=500)
sns.countplot(x=X_diff['hour_int'], hue=X_diff['diff_present'])
plt.xlabel('Hour', fontsize=14)
plt.ylabel('Difference', fontsize=14)
plt.title('Prediction Differences', fontsize=16, y=1.2)
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12)  
plt.legend(labels=["no differences", "differences"], fontsize=14, loc='upper center', 
           bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False)

plt.show()



#%%






# %%
# DecisionTreeRegressor ----
from sklearn.tree import DecisionTreeRegressor
from collections import Counter

#SEED = 7531
SEED = 42

hyperparams = {
    'random_state': SEED,
    'max_features': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_depth': None
}

# Erstelle das Entscheidungsbaum-Modell
dt = DecisionTreeRegressor(**hyperparams)

# Trainiere das Modell
dt.fit(X_train, y_train)
print("X_train: ", X_train)

# Vorhersagen treffen
y_pred = dt.predict(X_test)

feature_indices = dt.tree_.feature

# Entferne -2 Werte (dies steht für Blattknoten, die keine Splits enthalten)
split_feature_indices = feature_indices[feature_indices != -2]

# Zähle, wie oft jedes Feature gesplittet wurde
feature_split_counts = Counter(split_feature_indices)

# Anzahl der einzigartigen gesplitteten Features
unique_splits = len(feature_split_counts)

# Ausgabe der gesplitteten Features
print(f"Anzahl der gesplitteten Features: {unique_splits}")

# Detailierte Anzeige der Split-Häufigkeiten
print("Feature-Split-Häufigkeiten:")
for feature_idx, count in feature_split_counts.items():
    feature_name = X_train.columns[feature_idx]
    print(f"Feature '{feature_name}' wurde {count} mal gesplittet.")


# Speichern der Vorhersagen
predictions_df = pd.DataFrame(y_pred, columns=["predictions"])
#predictions_df.to_csv("predictions_d_mtryp_42.csv", index=False)
#print("Vorhersagen wurden in 'predictions.csv' gespeichert.")

# Visualisierung des Baums
fig = plt.figure(figsize=(23, 14), facecolor="white", dpi=500)
ax = fig.add_subplot(111)
plot_tree(dt, max_depth=2, filled=True, feature_names=X_train.columns, rounded=False, fontsize=12, ax=ax, impurity=False)
plt.title("Visualisierung der ersten drei Ebenen des Entscheidungsbaums")
plt.show()

# Maximale Tiefe des Baumes ausgeben
print(f"Maximale Tiefe des Entscheidungsbaums: {dt.get_depth()}")
print(f"Anzahl der Blätter im Entscheidungsbaum: {dt.get_n_leaves()}")
#%%
# Predictions Ranger ---
pred_r = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/predictions_r_mtryp.csv")

# Predictions Sklearn ---
pred_s = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/predictions_s_mtryp.csv")

# Predictions DecisionTree ---
#pred_s_1 = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/predictions_d_mtryp.csv")
#pred_s_1 = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/predictions_d_mtryp_7531.csv")
pred_s_1 = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/predictions_d_mtryp_42.csv")

#%%
# Die drei Predictions vergleichen ----
pred_r_rf = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/predictions_r_mtryp.csv")
pred_s_rf = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/predictions_s_mtryp.csv")
#pred_s_dt = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/predictions_d_mtryp.csv")
#pred_s_dt = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/predictions_d_mtryp_42.csv")
pred_s_dt = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/predictions_d_mtryp_7531.csv")

# DataFrame erstellen
df = pd.DataFrame({
    'pred_r_rf': pred_r_rf.iloc[:, 0],  
    'pred_s_rf': pred_s_rf.iloc[:, 0],  
    'pred_s_dt': pred_s_dt.iloc[:, 0]   
})

df['all_equal'] = (df['pred_r_rf'] == df['pred_s_rf']) & (df['pred_r_rf'] == df['pred_s_dt'])
df['r_rf__s_rf_equal'] = (df['pred_r_rf'] == df['pred_s_rf'])
df['r_rf__s_dt_equal'] = (df['pred_r_rf'] == df['pred_s_dt'])
df['r_s_dt__s_rf_equal'] = (df['pred_s_dt'] == df['pred_s_rf'])

count_all_equal = df['all_equal'].value_counts()
count_r_rf__s_rf_equal = df['r_rf__s_rf_equal'].value_counts()
count_r_rf__s_dt_equal = df['r_rf__s_dt_equal'].value_counts()
count_r_s_dt__s_rf_equal = df['r_s_dt__s_rf_equal'].value_counts()

# Ausgabe der Häufigkeit
print("Häufigkeit für all_equal:")
print(count_all_equal)
print("\nHäufigkeit für r_rf__s_rf_equal:")
print(count_r_rf__s_rf_equal)
print("\nHäufigkeit für r_rf__s_dt_equal:")
print(count_r_rf__s_dt_equal)
print("\nHäufigkeit für r_s_dt__s_rf_equal:")
print(count_r_s_dt__s_rf_equal)
#%%
print(df)
#%%
X_merged = pd.concat([X_test.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
X_merged



#%%
#=================================
# Plot Prediction Difference --- (A)
# Between the three models
# as a subplot -----
import numpy as np
import matplotlib.pyplot as plt

# Differenzen berechnen
diff_r_s_rf = pred_r_rf.iloc[:, 0] - pred_s_rf.iloc[:, 0]
diff_r_s_dt = pred_r_rf.iloc[:, 0] - pred_s_dt.iloc[:, 0]
diff_s_rf_s_dt = pred_s_rf.iloc[:, 0] - pred_s_dt.iloc[:, 0]

# x-Achse definieren
x_values = np.arange(len(diff_r_s_rf))

# Plot
fig, axs = plt.subplots(3, 1, figsize=(14, 12), dpi=500, sharex=True)

# Plot 1: ranger - sklearn RF
axs[0].scatter(x_values, diff_r_s_rf, color='green', alpha=0.6, s=10)
axs[0].axhline(0, color='black', linestyle='--', linewidth=1)
axs[0].set_title('Prediction Difference: RF ranger - RF sklearn', fontsize=14)
axs[0].set_ylabel('Difference', fontsize=12)
axs[0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: ranger - sklearn DT
axs[1].scatter(x_values, diff_r_s_dt, color='blue', alpha=0.6, s=10)
axs[1].axhline(0, color='black', linestyle='--', linewidth=1)
axs[1].set_title('Prediction Difference: RF ranger - Decision Tree sklearn', fontsize=14)
axs[1].set_ylabel('Difference', fontsize=12)
axs[1].grid(True, linestyle='--', alpha=0.6)

# Plot 3: sklearn RF - sklearn DT
axs[2].scatter(x_values, diff_s_rf_s_dt, color='orange', alpha=0.6, s=10)
axs[2].axhline(0, color='black', linestyle='--', linewidth=1)
axs[2].set_title('Prediction Difference: sklearn RF - Decision Tree sklearn', fontsize=14)
axs[2].set_xlabel('Time Step', fontsize=12)
axs[2].set_ylabel('Difference', fontsize=12)
axs[2].grid(True, linestyle='--', alpha=0.6)

# Layout anpassen
plt.tight_layout()
plt.show()


#%%
#=================================
# Plot Prediction Difference --- (B)

# Differenzen berechnen
diff_r_s_rf = pred_r_rf.iloc[:, 0] - pred_s_rf.iloc[:, 0]
diff_r_s_dt = pred_r_rf.iloc[:, 0] - pred_s_dt.iloc[:, 0]
diff_s_rf_s_dt = pred_s_rf.iloc[:, 0] - pred_s_dt.iloc[:, 0]

# x-Achse definieren
x_values = np.arange(len(diff_r_s_rf))

# Plot 1: ranger - sklearn RF
plt.figure(figsize=(14, 6), dpi=500)

plt.scatter(x_values, diff_r_s_rf, color='red', alpha=0.6, s=10)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Prediction Difference: RF ranger - RF sklearn', fontsize=16)
plt.xlabel('Time Step', fontsize=14)
plt.ylabel('Difference', fontsize=14)
plt.ylim(-4500, 4500)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plot 2: ranger - sklearn Decision Tree
plt.figure(figsize=(14, 6), dpi=500)
plt.scatter(x_values, diff_r_s_dt, color='blue', alpha=0.6, s=10)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Prediction Difference: RF ranger - Decision Tree sklearn', fontsize=16)
plt.xlabel('Time Step', fontsize=14)
plt.ylabel('Difference', fontsize=14)
plt.ylim(-4500, 4500)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plot 3: sklearn RF - sklearn Decision Tree
plt.figure(figsize=(14, 6), dpi=500)
plt.scatter(x_values, diff_s_rf_s_dt, color='orange', alpha=0.6, s=10)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Prediction Difference: RF sklearn - Decision Tree sklearn', fontsize=16)
plt.xlabel('Time Step', fontsize=14)
plt.ylabel('Difference', fontsize=14)
plt.ylim(-4500, 4500)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()




#%%

#%%
#=================================
# Plot  ---
# Erstelle die 2x2 Subplot-Gitter ---
# zeigt: Vergleich der Vorhersageübereinstimmung von drei Modellen über 24 Stunden
# HOUR ---

fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=500)
axs = axs.flatten()

fig.suptitle('Comparison of Prediction Accuracy', fontsize=16, y=1.0)

# Plot für den Fall all_equal
sns.countplot(x='hour_int', data=X_merged, hue='all_equal', ax=axs[0], 
              palette={True: 'green', False: 'red'}, hue_order=[True, False])
axs[0].set_title("RF ranger, RF sklearn and DT sklearn", fontsize=14)
axs[0].set_xlabel("Hour", fontsize=14)
axs[0].set_ylabel("Frequency", fontsize=14)
axs[0].set_ylim(0, 550)  # Y-Achse von 0 bis 550
axs[0].legend_.remove()  # Entferne die Legende aus dem Subplot

# Plot für den Fall r_rf__s_rf_equal
sns.countplot(x='hour_int', data=X_merged, hue='r_rf__s_rf_equal', ax=axs[1], 
              palette={True: 'green', False: 'red'}, hue_order=[True, False])
axs[1].set_title("RF ranger and RF sklearn", fontsize=14)
axs[1].set_xlabel("Hour", fontsize=14)
axs[1].set_ylabel("Frequency", fontsize=14)
axs[1].set_ylim(0, 550)  # Y-Achse von 0 bis 550
axs[1].legend_.remove()  # Entferne die Legende aus dem Subplot

# Plot für den Fall r_rf__s_dt_equal
sns.countplot(x='hour_int', data=X_merged, hue='r_rf__s_dt_equal', ax=axs[2], 
              palette={True: 'green', False: 'red'}, hue_order=[True, False])
axs[2].set_title("RF ranger and DT sklearn", fontsize=14)
axs[2].set_xlabel("Hour", fontsize=14)
axs[2].set_ylabel("Frequency", fontsize=14)
axs[2].set_ylim(0, 550)  # Y-Achse von 0 bis 550
axs[2].legend_.remove()  # Entferne die Legende aus dem Subplot

# Plot für den Fall r_s_dt__s_rf_equal
sns.countplot(x='hour_int', data=X_merged, hue='r_s_dt__s_rf_equal', ax=axs[3], 
              palette={True: 'green', False: 'red'}, hue_order=[True, False])
axs[3].set_title(" RF sklearn and DT sklearn ", fontsize=14)
axs[3].set_xlabel("Hour", fontsize=14)
axs[3].set_ylabel("Frequency", fontsize=14)
axs[3].set_ylim(0, 550)  # Y-Achse von 0 bis 550
axs[3].legend_.remove()  # Entferne die Legende aus dem Subplot

# Gemeinsame Legende unten mit neuen Beschriftungen
handles, labels = axs[0].get_legend_handles_labels()  # Hole die Legende aus dem ersten Plot
new_labels = ['similar predictions', 'different predictions']
fig.legend(handles, new_labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95), fontsize=14)

# Achsenticks anpassen
plt.xticks(fontsize=12)  # X-Achsenticks
plt.yticks(fontsize=12)  # Y-Achsenticks

# Tight layout für saubere Anordnung der Subplots
plt.tight_layout(rect=[0, 0, 1, 0.90])  # Platz für die Legende unten lassen
plt.show()

#%%
# Feature Weekday ---

fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=500)
axs = axs.flatten()

# Aktualisierter Titel für das Feature "weekday_int"
fig.suptitle('Comparison of Prediction Accuracy ', fontsize=16, y=1.0)

# Plot für den Fall all_equal
sns.countplot(x='weekday_int', data=X_merged, hue='all_equal', ax=axs[0], 
              palette={True: 'green', False: 'red'}, hue_order=[True, False])
axs[0].set_title("RF ranger, RF sklearn and DT sklearn", fontsize=14)
axs[0].set_xlabel("Weekday", fontsize=14)
axs[0].set_ylabel("Frequency", fontsize=14)
axs[0].set_ylim(0, 1800)  # Y-Achse von 0 bis 550
axs[0].legend_.remove()  # Entferne die Legende aus dem Subplot

# Plot für den Fall r_rf__s_rf_equal
sns.countplot(x='weekday_int', data=X_merged, hue='r_rf__s_rf_equal', ax=axs[1], 
              palette={True: 'green', False: 'red'}, hue_order=[True, False])
axs[1].set_title("RF ranger and RF sklearn", fontsize=14)
axs[1].set_xlabel("Weekday", fontsize=14)
axs[1].set_ylabel("Frequency", fontsize=14)
axs[1].set_ylim(0, 1800)  # Y-Achse von 0 bis 550
axs[1].legend_.remove()  # Entferne die Legende aus dem Subplot

# Plot für den Fall r_rf__s_dt_equal
sns.countplot(x='weekday_int', data=X_merged, hue='r_rf__s_dt_equal', ax=axs[2], 
              palette={True: 'green', False: 'red'}, hue_order=[True, False])
axs[2].set_title("RF ranger and DT sklearn", fontsize=14)
axs[2].set_xlabel("Weekday", fontsize=14)
axs[2].set_ylabel("Frequency", fontsize=14)
axs[2].set_ylim(0, 1800)  # Y-Achse von 0 bis 550
axs[2].legend_.remove()  # Entferne die Legende aus dem Subplot

# Plot für den Fall r_s_dt__s_rf_equal
sns.countplot(x='weekday_int', data=X_merged, hue='r_s_dt__s_rf_equal', ax=axs[3], 
              palette={True: 'green', False: 'red'}, hue_order=[True, False])
axs[3].set_title("RF sklearn and DT sklearn", fontsize=14)
axs[3].set_xlabel("Weekday", fontsize=14)
axs[3].set_ylabel("Frequency", fontsize=14)
axs[3].set_ylim(0, 1800)  # Y-Achse von 0 bis 550
axs[3].legend_.remove()  # Entferne die Legende aus dem Subplot

# Gemeinsame Legende unten mit neuen Beschriftungen
handles, labels = axs[0].get_legend_handles_labels()  # Hole die Legende aus dem ersten Plot
new_labels = ['similar predictions', 'different predictions']
fig.legend(handles, new_labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95), fontsize=14)

# Achsenticks anpassen
plt.xticks(fontsize=12)  # X-Achsenticks
plt.yticks(fontsize=12)  # Y-Achsenticks

# Tight layout für saubere Anordnung der Subplots
plt.tight_layout(rect=[0, 0, 1, 0.90])  # Platz für die Legende unten lassen
plt.show()
#%%
#=================================
# Plot  ---
# Holiday ----

fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=500)
axs = axs.flatten()

# Aktualisierter Titel für das Feature "holiday"
fig.suptitle('Comparison of Prediction Accuracy', fontsize=16, y=1.0)

# Plot für den Fall all_equal
sns.countplot(x='holiday', data=X_merged, hue='all_equal', ax=axs[0], 
              palette={True: 'green', False: 'red'}, hue_order=[True, False])
axs[0].set_title("RF ranger, RF sklearn and DT sklearn", fontsize=14)
axs[0].set_xlabel("Holiday", fontsize=14)
axs[0].set_ylabel("Frequency", fontsize=14)
axs[0].set_ylim(0, 12000)  # Y-Achse von 0 bis 550
axs[0].legend_.remove()  # Entferne die Legende aus dem Subplot

# Plot für den Fall r_rf__s_rf_equal
sns.countplot(x='holiday', data=X_merged, hue='r_rf__s_rf_equal', ax=axs[1], 
              palette={True: 'green', False: 'red'}, hue_order=[True, False])
axs[1].set_title("RF ranger and RF sklearn", fontsize=14)
axs[1].set_xlabel("Holiday", fontsize=14)
axs[1].set_ylabel("Frequency", fontsize=14)
axs[1].set_ylim(0, 12000)  # Y-Achse von 0 bis 550
axs[1].legend_.remove()  # Entferne die Legende aus dem Subplot

# Plot für den Fall r_rf__s_dt_equal
sns.countplot(x='holiday', data=X_merged, hue='r_rf__s_dt_equal', ax=axs[2], 
              palette={True: 'green', False: 'red'}, hue_order=[True, False])
axs[2].set_title("RF ranger and DT sklearn", fontsize=14)
axs[2].set_xlabel("Holiday", fontsize=14)
axs[2].set_ylabel("Frequency", fontsize=14)
axs[2].set_ylim(0, 12000)  # Y-Achse von 0 bis 550
axs[2].legend_.remove()  # Entferne die Legende aus dem Subplot

# Plot für den Fall r_s_dt__s_rf_equal
sns.countplot(x='holiday', data=X_merged, hue='r_s_dt__s_rf_equal', ax=axs[3], 
              palette={True: 'green', False: 'red'}, hue_order=[True, False])
axs[3].set_title("RF sklearn and DT sklearn", fontsize=14)
axs[3].set_xlabel("Holiday", fontsize=14)
axs[3].set_ylabel("Frequency", fontsize=14)
axs[3].set_ylim(0, 12000)  # Y-Achse von 0 bis 550
axs[3].legend_.remove()  # Entferne die Legende aus dem Subplot

# Gemeinsame Legende unten mit neuen Beschriftungen
handles, labels = axs[0].get_legend_handles_labels()  # Hole die Legende aus dem ersten Plot
new_labels = ['similar predictions', 'different predictions']
fig.legend(handles, new_labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95), fontsize=14)

# Achsenticks anpassen
plt.xticks(fontsize=12)  # X-Achsenticks
plt.yticks(fontsize=12)  # Y-Achsenticks

# Tight layout für saubere Anordnung der Subplots
plt.tight_layout(rect=[0, 0, 1, 0.90])  # Platz für die Legende unten lassen
plt.show()



#%%
#=================================
# Plot  ---
# Load Lag1

fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=500)
axs = axs.flatten()

# Suptitle
fig.suptitle('Comparison of Prediction Accuracy', fontsize=16)

# Vergleichs-Spalten
comparison_cols = [
    'all_equal',
    'r_rf__s_rf_equal',
    'r_rf__s_dt_equal',
    'r_s_dt__s_rf_equal'
]

# Benutzerdefinierte Titel für jeden Subplot
custom_titles = [
    'RF ranger, RF sklearn and DT sklearn',
    'RF ranger and RF sklearn',
    'RF ranger and DT sklearn',
    'RF sklearn and DT sklearn'
]

# Farben
palette = {True: 'green', False: 'red'}

# Plots zeichnen
for i, col in enumerate(comparison_cols):
    sns.histplot(
        data=X_merged,
        x='load_lag1',
        hue=col,
        ax=axs[i],
        kde=False,
        palette=palette,
        hue_order=[True, False],
        bins=50,
        legend=False
    )
    axs[i].set_title(custom_titles[i], fontsize=14)
    axs[i].set_xlabel('Load_lag1', fontsize=14)
    axs[i].set_ylabel('Frequency', fontsize=14)
    axs[i].set_ylim(-0.02*500, 500)

# Manuelle Legende
legend_labels = ['similar predictions', 'different predictions']
legend_colors = ['green', 'red']
patches = [mpatches.Patch(color=col, label=lab) for col, lab in zip(legend_colors, legend_labels)]

# Legende zentriert oberhalb der Plots
fig.legend(handles=patches, loc='upper center', ncol=2, fontsize=14, bbox_to_anchor=(0.5, 0.93))

# Layout anpassen (Platz für Titel + Legende)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()


#%%
#=================================
# Plot  ---
# Yearday

# Subplots erstellen
fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=500)
axs = axs.flatten()

# Suptitle
fig.suptitle('Comparison of Prediction Accuracy', fontsize=16)

# Vergleichs-Spalten
comparison_cols = [
    'all_equal',
    'r_rf__s_rf_equal',
    'r_rf__s_dt_equal',
    'r_s_dt__s_rf_equal'
]

# Benutzerdefinierte Titel für jeden Subplot
custom_titles = [
    'RF ranger, RF sklearn and DT sklearn',
    'RF ranger and RF sklearn',
    'RF ranger and DT sklearn',
    'RF sklearn and DT sklearn'
]

# Farben
palette = {True: 'green', False: 'red'}

# Plots zeichnen
for i, col in enumerate(comparison_cols):
    sns.histplot(
        data=X_merged,
        x='yearday',
        hue=col,
        ax=axs[i],
        kde=False,
        palette=palette,
        hue_order=[True, False],
        bins=50,  # 365 Tage in ~50 Bins = ~7 Tage pro Bin
        legend=False
    )
    axs[i].set_title(custom_titles[i], fontsize=14)
    axs[i].set_xlabel('Yearday', fontsize=14)
    axs[i].set_ylabel('Frequency', fontsize=14)
    axs[i].set_ylim(-0.02*300, 300)
    axs[i].set_xlim(-1, 370)
    axs[i].set_xticks(np.arange(0, 366, 50))  # alle 50 Tage ein Tick

# Manuelle Legende
legend_labels = ['similar predictions', 'different predictions']
legend_colors = ['green', 'red']
patches = [mpatches.Patch(color=col, label=lab) for col, lab in zip(legend_colors, legend_labels)]

# Legende zentriert oberhalb der Plots
fig.legend(handles=patches, loc='upper center', ncol=2, fontsize=14, bbox_to_anchor=(0.5, 0.93))

# Layout anpassen
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()


#%%


#%%


























# %%
# Test mit dummy features welchen Wert mtry hat
from sklearn.preprocessing import OneHotEncoder


winter_time = ['2018-10-28 02:00:00',
               '2019-10-27 02:00:00',
               '2020-10-25 02:00:00',
               '2021-10-31 02:00:00',
               '2022-10-30 02:00:00',
               '2023-10-29 02:00:00']

df_orig = load_energy('/home/siefert/projects/Masterarbeit/dat_energy/rf_data_1823_clean.csv')
df_orig = df_orig[df_orig.load > 0]
df_orig = df_orig[df_orig.load < 82000]
df_orig = df_orig[~df_orig.date_time.isin(winter_time)]

df_orig['load_lag1'] = df_orig['load'].shift(1)

# ============ 🔁 One-Hot-Encoding der month_int-Spalte =============
df_orig['month_int'] = df_orig['month_int'].astype(int)

ohe = OneHotEncoder(sparse=False, drop='first')
month_encoded = ohe.fit_transform(df_orig[['month_int']])
month_encoded_cols = [f"month_{i+2}" for i in range(month_encoded.shape[1])]  # Start bei 2 wegen drop='first'
df_month_ohe = pd.DataFrame(month_encoded, columns=month_encoded_cols, index=df_orig.index)
print(df_month_ohe)

df_orig = pd.concat([df_orig, df_month_ohe], axis=1)

# ============ Konfiguration der Feature-Auswahl =============
data_encoding = dict(time_trend=False,
                     time_trend_sq=False,
                     cat_features=False,
                     fine_resolution=False,
                     sin_cos_features=False,
                     last_obs=False)

base_fml = ['hour_int', 'weekday_int', 'holiday']
month_ohe_cols = df_month_ohe.columns.tolist()

fmls = []
fmls.append(base_fml + ['month_int'])  # Variante ohne OHE
fmls.append(base_fml + month_ohe_cols)  # Variante mit OHE
# weitere Kombinationen möglich

SEED = 7531
N_TREES = 100  # Default sklearn

df_orig.dropna(inplace=True)

for fml in fmls:
    print("--------------------------------------------------")
    print(f"Verwendete Features: {fml}")
    print('Anzahl Features p: ', len(fml))
    MTRY_RF = int(len(fml) / 3)
    print('mtry BT = p: ', len(fml))
    print('mtry RF = p/3 :',  MTRY_RF)

    df, _, _ = prep_energy(df=df_orig, **data_encoding)

    tp_start = "2022-01-01 00:00:00"
    dat_train = df.set_index("date_time")[:tp_start]
    dat_test = df.set_index("date_time")[tp_start:]

    y_train = dat_train['load'].values
    y_test = dat_test['load'].values
    dat_train.drop(columns=['load'], inplace=True)
    dat_test.drop(columns=['load'], inplace=True)

    X_train = dat_train[fml]
    X_test = dat_test[fml]

    hyperparams = dict(n_estimators=N_TREES,
                       random_state=SEED,
                       n_jobs=-1,
                       max_features=MTRY_RF,
                       min_samples_split=2,
                       min_samples_leaf=1,
                       max_depth=None)

    rf = RandomForestWeight(hyperparams=hyperparams)
    rf.fit(X_train, y_train)

    # Bagged Trees
    hyperparams_bt = hyperparams.copy()
    hyperparams_bt["max_features"] = len(fml)
    bt = RandomForestWeight(hyperparams=hyperparams_bt)
    bt.fit(X_train, y_train)

    _, w_hat_rf = rf.weight_predict(X_test)
    _, w_hat_bt = bt.weight_predict(X_test)

    y_pred_rf = rf.predict(X_test)
    y_pred_bt = bt.predict(X_test)

    print("Ergebnisse (RF vs BT):")
    print("y_pred_rf:", y_pred_rf[:5])
    print("y_pred_bt:", y_pred_bt[:5])
    print("Wichtig: Unterschiedlicher mtry → Strukturunterschiede möglich")
