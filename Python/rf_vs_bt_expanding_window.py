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


def calculate_individual_ae(y_test, y_pred):

    individual_ae = ae(y_test, y_pred)
    print('shape se_values ', individual_ae.shape)
    
    return individual_ae

#%%
output_folder = r"/home/siefert/projects/Masterarbeit/Data/Results/Python_Results/python_res_expanding_window_different_mtry"
os.makedirs(output_folder, exist_ok=True)

#output_folder = r"/home/siefert/projects/Masterarbeit/Data/Results/Python_Results/python_res_linear_model"
#os.makedirs(output_folder, exist_ok=True)
#%%
def save_results_expanding_window(individual_crps_arr, individual_se_arr, individual_ae_arr, dat_test, prefix="python_res_different_mtry_expanding/"):

    '''
    Funktion um die Ergebnisse von sklearn zu speichern
    
    Output: CSV-Dateien, die die CRPS- sowie die SE-Werte für jeden Datenpunkt im Testdatensatz enthalten.
    '''
    os.makedirs(prefix, exist_ok=True)  # Erstelle das Verzeichnis, falls es nicht existiert
    
    for crps_result, se_result, ae_result in zip(individual_crps_arr, individual_se_arr, individual_ae_arr):
        # Extrahiere Parameter aus den Ergebnissen
        time_trend_part = 'tt' if crps_result['time_trend'] == 'yes' else 'nott'
        day_part = 'day' if crps_result['day_of_year'] == 'yes' else 'month'
        mtry_value = f"mtry{crps_result['m_try']}"
        #date_time_str = crps_result['date_time'].iloc[0].strftime("%Y-%m-%d")
        month_end = crps_result['date_time'].max() + pd.offsets.MonthEnd(0)
        month_end_str = month_end.strftime('%Y-%m-%d')
        
        # Dynamischer Dateiname
        save_name = f"sklearn_{time_trend_part}_{day_part}_{mtry_value}_{month_end_str}.csv"
        
        # Erstelle einen DataFrame
        df = pd.DataFrame({
            'date_time': crps_result['date_time'],   
            'crps': crps_result['CRPS'],  # CRPS-Werte
            'se': se_result['SE'],      
            'ae': ae_result['AE']  # SE-Werte
        })
        
        # Speichere die Ergebnisse als CSV
        df.to_csv(save_name, index=False)
        
        print(f"Saved: {save_name}")
   

   
#%%
def save_results_different_mtry(individual_crps_arr, individual_se_arr, individual_ae_arr ,dat_test, prefix="/home/siefert/projects/Masterarbeit/Results/Python/python_res_expanding_window_different_Mtry_a/"):
    """
    Speichert die Ergebnisse (CRPS und SE) für jeden Testdatensatzpunkt in CSV-Dateien.
    
    Parameters:
        individual_crps_arr (list): Liste mit CRPS-Ergebnissen für verschiedene m_try-Werte und Zeitpunkte.
        individual_se_arr (list): Liste mit SE-Ergebnissen für verschiedene m_try-Werte und Zeitpunkte.
        individual_ae_arr (list): Liste mit AE-Ergebnissen für verschiedene m_try-Werte und Zeitpunkte.
        prefix (str): Verzeichnis, in dem die Dateien gespeichert werden sollen.
    """
      # Erstelle das Verzeichnis, falls es nicht existiert
    
    for crps_result, se_result, ae_result in zip(individual_crps_arr, individual_se_arr, individual_ae_arr):
        # Extrahiere Parameter aus den Ergebnissen
        time_trend_part = 'tt' if crps_result['time_trend'] == 'yes' else 'nott'
        day_part = 'day' if crps_result['day_of_year'] == 'yes' else 'month'
        mtry_value = f"mtry{crps_result['m_try']}"
        #date_time_str = crps_result['date_time'].iloc[0].strftime("%Y-%m-%d")
        month_end = crps_result['date_time'].max() + pd.offsets.MonthEnd(0)
        month_end_str = month_end.strftime('%Y-%m-%d')
        
        # Dynamischer Dateiname
        save_name = f"sklearn_{time_trend_part}_{day_part}_{mtry_value}_{month_end_str}.csv"
        
        # Erstelle einen DataFrame
        df = pd.DataFrame({
            'date_time': crps_result['date_time'],   
            'crps': crps_result['CRPS'],  # CRPS-Werte
            'se': se_result['SE'],      
            'ae': ae_result['AE']  # SE-Werte
        })
        
        # Speichere die Ergebnisse als CSV
        df.to_csv(save_name, index=False)
        
        print(f"Saved: {save_name}")

#%%
def save_results_linear_model(individual_crps_arr, individual_se_arr, individual_ae_arr ,dat_test, prefix="/home/siefert/projects/Masterarbeit/Data/Results/Python_Results/python_res_linear_model/"):
    """
    Speichert die Ergebnisse (CRPS und SE) für jeden Testdatensatzpunkt in CSV-Dateien.
    
    Parameters:
        individual_crps_arr (list): Liste mit CRPS-Ergebnissen für verschiedene m_try-Werte und Zeitpunkte.
        individual_se_arr (list): Liste mit SE-Ergebnissen für verschiedene m_try-Werte und Zeitpunkte.
        prefix (str): Verzeichnis, in dem die Dateien gespeichert werden sollen.
    """
      # Erstelle das Verzeichnis, falls es nicht existiert
    
    for i, crps_result in enumerate(individual_crps_arr):
        # Extrahiere Parameter aus den Ergebnissen
        time_trend_part = 'tt' if crps_result['time_trend'] == 'yes' else 'nott'
        day_part = 'day' if crps_result['day_of_year'] == 'yes' else 'month'
        
        crps_result['date_time'] = pd.to_datetime(crps_result['date_time'])
        
        month_end = crps_result['date_time'].max() + pd.offsets.MonthEnd(0)
        month_end_str = month_end.strftime('%Y-%m-%d')
        
        # Dynamischer Dateiname
        save_name = f"sklearn_linear_{time_trend_part}_{day_part}_{month_end_str}.csv"
        
        # Erstelle einen DataFrame
        df = pd.DataFrame({
            'date_time': crps_result['date_time'], 
            'crps': crps_result['CRPS'],  # CRPS-Werte
            'se': individual_se_arr[i]['SE'],       # SE-Werte
            'ae': individual_ae_arr[i]['AE'],
        })
        
        # Speichere die Ergebnisse als CSV
        df.to_csv(save_name, index=False)
        
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

#df_orig['load_lag1'] = df_orig['load'].shift(1)

data_encoding = dict(time_trend=False,
                     time_trend_sq=False,
                     cat_features=False,
                     fine_resolution=False,
                     sin_cos_features=False,
                     last_obs=False)
    
base_fml = ['hour_int', 'weekday_int', 'holiday']

fmls = []
fmls.append(base_fml + ['month_int']) # without timetrend
fmls.append(base_fml + ['yearday']) # without timetrend
fmls.append(base_fml + ['time_trend', 'month_int'])
fmls.append(base_fml + ['time_trend', 'yearday'])

#%%
# Expanding Window Ansatz ------
# RF Model ---
from dateutil.relativedelta import relativedelta
df_orig['date_time'] = pd.to_datetime(df_orig['date_time'])

start_date = pd.Timestamp("2022-01-01 00:00:00")
end_date = pd.Timestamp("2023-11-01 00:00:00")
time_intervals = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monatsanfänge

# Ergebnislisten initialisieren
crps_arr = []
se_arr = []
mse_arr = []
mae_arr = []
individual_crps_arr = []
individual_se_arr = []
individual_ae_arr = []

N_TREES = 100

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
    for current_date in time_intervals:
        print(f"Processing data up to {current_date}")

    
        train_end_date = current_date - relativedelta(hours=1)
        print("train_end_date: " ,train_end_date)
        
        dat_train = df_orig[df_orig['date_time'] <= train_end_date].copy()
        print(dat_train.head(10))
        print(dat_train.tail(10))
    
        test_start_date = current_date
        print("test_start_date: ", test_start_date)
        test_end_date = current_date + relativedelta(months=1) - relativedelta(hours=1)
        print("test_end_date: ", test_end_date)

        dat_test = df_orig[(df_orig['date_time'] >= test_start_date) & (df_orig['date_time'] <= test_end_date)].copy()
        print(dat_test.head(10))
        print(dat_test.tail(10))

        dat_train['date_time'] = pd.to_datetime(dat_train['date_time'])
        dat_test['date_time'] = pd.to_datetime(dat_test['date_time'])    
    
        y_train = dat_train['load'].values
        y_test = dat_test['load'].values
        X_train = dat_train[fml]  # Base formula 
        X_test = dat_test[fml]

        first_train_date = dat_train['date_time'].min()
        print(f"Erster Tag der Trainingsdaten: {first_train_date}")
        last_train_date = dat_train['date_time'].max()
        print(f"Letzter Tag der Trainingsdaten: {last_train_date}")

        # Überprüfe den ersten Tag der Testdaten
        first_test_date = dat_test['date_time'].min()
        print(f"Erster Tag der Testdaten: {first_test_date}")
        last_test_date = dat_test['date_time'].max()
        print(f"Letzter Tag der Testdaten: {last_test_date}")

        for m_try in range(1, len(fml) + 1):
            print(f"Training with m_try = {m_try}")

        
            hyperparams = {
                'n_estimators': N_TREES,
                'random_state': SEED,
                'n_jobs': -1,
                'max_features': m_try,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_depth': None
            }

            hyperparams['random_state'] = hyperparams['random_state'] 
            rf = RandomForestWeight(hyperparams=hyperparams)
            rf.fit(X_train, y_train)

            _, w_hat = rf.weight_predict(X_test)
            y_pred = rf.predict(X_test)
    
            #print(y_test.shape)
            #print(X_test.shape)
            #print(y_train.shape)
            #print(y_pred.shape)
            #print(w_hat.shape)

            #print(y_test)
            #print(y_train)
            #print(w_hat)

            crps = crps_sample(y_test, y_train, w_hat)
            print(crps.shape)

            crps_arr.append({
            'm_try': m_try,
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'CRPS': crps
            })

            individual_crps = calculate_individual_crps(y_test, y_train, w_hat)
        
            individual_crps_arr.append({
                'm_try': m_try,
                'time_trend': 'yes' if 'time_trend' in fml else 'no',
                'day_of_year': 'yes' if 'yearday' in fml else 'no',
                'CRPS': individual_crps,
                'date_time': dat_test['date_time']
                })

         # SE ---
            se_val = se(y_test, y_pred)

            se_arr.append({
            'm_try': m_try,
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'SE': se_val
            })


        # Individual SE ---
            individual_se = calculate_individual_se(y_test, y_pred)


            individual_se_arr.append({
            'm_try': m_try,
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'SE': individual_se,
            'date_time': dat_test['date_time']
        })

        # MSE ---
            mse_val = mse(y_test, y_pred)

            mse_arr.append({
            'm_try': m_try,
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'MSE': mse_val
            })


        # Individual AE ---
            individual_ae = calculate_individual_ae(y_test, rf.quantile_predict(q=.5, X_test=X_test))


            individual_ae_arr.append({
            'm_try': m_try,
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'AE': individual_ae,
            'date_time': dat_test['date_time'] })


        # MAE ---
            #mae_val = mae(y_test, y_pred)
            mae_val = mae(y_test, rf.quantile_predict(q=.5, X_test=X_test))

            mae_arr.append({
            'm_try': m_try,
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
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
df_individual_se_arr = pd.DataFrame(individual_se_arr)
df_individual_se_arr
#%%
# MSE ---
df_mse = pd.DataFrame(mse_arr)
df_mse

#%%
df_ae = pd.DataFrame(individual_ae_arr)
df_ae
#%%
# MAE ---
df_mae = pd.DataFrame(mae_arr)
df_mae

#%%
# Save the results of the Random Forest Model 
#save_results_different_mtry(individual_crps_arr, individual_se_arr, individual_ae_arr ,dat_test, prefix="/home/siefert/projects/Masterarbeit/Results/Python/python_res_expanding_window_different_Mtry_a/" )
save_results_expanding_window(individual_crps_arr, individual_se_arr, individual_ae_arr, dat_test ,prefix="python_res_different_mtry_expanding/")


#%%
# LINEAR MODEL ---

from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta

df_orig['date_time'] = pd.to_datetime(df_orig['date_time'])

start_date = pd.Timestamp("2022-01-01 00:00:00")
end_date = pd.Timestamp("2023-11-01 00:00:00")

time_intervals = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monatsanfänge

crps_arr = []
se_arr = []
mse_arr = []
mae_arr = []
individual_crps_arr = []
individual_se_arr = []
individual_ae_arr = []

df_orig.dropna(inplace=True)

for fml in fmls:
    print(fml)
    print(f"Verwendete Formel: {fml}")

    df, _, _ = prep_energy(df=df_orig, **data_encoding)

    for current_date in time_intervals:
        print(f"Processing data up to {current_date}")

        # Trainings- und Testdaten aufteilen
        train_end_date = current_date - relativedelta(seconds=1)
        dat_train = df_orig[df_orig['date_time'] <= train_end_date].copy()

        test_start_date = current_date
        test_end_date = current_date + relativedelta(months=1) - relativedelta(seconds=1)
        dat_test = df_orig[(df_orig['date_time'] >= test_start_date) & (df_orig['date_time'] <= test_end_date)].copy()

        y_train = dat_train['load'].values
        y_test = dat_test['load'].values
        X_train = dat_train[fml]
        X_test = dat_test[fml]

        # Debug-Informationen
        print(f"Trainingszeitraum: {dat_train['date_time'].min()} bis {dat_train['date_time'].max()}")
        print(f"Testzeitraum: {dat_test['date_time'].min()} bis {dat_test['date_time'].max()}")

        # Lineares Modell trainieren
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        y_pred = lin_reg.predict(X_test)
        residuals = y_test - y_pred
        sigma = np.std(residuals)  # Schätzung der Unsicherheit

        individual_crps = calculate_individual_crps_linear(y_test, y_pred, sigma)

        individual_crps_arr.append({
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'CRPS': individual_crps,
            'date_time': dat_test['date_time'].values
        })

        # SE berechnen
        se_val = se(y_test, y_pred)
        se_arr.append({
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'SE': se_val
        })

        individual_se = calculate_individual_se(y_test, y_pred)
        individual_se_arr.append({
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'SE': individual_se,
            'date_time': dat_test['date_time']
        })

        individual_ae = calculate_individual_ae(y_test, y_pred)
        individual_ae_arr.append({
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'AE': individual_ae,
            'date_time': dat_test['date_time']
        })

        # MSE berechnen
        mse_val = mse(y_test, y_pred)
        mse_arr.append({
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'MSE': mse_val
        })

        # MAE berechnen
        mae_val = mae(y_test, y_pred)
        mae_arr.append({
            'time_trend': 'yes' if 'time_trend' in fml else 'no',
            'day_of_year': 'yes' if 'yearday' in fml else 'no',
            'MAE': mae_val
        })

#%%
# Individual CRPS ---
df_individual_crps = pd.DataFrame(individual_crps_arr)
df_individual_crps

#%%
df_individual_se_arr = pd.DataFrame(individual_se_arr)
df_individual_se_arr
#%%
df_individual_ae_arr = pd.DataFrame(individual_ae_arr)
df_individual_ae_arr

#%%

save_results_linear_model(individual_crps_arr, individual_se_arr, individual_ae_arr ,dat_test, prefix="/home/siefert/projects/Masterarbeit/Data/Results/Python_Results/python_res_linear_model/")


#%%
#=================================
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
#=================================
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
#=================================
# Plot -----
# Expanding Window ----

import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

df_orig['date_time'] = pd.to_datetime(df_orig['date_time'])

# Start- und Enddatum
start_date = pd.Timestamp("2022-01-01")
end_date = pd.Timestamp("2023-11-01")
time_intervals = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monatsanfänge

# Konsistente Farben
train_color = 'lightskyblue'
test_color = 'lightgreen'

plt.rcParams.update(plt.rcParamsDefault)

# Anzahl der Zeitfenster, die visualisiert werden sollen
num_intervals = 3  

# Subplots erstellen
fig, axes = plt.subplots(num_intervals, 1, figsize=(16, 6 * num_intervals), sharex=False, dpi = 500)

# Subplot für jedes Zeitfenster
for idx, current_date in enumerate(time_intervals[:num_intervals]):
    train_end_date = current_date - relativedelta(hours=1)
    test_start_date = current_date
    test_end_date = current_date + relativedelta(months=1) - relativedelta(hours=1)
    
    # Trainings- und Testdaten
    dat_train = df_orig[df_orig['date_time'] <= train_end_date]
    dat_test = df_orig[(df_orig['date_time'] >= test_start_date) & (df_orig['date_time'] <= test_end_date)]

    # Plot in Subplot
    ax = axes[idx]
    ax.plot(df_orig['date_time'], df_orig['load'], label='load', color='black', alpha=0.3)
    ax.plot(dat_train['date_time'], dat_train['load'], color=train_color, alpha=0.6, label=f'train up to {train_end_date}')
    ax.plot(dat_test['date_time'], dat_test['load'], color=test_color, alpha=0.9, label=f'test from {test_start_date} to {test_end_date}')
    
    ax.axvline(test_start_date, color='red', linestyle='-', alpha=1.0, label='start test interval')
    test_start_color = "red"
    # Titel und Legende
    ax.set_title(f'Expanding Window - Time Interval {idx + 1}', fontsize=16)
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel('Energy Load', fontsize=16)
    ax.tick_params(axis='x', labelsize=12)  # X-Achse Ticks
    ax.tick_params(axis='y', labelsize=12)  
    #plt.xticks(fontsize=12)  # Increase the font size of the x-axis ticks
    #plt.yticks(fontsize=12)    
    
    # Manuelle Farbanpassung der Legende für Train und Test und dickere Linien
    legend = ax.legend(fontsize=14, loc='upper left')

    
    ax.grid(True)
    
    # Entferne oberste und rechte Spines und setze ihre Farben auf hellgrau
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgrey')  # Linke Spine in hellgrau
    ax.spines['bottom'].set_color('lightgrey')  # Untere Spine in hellgrau

# Gemeinsames Layout
plt.tight_layout()
plt.show()

# %%
