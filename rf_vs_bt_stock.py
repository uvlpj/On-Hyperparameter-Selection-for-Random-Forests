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
import glob

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

def calculate_individual_crps(y_test, dat_test, weights_test):

    individual_crps = crps_sample(y_test, dat_test, weights_test, return_mean=False)
    print('shape crps_values ', individual_crps.shape)
    
    
    return individual_crps
        

def calculate_individual_se(y_test, y_pred):

    individual_se = se(y_test, y_pred)
    print('shape se_values ', individual_se.shape)
    
    return individual_se

#%%
def save_results(individual_crps_arr, individual_se_arr, dat_name ,dat_test, prefix="res/"):
    '''
    Funktion um die Ergebnisse von sklearn zu speichern
    
    Output: CSV-Dateien, die die CRPS- sowie die SE-Werte für jeden Datenpunkt im Testdatensatz enthalten.
    '''
    os.makedirs(prefix, exist_ok=True)  # Erstelle das Verzeichnis, falls es nicht existiert
    
    for i, result in enumerate(individual_crps_arr):
        # Dynamischer Dateiname basierend auf den Parametern (time_trend, day_of_year)
        time_trend_part = 'tt' if result['time_trend'] == 'yes' else 'nott'
        day_part = 'day' if result['day_of_year'] == 'yes' else 'month'
        lag_part = 'lagged' if result['load_lag1'] == 'yes' else 'notlagged'
        
        # Speichern für Random Forest
        save_name_rf = f"{prefix}sklearn_{time_trend_part}_{day_part}_{lag_part}_rf.csv"
        # Speichern für Bagged Trees
        save_name_bt = f"{prefix}sklearn_{time_trend_part}_{day_part}_{lag_part}_bt.csv"

        # Erstelle einen DataFrame für Random Forest
        df_rf = pd.DataFrame({
            'date_time': dat_test.index, 
            'crps': result['CRPS_RF'],  # Individuelle CRPS-Werte für RF
            'se': se_arr[i]['SE_RF'],    # Individuelle SE-Werte für RF
        })
        
        # Erstelle einen DataFrame für Bagged Trees
        df_bt = pd.DataFrame({
            'date_time': dat_test.index, 
            'crps': result['CRPS_BT'],  # Individuelle CRPS-Werte für BT
            'se': se_arr[i]['SE_BT'],    # Individuelle SE-Werte für BT
        })
        
        # Speichern der Ergebnisse als CSV-Datei mit Index
        df_rf.to_csv(save_name_rf, index=False)  # Index wird mitgespeichert
        df_bt.to_csv(save_name_bt, index=False)  # Index wird mitgespeichert
        
        print(f"Saved: {save_name_rf}")
        print(f"Saved: {save_name_bt}")

        
#%%
output_folder = r"/home/siefert/projects/Masterarbeit/sophia_code/python_res_stock_2"
os.makedirs(output_folder, exist_ok=True)

#%%
def save_results_different_mtry(individual_crps_arr, individual_se_arr, dat_name, dat_test, prefix="python_res_stock_different_mtry_2/"):
    '''
    Funktion um die Ergebnisse von sklearn zu speichern
    
    Output: CSV-Dateien, die die CRPS- sowie die SE-Werte für jeden Datenpunkt im Testdatensatz enthalten.
    '''
    os.makedirs(prefix, exist_ok=True)  # Erstelle das Verzeichnis, falls es nicht existiert
    
    # Schleife durch alle CRPS-Ergebnisse
    for i, result in enumerate(individual_crps_arr):
        # Der mtry-Wert und der Datensatzname für jede Iteration
        mtry_value = f"mtry{result['m_try']}"
        current_dat_name = result['name']
        
        # Dynamischer Dateiname basierend auf den Parametern (Datensatzname und mtry)
        save_name = f"{prefix}sklearn_{current_dat_name}_{mtry_value}.csv"
        
        # Erstelle einen DataFrame mit den entsprechenden Ergebnissen
        df = pd.DataFrame({
            'date_time': dat_test['date'], 
            'name': current_dat_name,
            'crps': result['CRPS'],  # CRPS-Werte für den aktuellen mtry
            'se': individual_se_arr[i]['SE'],    # SE-Werte für den aktuellen mtry
        })
        
        # Speichern der CSV-Datei
        df.to_csv(save_name, index=False)

        print(f"Saved: {save_name}")

   
#%%
ordner_path = '/home/siefert/projects/Masterarbeit/dat_stock'
dateien = glob.glob(os.path.join(ordner_path, "*.csv"))

datensaetze = {}

for datei in dateien:
    dat_name = os.path.splitext(os.path.basename(datei))[0]
    datensaetze[dat_name] = pd.read_csv(datei)


for name, df in datensaetze.items():
    print(f"Datensatzname: {name}, Anzahl Zeilen: {len(df)}")
    print(df.head())


#%%
base_fml = ['aret_l1' ,'aret_l5', 'aret_l22', 
                       'vol_l1',  'vol_l5',  'vol_l22', 
                       'hml_l1' , 'hml_l5' , 'hml_l22' ,
                       'vxv_l1' , 'vxv_l5' , 'vxv_l22' ,
                       'vxn_l1' , 'vxn_l5' , 'vxn_l22' ,
                       'vxd_l1' , 'vxd_l5' , 'vxd_l22']
fmls = []
fmls.append(base_fml) # without timetrend
#%%
# Hyperparameter
N_TREES = 100

mtry_values = [1, 4, 6, 12, 18]

# Ergebnis-Arrays
crps_arr = []
se_arr = []
mse_arr = []
mae_arr = []
individual_crps_arr = []
individual_se_arr = []

train_ratio = 0.7

# Beginne das Training für jeden Datensatz und jeden Wert von m_try
for dat_name, df in datensaetze.items():
    print(f"Datensatz: {dat_name}")

    df['date_time'] = pd.to_datetime(df['date'])
    
    # Gesamtanzahl der Daten (Gesamtzahl der Zeilen)
    total_rows = len(df)
    
    # Berechne die Anzahl der Trainingsdaten basierend auf dem gewünschten Anteil (z.B. 70%)
    train_size = int(total_rows * train_ratio)
    
    # Splitte den DataFrame in Trainings- und Testdaten
    dat_train = df.iloc[:train_size]  # Trainingsdaten (erste 70% der Daten)
    dat_test = df.iloc[train_size:]
    dat_train = dat_train.copy()
    dat_test = dat_test.copy()

    dat_train['date_time'] = pd.to_datetime(dat_train['date'])
    dat_test['date_time'] = pd.to_datetime(dat_test['date'])



    
    # Trainings- und Testzeitraum
    #tp_start = "2022-01-01 00:00:00"
    #df['date_time'] = pd.to_datetime(df['date'])
    #dat_train = df.set_index("date_time")[:tp_start]
    #dat_test = df.set_index("date_time")[tp_start:]
    
    y_train = dat_train['aret'].values
    y_test = dat_test['aret'].values

    # Training und Test-Sets vorbereiten
    X_train = dat_train[base_fml]
    X_test = dat_test[base_fml]
    
    for m_try in mtry_values:
        print(f"Training RandomForest mit mtry = {m_try}")

        hyperparams = {
            'n_estimators': N_TREES,
            'random_state': SEED,
            'n_jobs': -1,
            'max_features': m_try,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }

        hyperparams['random_state'] = hyperparams['random_state'] 
        rf = RandomForestWeight(hyperparams=hyperparams)
        rf.fit(X_train, y_train)

        _, w_hat = rf.weight_predict(X_test)
        y_pred = rf.predict(X_test)

        # CRPS ---

        crps = crps_sample(y_test, y_train, w_hat)
        crps_arr.append({
            'm_try': m_try,
            'name': dat_name,
            'CRPS': crps
        })

        # Individual CRPS ---
        individual_crps = calculate_individual_crps(y_test, y_train, w_hat)
        
        individual_crps_arr.append({
            'm_try': m_try,
            'name': dat_name,
            'CRPS': individual_crps,
            'date_time': dat_test['date']})

         # SE ---
        se_val = se(y_test, y_pred)

        se_arr.append({
            'm_try': m_try,
            'name': dat_name,
            'SE': se_val})


        # Individual SE ---
        individual_se = calculate_individual_se(y_test, y_pred)


        individual_se_arr.append({
            'm_try': m_try,
            'name': dat_name,
            'SE': individual_se,
            'date_time': dat_test['date']})

        # MSE ---
        mse_val = mse(y_test, y_pred)

        mse_arr.append({
            'm_try': m_try,
            'name': dat_name,
            'MSE': mse_val})

        # MAE ---
        mae_val = mae(y_test, y_pred)

        mae_arr.append({
            'm_try': m_try,
            'name': dat_name,
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
# MAE ---
df_mae = pd.DataFrame(mae_arr)
df_mae

#%%
save_results_different_mtry(individual_crps_arr, individual_se_arr, dat_name, dat_test ,prefix="python_res_stock_different_mtry_2/")

# %%
# =========================
# Plotting the timeseries ------
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False
plt.style.use('seaborn-whitegrid')

ordner_path = '/home/siefert/projects/Masterarbeit/dat_stock'
dateien = glob.glob(os.path.join(ordner_path, "*.csv"))

datensaetze = {}
# Einlesen der CSV-Dateien
for datei in dateien:
    dat_name = os.path.splitext(os.path.basename(datei))[0]
    try:
        datensaetze[dat_name] = pd.read_csv(datei)
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei {datei}: {e}")
        continue

# Plotten der Zeitreihen für die Spalte 'aret'
for name, df in datensaetze.items():
    print(f"Datensatzname: {name}, Anzahl Zeilen: {len(df)}")
    print(df.head())
    
    # Überprüfen, ob die Spalten 'date' und 'aret' vorhanden sind
    if 'date' in df.columns and 'aret' in df.columns:
        # Konvertiere 'date' zu datetime und sortiere
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Plot erstellen
        fig, ax = plt.subplots(figsize=(12, 6), dpi = 500)
        ax.plot(df['date'], df['aret'], label=f"{name} - aret")
        ax.set_title(f"Stock {name}", fontsize = 16)
        ax.set_xlabel("Date", fontsize = 14)
        ax.set_ylabel("aret", fontsize = 14)
        #ax.legend()
        ax.grid(True)
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        plt.xticks(fontsize=12)  # Increase the font size of the x-axis ticks
        plt.yticks(fontsize=12) 
        ax.spines['top'].set_color('lightgrey')
        ax.spines['right'].set_color('lightgrey')
        ax.spines['left'].set_color('lightgrey')  # Linke Spine in hellgrau
        ax.spines['bottom'].set_color('lightgrey')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Spalten 'date' oder 'aret' fehlen in {name}.")


# %%
# =========================
# Stock Subplot ---

plt.rcParams['text.usetex'] = False
plt.style.use('seaborn-whitegrid')


# Ordnerpfad und CSV-Dateien
ordner_path = '/home/siefert/projects/Masterarbeit/dat_stock'
dateien = glob.glob(os.path.join(ordner_path, "*.csv"))

datensaetze = {}
# Einlesen der CSV-Dateien
for datei in dateien:
    dat_name = os.path.splitext(os.path.basename(datei))[0]
    try:
        datensaetze[dat_name] = pd.read_csv(datei)
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei {datei}: {e}")
        continue

if "AMGN" in datensaetze:
    del datensaetze["AMGN"] 

# Anzahl der Plots
anzahl_plots = len(datensaetze)
spalten = 4  # Anzahl der Spalten im Subplot
zeilen = -(-anzahl_plots // spalten)  # Rundet nach oben, um genügend Zeilen zu haben

# Gesamten Subplot-Bereich vorbereiten
fig, axes = plt.subplots(zeilen, spalten, figsize=(20, zeilen * 4))
axes = axes.flatten()  # Um die Achsen einfach zu iterieren

# Plotten der Zeitreihen für die Spalte 'aret'
for i, (name, df) in enumerate(datensaetze.items()):
    print(f"Datensatzname: {name}, Anzahl Zeilen: {len(df)}")
    print(df.head())
    
    # Überprüfen, ob die Spalten 'date' und 'aret' vorhanden sind
    if 'date' in df.columns and 'aret' in df.columns:
        # Konvertiere 'date' zu datetime und sortiere
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Plot in die entsprechende Achse
        ax = axes[i]
        ax.plot(df['date'], df['aret'], label=f"{name} - aret")
        ax.set_title(f"Stock {name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("aret")
        #ax.legend()
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('lightgrey')  # Linke Spine in hellgrau
        ax.spines['bottom'].set_color('lightgrey')
        
        # Y-Achse auf 0 bis 20 begrenzen und Ticks setzen
        ax.set_ylim(0, 26)
        ax.set_yticks(range(0, 26, 5))  # Setzt die Y-Ticks von 0 bis 20 in Schritten von 5
    else:
        print(f"Spalten 'date' oder 'aret' fehlen in {name}.")
        # Leere Achse deaktivieren, falls nicht verwendet
        axes[i].axis('off')

# Überflüssige Achsen deaktivieren
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
#%%

#%%
# %%
# Plot ----
# Shows the Correlationplot of each of the 30 datasets in the stock dataset

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['text.usetex'] = False
plt.rcdefaults() 

# Pfad zu den CSV-Dateien
ordner_path = '/home/siefert/projects/Masterarbeit/dat_stock'
dateien = glob.glob(os.path.join(ordner_path, "*.csv"))

datensaetze = {}

# CSV-Dateien einlesen
for datei in dateien:
    dat_name = os.path.splitext(os.path.basename(datei))[0]
    datensaetze[dat_name] = pd.read_csv(datei)

# Korrelationen berechnen und visualisieren
for name, df in datensaetze.items():
    print(f"Korrelationen für {name}:")
    
    df_numeric = df.select_dtypes(include=['number'])  
    corr_matrix = df_numeric.corr()

    # Maske erstellen, um nur die obere Dreieckshälfte zu verbergen
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Heatmap mit der Maske plotten
    plt.figure(figsize=(14, 10), dpi = 500)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title(f'Correlation matrix for {name}')
    plt.show()