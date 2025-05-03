#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from data_preprocessor import *
import pandas as pd
import matplotlib.pyplot as plt
import glob
#%%
# Pfad zu den CSV-Dateien (alle CSV-Dateien im Ordner "res")

#csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Data/Results/R_Results/res_different_mtry/*.csv")
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Data/Results/R_Results/res_different_Mtry/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/Data/Results/Python_Results/python_res_different_mtry/*.csv")

#%%
# Pfad zu den Ergebnisse (keine Konstante in quantregForest, angepasste Hyperparameter, BT und RF )
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Data/Results/R_Results/res_without_Intercept_Hyperpara_const/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_without_intercept_hyper_const/*.csv")

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

#%%
csv_files = csv_files_r + csv_files_python
#%%
#specifications = [
#    'nott_doy_bt', 'nott_doy_rf',
#    'nott_month_bt', 'nott_month_rf',
#    'tt_doy_bt', 'tt_doy_rf',
#    'tt_month_bt', 'tt_month_rf'
#]

#specifications = [
#    'nott_day_lagged_bt', 'nott_day_lagged_rf',
#    'nott_month_lagged_bt', 'nott_month_lagged_rf',
#    'tt_day_lagged_bt', 'tt_day_lagged_rf',
#    'tt_month_lagged_bt', 'tt_month_lagged_rf'
#    'nott_day_notlagged_bt', 'nott_day_notlagged_rf',
#    'nott_month_notlagged_bt', 'nott_month_notlagged_rf',
#    'tt_day_notlagged_bt', 'tt_day_notlagged_rf',
#    'tt_month_notlagged_bt', 'tt_month_notlagged_rf'
#]


specifications = [
    'nott_day_lagged_bt', 'nott_day_lagged_rf',
    'nott_month_lagged_bt', 'nott_month_lagged_rf',
    'tt_day_lagged_bt', 'tt_day_lagged_rf',
    'tt_month_lagged_bt', 'tt_month_lagged_rf'
    'nott_day_notlagged_bt', 'nott_day_notlagged_rf',
    'nott_month_notlagged_bt', 'nott_month_notlagged_rf',
    'tt_day_notlagged_bt', 'tt_day_notlagged_rf',
    'tt_month_notlagged_bt', 'tt_month_notlagged_rf'
]

#%%
#=================================
# Plot (A) -----
'''
Plot zeigt den Mean CRPS für jede Feature Combination 
Hyperparameter wie in Schritt 1 sind nicht angepasst
Die Konstante in quantregForest ist vorhanden
'''
# Pfad zu den Ergebnisse
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_with_Intercept_Hyperpara_notconst/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_without_intercept_hyper_not_const/*.csv")

# Kombiniere alle CSV-Dateien
csv_files = csv_files_r + csv_files_python

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

def plot_mean_crps_subplots_1(specifications):
    """
    Plot the mean CRPS for the three packages for multiple specifications in a 4x4 subplot.
    """
    num_specs = len(specifications)
    num_cols = 4  # Number of columns in subplot grid
    num_rows = (num_specs + num_cols - 1) // num_cols  
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=300)
    fig.subplots_adjust(left=0.06, right=0.94, wspace=0.6, top=0.85)  # Adjust subplot spacing for more left/right space
    fig.suptitle('Mean CRPS comparison across packages\n with Intercept in quantregForest and hyperparameters not aligned', fontsize=16)

    axes = axes.flatten()

    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}

    for idx, spec in enumerate(specifications):
        ax = axes[idx]
        mean_crps_values = {}
        plotted = False

        for package in ['sklearn', 'ranger', 'quantregForest']:
            pattern = re.compile(f"^{package}_{spec}\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]

            if filtered_files:
                df = pd.read_csv(filtered_files[0])
                mean_crps = df['crps'].mean()
                mean_crps_values[package] = mean_crps
                ax.scatter(package, mean_crps, color=colors[package], s = 20)
                ax.text(package, mean_crps - 150, f'{mean_crps:.2f}', ha='center', va='top', fontsize=12.5, color=colors[package])
                plotted = True

        if not plotted:
            ax.plot([], [], color='gray')  # Dummy plot for legend


        ax.set_title(f'{spec}', fontsize=12)
        ax.set_ylabel('Mean CRPS', fontsize=10)
        ax.set_ylim(0, 4300)  # Adjusted Y-axis limit
        ax.set_yticks(range(0, 4001, 1000))  # Set y-axis grid in 1000 steps
        ax.set_xlim(-0.5, 2.5)  # Add more padding on the sides
        ax.tick_params(labelsize=12)
        ax.grid(True)
        

    handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package]) for package in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors), bbox_to_anchor=(0.5, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout
    plt.show()

#%%

specifications = [
    'nott_day_lagged_bt', 'nott_day_lagged_rf',
    'nott_month_lagged_bt', 'nott_month_lagged_rf',
    'tt_day_lagged_bt', 'tt_day_lagged_rf',
    'tt_month_lagged_bt', 'tt_month_lagged_rf',
    'nott_day_notlagged_bt', 'nott_day_notlagged_rf',
    'nott_month_notlagged_bt', 'nott_month_notlagged_rf',
    'tt_day_notlagged_bt', 'tt_day_notlagged_rf',
    'tt_month_notlagged_bt', 'tt_month_notlagged_rf'
]

plot_mean_crps_subplots_1(specifications)

#%%

#%%
#=================================
# Plot (B) -----
'''
Plot zeigt den Mean CRPS für jede Feature Combination 
Hyperparameter wie in Schritt 2 sind angepasst
Die Konstante in quantregForest ist entfernt 
'''
# Pfad zu den Ergebnisse
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_without_Intercept_Hyperpara_const/*.csv")
#csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_without_intercept_hyper_const/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_rf_bt/*.csv")

# Kombiniere alle CSV-Dateien
csv_files = csv_files_r + csv_files_python

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

def plot_mean_crps_subplots_2(specifications):
    """
    Plot the mean CRPS for the three packages for multiple specifications in a 4x4 subplot.
    """
    num_specs = len(specifications)
    num_cols = 4  # Number of columns in subplot grid
    num_rows = (num_specs + num_cols - 1) // num_cols  
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=300)
    fig.subplots_adjust(left=0.06, right=0.94, wspace=0.6, top=0.85)  # Adjust subplot spacing for more left/right space

    fig.suptitle('Mean CRPS comparison across packages\n without Intercept in quantregForest and aligned hyperparameters', fontsize=16)

    axes = axes.flatten()

    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}

    for idx, spec in enumerate(specifications):
        ax = axes[idx]
        mean_crps_values = {}
        plotted = False

        for package in ['sklearn', 'ranger', 'quantregForest']:
            pattern = re.compile(f"^{package}_{spec}\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]

            if filtered_files:
                df = pd.read_csv(filtered_files[0])
                mean_crps = df['crps'].mean()
                mean_crps_values[package] = mean_crps
                ax.scatter(package, mean_crps, color=colors[package], s = 20)
                ax.text(package, mean_crps - 150, f'{mean_crps:.2f}', ha='center', va='top', fontsize=12.5, color=colors[package])
                plotted = True

        if not plotted:
            ax.plot([], [], color='gray')  # Dummy plot for legend


        ax.set_title(f'Mean CRPS for {spec}', fontsize=12)
        ax.set_ylabel('Mean CRPS')
        ax.set_ylim(-0.02*4300, 4300)  # Adjusted Y-axis limit
        ax.set_yticks(range(0, 4001, 1000))  # Set y-axis grid in 1000 steps
        ax.set_xlim(-0.5, 2.5)  # Add more padding on the sides

        ax.tick_params(labelsize=12)
        ax.grid(True)
        

    handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package]) for package in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors), bbox_to_anchor=(0.5, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout
    plt.show()

#%%
specifications = [
    'nott_day_lagged_bt', 'nott_day_lagged_rf',
    'nott_month_lagged_bt', 'nott_month_lagged_rf',
    'tt_day_lagged_bt', 'tt_day_lagged_rf',
    'tt_month_lagged_bt', 'tt_month_lagged_rf',
    'nott_day_notlagged_bt', 'nott_day_notlagged_rf',
    'nott_month_notlagged_bt', 'nott_month_notlagged_rf',
    'tt_day_notlagged_bt', 'tt_day_notlagged_rf',
    'tt_month_notlagged_bt', 'tt_month_notlagged_rf'
]

plot_mean_crps_subplots_2(specifications)


#%%
#=================================
# PLOT ----
# BT vs RF
# CRPS ----

# Pfad zu den Ergebnisse
#csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_ranger_quantreg_rf_bt_a1/*.csv")
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_rf_bt_a1/*.csv")
#csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_without_Intercept_Hyperpara_const/*.csv")



#csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_without_intercept_hyper_const/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_rf_bt/*.csv")

# Kombiniere alle CSV-Dateien
csv_files = csv_files_r + csv_files_python

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')


def plot_mean_crps_subplots(specifications):
    """
    Plot the mean CRPS for Boosted Trees (BT) and Random Forest (RF) for each feature combination in separate subplots.
    """
    num_specs = len(specifications) // 2  # We assume specifications are in pairs
    num_cols = 4  # Number of columns in subplot grid
    num_rows = (num_specs + num_cols - 1) // num_cols  
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=500)
    fig.subplots_adjust(left=0.06, right=0.94, wspace=0.3, top=0.85)  # Adjust subplot spacing to fit labels
    
    fig.suptitle('Mean CRPS comparison: BT vs RF across packages', fontsize=16)
    
    axes = axes.flatten()
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}
    
    for idx in range(0, len(specifications), 2):
        feature_name = specifications[idx].rsplit('_', 1)[0]  # Remove _bt or _rf
        ax = axes[idx // 2]
        
        # Neue Positionen für BT und RF, enger zusammen
        x_labels = ['RF', 'BT']
        x_positions = [0.1, 0.5]  # Wir setzen BT bei 0.3 und RF bei 0.7, damit sie nah beieinander sind

        # Plotten der Daten für BT und RF
        for x_pos, model in zip(x_positions, ['rf', 'bt']):
            for package in ['sklearn', 'ranger', 'quantregForest']:
                spec = specifications[idx] if model == 'bt' else specifications[idx + 1]
                pattern = re.compile(f"^{package}_{spec}\.csv$", )
                filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
                
                if filtered_files:
                    df = pd.read_csv(filtered_files[0])
                    mean_crps = df['crps'].mean()
                    ax.scatter(x_pos, mean_crps, color=colors[package], s=60, alpha=0.6, marker = 'x',
                               label=labels[package] if x_pos == 0.3 else "")
                    
        ax.set_title(f'{feature_name}', fontsize=14)
        ax.set_ylabel('Mean CRPS', fontsize = 14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=14)
        ax.set_ylim(-0.02*4500, 4500)
        ax.set_xlim(0, 0.6)
        ax.set_yticks(range(0, 4501, 1000))
        #ax.set_yticks(range(0, 4501, 500))
        ax.grid(True)

        ax.tick_params(axis='both', which='major', labelsize=13)
    
    handles = [plt.Line2D([0], [0], marker='x', color='w', markerfacecolor=colors[p], markeredgecolor=colors[p], markersize=10, label=labels[p]) for p in colors]
    #handles = [plt.Line2D([0], [0], marker='x', color=colors[p], markersize=8, label=labels[p]) for p in colors]
    #handles = [plt.Line2D([0], [0], marker='o', color=colors[p], markersize=8, label=labels[p]) for p in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors), bbox_to_anchor=(0.5, 0.95))
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()



specifications = [
    'nott_day_lagged_bt', 'nott_day_lagged_rf',
    'nott_month_lagged_bt', 'nott_month_lagged_rf',
    'tt_day_lagged_bt', 'tt_day_lagged_rf',
    'tt_month_lagged_bt', 'tt_month_lagged_rf',
    'nott_day_notlagged_bt', 'nott_day_notlagged_rf',
    'nott_month_notlagged_bt', 'nott_month_notlagged_rf',
    'tt_day_notlagged_bt', 'tt_day_notlagged_rf',
    'tt_month_notlagged_bt', 'tt_month_notlagged_rf'
]

plot_mean_crps_subplots(specifications)

#%%

#%%
#=================================
# Plot ----
# Root Mean SE
# BT vs RF 

#csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_ranger_quantreg_rf_bt_a1/*.csv")
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_rf_bt_a1/*.csv")

#csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_without_Intercept_Hyperpara_const/*.csv")

#csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_without_intercept_hyper_const/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_rf_bt/*.csv")

# Kombiniere alle CSV-Dateien
csv_files = csv_files_r + csv_files_python

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')


def plot_mean_se_subplots(specifications):
    """
    Plot the mean SE for Boosted Trees (BT) and Random Forest (RF) for each feature combination in separate subplots.
    """
    num_specs = len(specifications) // 2  # We assume specifications are in pairs
    num_cols = 4  # Number of columns in subplot grid
    num_rows = (num_specs + num_cols - 1) // num_cols  
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=500)
    fig.subplots_adjust(left=0.06, right=0.94, wspace=0.3, top=0.85)  # Adjust subplot spacing to fit labels
    
    fig.suptitle('Root Mean SE comparison: BT vs RF across packages', fontsize=16)
    
    axes = axes.flatten()
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}
    
    for idx in range(0, len(specifications), 2):
        feature_name = specifications[idx].rsplit('_', 1)[0]  # Remove _bt or _rf
        ax = axes[idx // 2]
        
        # Neue Positionen für BT und RF, enger zusammen
        x_labels = ['RF', 'BT']
        x_positions = [0.1, 0.5]  # Wir setzen BT bei 0.3 und RF bei 0.7, damit sie nah beieinander sind

        # Plotten der Daten für BT und RF
        for x_pos, model in zip(x_positions, ['rf', 'bt']):
            for package in ['sklearn', 'ranger', 'quantregForest']:
                spec = specifications[idx] if model == 'bt' else specifications[idx + 1]
                pattern = re.compile(f"^{package}_{spec}\.csv$", )
                filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
                
                if filtered_files:
                    df = pd.read_csv(filtered_files[0])
                    mean_crps = np.sqrt(df['se'].mean())
                    ax.scatter(x_pos, mean_crps, color=colors[package], s=60, alpha=0.8, marker = 'x',
                               label=labels[package] if x_pos == 0.3 else "")
                    
        ax.set_title(f'{feature_name}', fontsize=14)
        ax.set_ylabel('Root Mean SE', fontsize = 14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=14)
        ax.set_ylim(-0.02*7000, 7000)
        ax.set_xlim(0, 0.6)
        ax.set_yticks(range(0, 7001, 1000))
        #ax.set_yticks(range(0, 4501, 500))
        ax.grid(True)

        ax.tick_params(axis='both', which='major', labelsize=13)
    
    handles = [plt.Line2D([0], [0], marker='x', color='w', markerfacecolor=colors[p], markeredgecolor=colors[p], markersize=10, label=labels[p]) for p in colors]
    #handles = [plt.Line2D([0], [0], marker='x', color=colors[p], markersize=8, label=labels[p]) for p in colors]
    #handles = [plt.Line2D([0], [0], marker='o', color=colors[p], markersize=8, label=labels[p]) for p in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors), bbox_to_anchor=(0.5, 0.95))
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()



specifications = [
    'nott_day_lagged_bt', 'nott_day_lagged_rf',
    'nott_month_lagged_bt', 'nott_month_lagged_rf',
    'tt_day_lagged_bt', 'tt_day_lagged_rf',
    'tt_month_lagged_bt', 'tt_month_lagged_rf',
    'nott_day_notlagged_bt', 'nott_day_notlagged_rf',
    'nott_month_notlagged_bt', 'nott_month_notlagged_rf',
    'tt_day_notlagged_bt', 'tt_day_notlagged_rf',
    'tt_month_notlagged_bt', 'tt_month_notlagged_rf'
]

plot_mean_se_subplots(specifications)


#%%
#=================================
# Plot ----
# Mean AE
# BT vs RF 

#csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_ranger_quantreg_rf_bt_a1/*.csv")
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_rf_bt_a1/*.csv")

#csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_without_Intercept_Hyperpara_const/*.csv")

#csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_without_intercept_hyper_const/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_rf_bt/*.csv")

# Kombiniere alle CSV-Dateien
csv_files = csv_files_r + csv_files_python

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')


def plot_mean_ae_subplots(specifications):
    """
    Plot the mean AE for Boosted Trees (BT) and Random Forest (RF) for each feature combination in separate subplots.
    """
    num_specs = len(specifications) // 2  # We assume specifications are in pairs
    num_cols = 4  # Number of columns in subplot grid
    num_rows = (num_specs + num_cols - 1) // num_cols  
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=500)
    fig.subplots_adjust(left=0.06, right=0.94, wspace=0.3, top=0.85)  # Adjust subplot spacing to fit labels
    
    fig.suptitle('Mean AE comparison: BT vs RF across packages', fontsize=16)
    
    axes = axes.flatten()
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}
    
    for idx in range(0, len(specifications), 2):
        feature_name = specifications[idx].rsplit('_', 1)[0]  # Remove _bt or _rf
        ax = axes[idx // 2]
        
        # Neue Positionen für BT und RF, enger zusammen
        x_labels = ['RF', 'BT']
        x_positions = [0.1, 0.5]  # Wir setzen BT bei 0.3 und RF bei 0.7, damit sie nah beieinander sind

        # Plotten der Daten für BT und RF
        for x_pos, model in zip(x_positions, ['rf', 'bt']):
            for package in ['sklearn', 'ranger', 'quantregForest']:
                spec = specifications[idx] if model == 'bt' else specifications[idx + 1]
                pattern = re.compile(f"^{package}_{spec}\.csv$", )
                filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
                
                if filtered_files:
                    df = pd.read_csv(filtered_files[0])
                    mean_crps = (df['ae'].mean())
                    print(f"Specification: {spec}, Package: {package}, Model: {model.upper()}, Mean AE: {mean_crps:.4f}")
                    ax.scatter(x_pos, mean_crps, color=colors[package], s=60, alpha=0.8, marker = 'x',
                               label=labels[package] if x_pos == 0.3 else "")
                    
        ax.set_title(f'{feature_name}', fontsize=14)
        ax.set_ylabel('Mean AE', fontsize = 14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=14)
        ax.set_ylim(-0.02*5500, 5500)
        ax.set_xlim(0, 0.6)
        ax.set_yticks(range(0, 5501, 1000))
        #ax.set_yticks(range(0, 4501, 500))
        ax.grid(True)

        ax.tick_params(axis='both', which='major', labelsize=13)
    
    handles = [plt.Line2D([0], [0], marker='x', color='w', markerfacecolor=colors[p], markeredgecolor=colors[p], markersize=10, label=labels[p]) for p in colors]
    #handles = [plt.Line2D([0], [0], marker='x', color=colors[p], markersize=8, label=labels[p]) for p in colors]
    #handles = [plt.Line2D([0], [0], marker='o', color=colors[p], markersize=8, label=labels[p]) for p in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors), bbox_to_anchor=(0.5, 0.95))
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()



specifications = [
    'nott_day_lagged_bt', 'nott_day_lagged_rf',
    'nott_month_lagged_bt', 'nott_month_lagged_rf',
    'tt_day_lagged_bt', 'tt_day_lagged_rf',
    'tt_month_lagged_bt', 'tt_month_lagged_rf',
    'nott_day_notlagged_bt', 'nott_day_notlagged_rf',
    'nott_month_notlagged_bt', 'nott_month_notlagged_rf',
    'tt_day_notlagged_bt', 'tt_day_notlagged_rf',
    'tt_month_notlagged_bt', 'tt_month_notlagged_rf'
]

plot_mean_ae_subplots(specifications)


#%%
#=================================
# Plot -----
# Cumulative SE Werte --
# Auf der x-achse Datum ---

#csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_without_Intercept_Hyperpara_const/*.csv")
#csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_without_intercept_hyper_const/*.csv")

csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_ranger_quantreg_rf_bt_a1/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_rf_bt/*.csv")

# Beinhaltet die Version in Python von Fabian
#csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_rf_bt_a2/*.csv")

# Kombiniere alle CSV-Dateien
csv_files = csv_files_r + csv_files_python

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

def plot_cumulative_se(specification):
    """
    Plot the cumulative Squared Error for the three packages for a specific specification.
    """
    plt.figure(figsize=(16, 8), dpi=500)

    # Farbschema und Linienstil für die verschiedenen Pakete
    colors = {'ranger': 'b', 'quantregForest': 'g', 'sklearn': 'r'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}
    linestyles = {'ranger': '-', 'quantregForest': '-', 'sklearn': '-'}  # Verschiedene Linienstile
    
    # Überprüfen, ob die CSV-Dateien die spezifische Spezifikation enthalten
    for package in ['sklearn', 'ranger', 'quantregForest']:
        # Verwende einen Regex, um exakte Übereinstimmung zu erzwingen (Paketname + Spezifikation)
        pattern = re.compile(f"^{package}_{specification}\.csv$")
        filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
        print(f"Filtered files for {package} => {filtered_files}") 
        
        # Überprüfen, ob eine passende Datei gefunden wurde
        if filtered_files:
            # Lese die erste passende Datei ein 
            df = pd.read_csv(filtered_files[0])
            print(df)
            print(f"Columns in {package} data: {df.columns.tolist()}")

            # Konvertiere 'date_time' zu datetime-Objekten
            df['date_time'] = pd.to_datetime(df['date_time'])

            # Kumulierten SE berechnen
            df['cumulative_se'] = np.sqrt(df['se'].cumsum())
            print(df['cumulative_se'].head(10))
            print(df['cumulative_se'].tail(10))

            # Plot der kumulierten SE mit spezifischem Linienstil
            plt.plot(df['date_time'], df['cumulative_se'], 
                     label=labels[package], 
                     color=colors[package], 
                     linestyle=linestyles[package],
                     alpha = 0.7)

    # Plot-Details
    plt.title(f'{specification}', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    
    # Skalierung der Y-Achse zur besseren Sichtbarkeit von Unterschieden
    plt.ylim(df['cumulative_se'].min() - 100, df['cumulative_se'].max() + 100)

    plt.ylabel('Cumulative SE', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper left')  # Kleine Legende links oben
    plt.tight_layout()  # Optimiert das Layout, um Überlappungen zu vermeiden

    # Plot anzeigen
    plt.show()


#%%
plot_cumulative_se('nott_month_lagged_rf')
#%%
plot_cumulative_se('nott_day_notlagged_bt')
#%%
plot_cumulative_se('nott_day_notlagged_rf')
#%%
plot_cumulative_se('nott_month_lagged_bt')
#%%
plot_cumulative_se('nott_month_lagged_rf')
#%%
plot_cumulative_se('tt_day_lagged_bt')
#%%
plot_cumulative_se('tt_day_lagged_rf')
#%%
plot_cumulative_se('tt_month_lagged_bt')
#%%
plot_cumulative_se('tt_month_lagged_rf')
#%%
plot_cumulative_se('nott_day_notlagged_bt')


# ====================================================================================================================
#%%
#=================================
# PLOT ---
# Cumulative CRPS
# BT vs RF

# csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_without_Intercept_Hyperpara_const/*.csv")
# csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_without_intercept_hyper_const/*.csv")

csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_ranger_quantreg_rf_bt_a1/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_rf_bt/*.csv")

# Kombiniere alle CSV-Dateien
csv_files = csv_files_r + csv_files_python

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

def plot_cumulative_crps(specification):
    """
    Plot the cumulative Squared Error for the three packages for a specific specification.
    """
    plt.figure(figsize=(14, 6), dpi=500)

    # Farbschema und Linienstil für die verschiedenen Pakete
    colors = {'ranger': 'b', 'quantregForest': 'g', 'sklearn': 'r'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}
    linestyles = {'ranger': '-', 'quantregForest': '-', 'sklearn': '-'}  # Verschiedene Linienstile
    
    # Überprüfen, ob die CSV-Dateien die spezifische Spezifikation enthalten
    for package in ['sklearn', 'ranger', 'quantregForest']:
        # Verwende einen Regex, um exakte Übereinstimmung zu erzwingen (Paketname + Spezifikation)
        pattern = re.compile(f"^{package}_{specification}\.csv$")
        filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
        print(f"Filtered files for {package} => {filtered_files}") 
        
        # Überprüfen, ob eine passende Datei gefunden wurde
        if filtered_files:
            # Lese die erste passende Datei ein 
            df = pd.read_csv(filtered_files[0])
            print(df)
            print(f"Columns in {package} data: {df.columns.tolist()}")

            # Konvertiere 'date_time' zu datetime-Objekten
            df['date_time'] = pd.to_datetime(df['date_time'])

            # Kumulierten SE berechnen
            df['cumulative_crps'] = (df['crps'].cumsum())
            print(df['cumulative_crps'].head(10))
            print(df['cumulative_crps'].tail(10))

            # Plot der kumulierten SE mit spezifischem Linienstil
            plt.plot(df['date_time'], df['cumulative_crps'], 
                     label=labels[package], 
                     color=colors[package], 
                     linestyle=linestyles[package],
                     alpha = 0.7)

    # Plot-Details
    plt.title(f'{specification}', fontsize=14, y = 1.1)
    plt.xlabel('Date', fontsize=12)

    plt.ylim(-0.02, 6.5e7)
    
    # Skalierung der Y-Achse zur besseren Sichtbarkeit von Unterschieden
    #plt.ylim(df['cumulative_crps'].min() - 100, df['cumulative_crps'].max() + 100)

    plt.ylabel('Cumulative CRPS', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=12)
    plt.grid(True)

    # Die Legende unter dem Titel platzieren und mit Padding versehen
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)  # ncol für mehrere Spalten
    #plt.tight_layout(pad=0.2)  # Pad vergrößert den Abstand zwischen Elementen

    # Plot anzeigen
    plt.show()


#%%
plot_cumulative_crps('nott_day_lagged_rf')
#%%
plot_cumulative_crps('nott_month_lagged_rf')
#%%
plot_cumulative_crps('nott_day_notlagged_rf')
#%%
plot_cumulative_crps('nott_day_notlagged_bt')
#%%
plot_cumulative_crps('nott_month_notlagged_rf')
#%%
plot_cumulative_crps('tt_day_notlagged_rf')
#%%
plot_cumulative_crps('tt_month_notlagged_rf')
#%%
plot_cumulative_crps('tt_month_notlagged_bt')


#%%
#=================================
# Plot ----
# Subplot (Part 1)
# Cumulative CRPS
# RF vs BT

csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_ranger_quantreg_rf_bt_a1/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_rf_bt/*.csv")

# Kombiniere alle CSV-Dateien
csv_files = csv_files_r + csv_files_python

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

def plot_cumulative_crps_subplot(specifications, overall_title):
    """
    Plot the cumulative Squared Error for the three packages for multiple specifications in subplots.
    Arrange 3 plots per row and only one legend at the top.
    """
    # Bestimme die Anzahl der Subplots basierend auf der Anzahl der Spezifikationen
    num_specs = len(specifications)
    num_cols = 3  # Drei Plots pro Reihe
    num_rows = (num_specs + num_cols - 1) // num_cols  # Berechne die nötige Anzahl an Zeilen

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 6 * num_rows), dpi=500)

    # Flatten der axes, falls es ein 2D-Array ist, um es einfach zu handhaben
    axes = axes.flatten()

    # Farbschema und Linienstil für die verschiedenen Pakete
    colors = {'ranger': 'b', 'quantregForest': 'g', 'sklearn': 'r'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}
    linestyles = {'ranger': '-', 'quantregForest': '-', 'sklearn': '-'}  # Verschiedene Linienstile

    # Erstelle eine Liste der Handles und Labels für die globale Legende
    handles = []
    legend_labels = []

    # Setze eine Variable, um sicherzustellen, dass jedes Paket nur einmal in die Legende aufgenommen wird
    added_packages = set()

    # Iteriere über die Spezifikationen und Plotte jedes Diagramm in einem eigenen Subplot
    for idx, specification in enumerate(specifications):
        ax = axes[idx]  # Hole die aktuelle Achse (Subplot)

        # Überprüfen, ob die CSV-Dateien die spezifische Spezifikation enthalten
        for package in ['sklearn', 'ranger', 'quantregForest']:
            # Verwende einen Regex, um exakte Übereinstimmung zu erzwingen (Paketname + Spezifikation)
            pattern = re.compile(f"^{package}_{specification}\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
            print(f"Filtered files for {package} => {filtered_files}") 
            
            # Überprüfen, ob eine passende Datei gefunden wurde
            if filtered_files:
                # Lese die erste passende Datei ein (wir nehmen an, dass es nur eine pro Paket und Spezifikation gibt)
                df = pd.read_csv(filtered_files[0])
                print(df)
                print(f"Columns in {package} data: {df.columns.tolist()}")

                # Konvertiere 'date_time' zu datetime-Objekten
                df['date_time'] = pd.to_datetime(df['date_time'])

                # Kumulierten SE berechnen
                df['cumulative_crps'] = df['crps'].cumsum()
                print(df['cumulative_crps'].head(10))
                print(df['cumulative_crps'].tail(10))
                cumulative_crps_last = df['cumulative_crps'].iloc[-1]
                cumulative_crps_last = df['cumulative_crps'].iloc[-1]

                mean_crps = np.sqrt(df['se'].mean())
                expected_cumulative_crps = mean_crps * 16449
                print("mean_crps:", mean_crps)

                print(f"Calculated cumulative SE at the last data point: {cumulative_crps_last}")
                print(f"Expected cumulative CRPS: {expected_cumulative_crps}")      


                # Plot der kumulierten SE mit spezifischem Linienstil auf der aktuellen Achse (Subplot)
                line, = ax.plot(df['date_time'], df['cumulative_crps'], 
                                color=colors[package], 
                                linestyle=linestyles[package],
                                alpha = 0.7)

                # Füge den Handle und Label der globalen Legende hinzu, aber nur wenn das Paket noch nicht hinzugefügt wurde
                if package not in added_packages:
                    handles.append(line)
                    legend_labels.append(labels[package])
                    added_packages.add(package)

        # **Individuellen Titel für jeden Subplot**
        ax.set_title(f'{specification}', fontsize=14, y = 1.0)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative SE', fontsize=12)
        ax.set_ylim(-0.02, 6.5e7)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)

    # Entferne unbenutzte Subplots, falls weniger als 3 * num_rows Plots erstellt wurden
    for idx in range(num_specs, len(axes)):
        axes[idx].axis('off')

    # **Gesamt-Titel für den gesamten Plot**
    plt.suptitle(overall_title, fontsize=16, y=1.02)

    # Erstelle eine globale Legende oben (nur einmal für alle Subplots)
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=14)

    # Das Layout der Subplots anpassen
    plt.tight_layout(pad=2.0)
    
    # Plot anzeigen
    plt.show()

specifications = [
    'nott_month_lagged_bt', 
    'nott_day_notlagged_bt', 
    'nott_month_notlagged_bt',
    'tt_day_notlagged_rf',
    'tt_day_lagged_rf',
    'tt_day_lagged_bt',
    'tt_month_lagged_bt',
    'tt_month_lagged_rf'
]

overall_title = "Cumulative SE comparison across packages"

# Funktion aufrufen und Subplots für alle Spezifikationen erstellen
plot_cumulative_crps_subplot(specifications, overall_title)


#%%
#=================================
# (Part 2)
# Liste der Spezifikationen
specifications = [
    'nott_day_lagged_rf',
    'nott_day_lagged_bt',

    'nott_month_lagged_rf', 
    'nott_day_notlagged_rf', 
    'tt_month_notlagged_rf', 
    'tt_month_notlagged_bt',

    'nott_month_notlagged_rf',
    'tt_day_notlagged_bt'
]

# Gesamt-Titel für den gesamten Plot
overall_title = "Cumulative SE comparison across packages"

# Funktion aufrufen und Subplots für alle Spezifikationen erstellen
plot_cumulative_crps_subplot(specifications, overall_title)



#%%

#%%
#=================================
# Plot ----
# Cumulative SE Plot als Subplot



specifications = [
    'nott_day_lagged_bt', 'nott_day_lagged_rf',
    'nott_month_lagged_bt', 'nott_month_lagged_rf',
    'tt_day_lagged_bt', 'tt_day_lagged_rf',
    'tt_month_lagged_bt', 'tt_month_lagged_rf',
    'nott_day_notlagged_bt', 'nott_day_notlagged_rf',
    'nott_month_notlagged_bt', 'nott_month_notlagged_rf',
    'tt_day_notlagged_bt', 'tt_day_notlagged_rf',
    'tt_month_notlagged_bt', 'tt_month_notlagged_rf'
]

csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_without_Intercept_Hyperpara_const/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_without_intercept_hyper_const/*.csv")
#csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_rf_bt/*.csv")


# Kombiniere alle CSV-Dateien
csv_files = csv_files_r + csv_files_python

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

def plot_cumulative_se(specification, ax):
    """
    Plot the cumulative Squared Error for the three packages for a specific specification on a given axis.
    """
    # Farbschema und Linienstil für die verschiedenen Pakete
    colors = {'ranger': 'b', 'quantregForest': 'g', 'sklearn': 'r'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}
    linestyles = {'ranger': '-', 'quantregForest': '-', 'sklearn': '-'}  # Verschiedene Linienstile
    
    # Überprüfen, ob die CSV-Dateien die spezifische Spezifikation enthalten
    for package in ['sklearn', 'ranger', 'quantregForest']:
        # Verwende einen Regex, um exakte Übereinstimmung zu erzwingen (Paketname + Spezifikation)
        pattern = re.compile(f"^{package}_{specification}\.csv$")
        filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
        print(f"Filtered files for {package} => {filtered_files}") 
        
        # Überprüfen, ob eine passende Datei gefunden wurde
        if filtered_files:
            # Lese die erste passende Datei ein (wir nehmen an, dass es nur eine pro Paket und Spezifikation gibt)
            df = pd.read_csv(filtered_files[0])
            print(df)
            print(f"Columns in {package} data: {df.columns.tolist()}")

            # Konvertiere 'date_time' zu datetime-Objekten
            df['date_time'] = pd.to_datetime(df['date_time'])

            # Kumulierten SE berechnen
            df['cumulative_se'] = df['se'].cumsum()
            print(df['cumulative_se'].head(10))
            print(df['cumulative_se'].tail(10))

            # Plot der kumulierten SE mit spezifischem Linienstil auf den Subplot (achse)
            ax.plot(df['date_time'], df['cumulative_se'], 
                     label=labels[package], 
                     color=colors[package], 
                     linestyle=linestyles[package])

    # Subplot-Details
    ax.set_title(f' {specification}', fontsize=12)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Cumulative SE', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8)
    ax.grid(True)

# Erstelle Subplots (4x4)
fig, axs = plt.subplots(4, 4, figsize=(16, 16), dpi=500)

# Durchlaufe die Spezifikationen und plotte sie in den Subplots
for i, specification in enumerate(specifications):
    ax = axs[i // 4, i % 4]  # Bestimme den Subplot (4x4)
    plot_cumulative_se(specification, ax)

# Tight layout um Überlappungen zu vermeiden
plt.tight_layout()

# Plot anzeigen
plt.show()
#%%

# %%
#=================================
# PLot ----
# CRPS plot for different mtry values ---
# Hyperparameters are constant set to the value of sklearn and quantregForest has no Intercept


# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

# Bei dem csv file ist min.node.size = 2
#csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_different_Mtry/*.csv")

# Bei dem csv file ist min.node.size = 1
#csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_different_mtry_a2/*.csv")
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_different_mtry_a1/*.csv")

csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_different_mtry/*.csv")

#csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_different_mtry/*.csv")
#csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_different_mtry_a2/*.csv")

csv_files = csv_files_r + csv_files_python




def plot_crps_vs_mtry(specifications_list):
    """
    Plot the mean CRPS values for each specification in a subplot grid with m_try values on the x-axis.

    """

    num_specs = len(specifications_list)
    num_cols = 4  # Number of columns in subplot grid
    num_rows = (num_specs + num_cols - 1) // num_cols  

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=500)
    axes = axes.flatten()

    # Overall title for the figure
    fig.suptitle('Mean CRPS for different m_try values (fixed train-test split)', fontsize=16)

    # Colors and labels for different packages
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}

    for i, specification in enumerate(specifications_list):
        ax = axes[i]
        
        for package in ['sklearn', 'ranger', 'quantregForest']:
            pattern = re.compile(f"^{package}_{specification}_mtry(\\d+)\\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
            print(filtered_files)

            mtry_values = []
            crps_values = []

            for file in filtered_files:
                print(f" Datei: {file}")
                mtry_match = re.search(r'_mtry(\d+)', file)
                if mtry_match:
                    mtry = int(mtry_match.group(1))
                    print(mtry)
                    df = pd.read_csv(file)
                    mean_crps = df['crps'].mean()

                    mtry_values.append(mtry)
                    crps_values.append(mean_crps)

            # Sort mtry and CRPS values for plotting
            sorted_indices = sorted(range(len(mtry_values)), key=lambda k: mtry_values[k])
            mtry_values = [mtry_values[idx] for idx in sorted_indices]
            crps_values = [crps_values[idx] for idx in sorted_indices]

            # Plot CRPS values over m_try for each package within a specification
            ax.plot(mtry_values, crps_values, marker='o', color=colors[package], 
                    label=labels[package], alpha=0.4)  # Set alpha for transparency

        # Set X-axis limits and ticks to ensure they are in steps of 1
        ax.set_xticks(range(min(mtry_values), max(mtry_values) + 1))  # +1 to include the max value

        # Add padding to the x-axis limits
        padding = 0.1  # Adjust the padding as needed
        ax.set_xlim(min(mtry_values) - padding, max(mtry_values) + padding)

        # Title and labels for each subplot
        ax.set_title(f'{specification}', fontsize=14)
        ax.set_xlabel('m_try')
        ax.set_ylabel('Mean CRPS')
        ax.set_yticks(range(0, 4001, 1000))
        ax.set_ylim(-0.02*4300, 4300)
        ax.grid(True)

    # Shared legend
    handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package]) for package in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors), bbox_to_anchor=(0.5, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.9])  
    plt.subplots_adjust(wspace=0.4, top=0.85)  
    plt.show()


plot_crps_vs_mtry(['nott_day_lagged', 'nott_month_lagged', 'tt_day_lagged', 'tt_month_lagged',
                    'nott_day_notlagged', 'nott_month_notlagged', 'tt_day_notlagged', 'tt_month_notlagged'])

#%%

# %%
#=================================
# PLOT ---
# Root Mean SE plot for different mtry values ---
# Hyperparameters are constant set to the value of sklearn and quantregForest has no Intercept

plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')


# Bei dem csv file ist min.node.size = 2
#csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_different_Mtry/*.csv")

# Bei dem csv file ist min.node.size = 1
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_different_mtry_a2/*.csv")

csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_different_mtry/*.csv")
#csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_different_mtry_a2/*.csv")

csv_files = csv_files_r + csv_files_python

def plot_se_vs_mtry(specifications_list):
    """
    Plot the mean SE values for each specification in a subplot grid with m_try values on the x-axis.

    """

    num_specs = len(specifications_list)
    num_cols = 4  # Number of columns in subplot grid
    num_rows = (num_specs + num_cols - 1) // num_cols  

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=500)
    axes = axes.flatten()

    # Overall title for the figure
    fig.suptitle(' Root Mean SE for different m_try values (fixed train-test split)', fontsize=16)

    # Colors and labels for different packages
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}

    for i, specification in enumerate(specifications_list):
        ax = axes[i]
        
        for package in ['sklearn', 'ranger', 'quantregForest']:
            pattern = re.compile(f"^{package}_{specification}_mtry(\\d+)\\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]

            mtry_values = []
            se_values = []

            for file in filtered_files:
                mtry_match = re.search(r'_mtry(\d+)', file)
                if mtry_match:
                    mtry = int(mtry_match.group(1))
                    df = pd.read_csv(file)
                    mean_se = np.sqrt(df['se'].mean())

                    mtry_values.append(mtry)
                    se_values.append(mean_se)

            # Sort mtry and CRPS values for plotting
            sorted_indices = sorted(range(len(mtry_values)), key=lambda k: mtry_values[k])
            mtry_values = [mtry_values[idx] for idx in sorted_indices]
            se_values = [se_values[idx] for idx in sorted_indices]

            # Plot CRPS values over m_try for each package within a specification
            ax.plot(mtry_values, se_values, marker='o', color=colors[package], 
                    label=labels[package], alpha=0.4)  # Set alpha for transparency

        # Set X-axis limits and ticks to ensure they are in steps of 1
        ax.set_xticks(range(min(mtry_values), max(mtry_values) + 1))  # +1 to include the max value

        # Add padding to the x-axis limits
        padding = 0.1  # Adjust the padding as needed
        ax.set_xlim(min(mtry_values) - padding, max(mtry_values) + padding)

        # Title and labels for each subplot
        ax.set_title(f'{specification}', fontsize=14)
        ax.set_xlabel('m_try')
        ax.set_ylabel('Root Mean SE ')
        #ax.set_yticks([0, 1e7, 2e7, 3e7, 4e7])
        ax.set_ylim(0, 7600)
        ax.set_ylim(-0.02 * 7600, 7600)  # 5% Padding unterhalb von 0

        ax.grid(True)

    # Shared legend
    handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package]) for package in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors), bbox_to_anchor=(0.5, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.9])  
    plt.subplots_adjust(wspace=0.4, top=0.85)  
    plt.show()

#%%

plot_se_vs_mtry(['nott_day_lagged', 'nott_month_lagged', 'tt_day_lagged', 'tt_month_lagged',
                    'nott_day_notlagged', 'nott_month_notlagged', 'tt_day_notlagged', 'tt_month_notlagged'])

#%%

#%%
#=================================
# PLOT ---
# AE plot for different mtry values ---
# Hyperparameters are constant set to the value of sklearn and quantregForest has no Intercept

plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_different_mtry_a2/*.csv")

csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_different_mtry/*.csv")

csv_files = csv_files_r + csv_files_python

def plot_ae_vs_mtry(specifications_list):
    """
    Plot the mean AE values for each specification in a subplot grid with m_try values on the x-axis.

    """

    num_specs = len(specifications_list)
    num_cols = 4  # Number of columns in subplot grid
    num_rows = (num_specs + num_cols - 1) // num_cols  

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=500)
    axes = axes.flatten()

    # Overall title for the figure
    fig.suptitle(' Mean AE for different m_try values (fixed train-test split)', fontsize=16)

    # Colors and labels for different packages
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}

    for i, specification in enumerate(specifications_list):
        ax = axes[i]
        
        for package in ['sklearn', 'ranger', 'quantregForest']:
            pattern = re.compile(f"^{package}_{specification}_mtry(\\d+)\\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
            print(f"Package: {package}, Specification: {specification}, Files: {filtered_files}")

            mtry_values = []
            se_values = []

            for file in filtered_files:
                mtry_match = re.search(r'_mtry(\d+)', file)
                if mtry_match:
                    mtry = int(mtry_match.group(1))
                    df = pd.read_csv(file)
                    mean_se = (df['ae'].mean())

                    mtry_values.append(mtry)
                    se_values.append(mean_se)

            # Sort mtry and CRPS values for plotting
            sorted_indices = sorted(range(len(mtry_values)), key=lambda k: mtry_values[k])
            mtry_values = [mtry_values[idx] for idx in sorted_indices]
            se_values = [se_values[idx] for idx in sorted_indices]

            # Plot CRPS values over m_try for each package within a specification
            ax.plot(mtry_values, se_values, marker='o', color=colors[package], 
                    label=labels[package], alpha=0.4)  # Set alpha for transparency

        # Set X-axis limits and ticks to ensure they are in steps of 1
        ax.set_xticks(range(min(mtry_values), max(mtry_values) + 1))  # +1 to include the max value

        # Add padding to the x-axis limits
        padding = 0.1  # Adjust the padding as needed
        ax.set_xlim(min(mtry_values) - padding, max(mtry_values) + padding)

        # Title and labels for each subplot
        ax.set_title(f'{specification}', fontsize=14)
        ax.set_xlabel('m_try')
        ax.set_ylabel('Mean AE ')
        #ax.set_yticks([0, 1e7, 2e7, 3e7, 4e7])
        ax.set_ylim(0, 6000)
        ax.set_ylim(-0.02 * 6000, 6000)  # 5% Padding unterhalb von 0

        ax.grid(True)

    # Shared legend
    handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package]) for package in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors), bbox_to_anchor=(0.5, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.9])  
    plt.subplots_adjust(wspace=0.4, top=0.85)  
    plt.show()

#%%

plot_ae_vs_mtry(['nott_day_lagged', 'nott_month_lagged', 'tt_day_lagged', 'tt_month_lagged',
                    'nott_day_notlagged', 'nott_month_notlagged', 'tt_day_notlagged', 'tt_month_notlagged'])


#%%


#%%
#=================================
# Plot ----
# Different mtry 
# 10 different seeds

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

# Alle CSV-Dateien aus den Seed- und Version-Ordnern laden
csv_files_r = []
for seed in range(1, 10):  # seed1 bis seed7
    csv_files_r += glob.glob(f"/home/siefert/projects/Masterarbeit/sophia_code/different_seed/res_different_mtry_a1_seed{seed}/*.csv")

csv_files_python = []
for version in range(1, 10):  # version1 bis version7
    csv_files_python += glob.glob(f"/home/siefert/projects/Masterarbeit/sophia_code/python_res_different_mtry_v{version}/*.csv")

# Gesamtdateiliste
csv_files = csv_files_r + csv_files_python


def plot_crps_vs_mtry(specifications_list):
    """
    Plot the mean CRPS values for each specification in a subplot grid with m_try values on the x-axis.
    """

    num_specs = len(specifications_list)
    num_cols = 4  # Anzahl der Spalten im Plot
    num_rows = (num_specs + num_cols - 1) // num_cols  

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=500)
    axes = axes.flatten()

    # Gesamt-Titel
    fig.suptitle('Mean CRPS for different m_try values (across 10 different seeds)', fontsize=16)

    # Farben und Labels für die Packages
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}

    for i, specification in enumerate(specifications_list):
        ax = axes[i]
        
        for package in ['sklearn', 'ranger', 'quantregForest']:
            pattern = re.compile(f"^{package}_{specification}_mtry(\\d+)\\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]

            print(f"Gefilterte Dateien für {specification} und {package}: {filtered_files}")

            mtry_crps = {}

            for file in filtered_files:
                mtry_match = re.search(r'_mtry(\d+)', file)
                if mtry_match:
                    mtry = int(mtry_match.group(1))
                    df = pd.read_csv(file)

                    if 'crps' in df.columns:
                        mean_crps = df['crps'].mean()
                    else:
                        print(f"Warnung: Keine 'crps'-Spalte in {file}")
                        continue

                    if mtry in mtry_crps:
                        mtry_crps[mtry].append(mean_crps)
                    else:
                        mtry_crps[mtry] = [mean_crps]

            # Mittelwerte über alle Seeds berechnen
            mtry_values = sorted(mtry_crps.keys())
            crps_values = [sum(mtry_crps[m]) / len(mtry_crps[m]) for m in mtry_values]

            # Plot für das Package
            ax.plot(mtry_values, crps_values, marker='o', color=colors[package], label=labels[package], alpha=0.4)

        # X-Achse anpassen
        if mtry_values:
            ax.set_xticks(range(min(mtry_values), max(mtry_values) + 1))  
            ax.set_xlim(min(mtry_values) - 0.1, max(mtry_values) + 0.1)

        # Titel und Achsenbeschriftung
        ax.set_title(f'{specification}', fontsize=14)
        ax.set_xlabel('m_try')
        ax.set_ylabel('Mean CRPS')
        ax.set_yticks(range(0, 4001, 1000))
        ax.set_ylim(-0.02 * 4300, 4300)
        ax.grid(True)

    # Gemeinsame Legende
    handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package]) for package in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors), bbox_to_anchor=(0.5, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.9])  
    plt.subplots_adjust(wspace=0.4, top=0.85)  
    plt.show()


# Aufruf mit allen Spezifikationen
plot_crps_vs_mtry(['nott_day_lagged', 'nott_month_lagged', 'tt_day_lagged', 'tt_month_lagged',
                    'nott_day_notlagged', 'nott_month_notlagged', 'tt_day_notlagged', 'tt_month_notlagged'])

#%%


#%%

# %%
#=================================
# Plot --------
# Zeigt jeweils für eines der drei pakete, die indiviudellen mean CRPS Werte und dann die Aggregation
# für sklearn
# für ranger
# für quantregForest


plt.rcParams['text.usetex'] = False
plt.style.use('seaborn-whitegrid')



csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_different_mtry_a/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/Results/Python/python_res_different_mtry/*.csv")

csv_files = csv_files_r + csv_files_python


def plot_crps_vs_mtry(specifications_list):
    """
    Plot the mean CRPS values for each specification considering only m_try = 1 and m_try = 5.
    Adjusts the spacing so that m_try = 1 and m_try = 5 are not equally far apart.
    """

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.suptitle('Mean CRPS for m_try = 1 and m_try = 5 (fixed train-test split)\n sklearn', fontsize=14, y=1.05)

    colors = {'nott_day_lagged': 'r', 'nott_month_lagged': 'b', 'tt_day_lagged': 'g', 'tt_month_lagged': 'purple'}
    
    adjusted_positions = {1: 1.2, 5: 4.2}  # Leichte Verschiebung

    mean_crps_mtry_1 = []  
    mean_crps_mtry_5 = []  

    for specification in specifications_list:
        for mtry in [1, 5]:  
            pattern = re.compile(f"^sklearn_{specification}_mtry{mtry}\\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
            
            crps_list = []

            for file in filtered_files:
                df = pd.read_csv(file)
                mean_crps = df['crps'].mean()
                crps_list.append(mean_crps)

            if crps_list:
                mean_crps = sum(crps_list) / len(crps_list)
                adjusted_x = adjusted_positions[mtry]  # Angepasste X-Position
                ax.scatter([adjusted_x] * len(crps_list), crps_list, color=colors[specification], alpha=0.6, s=50, label=specification if mtry == 1 else "")

                if mtry == 1:
                    mean_crps_mtry_1.extend(crps_list)
                else:
                    mean_crps_mtry_5.extend(crps_list)

    if mean_crps_mtry_1 and mean_crps_mtry_5:
        global_mean_1 = sum(mean_crps_mtry_1) / len(mean_crps_mtry_1)
        global_mean_5 = sum(mean_crps_mtry_5) / len(mean_crps_mtry_5)
        ax.scatter(adjusted_positions[1], global_mean_1, color='black', s=100, marker='X', label='Mean (m_try = 1 and 5) ')
        ax.scatter(adjusted_positions[5], global_mean_5, color='black', s=100, marker='X')

    ax.set_xticks([adjusted_positions[1], adjusted_positions[5]])
    ax.set_xticklabels([1, 5])
    ax.set_xlim(0.5, 5)  
    ax.set_ylim(0, 1500)  
    ax.set_xlabel('m_try')
    ax.set_ylabel('Mean CRPS')
    ax.grid(True)

    # Legende unter dem Titel
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=10, ncol=3, bbox_to_anchor=(0.5, 0.97)) 

    plt.tight_layout(rect=[0, 0, 1, 0.85])  
    plt.show()

# Spezifikationen nur mit "lagged"-Varianten
lagged_specs = ['nott_day_lagged', 'nott_month_lagged', 'tt_day_lagged', 'tt_month_lagged']

plot_crps_vs_mtry(lagged_specs)
# %%
# =========================
# Plot ---
# Für alle drei packete 
# für fixed-train-test split und mit feature lagged werden 
# individuellen und einzelnen CRPS Werte geplottet

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

csv_files = csv_files_r + csv_files_python

def plot_crps_vs_mtry_for_package(package, specifications_list):
    """
    Plot the mean CRPS values for each specification considering only m_try = 1 and m_try = 5.
    Adjusts the spacing so that m_try = 1 and m_try = 5 are not equally far apart for each package.
    """

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.suptitle(f'Mean CRPS for m_try = 1 and m_try = 5 (fixed train-test split)\n{package}', fontsize=14, y=1.05)

    colors = {'nott_day_lagged': 'r', 'nott_month_lagged': 'b', 'tt_day_lagged': 'g', 'tt_month_lagged': 'purple'}
    
    adjusted_positions = {1: 1.2, 5: 4.2}  # Leichte Verschiebung

    mean_crps_mtry_1 = []  
    mean_crps_mtry_5 = []  

    for specification in specifications_list:
        for mtry in [1, 5]:  
            pattern = re.compile(f"^{package}_{specification}_mtry{mtry}\\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
            
            crps_list = []

            for file in filtered_files:
                df = pd.read_csv(file)
                mean_crps = df['crps'].mean()
                crps_list.append(mean_crps)

            if crps_list:
                mean_crps = sum(crps_list) / len(crps_list)
                adjusted_x = adjusted_positions[mtry]  # Angepasste X-Position
                ax.scatter([adjusted_x] * len(crps_list), crps_list, color=colors[specification], alpha=0.6, s=50, label=specification if mtry == 1 else "")

                if mtry == 1:
                    mean_crps_mtry_1.extend(crps_list)
                else:
                    mean_crps_mtry_5.extend(crps_list)

    if mean_crps_mtry_1 and mean_crps_mtry_5:
        global_mean_1 = sum(mean_crps_mtry_1) / len(mean_crps_mtry_1)
        print(global_mean_1)
        global_mean_5 = sum(mean_crps_mtry_5) / len(mean_crps_mtry_5)
        print(global_mean_5)
        ax.scatter(adjusted_positions[1], global_mean_1, color='black', s=100, marker='X', label='Mean (m_try = 1 and 5) ')
        ax.scatter(adjusted_positions[5], global_mean_5, color='black', s=100, marker='X')

    ax.set_xticks([adjusted_positions[1], adjusted_positions[5]])
    ax.set_xticklabels([1, 5])
    ax.set_xlim(0.5, 5)  
    ax.set_ylim(0, 2500)  
    ax.set_xlabel('m_try')
    ax.set_ylabel('Mean CRPS')
    ax.grid(True)

    # Legende unter dem Titel
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=10, ncol=3, bbox_to_anchor=(0.5, 0.97)) 

    plt.tight_layout(rect=[0, 0, 1, 0.85])  
    plt.show()

# Spezifikationen nur mit "lagged"-Varianten
lagged_specs = ['nott_day_lagged', 'nott_month_lagged', 'tt_day_lagged', 'tt_month_lagged']

# Pakete, für die die Plots erstellt werden sollen
packages = ['sklearn', 'ranger', 'quantregForest']

# Für jedes Paket einen Plot erstellen
for package in packages:
    plot_crps_vs_mtry_for_package(package, lagged_specs)


#%%
#=================================

# Lade alle relevanten CSV-Dateien
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_different_mtry_a1/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_different_mtry_a4/*.csv")
csv_files = csv_files_r + csv_files_python

s_m1 = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/sklearn_mtry1.csv")
s_m5 = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/sklearn_mtry5.csv")

r_m1 = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/ranger_mtry1.csv")
r_m5 = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/ranger_mtry5.csv")

q_m1 = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/quantregForest_mtry1.csv")
q_m5 = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/quantregForest_mtry5.csv")

# Dictionary zur Speicherung der berechneten Mean-CRPS-Werte
mean_crps_results = {}

def plot_crps_vs_mtry_for_package(package, specifications_list, ax, colors, adjusted_positions):
    """
    Plot the mean CRPS values for each specification considering only m_try = 1 and m_try = 5
    in the provided axis (ax).
    """
    ax.set_title(f'{package}', fontsize=16)
    
    mean_crps_results[package] = {'mtry_1': [], 'mtry_5': []}  # Speichere Werte für das Package

    for specification in specifications_list:
        for mtry in [1, 5]:  
            pattern = re.compile(f"^{package}_{specification}_mtry{mtry}\\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
            
            crps_list = []

            for file in filtered_files:
                df = pd.read_csv(file)
                mean_crps = df['crps'].mean()
                crps_list.append(mean_crps)

            if crps_list:
                mean_crps = sum(crps_list) / len(crps_list)
                adjusted_x = adjusted_positions[mtry]  
                ax.scatter([adjusted_x] * len(crps_list), crps_list, color=colors[specification], 
                           alpha=0.8, s=100, marker='x', edgecolor='black', linewidth=2)

                mean_crps_results[package][f'mtry_{mtry}'].append(mean_crps)  # Speichere Mean-CRPS

    # Berechnung der kombinierten Mean-CRPS-Werte
    def get_mean_crps_combined(file_paths):
        crps_list = []
        for file in file_paths:
            df = pd.read_csv(file)
            crps_combined_mean = df['crps_combined'].mean()  
            crps_list.append(crps_combined_mean)
        if crps_list:
            return sum(crps_list) / len(crps_list)
        return None
    
    mean_crps_combined_m1, mean_crps_combined_m5 = None, None

    if package == 'sklearn':
        mean_crps_combined_m1 = get_mean_crps_combined(s_m1)
        mean_crps_combined_m5 = get_mean_crps_combined(s_m5)
    elif package == 'ranger':
        mean_crps_combined_m1 = get_mean_crps_combined(r_m1)
        mean_crps_combined_m5 = get_mean_crps_combined(r_m5)
    elif package == 'quantregForest':
        mean_crps_combined_m1 = get_mean_crps_combined(q_m1)
        mean_crps_combined_m5 = get_mean_crps_combined(q_m5)
    
    # Falls kombinierte Werte vorhanden sind, speichere sie ebenfalls
    if mean_crps_combined_m1 is not None:
        mean_crps_results[package]['combined_mtry_1'] = mean_crps_combined_m1
        ax.scatter(adjusted_positions[1], mean_crps_combined_m1, color='black', s=100, 
                   marker='x', edgecolor='black', linewidth=2, label=f'{package} Combined mtry=1')

    if mean_crps_combined_m5 is not None:
        mean_crps_results[package]['combined_mtry_5'] = mean_crps_combined_m5
        ax.scatter(adjusted_positions[5], mean_crps_combined_m5, color='black', s=100, 
                   marker='x', edgecolor='black', linewidth=2, label=f'{package} Combined mtry=5')

    ax.set_xticks([adjusted_positions[1], adjusted_positions[5]])
    ax.set_xticklabels([1, 5])
    ax.set_xlim(0.8, 2.7)
    ax.set_ylim(0, 2400)  
    ax.set_xlabel('m_try', fontsize=18)
    ax.set_ylabel('Mean CRPS', fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True)


def plot_crps_subplots(specifications_list):
    """
    Create subplots for each package, and add the legend only once for all subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=500)
    axes = axes.flatten()

    fig.suptitle(' Individual Forecasts vs Combined Forecast', fontsize=18)

    packages = ['sklearn', 'ranger', 'quantregForest']

    colors = {'nott_day_lagged': 'r', 'nott_month_lagged': 'b', 'tt_day_lagged': 'g', 'tt_month_lagged': 'purple'}
    
    adjusted_positions = {1: 1.1, 5: 2.4}

    for i, package in enumerate(packages):
        plot_crps_vs_mtry_for_package(package, specifications_list, axes[i], colors, adjusted_positions)

    # Die Marker in der Legende dicker machen
    handles = [plt.Line2D([0], [0], marker='x', color=color, markersize=10, linestyle='None', 
                          markeredgewidth=3, label=spec)
               for spec, color in colors.items()]
    
    # "Combined Forecast" in die Legende aufnehmen
    handles.append(plt.Line2D([0], [0], color='black', marker='x', markersize=12, 
                              markeredgewidth=3, label='combined distribution forecast', linestyle='None'))  

    # Erstelle die Legende
    fig.legend(handles, [handle.get_label() for handle in handles], loc='upper center', fontsize=16, ncol=5, bbox_to_anchor=(0.5, 0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.85])  
    plt.show()


lagged_specs = ['nott_day_lagged', 'nott_month_lagged', 'tt_day_lagged', 'tt_month_lagged']

plot_crps_subplots(lagged_specs)

# 🔹 Ausgabe der gespeicherten Mean-CRPS-Werte 🔹
print("\n======= MEAN CRPS RESULTS =======")
for package, results in mean_crps_results.items():
    print(f"\n{package}:")
    for key, value in results.items():
        print(f"  {key}: {value}")


#%%



# Plots ---
# ranger und quantregForest ---
# which show the estimated Quantiles -----

plt.rcParams['text.usetex'] = False

#forecast_quantiles = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/ranger_nott_day_lagged_mtry5.csv")
forecast_quantiles = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/quantile_predictions_R/quantregForest_nott_day_lagged_mtry5.csv")
forecast_quantiles = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/quantile_predictions_R/quantregForest_nott_day_lagged_mtry1.csv")



forecast_quantiles.iloc[:, 0] = pd.to_datetime(forecast_quantiles.iloc[:, 0])

# Die ersten 168 Stunden aus y_test extrahieren
y_subset = y_test[:168]

# Berechnung des Mittelwerts über die 100 Quantile
mean_quantile = forecast_quantiles.iloc[:168, 1:].mean(axis=1).values  # Alle Quantile (ab der 2. Spalte) und Mittelwert über alle Quantile

# Die Quantile extrahieren
q5 = forecast_quantiles.iloc[:168, forecast_quantiles.columns.get_loc("quantile..0.005")].values
q995 = forecast_quantiles.iloc[:168, forecast_quantiles.columns.get_loc("quantile..0.995")].values

plt.figure(figsize=(12, 6), dpi=500)

# Test Load (Echte Werte) plotten
plt.plot(range(len(y_subset)), y_subset, label='test load (y_test)', color='tab:blue')

# Quantil-Bereich einfärben
plt.fill_between(range(len(q5)), q5, q995, color='red', alpha=0.2, label='0.5%-99.5%')

# Mittelwert der Quantile plotten
plt.plot(range(len(mean_quantile)), mean_quantile, label='Mean of quantiles', color='purple', linestyle='-', linewidth=2, alpha = 0.4)

# Achsen und Titel
plt.xlabel('Hours', fontsize=14)
plt.ylabel('Energy Load', fontsize=14)
plt.title('quantregForest \n nott_day_lagged mtry = 5 ', fontsize=16, pad=30)
plt.ylim(35000, 80000)

# Layout anpassen
plt.tight_layout()

# Achsen anpassen und Spines setzen
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('lightgray')

ax.tick_params(axis='both', which='both', length=0)

plt.subplots_adjust(top=0.82)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=4, fontsize=12)

# Plot anzeigen
plt.show()

#%%
#=================================
# PLOT ---
# Quantie für entweder ranger oder quantregForest als Subplot

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False

# Daten laden
forecast_mtry5 = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/quantile_predictions_R/quantregForest_nott_month_notlagged_mtry4.csv")
forecast_mtry1 = pd.read_csv("/home/siefert/projects/Masterarbeit/sophia_code/quantile_predictions_R/quantregForest_nott_month_notlagged_mtry1.csv")

forecast_mtry5.iloc[:, 0] = pd.to_datetime(forecast_mtry5.iloc[:, 0])
forecast_mtry1.iloc[:, 0] = pd.to_datetime(forecast_mtry1.iloc[:, 0])

# Die ersten 168 Stunden aus y_test extrahieren
y_subset = y_test[:168]

# Funktion zur Erstellung eines Subplots
def plot_forecast(ax, forecast_quantiles, title, legend_handles=None, legend_labels=None, ylabel=True):
    mean_quantile = forecast_quantiles.iloc[:168, 1:].mean(axis=1).values
    q5 = forecast_quantiles.iloc[:168, forecast_quantiles.columns.get_loc("quantile..0.005")].values
    q995 = forecast_quantiles.iloc[:168, forecast_quantiles.columns.get_loc("quantile..0.995")].values
    
    line1, = ax.plot(range(len(y_subset)), y_subset, label='test load (y_test)', color='tab:blue', linewidth=2, alpha = 1)
    fill = ax.fill_between(range(len(q5)), q5, q995, color='red', alpha=0.2, label='0.5% - 99.5%')
    line2, = ax.plot(range(len(mean_quantile)), mean_quantile, label='mean of quantiles', color='purple', linestyle='-', linewidth=2, alpha=0.5)
    
    if legend_handles is not None and not legend_handles:
        legend_handles.extend([line1, fill, line2])
        legend_labels.extend(['true load', '0.5% - 99.5%', 'mean of quantiles'])
    
    ax.set_xlabel('Forecast Time - Hours', fontsize=14)
    if ylabel:
        ax.set_ylabel('Energy Load', fontsize=14)
        #ax.set_yticks([35000, 45000, 55000, 65000, 75000, 80000])
    else:
        ax.set_yticklabels([])
    
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_ylim(35000, 80000)
    ax.set_xlim(-5, 172)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    ax.spines['top'].set_edgecolor('lightgray')
    ax.spines['right'].set_edgecolor('lightgray')
    ax.spines['left'].set_edgecolor('lightgray')
    ax.spines['bottom'].set_edgecolor('lightgray')
    ax.tick_params(axis='both', which='both', length=0)

# Subplots erstellen
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=500)
legend_handles = []
legend_labels = []
plot_forecast(axes[0], forecast_mtry1, 'quantregForest \n nott_month_notlagged \n mtry = 1', legend_handles, legend_labels, ylabel=True,)
plot_forecast(axes[1], forecast_mtry5, 'quantregForest \n nott_month_notlagged \n mtry = 4', ylabel=False)

fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Durchschnittliche Breite des 0.05% und 99.5% Quantils
# Daten laden

# Zeitstempel umwandeln
forecast_mtry5.iloc[:, 0] = pd.to_datetime(forecast_mtry5.iloc[:, 0])
forecast_mtry1.iloc[:, 0] = pd.to_datetime(forecast_mtry1.iloc[:, 0])

# Berechnung der Differenz zwischen quantile..0.005 und quantile..0.995 für jedes Zeitintervall
q5_mtry5 = forecast_mtry5["quantile..0.005"].values  # Über den gesamten Testzeitraum
q995_mtry5 = forecast_mtry5["quantile..0.995"].values  # Über den gesamten Testzeitraum
q5_mtry1 = forecast_mtry1["quantile..0.005"].values  # Über den gesamten Testzeitraum
q995_mtry1 = forecast_mtry1["quantile..0.995"].values  # Über den gesamten Testzeitraum

# Berechnung der Breite des Quantils (Differenz zwischen 0.005 und 0.995 Quantilen)
width_mtry5 = q995_mtry5 - q5_mtry5
width_mtry1 = q995_mtry1 - q5_mtry1

# Berechnung der durchschnittlichen Breite des Quantils über den gesamten Zeitraum
avg_width_mtry5 = width_mtry5.mean()
avg_width_mtry1 = width_mtry1.mean()

# Ausgabe der durchschnittlichen Breite
print(f"Durchschnittliche Breite des Quantils (mtry = 5): {avg_width_mtry5:.2f}")
print(f"Durchschnittliche Breite des Quantils (mtry = 1): {avg_width_mtry1:.2f}")


#%%

#=================================
# Plot sklearn ----
q_hat_filepath = "/home/siefert/projects/Masterarbeit/sophia_code/sklearn_quantile_predictions_different_mtry/q_hat_mtry_1_['hour_int', 'weekday_int', 'holiday', 'month'].npz"
q_hat_filepath = "/home/siefert/projects/Masterarbeit/sophia_code/sklearn_quantile_predictions_different_mtry/q_hat_mtry_4_['hour_int', 'weekday_int', 'holiday',  'month'].npz"


loaded_data = np.load(q_hat_filepath)
q_hat = loaded_data["q_hat"]

n_quantiles = int(1e2)
grid_quantiles = (2 * (np.arange(1, n_quantiles + 1)) - 1) / (2 * n_quantiles)

# DataFrame erstellen mit den Quantilen als Spaltennamen
forecast_quantiles = pd.DataFrame(q_hat, columns=grid_quantiles[:q_hat.shape[1]])

# DataFrame anzeigen
print(forecast_quantiles)


forecast_quantiles.columns = forecast_quantiles.columns.astype(str)

# Die ersten 168 Stunden aus y_test extrahieren
y_subset = y_test[:168]

# Mittelwert der Quantile berechnen
mean_quantile = forecast_quantiles.iloc[:168, :].mean(axis=1).values  

# Die Quantile extrahieren
q5 = forecast_quantiles.iloc[:168, forecast_quantiles.columns.get_loc("0.005")].values
q995 = forecast_quantiles.iloc[:168, forecast_quantiles.columns.get_loc("0.995")].values

plt.figure(figsize=(12, 6), dpi=500)

# Test Load (Echte Werte) plotten
plt.plot(range(len(y_subset)), y_subset, label='test load (y_test)', color='tab:blue')

# Quantil-Bereich einfärben
plt.fill_between(range(len(q5)), q5, q995, color='red', alpha=0.2, label='0.5%-99.5%')

# Mittelwert der Quantile plotten
plt.plot(range(len(mean_quantile)), mean_quantile, label='mean of quantiles', color='purple', linestyle='-', linewidth=2, alpha = 0.4)

# Achsen und Titel
plt.xlabel('Forecast Horizon - Hours', fontsize=14)
plt.ylabel('Energy Load', fontsize=14)
plt.title('sklearn \n nott_day_lagged mtry = 1 ', fontsize=16, pad=30)
plt.ylim(35000, 80000)

# Layout anpassen
plt.tight_layout()

# Achsen anpassen und Spines setzen
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('lightgray')

ax.tick_params(axis='both', which='both', length=0)

plt.subplots_adjust(top=0.82)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=4, fontsize=12)

# Plot anzeigen
plt.show()

# %%
#=================================
# Plot Sklearn -----
# shows the predicted quantiles

plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

q_hat_filepaths = {
    "mtry = 1": "/home/siefert/projects/Masterarbeit/sophia_code/sklearn_quantile_predictions_different_mtry/q_hat_mtry_1_['hour_int', 'weekday_int', 'holiday', 'month_int'].npz",
    "mtry = 4": "/home/siefert/projects/Masterarbeit/sophia_code/sklearn_quantile_predictions_different_mtry/q_hat_mtry_4_['hour_int', 'weekday_int', 'holiday', 'month_int'].npz"
}

n_quantiles = int(1e2)
grid_quantiles = (2 * (np.arange(1, n_quantiles + 1)) - 1) / (2 * n_quantiles)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=500, sharey=True)

legend_handles = []
legend_labels = []

for ax, (title, q_hat_filepath) in zip(axes, q_hat_filepaths.items()):
    loaded_data = np.load(q_hat_filepath)
    q_hat = loaded_data["q_hat"]
    
    forecast_quantiles = pd.DataFrame(q_hat, columns=grid_quantiles[:q_hat.shape[1]])
    forecast_quantiles.columns = forecast_quantiles.columns.astype(str)
    
    y_subset = y_test[:168]
    mean_quantile = forecast_quantiles.iloc[:168, :].mean(axis=1).values  
    q5 = forecast_quantiles.iloc[:168, forecast_quantiles.columns.get_loc("0.005")].values
    q995 = forecast_quantiles.iloc[:168, forecast_quantiles.columns.get_loc("0.995")].values

    # Berechnung der Breite des Quantils (Differenz zwischen quantile 0.005 und 0.995)
    width_quantiles = q995 - q5
    avg_width_quantiles = width_quantiles.mean()  # Durchschnittliche Breite des Quantils
    
    # Ausgabe der durchschnittlichen Breite des Quantils
    print(f"Durchschnittliche Breite des Quantils für {title}: {avg_width_quantiles:.2f}")
    
    
    line1, = ax.plot(range(len(y_subset)), y_subset, label='test load (y_test)', color='tab:blue', linewidth = 2)
    fill = ax.fill_between(range(len(q5)), q5, q995, color='red', alpha=0.2, label='0.5% - 99.5%')
    line2, = ax.plot(range(len(mean_quantile)), mean_quantile, label='mean of quantiles', color='purple', linestyle='-', linewidth=2, alpha=0.4)
    
    if not legend_handles:
        legend_handles.extend([line1, fill, line2])
        legend_labels.extend(['true load', '0.5% - 99.5%', 'mean of quantiles'])
    
    ax.set_xlabel('Forecast Horizon - Hours', fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_title(f'sklearn \n nott_month_notlagged \n {title}', fontsize=18, pad=20)
    ax.set_ylim(35000, 80000)
    ax.set_xlim(-5, 172)
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgray')
    ax.tick_params(axis='both', which='both', length=0)
    
axes[0].set_ylabel('Energy Load', fontsize=14)
fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%

