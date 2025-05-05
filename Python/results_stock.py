#%%
import glob
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
# Pfad zu den CSV-Dateien 
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_stock_different_mtry/*.csv")
#csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_stock_different_Mtry/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_stock_different_m_try/*csv")


# Alle CSV-Dateien zusammenfügen
csv_files = csv_files_r + csv_files_python

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

# %%
# CRPS value ----
# Stock ---

def plot_crps_vs_mtry_split(specifications_list):
    """
    Plot the mean CRPS values for each specification in two separate subplot grids:
    - First 6 specifications in a 3x2 grid
    - Remaining specifications in a 6x4 grid
    """

    # Split the specifications list into two groups
    group1 = specifications_list[:6]  # First 6 specifications
    group2 = specifications_list[6:]  # Remaining specifications

    
    def plot_group(specifications, num_cols, num_rows, title):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=500)
        axes = axes.flatten()
        fig.suptitle(title, fontsize=16)

        colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
        labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}

        for i, specification in enumerate(specifications):
            ax = axes[i]

            mtry_values_dict = {'sklearn': [], 'ranger': [], 'quantregForest': []}
            crps_values_dict = {'sklearn': [], 'ranger': [], 'quantregForest': []}

            for package in ['sklearn', 'ranger', 'quantregForest']:
                pattern = re.compile(f"^{package}_{specification}_mtry(\\d+)\\.csv$")
                filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]

                for file in filtered_files:
                    mtry_match = re.search(r'_mtry(\d+)', file)
                    if mtry_match:
                        mtry = int(mtry_match.group(1))
                        df = pd.read_csv(file)
                        mean_crps = df['crps'].mean()
                        mtry_values_dict[package].append(mtry)
                        crps_values_dict[package].append(mean_crps)

            for package in ['sklearn', 'ranger', 'quantregForest']:
                mtry_values = mtry_values_dict[package]
                crps_values = crps_values_dict[package]

                if mtry_values:
                    sorted_indices = sorted(range(len(mtry_values)), key=lambda k: mtry_values[k])
                    mtry_values = [mtry_values[idx] for idx in sorted_indices]
                    crps_values = [crps_values[idx] for idx in sorted_indices]

                    ax.plot(mtry_values, crps_values, marker='o', color=colors[package],
                            label=labels[package], alpha=0.6)

            all_mtry_values = mtry_values_dict['sklearn'] + mtry_values_dict['ranger'] + mtry_values_dict['quantregForest']
            if all_mtry_values:
                ax.set_xticks(sorted(set(all_mtry_values)))
                padding = 0.4
                ax.set_xlim(min(all_mtry_values) - padding, max(all_mtry_values) + padding)

            ax.set_title(f' {specification}', fontsize=12)
            ax.set_xlabel('m_try', fontsize=12)
            ax.set_ylabel('Mean CRPS', fontsize=12)
            #ax.set_ylim(0, 1.3)
            ax.set_ylim(-0.02*1.3, 1.3)
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True)

        # Hide any unused subplots
        for j in range(len(specifications), len(axes)):
            fig.delaxes(axes[j])

        handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package])
                   for package in colors]
        fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors),
                   bbox_to_anchor=(0.5, 0.95))
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        #plt.subplots_adjust(wspace=0.2)
        plt.show()

    # Plot each group
    plot_group(group1, 3, 2, "Mean CRPS for different m_try values (first 6 stock data sets)")
    plot_group(group2, 4, 6, "Mean CRPS for different m_try values (remaining 24 stock data sets)")
#%%
# Call the function
plot_crps_vs_mtry_split(['AAPL', 'AMGN', 'AMZN', 
                         'AXP', 'BA', 'CAT', 'CRM',
                           'CSCO', 'CVX', 'DIS', 'GS', 
                           'HD', 'HON', 'IBM', 'JNJ', 
                           'JPM', 'KO', 'MCD', 'MMM', 'MRK', 
                           'MSFT', 'NKE', 'NVDA', 'PG', 'SHW', 
                           'TRV', 'UNH', 'V', 'VZ', 'WMT'])
# %%
#%%
# RMSE ---
# Stock ---

def plot_se_vs_mtry_split(specifications_list):
    """
    Plot the mean SE values for each specification in two separate subplot grids:
    - First 6 specifications in a 3x2 grid
    - Remaining specifications in a 6x4 grid
    """

    # Split the specifications list into two groups
    group1 = specifications_list[:6]  # First 6 specifications
    group2 = specifications_list[6:]  # Remaining specifications

    # Define a function to plot a group in a given grid layout
    def plot_group(specifications, num_cols, num_rows, title):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=500)
        axes = axes.flatten()
        fig.suptitle(title, fontsize=16)

        colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
        labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}

        for i, specification in enumerate(specifications):
            ax = axes[i]

            mtry_values_dict = {'sklearn': [], 'ranger': [], 'quantregForest': []}
            se_values_dict = {'sklearn': [], 'ranger': [], 'quantregForest': []}

            for package in ['sklearn', 'ranger', 'quantregForest']:
                pattern = re.compile(f"^{package}_{specification}_mtry(\\d+)\\.csv$")
                filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]

                for file in filtered_files:
                    mtry_match = re.search(r'_mtry(\d+)', file)
                    if mtry_match:
                        mtry = int(mtry_match.group(1))
                        df = pd.read_csv(file)
                        mean_se = np.sqrt(df['se'].mean())
                        mtry_values_dict[package].append(mtry)
                        se_values_dict[package].append(mean_se)

            for package in ['sklearn', 'ranger', 'quantregForest']:
                mtry_values = mtry_values_dict[package]
                se_values = se_values_dict[package]

                if mtry_values:
                    sorted_indices = sorted(range(len(mtry_values)), key=lambda k: mtry_values[k])
                    mtry_values = [mtry_values[idx] for idx in sorted_indices]
                    se_values = [se_values[idx] for idx in sorted_indices]

                    ax.plot(mtry_values, se_values, marker='o', color=colors[package],
                            label=labels[package], alpha=0.6)

            all_mtry_values = mtry_values_dict['sklearn'] + mtry_values_dict['ranger'] + mtry_values_dict['quantregForest']
            if all_mtry_values:
                ax.set_xticks(sorted(set(all_mtry_values)))
                padding = 0.4
                ax.set_xlim(min(all_mtry_values) - padding, max(all_mtry_values) + padding)

            ax.set_title(f' {specification}', fontsize=12)
            ax.set_xlabel('m_try', fontsize=12)
            ax.set_ylabel('RMSE', fontsize=12)
            #ax.set_ylim(0, 1.3)
            ax.set_ylim(-0.02*2.7, 2.7)
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True)

        # Hide any unused subplots
        for j in range(len(specifications), len(axes)):
            fig.delaxes(axes[j])

        handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package])
                   for package in colors]
        fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors),
                   bbox_to_anchor=(0.5, 0.95))
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        #plt.subplots_adjust(wspace=0.2)
        plt.show()

    # Plot each group
    plot_group(group1, 3, 2, "RMSE for different m_try values (first 6 stock data sets)")
    plot_group(group2, 4, 6, "RMSE for different m_try values (remaining 24 stock data sets)")
#%%
# Call the function 
plot_se_vs_mtry_split(['AAPL', 'AMGN', 'AMZN', 
                         'AXP', 'BA', 'CAT', 'CRM',
                           'CSCO', 'CVX', 'DIS', 'GS', 
                           'HD', 'HON', 'IBM', 'JNJ', 
                           'JPM', 'KO', 'MCD', 'MMM', 'MRK', 
                           'MSFT', 'NKE', 'NVDA', 'PG', 'SHW', 
                           'TRV', 'UNH', 'V', 'VZ', 'WMT'])

#%%
# =================
# MAE ---
# STock ---

def plot_ae_vs_mtry_split(specifications_list):
    """
    Plot the mean CRPS values for each specification in two separate subplot grids:
    - First 6 specifications in a 3x2 grid
    - Remaining specifications in a 6x4 grid
    """

    # Split the specifications list into two groups
    group1 = specifications_list[:6]  # First 6 specifications
    group2 = specifications_list[6:]  # Remaining specifications

    # Define a function to plot a group in a given grid layout
    def plot_group(specifications, num_cols, num_rows, title):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=500)
        axes = axes.flatten()
        fig.suptitle(title, fontsize=16)

        colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
        labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}

        for i, specification in enumerate(specifications):
            ax = axes[i]

            mtry_values_dict = {'sklearn': [], 'ranger': [], 'quantregForest': []}
            se_values_dict = {'sklearn': [], 'ranger': [], 'quantregForest': []}

            for package in ['sklearn', 'ranger', 'quantregForest']:
                pattern = re.compile(f"^{package}_{specification}_mtry(\\d+)\\.csv$")
                filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]

                for file in filtered_files:
                    mtry_match = re.search(r'_mtry(\d+)', file)
                    if mtry_match:
                        mtry = int(mtry_match.group(1))
                        df = pd.read_csv(file)
                        mean_se = (df['ae'].mean())
                        mtry_values_dict[package].append(mtry)
                        se_values_dict[package].append(mean_se)

            for package in ['sklearn', 'ranger', 'quantregForest']:
                mtry_values = mtry_values_dict[package]
                se_values = se_values_dict[package]

                if mtry_values:
                    sorted_indices = sorted(range(len(mtry_values)), key=lambda k: mtry_values[k])
                    mtry_values = [mtry_values[idx] for idx in sorted_indices]
                    se_values = [se_values[idx] for idx in sorted_indices]

                    ax.plot(mtry_values, se_values, marker='o', color=colors[package],
                            label=labels[package], alpha=0.6)

            all_mtry_values = mtry_values_dict['sklearn'] + mtry_values_dict['ranger'] + mtry_values_dict['quantregForest']
            if all_mtry_values:
                ax.set_xticks(sorted(set(all_mtry_values)))
                padding = 0.4
                ax.set_xlim(min(all_mtry_values) - padding, max(all_mtry_values) + padding)

            ax.set_title(f' {specification}', fontsize=12)
            ax.set_xlabel('m_try', fontsize = 12)
            ax.set_ylabel('MAE', fontsize = 12)
            ax.set_ylim(0, 2)
            ax.set_ylim(-0.02*2.0, 2.0)
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True)

        # Hide any unused subplots
        for j in range(len(specifications), len(axes)):
            fig.delaxes(axes[j])

        handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package])
                   for package in colors]
        fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors),
                   bbox_to_anchor=(0.5, 0.95))
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        #plt.subplots_adjust(wspace=0.2)
        plt.show()

    # Plot each group
    plot_group(group1, 3, 2, "MAE for different m_try values (first 6 stock data sets)")
    plot_group(group2, 4, 6, "MAE for different m_try values (remaining 24 stock data sets)")
#%%
# Call  the function
plot_ae_vs_mtry_split(['AAPL', 'AMGN', 'AMZN', 
                         'AXP', 'BA', 'CAT', 'CRM',
                           'CSCO', 'CVX', 'DIS', 'GS', 
                           'HD', 'HON', 'IBM', 'JNJ', 
                           'JPM', 'KO', 'MCD', 'MMM', 'MRK', 
                           'MSFT', 'NKE', 'NVDA', 'PG', 'SHW', 
                           'TRV', 'UNH', 'V', 'VZ', 'WMT'])
# %%

