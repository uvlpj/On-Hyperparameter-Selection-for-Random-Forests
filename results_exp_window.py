#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re


#%%
from data_preprocessor import *
# %%
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Pfad zu den CSV-Dateien (alle CSV-Dateien im Ordner "res")
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Results/R/res_different_mtry_exp_window_a/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_different_mtry_expanding/*.csv") 

csv_files_linear = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_expanding_linear/*.csv")  

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

#%%
#csv_files = csv_files_r + csv_files_python

csv_files = csv_files_r + csv_files_python + csv_files_linear


#%%

def plot_crps_vs_mtry(csv_files, specifications_list):
    """
    Plot the mean CRPS values for each specification in a subplot grid with m_try values on the x-axis.
    """
    num_specs = len(specifications_list)
    num_cols = 4  # Number of columns in subplot grid
    num_rows = (num_specs + num_cols - 1) // num_cols  # Calculate number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=500)
    axes = axes.flatten()

    # Title for the entire figure
    #fig.suptitle('Mean CRPS for different m_try values (expanding window)', fontsize=16)

    fig.suptitle(
    'Mean CRPS for different m_try values (expanding window)', 
    fontsize=16, 
    y=1.3  # Verschiebt den Titel weiter nach oben
)

    # Colors and labels for different packages
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}

    # Iterate through each specification
    for i, specification in enumerate(specifications_list):
        ax = axes[i]

        # Initialize data structures to hold aggregated CRPS values
        aggregated_crps = {mtry: [] for mtry in range(1, 11)}  # Assuming mtry ranges from 1 to 10

        # For each package (sklearn, ranger, quantregForest)
        for package in ['sklearn', 'ranger', 'quantregForest']:
            # Filter the CSV files that match the current package and specification
            pattern = re.compile(f"^{package}_{specification}_mtry(\\d+)_\\d{{4}}-\\d{{2}}-\\d{{2}}\\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]

            # Debug: print the filtered files for each package and specification
            print(f"Files found for {package} with specification {specification}:")
            print(filtered_files)

            # Process each file
            for file in filtered_files:
                # Extract m_try from filename
                mtry_match = re.search(r'_mtry(\d+)', file)
                if mtry_match:
                    mtry = int(mtry_match.group(1))
                    df = pd.read_csv(file)

                    # Calculate the mean CRPS for this file
                    mean_crps = df['crps'].mean()  # Assuming the CRPS column is named 'crps'
                    
                    # Append the mean CRPS value to the corresponding mtry group
                    aggregated_crps[mtry].append(mean_crps)

            # For each mtry value, calculate the average CRPS over all timeframes (files)
            avg_crps_per_mtry = {mtry: np.mean(crps) for mtry, crps in aggregated_crps.items() if crps}

            # Plot the average CRPS values over mtry for the current package
            mtry_values = list(avg_crps_per_mtry.keys())
            crps_values = list(avg_crps_per_mtry.values())

            # Plot CRPS values over m_try for the current package
            ax.plot(mtry_values, crps_values, marker='o', color=colors[package], 
                    label=labels[package], alpha=0.5)  # Set alpha for transparency

            # Calculate and print the average CRPS for the package across all m_try values
            overall_avg_crps = np.mean(crps_values)
            print(f"Package: {package} | Specification: {specification} | Average CRPS: {overall_avg_crps:.2f}")

        # Set X-axis limits and ticks, if mtry_values is not empty
        ax.set_xticks(mtry_values)
        ax.set_xlim(min(mtry_values) - 0.1, max(mtry_values) + 0.1)  # Add padding to x-axis
        ax.set_xticklabels(mtry_values)  # Ensure that mtry labels are correctly placed

        # Title and labels for each subplot
        ax.set_title(f'{specification}', fontsize=14)
        ax.set_xlabel('m_try')
        ax.set_ylabel('Mean CRPS')
        ax.set_yticks(range(0, 5501, 1000))
        ax.set_ylim(-0.02*5000, 5500)  # Adjust according to the CRPS range
        ax.grid(True)

    # Shared legend
    handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package]) for package in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors), bbox_to_anchor=(0.5, 1.1))

    plt.tight_layout(rect=[0, 0, 1, 0.9])  
    plt.subplots_adjust(wspace=0.4, top=0.85)  
    plt.show()



specifications_list = ['nott_day', 'nott_month', 'tt_day', 'tt_month']  
plot_crps_vs_mtry(csv_files, specifications_list)

#%%


#%%

def plot_crps_vs_mtry_linear(csv_files, csv_files_linear, specifications_list):
    """
    Plot the mean CRPS values for each specification in a subplot grid with m_try values on the x-axis.
    Additionally, add the mean CRPS value of the linear model (without mtry) as a horizontal line.
    """
    num_specs = len(specifications_list)
    num_cols = 4  # Number of columns in subplot grid
    num_rows = (num_specs + num_cols - 1) // num_cols  # Calculate number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=500)
    axes = axes.flatten()

    fig.suptitle(
        'RMSE for different m_try values (expanding window)', 
        fontsize=16, 
        y=1.3  # Verschiebt den Titel weiter nach oben
    )

    # Colors and labels for different packages
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g', 'linear': 'k'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest', 'linear': 'linear model'}

    # Iterate through each specification
    for i, specification in enumerate(specifications_list):
        ax = axes[i]

        # Initialize data structures to hold aggregated CRPS values
        aggregated_crps = {mtry: [] for mtry in range(1, 11)}  # Assuming mtry ranges from 1 to 10

        # For each package (sklearn, ranger, quantregForest)
        for package in ['sklearn', 'ranger', 'quantregForest']:
            # Filter the CSV files that match the current package and specification
            pattern = re.compile(f"^{package}_{specification}_mtry(\\d+)_\\d{{4}}-\\d{{2}}-\\d{{2}}\\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]

            # Process each file
            for file in filtered_files:
                # Extract m_try from filename
                mtry_match = re.search(r'_mtry(\d+)', file)
                if mtry_match:
                    mtry = int(mtry_match.group(1))
                    df = pd.read_csv(file)

                    # Calculate the mean CRPS for this file
                    mean_crps = np.sqrt(df['se'].mean())  # Assuming the CRPS column is named 'crps'
                    
                    # Append the mean CRPS value to the corresponding mtry group
                    aggregated_crps[mtry].append(mean_crps)

            # For each mtry value, calculate the average CRPS over all timeframes (files)
            avg_crps_per_mtry = {mtry: np.mean(crps) for mtry, crps in aggregated_crps.items() if crps}

            # Plot the average CRPS values over mtry for the current package
            mtry_values = list(avg_crps_per_mtry.keys())
            crps_values = list(avg_crps_per_mtry.values())

            # Plot CRPS values over m_try for the current package
            ax.plot(mtry_values, crps_values, marker='o', color=colors[package], 
                    label=labels[package], alpha=0.7)  # Set alpha for transparency

            # Calculate and print the average CRPS for the package across all m_try values
            overall_avg_crps = np.mean(crps_values)
            print(f"Package: {package} | Specification: {specification} | Average CRPS: {overall_avg_crps:.2f}")

        # Now handle the linear model data (without mtry, aggregated over time)
        linear_crps_values = []  # List to hold CRPS values for the linear model

        # Filter the CSV files for the linear model and specification
        linear_pattern = re.compile(f"^sklearn_linear_{specification}_\\d{{4}}-\\d{{2}}-\\d{{2}}\\.csv$")
        linear_files = [file for file in csv_files_linear if linear_pattern.search(os.path.basename(file))]

        # Debugging: Print the linear files being found
        print(f"Linear files for specification {specification}: {linear_files}")

        # Process each linear model file
        for file in linear_files:
            df = pd.read_csv(file)

            # Calculate the mean CRPS for this file
            mean_crps = np.sqrt(df['se'].mean())  # Assuming the CRPS column is named 'crps'
            linear_crps_values.append(mean_crps)

        # Debugging: Check if the linear CRPS values are being collected
        print(f"Linear CRPS values for specification {specification}: {linear_crps_values}")

        # Calculate the overall mean CRPS for the linear model (over all timeframes)
        if linear_crps_values:
            mean_crps_linear = np.mean(linear_crps_values)
            # Plot the mean CRPS for the linear model as a horizontal line
            ax.axhline(y=mean_crps_linear, color=colors['linear'], linestyle='-', label=labels['linear'], alpha = 0.8)
        else:
            print(f"No data for linear model in specification {specification}")

        # Set X-axis limits and ticks, if mtry_values is not empty
        ax.set_xticks(mtry_values)
        ax.set_xlim(min(mtry_values) - 0.1, max(mtry_values) + 0.1)  # Add padding to x-axis
        ax.set_xticklabels(mtry_values)  # Ensure that mtry labels are correctly placed

        # Title and labels for each subplot
        ax.set_title(f'{specification}', fontsize=14)
        ax.set_xlabel('m_try')
        ax.set_ylabel('RMSE')
        ax.set_ylim(-0.02*9000, 9000)  # Adjust according to the CRPS range
        ax.set_yticks(range(0, 9001, 1000))
        ax.grid(True)

    # Shared legend
    handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package]) for package in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors), bbox_to_anchor=(0.5, 1.1))

    plt.tight_layout(rect=[0, 0, 1, 0.9])  
    plt.subplots_adjust(wspace=0.4, top=0.85)  
    plt.show()

#%%
specifications_list = ['nott_day', 'nott_month', 'tt_day', 'tt_month']
plot_crps_vs_mtry_linear(csv_files, csv_files_linear, specifications_list)



#%%


# %%
def plot_se_vs_mtry(csv_files, specifications_list):
    """
    Plot the mean SE values for each specification in a subplot grid with m_try values on the x-axis.
    """
    num_specs = len(specifications_list)
    num_cols = 4  # Number of columns in subplot grid
    num_rows = (num_specs + num_cols - 1) // num_cols  # Calculate number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=300)
    axes = axes.flatten()

    # Title for the entire figure
    #fig.suptitle('Mean CRPS for different m_try values (expanding window)', fontsize=16)

    fig.suptitle(
    'Root Mean SE for different m_try values (expanding window)', 
    fontsize=16, 
    y=1.3  # Verschiebt den Titel weiter nach oben
)

    # Colors and labels for different packages
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}

    # Iterate through each specification
    for i, specification in enumerate(specifications_list):
        ax = axes[i]

        # Initialize data structures to hold aggregated CRPS values
        aggregated_crps = {mtry: [] for mtry in range(1, 11)}  # Assuming mtry ranges from 1 to 10

        # For each package (sklearn, ranger, quantregForest)
        for package in ['sklearn', 'ranger', 'quantregForest']:
            # Filter the CSV files that match the current package and specification
            pattern = re.compile(f"^{package}_{specification}_mtry(\\d+)_\\d{{4}}-\\d{{2}}-\\d{{2}}\\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]

            # Debug: print the filtered files for each package and specification
            print(f"Files found for {package} with specification {specification}:")
            print(filtered_files)

            # Process each file
            for file in filtered_files:
                # Extract m_try from filename
                mtry_match = re.search(r'_mtry(\d+)', file)
                if mtry_match:
                    mtry = int(mtry_match.group(1))
                    df = pd.read_csv(file)

                    # Calculate the mean CRPS for this file
                    mean_crps = np.sqrt(df['se'].mean())  # Assuming the CRPS column is named 'crps'
                    
                    # Append the mean CRPS value to the corresponding mtry group
                    aggregated_crps[mtry].append(mean_crps)

            # For each mtry value, calculate the average CRPS over all timeframes (files)
            avg_crps_per_mtry = {mtry: np.mean(crps) for mtry, crps in aggregated_crps.items() if crps}

            # Plot the average CRPS values over mtry for the current package
            mtry_values = list(avg_crps_per_mtry.keys())
            crps_values = list(avg_crps_per_mtry.values())

            # Plot CRPS values over m_try for the current package
            ax.plot(mtry_values, crps_values, marker='o', color=colors[package], 
                    label=labels[package], alpha=0.5)  # Set alpha for transparency

            # Calculate and print the average CRPS for the package across all m_try values
            overall_avg_crps = np.mean(crps_values)
            print(f"Package: {package} | Specification: {specification} | Average SE: {overall_avg_crps:.2f}")

        # Set X-axis limits and ticks, if mtry_values is not empty
        ax.set_xticks(mtry_values)
        ax.set_xlim(min(mtry_values) - 0.1, max(mtry_values) + 0.1)  # Add padding to x-axis
        ax.set_xticklabels(mtry_values)  # Ensure that mtry labels are correctly placed

        # Title and labels for each subplot
        ax.set_title(f'{specification}', fontsize=14)
        ax.set_xlabel('m_try')
        ax.set_ylabel('Mean SE')
        ax.set_yticks(range(0, 5500, 1000))
        ax.set_ylim(-0.02*5500, 5500)  # Adjust according to the CRPS range
        ax.grid(True)

    # Shared legend
    handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package]) for package in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=14, ncol=len(colors), bbox_to_anchor=(0.5, 1.1))

    plt.tight_layout(rect=[0, 0, 1, 0.9])  
    plt.subplots_adjust(wspace=0.4, top=0.85)  
    plt.show()


#%%
specifications_list = ['nott_day', 'nott_month', 'tt_day', 'tt_month']  # Example list of specifications
plot_se_vs_mtry(csv_files, specifications_list)

#%%


# %%
def plot_se_vs_mtry_linear(csv_files, csv_files_linear, specifications_list):
    """
    Plot the mean CRPS values for each specification in a subplot grid with m_try values on the x-axis.
    Additionally, add the mean CRPS value of the linear model (without mtry) as a horizontal line.
    """
    num_specs = len(specifications_list)
    num_cols = 4  # Number of columns in subplot grid
    num_rows = (num_specs + num_cols - 1) // num_cols  # Calculate number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=300)
    axes = axes.flatten()

    fig.suptitle(
        'Mean SE for different m_try values (expanding window)', 
        fontsize=16, 
        y=1.3  # Verschiebt den Titel weiter nach oben
    )

    # Colors and labels for different packages
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g', 'linear': 'k'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest', 'linear': 'linear model'}

    # Iterate through each specification
    for i, specification in enumerate(specifications_list):
        ax = axes[i]

        # Initialize data structures to hold aggregated CRPS values
        aggregated_se = {mtry: [] for mtry in range(1, 6)}  

        # For each package (sklearn, ranger, quantregForest)
        for package in ['sklearn', 'ranger', 'quantregForest']:
            # Filter the CSV files that match the current package and specification
            pattern = re.compile(f"^{package}_{specification}_mtry(\\d+)_\\d{{4}}-\\d{{2}}-\\d{{2}}\\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]

            # Process each file
            for file in filtered_files:
                # Extract m_try from filename
                mtry_match = re.search(r'_mtry(\d+)', file)
                if mtry_match:
                    mtry = int(mtry_match.group(1))
                    df = pd.read_csv(file)

                    # Calculate the mean CRPS for this file
                    mean_se = df['se'].mean()  # Assuming the CRPS column is named 'crps'
                    
                    # Append the mean CRPS value to the corresponding mtry group
                    aggregated_se[mtry].append(mean_se)

            # For each mtry value, calculate the average CRPS over all timeframes (files)
            avg_se_per_mtry = {mtry: np.mean(se) for mtry, se in aggregated_se.items() if se}

            # Plot the average CRPS values over mtry for the current package
            mtry_values = list(avg_se_per_mtry.keys())
            se_values = list(avg_se_per_mtry.values())

            # Plot CRPS values over m_try for the current package
            ax.plot(mtry_values, se_values, marker='o', color=colors[package], 
                    label=labels[package], alpha=0.7)  # Set alpha for transparency

            # Calculate and print the average CRPS for the package across all m_try values
            overall_avg_se = np.mean(se_values)
            print(f"Package: {package} | Specification: {specification} | Average CRPS: {overall_avg_se:.2f}")

        # Now handle the linear model data (without mtry, aggregated over time)
        linear_se_values = []  # List to hold CRPS values for the linear model

        # Filter the CSV files for the linear model and specification
        linear_pattern = re.compile(f"^sklearn_linear_{specification}_\\d{{4}}-\\d{{2}}-\\d{{2}}\\.csv$")
        linear_files = [file for file in csv_files_linear if linear_pattern.search(os.path.basename(file))]

        # Debugging: Print the linear files being found
        print(f"Linear files for specification {specification}: {linear_files}")

        # Process each linear model file
        for file in linear_files:
            df = pd.read_csv(file)

            # Calculate the mean CRPS for this file
            mean_se = df['se'].mean()  # Assuming the CRPS column is named 'crps'
            linear_se_values.append(mean_se)

        # Debugging: Check if the linear CRPS values are being collected
        print(f"Linear SE values for specification {specification}: {linear_se_values}")

        # Calculate the overall mean CRPS for the linear model (over all timeframes)
        if linear_se_values:
            mean_se_linear = np.mean(linear_se_values)
            # Plot the mean CRPS for the linear model as a horizontal line
            ax.axhline(y=mean_se_linear, color=colors['linear'], linestyle='-', label=labels['linear'], alpha = 0.7)
        else:
            print(f"No data for linear model in specification {specification}")

        # Set X-axis limits and ticks, if mtry_values is not empty
        ax.set_xticks(mtry_values)
        ax.set_xlim(min(mtry_values) - 0.1, max(mtry_values) + 0.1)  # Add padding to x-axis
        ax.set_xticklabels(mtry_values)  # Ensure that mtry labels are correctly placed

        # Title and labels for each subplot
        ax.set_title(f'Mean SE for {specification}', fontsize=10)
        ax.set_xlabel('m_try')
        ax.set_ylabel('Mean CRPS')
        ax.set_ylim(0, 75000000)  # Adjust according to the CRPS range
        ax.grid(True)

    # Shared legend
    handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package]) for package in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=10, ncol=len(colors), bbox_to_anchor=(0.5, 1.1))

    plt.tight_layout(rect=[0, 0, 1, 0.9])  
    plt.subplots_adjust(wspace=0.4, top=0.85)  
    plt.show()

#%%
plot_se_vs_mtry_linear(csv_files, csv_files_linear, specifications_list)

# %%
# Funktion zum Plotten der CRPS-Werte für jedes Datum in Subplots ---


def plot_crps_vs_mtry_date(csv_files, specification):
    """
    Plot the CRPS values for each date (timeFrame) in separate subplots showing the CRPS for different m_try values.
    """
    # Colors and labels for different packages
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}
    
    # Wir erstellen für jedes Datum (timeFrame) einen eigenen Plot
    date_grouped_files = {}

    # Filtere nur die Dateien, die zur Spezifikation 'nott_day' gehören
    filtered_files = [file for file in csv_files if f"nott_day" in file]

    # Gruppiere die gefilterten Dateien nach Jahr-Monat
    for file in filtered_files:
        # Extrahiere das Jahr-Monat (YYYY-MM) aus dem Dateinamen
        pattern = re.compile(r"_(\d{4}-\d{2})-\d{2}\.csv$")
        date_match = pattern.search(file)
        if date_match:
            date = date_match.group(1)  # Nur Jahr-Monat

            if date not in date_grouped_files:
                date_grouped_files[date] = []
            date_grouped_files[date].append(file)

    # Sortiere die Dateien nach Jahr-Monat (ascending order)
    sorted_dates = sorted(date_grouped_files.keys())

    # Berechne die Anzahl der Zeilen und Spalten für die Subplots
    num_dates = len(sorted_dates)
    num_cols = 6  # Beispiel: 5 Spalten
    num_rows = (num_dates + num_cols - 1) // num_cols  # Berechne die Anzahl der Zeilen, um alle zu passen

    # Erstelle Subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), dpi=300)
    axes = axes.flatten()  # Flach machen, um durch die Subplots zu iterieren

    # Große Überschrift für die gesamte Abbildung
    fig.suptitle('Mean CRPS for different m_try value (expanding window)\nnott_day', fontsize=16, y=1.05)


    # Plot für jedes Jahr-Monat (TimeFrame)
    for idx, date in enumerate(sorted_dates):
        ax = axes[idx]
        ax.set_title(f"{date}", fontsize=14)
        ax.set_xlabel('m_try', fontsize=12)
        ax.set_ylabel('Mean CRPS', fontsize=12)
        ax.grid(True)

        # Aggregiere CRPS für jedes m_try für jedes Paket
        aggregated_crps = {package: {mtry: [] for mtry in range(1, 11)} for package in colors}  # Für jedes Paket

        # Verarbeite jede Datei und berechne den Mean CRPS
        for file in date_grouped_files[date]:
            # Extrahiere Paket und mtry aus dem Dateinamen
            for package in colors:
                if f"{package}" in file:
                    mtry_match = re.search(r'_mtry(\d+)', file)
                    if mtry_match:
                        mtry = int(mtry_match.group(1))
                        df = pd.read_csv(file)

                        # Berechne den Mean CRPS für diese Datei
                        mean_crps = df['crps'].mean()  # Wir gehen davon aus, dass die Spalte 'crps' heißt

                        # Füge den CRPS-Wert der Liste für den entsprechenden mtry-Wert und Paket hinzu
                        aggregated_crps[package][mtry].append(mean_crps)

        # Plot für jedes Paket
        for package, crps_data in aggregated_crps.items():
            # Berechne den durchschnittlichen CRPS für jedes mtry
            avg_crps_per_mtry = {mtry: np.mean(crps) for mtry, crps in crps_data.items() if crps}

            print(f"Date: {date}, Package: {package}")
            for mtry, avg_crps in avg_crps_per_mtry.items():
                print(f"  m_try: {mtry}, Mean CRPS: {avg_crps:.2f}")

            # Plot für jedes mtry
            mtry_values = list(avg_crps_per_mtry.keys())
            crps_values = list(avg_crps_per_mtry.values())

            ax.plot(mtry_values, crps_values, marker='o', label=f"{labels[package]}", color=colors[package], alpha=0.7)

        # Setze Achsenlimits und -ticks
        ax.set_xticks(mtry_values)
        ax.set_xlim(min(mtry_values) - 0.2, max(mtry_values) + 0.2)  # Füge etwas Puffer zur X-Achse hinzu
        ax.set_xticklabels(mtry_values)  # Achte darauf, dass die mtry-Beschriftungen korrekt angezeigt werden
        ax.set_ylim(-0.02*6500, 6500)

    # Legende oben unter dem Titel
    handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package]) for package in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=12, ncol=len(colors), bbox_to_anchor=(0.5, 1.00))

    # Entferne unnötige leere Subplots, falls nicht alle verwendet wurden
    for idx in range(num_dates, len(axes)):
        fig.delaxes(axes[idx])

    # Layout anpassen und anzeigen
    plt.tight_layout()
    plt.show()

#%%
# Spezifikation 'nott_day' angeben
#specification = 'nott_day'
specification = 'nott_day'
#specification = 'tt_month'
#specification = 'nott_month'

# Plot die CRPS-Werte für die Spezifikation 'nott_day' in Subplots
plot_crps_vs_mtry_date(csv_files, specification)
# %%
