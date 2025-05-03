#%%
import glob
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
#%%

csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Data/Results/R_Results/res_simulated_data_different_mtry/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/Data/Results/Python_Results/python_res_simulated_data_different_mtry/*.csv")

# Alle CSV-Dateien zusammenfügen
csv_files = csv_files_r + csv_files_python

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

#%%

# %%

import glob
import re
import os
import pandas as pd
import matplotlib.pyplot as plt

# Liste der CSV-Dateien
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Data/Results/R_Results/res_simulated_data_different_mtry2/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/sophia_code/python_res_simulated_data_different_mtry2/*.csv")

# Alle CSV-Dateien zusammenfügen
csv_files = csv_files_r + csv_files_python

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

# Funktion zum Berechnen des Mittelwerts des CRPS und Extrahieren von mtry und Modell aus dem Dateinamen
def plot_crps_vs_mtry():
    """
    Plot the mean CRPS values for each model (sklearn, ranger, quantregForest) in a subplot grid with m_try values on the x-axis.
    """

    # Farben und Bezeichner für die Modelle
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}

    # Initialisiere ein Dictionary, um mtry und CRPS-Werte für jedes Modell zu speichern
    mtry_values_dict = {'sklearn': [], 'ranger': [], 'quantregForest': []}
    crps_values_dict = {'sklearn': [], 'ranger': [], 'quantregForest': []}
    
    for package in ['sklearn', 'ranger', 'quantregForest']:
        # Filtere die relevanten Dateien für jedes Modell
        pattern = re.compile(f"^{package}_mtry(\\d+)\\.csv$")
        filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]

        for file in filtered_files:
            mtry_match = re.search(r'mtry(\d+)', os.path.basename(file))
            if mtry_match:
                mtry = int(mtry_match.group(1))
                df = pd.read_csv(file)
                mean_crps = df['crps'].mean()

                # Speichere mtry und CRPS für jedes Modell
                mtry_values_dict[package].append(mtry)
                crps_values_dict[package].append(mean_crps)

    # Plot erstellen
    plt.figure(figsize=(10, 6))

    # Plotte die CRPS-Werte für jedes Modell
    for package in ['sklearn', 'ranger', 'quantregForest']:
        mtry_values = mtry_values_dict[package]
        crps_values = crps_values_dict[package]

        # Sortiere mtry und CRPS-Werte für das Plotten
        if mtry_values:
            sorted_indices = sorted(range(len(mtry_values)), key=lambda k: mtry_values[k])
            mtry_values = [mtry_values[idx] for idx in sorted_indices]
            crps_values = [crps_values[idx] for idx in sorted_indices]

            # Plot der CRPS-Werte
            plt.plot(mtry_values, crps_values, marker='o', color=colors[package], label=labels[package], alpha=0.6)

    # Titel und Achsenbezeichnungen
    plt.title('Mean CRPS across different mtry values (simulated dataset 0.7 correlation)')
    plt.xlabel('m_try')
    plt.ylabel('Mean CRPS')
    plt.legend(title='Package')
    plt.ylim(0, 0.7)
    plt.grid(True)
    plt.show()

# Funktion aufrufen, um den Plot zu erstellen
plot_crps_vs_mtry()
# %%
