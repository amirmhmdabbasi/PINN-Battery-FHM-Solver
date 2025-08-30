import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "/content/drive/MyDrive/Data.pkl" #Data was uploaded to google drive, as "Data.pkl"
with open(file_path, "rb") as file:
        data = pickle.load(file)

# Battery parameters

A_cell = 0.1037  # [m^2] Cell cross-sectional area
cs_neg_max = 27088  # [mol/m^3] Maximum concentration in negative electrode
cs_pos_max = 48700  # [mol/m^3] Maximum concentration in positive electrode
ce_init = 1200  # [mol/m^3] Initial electrolyte concentration
Ds_neg_eff = 1.4e-11  # [m^2/s] Effective diffusion coefficient in negative electrode
Ds_pos_eff = 1.4e-11  # [m^2/s] Effective diffusion coefficient in positive electrode
eps_n = 0.6260  # Volume fraction of active material in negative electrode
eps_p = 0.5741  # Volume fraction of active material in positive electrode
eps_en = 0.2979  # Volume fraction of electrolyte in negative electrode
eps_ep = 0.3591  # Volume fraction of electrolyte in positive electrode
eps_sep = 0.35  # Volume fraction of electrolyte in separator
kn_star = 180  # [A/mol] Reaction rate constant for negative electrode
kp_star = 90  # [A/mol] Reaction rate constant for positive electrode
L_neg = 53.2e-6  # [m] Thickness of negative electrode
L_sep = 24.7e-6  # [m] Thickness of separator
L_pos = 39.9e-6  # [m] Thickness of positive electrode
R_c = 0.0268  # [ohm] Contact resistance
xn_init = 0.7916  # Initial stoichiometry in negative electrode
xp_init = 0.3494  # Initial stoichiometry in positive electrode
sigma_neg = 100  # [S/m] Conductivity of negative electrode
sigma_pos = 3.8  # [S/m] Conductivity of positive electrode
Ks_eff_pos = sigma_pos * 0.513  # Effective ionic conductivity in positive electrode
Ks_eff_neg = sigma_neg * 0.584  # Effective ionic conductivity in negative electrode
t_plus = 0.363  # Transference number of Li
R = 8.314  # [J/(mol*K)] Ideal gas constant
F = 96485  # [A*s/mol] Faraday constant
I = 2.27  # [A] Applied current

# This section organizes the data for model training. 
# Warning messages are included as a failsafe to flag any missing data during preprocessing, ensuring data integrity.

temperatures = [288.0, 298.0, 308.0, 318.0]
C_rates = [0.5, 1, 2, 4, 6, 8, 10]

def process_data(data, temperatures, C_rates):
    input_list = []
    cs_list = []
    ce_list = []
    phie_list = []
    phis_list = []
    JLi_list = []

    for T in temperatures:
        for C in C_rates:
            T_str = f"T{T}"
            C_str = f"C{C}"
            if T_str in data and C_str in data[T_str]:
                time_data = data[T_str][C_str]
                X = torch.tensor(time_data[0]["X"]).reshape(-1, 1).float()
                for time_val_str, variables in time_data.items():
                    if time_val_str != "X":
                        try:
                            time_val = int(time_val_str)
                            ce_data = torch.tensor(variables["ce"]).reshape(-1, 1).float()
                            cs_data = torch.tensor(variables["cs"]).reshape(-1, 1).float()
                            phie_data = torch.tensor(variables["pe"]).reshape(-1, 1).float()
                            phis_data = torch.tensor(variables["ps"]).reshape(-1, 1).float()
                            JLi_data = torch.tensor(variables["J_Li"]).reshape(-1,1).float()
                            for x_idx, x_val in enumerate(X):
                                input_list.append([x_val.item(), time_val, C, T])
                                cs_list.append(cs_data[x_idx].item())
                                ce_list.append(ce_data[x_idx].item())
                                phie_list.append(phie_data[x_idx].item())
                                phis_list.append(phis_data[x_idx].item())
                                JLi_list.append(JLi_data[x_idx].item())
                        except ValueError:
                            print(f"Warning: Could not convert key {time_val_str} to integer for T={T}, C={C}. Skipping.")
            else:
                print(f"Warning: Data not found for T={T}, C={C}")
    inputs = torch.tensor(input_list, dtype=torch.float32)
    cs_targets = torch.tensor(cs_list, dtype=torch.float32).reshape(-1, 1)
    ce_targets = torch.tensor(ce_list, dtype=torch.float32).reshape(-1, 1)
    phie_targets = torch.tensor(phie_list, dtype=torch.float32).reshape(-1, 1)
    phis_targets = torch.tensor(phis_list, dtype=torch.float32).reshape(-1, 1)
    JLi_targets = torch.tensor(JLi_list, dtype=torch.float32).reshape(-1,1)
    return inputs, cs_targets, ce_targets, phie_targets, phis_targets, JLi_targets

inputs_all, cs_all, ce_all, phie_all, phis_all, JLi_all = process_data(data, temperatures, C_rates)

all_data = torch.cat([inputs_all, cs_all, ce_all, phie_all, phis_all, JLi_all], dim=1)

# Splitting the dataset into 70% training, 15% validation, and 15% test sets using scikit-learn's train_test_split for balanced model evaluation.
train_data, temp_data = train_test_split(all_data, test_size=0.3, random_state=42)  # Added random state
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

def separate_data(data):
    inputs = data[:, :4]
    cs = data[:, 4].reshape(-1, 1)
    ce = data[:, 5].reshape(-1, 1)
    phie = data[:, 6].reshape(-1, 1)
    phis = data[:, 7].reshape(-1, 1)
    JLi = data[:, 8].reshape(-1, 1)
    return inputs, cs, ce, phie, phis, JLi

train_inputs, train_cs, train_ce, train_phie, train_phis, train_JLi = separate_data(train_data)
val_inputs, val_cs, val_ce, val_phie, val_phis, val_JLi = separate_data(val_data)
test_inputs, test_cs, test_ce, test_phie, test_phis, test_JLi = separate_data(test_data)

# Normalization is performed using the z-score (standardization) method. 
# Although scikit-learn's StandardScaler could have been used, we opted for a manual implementation via a custom function for greater control and transparency.
T_min = train_inputs[:, 3].min().to(device)
T_max = train_inputs[:, 3].max().to(device)
ce_min = train_ce.min().to(device)
ce_max = train_ce.max().to(device)
JLi_min = train_JLi.min().to(device)
JLi_max = train_JLi.max().to(device)

x_mean_train = train_inputs[:, 0].mean()
x_std_train = train_inputs[:, 0].std()
t_mean_train = train_inputs[:, 1].mean()
t_std_train = train_inputs[:, 1].std()
C_mean_train = train_inputs[:, 2].mean()
C_std_train = train_inputs[:, 2].std()
T_mean_train = train_inputs[:, 3].mean()
T_std_train = train_inputs[:, 3].std()

cs_mean_train = train_cs.mean()
cs_std_train = train_cs.std()
ce_mean_train = train_ce.mean()
ce_std_train = train_ce.std()
phie_mean_train = train_phie.mean()
phie_std_train = train_phie.std()
phis_mean_train = train_phis.mean()
phis_std_train = train_phis.std()
JLi_mean_train = train_JLi.mean()
JLi_std_train = train_JLi.std()

def zscore_normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)

train_inputs[:, 0] = zscore_normalize(train_inputs[:, 0], x_mean_train, x_std_train)
train_inputs[:, 1] = zscore_normalize(train_inputs[:, 1], t_mean_train, t_std_train)
train_inputs[:, 2] = zscore_normalize(train_inputs[:, 2], C_mean_train, C_std_train)
train_inputs[:, 3] = zscore_normalize(train_inputs[:, 3], T_mean_train, T_std_train)

val_inputs[:, 0] = zscore_normalize(val_inputs[:, 0], x_mean_train, x_std_train)
val_inputs[:, 1] = zscore_normalize(val_inputs[:, 1], t_mean_train, t_std_train)
val_inputs[:, 2] = zscore_normalize(val_inputs[:, 2], C_mean_train, C_std_train)
val_inputs[:, 3] = zscore_normalize(val_inputs[:, 3], T_mean_train, T_std_train)

test_inputs[:, 0] = zscore_normalize(test_inputs[:, 0], x_mean_train, x_std_train)
test_inputs[:, 1] = zscore_normalize(test_inputs[:, 1], t_mean_train, t_std_train)
test_inputs[:, 2] = zscore_normalize(test_inputs[:, 2], C_mean_train, C_std_train)
test_inputs[:, 3] = zscore_normalize(test_inputs[:, 3], T_mean_train, T_std_train)

train_cs = zscore_normalize(train_cs, cs_mean_train, cs_std_train)
val_cs = zscore_normalize(val_cs, cs_mean_train, cs_std_train)
test_cs = zscore_normalize(test_cs, cs_mean_train, cs_std_train)

train_ce = zscore_normalize(train_ce, ce_mean_train, ce_std_train)
val_ce = zscore_normalize(val_ce, ce_mean_train, ce_std_train)
test_ce = zscore_normalize(test_ce, ce_mean_train, ce_std_train)

train_phie = zscore_normalize(train_phie, phie_mean_train, phie_std_train)
val_phie = zscore_normalize(val_phie, phie_mean_train, phie_std_train)
test_phie = zscore_normalize(test_phie, phie_mean_train, phie_std_train)

train_phis = zscore_normalize(train_phis, phis_mean_train, phis_std_train)
val_phis = zscore_normalize(val_phis, phis_mean_train, phis_std_train)
test_phis = zscore_normalize(test_phis, phis_mean_train, phis_std_train)

train_JLi = zscore_normalize(train_JLi, JLi_mean_train, JLi_std_train)
val_JLi = zscore_normalize(val_JLi, JLi_mean_train, JLi_std_train)
test_JLi = zscore_normalize(test_JLi, JLi_mean_train, JLi_std_train)

# This section prints the range of each parameter post-normalization to identify potential outliers or anomalies 
# that could negatively impact model training.
print("Training data after normalization:")
print("x_train:", train_inputs[:, 0].min(), train_inputs[:, 0].max())
print("t_train:", train_inputs[:, 1].min(), train_inputs[:, 1].max())
print("C_train:", train_inputs[:, 2].min(), train_inputs[:, 2].max())
print("T_train:", train_inputs[:, 3].min(), train_inputs[:, 3].max())
print("cs_train:", train_cs.min(), train_cs.max())
print("ce_train:", train_ce.min(), train_ce.max())
print("phie_train:", train_phie.min(), train_phie.max())
print("phis_train:", train_phis.min(), train_phis.max())
print("JLi_train:", train_JLi.min(), train_JLi.max())

print("\nValidation data after normalization:")
print("x_val:", val_inputs[:, 0].min(), val_inputs[:, 0].max())
print("t_val:", val_inputs[:, 1].min(), val_inputs[:, 1].max())
print("C_val:", val_inputs[:, 2].min(), val_inputs[:, 2].max())
print("T_val:", val_inputs[:, 3].min(), val_inputs[:, 3].max())
print("cs_val:", val_cs.min(), val_cs.max())
print("ce_val:", val_ce.min(), val_ce.max())
print("phie_val:", val_phie.min(), val_phie.max())
print("phis_val:", val_phis.min(), val_phis.max())
print("JLi_val:", val_JLi.min(), val_JLi.max())

print("\nTest data after normalization:")
print("x_test:", test_inputs[:, 0].min(), test_inputs[:, 0].max())
print("t_test:", test_inputs[:, 1].min(), test_inputs[:, 1].max())
print("C_test:", test_inputs[:, 2].min(), test_inputs[:, 2].max())
print("T_test:", test_inputs[:, 3].min(), test_inputs[:, 3].max())
print("cs_test:", test_cs.min(), test_cs.max())
print("ce_test:", test_ce.min(), test_ce.max())
print("phie_test:", test_phie.min(), test_phie.max())
print("phis_test:", test_phis.min(), test_phis.max())
print("JLi_test:", test_JLi.min(), test_JLi.max())

# Denormalization function defined for later use when converting normalized values back to their original scale.
def denormalize(normalized_data, original_mean, original_std):
    return (normalized_data*original_std) + original_mean
