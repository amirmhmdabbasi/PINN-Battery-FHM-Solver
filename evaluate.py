# This section calculates metrics such as RMSE, MAE, and R² for a given temperature and C-rate.
# It also generates discharge and SOC curves for performance evaluation.
def create_ordered_input_output(data_path, temperature, c_rate):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return None

    T_str = f"T{temperature}"
    C_str = f"C{c_rate}"

    time_data = data[T_str][C_str]
    X = torch.tensor(time_data[0]["X"]).reshape(-1, 1).float()
    time_points = sorted([key for key in time_data.keys() if key != "X"])

    input_list = []
    cs_list = []
    ce_list = []
    phie_list = []
    phis_list = []
    JLi_list = []

    for time_val in time_points:
        variables = time_data[time_val]
        cs_data = torch.tensor(variables["cs"]).reshape(-1, 1).float()
        ce_data = torch.tensor(variables["ce"]).reshape(-1, 1).float()
        phie_data = torch.tensor(variables["pe"]).reshape(-1, 1).float()
        phis_data = torch.tensor(variables["ps"]).reshape(-1, 1).float()
        JLi_data = torch.tensor(variables["J_Li"]).reshape(-1, 1).float()

        for x_idx, x_val in enumerate(X):
            input_list.append([x_val.item(), time_val, c_rate, temperature])
            cs_list.append(cs_data[x_idx].item())
            ce_list.append(ce_data[x_idx].item())
            phie_list.append(phie_data[x_idx].item())
            phis_list.append(phis_data[x_idx].item())
            JLi_list.append(JLi_data[x_idx].item())

    input_tensor = torch.tensor(input_list, dtype=torch.float32)
    cs_tensor = torch.tensor(cs_list, dtype=torch.float32).reshape(-1, 1)
    ce_tensor = torch.tensor(ce_list, dtype=torch.float32).reshape(-1, 1)
    phie_tensor = torch.tensor(phie_list, dtype=torch.float32).reshape(-1, 1)
    phis_tensor = torch.tensor(phis_list, dtype=torch.float32).reshape(-1, 1)
    JLi_tensor = torch.tensor(JLi_list, dtype=torch.float32).reshape(-1, 1)

    return input_tensor, cs_tensor, ce_tensor, phie_tensor, phis_tensor, JLi_tensor

data_path = "Data.pkl"
temperature = 298.0
c_rate = 1
data_tensors = create_ordered_input_output(data_path, temperature, c_rate)

if data_tensors:
    input_tensor, cs_tensor, ce_tensor, phie_tensor, phis_tensor, JLi_tensor = data_tensors

input_tensor_normalized = input_tensor.clone()
input_tensor_normalized[:, 0] = zscore_normalize(input_tensor_normalized[:, 0], x_mean_train, x_std_train)
input_tensor_normalized[:, 1] = zscore_normalize(input_tensor_normalized[:, 1], t_mean_train, t_std_train)
input_tensor_normalized[:, 2] = zscore_normalize(input_tensor_normalized[:, 2], C_mean_train, C_std_train)
input_tensor_normalized[:, 3] = zscore_normalize(input_tensor_normalized[:, 3], T_mean_train, T_std_train)

cs_tensor_normalized = zscore_normalize(cs_tensor, cs_mean_train, cs_std_train)
ce_tensor_normalized = zscore_normalize(ce_tensor, ce_mean_train, ce_std_train)
phie_tensor_normalized = zscore_normalize(phie_tensor, phie_mean_train, phie_std_train)
phis_tensor_normalized = zscore_normalize(phis_tensor, phis_mean_train, phis_std_train)
JLi_tensor_normalized = zscore_normalize(JLi_tensor, JLi_mean_train, JLi_std_train)

with torch.no_grad():
  outputs_normalized = net(input_tensor_normalized.to(device))
  cs_pred_normalized, ce_pred_normalized, phie_pred_normalized, phis_pred_normalized, JLi_pred_normalized = outputs_normalized.split(1, dim=1)

all_true_normalized = torch.cat([cs_tensor_normalized, ce_tensor_normalized, phie_tensor_normalized, phis_tensor_normalized, JLi_tensor_normalized], dim=1).cpu().numpy()
all_pred_normalized = torch.cat([cs_pred_normalized, ce_pred_normalized, phie_pred_normalized, phis_pred_normalized, JLi_pred_normalized], dim=1).cpu().numpy()

rmse = np.sqrt(mean_squared_error(all_true_normalized, all_pred_normalized))
r2 = r2_score(all_true_normalized, all_pred_normalized)
mae = mean_absolute_error(all_true_normalized, all_pred_normalized)

print(f"Metrics (Normalized Space) for T={temperature}, C={c_rate}:")
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# Discharge curve
phis_pred = (phis_pred_normalized * phis_std_train + phis_mean_train).cpu().numpy()
phis_true = phis_tensor.cpu().numpy()

x_values = input_tensor[:, 0].numpy()
time_values = input_tensor[:, 1].numpy()
time_points = sorted(list(set(time_values)))

voltage_diff_pred = []
voltage_diff_true = []
voltage_diff_mae = []

for time_val in time_points:
    mask = time_values == time_val
    x_at_time = x_values[mask]
    phis_pred_at_time = phis_pred[mask].flatten()
    phis_true_at_time = phis_true[mask].flatten()

    x_end_index = np.argmax(x_at_time)
    x_0_index = np.argmin(x_at_time)

    voltage_pred = phis_pred_at_time[x_end_index] - 0.0268 * 2.27
    voltage_true = phis_true_at_time[x_end_index] - 0.0268 * 2.27

    voltage_diff_pred.append(voltage_pred)
    voltage_diff_true.append(voltage_true)

    mae = np.mean(np.abs(voltage_pred - voltage_true))
    voltage_diff_mae.append(mae)

time_points = np.array(time_points)
voltage_diff_pred = np.array(voltage_diff_pred)
voltage_diff_mae = np.array(voltage_diff_mae)

marker_indices = np.arange(0, len(time_points), 50, dtype=int)
marker_time_points = time_points[marker_indices]
marker_voltage_diff_pred = voltage_diff_pred[marker_indices]

font_dir = '/content/drive/MyDrive/Times'

font_files = [
    os.path.join(font_dir, 'times.ttf'),
    os.path.join(font_dir, 'timesbd.ttf'),
    os.path.join(font_dir, 'timesbi.ttf'),
    os.path.join(font_dir, 'timesi.ttf')
]

for font_file in font_files:
    matplotlib.font_manager.fontManager.addfont(font_file)

plt.rcParams['font.family'] = 'Times New Roman'

plt.figure(figsize=(10, 6), facecolor='white')

plt.plot(time_points, voltage_diff_true, label='True Voltage Difference', color='#333333')  # or 'dimgray'

plt.plot(time_points, voltage_diff_pred, label='Predicted Voltage Difference', color='lightcoral')
plt.plot(marker_time_points, marker_voltage_diff_pred, '*', color='lightcoral', markersize=8)

plt.xlabel('Time (s)', fontsize=12, fontname='Times New Roman', color='steelblue')
plt.ylabel('Voltage Difference (V)', fontsize=12, fontname='Times New Roman', color='steelblue')
plt.title(f'Discharge Curve (T={int(temperature)-273}°C, C={c_rate})', fontsize=14, fontname='Times New Roman', color='steelblue')

plt.xticks(fontsize=10, fontname='Times New Roman', color='steelblue')
plt.yticks(fontsize=10, fontname='Times New Roman', color='steelblue')

plt.legend(fontsize=10, loc='best', frameon=False, prop={'family': 'Times New Roman', 'size': 10}, labelcolor='steelblue')

plt.grid(True, color='lightblue', linestyle='--', linewidth=0.5)
ax = plt.gca()
ax.spines['bottom'].set_color('lightblue')
ax.spines['top'].set_color('lightblue')
ax.spines['left'].set_color('lightblue')
ax.spines['right'].set_color('lightblue')

plt.show()

plt.figure(figsize=(10, 6), facecolor='white')

plt.plot(time_points, voltage_diff_mae, label='Mean Absolute Error', color='lightcoral', marker='o', markersize=3)

plt.xlabel('Time (s)', fontsize=12, fontname='Times New Roman', color='steelblue')
plt.ylabel('Mean Absolute Error (V)', fontsize=12, fontname='Times New Roman', color='steelblue')
plt.title(f'Mean Absolute Error of Predicted Voltage (T={int(temperature)-273}°C, C={c_rate})', fontsize=14, fontname='Times New Roman', color='steelblue')

plt.xticks(fontsize=10, fontname='Times New Roman', color='steelblue')
plt.yticks(fontsize=10, fontname='Times New Roman', color='steelblue')

plt.legend(fontsize=10, loc='best', frameon=False, prop={'family': 'Times New Roman', 'size': 10}, labelcolor='steelblue')

plt.grid(True, color='lightblue', linestyle='--', linewidth=0.5)
ax = plt.gca()
ax.spines['bottom'].set_color('lightblue')
ax.spines['top'].set_color('lightblue')
ax.spines['left'].set_color('lightblue')
ax.spines['right'].set_color('lightblue')

plt.show()

# SOC curve
cs_pred = (cs_pred_normalized * cs_std_train + cs_mean_train).cpu().numpy()
cs_true = cs_tensor.cpu().numpy()

x_values = denormalize(input_tensor_normalized[:, 0], x_mean_train, x_std_train).cpu().numpy()
time_values = input_tensor[:, 1].numpy()
time_points = sorted(list(set(time_values)))
soc_pred = []
soc_true = []
soc_mae = []

for time_val in time_points:
    mask = time_values == time_val
    x_at_time = x_values[mask]

    cs_pred_at_time = cs_pred[mask].flatten()
    cs_true_at_time = cs_true[mask].flatten()
    anode_mask = x_at_time <= L_neg
    cs_pred_anode = cs_pred_at_time[anode_mask]
    cs_true_anode = cs_true_at_time[anode_mask]

    mean_cs_pred_anode = np.mean(cs_pred_anode)
    mean_cs_true_anode = np.mean(cs_true_anode)

    theta_pred = mean_cs_pred_anode / cs_neg_max
    theta_true = mean_cs_true_anode / cs_neg_max

    soc_pred_val = (theta_pred - 0.007) * 100 / (0.792 - 0.007)
    soc_true_val = (theta_true - 0.007) * 100 / (0.792 - 0.007)

    soc_pred.append(soc_pred_val)
    soc_true.append(soc_true_val)

    mae = np.mean(np.abs(soc_pred_val - soc_true_val))
    soc_mae.append(mae)

time_points = np.array(time_points)
soc_pred = np.array(soc_pred)
soc_mae = np.array(soc_mae)

marker_indices = np.arange(0, len(time_points), 50, dtype=int)
marker_time_points = time_points[marker_indices]
marker_soc_pred = soc_pred[marker_indices]

plt.figure(figsize=(10, 6), facecolor='white')

plt.plot(time_points, soc_true, label='True SOC', color='#333333')  # or 'dimgray'

plt.plot(time_points, soc_pred, label='Predicted SOC', color='lightcoral')
plt.plot(marker_time_points, marker_soc_pred, '*', color='lightcoral', markersize=8)

plt.xlabel('Time (s)', fontsize=12, fontname='Times New Roman', color='seagreen')
plt.ylabel('SOC (%)', fontsize=12, fontname='Times New Roman', color='seagreen')
plt.title(f'State of Charge (T={int(temperature)-273}°C, C={c_rate})', fontsize=14, fontname='Times New Roman', color='seagreen')

plt.xticks(fontsize=10, fontname='Times New Roman', color='seagreen')
plt.yticks(fontsize=10, fontname='Times New Roman', color='seagreen')

plt.legend(fontsize=10, loc='best', frameon=False, prop={'family': 'Times New Roman', 'size': 10}, labelcolor='seagreen')

plt.grid(True, color='lightgreen', linestyle='--', linewidth=0.5)
ax = plt.gca()
ax.spines['bottom'].set_color('lightgreen')
ax.spines['top'].set_color('lightgreen')
ax.spines['left'].set_color('lightgreen')
ax.spines['right'].set_color('lightgreen')

plt.show()

plt.figure(figsize=(10, 6), facecolor='white')

plt.plot(time_points, soc_mae, label='Mean Absolute Error', color='lightcoral', marker='o', markersize=3)

plt.xlabel('Time (s)', fontsize=12, fontname='Times New Roman', color='seagreen')
plt.ylabel('Mean Absolute Error (%)', fontsize=12, fontname='Times New Roman', color='seagreen')
plt.title(f'Mean Absolute Error of SOC (T={int(temperature)-273}°C, C={c_rate})', fontsize=14, fontname='Times New Roman', color='seagreen')

plt.xticks(fontsize=10, fontname='Times New Roman', color='seagreen')
plt.yticks(fontsize=10, fontname='Times New Roman', color='seagreen')

plt.legend(fontsize=10, loc='best', frameon=False, prop={'family': 'Times New Roman', 'size': 10}, labelcolor='seagreen')

plt.grid(True, color='lightgreen', linestyle='--', linewidth=0.5)
ax = plt.gca()
ax.spines['bottom'].set_color('lightgreen')
ax.spines['top'].set_color('lightgreen')
ax.spines['left'].set_color('lightgreen')
ax.spines['right'].set_color('lightgreen')

plt.show()
