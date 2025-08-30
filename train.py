# Pretraining phase: fully data-driven to guide the model into a physically reasonable value space, 
# minimizing anomalies during subsequent residual-based calculations.

# Hyperparameters
config = {
    'lr': 5e-3,
    'epochs': 50,
    'batch_size': 8192,
    'grad_clip': 1.0,
}
pw1 = 1.0
pw2 = 1.0
pw3 = 1.0
pw4 = 1.0
pw5 = 1.0

net = PINN(in_features=4, hidden_features=256, hidden_layers=6, out_features=5).to(device)
optimizer = optim.Adam(net.parameters(), lr=config['lr'])

train_dataset = TensorDataset(train_inputs, train_cs, train_ce, train_phie, train_phis, train_JLi)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

val_dataset = TensorDataset(val_inputs, val_cs, val_ce, val_phie, val_phis, val_JLi)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

best_val_loss = float('inf')
history = {
    'train': [],
    'val': [],
    'lr': [],
    'metrics': []
}

for epoch in range(config['epochs']):
    net.train()
    epoch_loss = 0.0

    for batch in train_loader:
        inputs = batch[0].to(device, non_blocking=True)
        targets = [t.to(device, non_blocking=True) for t in batch[1:]]

        x = inputs[:, 0:1]
        t = inputs[:, 1:2]
        C = inputs[:, 2:3]
        T = inputs[:, 3:4]

        optimizer.zero_grad()

        outputs = net(inputs)
        cs_pred, ce_pred, phie_pred, phis_pred, JLi_pred = outputs.split(1, dim=1)
        cs_target, ce_target, phie_target, phis_target, JLi_target = targets

        loss_cs = torch.mean(torch.abs(cs_pred - cs_target))
        loss_ce = torch.mean(torch.abs(ce_pred - ce_target))
        loss_phie = torch.mean(torch.abs(phie_pred - phie_target))
        loss_phis = torch.mean(torch.abs(phis_pred - phis_target))
        loss_JLi = torch.mean(torch.abs(JLi_pred - JLi_target))

        loss = pw1*loss_cs + pw2*loss_ce + pw3*loss_phie + pw4*loss_phis + pw5*loss_JLi

        torch.nn.utils.clip_grad_norm_(net.parameters(), config['grad_clip'])

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)

    val_loss = 0.0
    epoch_metrics = {
        'loss': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
        'mse': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
        'rmse': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
        'mae': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
        'r2': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
        'total_samples': 0
    }

    all_preds = {'cs': [], 'ce': [], 'phie': [], 'phis': [], 'JLi': []}
    all_targets = {'cs': [], 'ce': [], 'phie': [], 'phis': [], 'JLi': []}

    net.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device, non_blocking=True)
            targets = [t.to(device, non_blocking=True) for t in batch[1:]]

            x = inputs[:, 0:1]
            t = inputs[:, 1:2]
            C = inputs[:, 2:3]
            T = inputs[:, 3:4]

            outputs = net(inputs)
            cs_pred, ce_pred, phie_pred, phis_pred, JLi_pred = outputs.split(1, dim=1)
            cs_target, ce_target, phie_target, phis_target, JLi_target = targets

            loss_cs = torch.mean(torch.abs(cs_pred - cs_target))
            loss_ce = torch.mean(torch.abs(ce_pred - ce_target))
            loss_phie = torch.mean(torch.abs(phie_pred - phie_target))
            loss_phis = torch.mean(torch.abs(phis_pred - phis_target))
            loss_JLi = torch.mean(torch.abs(JLi_pred - JLi_target))

            batch_loss = loss_cs + loss_ce + loss_phie + loss_phis + loss_JLi

            epoch_metrics['total_samples'] += inputs.size(0)
            val_loss += batch_loss.item() * inputs.size(0)

            all_preds['cs'].append(cs_pred.cpu())
            all_preds['ce'].append(ce_pred.cpu())
            all_preds['phie'].append(phie_pred.cpu())
            all_preds['phis'].append(phis_pred.cpu())
            all_preds['JLi'].append(JLi_pred.cpu())

            all_targets['cs'].append(cs_target.cpu())
            all_targets['ce'].append(ce_target.cpu())
            all_targets['phie'].append(phie_target.cpu())
            all_targets['phis'].append(phis_target.cpu())
            all_targets['JLi'].append(JLi_target.cpu())

    val_loss /= epoch_metrics['total_samples']

    for metric in ['loss', 'mse', 'rmse', 'mae']:
        for var in ['cs', 'ce', 'phie', 'phis', 'JLi']:
            epoch_metrics[metric][var] = torch.mean(torch.abs(torch.cat(all_preds[var]) - torch.cat(all_targets[var]))).item()

    for var in ['cs', 'ce', 'phie', 'phis', 'JLi']:
        all_var_preds = torch.cat(all_preds[var])
        all_var_targets = torch.cat(all_targets[var])

        ss_tot = torch.sum((all_var_targets - all_var_targets.mean())**2)
        ss_res = torch.sum((all_var_targets - all_var_preds)**2)
        epoch_metrics['r2'][var] = (1 - (ss_res / (ss_tot + 1e-8))).item()

    history['metrics'].append(epoch_metrics)
    history['train'].append(epoch_loss / len(train_dataset))
    history['val'].append(val_loss)
    history['lr'].append(optimizer.param_groups[0]['lr'])

    print(f"Epoch {epoch + 1:03d} | Train Loss: {history['train'][-1]:.3e} | Val Loss: {val_loss:.3e} | LR: {history['lr'][-1]:.2e}")
    print(f"R² Values - CS: {epoch_metrics['r2']['cs']:.4f}, CE: {epoch_metrics['r2']['ce']:.4f}, Phie: {epoch_metrics['r2']['phie']:.4f}, Phis: {epoch_metrics['r2']['phis']:.4f}, JLi: {epoch_metrics['r2']['JLi']:.4f}")

# Saving the pretrained model
PATH = "/content/drive/MyDrive"
os.makedirs(PATH, exist_ok=True)
torch.save(net.state_dict(), os.path.join(PATH, 'pretrained_model.pth'))

# Main training loop
# Hyperparameters
config = {
    'lr': 5e-4,
    'epochs': 100,
    'batch_size': 8192,
    'patience': 100,
    'warmup_steps': 100,
    'grad_clip': 1.0,
    'weight_decay': 1e-4
}

# These weights were selected using a grid search method. 
# Note: the grid search implementation is not included in this repository.
w1 = 8e-15
w2 = 1e-14
w3 = 1e-2
w4 = 1.0
w5 = 2.0
w6 = 1.0
w7 = 1.0
w8 = 1.0

PATH = "/content/drive/MyDrive"
os.makedirs(PATH, exist_ok=True)

net = PINN(in_features=4, hidden_features=256, hidden_layers = 6, out_features = 5).to(device)
net.load_state_dict(torch.load(os.path.join(PATH, 'pretrained_model.pth'), weights_only=True))
optimizer = optim.AdamW(net.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

train_dataset = TensorDataset(train_inputs, train_cs, train_ce, train_phie, train_phis, train_JLi)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

val_dataset = TensorDataset(val_inputs, val_cs, val_ce, val_phie, val_phis, val_JLi)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config['lr'],
    total_steps=config['epochs'] * (len(train_loader)),
    pct_start=0.1,
    div_factor = 25,
    final_div_factor = 1000
)

best_val_loss = float('inf')
history = {
    'train': [],
    'val': [],
    'lr': [],
    'metrics': []
}

for epoch in range(config['epochs']):
    net.train()
    epoch_loss = 0.0

    for batch in train_loader:
        inputs = batch[0].to(device, non_blocking=True)
        targets = [t.to(device, non_blocking=True) for t in batch[1:]]

        x = inputs[:, 0:1]
        t = inputs[:, 1:2]
        C = inputs[:, 2:3]
        T = inputs[:, 3:4]

        optimizer.zero_grad()

        loss = loss_function(net, x, t, C, T, *targets, w1, w2, w3, w4, w5, w6, w7, w8)

        torch.nn.utils.clip_grad_norm_(net.parameters(), config['grad_clip'])

        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item() * inputs.size(0)

    val_loss = 0.0
    epoch_metrics = {
        'loss': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
        'mse': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
        'rmse': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
        'mae': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
        'r2': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
        'total_samples': 0
    }

    all_preds = {'cs': [], 'ce': [], 'phie': [], 'phis': [], 'JLi': []}
    all_targets = {'cs': [], 'ce': [], 'phie': [], 'phis': [], 'JLi': []}

    net.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device, non_blocking=True)
            targets = [t.to(device, non_blocking=True) for t in batch[1:]]

            x = inputs[:, 0:1]
            t = inputs[:, 1:2]
            C = inputs[:, 2:3]
            T = inputs[:, 3:4]

            batch_loss, batch_metrics = validation_loss_function(net, x, t, C, T, *targets)

            batch_size = batch_metrics['batch_size']
            epoch_metrics['total_samples'] += batch_size
            val_loss += batch_metrics['total_loss'] * batch_size

            for metric in ['loss', 'mse', 'rmse', 'mae']:
                for var in ['cs', 'ce', 'phie', 'phis', 'JLi']:
                    epoch_metrics[metric][var] += batch_metrics[metric][var] * batch_size

            outputs = net(inputs)
            cs, ce, phie, phis, JLi = outputs.split(1, dim=1)

            all_preds['cs'].append(cs.cpu())
            all_preds['ce'].append(ce.cpu())
            all_preds['phie'].append(phie.cpu())
            all_preds['phis'].append(phis.cpu())
            all_preds['JLi'].append(JLi.cpu())

            all_targets['cs'].append(targets[0].cpu())
            all_targets['ce'].append(targets[1].cpu())
            all_targets['phie'].append(targets[2].cpu())
            all_targets['phis'].append(targets[3].cpu())
            all_targets['JLi'].append(targets[4].cpu())

    val_loss /= epoch_metrics['total_samples']

    for metric in ['loss', 'mse', 'rmse', 'mae']:
        for var in ['cs', 'ce', 'phie', 'phis', 'JLi']:
            epoch_metrics[metric][var] /= epoch_metrics['total_samples']

    for var in ['cs', 'ce', 'phie', 'phis', 'JLi']:
        all_var_preds = torch.cat(all_preds[var])
        all_var_targets = torch.cat(all_targets[var])

        ss_tot = torch.sum((all_var_targets - all_var_targets.mean())**2)
        ss_res = torch.sum((all_var_targets - all_var_preds)**2)
        epoch_metrics['r2'][var] = (1 - (ss_res / (ss_tot + 1e-8))).item()

    history['metrics'].append(epoch_metrics)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(net.state_dict(), os.path.join(PATH, 'best_model3.pth'))
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    history['train'].append(epoch_loss / len(train_dataset))
    history['val'].append(val_loss)
    history['lr'].append(optimizer.param_groups[0]['lr'])

    print(f"Epoch {epoch+1:03d} | "
          f"Train Loss: {history['train'][-1]:.3e} | "
          f"Val Loss: {val_loss:.3e} | "
          f"LR: {history['lr'][-1]:.2e}")

    print(f"R² Values - CS: {epoch_metrics['r2']['cs']:.4f}, CE: {epoch_metrics['r2']['ce']:.4f}, "
          f"Phie: {epoch_metrics['r2']['phie']:.4f}, Phis: {epoch_metrics['r2']['phis']:.4f}, "
          f"JLi: {epoch_metrics['r2']['JLi']:.4f}")

    if epochs_no_improve >= config['patience']:
        print(f"Early stopping at epoch {epoch+1}")
        break

net.load_state_dict(torch.load(os.path.join(PATH, 'best_model3.pth')))
test_dataset = TensorDataset(test_inputs, test_cs, test_ce, test_phie, test_phis, test_JLi)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

test_metrics = {
    'loss': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
    'mse': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
    'rmse': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
    'mae': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
    'r2': {'cs': 0, 'ce': 0, 'phie': 0, 'phis': 0, 'JLi': 0},
    'total_samples': 0
}

all_preds = {'cs': [], 'ce': [], 'phie': [], 'phis': [], 'JLi': []}
all_targets = {'cs': [], 'ce': [], 'phie': [], 'phis': [], 'JLi': []}

net.eval()
with torch.no_grad():
    for batch in test_loader:
        inputs = batch[0].to(device, non_blocking=True)
        targets = [t.to(device, non_blocking=True) for t in batch[1:]]

        x = inputs[:, 0:1]
        t = inputs[:, 1:2]
        C = inputs[:, 2:3]
        T = inputs[:, 3:4]

        _, batch_metrics = validation_loss_function(net, x, t, C, T, *targets)

        batch_size = batch_metrics['batch_size']
        test_metrics['total_samples'] += batch_size

        for metric in ['loss', 'mse', 'rmse', 'mae']:
            for var in ['cs', 'ce', 'phie', 'phis', 'JLi']:
                test_metrics[metric][var] += batch_metrics[metric][var] * batch_size

        outputs = net(inputs)
        cs, ce, phie, phis, JLi = outputs.split(1, dim=1)

        all_preds['cs'].append(cs.cpu())
        all_preds['ce'].append(ce.cpu())
        all_preds['phie'].append(phie.cpu())
        all_preds['phis'].append(phis.cpu())
        all_preds['JLi'].append(JLi.cpu())

        all_targets['cs'].append(targets[0].cpu())
        all_targets['ce'].append(targets[1].cpu())
        all_targets['phie'].append(targets[2].cpu())
        all_targets['phis'].append(targets[3].cpu())
        all_targets['JLi'].append(targets[4].cpu())

for metric in ['loss', 'mse', 'rmse', 'mae']:
    for var in ['cs', 'ce', 'phie', 'phis', 'JLi']:
        test_metrics[metric][var] /= test_metrics['total_samples']

for var in ['cs', 'ce', 'phie', 'phis', 'JLi']:
    all_var_preds = torch.cat(all_preds[var])
    all_var_targets = torch.cat(all_targets[var])

    ss_tot = torch.sum((all_var_targets - all_var_targets.mean())**2)
    ss_res = torch.sum((all_var_targets - all_var_preds)**2)
    test_metrics['r2'][var] = (1 - (ss_res / (ss_tot + 1e-8))).item()

test_metrics['total_loss'] = sum(test_metrics['loss'].values())
test_metrics['total_rmse'] = sum(test_metrics['rmse'].values()) / len(test_metrics['rmse'])
test_metrics['total_r2'] = sum(test_metrics['r2'].values()) / len(test_metrics['r2'])

print("\n TEST RESULTS ")
print(f"Total Loss: {test_metrics['total_loss']:.4f}")
print("\nVariable-specific metrics:")
print(f"{'Variable':<8} {'RMSE':<10} {'R²':<10} {'MAE':<10}")
print("-" * 40)
for var in ['cs', 'ce', 'phie', 'phis', 'JLi']:
    print(f"{var:<8} {test_metrics['rmse'][var]:<10.4f} {test_metrics['r2'][var]:<10.4f} {test_metrics['mae'][var]:<10.4f}")

def plot_training_curves(history):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.semilogy(history['train'], label='Train')
    plt.semilogy(history['val'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['lr'], '.-', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    epochs = list(range(1, len(history['metrics'])+1))

    for var in ['cs', 'ce', 'phie', 'phis', 'JLi']:
        r2_values = [metrics['r2'][var] for metrics in history['metrics']]
        plt.plot(epochs, r2_values, marker='.', label=f'{var}')

    plt.axhline(y=0.98, color='r', linestyle='--', label='Target R²=0.98')
    plt.xlabel('Epoch')
    plt.ylabel('R² Value')
    plt.title('R² Evolution')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)

    for var in ['cs', 'ce']:
        rmse_values = [metrics['rmse'][var] for metrics in history['metrics']]
        plt.semilogy(epochs, rmse_values, marker='.', label=f'{var}')

    plt.xlabel('Epoch')
    plt.ylabel('RMSE (log scale)')
    plt.title('Concentration RMSE Evolution')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_report.svg', dpi=300)
    plt.show()

plot_training_curves(history)
