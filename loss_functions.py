def loss_function(net, x, t, C, T, cs_data, ce_data, phie_data, phis_data, JLi_data, w1, w2, w3, w4, w5, w6, w7, w8):

    x.requires_grad_(True)
    t.requires_grad_(True)

    inputs = torch.cat([x, t, C, T], dim=1)
    outputs = net(inputs)
    cs, ce, phie, phis, JLi = outputs.split(1, dim=1)

    data_losses = {
        'cs': torch.mean(torch.abs(cs - cs_data)),
        'ce': torch.mean(torch.abs(ce - ce_data)),
        'phie': torch.mean(torch.abs(phie - phie_data)),
        'phis': torch.mean(torch.abs(phis - phis_data)),
        'JLi': torch.mean(torch.abs(JLi - JLi_data))
    }

    def compute_gradients(y, x):
        dy = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]
        return dy

    cs_t = compute_gradients(cs, t)
    cs_x = compute_gradients(cs, x)
    ce_t = compute_gradients(ce, t)
    ce_x = compute_gradients(ce, x)
    phie_x = compute_gradients(phie, x)
    phis_x = compute_gradients(phis, x)

    cs_xx = compute_gradients(cs_x, x)
    ce_xx = compute_gradients(ce_x, x)
    phie_xx = compute_gradients(phie_x, x)
    phis_xx = compute_gradients(phis_x, x)

    x_eval = denormalize(x, x_mean_train, x_std_train)

    def region_mask(x, lower, upper, sharpness=1e+7):
      sig1 = torch.sigmoid(sharpness * (x - lower))
      sig2 = torch.sigmoid(sharpness * (upper - x))
      return sig1 * sig2

    masks = {
        'anode': region_mask(x_eval, 0, L_neg),
        'separator': region_mask(x_eval, L_neg, L_neg + L_sep),
        'cathode': region_mask(x_eval, L_neg + L_sep, L_neg + L_sep + L_pos)
    }

    residuals = {
        'anode': residual_anode(cs, ce, phis, phie, JLi, cs_t, cs_xx, ce_t, ce_x, ce_xx, phie_xx, phis_xx, T, Ds_neg_eff, Ks_eff_neg, eps_en),
        'separator': residual_separator(ce, phie, JLi, ce_t, ce_x, ce_xx, phie_xx, T, eps_sep),
        'cathode': residual_cathode(cs, ce, phis, phie, JLi, cs_t, cs_xx, ce_t, ce_x, ce_xx, phie_xx, phis_xx, T, Ds_pos_eff, Ks_eff_pos, eps_ep)
    }

    residual_loss = 0.0
    for region in ['anode', 'separator', 'cathode']:
        mask = masks[region]
        region_residuals = residuals[region] if isinstance(residuals[region], tuple) \
                         else (residuals[region],)

        for res in region_residuals:
            residual_loss += torch.mean(mask * nn.L1Loss()(res, torch.zeros_like(res)))

    bc_loss = boundary_loss(net, x, t, C, T)
    ic_loss = initial_loss(net, x, t, C, T)

    with torch.no_grad():
        loss_components = torch.stack([
            residual_loss,
            bc_loss,
            ic_loss,
            data_losses['cs'],
            data_losses['ce'],
            data_losses['phie'],
            data_losses['phis'],
            data_losses['JLi']
        ])

    loss = (w1 * residual_loss +
            w2 * bc_loss +
            w3 * ic_loss +
            w4 * data_losses['cs'] +
            w5 * data_losses['ce'] +
            w6 * data_losses['phie'] +
            w7 * data_losses['phis'] +
            w8 * data_losses['JLi']
            )

    return loss

def validation_loss_function(net, x, t, C, T, cs_data, ce_data, phie_data, phis_data, JLi_data):
    with torch.no_grad():
        x = x.to(device)
        t = t.to(device)
        C = C.to(device)
        T = T.to(device)

        cs_data = cs_data.to(device)
        ce_data = ce_data.to(device)
        phie_data = phie_data.to(device)
        phis_data = phis_data.to(device)
        JLi_data = JLi_data.to(device)

        inputs = torch.cat([x, t, C, T], dim=1)
        outputs = net(inputs)

        cs, ce, phie, phis, JLi = outputs.split(1, dim=1)

        loss_fn = {
          'cs': torch.mean(torch.abs(cs - cs_data)),
          'ce': torch.mean(torch.abs(ce - ce_data)),
          'phie': torch.mean(torch.abs(phie - phie_data)),
          'phis': torch.mean(torch.abs(phis - phis_data)),
          'JLi': torch.mean(torch.abs(JLi - JLi_data))
        }

        variable_pairs = [
            ('cs', cs, cs_data),
            ('ce', ce, ce_data),
            ('phie', phie, phie_data),
            ('phis', phis, phis_data),
            ('JLi', JLi, JLi_data)
        ]

        losses = {}
        mse = {}
        mae = {}
        r2 = {}

        for name, pred, target in variable_pairs:
            losses[name] = loss_fn[name]

            mse[name] = torch.mean((pred - target)**2)

            mae[name] = torch.mean(torch.abs(pred - target))

            ss_tot = torch.sum((target - torch.mean(target))**2)
            ss_res = torch.sum((target - pred)**2)
            r2[name] = 1 - (ss_res / (ss_tot + 1e-8))

        total_loss = sum(torch.nan_to_num(loss, nan=1e6) for loss in losses.values())
        total_mse = sum(mse.values()) / len(mse)
        total_mae = sum(mae.values()) / len(mae)
        total_r2 = sum(r2.values()) / len(r2)

        rmse = {name: torch.sqrt(val) for name, val in mse.items()}
        total_rmse = torch.sqrt(total_mse)

        metrics = {
            'loss': {name: val.item() for name, val in losses.items()},
            'mse': {name: val.item() for name, val in mse.items()},
            'rmse': {name: val.item() for name, val in rmse.items()},
            'mae': {name: val.item() for name, val in mae.items()},
            'r2': {name: val.item() for name, val in r2.items()},
            'total_loss': total_loss.item(),
            'total_mse': total_mse.item(),
            'total_rmse': total_rmse.item(),
            'total_mae': total_mae.item(),
            'total_r2': total_r2.item(),
            'batch_size': x.size(0)
        }

    return total_loss, metrics

