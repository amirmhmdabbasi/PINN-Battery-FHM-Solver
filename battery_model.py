# This section defines helper functions used to compute various battery parameters, 
# including current, electrolyte ionic conductivity, and diffusion coefficient.
def I_app(C):
    C = denormalize(C, C_mean_train, C_std_train)
    I_app = I*C
    return I_app

def Ke(ce, T):
    ce = ce / 1000.0
    k = torch.tensor([[-10.5, 0.0740, -6.96e-5],
                      [0.668, -0.0178, 2.80e-5],
                      [0.494, -8.86e-4, 0.0]], device=ce.device)
    x = torch.zeros_like(ce)
    for i in range(3):
        for j in range(3):
            x += k[i, j] * ce**i * T**j
    K = (x**2) * ce
    Ke = K / 10.0
    return Ke

def De(ce, T):
    ce = ce / 1000.0
    d = torch.tensor([[-4.43, -54.0],
                      [-0.22, 0.0]], device=ce.device)
    T_g0 = 229.0
    T_g1 = 5.0
    T_g = T_g0 + ce * T_g1
    d1 = d[0, 0] + d[0, 1] / (T - T_g)
    d2 = d[1, 0] + d[1, 1] / (T - T_g)
    x = d1 + d2 * ce
    DD = 10**x
    De = DD * 1e-4
    return De

# Nondimensionalization scales
x_s = L_neg + L_sep + L_pos
t_s = 3600.0 #[s]
T_s = 298.0 #[K]
cs_s = cs_pos_max
ce_s = ce_init
phis_s = 2.5 #[V]
phie_s = 1.0 #[V]
JLi_s = 0.0001 #[A/m^3]

# This section contains the core functions that define the FHM (Full Homogenized Model) equations 
# and compute the corresponding residuals.
def residual_anode(cs, ce, phis, phie, J_Li, cs_t, cs_xx, ce_t, ce_x, ce_xx, phie_xx, phis_xx, T, Ds_eff, Ks_eff, eps_e):
    T = denormalize(T, T_mean_train, T_std_train)
    JLi = denormalize(J_Li, JLi_mean_train, JLi_std_train)
    ce = denormalize(ce, ce_mean_train, ce_std_train)
    Ke_eff = Ke(ce, T)*0.192
    De_eff = De(ce, T)*0.192
    a1 = (Ds_eff*t_s)/(x_s**2)
    a2 = -(JLi_s*t_s)/(F*cs_s)
    a3 = (De_eff*t_s)/(eps_e*(x_s**2))
    a4 = (JLi_s*t_s)/(F*eps_e*ce_s)
    a5 = (R*T_s*T*(t_plus**2)*t_s*Ke_eff)/((x_s**2)*(F**2)*eps_e*ce_s)
    a6 = (t_plus*t_s*Ke_eff*phie_s)/(F*(x_s**2)*eps_e*ce_s)
    a7 = (JLi_s*(x_s**2))/(Ks_eff*phis_s)
    a8 = (R*T_s*T*t_plus)/(F*phie_s)
    a9 = -(JLi_s*(x_s**2))/(Ke_eff*phie_s)

    residual_cs = (cs_std_train/t_std_train)*cs_t - a1*(cs_std_train/(x_std_train**2))*cs_xx - a2*JLi
    residual_ce = (ce_std_train/t_std_train)*ce_t - a3*(ce_std_train/(x_std_train**2))*ce_xx - a4*JLi - a5*(((ce_std_train/(x_std_train**2))*(ce_xx/ce))-((ce_std_train/x_std_train)*(ce_x/ce))**2) - a6*(phie_std_train/(x_std_train)**2)*phie_xx
    residual_phis = (phis_std_train/(x_std_train**2))*phis_xx - a7*JLi
    residual_phie = (phie_std_train/(x_std_train)**2)*phie_xx - a8*(((ce_std_train/x_std_train)*(ce_x/ce))**2 - ((ce_std_train/(x_std_train**2))*(ce_xx/ce))) - a9*JLi
    return residual_cs, residual_ce, residual_phis, residual_phie

def residual_separator(ce, phie, J_Li, ce_t, ce_x, ce_xx, phie_xx, T, eps_e):
    T = denormalize(T, T_mean_train, T_std_train)
    JLi = denormalize(J_Li, JLi_mean_train, JLi_std_train)
    ce = denormalize(ce, ce_mean_train, ce_std_train)
    Ke_eff = Ke(ce, T)*0.244
    De_eff = De(ce, T)*0.244
    a3 = (De_eff*t_s)/(eps_e*(x_s**2))
    a4 = (JLi_s*t_s)/(F*eps_e*ce_s)
    a5 = (R*T_s*T*(t_plus**2)*t_s*Ke_eff)/((x_s**2)*(F**2)*eps_e*ce_s)
    a6 = (t_plus*t_s*Ke_eff*phie_s)/(F*(x_s**2)*eps_e*ce_s)
    a8 = (R*T_s*T*t_plus)/(F*phie_s)
    a9 = -(JLi_s*(x_s**2))/(Ke_eff*phie_s)

    residual_ce = (ce_std_train/t_std_train)*ce_t - a3*(ce_std_train/(x_std_train**2))*ce_xx - a4*JLi - a5*(((ce_std_train/(x_std_train**2))*(ce_xx/ce))-((ce_std_train/x_std_train)*(ce_x/ce))**2) - a6*(phie_std_train/(x_std_train)**2)*phie_xx
    residual_phie = (phie_std_train/(x_std_train)**2)*phie_xx - a8*(((ce_std_train/x_std_train)*(ce_x/ce))**2 - ((ce_std_train/(x_std_train**2))*(ce_xx/ce))) - a9*JLi
    return residual_ce, residual_phie

def residual_cathode(cs, ce, phis, phie, J_Li, cs_t, cs_xx, ce_t, ce_x, ce_xx, phie_xx, phis_xx, T, Ds_eff, Ks_eff, eps_e):
    T = denormalize(T, T_mean_train, T_std_train)
    JLi = denormalize(J_Li, JLi_mean_train, JLi_std_train)
    ce = denormalize(ce, ce_mean_train, ce_std_train)
    Ke_eff = Ke(ce, T)*0.254
    De_eff = De(ce, T)*0.254
    a1 = (Ds_eff*t_s)/(x_s**2)
    a2 = -(JLi_s*t_s)/(F*cs_s)
    a3 = (De_eff*t_s)/(eps_e*(x_s**2))
    a4 = (JLi_s*t_s)/(F*eps_e*ce_s)
    a5 = (R*T_s*T*(t_plus**2)*t_s*Ke_eff)/((x_s**2)*(F**2)*eps_e*ce_s)
    a6 = (t_plus*t_s*Ke_eff*phie_s)/(F*(x_s**2)*eps_e*ce_s)
    a7 = (JLi_s*(x_s**2))/(Ks_eff*phis_s)
    a8 = (R*T_s*T*t_plus)/(F*phie_s)
    a9 = -(JLi_s*(x_s**2))/(Ke_eff*phie_s)

    residual_cs = (cs_std_train/t_std_train)*cs_t - a1*(cs_std_train/(x_std_train**2))*cs_xx - a2*JLi
    residual_ce = (ce_std_train/t_std_train)*ce_t - a3*(ce_std_train/(x_std_train**2))*ce_xx - a4*JLi - a5*(((ce_std_train/(x_std_train**2))*(ce_xx/ce))-((ce_std_train/x_std_train)*(ce_x/ce))**2) - a6*(phie_std_train/(x_std_train)**2)*phie_xx
    residual_phis = (phis_std_train/(x_std_train**2))*phis_xx - a7*JLi
    residual_phie = (phie_std_train/(x_std_train)**2)*phie_xx - a8*(((ce_std_train/x_std_train)*(ce_x/ce))**2 - ((ce_std_train/(x_std_train**2))*(ce_xx/ce))) - a9*JLi
    return residual_cs, residual_ce, residual_phis, residual_phie

# This section computes the residuals for the boundary condition equations. 
# Smooth masking is applied to ensure a seamless transition between battery regions, 
# preventing discontinuities in derivative calculations.
def boundary_loss(net, x, t, C, T):
    bc_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    mae_loss = nn.L1Loss()

    x_eval = denormalize(x[:, 0:1], x_mean_train, x_std_train).squeeze()

    def smooth_mask(x, threshold, sharpness=1e+7, tolerance=1e-4):
        lower = torch.sigmoid(sharpness * (x - (threshold - tolerance)))
        upper = torch.sigmoid(sharpness * ((threshold + tolerance) - x))
        return lower * upper

    boundaries = {
        0.0: {"phis": lambda out: mae_loss(out[:, 3], torch.zeros_like(out[:, 3])),
              "dcs_dx": lambda grad_cs_x: mae_loss(grad_cs_x, torch.zeros_like(grad_cs_x)),
              "dce_dx": lambda grad_ce_x: mae_loss(grad_ce_x, torch.zeros_like(grad_ce_x)),
              "dphie_dx": lambda grad_phie_x: mae_loss(grad_phie_x, torch.zeros_like(grad_phie_x))},
        L_neg: {"dcs_dx": lambda grad_cs_x, JLi: mae_loss(grad_cs_x, (-JLi_s * JLi) / (F * L_neg)),
                "dphis_dx": lambda grad_phis_x: mae_loss(grad_phis_x, torch.zeros_like(grad_phis_x))},
        L_neg + L_sep: {"dcs_dx": lambda grad_cs_x, JLi: mae_loss(grad_cs_x, (-JLi_s * JLi) / (F * L_pos)),
                        "dphis_dx": lambda grad_phis_x: mae_loss(grad_phis_x, torch.zeros_like(grad_phis_x))},
        L_neg + L_sep + L_pos: {"dcs_dx": lambda grad_cs_x: mae_loss(grad_cs_x, torch.zeros_like(grad_cs_x)),
                                "dce_dx": lambda grad_ce_x: mae_loss(grad_ce_x, torch.zeros_like(grad_ce_x)),
                                "dphie_dx": lambda grad_phie_x: mae_loss(grad_phie_x, torch.zeros_like(grad_phie_x)),
                                "dphis_dx": lambda grad_phis_x, C: mae_loss(grad_phis_x, -I_app(C) / A_cell)},
    }

    for x_boundary, conditions in boundaries.items():
        mask = smooth_mask(x_eval, x_boundary)
        mask = mask > 0.01

        if mask.sum() == 0:
            continue

        x_bc = x[mask].requires_grad_(True)
        t_bc = t[mask].requires_grad_(True)
        C_bc = C[mask]
        T_bc = T[mask]
        inputs_bc = torch.cat([x_bc, t_bc, C_bc, T_bc], dim=1)
        outputs_bc = net(inputs_bc)
        cs_bc, ce_bc, phie_bc, phis_bc, JLi_bc = outputs_bc.split(1, dim=1)

        JLi = denormalize(JLi_bc, JLi_mean_train, JLi_std_train)
        cs_x_bc = torch.autograd.grad(cs_bc.sum(), x_bc, create_graph=True, retain_graph=True)[0]
        ce_x_bc = torch.autograd.grad(ce_bc.sum(), x_bc, create_graph=True, retain_graph=True)[0]
        phie_x_bc = torch.autograd.grad(phie_bc.sum(), x_bc, create_graph=True, retain_graph=True)[0]
        phis_x_bc = torch.autograd.grad(phis_bc.sum(), x_bc, create_graph=True, retain_graph=True)[0]

        for condition, target_func in conditions.items():
            if condition == "phis":
                bc_loss += torch.mean(mask.float().unsqueeze(1) * target_func(outputs_bc))
            elif condition == "dcs_dx":
                flux_scale_neg = Ds_neg_eff * (cs_s / x_s) * (cs_std_train / x_std_train)
                flux_neg = flux_scale_neg * cs_x_bc
                if x_boundary == L_neg or x_boundary == L_neg + L_sep:
                    bc_loss += torch.mean(mask * target_func(flux_neg, JLi_bc))
                else:
                    bc_loss += torch.mean(mask * target_func(flux_neg))
            elif condition == "dce_dx":
                electrolyte_flux_scale = ((ce_s / x_s) * (ce_std_train / x_std_train))
                electrolyte_flux = electrolyte_flux_scale * ce_x_bc
                bc_loss += torch.mean(mask * target_func(electrolyte_flux))
            elif condition == "dphie_dx":
                electrolyte_conduct_scale = ((phie_s / x_s) * (phie_std_train / x_std_train))
                electrolyte_current = -electrolyte_conduct_scale * phie_x_bc
                bc_loss += torch.mean(mask * target_func(electrolyte_current))
            elif condition == "dphis_dx":
                solid_conduct_scale = (Ks_eff_pos * (phis_s / x_s) * (phis_std_train / x_std_train) if x_boundary == L_neg + L_sep + L_pos else 1.0)
                solid_current = -solid_conduct_scale * phis_x_bc
                if x_boundary == L_neg + L_sep + L_pos:
                    bc_loss += torch.mean(mask * target_func(solid_current, C_bc))
                else:
                    bc_loss += torch.mean(mask * target_func(solid_current))

    return bc_loss

# The same smooth masking technique is applied here to compute residuals for the initial conditions, 
# ensuring continuity and avoiding derivative discontinuities at the start of the simulation.
def initial_loss(net, x, t, C, T):
    ic_loss = torch.tensor(0.0, dtype=torch.float32, device=x.device)
    mae_loss = nn.L1Loss()
    t_tolerance=1e-3
    t_eval = denormalize(t, t_mean_train, t_std_train)
    mask_initial = torch.abs(t_eval - 0.0) < t_tolerance

    if not mask_initial.any():
        return ic_loss

    def U0_n_ic(x):
        return 0.1493 + 0.8493 * np.exp(-61.79 * x) + 0.3824 * np.exp(-665.8 * x) - \
               np.exp(39.42 * x - 41.92) - 0.03131 * np.arctan(25.59 * x - 4.099) - \
               0.009434 * np.arctan(32.49 * x - 15.74)

    def U0_p_ic(x):
        return -10.72 * x**4 + 23.88 * x**3 - 16.77 * x**2 + 2.595 * x + 4.563

    indices = torch.where(mask_initial)[0]
    batch_size = indices.size(0)

    x_ic = x[indices].view(batch_size, 1)
    t_ic = torch.zeros_like(x_ic)
    C_ic = C[indices].view(batch_size, 1)
    T_ic = T[indices].view(batch_size, 1)

    inputs_ic = torch.cat([x_ic, t_ic, C_ic, T_ic], dim=1)
    outputs_ic = net(inputs_ic)
    cs_ic, ce_ic, phie_ic, phis_ic, JLi_ic = outputs_ic.split(1, dim=1)

    x_physical = denormalize(x_ic, x_mean_train, x_std_train)

    cs_neg_init = xn_init * cs_neg_max
    cs_pos_init = xp_init * cs_pos_max
    ocv_init = U0_p_ic(cs_pos_init / cs_pos_max) - U0_n_ic(cs_neg_init / cs_neg_max)

    anode_mask = ((0 <= x_physical) & (x_physical <= L_neg)).float().view(batch_size, 1)
    separator_mask = ((L_neg < x_physical) & (x_physical < L_neg + L_sep)).float().view(batch_size, 1)
    cathode_mask = ((L_neg + L_sep <= x_physical) & (x_physical <= L_neg + L_sep + L_pos)).float().view(batch_size, 1)

    ic_loss += torch.mean(anode_mask * mae_loss(phis_ic, torch.zeros_like(phis_ic)))
    ic_loss += torch.mean(anode_mask * mae_loss(phie_ic, torch.zeros_like(phie_ic)))
    ic_loss += torch.mean(anode_mask * mae_loss(cs_ic, torch.full_like(cs_ic, ((xn_init * cs_neg_max/cs_s) - cs_mean_train) / cs_std_train)))
    ic_loss += torch.mean(anode_mask * mae_loss(ce_ic, torch.full_like(ce_ic, ((ce_init/ce_s) - ce_mean_train) / ce_std_train)))

    ic_loss += torch.mean(separator_mask * mae_loss(phie_ic, torch.zeros_like(phie_ic)))
    ic_loss += torch.mean(separator_mask * mae_loss(ce_ic, torch.full_like(ce_ic, ((ce_init/ce_s) - ce_mean_train) / ce_std_train)))

    ic_loss += torch.mean(cathode_mask * mae_loss(phis_ic, torch.full_like(phis_ic, ((ocv_init/phis_s) - phis_mean_train) / phis_std_train)))
    ic_loss += torch.mean(cathode_mask * mae_loss(phie_ic, torch.zeros_like(phie_ic)))
    ic_loss += torch.mean(cathode_mask * mae_loss(cs_ic, torch.full_like(cs_ic, ((xp_init * cs_pos_max/cs_s) - cs_mean_train) / cs_std_train)))
    ic_loss += torch.mean(cathode_mask * mae_loss(ce_ic, torch.full_like(ce_ic, ((ce_init/ce_s) - ce_mean_train) / ce_std_train)))

    return ic_loss
