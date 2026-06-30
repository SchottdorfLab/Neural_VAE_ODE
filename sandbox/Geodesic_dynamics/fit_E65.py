import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ======================================================
# ================ LOAD THE E65 DATASET ================
# ======================================================

print("Loading and preparing data...")
mat_path = os.environ.get("E65_MAT_PATH", "./E65.mat")
mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
nic_output = mat['nic_output']

# Extract raw arrays
ROIactivities = nic_output.ROIactivities # shape: [Time, Neurons]
trialn = nic_output.trialn               # shape: [Time]
sensory_input = nic_output.sensory_input # shape: [Time]

# Filter for active timepoints and neurons
Datarange = ROIactivities.sum(axis=1) > 0
Neurons_active = ROIactivities.sum(axis=0) > 0

all_data = ROIactivities[Datarange][:, Neurons_active]
trialn = trialn[Datarange]
sensory_input = sensory_input[Datarange]

# Use the specific subset of trials
trials_train = np.array([
    9,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,
    38,39,40,41,42,43,44,46,47,48,49,50,51,53,55,57,58,59,60,61,62,63,64,65,66,67,
    68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,
    94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,
    115,117,118,119,120,122,123,124,125,126,127,128,129,130,131,132,133,134,135,
    136,137,138,139,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,
    157,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,
    177,178,179,180,181,182,183,184,185,186,187,189,190,191,192,194,195,196,197,
    198,199,200,201,202,203,204,205,206,207,208,209,210
])

trial_to_idx = {t: i for i, t in enumerate(trials_train)}
num_trials = len(trials_train)

# Group data by trial for the PyTorch training loop
dataset_train = []
for t_id in trials_train:
    mask = (trialn == t_id)
    dataset_train.append({
        'idx': trial_to_idx[t_id],
        'rates': torch.tensor(all_data[mask], dtype=torch.float32),
        'sensory': torch.tensor(sensory_input[mask], dtype=torch.float32).unsqueeze(1),
        'seq_len': mask.sum()
    })

n_neurons = all_data.shape[1]
print(f"Data ready. {len(dataset_train)} trials in subset, {n_neurons} active neurons.")

# ======================================================
# ============ Define the metric tensor net ============
# ======================================================

torch.manual_seed(42)

class MetricNetwork(nn.Module):
    def __init__(self, latent_dim=5, hidden_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        out_dim = latent_dim * (latent_dim + 1) // 2
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.eps = 1e-4

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.net(x)
        L = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=x.device)
        
        # Generalized mapping for the Cholesky factor to support any latent_dim
        idx = 0
        for i in range(self.latent_dim):
            for j in range(i + 1):
                if i == j:
                    # Diagonal elements must be positive
                    L[:, i, j] = F.softplus(out[:, idx]) + 1e-3
                else:
                    # Off-diagonal elements can be any real number
                    L[:, i, j] = out[:, idx]
                idx += 1
                
        I = torch.eye(self.latent_dim, device=x.device).unsqueeze(0)
        g = torch.bmm(L, L.transpose(1, 2)) + self.eps * I
        return g

# ======================================================
# =============== Define Geodesic Dynamics =============
# ======================================================

class GeodesicDynamics(nn.Module):
    def __init__(self, metric_net):
        super().__init__()
        self.metric_net = metric_net
        self.sensory_drive = nn.Linear(1, metric_net.latent_dim, bias=False)
        self.log_friction = nn.Parameter(torch.tensor(-2.0)) 
    
    def compute_christoffel(self, x):
        x.requires_grad_(True)
        g = self.metric_net(x) 
        batch_size, d, _ = g.shape
        
        dg = torch.zeros(batch_size, d, d, d, device=x.device)
        for i in range(d):
            for j in range(d):
                grad_g = torch.autograd.grad(
                    outputs=g[:, i, j].sum(), 
                    inputs=x, 
                    create_graph=True, 
                    retain_graph=True
                )[0]
                dg[:, i, j, :] = grad_g 

        g_inv = torch.inverse(g)
        Gamma = torch.zeros(batch_size, d, d, d, device=x.device)
        for k in range(d):
            for m in range(d):
                for n in range(d):
                    term = 0
                    for l in range(d):
                        term += 0.5 * g_inv[:, k, l] * (
                            dg[:, n, l, m] + dg[:, m, l, n] - dg[:, m, n, l]
                        )
                    Gamma[:, k, m, n] = term
        return Gamma

    
    def get_acceleration(self, x, v, sensory_t):
        Gamma = self.compute_christoffel(x)
        d = x.shape[1]
        a = torch.zeros_like(v)
        for k in range(d):
            for m in range(d):
                for n in range(d):
                    a[:, k] -= Gamma[:, k, m, n] * v[:, m] * v[:, n]
        friction = torch.exp(self.log_friction) * v
        return a + self.sensory_drive(sensory_t) - friction
    
    def forward(self, state, dt, sensory_t):
        # RK4 Integration
        x, v = state
        
        k1_v = self.get_acceleration(x, v, sensory_t)
        k1_x = v
        
        k2_v = self.get_acceleration(x + 0.5 * dt * k1_x, v + 0.5 * dt * k1_v, sensory_t)
        k2_x = v + 0.5 * dt * k1_v
        
        k3_v = self.get_acceleration(x + 0.5 * dt * k2_x, v + 0.5 * dt * k2_v, sensory_t)
        k3_x = v + 0.5 * dt * k2_v
        
        k4_v = self.get_acceleration(x + dt * k3_x, v + dt * k3_v, sensory_t)
        k4_x = v + dt * k3_v
        
        v_next = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        x_next = x + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        
        return x_next, v_next

# ======================================================
# ============= MLP from latent to neural ==============
# ======================================================

class NeuralDecoder(nn.Module):
    def __init__(self, latent_dim=5, n_neurons=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_neurons),
            nn.Softplus()
        )
        
    def forward(self, x):
        return self.net(x)

# ======================================================
# ============= Learn geometry from data ===============
# ======================================================

class InverseGeodesicModel(nn.Module):
    def __init__(self, num_trials, latent_dim=5, n_neurons=300):
        super().__init__()
        self.metric = MetricNetwork(latent_dim)
        self.dynamics = GeodesicDynamics(self.metric)
        self.decoder = NeuralDecoder(latent_dim, n_neurons)
        
        self.x0 = nn.Parameter(torch.randn(num_trials, latent_dim) * 0.1) # Every trial has different initial conditions
        self.v0 = nn.Parameter(torch.randn(num_trials, latent_dim) * 0.1)
        self.log_tau = nn.Parameter(torch.tensor(0.0)) 

    def forward(self, trial_idx, sensory_seq, seq_len):
        dt_physical = 1.0
        dt_scaled = dt_physical * torch.exp(self.log_tau) 
        
        x = self.x0[trial_idx].unsqueeze(0) # [1, latent_dim]
        v = self.v0[trial_idx].unsqueeze(0)
        
        latents = []
        for i in range(seq_len):
            latents.append(x)
            s_t = sensory_seq[i].unsqueeze(0) # [1, 1]
            x, v = self.dynamics((x, v), dt_scaled, s_t) # Geodesic dynamics, starting at x|v.
            
        latents = torch.cat(latents, dim=0) 
        rates = self.decoder(latents) 
        return latents, rates

# ======================================================
# ============== Baseline: Neural ODE ==================
# ======================================================

class FreeDynamics(nn.Module):
    def __init__(self, latent_dim=5, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + 1, hidden_dim), # +1 for sensory input
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, state, dt, sensory_t):
        x, v = state
        state_vec = torch.cat([x, v, sensory_t], dim=-1) 
        a = self.net(state_vec) 
        
        v_next = v + a * dt
        x_next = x + v * dt
        return x_next, v_next

class InverseFreeModel(nn.Module):
    def __init__(self, num_trials, latent_dim=5, n_neurons=300):
        super().__init__()
        self.dynamics = FreeDynamics(latent_dim)
        self.decoder = NeuralDecoder(latent_dim, n_neurons)
        
        self.x0 = nn.Parameter(torch.randn(num_trials, latent_dim) * 0.1)
        self.v0 = nn.Parameter(torch.randn(num_trials, latent_dim) * 0.1)
        self.log_tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, trial_idx, sensory_seq, seq_len):
        dt_physical = 1.0
        dt_scaled = dt_physical * torch.exp(self.log_tau)
        
        x = self.x0[trial_idx].unsqueeze(0)
        v = self.v0[trial_idx].unsqueeze(0)
        
        latents = []
        for i in range(seq_len):
            latents.append(x)
            s_t = sensory_seq[i].unsqueeze(0)
            x, v = self.dynamics((x, v), dt_scaled, s_t)
            
        latents = torch.cat(latents, dim=0)
        rates = self.decoder(latents)
        return latents, rates

# ======================================================
# ============= Training and comparison ================
# ======================================================

def train_and_evaluate(model, dataset, epochs=300, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.PoissonNLLLoss(log_input=False, reduction='sum')
    loss_history = []
    
    print(f"Training {model.__class__.__name__}...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_nll = 0.0
        
        for trial in dataset:
            _, pred_rates = model(trial['idx'], trial['sensory'], trial['seq_len'])
            total_nll += loss_fn(pred_rates, trial['rates'])
            
        total_nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        loss_history.append(total_nll.item())
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | NLL Sum: {total_nll.item():.2f}")

    return model, loss_history, total_nll.item()

def calculate_ic(nll_sum, num_params, num_obs):
    aic = 2 * num_params + 2 * nll_sum
    bic = num_params * np.log(num_obs) + 2 * nll_sum
    return aic, bic

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ======================================================
# ================= Results and Plot ===================
# ======================================================

# latent_dim should be 5. Can be overridden without changing the script body.
latent_dim = int(os.environ.get("GEODESIC_LATENT_DIM", "5"))
model_geo = InverseGeodesicModel(num_trials=num_trials, latent_dim=latent_dim, n_neurons=n_neurons)
model_free = InverseFreeModel(num_trials=num_trials, latent_dim=latent_dim, n_neurons=n_neurons)

params_geo = count_parameters(model_geo)
params_free = count_parameters(model_free)
print(f"\nGeodesic Model Parameters: {params_geo}")
print(f"Free Dynamics Model Parameters: {params_free}\n")

num_obs = sum(trial['seq_len'] * n_neurons for trial in dataset_train)

# Train Models
EPOCHS = int(os.environ.get("GEODESIC_EPOCHS", "500"))
model_geo, loss_geo, nll_geo = train_and_evaluate(model_geo, dataset_train, epochs=EPOCHS)
print("-" * 20)
model_free, loss_free, nll_free = train_and_evaluate(model_free, dataset_train, epochs=EPOCHS)

# Get AIC/BIC
aic_geo, bic_geo = calculate_ic(nll_geo, params_geo, num_obs)
aic_free, bic_free = calculate_ic(nll_free, params_free, num_obs)

print("\n" + "="*45)
print("            MODEL COMPARISON RESULTS ")
print("="*45)
print(f"{'Metric':<15} | {'Geodesic Model':<15} | {'Free Model':<15}")
print("-" * 48)
print(f"{'Parameters (k)':<15} | {params_geo:<15} | {params_free:<15}")
print(f"{'Final NLL':<15} | {nll_geo:<15.2f} | {nll_free:<15.2f}")
print(f"{'AIC':<15} | {aic_geo:<15.2f} | {aic_free:<15.2f}")
print(f"{'BIC':<15} | {bic_geo:<15.2f} | {bic_free:<15.2f}")
print("="*45)

best_aic = "Geodesic" if aic_geo < aic_free else "Free Dynamics"
best_bic = "Geodesic" if bic_geo < bic_free else "Free Dynamics"

print(f"\nPreferred Model by AIC: {best_aic}")
print(f"Preferred Model by BIC: {best_bic}")



# === And a new plot routine ==


def plot_model_heatmap(dataset, model_g, model_f, dataset_idx=0, num_neurons=50):
    """Plots true vs predicted population rates as side-by-side heatmaps."""
    if len(dataset) == 0:
        print("Dataset is empty, nothing to plot.")
        return

    trial_data = dataset[dataset_idx]
    rates_true = trial_data['rates'].detach().numpy()
    
    # 1. Geodesic model MUST evaluate with gradients enabled
    _, rates_geo_pred = model_g(trial_data['idx'], trial_data['sensory'], trial_data['seq_len'])
    
    # 2. The free ODE model can safely use torch.no_grad()
    with torch.no_grad():
        _, rates_free_pred = model_f(trial_data['idx'], trial_data['sensory'], trial_data['seq_len'])
        
    # Detach predictions and convert to NumPy
    rates_geo_pred = rates_geo_pred.detach().numpy()
    rates_free_pred = rates_free_pred.detach().numpy()
    
    # Select up to `num_neurons` (or all if fewer than requested)
    n_plot = min(num_neurons, rates_true.shape[1])
    
    # Transpose so time is x-axis (columns), neurons are y-axis (rows)
    mat_true = rates_true[:, :n_plot].T
    mat_geo = rates_geo_pred[:, :n_plot].T
    mat_free = rates_free_pred[:, :n_plot].T
    
    # Determine global min and max for consistent color scaling across all plots
    vmin = min(mat_true.min(), mat_geo.min(), mat_free.min())
    vmax = 1 #max([mat_true.max(), mat_geo.max(), mat_free.max()])
    
    # Create a 1x3 grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    
    # Plot 1: True Data
    im0 = axes[0].imshow(mat_true, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title('True Data')
    axes[0].set_ylabel('Neuron Index')
    axes[0].set_xlabel('Time Step')
    
    # Plot 2: Geodesic Fit
    im1 = axes[1].imshow(mat_geo, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title('Geodesic Fit')
    axes[1].set_xlabel('Time Step')
    
    # Plot 3: Free ODE Fit
    im2 = axes[2].imshow(mat_free, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axes[2].set_title('Free ODE Fit')
    axes[2].set_xlabel('Time Step')
    
    # Add a single shared colorbar for the whole figure
    fig.colorbar(im2, ax=axes.ravel().tolist(), label='Firing Rate', shrink=0.8)
    
    # Use trials_train if it's in the global scope, otherwise just use the index
    try:
        trial_name = trials_train[dataset_idx]
    except NameError:
        trial_name = dataset_idx
        
    plt.suptitle(f'Population Activity Heatmaps for Trial {trial_name}', fontsize=14, y=1.02)
    out_path = os.environ.get("GEODESIC_HEATMAP_PATH")
    if out_path:
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"Saved heatmap to {out_path}")
    else:
        plt.show()

plot_model_heatmap(dataset_train, model_geo, model_free, dataset_idx=12, num_neurons=50)
