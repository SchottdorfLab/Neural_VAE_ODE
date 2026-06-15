import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random
import torch
import torch.nn as nn
import torch.optim as optim

def spherical_geodesic(t, y):
    """
    Computes the derivatives for the geodesic equations on a sphere.
    y = [theta, phi, dtheta/dt, dphi/dt]
    """
    theta, phi, dtheta, dphi = y
    
    epsilon = 1e-8  # Avoid exact division by zero at the poles
    
    # Geodesic equations derived from Christoffel symbols:
    # d^2(theta)/dt^2 = sin(theta) * cos(theta) * (dphi/dt)^2
    # d^2(phi)/dt^2   = -2 * cot(theta) * (dtheta/dt) * (dphi/dt)
    ddtheta = np.sin(theta) * np.cos(theta) * dphi**2
    ddphi = -2.0 * (np.cos(theta) / (np.sin(theta) + epsilon)) * dtheta * dphi
    
    return [dtheta, dphi, ddtheta, ddphi]

# Initial conditions: Start at the equator, moving at an angle to trace a tilted great circle
y0 = [np.pi/2, 0.0, 0.5, 0.5] 
t_span = (0, 4 * np.pi) # Integrate long enough to wrap around the sphere
t_eval = np.linspace(t_span[0], t_span[1], 600)

# Solve the ODE
sol = solve_ivp(spherical_geodesic, t_span, y0, t_eval=t_eval, method='RK45')
theta_t = sol.y[0]
phi_t = sol.y[1]

# Place Fields Simulation
N_neurons = 300
kappa = 1.5 # Tuning Width

# Tile the sphere with "place field" 
indices = np.arange(0, N_neurons, dtype=float) + 0.5
phi_centers = np.pi * (1 + 5**0.5) * indices
theta_centers = np.arccos(1 - 2 * indices / N_neurons)

# Calculate Neural Activity for all neurons over time
activity = np.zeros((N_neurons, len(t_eval)))

for i in range(N_neurons):
    # von-Mises tuning: the dot product of the unit vectors pointing to the neuron's center and the trajectory's current position.
    # I chose this ad hoc. Shouldn't really matter I think.
    product = (np.cos(theta_centers[i]) * np.cos(theta_t) + 
                   np.sin(theta_centers[i]) * np.sin(theta_t) * np.cos(phi_t - phi_centers[i]))
    activity[i, :] = np.exp(kappa * product)


fig = plt.figure(figsize=(14, 6))

# Plot A: 3D Sphere, Trajectory, and Neuron Centers
ax1 = fig.add_subplot(131, projection='3d')

# Draw transparent sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
ax1.plot_surface(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)), np.outer(np.ones(np.size(u)), np.cos(v)), color='cyan', alpha=0.1, edgecolor='none')

# Convert trajectory to Cartesian
x_t = np.sin(theta_t) * np.cos(phi_t)
y_t = np.sin(theta_t) * np.sin(phi_t)
z_t = np.cos(theta_t)
ax1.plot(x_t, y_t, z_t, color='red', linewidth=3, label='Agent Trajectory')

# Convert neuron centers to Cartesian
x_c = np.sin(theta_centers) * np.cos(phi_centers)
y_c = np.sin(theta_centers) * np.sin(phi_centers)
z_c = np.cos(theta_centers)
ax1.scatter(x_c, y_c, z_c, color='black', s=10, alpha=0.6, label='Place Cell Centers')

ax1.set_title('Geodesic trajectory + place field centers')
ax1.legend()

ax2 = fig.add_subplot(132) # Raster plot of Neural Activity
peak_times = np.argmax(activity, axis=1)
peak_idx = np.argsort(peak_times)
random.shuffle(peak_idx)
random_activity = activity[peak_idx, :] # Random ordering of neurons

im = ax2.imshow(random_activity, aspect='auto', origin='lower', cmap='magma', extent=[t_eval[0], t_eval[-1], 0, N_neurons])
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Neuron ID (Sorted by Peak Activation)')
ax2.set_title('Neural Population Activity')
plt.colorbar(im, ax=ax2, label='Normalized Firing Rate')

ax3 = fig.add_subplot(133) # Sort neurons by their peak activity time to visualize the sequence
peak_idx = np.argsort(peak_times)
sorted_activity = activity[peak_idx, :] # Order neurons by peak time. Shows the sequential activation of the place fields
im = ax3.imshow(sorted_activity, aspect='auto', origin='lower', cmap='magma', extent=[t_eval[0], t_eval[-1], 0, N_neurons])
ax3.set_xlabel('Time (t)')
ax3.set_ylabel('Neuron ID (Sorted by Peak Activation)')
ax3.set_title('Neural Population Activity (sorted))')
plt.colorbar(im, ax=ax3, label='Normalized Firing Rate')


plt.tight_layout()
plt.show()


#=============== Geodesic fit ===================#

torch.manual_seed(42)

class MetricNetwork(nn.Module):
    """
    Parametrizes the metric tensor as a neural net.
    """
    def __init__(self, latent_dim=2, hidden_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        # MLP to predict the elements of the Cholesky factor L given coordinates.
        # For a d-dimensional space, L has d(d+1)/2 non-zero elements
        out_dim = latent_dim * (latent_dim + 1) // 2
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.eps = 1e-4 # Minimum bound to prevent singular matrices

    def forward(self, x):
        """
        Maps latent coordinate x to a positive-definite metric tensor g(x).
        x shape: [batch_size, latent_dim]
        """
        batch_size = x.shape[0]
        out = self.net(x)
        
        # Construct lower triangular matrix L
        L = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=x.device)
        
        # For d=2, indices are: L11, L21, L22
        # Exponentiate the diagonal to ensure positivity
        L[:, 0, 0] = torch.exp(out[:, 0]) 
        L[:, 1, 0] = out[:, 1]
        L[:, 1, 1] = torch.exp(out[:, 2])
        
        # g = L * L^T + eps * I
        I = torch.eye(self.latent_dim, device=x.device).unsqueeze(0)
        g = torch.bmm(L, L.transpose(1, 2)) + self.eps * I
        return g


class GeodesicDynamics(nn.Module):
    def __init__(self, metric_net):
        super().__init__()
        self.metric_net = metric_net

    def compute_christoffel(self, x):
        """
        Computes Christoffel as derivatives of metic tensory, using PyTorch Autograd.
        """
        x.requires_grad_(True)
        g = self.metric_net(x) # [batch, d, d]
        batch_size, d, _ = g.shape
        
        # Step 1. Compute spatial derivatives of the metric tensor (dg_ij / dx_k)
        dg = torch.zeros(batch_size, d, d, d, device=x.device)
        for i in range(d):
            for j in range(d):
                # Gradients of g_{ij} with respect to all x
                grad_g = torch.autograd.grad(
                    outputs=g[:, i, j].sum(), 
                    inputs=x, 
                    create_graph=True, 
                    retain_graph=True
                )[0]
                dg[:, i, j, :] = grad_g # Shape: [batch, d]

        # 2. Compute inverse metric g^{kl}
        g_inv = torch.inverse(g)
        
        # 3. Construct Christoffel symbols
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

    def forward(self, state, dt):
        """
        Performs one integration step (Euler) of the second-order ODE.
        This produces the geodesic trajectory, given the Gamma/g
        state: [x, v] where x and v are [batch, latent_dim]
        """
        x, v = state
        Gamma = self.compute_christoffel(x)
        d = x.shape[1]
        
        # Compute acceleration: a^k = - \sum_{m,n} Gamma^k_{mn} v^m v^n
        a = torch.zeros_like(v)
        for k in range(d):
            for m in range(d):
                for n in range(d):
                    a[:, k] -= Gamma[:, k, m, n] * v[:, m] * v[:, n]
        
        # Euler integration
        v_next = v + a * dt
        x_next = x + v * dt
        return x_next, v_next

class NeuralDecoder(nn.Module):
    """
    MLP from latent space to neural activity.
    """
    def __init__(self, latent_dim=2, n_neurons=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_neurons),
            nn.Softplus() # Firing rates must be strictly positive
        )
        
    def forward(self, x):
        return self.net(x)

class InverseGeodesicModel(nn.Module):
    """
    Assembles the model:
    1. Metric network is the NN model for gij
    2. Geodesic dynamics are the dynamics produced from (1)
    3. Neural decoder produces firing rates.
    ->  Gradient-descent the whole thing
    """
    def __init__(self, latent_dim=2, n_neurons=300):
        super().__init__()
        self.metric = MetricNetwork(latent_dim)
        self.dynamics = GeodesicDynamics(self.metric)
        self.decoder = NeuralDecoder(latent_dim, n_neurons)
        
        # Treat the initial state as a learnable parameter.
        self.x0 = nn.Parameter(torch.randn(1, latent_dim))
        self.v0 = nn.Parameter(torch.randn(1, latent_dim))

    def forward(self, t_eval):
        """
        Rolls out the latent trajectory and decodes to neural activity.
        """
        dt = t_eval[1] - t_eval[0] # Assuming uniform time steps
        seq_len = len(t_eval)
        
        x, v = self.x0, self.v0
        latents = []
        
        # Roll out dynamics
        for _ in range(seq_len):
            latents.append(x)
            x, v = self.dynamics((x, v), dt)
            
        latents = torch.cat(latents, dim=0) # [seq_len, latent_dim]
        
        # Decode to neural firing rates
        rates = self.decoder(latents) # [seq_len, n_neurons]
        return latents, rates


def train_inverse_model(target_activity, t_eval_np, epochs=300, lr=1e-3):
    # Convert numpy arrays to torch tensors
    # target_activity shape from your code: [N_neurons, Time] -> transpose to [Time, N_neurons]
    target_rates = torch.tensor(target_activity.T, dtype=torch.float32)
    t_eval = torch.tensor(t_eval_np, dtype=torch.float32)
    
    model = InverseGeodesicModel(latent_dim=2, n_neurons=target_activity.shape[0])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # We use a Poisson Negative Log-Likelihood (NLL) loss, standard for neural firing rates
    loss_fn = nn.PoissonNLLLoss(log_input=False)

    loss_history = []

    print("Starting optimization...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        pred_latents, pred_rates = model(t_eval)
        
        # Compute loss
        loss = loss_fn(pred_rates, target_rates)
        loss.backward()
        
        # Clip gradients (crucial for ODE solvers and autograd Christoffel symbols)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        loss_history.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return model, loss_history, pred_latents.detach()


# Finally train!
model, losses, recovered_latents = train_inverse_model(activity, t_eval, epochs=1000)
plt.plot(losses)

plt.scatter(recovered_latents[:,0], recovered_latents[:,1])
# Because the MLP can absorb coordinate transformations, the latent space might learn a stretched/warped sphere alongside a compensating metric tensor. 
# Think about Gnome projections


#=============== Model comparison with a usual neural ODE ===================#




class FreeDynamics(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=128):
        super().__init__()
        # Maps the concatenated state [x, v] directly to acceleration. No geodesics here.
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, state, dt):
        """
        Performs one Euler integration step using unconstrained neural dynamics.
        """
        x, v = state
        state_vec = torch.cat([x, v], dim=-1) # Shape: [batch, latent_dim * 2]
        
        # Free acceleration prediction
        a = self.net(state_vec) 
        
        # Euler integration
        v_next = v + a * dt
        x_next = x + v * dt
        return x_next, v_next

class InverseFreeModel(nn.Module):
    def __init__(self, latent_dim=2, n_neurons=300):
        super().__init__()
        self.dynamics = FreeDynamics(latent_dim)
        self.decoder = NeuralDecoder(latent_dim, n_neurons)
        
        # Learnable initial conditions
        self.x0 = nn.Parameter(torch.randn(1, latent_dim))
        self.v0 = nn.Parameter(torch.randn(1, latent_dim))

    def forward(self, t_eval):
        dt = t_eval[1] - t_eval[0]
        seq_len = len(t_eval)
        
        x, v = self.x0, self.v0
        latents = []
        
        for _ in range(seq_len):
            latents.append(x)
            x, v = self.dynamics((x, v), dt)
            
        latents = torch.cat(latents, dim=0)
        rates = self.decoder(latents)
        return latents, rates
    

def train_and_evaluate(model, target_activity, t_eval_np, epochs=300, lr=1e-3):
    target_rates = torch.tensor(target_activity.T, dtype=torch.float32)
    t_eval = torch.tensor(t_eval_np, dtype=torch.float32)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # We need the SUM of the negative log-likelihood for exact AIC/BIC scaling. I messed this up
    loss_fn = nn.PoissonNLLLoss(log_input=False, reduction='sum')

    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        pred_latents, pred_rates = model(t_eval)
        
        # The loss here is the negative log-likelihood (NLL) 
        # (excluding the constant term, which drops out in model comparison)
        nll_sum = loss_fn(pred_rates, target_rates)
        
        nll_sum.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        loss_history.append(nll_sum.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | NLL Sum: {nll_sum.item():.2f}")

    return model, loss_history, pred_latents.detach(), nll_sum.item()


def calculate_ic(nll_sum, num_params, num_obs):
    aic = 2 * num_params + 2 * nll_sum
    bic = num_params * np.log(num_obs) + 2 * nll_sum
    return aic, bic

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# PArameters
latent_dim = 2
n_neurons = activity.shape[0]
n_timepoints = len(t_eval)
num_obs = n_neurons * n_timepoints  # Total scalar data points

# Init models
model_geo = InverseGeodesicModel(latent_dim=latent_dim, n_neurons=n_neurons)
model_free = InverseFreeModel(latent_dim=latent_dim, n_neurons=n_neurons)

params_geo = count_parameters(model_geo)
params_free = count_parameters(model_free)
print(f"Geodesic Model Parameters: {params_geo}")
print(f"Free Dynamics Model Parameters: {params_free}\n")

#Train models
model_geo, loss_geo, latents_geo, nll_geo = train_and_evaluate(model_geo, activity, t_eval, epochs=300)
model_free, loss_free, latents_free, nll_free = train_and_evaluate(model_free, activity, t_eval, epochs=300)

#Get AIC/BIC
aic_geo, bic_geo = calculate_ic(nll_geo, params_geo, num_obs)
aic_free, bic_free = calculate_ic(nll_free, params_free, num_obs)

# Print in pretty
print("\n" + "="*40)
print(" MODEL COMPARISON RESULTS ")
print("="*40)

print(f"{'Metric':<15} | {'Geodesic Model':<15} | {'Free Model':<15}")
print("-" * 45)
print(f"{'Parameters (k)':<15} | {params_geo:<15} | {params_free:<15}")
print(f"{'Final NLL':<15} | {nll_geo:<15.2f} | {nll_free:<15.2f}")
print(f"{'AIC':<15} | {aic_geo:<15.2f} | {aic_free:<15.2f}")
print(f"{'BIC':<15} | {bic_geo:<15.2f} | {bic_free:<15.2f}")
print("="*40)

# Identify the preferred model
best_aic = "Geodesic" if aic_geo < aic_free else "Free Dynamics"
best_bic = "Geodesic" if bic_geo < bic_free else "Free Dynamics"

print(f"\nPreferred Model by AIC: {best_aic}")
print(f"Preferred Model by BIC: {best_bic}")