import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random

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



