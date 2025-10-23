import numpy as np, os
# Random seed for reproducibility
np.random.seed(1)

# Parameters
N = 300   # number of neurons
T = 4000 # Simulation time
dim = 2 # 2D arena -> 2D latents hidden among N time series.

# Hyperparameters for the simulated foraging rat
s = 0.1 # Define geometric properties of the path. s is stepsize
noise = 0  # 0 = no noise

# -------------------------------------------------------
# Generate random walk trajectory inside unit circle
# -------------------------------------------------------
Tpool = round(10 * T)
xpool = []

while len(xpool) < Tpool:
    pos_candidates = 2 * (np.random.rand(dim) - 0.5)  # uniform in [-1,1]^dim
    if np.sum(pos_candidates**2) < 1:  # keep only inside unit circle
        xpool.append(pos_candidates)

xpool = np.array(xpool)
trajectory = np.zeros((T, dim))
unusedflag = np.ones(Tpool, dtype=bool)

trajectory[0, :] = xpool[0, :]
unusedflag[0] = False

# Choose 2nd position
distances = np.sqrt(np.sum((trajectory[0, :] - xpool)**2, axis=1))
val = (distances - s) ** 2
ind = np.argsort(val)
trajectory[1, :] = xpool[ind[0], :]
unusedflag[ind[0]] = False

# Build trajectory
for i in range(2, T):
    x_tminus = trajectory[i - 2, :]
    x_t = trajectory[i - 1, :]
    r = np.random.rand()

    allowed_list = xpool[unusedflag, :]
    distances = np.sqrt(np.sum((x_t - allowed_list) ** 2, axis=1))
    val = (distances - s * r) ** 2
    ind = np.argsort(val)

    testpoint_inacceptable = True
    idx_test = 0

    while testpoint_inacceptable and idx_test < len(allowed_list):
        x_cand = allowed_list[ind[idx_test], :]

        not_moving_forward = (
            np.sum((x_cand - x_t) ** 2) > np.sum((x_cand - x_tminus) ** 2)
        )
        step_too_large = np.sqrt(np.sum((x_cand - x_t) ** 2)) > 2 * s
        testpoint_inacceptable = not_moving_forward or step_too_large
        idx_test += 1

    if idx_test == len(allowed_list):
        x_cand = allowed_list[ind[0], :]

    trajectory[i, :] = x_cand

    # Find true index in xpool
    dists = np.sum((xpool - x_cand) ** 2, axis=1)
    idx_true = np.argmin(dists)
    unusedflag[idx_true] = False

# -------------------------------------------------------
# Place field activity
# -------------------------------------------------------
ff_pos = np.random.randint(0, 2, size=(N, dim)) - 1.5  # random edges
ff_pos = np.unique(ff_pos, axis=0)

while len(ff_pos) < N:
    ff_pos = np.vstack([ff_pos, 2 * (np.random.rand(1, dim) - 0.5)])

ff_rate = 5.0     # maximum firing rate
ff_width = 0.25   # place field width
bc_width = 0.25   # (not used later in MATLAB either)

activity = np.zeros((N, T))
for neuron in range(N):
    diff = trajectory - ff_pos[neuron, :]
    activity[neuron, :] = ff_rate * np.exp(
        -np.sum(diff**2, axis=1) / (2 * ff_width**2)
    )

data_rat = activity.T + noise * np.random.randn(T, N)  # neuron activity over time
data_rat[data_rat < 0] = 0

# assuming `data_rat` exists and shape is (T, N)
T, N = data_rat.shape
trial_len = 200          # 400 timepoints per trial (=> 20 trials)
num_trials = T // trial_len

roi = []
trial_idx = []
time = []

for i in range(num_trials):
    start = i * trial_len
    end = start + trial_len
    roi.append(data_rat[start:end].T)
    trial_idx.extend([i] * trial_len)
    time.extend(np.arange(trial_len))

roi = np.hstack(roi)  # shape [N, total_time]
trial_idx = np.array(trial_idx)
time = np.array(time)

save_dir = "src/random_walk_data"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "synthetic_rat_data.npz")

np.savez(save_path, roi=roi, Trial=trial_idx, Time=time)
print(f"Saved chunked synthetic dataset to: {save_path}")

np.savez("synthetic_rat_data.npz", roi=data_rat.T, Time=np.arange(T), Trial=np.zeros(T))
