from nlb_tools.nwb_interface import NWBDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

## Load dataset
dataset = NWBDataset("~/lvm/code/dandi/000129/sub-Indy", "*train", split_heldout=False)
dataset.data

# Make trial data

# Find when target pos changes
has_change = dataset.data.target_pos.fillna(-1000).diff(axis=0).any(axis=1) # filling NaNs with arbitrary scalar to treat as one block
# Find if target pos change corresponds to NaN-padded gap between files
change_nan = dataset.data[has_change].isna().any(axis=1)
# Drop trials containing the gap and immediately before and after, as those trials may be cut short
drop_trial = (change_nan | change_nan.shift(1, fill_value=True) | change_nan.shift(-1, fill_value=True))[:-1]
# Add start and end times to trial info
change_times = dataset.data.index[has_change]
start_times = change_times[:-1][~drop_trial]
end_times = change_times[1:][~drop_trial]
# Get target position per trial
target_pos = dataset.data.target_pos.loc[start_times].to_numpy().tolist()
# Compute reach distance and angle
reach_dist = dataset.data.target_pos.loc[end_times - pd.Timedelta(1, 'ms')].to_numpy() - dataset.data.target_pos.loc[start_times - pd.Timedelta(1, 'ms')].to_numpy()
reach_angle = np.arctan2(reach_dist[:, 1], reach_dist[:, 0]) / np.pi * 180
# Create trial info
dataset.trial_info = pd.DataFrame({
    'trial_id': np.arange(len(start_times)),
    'start_time': start_times,
    'end_time': end_times,
    'target_pos': target_pos,
    'reach_dist_x': reach_dist[:, 0],
    'reach_dist_y': reach_dist[:, 1],
    'reach_angle': reach_angle,
})

dataset.resample(5)

# Extract start and end times, target positions
start = dataset.trial_info.iloc[0].start_time # arbitrarily using 1st through 6th reaches
end = dataset.trial_info.iloc[7].end_time - pd.Timedelta(1, 'ms')
targets = dataset.trial_info.target_pos.iloc[0:8].to_numpy().tolist()
tts = dataset.trial_info.end_time.iloc[0:8] - start

# Get cursor position data
reach_data = dataset.data.cursor_pos.loc[start:end].to_numpy().reshape(-1, 1, 2)
reach_seg = np.concatenate([reach_data[:-1], reach_data[1:]], axis=1)
# Split into collection of lines for color gradient
lc = LineCollection(reach_seg, cmap='rainbow', norm=plt.Normalize(0, len(reach_seg) * dataset.bin_width / 1000))
lc.set_array(np.arange(len(reach_seg)) * dataset.bin_width / 1000)

# Plot lines and add targets
ax = plt.axes()
lines = ax.add_collection(lc)
for tt, target in zip(tts, targets):
    ax.plot(target[0], target[1], marker='o', markersize=8, color=plt.get_cmap('rainbow')(tt.total_seconds() * 1000 / dataset.bin_width / len(reach_seg)))
ax.set_xlim(-60, 60)
ax.set_ylim(-10, 130)
plt.colorbar(lines, label='time (s)')
plt.show()

# Calculate speed and call `calculate_onset`
speed = np.linalg.norm(dataset.data.finger_vel, axis=1)
dataset.data['speed'] = speed
peak_times = dataset.calculate_onset('speed', 0.05)

# Smooth spikes with 50 ms std Gaussian
dataset.smooth_spk(50, name='smth_50', ignore_nans=True)

# Lag velocity by 120 ms relative to neural data
lag = 120
lag_bins = int(round(lag / dataset.bin_width))
nans = dataset.data.finger_vel.x.isna().reset_index(drop=True)
rates = dataset.data.spikes_smth_50[~nans.to_numpy() & ~nans.shift(-lag_bins, fill_value=True).to_numpy()].to_numpy()
vel = dataset.data.finger_vel[~nans.to_numpy() & ~nans.shift(lag_bins, fill_value=True).to_numpy()].to_numpy()
vel_index = dataset.data.finger_vel[~nans.to_numpy() & ~nans.shift(lag_bins, fill_value=True).to_numpy()].index

# Fit decoder and evaluate
gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
gscv.fit(rates, vel)
print(f'Decoding R2: {gscv.best_score_}')
pred_vel = gscv.predict(rates)

# Add data back to main dataframe
pred_vel_df = pd.DataFrame(pred_vel, index=vel_index, columns=pd.MultiIndex.from_tuples([('pred_vel', 'x'), ('pred_vel', 'y')]))
dataset.data = pd.concat([dataset.data, pred_vel_df], axis=1)

# Coloring function
get_color = lambda idx, series: plt.get_cmap('Greens')((series[idx] - series.min()) / (series.max() - series.min()))
# Extract trial data aligned to movement onset
trial_data = dataset.make_trial_data(align_field='speed_onset', align_range=(-100, 400), allow_nans=True) 

# Initialize figure
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
# Loop through trials and plot on appropriate subplots
for tid, trial in trial_data.groupby('trial_id'):
    axs[0][0].plot(np.arange(-100, 400, dataset.bin_width), trial.finger_vel.x, color=get_color(tid, dataset.trial_info.reach_dist_x))
    axs[0][1].plot(np.arange(-100, 400, dataset.bin_width), trial.finger_vel.y, color=get_color(tid, dataset.trial_info.reach_dist_y))
    axs[1][0].plot(np.arange(-100, 400, dataset.bin_width), trial.pred_vel.x, color=get_color(tid, dataset.trial_info.reach_dist_x))
    axs[1][1].plot(np.arange(-100, 400, dataset.bin_width), trial.pred_vel.y, color=get_color(tid, dataset.trial_info.reach_dist_y))

# Add labels
axs[0][0].set_title('X velocity')
axs[0][1].set_title('Y velocity')
axs[0][0].set_ylabel('True velocity')
axs[1][0].set_ylabel('Predicted velocity')
axs[1][0].set_xlabel('Time after move onset (ms)')
axs[1][1].set_xlabel('Time after move onset (ms)')
axs[0][0].set_xlim(-100, 400)
plt.show()

# Seed generator for consistent plots
np.random.seed(2021)
n_conds = 27 # number of conditions to plot

# Get unique conditions
conds = dataset.trial_info.set_index(['trial_type', 'trial_version']).index.unique().tolist()

# Loop through conditions
rates = []
colors = []
for i in np.random.choice(len(conds), n_conds):
    cond = conds[i]
    # Find trials in condition
    mask = np.all(dataset.trial_info[['trial_type', 'trial_version']] == cond, axis=1)
    # Extract trial data
    trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(-50, 450), ignored_trials=(~mask))
    # Append averaged smoothed spikes for condition
    rates.append(trial_data.groupby('align_time')[trial_data[['spikes_smth_50']].columns].mean().to_numpy())
    # Append reach angle-based color for condition
    active_target = dataset.trial_info[mask].target_pos.iloc[0][dataset.trial_info[mask].active_target.iloc[0]]
    reach_angle = np.arctan2(*active_target[::-1])
    colors.append(plt.cm.hsv(reach_angle / (2*np.pi) + 0.5))

# Stack data and apply PCA
rate_stack = np.vstack(rates)
rate_scaled = StandardScaler().fit_transform(rate_stack)
pca = PCA(n_components=3)
traj_stack = pca.fit_transform(rate_scaled)
traj_arr = traj_stack.reshape((n_conds, len(rates[0]), -1))

# Loop through trajectories and plot
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
for traj, col in zip(traj_arr, colors):
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=col)
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=col) 

# Add labels
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

## Plot predicted vs true kinematics

# Choose 23rd condition to plot
cond = conds[23]

# Find trials in condition and extract data
mask = np.all(dataset.trial_info[['trial_type', 'trial_version']] == cond, axis=1)
trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(-50, 450), ignored_trials=(~mask))

# Initialize figure
fig, axs = plt.subplots(2, 3, figsize=(10, 4))
t = np.arange(-50, 450, dataset.bin_width)

# Loop through trials in condition
for _, trial in trial_data.groupby('trial_id'):
    # True and predicted x velocity
    axs[0][0].plot(t, trial.hand_vel.x, linewidth=0.7, color='black')
    axs[1][0].plot(t, trial.pred_vel.x, linewidth=0.7, color='blue')
    # True and predicted y velocity
    axs[0][1].plot(t, trial.hand_vel.y, linewidth=0.7, color='black')
    axs[1][1].plot(t, trial.pred_vel.y, linewidth=0.7, color='blue')
    # True and predicted trajectories
    true_traj = np.cumsum(trial.hand_vel.to_numpy(), axis=0) * dataset.bin_width / 1000
    pred_traj = np.cumsum(trial.pred_vel.to_numpy(), axis=0) * dataset.bin_width / 1000
    axs[0][2].plot(true_traj[:, 0], true_traj[:, 1], linewidth=0.7, color='black')
    axs[1][2].plot(pred_traj[:, 0], pred_traj[:, 1], linewidth=0.7, color='blue')

# Set up shared axes
for i in range(2):
    axs[i][0].set_xlim(-50, 450)
    axs[i][1].set_xlim(-50, 450)
    axs[i][2].set_xlim(-180, 180)
    axs[i][2].set_ylim(-130, 130)

# Add labels
axs[0][0].set_title('X velocity (mm/s)')
axs[0][1].set_title('Y velocity (mm/s)')
axs[0][2].set_title('Reach trajectory')
plt.show()

# Extract neural data and lagged hand velocity
trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(-130, 370))
lagged_trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(-50, 450))
rates = trial_data.spikes_smth_50.to_numpy()
vel = lagged_trial_data.hand_vel.to_numpy()

# Fit and evaluate decoder
gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 5)})
gscv.fit(rates, vel)
pred_vel = gscv.predict(rates)
print(f"Decoding R2: {gscv.best_score_}")

# Merge predictions back to continuous data
pred_vel_df = pd.DataFrame(pred_vel, index=lagged_trial_data.clock_time, columns=pd.MultiIndex.from_tuples([('pred_vel', 'x'), ('pred_vel', 'y')]))
dataset.data = pd.concat([dataset.data, pred_vel_df], axis=1)

# Seed generator for consistent plots
np.random.seed(2468)
n_conds = 8 # number of conditions to plot

# Smooth spikes with 50 ms std Gaussian
dataset.smooth_spk(50, name='smth_50')

# Plot random neuron
neur_num = np.random.choice(dataset.data.spikes.columns)

# Find unique conditions
conds = dataset.trial_info.set_index(['trial_type', 'trial_version']).index.unique().tolist()

# Plot random subset of conditions
for i in np.random.choice(len(conds), size=n_conds, replace=False):
    cond = conds[i]
    # Find trials in condition
    mask = np.all(dataset.trial_info[['trial_type', 'trial_version']] == cond, axis=1)
    # Extract trial data
    trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(-50, 450), ignored_trials=(~mask))
    # Average hand position across trials
    psth = trial_data.groupby('align_time')[[('spikes_smth_50', neur_num)]].mean().to_numpy() / dataset.bin_width * 1000
    # Color PSTHs by reach angle
    active_target = dataset.trial_info[mask].target_pos.iloc[0][dataset.trial_info[mask].active_target.iloc[0]]
    reach_angle = np.arctan2(*active_target[::-1])
    # Plot reach
    plt.plot(np.arange(-50, 450, dataset.bin_width), psth, label=cond, color=plt.cm.hsv(reach_angle / (2*np.pi) + 0.5))

# Add labels
plt.ylim(bottom=0)
plt.xlabel('Time after movement onset (ms)')
plt.ylabel('Firing rate (spk/s)')
plt.title(f'Neur {neur_num} PSTH')
plt.legend(title='condition', loc='upper right')
plt.show()