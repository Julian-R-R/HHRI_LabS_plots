import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# === Global Plotting Parameters ===
TITLE_FONTSIZE = 28
LABEL_FONTSIZE = 24
TICK_FONTSIZE = 22
LEGEND_FONTSIZE = 18
FIGURE_SIZE = (16, 10)
SAVE = True


# Target Cyprien 14° 

# === Data Configuration ===
filnames = {
    'raw': 'data/emg_raw_filt_mean.csv',
    'torque': 'data/torque_control.csv',
    'eval_torque_J_r': 'data/eval_torque_jul_r_2.csv',
    'eval_torque_J_l': 'data/eval_torque_jul_l_1.csv',
    'eval_torque_F': 'data/eval_torque_forarm_r_1.csv',
    'eval_torque_G': 'data/eval_torque_giu_r_1.csv',
    'eval_torque_C_r': 'data/eval_torque_C_r.csv',
    'position': 'data/position_control.csv',
    'eval_position_J_r': 'data/eval_pos_jul_r_2.csv', 
    'eval_position_J_l': 'data/eval_pos_jul_l_1.csv',
    'eval_position_F': 'data/eval_pos_forarm_r_1.csv',
    'eval_position_G': 'data/eval_pos_giu_r_1.csv',
    'eval_position_C_r': 'data/eval_pos_C_r.csv',
    'precise_pos': 'data/eval_precise_pos_jul_r_1_train.csv',
    'eval_precise_pos_J_r_train': 'data/eval_precise_pos_jul_r_1_train.csv',
    'eval_precise_pos_J_r': 'data/eval_precise_pos_jul_r_3.csv',
    'eval_precise_pos_J_l_train': 'data/eval_precise_pos_jul_l_1_train.csv',
    'eval_precise_pos_J_l': 'data/eval_precise_pos_jul_l_3.csv',
    'eval_precise_pos_F_train': 'data/eval_precise_pos_forarm_r_1_train.csv',
    'eval_precise_pos_F': 'data/eval_precise_pos_forarm_r_1.csv',
    'eval_precise_pos_C_r_train': 'data/eval_precise_pos_C_r_train.csv',
    'eval_precise_pos_C_r': 'data/eval_precise_pos_C_r.csv'

}

keep_data = {
    'raw': ('EMG value [V]', 'Filtered EMG value [V]', 'EMG mean [V]'),
    'torque': ('motor_torque [N.m]', 'encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_torque_J_r': ('motor_torque [N.m]', 'encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_torque_J_l': ('motor_torque [N.m]', 'encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_torque_F': ('motor_torque [N.m]', 'encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_torque_G': ('motor_torque [N.m]', 'encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_torque_C_r': ('motor_torque [N.m]', 'encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'position': ('encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_position_J_r': ('encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_position_J_l': ('encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_position_F': ('encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_position_G': ('encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_position_C_r': ('encoder_paddle_pos [deg]', 'EMG mean [V]'), #r_2
    'precise_pos': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_J_r_train': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_J_r': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_J_l_train': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_J_l': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_F_train': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_F': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_C_r_train': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_C_r': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]')
}

data_labels = {
    'raw': ['EMG value', 'Filtered EMG value', 'EMG mean'],
    'torque': ['Torque', 'Position', 'EMG mean'],
    'eval_torque_J_r': ['Torque', 'Position', 'EMG mean'],
    'eval_torque_J_l': ['Torque', 'Position', 'EMG mean'],
    'eval_torque_F': ['Torque', 'Position', 'EMG mean'],
    'eval_torque_G': ['Torque', 'Position', 'EMG mean'],
    'eval_torque_C_r': ['Torque', 'Position', 'EMG mean'],
    'position': ['Position', 'EMG mean'],
    'eval_position_J_r': ['Position', 'EMG mean'],
    'eval_position_J_l': ['Position', 'EMG mean'],
    'eval_position_F': ['Position', 'EMG mean'],
    'eval_position_G': ['Position', 'EMG mean'],
    'eval_position_C_r': ['Position', 'EMG mean'], #r_2
    'precise_pos': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_J_r_train': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_J_r': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_J_l_train': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_J_l': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_F_train': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_F': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_C_r_train': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_C_r': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall']
}

ranges = {
    'raw': (186, 188.8),
    'torque': (118.5, 123),
    'eval_torque_J_r': (726, 733),
    'eval_torque_J_l': (1692, 1701),
    'eval_torque_F': (120.5, 128),
    'eval_torque_G': (3554.5, 3561.5),
    'eval_torque_C_r': (-np.inf, np.inf), 
    'position': (56.5, 59.5),
    'eval_position_J_r': (1119, 1126),
    'eval_position_J_l': (2011, 2018),
    'eval_position_F': (79.5, 85),
    'eval_position_G': (3171, 3177),
    'eval_position_C_r': (np.inf, np.inf), 
    'precise_pos': (176, 182.25),
    'eval_precise_pos_J_r_train': (165.5, 171.5),
    'eval_precise_pos_J_r': (205, 211.8),
    'eval_precise_pos_J_l_train': (2335.5, 2341.2),
    'eval_precise_pos_J_l': (2718.5, 2724.5),
    'eval_precise_pos_F_train': (137.2, 141.2),
    'eval_precise_pos_F': (390.5, 394.75),
    'eval_precise_pos_C_r_train': (np.inf, np.inf),
    'eval_precise_pos_C_r': (np.inf, np.inf)
}

ranges_gauss = {
    'raw': (-np.inf, np.inf),
    'torque': (-np.inf, np.inf),
    'eval_torque_J_r': (-np.inf, np.inf),
    'eval_torque_J_l': (1682, 1737),
    'eval_torque_F': (-np.inf, np.inf),
    'eval_torque_G': (-np.inf, np.inf),
    'eval_torque_C_r': (1029, 1074),
    'position': (-np.inf, np.inf),
    'eval_position_J_r': (1119, 1154),
    'eval_position_J_l': (1984, 2018),
    'eval_position_F': (-np.inf, np.inf),
    'eval_position_G': (-np.inf, np.inf),
    'eval_position_C_r': (-np.inf, np.inf),
    'precise_pos': (-np.inf, np.inf),
    'eval_precise_pos_J_r_train': (155, 182),
    'eval_precise_pos_J_r': (173, 202),
    'eval_precise_pos_J_l_train': (2330.5, 2353),
    'eval_precise_pos_J_l': (2719, 2747),
    'eval_precise_pos_F_train': (134, 149),
    'eval_precise_pos_F': (383, 402),
    'eval_precise_pos_C_r_train': (491, 529.20),
    'eval_precise_pos_C_r': (724.95, 764.34)
}

target_values = {
    'raw': None,
    'torque': None,
    'eval_torque_J_r': 20,
    'eval_torque_J_l': 20,
    'eval_torque_F': 42,
    'eval_torque_G': 20,
    'eval_torque_C_r': 14,
    'position': None,
    'eval_position_J_r': 20,
    'eval_position_J_l': 20,
    'eval_position_F': 20,
    'eval_position_G': 20,
    'eval_position_C_r': 14,
    'precise_pos': None,
    'eval_precise_pos_J_r_train': 20,
    'eval_precise_pos_J_r': 20,
    'eval_precise_pos_J_l_train': 20,
    'eval_precise_pos_J_l': 20,
    'eval_precise_pos_F_train': 20,
    'eval_precise_pos_F': 20,
    'eval_precise_pos_C_r_train': 14,
    'eval_precise_pos_C_r': 14
}

# === DataFrame Construction ===
def build_df_info():
    """Builds a DataFrame with all run metadata."""
    df_dict = {
        'run': [],
        'filename': [],
        'columns': [],
        'data_labels': [],
        'ranges': [],
        'ranges_gauss': [],
        'target': []
    }
    for key in filnames:
        df_dict['run'].append(key)
        df_dict['filename'].append(filnames[key])
        df_dict['columns'].append(keep_data.get(key, None))
        df_dict['data_labels'].append(data_labels.get(key, None))
        df_dict['ranges'].append(ranges.get(key, None))
        df_dict['ranges_gauss'].append(ranges_gauss.get(key, None))
        df_dict['target'].append(target_values.get(key, None))
    df_info = pd.DataFrame(df_dict)
    df_info.to_csv('data_info.csv', index=False)
    return df_info

df_info = build_df_info()

# === Data Loading ===
def load_data(df_input, run="", trim_type='ranges', trim=False):
    """
    Load data from a CSV file into a pandas DataFrame.
    Optionally trims the data to a specified range.
    """
    filename = df_input.loc[df_input['run'] == run, 'filename'].values[0]
    try:
        df = pd.read_csv(filename, sep=';', index_col=0)
        print(f"Run {run} loaded successfully with index column.")
    except FileNotFoundError:
        print(f"Run {run} not found. Loading without index column.")
        df = None

    if df is None:
        return None

    if trim:
        start, end = df_input.loc[df_input['run'] == run, trim_type].values[0]
        if start > df.index[0] and end < df.index[-1]:
            df = df.loc[start:end]
            df.index = (df.index - df.index[0])

    keep = df_input.loc[df_input['run'] == run, 'columns'].values[0]
    df = df[list(keep)]
    return df

# === Utility Functions ===
def get_half_ticks(data_min, data_max):
    """Return ticks from below data_min to above data_max with 0.5 increments."""
    start = np.floor(data_min * 2) / 2
    end = np.ceil(data_max * 2) / 2
    return np.arange(start, end + 0.5, 0.5)

def divide_chunks(df, column='encoder_paddle_pos [deg]', threshold=15):
    """
    Divide the DataFrame into chunks based on when a signal rises above a threshold and stays above it.
    Only the specified column is kept in the output chunks.
    Discard chunks with less than 1000 values.
    """
    chunks = []
    current_chunk = []
    above_threshold = False
    for idx, value in df[column].items():
        if value > threshold:
            current_chunk.append(idx)
            above_threshold = True
        elif above_threshold:
            if current_chunk:
                chunk_array = df.loc[current_chunk, column].to_numpy()
                if len(chunk_array) >= 1000:
                    chunks.append(chunk_array)
                current_chunk = []
            above_threshold = False
    if current_chunk:
        chunk_array = df.loc[current_chunk, column].to_numpy()
        if len(chunk_array) >= 1000:
            chunks.append(chunk_array)
    return chunks

def divide_chunks_C(df, column='encoder_paddle_pos [deg]'):
    """
    Divide the DataFrame into chunks based on when a signal rises above a threshold and stays above it.
    Only the specified column is kept in the output chunks.
    Discard chunks with less than 1000 values.
    """
    chunks = []
    current_chunk = []
    time_windows = [(61.018, 67.945), (73, 79.888), (82.621, 90.686), (92.472, 98.555), (101.645, 110.080)]

    #cut a new chunk for ecah time window
    for start, end in time_windows:
        chunk = df[(df.index >= start) & (df.index <= end)]
        if not chunk.empty:
            chunk_array = chunk[column].to_numpy()
            if len(chunk_array) >= 1000:
                chunks.append(chunk_array)
    return chunks

def calculate_mean_std_precise(chunks):
    """Calculate the mean and standard deviation of the max value in each chunk."""
    maxs = [np.max(chunk) for chunk in chunks if len(chunk) > 0]
    return np.mean(maxs), np.std(maxs)

def calculate_mean_std_pos(chunks):
    """Calculate the mean and standard deviation of each chunk."""
    means = [np.mean(chunk) for chunk in chunks if len(chunk) > 0]
    stds = [np.std(chunk) for chunk in chunks if len(chunk) > 0]
    return means, stds

# === Plotting Functions ===
def plot_data(df, run, plot_name, save=False, all=False):
    """
    Plot the data for a given run, with appropriate formatting for each control strategy.
    """
    colors = ['royalblue', 'darkorange', 'teal', 'crimson', 'darkgreen', 'purple', 'gold']
    title_fontsize = TITLE_FONTSIZE
    label_fontsize = LABEL_FONTSIZE
    tick_fontsize = TICK_FONTSIZE
    legend_fontsize = LEGEND_FONTSIZE
    figsize = FIGURE_SIZE

    data_label = df_info.loc[df_info['run'] == run, 'data_labels'].values[0]
    target = df_info.loc[df_info['run'] == run, 'target'].values[0]
    x_ticks = get_half_ticks(df.index.min(), df.index.max())

    if "raw" in run:
        plt.figure(figsize=figsize)
        for i, col in enumerate(df.columns):
            plt.subplot(3, 1, i + 1)
            if i != 2:
                plt.plot(df.index, df[col], color=colors[i], label=data_label[i])
            else:
                plt.plot(df.index, df[col]*1000, color=colors[i], label=data_label[i])
            plt.title(f'Plot of {data_label[i]}', fontsize=title_fontsize)
            plt.xlabel('Time [s]', fontsize=label_fontsize)
            if i != 2:
                plt.ylabel('Values [V]', fontsize=label_fontsize)
            else:
                plt.ylabel('Values [mV]', fontsize=label_fontsize)
            plt.xticks(x_ticks, fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            if i != 2:
                plt.ylim(0.5, 1.8)
            else:
                plt.ylim(0, 8)
            plt.xlim(df.index[0], df.index[-1])
            plt.grid()
            plt.legend(fontsize=legend_fontsize, loc='upper right')
    elif 'torque' in run:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel('Time [s]', fontsize=label_fontsize)
        ax1.set_ylabel(data_label[1] + ' [deg]', color=colors[0], fontsize=label_fontsize)
        ax1.plot(df.index, df[df.columns[1]], color=colors[0], label=data_label[1])
        ax1.tick_params(axis='y', labelcolor=colors[0], labelsize=tick_fontsize)
        ax1.tick_params(axis='x', labelsize=tick_fontsize)
        ax1.grid()

        if 'eval' in run:
            ax1.axhline(y=target, color='green', linestyle='--', label=f'Target: {int(target)}°')
            ax1.axhspan(target-2.5, target+2.5, color='green', alpha=0.2, label='±2.5° zone')

        ax1.set_xticks(x_ticks)

        ax2 = ax1.twinx()
        color2 = colors[1]
        color3 = colors[2]
        ylabels = []
        if len(data_label) > 0:
            ylabels.append(data_label[0] + ' [mN.m]')
            # ylabels.append(' [N.m]')
        else:
            ylabels.append(df.columns[0])
        if len(data_label) > 2:
            ylabels.append(data_label[2] + ' [mV]')
            # ylabels.append(' [V]')
        else:
            ylabels.append(df.columns[2])
        ax2.set_ylabel(' / '.join(ylabels), color='black', fontsize=label_fontsize)
        ax2.plot(df.index, df[df.columns[0]]*1000, color=color2, label=data_label[0])
        ax2.plot(df.index, df[df.columns[2]]*1000, color=color3, label=data_label[2])
        ax2.tick_params(axis='y', labelcolor='black', labelsize=tick_fontsize)
        ax2.tick_params(axis='x', labelsize=tick_fontsize)
        ax2.set_xticks(x_ticks)

            # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=legend_fontsize)

        plt.title('Torque Control', fontsize=title_fontsize)
    elif 'position' in run:
        fig, ax1 = plt.subplots(figsize=figsize)
        color1 = colors[0]
        ax1.set_xlabel('Time [s]', fontsize=label_fontsize)
        ax1.set_ylabel(data_label[0] + ' [deg]', color=color1, fontsize=label_fontsize)
        ax1.plot(df.index, df[df.columns[0]], color=color1, label=data_label[0])
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=tick_fontsize)
        ax1.tick_params(axis='x', labelsize=tick_fontsize)
        ax1.grid()

        if 'eval' in run:
            ax1.axhline(y=target, color='green', linestyle='--', label=f'Target: {int(target)}°')
            ax1.axhspan(target-2.5, target+2.5, color='green', alpha=0.2, label='±2.5° zone')

        ax1.set_xticks(x_ticks)

        ax2 = ax1.twinx()
        color2 = colors[1]
        ax2.set_ylabel(data_label[1] + ' [mV]', color='black', fontsize=label_fontsize)
        ax2.plot(df.index, df[df.columns[1]]*1000, color=color2, label=data_label[1])
        ax2.tick_params(axis='y', labelcolor='black', labelsize=tick_fontsize)
        ax2.tick_params(axis='x', labelsize=tick_fontsize)
        ax2.set_xticks(x_ticks)

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=legend_fontsize)
        plt.title('Position Control', fontsize=title_fontsize)
    elif "precise" in run:
        fig, ax1 = plt.subplots(figsize=figsize)
        color1 = colors[0]
        ax1.set_xlabel('Time [s]', fontsize=label_fontsize)
        ax1.set_ylabel(data_label[0] + ' [deg]', color=color1, fontsize=label_fontsize)
        ax1.plot(df.index, df[df.columns[0]], color=color1, label=data_label[0])
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=tick_fontsize)
        ax1.tick_params(axis='x', labelsize=tick_fontsize)
        ax1.grid()

        if 'eval' in run:
            ax1.axhline(y=round(target), color='green', linestyle='--', label=f'Target: {int(target)}°')
            ax1.axhspan(target-2.5, target+2.5, color='green', alpha=0.2, label='±2.5° zone')

        ax1.set_xticks(x_ticks)

        ax2 = ax1.twinx()
        colors2 = colors[1:4]
        for i, col in enumerate(df.columns[1:]):
            label = data_label[i+1]
            ax2.plot(df.index, df[col]*1000, color=colors2[i], label=label)

        ylabels = []
        ylabels.append(data_label[1] + ' [mV]')
        if len(data_label) > 2:
            ylabels.append(data_label[2] + ' [mV]')
        if len(data_label) > 3:
            ylabels.append(data_label[3] + ' [mV]')
        ax2.set_ylabel(' / '.join(ylabels), color='black', fontsize=label_fontsize)
        ax2.tick_params(axis='y', labelcolor='black', labelsize=tick_fontsize)
        ax2.tick_params(axis='x', labelsize=tick_fontsize)
        ax2.set_xticks(x_ticks)

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=legend_fontsize)

        plt.title('Precise Position Control', fontsize=title_fontsize)
    else:
        plt.figure(figsize=(figsize))
        for i, col in enumerate(df.columns):
            label = data_label[i] if i < len(data_label) else col
            plt.plot(df.index, df[col], label=label, color=colors[i % len(colors)])
        plt.xlabel('Time [s]', fontsize=label_fontsize)
        plt.ylabel('Values [V]', fontsize=label_fontsize)
        plt.title(f'Plot of {run} data', fontsize=title_fontsize)
        plt.xticks(x_ticks, fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.legend(fontsize=legend_fontsize, ncol=len(df.columns))

    plt.xlim(df.index[0], df.index[-1])
    plt.tight_layout()
    if save:
        plt.savefig(f'{plot_name}', dpi=150)
    if not save:
        plt.show()

def plot_gaussian(mean, std, targets=None, labels=None, show=False):
    """
    Plot one or several Gaussian distributions based on the provided mean(s) and std(s).
    mean and std can be scalars or arrays/lists of the same length.
    Optionally, targets and labels can be provided for reference lines and legend.
    """
    import collections.abc

    title_fontsize = TITLE_FONTSIZE
    label_fontsize = LABEL_FONTSIZE
    tick_fontsize = TICK_FONTSIZE
    legend_fontsize = LEGEND_FONTSIZE
    figsize = FIGURE_SIZE

    # Ensure mean and std are iterable
    if not isinstance(mean, collections.abc.Iterable):
        mean = [mean]
    if not isinstance(std, collections.abc.Iterable):
        std = [std]

    if labels is None:
        labels = [fr'Gaussian {i+1}: $\mu$={m:.2f}, $\sigma$={s:.2f}' for i, (m, s) in enumerate(zip(mean, std))]
    else:
        labels = [fr'{label}: $\mu$={m:.2f}, $\sigma$={s:.2f}' for label, m, s in zip(labels, mean, std)]
    if targets is not None and not isinstance(targets, collections.abc.Iterable):
        targets = [targets]

    plt.figure(figsize=figsize)
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    for i, (m, s) in enumerate(zip(mean, std)):
        x = np.linspace(m - 4*s, m + 4*s, 1000)
        y = (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / s) ** 2)
        plt.plot(x, y, label=labels[i], color=colors[i])
    if targets is not None:
        for t in targets:
            plt.axvline(x=t, color='green', linestyle='--', label=f'Target: {int(t)}°')
            plt.axvspan(t-2.5, t+2.5, color='green', alpha=0.2, label='±2.5° zone')
    plt.title('Gaussian Distribution', fontsize=title_fontsize)
    plt.xlabel('Position [deg]', fontsize=label_fontsize)
    plt.ylabel('Probability Density', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.grid()
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    if show:
        plt.show()

def plot_mean_std(mean, std, targets=None, labels=None, show=False):
    """
    Plot means and std as scatter points with error bars.
    mean and std can be scalars or arrays/lists of the same length.
    Optionally, targets and labels can be provided for reference lines and legend.
    """
    import collections.abc

    title_fontsize = TITLE_FONTSIZE
    label_fontsize = LABEL_FONTSIZE
    tick_fontsize = TICK_FONTSIZE
    legend_fontsize = LEGEND_FONTSIZE
    figsize = FIGURE_SIZE

    if not isinstance(mean, collections.abc.Iterable):
        mean = [mean]
    if not isinstance(std, collections.abc.Iterable):
        std = [std]

    if labels is None:
        labels = [f'Mean {i+1}' for i in range(len(mean))]
    else:
        labels = [f'{label}: Mean={m:.2f}, Std={s:.2f}' for label, m, s in zip(labels, mean, std)]
    if targets is not None and not isinstance(targets, collections.abc.Iterable):
        targets = [targets]

    x = np.arange(len(mean))
    plt.figure(figsize=figsize)
    plt.errorbar(x, mean, yerr=std, fmt='o', capsize=5, color='skyblue', ecolor='black', elinewidth=1.5, markersize=8)
    for i, (xm, ym, ys) in enumerate(zip(x, mean, std)):
        text = f'{ym:.2f} ± {ys:.2f}'
        plt.text(xm + 0.05, ym, text, fontsize=14, va='center', ha='left', color='black', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    if targets is not None:
        for t in targets:
            plt.axhline(y=t, color='green', linestyle='--', label=f'Target: {int(t)}°')
            plt.axhspan(t-2.5, t+2.5, color='green', alpha=0.2, label='±2.5° zone')
    plt.title('Mean and Standard Deviation', fontsize=title_fontsize)
    plt.ylabel('Position [deg]', fontsize=label_fontsize)
    plt.xlabel('Runs', fontsize=label_fontsize)
    # Set x-axis ticks to 1, 2, 3, ... (instead of 0-based)
    plt.xticks(ticks=x, labels=[str(i+1) for i in x], fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.grid(axis='y')
    plt.legend(fontsize=legend_fontsize, loc='upper right')
    plt.tight_layout()
    if show:
        plt.show()

# === Analysis Functions ===
def compute_gaussian_precise(df, save_fig=True):
    """
    Compute and plot Gaussian distributions for precise position runs.
    """
    runs = ['eval_precise_pos_J_r', 'eval_precise_pos_J_l', 'eval_precise_pos_F', 'eval_precise_pos_C_r']
    for run in runs:
        target = df.loc[df['run'] == run, 'target'].values[0]
        data_train = load_data(df, run + '_train', trim_type='ranges_gauss', trim=True)
        data_val = load_data(df, run, trim_type='ranges_gauss', trim=True)
        new_data_train = divide_chunks(data_train, threshold=10 if run == 'eval_precise_pos_C_r' else 15)
        new_data_val = divide_chunks(data_val, threshold=10 if run == 'eval_precise_pos_C_r' else 15)
        mean, std = [], []
        mean_train, std_train = calculate_mean_std_precise(new_data_train)
        mean_val, std_val = calculate_mean_std_precise(new_data_val)
        if target == 42:
            shift = -22 
        elif target == 14:
            shift = 6 
        else:
            shift = 0
        mean_train += shift
        mean_val += shift
        target += shift
        mean.append(mean_train)
        std.append(std_train)
        mean.append(mean_val)
        std.append(std_val)
        labels = ['Train', 'Validation']
        plot_gaussian(mean, std, targets=[target], labels=labels, show=not save_fig)
        if save_fig:
            plot_name = 'plots_3/gaussian_' + run + '.png'
            plt.savefig(plot_name, dpi=150)
            print(f"Gaussian plot saved as {plot_name}")
        plt.close()

def compute_mean_std(df, run_type="pos", save_fig=True):
    """
    Compute and plot mean/std for all runs of a given type (position or torque).
    """
    if run_type == "pos":
        runs = ['eval_position_J_r', 'eval_position_J_l', 'eval_position_F', 'eval_position_G', 'eval_position_C_r']
    elif run_type == "torque":
        runs = ['eval_torque_J_r', 'eval_torque_J_l', 'eval_torque_F', 'eval_torque_G', 'eval_torque_C_r']
    labels = ['P1 right arm', 'P1 left arm', 'P1 forearm', 'P2 right arm', 
              'P3 right arm']
    global_mean = []
    global_std = []
    targets = []
    for run in runs:
        target = df.loc[df['run'] == run, 'target'].values[0]
        data_val = load_data(df, run, trim_type='ranges_gauss', trim=True)
        if run_type == "pos" and run == 'eval_position_C_r':
            new_data_val = divide_chunks_C(data_val)
        else:
            new_data_val = divide_chunks(data_val, threshold=5 if run_type == "pos" else target - 10)
        if len(new_data_val) == 0:
            print(f"No valid chunks found for run {run}. Skipping.")
            continue
        mean_val, std_val = calculate_mean_std_pos(new_data_val)
        if target == 42:
            shift = np.ones_like(mean_val) * -22 
        elif target == 14:
            shift = np.ones_like(mean_val) * 6 
        else:
            shift = np.zeros_like(mean_val)
        target += shift[0]
        mean_val = mean_val + shift
        global_mean.append(np.mean(mean_val))
        global_std.append(np.mean(std_val))
        # print(f"Run: {run}, Mean: {np.mean(mean_val):.2f}, Std: {np.mean(std_val):.2f}, Target: {target}, chunks: {len(new_data_val)}")
        plot_mean_std(mean_val, std_val, targets=[target], labels=['mean', 'std'], show=not save_fig)
        if save_fig:
            plot_name = 'plots_3/mean_std_' + run + '.png'
            plt.savefig(plot_name, dpi=150)
            print(f"Position plot saved as {plot_name}")
        # plt.close()
        if target not in targets:
                targets.append(target)
    
    plot_gaussian(global_mean, global_std, targets=targets, labels=labels, show=not save_fig)
    print("Global Mean and Std computed")

    if save_fig:
        plot_name = f'plots_3/global_mean_std_{run_type}.png'
        plt.savefig(plot_name, dpi=150)
        print(f"Global Mean and Std plot saved as {plot_name}")
    else:
        plt.show()
    plt.close()

def compute_all_gaussian(df, save_fig=True):
    """
    Plot Gaussian distributions for each subject (J_r, J_l, F, G) across all three control strategies:
    torque, position, and precise position.
    If the target is 42, shift all means and targets by -22 to recenter the target to 20.
    """
    subjects = [
        ("J_r", "P1 right arm"),
        ("J_l", "P1 left arm"),
        ("F", "P1 forearm"),
        ("G", "P2 right arm"),
        ("C_r", "P3 right arm")
    ]
    strategies = [
        ("torque", "eval_torque_{}"),
        ("position", "eval_position_{}"),
        ("precise_pos", "eval_precise_pos_{}"),
    ]
    for subj_code, subj_label in subjects:
        means = []
        stds = []
        labels = []
        targets = []
        shift = 0
        for strat_name, run_fmt in strategies:
            run = run_fmt.format(subj_code)
            if strat_name == "precise_pos" and run not in df['run'].values:
                run = f"eval_precise_pos_{subj_code}"
            if run not in df['run'].values:
                continue
            data = load_data(df, run, trim_type='ranges_gauss', trim=True)
            if data is None or data.empty:
                continue
            if strat_name == "precise_pos":
                chunks = divide_chunks(data, threshold=10 if run == 'eval_precise_pos_C_r' else 15)
                mean, std = calculate_mean_std_precise(chunks)
            else:
                threshold = 5 if strat_name == "position" else df.loc[df['run'] == run, 'target'].values[0] - 10
                if run == 'eval_position_C_r':
                    chunks = divide_chunks_C(data)
                else:
                    chunks = divide_chunks(data, column='encoder_paddle_pos [deg]', threshold=threshold)
                mean_list, std_list = calculate_mean_std_pos(chunks)
                mean = np.mean(mean_list)
                std = np.mean(std_list)
            target = df.loc[df['run'] == run, 'target'].values[0]
            if target == 42:
                shift = -22 
            elif target == 14:
                shift = 6 
            else:
                shift = 0
            means.append(mean + shift)
            stds.append(std)
            labels.append(strat_name.replace("_", " ").capitalize())
            t_shifted = target + shift if target is not None else None
            if t_shifted not in targets:
                targets.append(t_shifted)
        if means and stds:
            plot_gaussian(means, stds, targets=targets, labels=labels, show=not save_fig)
            if save_fig:
                plot_name = f'plots_3/gaussian_{subj_code}_all_strategies.png'
                plt.savefig(plot_name, dpi=150)
                print(f"Gaussian plot for {subj_label} saved as {plot_name}")
            plt.close()

def save_all_plots(save_figures=True):
    for run in filnames:
        data = load_data(df_info, run, trim=True)
        plot_name = 'plots_3/' + run + f'_plot.png'
        plot_data(data, run, plot_name, save=save_figures, all=True)
        print(f"Plots for {run} saved.")

# === Main Execution ===
if __name__ == "__main__":
    # run = 'eval_position_C_r'  # Example run to load and plot

    # SAVE = False

    # # Load data for a specific run
    # data = load_data(df_info, run, trim_type='ranges_gauss', trim=True)
    # new_data_val = divide_chunks_C(data, column='encoder_paddle_pos [deg]')

    # print(f"Loaded data for run {run} with {len(new_data_val)} chunks.")
    # mean_val, std_val = calculate_mean_std_pos(new_data_val)
    # print(f"Mean: {mean_val}, Std: {std_val}")

    # plot_data(data, run, '', save=SAVE, all=False)

    # Uncomment the desired analysis
    save_all_plots(save_figures=SAVE)
    compute_gaussian_precise(df_info, save_fig=SAVE)
    compute_mean_std(df_info, save_fig=SAVE)
    compute_mean_std(df_info, run_type="torque", save_fig=SAVE)
    compute_all_gaussian(df_info, save_fig=SAVE)

    if SAVE:
        print("All plots generated and saved successfully.")
    else:
        print("All plots generated and displayed interactively.")

