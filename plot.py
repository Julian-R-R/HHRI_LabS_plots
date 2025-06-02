import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

filnames = {
    'raw': 'data/emg_raw_filt_mean.csv',
    'torque': 'data/torque_control.csv',
    'eval_torque_J_r': 'data/eval_torque_jul_r_2.csv',
    'eval_torque_J_l': 'data/eval_torque_jul_l_1.csv',
    'eval_torque_F': 'data/eval_torque_forarm_r_1.csv',
    'eval_torque_G': 'data/eval_torque_giu_r_1.csv',
    'position': 'data/position_control.csv',
    'eval_position_J_r': 'data/eval_pos_jul_r_2.csv',
    'eval_position_J_l': 'data/eval_pos_jul_l_1.csv',
    'eval_position_F': 'data/eval_pos_forarm_r_1.csv',
    'eval_position_G': 'data/eval_pos_giu_r_1.csv',
    'precise_pos': 'data/eval_precise_pos_jul_r_1_train.csv',
    'eval_precise_pos_J_r_train': 'data/eval_precise_pos_jul_r_1_train.csv',
    'eval_precise_pos_J_r': 'data/eval_precise_pos_jul_r_3.csv',
    'eval_precise_pos_J_l_train': 'data/eval_precise_pos_jul_l_1_train.csv',
    'eval_precise_pos_J_l': 'data/eval_precise_pos_jul_l_3.csv',
    'eval_precise_pos_F_train': 'data/eval_precise_pos_forarm_r_1_train.csv',
    'eval_precise_pos_F': 'data/eval_precise_pos_forarm_r_1.csv'
}

keep_data = {
    'raw': ('EMG value [V]', 'Filtered EMG value [V]', 'EMG mean [V]'),
    'torque': ('motor_torque [N.m]', 'encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_torque_J_r': ('motor_torque [N.m]', 'encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_torque_J_l': ('motor_torque [N.m]', 'encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_torque_F': ('motor_torque [N.m]', 'encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_torque_G': ('motor_torque [N.m]', 'encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'position': ('encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_position_J_r': ('encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_position_J_l': ('encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_position_F': ('encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'eval_position_G': ('encoder_paddle_pos [deg]', 'EMG mean [V]'),
    'precise_pos': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_J_r_train': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_J_r': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_J_l_train': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_J_l': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_F_train': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]'),
    'eval_precise_pos_F': ('encoder_paddle_pos [deg]', 'EMG mean [V]', 'EMG thres rise [V]', 'EMG thres fall [V]')
}

data_labels = {
    'raw': ['EMG value', 'Filtered EMG value', 'EMG mean'],
    'torque': ['Torque', 'Position', 'EMG mean'],
    'eval_torque_J_r': ['Torque', 'Position', 'EMG mean'],
    'eval_torque_J_l': ['Torque', 'Position', 'EMG mean'],
    'eval_torque_F': ['Torque', 'Position', 'EMG mean'],
    'eval_torque_G': ['Torque', 'Position', 'EMG mean'],
    'position': ['Position', 'EMG mean'],
    'eval_position_J_r': ['Position', 'EMG mean'],
    'eval_position_J_l': ['Position', 'EMG mean'],
    'eval_position_F': ['Position', 'EMG mean'],
    'eval_position_G': ['Position', 'EMG mean'],
    'precise_pos': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_J_r_train': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_J_r': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_J_l_train': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_J_l': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_F_train': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall'],
    'eval_precise_pos_F': ['Position', 'EMG mean', 'Threshold rise', 'Threshold fall']
}

ranges = {
    'raw': (186, 188.8),
    'torque': (119.5, 123),
    'eval_torque_J_r': (726, 733),
    'eval_torque_J_l': (1692, 1701),
    'eval_torque_F': (120.5, 128),
    'eval_torque_G': (3554.5, 3561.5),
    'position': (56.5, 59.5),
    'eval_position_J_r': (1119, 1126),
    'eval_position_J_l': (2011, 2018),
    'eval_position_F': (79.5, 85),
    'eval_position_G': (3171, 3177),
    'precise_pos': (176, 182.25),
    'eval_precise_pos_J_r_train': (165.5, 171.5),
    'eval_precise_pos_J_r': (205, 211.8),
    'eval_precise_pos_J_l_train': (2335.5, 2341.2),
    'eval_precise_pos_J_l': (2718.5, 2724.5),
    'eval_precise_pos_F_train': (137.2, 141.2),
    'eval_precise_pos_F': (390.5, 394.75)
}

ranges_gauss = {
    'raw': (-np.inf, np.inf),
    'torque': (-np.inf, np.inf),
    'eval_torque_J_r': (-np.inf, np.inf),
    'eval_torque_J_l': (1682, 1737),
    'eval_torque_F': (-np.inf, np.inf),
    'eval_torque_G': (-np.inf, np.inf),
    'position': (-np.inf, np.inf),
    'eval_position_J_r': (1119, 1154),
    'eval_position_J_l': (1984, 2018),
    'eval_position_F': (-np.inf, np.inf),
    'eval_position_G': (-np.inf, np.inf),
    'precise_pos': (-np.inf, np.inf),
    'eval_precise_pos_J_r_train': (155, 182),
    'eval_precise_pos_J_r': (173, 202),
    'eval_precise_pos_J_l_train': (2330.5, 2353),
    'eval_precise_pos_J_l': (2719, 2747),
    'eval_precise_pos_F_train': (134, 149),
    'eval_precise_pos_F': (383, 402)
}

target_values = {
    'raw': None,
    'torque': None,
    'eval_torque_J_r': 20,
    'eval_torque_J_l': 20,
    'eval_torque_F': 42,
    'eval_torque_G': 20,
    'position': None,
    'eval_position_J_r': 20,
    'eval_position_J_l': 20,
    'eval_position_F': 20,
    'eval_position_G': 20,
    'precise_pos': None,
    'eval_precise_pos_J_r_train': 20,
    'eval_precise_pos_J_r': 20,
    'eval_precise_pos_J_l_train': 20,
    'eval_precise_pos_J_l': 20,
    'eval_precise_pos_F_train': 20,
    'eval_precise_pos_F': 20
}

# Create a DataFrame from the dictionaries

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

# save the DataFrame to a CSV file
df_info.to_csv('data_info.csv', index=False)

# start = df_info.loc[df_info['run'] == 'torque', 'ranges'].values[0][0]

# print(start)


def load_data(df_input, run="", trim_type='ranges', trim=False):
    """
    Load data from a CSV file into a pandas DataFrame.
    If the file is not found, it returns None.
    """
    filename = df_input.loc[df_input['run'] == run, 'filename'].values[0]

    try:
        df = pd.read_csv(filename, sep=';', index_col=0)
        print(f"Run {run} loaded successfully with index column.")
    except FileNotFoundError:
        print(f"Run {run} not found. Loading without index column.")
        # Attempt to load without specifying index_col
        df = None

    if df is None:
        return None
    
    if trim:
        start = df_input.loc[df_input['run'] == run, trim_type].values[0][0]
        end = df_input.loc[df_input['run'] == run, trim_type].values[0][1]
        # Select rows based on index values (not position)
        if start > df.index[0] and end < df.index[-1]:
            df = df.loc[start:end]
            # offset the index to start from 0
            df.index = (df.index - df.index[0])

    keep = df_input.loc[df_input['run'] == run, 'columns'].values[0]

    df = df[list(keep)]
        
    return df

def plot_data(df, run, plot_name, save=False, all=False):
    colors = ['royalblue', 'darkorange', 'teal', 'crimson', 'darkgreen', 'purple', 'gold']
    title_fontsize = 22
    label_fontsize = 14
    tick_fontsize = 14
    legend_fontsize = 14

    figsize = (14, 9)

    data_label = df_info.loc[df_info['run'] == run, 'data_labels'].values[0]
    target = df_info.loc[df_info['run'] == run, 'target'].values[0]

    if "raw" in run:
        plt.figure(figsize=figsize)
        for i, col in enumerate(df.columns):
            plt.subplot(3, 1, i + 1)
            plt.plot(df.index, df[col], color=colors[i], label=data_label[i])
            plt.title(f'Plot of {data_label[i]}', fontsize=title_fontsize)
            plt.xlabel('Time [s]', fontsize=label_fontsize)
            plt.ylabel('Values [V]', fontsize=label_fontsize)
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            if i != 2:
                plt.ylim(0.5, 1.8)
            else:
                plt.ylim(0, 0.008)
            plt.xlim(df.index[0], df.index[-1])
            plt.grid()
            plt.legend(fontsize=legend_fontsize, loc='upper right')
    elif 'torque' in run:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel('Time [s]', fontsize=label_fontsize)
        ax1.set_ylabel(data_label[1] + ' [deg]', color=colors[0], fontsize=label_fontsize)
        ax1.plot(df.index, df[df.columns[1]], color=colors[0], label=data_label[1] if len(data_label) > 1 else df.columns[1])
        ax1.tick_params(axis='y', labelcolor=colors[0], labelsize=tick_fontsize)
        ax1.tick_params(axis='x', labelsize=tick_fontsize)
        ax1.grid()

        if 'eval' in run:
            ax1.axhline(y=target, color='green', linestyle='--', label='Target')
            ax1.axhspan(target-2.5, target+2.5, color='green', alpha=0.2, label='Target region')

        ax2 = ax1.twinx()
        color2 = colors[1]
        color3 = colors[2]
        ylabels = []
        if len(data_label) > 0:
            ylabels.append(data_label[0])
            ylabels.append(' [N.m]')
        else:
            ylabels.append(df.columns[0])
        if len(data_label) > 2:
            ylabels.append(data_label[2])
            ylabels.append(' [V]')
        else:
            ylabels.append(df.columns[2])
        ax2.set_ylabel(' / '.join(ylabels), color=color2, fontsize=label_fontsize)
        ax2.plot(df.index, df[df.columns[0]], color=color2, label=ylabels[0])
        ax2.plot(df.index, df[df.columns[2]], color=color3, label=ylabels[1])
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=tick_fontsize)
        ax2.tick_params(axis='x', labelsize=tick_fontsize)

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
        ax1.plot(df.index, df[df.columns[0]], color=color1, label=data_label[0] if len(data_label) > 0 else df.columns[0])
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=tick_fontsize)
        ax1.tick_params(axis='x', labelsize=tick_fontsize)
        ax1.grid()

        if 'eval' in run:
            ax1.axhline(y=target, color='green', linestyle='--', label='Target')
            ax1.axhspan(target-2.5, target+2.5, color='green', alpha=0.2, label='Target region')

        ax2 = ax1.twinx()
        color2 = colors[1]
        ax2.set_ylabel(data_label[1] + ' [V]', color=color2, fontsize=label_fontsize)
        ax2.plot(df.index, df[df.columns[1]], color=color2, label=data_label[1] if len(data_label) > 1 else df.columns[1])
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=tick_fontsize)
        ax2.tick_params(axis='x', labelsize=tick_fontsize)

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
            ax1.axhline(y=target, color='green', linestyle='--', label='Target')
            ax1.axhspan(target-2.5, target+2.5, color='green', alpha=0.2, label='Target region')

        ax2 = ax1.twinx()
        colors2 = colors[1:4]
        for i, col in enumerate(df.columns[1:]):
            label = data_label[i+1]
            ax2.plot(df.index, df[col], color=colors2[i], label=label)

        ylabels = []
        ylabels.append(data_label[1] + ' [V]')
        if len(data_label) > 2:
            ylabels.append(data_label[2] + ' [V]')
        if len(data_label) > 3:
            ylabels.append(data_label[3] + ' [V]')
        ax2.set_ylabel(' / '.join(ylabels), color='black', fontsize=label_fontsize)
        ax2.tick_params(axis='y', labelcolor='black', labelsize=tick_fontsize)
        ax2.tick_params(axis='x', labelsize=tick_fontsize)

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
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.legend(fontsize=legend_fontsize, ncol=len(df.columns))

    plt.xlim(df.index[0], df.index[-1])
    plt.tight_layout()

    if save:
        plt.savefig(f'{plot_name}', dpi=150)
        # print(f"Plot saved as plots/{plot_name}")

    if not all:
        plt.show()


def save_all_plots():
    for run in filnames:
        data = load_data(df_info, run, trim=True)
        plot_name = 'plots_3/' + run + f'_plot.png'
        plot_data(data, run, plot_name, save=True, all=True)
        print(f"Plots for {run} saved.")


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

def calculate_mean_std_precise(chunks):
    """
    Calculate the mean and standard deviation of each chunk.
    """
    maxs = []
    
    for chunk in chunks:
        if len(chunk) > 0:
            maxs.append(np.max(chunk))
        else:
            continue

    return np.mean(maxs), np.std(maxs)

def calculate_mean_std_pos(chunks):
    """
    Calculate the mean and standard deviation of each chunk.
    """
    means = []
    stds = []
    
    for chunk in chunks:
        if len(chunk) > 0:
            means.append(np.mean(chunk))
            stds.append(np.std(chunk))
        else:
            continue

    return means, stds

def plot_gaussian(mean, std, targets=None, labels=None, show=False):
    """
    Plot one or several Gaussian distributions based on the provided mean(s) and std(s).
    mean and std can be scalars or arrays/lists of the same length.
    Optionally, targets and labels can be provided for reference lines and legend.
    """
    import collections.abc

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

    plt.figure(figsize=(14, 9))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    for i, (m, s) in enumerate(zip(mean, std)):
        x = np.linspace(m - 4*s, m + 4*s, 1000)
        y = (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / s) ** 2)
        plt.plot(x, y, label=labels[i], color=colors[i])

    # Plot target(s) if provided
    if targets is not None:
        for t in targets:
            plt.axvline(x=t, color='green', linestyle='--', label='Target')
            plt.axvspan(t-2.5, t+2.5, color='green', alpha=0.2, label='Target region')

    plt.title('Gaussian Distribution', fontsize=22)
    plt.xlabel('Position [deg]', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.grid()
    plt.legend(fontsize=14)
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

    # Ensure mean and std are iterable
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
    
    plt.figure(figsize=(14, 9))
    plt.errorbar(x, mean, yerr=std, fmt='o', capsize=5, color='skyblue', ecolor='black', elinewidth=1.5, markersize=8)
    # Annotate each point with its mean and std
    for i, (xm, ym, ys) in enumerate(zip(x, mean, std)):
        text = f'{ym:.2f} ± {ys:.2f}'
        plt.text(xm + 0.05, ym, text, fontsize=14, va='center', ha='left', color='black', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    
    # Plot target(s) if provided
    if targets is not None:
        for t in targets:
            plt.axhline(y=t, color='green', linestyle='--', label='Target')
            plt.axhspan(t-2.5, t+2.5, color='green', alpha=0.2, label='Target region')

    # plt.xticks(x, labels, rotation=45)
    plt.title('Mean and Standard Deviation', fontsize=22)
    plt.ylabel('Position [deg]', fontsize=14)
    plt.xlabel('Runs', fontsize=14)
    plt.grid(axis='y')
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    
    if show:
        plt.show()

def compute_gaussian_precise(df, save_fig=True):
    runs = ['eval_precise_pos_J_r',
            'eval_precise_pos_J_l',
            'eval_precise_pos_F']
    
    for run in runs:
        data_train = load_data(df, run + '_train', trim_type='ranges_gauss', trim=True)
        data_val = load_data(df, run, trim_type='ranges_gauss', trim=True)

        new_data_train = divide_chunks(data_train)
        new_data_val = divide_chunks(data_val)

        mean, std = [], []

        mean_train, std_train = calculate_mean_std_precise(new_data_train)
        mean_val, std_val = calculate_mean_std_precise(new_data_val)

        mean.append(mean_train)
        std.append(std_train)
        mean.append(mean_val)
        std.append(std_val)
        labels = ['Train', 'Validation']

        plot_gaussian(mean, std, targets=[df.loc[df['run'] == run, 'target'].values[0]], labels=labels, show=not save_fig)

        if save_fig:
            plot_name = 'plots_3/gaussian_' + run + '.png'
            plt.savefig(plot_name, dpi=150)
            print(f"Gaussian plot saved as {plot_name}")

        plt.close()


def compute_mean_std(df, run_type="pos", save_fig=True):
    if run_type == "pos":
        runs = ['eval_position_J_r',
                'eval_position_J_l',
                'eval_position_F',
                'eval_position_G']
    elif run_type == "torque":
        runs = ['eval_torque_J_r',
                'eval_torque_J_l',
                'eval_torque_F',
                'eval_torque_G']
    
    labels = ['Subject 1 right arm', 'Subject 1 left arm', 'Subject 1 forearm', 'Subject 2 right arm']
    global_mean = []
    global_std = []
    
    for run in runs:
        data_val = load_data(df, run, trim_type='ranges_gauss', trim=True)

        new_data_val = divide_chunks(data_val, column='encoder_paddle_pos [deg]', threshold=5 if run_type == "pos" else df.loc[df['run'] == run, 'target'].values[0] - 10)

        if len(new_data_val) == 0:
            print(f"No valid chunks found for run {run}. Skipping.")
            continue

        mean_val, std_val = calculate_mean_std_pos(new_data_val)

        global_mean.append(np.mean(mean_val))
        global_std.append(np.mean(std_val))

        plot_mean_std(mean_val, std_val, targets=[df.loc[df['run'] == run, 'target'].values[0]], labels=['Validation Mean', 'Validation Std'], show=not save_fig)

        if save_fig:
            plot_name = 'plots_3/mean_std_' + run + '.png'
            plt.savefig(plot_name, dpi=150)
            print(f"Position plot saved as {plot_name}")

        plt.close()

    # Get all unique targets for the selected runs
    unique_targets = list({df.loc[df['run'] == run, 'target'].values[0] for run in runs if df.loc[df['run'] == run, 'target'].values[0] is not None})
    plot_gaussian(global_mean, global_std, targets=unique_targets, labels=labels, show=not save_fig)

    if save_fig:
        plot_name = f'plots_3/global_mean_std_{run_type}.png'
        plt.savefig(plot_name, dpi=150)
        print(f"Global Mean and Std plot saved as {plot_name}")
    
    plt.close()

run = 'raw'  # Change this to 'raw', 'torque', 'pos', or 'precise_pos' as needed
save_fig = False
plot_name = 'plots_3/' + run + '_plot.png'
data = load_data(df_info, run, trim=True)

# new_data = divide_chunks(data, column='encoder_paddle_pos [deg]', threshold=10)

# mean, std = calculate_mean_std_pos(new_data)
# plot_mean_std(mean, std, targets=[df_info.loc[df_info['run'] == run, 'target'].values[0]], labels=['Mean', 'Std'], show=True)

plot_data(data, run, plot_name, save=True)

# save_all_plots()
# compute_gaussian_precise(df_info)
# compute_mean_std(df_info)
# compute_mean_std(df_info, run_type="torque")

