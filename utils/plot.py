import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path
from scipy.fft import fft, fftfreq
from statistics import stdev, mean

import utils.constants as cst
import utils.fetch as fetch
import utils.find as find
import utils.misc as misc

# * ===================================================================================================

def coeff_variation(run):
    run_path = find.find_runs(run)[0]
    log_list = find.find_logs(run_path)
    timesteps = []

    for log in log_list:
        with open(log) as f:
            timesteps += [np.float64(line.split()[-1].rstrip('s')) for line in f if line.startswith("Time =")]
    diff = [timesteps[i+1] - timesteps[i] for i in range(len(timesteps)-1)]
    cv = stdev(diff) / mean(diff)
    print("Coefficient of variation of timesteps:", f"{cst.bmag}{cv:.1%}{cst.reset}\n")

# * ===================================================================================================

def gather_runs(runs: str) -> tuple:
    # Convert CSV data to DataFrame
    csv_df = find.csv_postpro_to_df()

    # Find runs in the specified directory
    run_dirs = []
    run_dirs += find.find_runs(runs)

    # Count the number of unique run directories
    runs_nb = len({r.name for r in run_dirs})

    # Raise an error if no runs are found
    if len(run_dirs) == 0:
        raise ValueError(f"No run found with {cst.bred}'{runs}'{cst.reset}.")

    return run_dirs, runs_nb, csv_df

# * ===================================================================================================

def gather_data(runs_dir: list[Path], specdir: str, probe: str, **kwargs) -> list:
    # List to store processed run DataFrames
    run_pp_df_list = []

    # Iterate over each run directory
    for run_path in runs_dir:
        print(f"\n{cst.bmag}--------\n# {run_path.name}\n--------{cst.reset}\n")
        
        # Get data from the run and add it to the list
        run_pp_df_list += [data for data in fetch.fetch_run_data(run_path, specdir=specdir,
                                                                  probe=probe, **kwargs)]
        
        # Raise an error if no data is found for the input
        if not run_pp_df_list:
            raise ValueError("No data found with such input.")
    return run_pp_df_list

# * ===================================================================================================

def plot_time_data(ax, df, handle_prefix, frmt_legend):
    # Iterate over each column (excluding 'Time') in the DataFrame
    for col in [c for c in df.columns if c != 'Time']:
        handle = f"{handle_prefix}{col}{frmt_legend}"
        
        # Plot the data as a line plot
        sns.lineplot(data=df, x='Time', y=col, label=handle, ax=ax, linewidth=cst.LINEWIDTH)
        sns.despine(left=True)
        
        # Yield the handle for each iteration
        yield handle

# * ===================================================================================================

def plot_freq_data(run, ax, df, handle_prefix, frmt_legend, sampling_rate, **kwargs):

    coeff_variation(run)
    
    # Iterate over each column (excluding 'Time') in the DataFrame
    for col in [c for c in df.columns if c != 'Time']:
        frmt_col = f"{handle_prefix}{col}"
        handle = f"{frmt_col}{frmt_legend}"

        print(f"Calculating FFT for {cst.bmag}{col}{cst.reset}...")
        
        # Calculate the FFT and frequency values
        signal_fft = fft(df[col].values)
        freqs = fftfreq(len(df[col])) * sampling_rate

        normalized_spectrum = np.abs(signal_fft) / np.max(np.abs(signal_fft))
        pos_freqs = freqs[freqs >= 0]

        if "lowpass" in kwargs:
            pos_freqs = pos_freqs[pos_freqs <= int(kwargs.get("lowpass"))]

        # Plot the frequency data as a line plot
        sns.lineplot(x=pos_freqs, y=normalized_spectrum[:len(pos_freqs)], label=handle, ax=ax, linewidth=cst.LINEWIDTH)
        sns.despine(left=True)
        
        # Yield the handle for each iteration
        yield handle

# * ===================================================================================================

def set_figure_params(probe, specdir):
    # Create a new figure and axis with specified parameters
    _, ax = plt.subplots(figsize=(12, 27 / 4))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.grid(axis='y', linewidth=0.1, color='00000')
    plt.grid(axis='x', linewidth=0)

    # Set y-axis scale to logarithmic if probe and specdir are None
    if probe is None and specdir is None:
        plt.yscale('log')

    return ax

# * ===================================================================================================

def format_legend(ax, handles):
    # Set legend properties
    ax.legend(
        loc='upper center',
        bbox_to_anchor=[0.5, -0.2],
        framealpha=1,
        frameon=False,
        ncol=misc.ncol(handles),
        borderaxespad=0,
        fontsize=12
    )

# * ===================================================================================================

def set_axis_labels(ax, freq=False, unit=None):
    # Set axis labels based on frequency or time data
    if freq:
        ax.set_ylabel("Normalized Amplitude", labelpad=10, fontsize=15)
        ax.set_xlabel(f"Frequency (Hz)", labelpad=18, fontsize=15)
    else:
        ax.set_xlabel("Iterations | Time (s)", labelpad=18, fontsize=15)
        ax.set_ylabel(unit, labelpad=10, fontsize=15)