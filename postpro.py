import os
import re
import socket
import getpass
import datetime
import importlib 
import matplotlib
import numpy as np
import pandas as pd
import toolbox as tb
import seaborn as sns
import constants as cst
import matplotlib.pyplot as plt

from functools import partial
from file_read_backwards import FileReadBackwards

# Args for the figures
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['lines.markeredgewidth'] = 1
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['lines.linewidth'] = 2.5
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.use('QtAgg')

sns.set()

# * ===================================================================================================

def reload():
    module = importlib.import_module(__name__)
    importlib.reload(module)
    importlib.reload(tb)
    importlib.reload(cst)
    print('Reloaded.')

# * ===================================================================================================

def plot_data(target, *, specdir, csv_path, graph_type='data', probe: str = None, **kwargs):
    
    # Initialization
    marker, space, underscore, sep = tb._display_settings(**kwargs)
    csv_df = tb._csv_postpro_to_df(csv_path)
    runs_dir = []
    run_pp_df_list = []

    # Get number of different runs and their dir
    if isinstance(target, str):
        target = [target]
    for tar in target:
        runs_dir += tb._find_runs(tar)
    runs_nb =  len({os.path.basename(r) for r in runs_dir})

    # ! If no run found
    if len(runs_dir) == 0:
        raise ValueError(f"No run found with {tb._bred}'{target}'{tb._reset}.")
    
    # Verbose
    tb._print_header(runs_dir)

    # Set the figure and axis
    handles = []
    _, ax = plt.subplots(figsize=(12, 27/4))
    ax.set_xlabel(f'{marker}Iterations{sep}Time{space}(s){marker}', labelpad=18, fontsize=15)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Loop over the runs
    for run_path in runs_dir:

        # Verbose
        print(f'\n{tb.bmag}# {os.path.basename(run_path)}{tb.reset}')
        print('')

        # Get data from run
        run_pp_df_list += [data for data in tb._get_data_from_run(run_path,
                                                          specdir = specdir,
                                                          graph_type = graph_type,
                                                          probe = probe,
                                                          **kwargs)]

    # * ========================== PLOT PARAMETERS ==========================

    # Loop over all the DataFrames to plot    
    for data in run_pp_df_list:
        run_id, pp_dir, df = data.values()

        # If there are multiple runs, display runs, else display the column name
        if runs_nb > 1: 
            frmt_legend = f'{sep}{run_id}'
        else: 
            frmt_legend = ''

        # Set the unit on the y axis
        if graph_type == 'data':
            unit = tb._get_unit(df, pp_dir, csv_df)
        elif graph_type == 'residuals':
            unit = f'{marker}Residuals{marker}'
            plt.yscale('log')
            plt.grid(axis='y', linewidth=0.5)

        # Format the column name displayed on the figure
        for col in [c for c in df.columns if c != 'Time']:
            format_string = lambda x: x.replace('_', underscore).replace(' ', space)
            frmt_col = format_string(col)

            # Display 'Probe #' for probe legend
            if graph_type == 'probes':
                frmt_col = f'Probe{space}{frmt_col}'

                # If a unit is specified for the y axis, display it
                if 'unit' in kwargs:
                    unit = kwargs.get("unit")
                else:
                    if probe == 'p': unit = 'Pa'
                    elif probe == 'k': unit = 'J/kg'
                    else: unit = None

            # Format legend handle and append the handles list
            handle = f'{marker}{frmt_col}{frmt_legend}{marker}'
            handles.append(handle)
            
            # Plot the curve with Time on x axis and the selected column on y axis
            sns.lineplot(data=df, x='Time', y=col, label=handle, ax=ax)

            # Set the unit as y label or hide y label if no unit
            if unit == None:
                plt.gca().set_ylabel(None)
            else: 
                ax.set_ylabel(f"{marker}{unit}{marker}", labelpad=10, fontsize=15)

        # Calculate columns number in the legend
        ncol = tb._ncol(handles)

        # Format the legend          
        ax.legend(loc='upper center',
                  bbox_to_anchor = [0.5, -0.2],
                  framealpha = 1,
                  frameon = False,
                  ncol = ncol,
                  borderaxespad=0,
                  fontsize=12)
        
        # Set title
        if 'title' in kwargs:
            title = format_string(kwargs.get('title'))
            ax.set_title(f'{marker}{title}{marker}', fontsize=20)

        # Verbose
        print('\nDisplaying the figure...\n')

        # Print figure
        plt.tight_layout()
        plt.show()

# * ========================= PARTIAL FUNCTIONS =========================

plot_probes = partial(plot_data, specdir=None, graph_type='probes')
plot_residuals = partial(plot_data, specdir=None, graph_type='residuals')

# * ===================================================================================================

def plot_freq(target, *, specdir, csv_path, graph_type='data', probe: str = None, **kwargs):
    ...

# * ===================================================================================================

def aero_balance(run, z=-0.359, wb=2.96):
    """ 
    Calculate the aerodynamic balance (i.e. the % of total downforce applied 
    on the front axle)

    Args:
        z   : z coordinate of the contact patches
        wb  : wheelbase

    Return:
        aero_bal: % of total downforce applied on the front axle
    """
    # Find the path representing the file forceCoeffs.dat
    f = tb._concat_data_files(tb._find_data_files(run, specdir='forceCoeffs'))

    # Mean values of the coefficients
    Cm, Cd, Cl = tb._final_avg(f, 'Cm'), tb._final_avg(
        f, 'Cd'), tb._final_avg(f, 'Cl')

    #  Aero balance, i.e. the % of total lift/downforce applied at the front contact patch
    aero_bal = (1 - ((Cm - z * Cd) / wb) / Cl) * 100
    return aero_bal

# * ===================================================================================================

def bar_chart(target, *,
              rng: int = cst.RNG,
              specdir: str = None,
              probe: str = None,
              **kwargs) -> None:
    
    # ! Mutual exclusion of arguments
    if probe != None and specdir != None:
        raise ValueError('probe and specdir are mutually exclusive.')
    
    # Label formatting
    marker, space, underscore, sep = tb._display_settings(**kwargs)
    
    # Find the run(s) path
    run_dirs = []
    if isinstance(target, str):
        target = [target]
    for tar in target:
        run_dirs += tb._find_runs(tar)

    # ! If no run found
    if len(run_dirs) == 0:
        raise ValueError(f"No run directory found with this list.")
    
    # Else, print the table showing the project and runs
    else:
        tb._print_header(run_dirs)
        
        # If no probe, legend not modified
        if probe == None:
            lgd = ''
            sig_list = tb._get_sig_list(run_dirs,
                                        specdir=specdir,
                                        graph_type='data',
                                        **kwargs)
        # If probe, "Probe #" added to the legend 
        else:
            lgd = f'Probe{space}'
            sig_list = tb._get_sig_list(run_dirs,
                                        graph_type='probes', 
                                        probe=probe,
                                        **kwargs)  

        # Initialization
        handles, xlabel = [], []
        df_mean = pd.DataFrame()
        _, ax = plt.subplots(figsize=(12, 27/4))
        run_number = len({sig['run_id'] for sig in sig_list})
        pp_dir_number = len({sig['pp_dir'] for sig in sig_list})
        xpos = np.arange(len(sig_list))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 0))
        
        # Loop over the datasets to be plotted
        for sig in sig_list:
            run_id = sig['run_id'] 
            pp_dir = sig['pp_dir']
            df = sig['df'].iloc[:, 1:] # remove time column
            
            # Initialize a dict representing the mean data for a run/pp_dir combination
            mean_dict = {'run': run_id, 'pp_dir': pp_dir}
            mean_dict.update({col: df[col].tail(rng).mean() for col in df.columns})
            
            # Add a new row of mean values in df_mean for each run/pp_dir combination
            df_mean = pd.concat([df_mean, pd.Series(mean_dict).to_frame().T])
                            
            # Format strings for LateX font
            format_string = lambda x: x.replace('_', underscore).replace(' ', space)
            frmt_run = format_string(run_id)
            frmt_pp_dir = format_string(pp_dir)
            
            # Format xlabel with pp_dir and/or run_id
            if run_number > 1 and pp_dir_number > 1:  
                frmt_legend = f'{marker}{frmt_pp_dir}{sep}run{frmt_run}{marker}'
            elif pp_dir_number > 1:
                frmt_legend = f'{marker}{frmt_pp_dir}{marker}'
            else:
                frmt_legend = f'{marker}run{frmt_run}{marker}'
            xlabel.append(frmt_legend)
            
        # Set a multi index with the run/pp_dir combination
        df_mean = df_mean.set_index(['run', 'pp_dir'])

        # Set the width of the rectangles
        width = 1.5 / (len(xpos) * len(df.columns))

        # Loop over the columns to plot each series of data with their handle
        for i, col in enumerate(df_mean.columns):
            handle = f'{marker}{lgd + col}{marker}'
            handles.append(handle)
            rect = ax.bar(xpos + i * width,
                            df_mean[col],
                            width = width,
                            label = handle)
            if 'fancyplot' in kwargs or 'fp' in kwargs:
                frmt = '${:.2e}$'
            else:
                frmt = '{:.2e}'
            ax.bar_label(rect, padding=3, fmt=frmt)
            
        # Plot parameters
        ax.legend(loc='upper center',
                    bbox_to_anchor = [0.5, -0.1],
                    framealpha = 1,
                    frameon = False,
                    ncol = tb._ncol(handles),
                    borderaxespad=0,
                    fontsize=12)
        
        # If a unit is specified for the y axis
        if 'unit' in kwargs:
            ax.set_ylabel(f'{marker}{kwargs.get("unit")}{marker}', labelpad=10)

        # If a title is specified
        if 'title' in kwargs:
            title = format_string(kwargs.get('title'))
            ax.set_title(f'{marker}{title}{marker}', fontsize=20, fontweight='bold')
            
        # Set the xticks at the center of the grouped rectangles
        ax.set_xticks(xpos + width * (len(df_mean.columns) - 1) / 2, xlabel, fontsize=15)
        
        # Verbose
        print('\nDisplaying the figure...\n')
        
        plt.tight_layout()
        plt.show()

# * ===================================================================================================

def plot_time(target, *, x='iterations', skipstart=10, **kwargs):
    col1 = cst.HOPIUM_PALETTE['hopium']
    col2 = cst.HOPIUM_PALETTE['red']

    if x == 'iterations' or x == 'time':
        fig, ax = plt.subplots(figsize=(12, 27/4))
        ax_cumul = ax.twinx()
    elif x =='both':
        fig, (ax_time, ax_iter) = plt.subplots(1, 2, figsize=(12, 27/4), sharey=False)
        ax_time_cumul = ax_time.twinx()
        ax_iter_cumul = ax_iter.twinx()
    
    # Label formatting
    marker, space, underscore, sep = tb._display_settings(**kwargs)
    # Find the run(s) path
    run_dirs = []
    log_files = []
    if isinstance(target, str):
        target = [target]
    for tar in target:
        run_dirs += tb._find_runs(tar)

    # ! If no run found
    if len(run_dirs) == 0:
        raise ValueError(f"No run directory found with this list.")
    
    # If at least one run found
    else:
        time_pattern = re.compile(r'Time = ([\d.]+)s')
        exec_pattern = re.compile(r'ExecutionTime\s*=\s*([\d.]+)\s*s')
        current_iter = 0
        prev_iter = 0
        restarts = []
        cumul_time = []
        tb._print_header(run_dirs)
        for run in run_dirs:
            run_id = re.search('(?<=run)\d{3}\w*', run).group(0)
            print(f'\nRun {tb._bmag}{run_id}{tb._reset}')
            log_files += tb._find_logs(run)
            data_iter = {'timestep': [], 'exec_time': []}

            for log in log_files:
                print('Log: ' + log)
                if len(data_iter['exec_time']) > 0:
                    current_iter = data_iter['exec_time'][-1]

                with open(log, 'r') as f:
                    time_bool = False
                    for line in f:
                        # Patterns to find the lines starting with 'Time' and 'ExecutionTime' and extract the values
                        time_match = re.match(time_pattern, line)
                        exec_match = re.match(exec_pattern, line)
                        # If a line starts with 'Time' and time_bool is False
                        if time_match and not time_bool:
                            # Extract the float value of Time
                            time_value = float(time_match.group(1))
                            # Set time_bool to True because a line starting with 'Time' has been found
                            time_bool = True
                        # If a line starts with 'ExecutionTime' and time_bool is True
                        elif exec_match and time_bool:
                            # The value of the previous execution time is kept to calculate the time difference
                            prev_iter = current_iter
                            # The current execution is extracted
                            current_iter = float(line.split()[2])
                            # Set time_bool to False to reset the boolean indicating if a 'Time' line has been found
                            time_bool = False
                            # The 'Time' value is added to the timestep list
                            data_iter['timestep'].append(time_value)
                            # If the iteration is not the first one
                            if len(data_iter['exec_time']) > 0:
                                # The timestep duration is calculated and added to the exec_time list
                                data_iter['exec_time'].append(current_iter - prev_iter)
                                cumul_time.append(cumul_time[-1] + current_iter - prev_iter)
                            # If it is the first iteration
                            else:
                                # The duration of the first timestep is added to the exec_time list
                                data_iter['exec_time'].append(current_iter)
                                cumul_time.append(current_iter)
                restarts.append((data_iter['timestep'][-1], len(data_iter['timestep'])))
            ax_time.scatter(data_iter['timestep'][skipstart:],
                            data_iter['exec_time'][skipstart:],
                            color=col1, marker='.', s=1.5)
            
            ax_time_cumul.plot(data_iter['timestep'][skipstart:],
                            cumul_time[skipstart:], color=col2)
            
            ax_iter.scatter(np.arange(skipstart, len(data_iter['timestep'])),
                            data_iter['exec_time'][skipstart:],
                            color=col1, marker='.', s=1.5)
            
            ax_iter_cumul.plot(np.arange(skipstart, len(data_iter['timestep'])),
                            cumul_time[skipstart:], color=col2)

    for r in [element[0] for element in restarts[:1]]:
        ax_time.axvline(r, linestyle=':')
    for p in [element[1] for element in restarts[:1]]:
        ax_iter.axvline(p, linestyle=':')

    ax_time.set_xlabel(f'{marker}Time{space}(s){marker}', labelpad=10)
    ax_iter.set_xlabel(f'{marker}Iterations{marker}', labelpad=10)

    ax_time.tick_params(labelleft=True, labelright=False,  left=True, right=False)
    ax_time_cumul.tick_params(labelleft=False, labelright=False,  left=False, right=True)
    ax_iter.tick_params(labelleft=False, labelright=False,  left=True, right=False)
    ax_iter_cumul.tick_params(labelleft=False, labelright=True,  left=False, right=True)

    for ticklabel in ax_time.get_yticklabels():
        ticklabel.set_color(col1)
    for ticklabel in ax_iter_cumul.get_yticklabels():
        ticklabel.set_color(col2)

    fig.text(-0.03, 0.35, f'{marker}Time{space}per{space}Iteration{space}(s){marker}',
             fontsize=15, rotation=90, color='#ef476f')
    fig.text(1, 0.42, f'{marker}Total{space}Time{space}(s){marker}',
             fontsize=15, rotation=-90, color='#0096c7')
    fig.tight_layout()
    plt.show()

# * ===================================================================================================

def sim_time(run):

    # Get run and log files  
    run_path = tb._find_runs(run)[0]
    log_files = tb._find_logs(run_path)
    times_list = []

    # Parsing log files to find the line containing the last timestep
    for log in log_files: 
        with FileReadBackwards(log, encoding='utf-8') as frb:
            for line in frb:
                # Steady simulations
                if line.startswith('ExecutionTime') and tb._issteady(run):
                    stime = int(line.split()[-2])
                    break
                # Unsteady simulations
                elif line.startswith('Time') and not tb._issteady(run):
                    stime = float(line.split()[-1][:-1])
                    break
        times_list.append(stime)
    
    # Total time calculation
    total_time = sum(times_list)
    return total_time

# * ===================================================================================================

def stop_sim(run):
    """
    This function stops the simulation by modifying the 'stopAt' line in the controlDict file
    of the specified run. It finds the path to the controlDict file and opens it, 
    then it replaces the first occurrence of 'endTime' with 'writeNow' and writes the modified line
    to the temporary file. Finally, the script uses the os.replace function to replace the original
    file with the temporary file.
    
    Parameters:
    - run (str): The name of the run folder containing the controlDict file
    
    Returns:
    - None
    """
    run_path = tb._find_runs(run)[0]
    # Path to the controlDict file in the run
    controlDict_path = tb._find_files("controlDict", root_dir=run_path + '/system/')[0]
    # Copy of the controlDict file where the new line will be written
    temp_file_path = controlDict_path + ".tmp"
    # Read the controlDict file to find the line starting with "stopAt"
    with open(controlDict_path, "r") as f, open(temp_file_path, "w") as temp:
            for line in f:
                if line.startswith("stopAt"):
                    # Replace "endTime" by "writeNow" in the temp file 
                    temp.write(line.replace("endTime", "writeNow"))
                else:
                    temp.write(line)
    print('Simulation stopping...')

    # Replace the original controlDict file by the temp file
    os.replace(temp_file_path, controlDict_path)

# * ===================================================================================================

def recap_sim(runs: str, *,
              xl_path: str = '/mnt/pocbinaryfiles/python_package/recap_sim.xlsx',
              geometry_name: str = None,
              probe: str = None,
              specdir: str = None,
              rng: int = cst.RNG,
              **kwargs) -> None:
    
    test = pd.DataFrame()
    
    try:
        test.to_excel(xl_path)
    except OSError as e:
        raise
    
    # ! Mutual exclusion of arguments
    if probe != None and specdir != None:
        raise ValueError('probe and specdir are mutually exclusive.')

    new_rows = pd.DataFrame()
    run_paths = tb._find_runs(runs)
    tb._print_header(run_paths)

    for run_path in run_paths:
        run_id = os.path.basename(run_path)
        steady = tb._issteady(run_id)
        
        # Find the path to the run directory (assuming the run directory is unique)
        date_check = False
        n_procs_check = False
        date = ''
        n_procs = ''
        turbulence_model = ''
        
        # Get the turbulence model from the "turbulenceProperties" file
        momentumTransport_path = os.path.join(run_path, "constant", "momentumTransport")
        with open(momentumTransport_path, 'r') as f:
            for line in f:
                if 'model' in line:
                    turbulence_model = line.split()[-1].strip(";")
                    break

        # Get the number of iterations from the "controlDict" file
        log_files = sorted(tb._find_logs(run_path))
        
        data_dict = {
            'Project': [os.path.basename(os.path.dirname(run_path))],
            'Run': [run_id],
            'User': [getpass.getuser()],
            'Workstation': [socket.gethostname()],
            'Geometry': [geometry_name],
            'Clock Time': [str(datetime.timedelta(seconds=int(sim_time(run_id))))],
            'Turbulence Model': [turbulence_model],
        }

        for i, log in enumerate(log_files):
            if i == 0:
                with open(log, 'r') as f:
                    for line in f:
                        if line.startswith('Date'):
                            date = ' '.join(line.split()[-3:])
                            date_check = True
                        elif line.startswith('nProcs'):
                            n_procs = int(line.split()[-1])
                            n_procs_check = True
                        if date_check and n_procs_check:
                            break
            if i == len(log_files) - 1:
                with FileReadBackwards(log) as frb:
                    if steady:
                        for line in frb:
                            if line.startswith("Time ="):
                                data_dict['Iterations'] = int(line.split()[-1])
                                break
                    else:
                        for line in frb:
                            if line.startswith("Time ="):
                                data_dict['Simulated Time (s)'] = float(line.split()[-1][:-1])
                                break
        
        data_dict.update({'Date': [date], '# Procs': [n_procs]})
        
        if specdir != None:
            sig_list = tb._get_sig_list([run_path], specdir=specdir, **kwargs)
            
        elif probe != None:
            sig_list = tb._get_sig_list([run_path], graph_type='probes', probe=probe, **kwargs)

        mean_dict = {}
        for sig in sig_list:
            df = sig['df']
            # Remove the Time column
            df = df.iloc[:,1:]
            # Initialize a dict representing the mean data for a run/pp_dir combination
            mean_dict.update({col: df[col].tail(rng).mean() for col in df.columns})
            mean_dict = {(probe + col) if col[0].isdigit() else col: value for col, value in mean_dict.items()}

            data_dict.update(mean_dict)

        # add the data to the dataframe
        df = pd.DataFrame.from_dict(data_dict)

        # Move column date to the first position
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(8))
        df = df[cols].round(3)
        
        new_rows = pd.concat([new_rows, df], axis=0)
        new_cols = new_rows.columns
        
    if os.path.isfile(xl_path):
        existing_rows = pd.read_excel(xl_path)
        total_rows = existing_rows
        existing_cols = existing_rows.columns
        
    # ! Wrong path
    else:
        raise ValueError('The path to the Excel file does not exist.')

    # Find columns in new_rows that are missing in existing_df
    missing_cols_xl = list(set(new_cols) - set(existing_cols))
    for col in missing_cols_xl:
        total_rows[col] = ''
    if missing_cols_xl:
        print('\nNew column(s) added:',
              f'{tb.bmag}{f"{tb.reset}, {tb.bmag}".join(sorted([str(i) for i in missing_cols_xl]))}{tb.reset}.')
    
    # Find columns in new_rows that are missing in new_rows
    missing_cols_new_rows = list(set(existing_cols) - set(new_cols))
    for col in missing_cols_new_rows:
       new_rows[col] = ''

    # Append the new rows to the existing DataFrame
    appended_df = pd.concat([total_rows, new_rows], ignore_index=True) 
    sorted_missing_cols = sorted(missing_cols_xl)
    sorted_cols = existing_cols.to_list() + sorted_missing_cols
    appended_df = appended_df[sorted_cols]
    # Save the updated DataFrame to a new Excel file
    appended_df.to_excel(xl_path, index=False)
    
    tb._format_excel(xl_path)